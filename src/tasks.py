"""Task adapters — turn ANY PyTorch model family into the harness's shape.

A TaskAdapter isolates the three model-specific concerns from the measurement
core (run.py):
  * what the executable nn.Module is (hooks_root),
  * the fixture(s) — the input tensor(s) that drive one unit of work per mode,
  * the oracle — a per-task sanity check that the model is genuinely working
    (replaces the object-detection conf check that used to gate everything).

Two canonical drivers (per the adversarial review):
  * ImageTask  — one image tensor in, one forward out. Covers classification
    (arbitrary nn.Module) and object detection (ultralytics YOLO `.model`,
    with optional --check-image object-conf oracle).
  * TokenLMTask — causal language model with two work modes: PREFILL (full
    prompt forward) and DECODE (single token). Ships a self-contained MiniGPT
    (torch-only, exercises int64 embeddings, LayerNorm, nn.MultiheadAttention
    multi-tensor leaves) plus an optional transformers path.

Adding a new model family later = one small adapter file; no data-plane change.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def finite_ok(out) -> bool:
    """True if the tensor output is finite and non-empty (sanity for a forward)."""
    flatten = []
    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, torch.Tensor):
            flatten.append(x)
        elif x is None:
            pass
        else:
            pass
    walk(out)
    if not flatten:
        return False
    for t in flatten:
        if t.numel() == 0 or not bool(torch.isfinite(t).all()):
            return False
    return True


class TaskAdapter:
    name = "base"

    def load(self, model, opt) -> "TaskAdapter":
        raise NotImplementedError

    @property
    def module(self) -> nn.Module:
        raise NotImplementedError

    def modes(self) -> list[str]:
        return ["forward"]

    def run_one(self, mode: str):
        raise NotImplementedError

    def oracle(self) -> bool:
        return True


# --------------------------------------------------------------------------
# Image task (classification + detection)
# --------------------------------------------------------------------------
class ImageTask(TaskAdapter):
    name = "image"

    def __init__(self):
        self._module = None
        self._is_detection = False
        self._imgsz = 640
        self._conf = 0.25
        self._check_image = None
        self._checked = False
        self._live = False
        self._cam = None
        self._frame_iter = None

    def load(self, model: str, opt):
        self._imgsz = opt.imgsz
        self._conf = opt.conf
        self._check_image = opt.check_image
        self._live = getattr(opt, "live_camera", False)
        # torchvision models (e.g. torchvision:efficientnet_b0)
        if model.startswith(("torchvision:", "tv:")):
            import torchvision.models as tvm
            m_name = model.split(":", 1)[1]
            if not hasattr(tvm, m_name):
                raise ValueError(f"unknown torchvision model: {m_name}")
            self._module = getattr(tvm, m_name)(weights=None).eval()
            self._is_detection = False
        # ultralytics detection (yolo11n.pt / yolov8n.pt / .model)
        elif model.endswith(".pt") or model.startswith(("yolo", "yolov")):
            import ultralytics  # noqa: WPS433
            yolo = ultralytics.YOLO(model)
            self._module = yolo.model            # the DetectionModel nn.Module
            self._is_detection = True
            self._yolo = yolo
        else:
            # generic classification / any nn.Module weights
            from torch.serialization import add_safe_globals  # not needed
            sd = torch.load(model, map_location="cpu")
            self._module = sd.get("model", sd) if isinstance(sd, dict) else sd
            if not isinstance(self._module, nn.Module):
                raise TypeError(f"{model} did not load to an nn.Module")
            self._module.eval()
        return self

    @property
    def module(self) -> nn.Module:
        return self._module

    def modes(self) -> list[str]:
        return ["forward"]

    def _fixture(self):
        if self._live:
            if self._frame_iter is None:
                from cam import Camera
                self._cam = Camera(size=(640, 480))
                meta = self._cam.start()
                self._frame_iter = self._cam.frames()
                self.cam_meta = meta
            import numpy as np
            frame = next(self._frame_iter)                     # HWC uint8 RGB888
            t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            return t.unsqueeze(0).float() / 255.0              # 1,3,H,W float32
        return torch.rand(1, 3, self._imgsz, self._imgsz)

    def run_one(self, mode: str):
        with torch.no_grad():
            return self._module(self._fixture())

    def stop_camera(self):
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
            self._frame_iter = None

    def oracle(self) -> bool:
        if self._is_detection and self._check_image:
            from detector import Detector
            r = Detector.max_conf(self._yolo.predict(self._check_image,
                                                     imgsz=self._imgsz, verbose=False)[0])
            ok = r >= self._conf
            self._checked = r
            return ok
        # open-loop: a real forward that stays finite is the sanity check
        return finite_ok(self.run_one("forward"))


# --------------------------------------------------------------------------
# MiniGPT — torch-only causal LM (verification vehicle; no transformers dep)
# Exercises: int64 Embedding input, LayerNorm, nn.MultiheadAttention (multi-
# tensor leaf), Linear head. Modes prefill (full seq) + decode (1 token).
# --------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)      # causal, but kernels identical for power
        x = x + a
        x = x + self.norm2(x)
        return x


class MiniGPT(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayer=2, vocab=256):
        super().__init__()
        self.d = d_model
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))
        self.blocks = nn.ModuleList([_Block(d_model, nhead) for _ in range(nlayer)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):            # idx: [B, T] int64
        T = idx.shape[1]
        x = self.tok(idx) * math.sqrt(self.d)
        x = x + self.pos[:, :T]
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class MiniGPTExplicit(torch.nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayer=2, vocab=256):
        super().__init__()
        import math
        self.d=d_model; self.nhead=nhead
        self.tok=torch.nn.Embedding(vocab,d_model)
        self.pos=torch.nn.Parameter(torch.zeros(1,256,d_model))
        self.qkv=torch.nn.ModuleList([torch.nn.Linear(d_model,3*d_model) for _ in range(nlayer)])
        self.proj=torch.nn.ModuleList([torch.nn.Linear(d_model,d_model) for _ in range(nlayer)])
        self.n1=torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(nlayer)])
        self.n2=torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(nlayer)])
        self.head=torch.nn.Linear(d_model,vocab)
    def forward(self, idx):
        import math, torch.nn.functional as F
        T=idx.shape[1]; x=self.tok(idx)*math.sqrt(self.d)+self.pos[:,:T]
        B,T,C=x.shape; H=self.nhead; Dh=C//H
        for qkv,proj,n1,n2 in zip(self.qkv,self.proj,self.n1,self.n2):
            h=n1(x); q,k,v=qkv(h).chunk(3,dim=-1)
            q=q.view(B,T,H,Dh).transpose(1,2); k=k.view(B,T,H,Dh).transpose(1,2); v=v.view(B,T,H,Dh).transpose(1,2)
            a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            a=a.transpose(1,2).reshape(B,T,C); x=x+proj(a); x=x+n2(x)
        return self.head(x)


class TokenLMTask(TaskAdapter):
    name = "lm"

    def __init__(self):
        self._module = None
        self._seq = 64
        self._vocab = 256
        self._mode = "prefill"

    def load(self, model: str, opt):
        self._seq = opt.seq
        if model == "minigpt":
            self._module = MiniGPT(vocab=self._vocab)
            self._vocab = 256
        elif model == "minigpt-explicit":
            self._module = MiniGPTExplicit()
            self._vocab = 256
        elif model.startswith("hf:"):
            # optional transformers-backed causal LM (needs: uv add transformers)
            from transformers import AutoModelForCausalLM  # noqa: WPS433
            if self._seq <= 0:
                self._seq = 64
            self._module = AutoModelForCausalLM.from_pretrained(
                model[3:], torch_dtype="auto", low_cpu_mem_usage=True).eval()
            self._vocab = self._module.config.vocab_size
        else:
            raise ValueError(f"unknown LM '{model}' (use 'minigpt' or 'hf:<name>')")
        return self

    @property
    def module(self) -> nn.Module:
        return self._module

    def modes(self) -> list[str]:
        return ["prefill", "decode"]

    def run_one(self, mode: str):
        T = self._seq if mode == "prefill" else 1
        idx = torch.randint(0, self._vocab, (1, T), dtype=torch.int64)
        with torch.no_grad():
            return self._module(idx)

    def oracle(self) -> bool:
        ok_pre = finite_ok(self.run_one("prefill"))
        ok_dec = finite_ok(self.run_one("decode"))
        return ok_pre and ok_dec


_REGISTRY = {"image": ImageTask, "lm": TokenLMTask}


def make_task(name: str, model: str, opt) -> TaskAdapter:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown --task '{name}'; valid: {sorted(_REGISTRY)}")
    return _REGISTRY[name]().load(model, opt)