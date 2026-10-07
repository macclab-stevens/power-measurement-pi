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

import contextlib
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


def _no_step(name: str):  # noqa: ARG001 - default step sink (PMIC mode, tests)
    return contextlib.nullcontext()


class TaskAdapter:
    """`step` is a context-manager factory, step(name), set by run.py to EventLog.step in
    monsoon mode; adapters wrap each sub-phase of one iteration in it."""
    name = "base"
    step = staticmethod(_no_step)

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
        self._frame = None          # fixed uint8 HWC RGB source frame (non-live)
        self._model_name = ""

    def load(self, model: str, opt):
        self._model_name = model
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

    def _source_frame(self):
        """uint8 HWC RGB frame: the check image if given, else a fixed synthetic 640x480."""
        if self._frame is None:
            import numpy as np
            img = None
            if self._check_image:
                import cv2
                bgr = cv2.imread(self._check_image)
                if bgr is not None:
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img is None:
                img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
            self._frame = np.ascontiguousarray(img)
        return self._frame

    def preprocess(self, frame):
        """uint8 HWC RGB -> float 1x3xSxS. Detection: letterbox (pad 114) to imgsz, /255.
        Classification: resize to imgsz, ImageNet mean/std normalization."""
        import cv2
        import numpy as np
        S = self._imgsz
        if self._is_detection:
            h, w = frame.shape[:2]
            r = min(S / h, S / w)
            nh, nw = int(round(h * r)), int(round(w * r))
            canvas = np.full((S, S, 3), 114, dtype=np.uint8)
            top, left = (S - nh) // 2, (S - nw) // 2
            canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh),
                                                              interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float() / 255.0
        else:
            img = cv2.resize(frame, (S, S), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = (t - mean) / std
        return t.unsqueeze(0)

    def postprocess(self, out):
        """Detection: NMS (conf 0.25, IoU 0.45). Classification: softmax + top-5."""
        if self._is_detection:
            from ultralytics.utils.nms import non_max_suppression
            return non_max_suppression(out, conf_thres=self._conf, iou_thres=0.45)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        return torch.softmax(logits, dim=-1).topk(5, dim=-1)

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
            if self._live:
                with self.step("camera_capture"):
                    x = self._fixture()
            else:
                with self.step("preprocess"):
                    x = self.preprocess(self._source_frame())
            with self.step("forward"):
                out = self._module(x)
            with self.step("postprocess"):
                self.postprocess(out)
            return out

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
        with torch.no_grad():
            with self.step("preprocess"):
                idx = torch.randint(0, self._vocab, (1, T), dtype=torch.int64)
            with self.step("forward"):
                out = self._module(idx)
            with self.step("postprocess"):
                logits = out.logits if hasattr(out, "logits") else out
                torch.softmax(logits[:, -1, :].float(), dim=-1).argmax(dim=-1)
            return out

    def oracle(self) -> bool:
        ok_pre = finite_ok(self.run_one("prefill"))
        ok_dec = finite_ok(self.run_one("decode"))
        return ok_pre and ok_dec


# --------------------------------------------------------------------------
# Training — one optimizer step per iteration, every sub-phase stepped
# --------------------------------------------------------------------------
TRAIN_STEPS = ("dataload", "zero_grad", "forward", "loss", "backward", "optimizer_step")


class TrainTask(TaskAdapter):
    """Synthetic-data training: torchvision classifiers on images, MiniGPT-explicit on tokens.

    dataload draws a batch from a fixed uint8 pool (random flip + normalize for images), so the
    model can memorize it and the loss is meaningful. zero_grad uses set_to_none=False (a real
    memset). Steps per iteration: dataload, zero_grad, forward, loss, backward, optimizer_step.
    """
    name = "train"
    POOL_BATCHES = 4
    N_CLASSES = 10

    def __init__(self):
        self._module = None
        self._opt = None
        self._is_lm = False
        self._batch = 8
        self._imgsz = 128
        self._seq = 64
        self._vocab = 256
        self._pool_x = None
        self._pool_y = None
        self._k = 0
        self.losses: list[float] = []

    def load(self, model: str, opt):
        self._batch = getattr(opt, "batch", 8)
        self._imgsz = opt.imgsz
        self._seq = opt.seq
        if self._batch < 2:
            raise ValueError("--task train needs --batch >= 2 (BatchNorm in train mode)")
        g = torch.Generator().manual_seed(0)
        if model.startswith(("torchvision:", "tv:")):
            import torchvision.models as tvm
            m_name = model.split(":", 1)[1]
            if not hasattr(tvm, m_name):
                raise ValueError(f"unknown torchvision model: {m_name}")
            self._module = getattr(tvm, m_name)(weights=None, num_classes=self.N_CLASSES)
            n = self.POOL_BATCHES * self._batch
            self._pool_x = torch.randint(0, 256, (n, 3, self._imgsz, self._imgsz),
                                         dtype=torch.uint8, generator=g)
            self._pool_y = torch.randint(0, self.N_CLASSES, (n,), generator=g)
        elif model == "minigpt-explicit":
            self._module = MiniGPTExplicit(vocab=self._vocab)
            self._is_lm = True
            n = self.POOL_BATCHES * self._batch
            self._pool_x = torch.randint(0, self._vocab, (n, self._seq + 1), generator=g)
        else:
            raise ValueError(f"unknown train model '{model}' "
                             "(torchvision:<name> or minigpt-explicit)")
        self._module.train()
        name = getattr(opt, "optimizer", "sgd")
        params = self._module.parameters()
        if name == "sgd":
            self._opt = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        elif name == "adam":
            self._opt = torch.optim.Adam(params, lr=1e-3)
        else:
            raise ValueError(f"unknown --optimizer '{name}' (sgd | adam)")
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self

    @property
    def module(self) -> nn.Module:
        return self._module

    def modes(self) -> list[str]:
        return ["train"]

    def _dataload(self):
        b = self._batch
        i = (self._k % self.POOL_BATCHES) * b
        self._k += 1
        if self._is_lm:
            ids = self._pool_x[i:i + b]
            return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
        x = self._pool_x[i:i + b].float().div_(255.0)
        flip = torch.rand(b) < 0.5
        x[flip] = x[flip].flip(-1)
        return (x - self._mean) / self._std, self._pool_y[i:i + b]

    def run_one(self, mode: str):
        import torch.nn.functional as F
        with self.step("dataload"):
            x, y = self._dataload()
        with self.step("zero_grad"):
            self._opt.zero_grad(set_to_none=False)
        with self.step("forward"):
            out = self._module(x)
        with self.step("loss"):
            if self._is_lm:
                loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            else:
                loss = F.cross_entropy(out, y)
        with self.step("backward"):
            loss.backward()
        with self.step("optimizer_step"):
            self._opt.step()
        v = float(loss.detach())
        self.losses.append(v)
        return loss.detach()

    def oracle(self) -> bool:
        return bool(self.losses) and all(math.isfinite(v) for v in self.losses)


_REGISTRY = {"image": ImageTask, "lm": TokenLMTask, "train": TrainTask}


def make_task(name: str, model: str, opt) -> TaskAdapter:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown --task '{name}'; valid: {sorted(_REGISTRY)}")
    return _REGISTRY[name]().load(model, opt)