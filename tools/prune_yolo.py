"""Structured channel pruning of a YOLO model (no fine-tuning) with torch-pruning.

  cd tools
  uv run python prune_yolo.py --model ../yolo11n.pt                    # -> ../models/yolo11n_p30.pt and _p50.pt
  uv run python prune_yolo.py --model ../yolo11n.pt --ratios 0.3       # one ratio
  uv run python prune_yolo.py --model ../yolo11n.pt --ratios 0.3 --verify-only   # reload + forward, fresh process

Each artifact is the Task 5 dict (src/yolo_variants.py PRUNED_FORMAT 'pi-power-pruned/1') that
`src/run.py --prune models/<stem>_p30.pt` loads:
    format, model (pruned FP32 nn.Module, eval), base_model, ratio, params_before, params_after,
    macs_before, macs_after (both at 640x640, batch 1; a MAC is one multiply-add), plus `method`.

Method (the honest part, also written into the artifact's `method`):
  * Importance: L2 norm of the conv weights of a channel group (torch-pruning GroupMagnitudeImportance,
    p=2, mean group reduction and mean normalisation), on the UNFUSED Conv+BN model (the model the Pi
    runs for FP32, so latency is comparable). No data is used.
  * Global channel ratio: `pruning_ratio` is the fraction of all prunable channels removed over the
    whole network (torch-pruning global_pruning), so layers are pruned unevenly. It is NOT a parameter
    ratio: removing channels shrinks the input and the output side of a conv, parameters fall roughly
    as (1 - ratio)^2 (see params_after / macs_after in the artifact).
  * Ignored: the output convs of the Detect head (cv2[i][-1] box, cv3[i][-1] class, dfl) so the
    prediction layout (4 * reg_max + nc) is unchanged, and the self-attention blocks of C2PSA
    (`Attention` reshapes and splits qkv by fixed head widths).
  * No fine-tuning: accuracy drops, the drop is recorded by tools/eval_map.py (data/map_reference.json).
  * C2f / C3k2 / C2PSA `chunk`/`split` the tensor at a fixed width `self.c`, which breaks channel
    pruning. Before pruning they are replaced by yolo_variants.C2fSplitFree / C2PSASplitFree (cv1 becomes
    two convs holding the two halves of the weights): outputs are identical, and the classes live in
    src/yolo_variants.py so the Pi can unpickle the artifact.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import yolo_variants as yv  # noqa: E402  (artifact format, split-free classes, variant ids)

MODELS_DIR = os.path.join(REPO, "models")
IMGSZ = 640
DEFAULT_RATIOS = (0.30, 0.50)
IMPORTANCE = "GroupMagnitudeImportance(p=2): L2 norm of conv weights, mean group reduction, mean normaliser"
IGNORED = ("Detect output convs: cv2[i][-1] (box), cv3[i][-1] (class), dfl",
           "C2PSA Attention blocks (fixed-width qkv split and head reshape)")


# ---- pure pieces (unit tested) ------------------------------------------------------------------
def artifact_path(model_path: str, ratio: float, out_dir: str = MODELS_DIR) -> str:
    """models/yolo11n_p30.pt for ../yolo11n.pt at 0.30 (the name run.py's --prune is given)."""
    return os.path.join(out_dir, yv.variant_id(model_path, "fp32", ratio) + ".pt")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, imgsz: int = IMGSZ) -> int:
    """Multiply-accumulates of one 1x3xSxS forward (torch-pruning's counter: conv/linear MACs)."""
    import torch_pruning as tp
    macs, _ = tp.utils.count_ops_and_params(copy.deepcopy(model).eval(), torch.randn(1, 3, imgsz, imgsz))
    return int(macs)


def make_split_free(model: nn.Module) -> tuple[nn.Module, dict]:
    """A copy of an ultralytics DetectionModel with every C2f / C3k2 / C2PSA replaced by its
    split-free equivalent (same weights, same outputs), and {class name: how many replaced}."""
    from ultralytics.nn.modules import block
    new = copy.deepcopy(model).eval()
    counts: dict[str, int] = {}

    def swap(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, block.C2f):             # C3k2 is a C2f subclass with the same forward
                repl = yv.C2fSplitFree.from_c2f(child)
            elif isinstance(child, block.C2PSA):
                repl = yv.C2PSASplitFree.from_c2psa(child)
            else:
                swap(child)
                continue
            counts[type(child).__name__] = counts.get(type(child).__name__, 0) + 1
            setattr(parent, name, repl)

    swap(new)
    return new, counts


def ignored_layers(model: nn.Module) -> list[nn.Module]:
    """Modules torch-pruning must not prune: Detect's output convs and C2PSA attention."""
    from ultralytics.nn.modules import block, head
    detect = next(m for m in model.modules() if isinstance(m, head.Detect))
    ignored: list[nn.Module] = [detect.dfl]
    for branches in (detect.cv2, detect.cv3):
        ignored += [b[-1] for b in branches]
    ignored += [m for m in model.modules() if isinstance(m, block.Attention)]
    return ignored


def prune_model(split_free: nn.Module, ratio: float, imgsz: int = IMGSZ) -> nn.Module:
    """A pruned copy of a split-free DetectionModel: global L2-magnitude channel pruning, eval mode."""
    import torch_pruning as tp
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"pruning ratio must be in (0, 1), got {ratio!r}")
    model = copy.deepcopy(split_free).eval().requires_grad_(True)
    example = torch.randn(1, 3, imgsz, imgsz)
    pruner = tp.pruner.BasePruner(
        model, example, importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio, global_pruning=True, ignored_layers=ignored_layers(model))
    pruner.step()
    return model.eval()


def build_artifact(pruned: nn.Module, base_model: str, ratio: float, params_before: int,
                   macs_before: int, imgsz: int = IMGSZ, split_counts: dict | None = None) -> dict:
    """The pi-power-pruned/1 dict (counts of the pruned model are measured here)."""
    import torch_pruning as tp
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"pruning ratio must be in (0, 1), got {ratio!r}")
    return {"format": yv.PRUNED_FORMAT, "model": pruned, "base_model": os.path.basename(base_model),
            "ratio": float(ratio), "params_before": int(params_before), "params_after": count_params(pruned),
            "macs_before": int(macs_before), "macs_after": count_macs(pruned, imgsz), "imgsz": imgsz,
            "method": {"library": f"torch-pruning {tp.__version__}", "importance": IMPORTANCE,
                       "global_pruning": True, "ignored": list(IGNORED), "fine_tuned": False,
                       "split_free_replaced": split_counts or {}},
            "torch_version": torch.__version__}


# ---- verification: what the Pi does with the file ---------------------------------------------
_VERIFY_CODE = r"""
import json, sys
sys.path.insert(0, {src!r})
import torch
import yolo_variants as yv
art = yv.load_artifact({path!r}, yv.PRUNED_FORMAT)
model = art["model"].eval()
with torch.inference_mode():
    out = model(torch.zeros(1, 3, {imgsz}, {imgsz}))[0]
leaked = [m for m in ("torch_pruning", "prune_yolo", "eval_map") if m in sys.modules]
print(json.dumps({{"output_shape": list(out.shape), "ratio": art["ratio"], "leaked_modules": leaked,
                  "params_after": sum(p.numel() for p in model.parameters())}}))
"""


def verify_artifact(path: str, expected_shape: list[int], imgsz: int = IMGSZ) -> dict:
    """Reload `path` in a FRESH interpreter that has only src/ on its path (python -I: no
    PYTHONPATH, no script directory, cwd elsewhere), forward one zero image at `imgsz`, and require
    the FP32 output shape and that torch-pruning / this tool were not needed to unpickle."""
    code = _VERIFY_CODE.format(src=os.path.join(REPO, "src"), path=os.path.abspath(path), imgsz=imgsz)
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise AssertionError(f"{path}: fresh-process load failed:\n{r.stderr[-800:]}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    if info["output_shape"] != list(expected_shape):
        raise AssertionError(f"{path}: output shape {info['output_shape']} != FP32 {list(expected_shape)}")
    if info["leaked_modules"]:
        raise AssertionError(f"{path}: loading needed {info['leaked_modules']}, which the Pi does not have")
    info["size_bytes"] = os.path.getsize(path)
    return info


def fp32_output_shape(model: nn.Module, imgsz: int = IMGSZ) -> list[int]:
    with torch.inference_mode():
        return list(model.eval()(torch.zeros(1, 3, imgsz, imgsz))[0].shape)


# ---- command line --------------------------------------------------------------------------------
def prune_one(split_free: nn.Module, split_counts: dict, base_model: str, ratio: float,
              out_dir: str, params_before: int, macs_before: int, log=print) -> str:
    pruned = prune_model(split_free, ratio)
    art = build_artifact(pruned, base_model, ratio, params_before, macs_before, split_counts=split_counts)
    path = artifact_path(base_model, ratio, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(art, path)
    log(f"ratio {ratio:.2f}: params {art['params_before']:,} -> {art['params_after']:,} "
        f"({art['params_after'] / art['params_before']:.1%}), MACs {art['macs_before'] / 1e9:.2f}G -> "
        f"{art['macs_after'] / 1e9:.2f}G ({art['macs_after'] / art['macs_before']:.1%}); "
        f"saved {path} ({os.path.getsize(path)} bytes)")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base ultralytics weights, e.g. ../yolo11n.pt")
    ap.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS),
                    help="global channel pruning ratios (default 0.30 0.50)")
    ap.add_argument("--out-dir", default=MODELS_DIR)
    ap.add_argument("--verify-only", action="store_true",
                    help="only reload the existing artifacts in a fresh process and check them")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    fp32 = YOLO(args.model).model.eval()
    shape = fp32_output_shape(fp32)
    if not args.verify_only:
        split_free, counts = make_split_free(fp32)
        x = torch.randn(1, 3, IMGSZ, IMGSZ)
        with torch.inference_mode():
            a, b = split_free(x)[0], fp32(x)[0]
        diff = (a - b).abs().max().item()
        print(f"split-free replacement: {counts}, max |output difference| vs FP32 = {diff:.2e}")
        if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
            raise AssertionError(f"split-free model differs from the original by {diff}")
        before = (count_params(fp32), count_macs(fp32))
        print(f"{os.path.basename(args.model)}: params {before[0]:,}, MACs {before[1] / 1e9:.2f}G at {IMGSZ}")
        for ratio in args.ratios:
            prune_one(split_free, counts, args.model, ratio, args.out_dir, *before)
    for ratio in args.ratios:
        path = artifact_path(args.model, ratio, args.out_dir)
        print("verify (fresh process, only src/ on the path):", json.dumps(verify_artifact(path, shape)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
