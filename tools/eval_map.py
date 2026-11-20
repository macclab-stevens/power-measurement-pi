"""One-time mAP of a YOLO model / variant / imgsz on the fixed 500-image COCO val2017 subset.

Writes ../data/map_reference.json entries keyed "<variant_id>@<imgsz>" (src/yolo_variants.py
`variant_id`); src/run.py looks them up and copies them into every cell's run_meta["map_reference"].
Entry: {map50_95, map50, n_images, dataset, tool_version (git short hash, '-dirty' if tracked files were modified), measured (ISO date)}.

Every entry is measured by ONE custom loop so FP32, fp16, INT8 and pruned variants are comparable:
the Pi's own preprocess (src/tasks.py ImageTask.preprocess: square letterbox, pad 114, centred, /255),
the model forward, ultralytics NMS (conf 0.001, IoU 0.7, multi-label, max 300 detections, as in
`.val()`) and ultralytics' own matching and `ap_per_class` over the COCO IoU thresholds
0.50:0.05:0.95. `--sanity` runs an FP32 model through this loop AND through ultralytics `.val()`
and checks mAP50-95 agrees within 0.01.

  cd tools
  uv run python eval_map.py --model ../yolo11n.pt --imgsz 640
  uv run python eval_map.py --model ../yolo11n.pt --quant fp16 --imgsz 640
  uv run python eval_map.py --model ../yolo11n.pt --artifact ../models/yolo11n_int8s.pt --imgsz 640
  uv run python eval_map.py --model ../yolo11n.pt --imgsz 640 --sanity      # loop vs .val(), writes nothing
  uv run python eval_map.py --all [--skip-existing]

`--all`: FP32 n/s/m of both generations at 640; yolo11 n/s/m at 320, 480, 960; fp16 yolo11n@640; and
every ../models/*.pt artifact (INT8 static, pruned) at 640. int8-dynamic is not in `--all`: it
quantizes 0 modules on YOLO, so its mAP would be FP32's (`--quant int8-dynamic` still works and
prints the quantized-module count). A single evaluation is aborted (exit 2) if the projected time
exceeds --budget-min; the table is merged after every evaluation, so reruns can `--skip-existing`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import yolo_variants as yv  # noqa: E402  (Task 5: variant_id, artifact formats)
from tasks import ImageTask, _to_float32, read_rgb  # noqa: E402  (the Pi's preprocess)

DATASET = "coco-val2017-first500"
MAP_REFERENCE_PATH = os.path.join(REPO, "data", "map_reference.json")
DATA_DIR = os.path.join(HERE, "data")
CONF_THRES, IOU_THRES, MAX_DET = 0.001, 0.7, 300
SANITY_TOLERANCE = 0.01
DEFAULT_BUDGET_MIN = 40.0
PROBE_IMAGES = 20
GEN_N_S_M = [f"{g}{s}" for g in ("yolo11", "yolov8") for s in "nsm"]


class TooSlow(RuntimeError):
    """The projected time of one evaluation exceeds the budget; carries the measured rate."""

    def __init__(self, s_per_image: float, n_images: int, budget_s: float):
        self.s_per_image = s_per_image
        super().__init__(f"{s_per_image:.2f} s/image x {n_images} images = "
                         f"{s_per_image * n_images / 60:.0f} min > budget {budget_s / 60:.0f} min")


# ---- table entries ----------------------------------------------------------------------------
def entry_key(variant: str, imgsz: int) -> str:
    return f"{variant}@{imgsz}"


def tool_version(run=subprocess.run) -> str:
    """Git short hash of this checkout, '<hash>-dirty' if tracked files have uncommitted changes
    (the numbers then come from code that is not that commit), or 'unknown'."""
    try:
        r = run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return "unknown"
        head = r.stdout.strip()
        s = run(["git", "-C", REPO, "status", "--porcelain", "-uno"], capture_output=True, text=True,
                timeout=30)
        return f"{head}-dirty" if s.returncode == 0 and s.stdout.strip() else head
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_entry(map50_95: float, map50: float, n_images: int, tool_version: str, measured: str) -> dict:
    entry = {"map50_95": round(float(map50_95), 5), "map50": round(float(map50), 5),
             "n_images": int(n_images), "dataset": DATASET, "tool_version": tool_version,
             "measured": measured}
    assert set(entry) == set(yv.MAP_REFERENCE_FIELDS)
    return entry


def load_table(path: str) -> dict:
    """The table, {} if the file does not exist; ValueError if it exists but is not a JSON object."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            table = json.load(f)
        except ValueError as e:
            raise ValueError(f"{path}: not valid JSON ({e}); refusing to overwrite it") from e
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a JSON object, refusing to overwrite it")
    return table


def merge_entry(path: str, key: str, entry: dict) -> dict:
    """Set one key, keep every other key, rewrite atomically (sorted keys, stable diffs)."""
    table = load_table(path)
    table[key] = entry
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return table


# ---- metric -----------------------------------------------------------------------------------
class MapAccumulator:
    """COCO-style mAP over images, using ultralytics' own matching and `ap_per_class`."""

    def __init__(self):
        from types import SimpleNamespace
        self._iouv = torch.linspace(0.5, 0.95, 10)
        self._ns = SimpleNamespace(iouv=self._iouv)   # match_predictions only reads self.iouv
        self._tp, self._conf, self._pred_cls, self._target_cls = [], [], [], []
        self.n_images = 0

    def add_image(self, pred: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor) -> None:
        """pred: (n, 6) xyxy, conf, cls; gt_boxes: (m, 4) xyxy; gt_cls: (m,); one coordinate frame."""
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import box_iou
        self.n_images += 1
        self._target_cls.append(gt_cls.cpu().numpy())
        if pred.shape[0] == 0:
            return
        if gt_boxes.shape[0] == 0:
            tp = np.zeros((pred.shape[0], 10), dtype=bool)
        else:
            iou = box_iou(gt_boxes.float(), pred[:, :4].float())
            tp = BaseValidator.match_predictions(self._ns, pred[:, 5], gt_cls, iou).cpu().numpy()
        self._tp.append(tp)
        self._conf.append(pred[:, 4].cpu().numpy())
        self._pred_cls.append(pred[:, 5].cpu().numpy())

    def result(self) -> dict:
        from ultralytics.utils.metrics import ap_per_class
        target = np.concatenate(self._target_cls) if self._target_cls else np.zeros(0)
        if not self._tp or target.size == 0:
            return {"map50_95": 0.0, "map50": 0.0, "n_images": self.n_images}
        ap = ap_per_class(np.concatenate(self._tp), np.concatenate(self._conf),
                          np.concatenate(self._pred_cls), target)[5]          # (nc_with_labels, 10)
        return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()), "n_images": self.n_images}


# ---- preprocessing shared with the Pi ---------------------------------------------------------
def make_preprocess(imgsz: int, half: bool):
    """uint8 HWC RGB -> 1x3xSxS exactly as the Pi does it: ImageTask.preprocess (detection mode).

    ImageTask is configured through its attributes so the letterbox is shared, not copied."""
    task = ImageTask()
    task._imgsz, task._is_detection, task._half = imgsz, True, half
    return task.preprocess


def letterbox_geometry(h: int, w: int, imgsz: int) -> tuple[int, int, int, int]:
    """(nw, nh, left, top) of the resized image inside the letterbox canvas (mirrors preprocess)."""
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    return nw, nh, (imgsz - nw) // 2, (imgsz - nh) // 2


def unletterbox(boxes: torch.Tensor, h: int, w: int, imgsz: int) -> torch.Tensor:
    """xyxy boxes in letterbox pixels -> original image pixels, clipped to the image."""
    nw, nh, left, top = letterbox_geometry(h, w, imgsz)
    out = boxes.clone().float()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) * (w / nw)
    out[:, [1, 3]] = (out[:, [1, 3]] - top) * (h / nh)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
    return out


def read_labels(path: str, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """YOLO txt (cls xc yc w h, normalised) -> ((m, 4) xyxy pixels, (m,) classes); missing file = none."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [ln.split() for ln in f if ln.strip()]
    if not rows:
        return torch.zeros((0, 4)), torch.zeros(0)
    a = torch.tensor([[float(v) for v in r[:5]] for r in rows])
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return torch.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], dim=1), a[:, 0]


# ---- the custom evaluation loop ---------------------------------------------------------------
def evaluate(module, imgsz: int, half: bool, images_dir: str, labels_dir: str, log=print,
             budget_s: float | None = None, probe_images: int = PROBE_IMAGES,
             clock=time.perf_counter) -> dict:
    """Run `module` over every *.jpg in images_dir and return {map50_95, map50, n_images}.

    Raises TooSlow (after `probe_images` images) when the projected total exceeds budget_s."""
    from ultralytics.utils.nms import non_max_suppression
    names = sorted(n for n in os.listdir(images_dir) if n.lower().endswith(".jpg"))
    if not names:
        raise ValueError(f"no *.jpg in {images_dir}")
    preprocess = make_preprocess(imgsz, half)
    acc = MapAccumulator()
    t0 = clock()
    for i, name in enumerate(names, 1):
        frame = read_rgb(os.path.join(images_dir, name))
        h, w = frame.shape[:2]
        with torch.inference_mode():
            out = module(preprocess(frame))
            if half:
                out = _to_float32(out)
            det = non_max_suppression(out, conf_thres=CONF_THRES, iou_thres=IOU_THRES,
                                      multi_label=True, max_det=MAX_DET)[0]
        pred = det.clone()
        pred[:, :4] = unletterbox(det[:, :4], h, w, imgsz)
        gt_boxes, gt_cls = read_labels(os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt"), h, w)
        acc.add_image(pred, gt_boxes, gt_cls)
        elapsed = clock() - t0
        if budget_s is not None and i == probe_images and elapsed / i * len(names) > budget_s:
            raise TooSlow(elapsed / i, len(names), budget_s)
        if i % 50 == 0 or i == len(names):
            log(f"  {i}/{len(names)} images, {elapsed:.0f} s ({elapsed / i:.2f} s/image)")
    return acc.result()


def ultralytics_val(model_path: str, imgsz: int, data_yaml: str) -> dict:
    """Reference `.val()` for the FP32 sanity check (rect batch=1, ultralytics' own preprocess)."""
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as tmp:
        r = YOLO(model_path).val(data=data_yaml, imgsz=imgsz, batch=1, device="cpu", conf=CONF_THRES,
                                 iou=IOU_THRES, plots=False, verbose=False, project=tmp, name="val",
                                 save=False)
    return {"map50_95": float(r.box.map), "map50": float(r.box.map50)}


# ---- jobs: what to evaluate -------------------------------------------------------------------
@dataclass
class Job:
    variant: str
    model: str                      # base ultralytics weights path
    imgsz: int
    quant: str = "fp32"
    artifact: str | None = None
    prune_ratio: float | None = None


def fp32_job(model_path: str, imgsz: int, quant: str = "fp32") -> Job:
    return Job(yv.variant_id(model_path, quant), model_path, imgsz, quant)


def artifact_job(path: str, root: str, imgsz: int, model: str | None = None) -> Job:
    """Identify a Task 8/9 artifact by its `format` (int8s / pruned) and its `base_model`."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    fmt = obj.get("format") if isinstance(obj, dict) else None
    if fmt not in (yv.INT8S_FORMAT, yv.PRUNED_FORMAT):
        raise ValueError(f"{path}: unknown artifact format {fmt!r}")
    art = yv.validate_artifact(obj, fmt, path)
    base = art.get("base_model") or model
    if not base:
        raise ValueError(f"{path}: artifact has no base_model; pass --model")
    base_path = model if model and os.path.basename(model) == os.path.basename(base) \
        else os.path.join(root, os.path.basename(base))
    if fmt == yv.INT8S_FORMAT:
        return Job(yv.variant_id(base_path, "int8-static"), base_path, imgsz, "int8-static", path)
    ratio = float(art["ratio"])
    return Job(yv.variant_id(base_path, "fp32", ratio), base_path, imgsz, "fp32", path, ratio)


def discover_artifacts(root: str) -> list[str]:
    d = os.path.join(root, "models")
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".pt")) if os.path.isdir(d) else []


def all_jobs(root: str, artifacts: list[str]) -> list[Job]:
    w = lambda name: os.path.join(root, f"{name}.pt")  # noqa: E731
    jobs = [fp32_job(w(m), 640) for m in GEN_N_S_M]
    jobs += [fp32_job(w(f"yolo11{s}"), sz) for sz in (320, 480, 960) for s in "nsm"]
    jobs.append(fp32_job(w("yolo11n"), 640, "fp16"))
    jobs += [artifact_job(p, root, 640) for p in artifacts]
    return jobs


# ---- loading and running a job ----------------------------------------------------------------
def ensure_weights(path: str) -> None:
    """Download an ultralytics weight into `path` if it is missing (*.pt is gitignored)."""
    if not os.path.exists(path):
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def load_job_module(job: Job) -> tuple[torch.nn.Module, bool, int | None]:
    """(eval-mode module, half, quantized-module count or None) for the variant a job names."""
    if job.artifact:
        torch.backends.quantized.engine = "qnnpack"    # as run.py does for INT8 artifacts
        fmt = yv.INT8S_FORMAT if job.quant == "int8-static" else yv.PRUNED_FORMAT
        module = yv.load_artifact(job.artifact, fmt)["model"].eval()
        return module, False, (yv.count_quantized_modules(module) if job.quant == "int8-static" else None)
    from ultralytics import YOLO
    ensure_weights(job.model)
    module = YOLO(job.model).model.eval()
    if job.quant == "fp16":
        return module.half(), True, None
    if job.quant == "int8-dynamic":
        q, n = yv.quantize_dynamic_counted(module)
        return q.eval(), False, n
    return module, False, None


def run_job(job: Job, data_dir: str, table_path: str | None, budget_s: float, log=print) -> dict:
    """Evaluate one job, merge its entry into table_path (when given) and return a result row."""
    key = entry_key(job.variant, job.imgsz)
    log(f"[{key}] start")
    t0 = time.perf_counter()
    module, half, nq = load_job_module(job)
    if job.quant in ("int8-dynamic", "int8-static"):
        log(f"[{key}] quantized modules: {nq}")
        for err in yv.quant_count_errors(job.quant, nq):
            log(f"[{key}] WARNING: {err}")
    res = evaluate(module, job.imgsz, half, os.path.join(data_dir, "coco500", "images"),
                   os.path.join(data_dir, "coco500", "labels"), log=log, budget_s=budget_s)
    wall = time.perf_counter() - t0
    row = {"key": key, **res, "wall_s": round(wall, 1), "quantized_modules": nq}
    if table_path:
        entry = make_entry(res["map50_95"], res["map50"], res["n_images"], tool_version(),
                           datetime.date.today().isoformat())
        merge_entry(table_path, key, entry)
    log(f"RESULT {key} mAP50-95={res['map50_95']:.4f} mAP50={res['map50']:.4f} "
        f"n={res['n_images']} wall={wall:.0f}s")
    return row


def sanity(job: Job, data_dir: str, budget_s: float, log=print) -> dict:
    """FP32 model through the custom loop and through ultralytics .val(): the difference."""
    if job.quant != "fp32" or job.artifact:
        raise ValueError("--sanity is for plain FP32 models")
    row = run_job(job, data_dir, None, budget_s, log)
    t0 = time.perf_counter()
    ref = ultralytics_val(job.model, job.imgsz, os.path.join(data_dir, "coco500.yaml"))
    diff = row["map50_95"] - ref["map50_95"]
    log(f"SANITY {job.variant}@{job.imgsz}: loop mAP50-95={row['map50_95']:.4f} "
        f".val()={ref['map50_95']:.4f} diff={diff:+.4f} (tolerance {SANITY_TOLERANCE}); "
        f"loop mAP50={row['map50']:.4f} .val() mAP50={ref['map50']:.4f}; .val() wall "
        f"{time.perf_counter() - t0:.0f}s")
    return {"loop": row, "val": ref, "diff": diff, "ok": abs(diff) <= SANITY_TOLERANCE}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="base ultralytics weights, e.g. ../yolo11n.pt")
    ap.add_argument("--quant", choices=["fp32", "fp16", "int8-dynamic"], default="fp32")
    ap.add_argument("--artifact", help="Task 8/9 artifact (models/*.pt); variant id comes from it")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--all", action="store_true", help="the plan's full set (see module docstring)")
    ap.add_argument("--skip-existing", action="store_true", help="skip keys already in the table")
    ap.add_argument("--sanity", action="store_true", help="FP32 loop vs ultralytics .val(); writes nothing")
    ap.add_argument("--out", default=MAP_REFERENCE_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--budget-min", type=float, default=DEFAULT_BUDGET_MIN,
                    help="abort one evaluation whose projected time exceeds this (exit 2)")
    args = ap.parse_args(argv)
    budget_s = args.budget_min * 60

    if args.all:
        jobs = all_jobs(REPO, discover_artifacts(REPO))
    elif args.artifact:
        jobs = [artifact_job(args.artifact, REPO, args.imgsz, args.model)]
    elif args.model:
        jobs = [fp32_job(args.model, args.imgsz, args.quant)]
    else:
        ap.error("give --model, --artifact or --all")

    def log(msg):
        print(msg, flush=True)

    existing = load_table(args.out)
    rc = 0
    for job in jobs:
        if args.skip_existing and entry_key(job.variant, job.imgsz) in existing:
            log(f"[{entry_key(job.variant, job.imgsz)}] already in table, skipped")
            continue
        try:
            if args.sanity:
                ensure_weights(job.model)
                rc = rc or (0 if sanity(job, args.data_dir, budget_s, log)["ok"] else 1)
            else:
                run_job(job, args.data_dir, args.out, budget_s, log)
        except TooSlow as e:
            log(f"[{entry_key(job.variant, job.imgsz)}] STOPPED: {e}")
            return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
