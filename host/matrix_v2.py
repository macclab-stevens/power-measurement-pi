"""Dataset v2 experiment matrix: families A, A-input, B, C, D, soak, and a small pilot.

Builds manifest rows (same shape as `host/matrix.py` rows, plus `family`, `holdout`, and
`timeout_s` where it differs from the orchestrator default) for `experiment.py run`. Every cell
is a `src/run.py` argument list that passes `--holdout` (so run_meta gets dataset_version 2) and
`--perf-counters on`. Pure stdlib: no torch, so it runs on the EliteBook.

    cells(which="all", seed=7, include_fp16=False, ...) -> rows in round order
    pilot(seed=7, ...)                                     -> one cell per configuration group

Family `ref` re-runs the v1 presets (yolo11n, yolov8n at 1500/2000/2400 MHz x 3) with the v1 args
unchanged plus `--perf-counters off` and no `--holdout`, so they stay dataset_version 1: a kernel
(6.18.50 -> 6.18.60) cross-check against the 224 v1 cells (Ruling R26). Ids `v1ref_<key>_<mhz>_rNN`.

Family D's concurrency cells (`--concurrent yolo11n.pt` / `--concurrent spinner`, run.py Task 13)
are on by default (`include_concurrent=False` leaves them out; experiment.py plan --no-concurrent).

Cell id: `{model}[_{variant}][_{condition}]_{imgsz}_t{threads}[_b{batch}]_{clock}_r{repeat:02d}`
with `clock` = pinned MHz or `dvfs`; pilot ids carry the prefix `pilot_`.

Rounds: round r holds repeat r of every configuration; rows are shuffled within a round with
`random.Random(seed)`. A DVFS cell pins `--cpu-mhz 2400` only around the sync pulses, so its
row's `cpu_mhz` is 2400 as well.

tests/test_matrix_v2_parse.py (Pi suite) parses every generated cell with run.parse_args.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

MAIN_MHZ = 2400
THREADS = 4
IMGSZ = 640
SOAK_EXTRA_S = 20 * 60          # soak timeout = soak length + this
FRAMES_NS, FRAMES_M = 30, 20    # measured frames for n/s models, for m models
DEFAULT_TIMEOUT_S = 40 * 60     # experiment.RUN_TIMEOUT_S; timeout_s is set only above this
CONFIG_FAMILIES = ("A", "A-input", "B", "C", "D", "soak")
FAMILIES = (*CONFIG_FAMILIES, "ref")
CHOICES = (*FAMILIES, "all", "pilot")
REF_KEYS = ("yolo11n", "yolov8n")
REF_MHZ = (1500, 2000, 2400)
REF_REPEATS = 3


@dataclass(frozen=True)
class Config:
    """One configuration group; `repeats` rows (repeat 0..repeats-1) are built from it."""
    family: str
    model: str                          # model_key, e.g. "yolo11s"
    repeats: int
    variant: str = ""                   # int8d int8s fp16 p30 p50
    condition: str = ""                 # fanauto fanoff soak30 serve1 stream empty conc2 ...
    imgsz: int = IMGSZ
    threads: int = THREADS
    batch: int | None = None
    mhz: int = MAIN_MHZ
    dvfs: bool = False
    fan: str = "full"
    bench: bool = False                 # --bench 2 (families A and C) or --no-bench
    frames: int | None = None           # None: 30 (n/s) or 20 (m), batch-scaled
    serve_fps: int | None = None        # None: --serve 10 --serve-gap-ms 200
    soak_minutes: int | None = None
    extra: tuple[str, ...] = field(default=())   # variant / input / concurrency args


def holdout_tag(model: str) -> str | None:
    """Holdout tag for a model key (Global Constraint 5): s models primary, m models secondary."""
    if model.endswith("s"):
        return "primary-s"
    if model.endswith("m"):
        return "secondary-m"
    return None


def _frames(cfg: Config) -> int:
    base = cfg.frames or (FRAMES_M if cfg.model.endswith("m") else FRAMES_NS)
    if cfg.batch and cfg.frames is None:
        return max(5, math.ceil(base / cfg.batch))
    return base


def cell_id(cfg: Config, repeat: int, prefix: str = "") -> str:
    parts = [cfg.model, cfg.variant, cfg.condition, str(cfg.imgsz), f"t{cfg.threads}",
             f"b{cfg.batch}" if cfg.batch else "", "dvfs" if cfg.dvfs else str(cfg.mhz),
             f"r{repeat:02d}"]
    return prefix + "_".join(p for p in parts if p)


def timeout_s(cfg: Config) -> int | None:
    """Soak length + 20 min, only when that exceeds the orchestrator's default timeout."""
    if not cfg.soak_minutes:
        return None
    t = cfg.soak_minutes * 60 + SOAK_EXTRA_S
    return t if t > DEFAULT_TIMEOUT_S else None


def args_for(cfg: Config) -> list[str]:
    tag = holdout_tag(cfg.model)
    a = ["--task", "image", "--model", f"{cfg.model}.pt", "--check-image", "bus.jpg",
         "--sampler", "monsoon", "--perf-counters", "on", "--holdout", tag or "none",
         "--threads", str(cfg.threads), "--imgsz", str(cfg.imgsz), "--cpu-mhz", str(cfg.mhz),
         "--fan", cfg.fan]
    if cfg.dvfs:
        a += ["--governor", "ondemand"]
    if cfg.batch:
        a += ["--batch", str(cfg.batch)]
    a += ["--frames", str(_frames(cfg)), "--warmup", "10"]
    a += ["--bench", "2"] if cfg.bench else ["--no-bench"]
    if cfg.serve_fps:
        a += ["--serve", str(round(20 * cfg.serve_fps)), "--serve-fps", str(cfg.serve_fps)]
    else:
        a += ["--serve", "10", "--serve-gap-ms", "200"]
    if cfg.soak_minutes:
        a += ["--soak-minutes", str(cfg.soak_minutes)]
    return a + list(cfg.extra)


def row(cfg: Config, repeat: int, prefix: str = "") -> dict:
    r = {"cell_id": cell_id(cfg, repeat, prefix), "family": cfg.family, "model_key": cfg.model,
         "cpu_mhz": cfg.mhz, "repeat": repeat, "round": repeat, "holdout": holdout_tag(cfg.model),
         "args": args_for(cfg)}
    t = timeout_s(cfg)
    if t:
        r["timeout_s"] = t
    return r


# ---- configuration groups --------------------------------------------------------------------
_BIG = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
_INT8S = ("--quant", "int8-static")


def _int8s(model: str) -> tuple[str, ...]:
    return (*_INT8S, "--quant-weights", f"models/{model}_int8s.pt")


def configs_a() -> list[Config]:
    return [Config("A", m, reps, mhz=mhz, bench=True)
            for m in _BIG for mhz, reps in ((2400, 5), (2000, 3), (1500, 3))]


def configs_a_input() -> list[Config]:
    return [Config("A-input", "yolo11n", 3, condition=c, bench=True,
                   extra=("--input-image", f"inputs/{c}.jpg")) for c in ("empty", "crowded")]


def configs_b(include_fp16=False, include_int8d=False, include_prune=False) -> list[Config]:
    out = [Config("B", m, 3, variant="int8s", extra=_int8s(m)) for m in ("yolo11n", "yolo11s")]
    out += [Config("B", "yolo11n", 3, variant="int8s", mhz=mhz, extra=_int8s("yolo11n"))
            for mhz in (2000, 1500)]
    if include_fp16:
        out.append(Config("B", "yolo11n", 3, variant="fp16", extra=("--quant", "fp16")))
    if include_int8d:
        out += [Config("B", m, 3, variant="int8d", extra=("--quant", "int8-dynamic"))
                for m in ("yolo11n", "yolo11s", "yolov8n")]
    if include_prune:
        out += [Config("B", "yolo11n", 3, variant=f"p{p}",
                       extra=("--prune", f"models/yolo11n_p{p}.pt")) for p in (30, 50)]
    return out


def configs_c(include_optional=False) -> list[Config]:
    out = [Config("C", m, 3, imgsz=s, bench=True)
           for s in (320, 480, 960) for m in ("yolo11n", "yolo11s", "yolo11m")]
    out += [Config("C", "yolo11n", 3, batch=b, bench=True) for b in (2, 4, 8)]
    out += [Config("C", m, 3, threads=t, bench=True)
            for t in (1, 2) for m in ("yolo11n", "yolo11s")]
    out += [Config("C", "yolo11n", 3, threads=t, imgsz=s, bench=True)
            for t in (1, 2) for s in (320, 960)]
    if include_optional:
        out += [Config("C", "yolo11m", 3, threads=t, bench=True) for t in (1, 2)]
    return out


def configs_d(include_concurrent=True) -> list[Config]:
    out = [Config("D", m, 5, dvfs=True) for m in ("yolo11n", "yolo11s")]
    out += [Config("D", "yolo11n", 3, condition=f"fan{m}", fan=m) for m in ("auto", "off")]
    out += [Config("D", "yolo11n", 3, condition=f"serve{f}", serve_fps=f) for f in (1, 5, 15)]
    out.append(Config("D", "yolo11n", 3, condition="stream", frames=50,
                      extra=("--frames-source", "stream")))
    if include_concurrent:
        out += [Config("D", "yolo11n", 3, condition="conc2",
                       extra=("--concurrent", "yolo11n.pt")),
                Config("D", "yolo11n", 3, condition="concspin", extra=("--concurrent", "spinner"))]
    return out


def configs_soak() -> list[Config]:
    return [Config("soak", m, 1, condition="soak30", fan="auto", soak_minutes=30)
            for m in ("yolo11n", "yolo11m")]


def ref_row(key: str, mhz: int, repeat: int, prefix: str = "") -> dict:
    """A v1 preset cell (args identical to matrix._cell) with perf counters off, no --holdout."""
    import matrix
    v1 = matrix._cell(key, mhz, repeat, repeat)
    return {"cell_id": f"{prefix}v1ref_{key}_{mhz}_r{repeat:02d}", "family": "ref",
            "model_key": key, "cpu_mhz": mhz, "repeat": repeat, "round": repeat, "holdout": None,
            "args": [*v1["args"], "--perf-counters", "off"]}


def configs(which: str, *, include_fp16=False, include_optional=False, include_concurrent=True,
            include_int8d=False, include_prune=False) -> list[Config]:
    """Configuration groups of one family, or of all six."""
    if which == "all":
        return [c for f in CONFIG_FAMILIES for c in configs(
            f, include_fp16=include_fp16, include_optional=include_optional,
            include_concurrent=include_concurrent, include_int8d=include_int8d,
            include_prune=include_prune)]
    if which == "A":
        return configs_a()
    if which == "A-input":
        return configs_a_input()
    if which == "B":
        return configs_b(include_fp16, include_int8d, include_prune)
    if which == "C":
        return configs_c(include_optional)
    if which == "D":
        return configs_d(include_concurrent)
    if which == "soak":
        return configs_soak()
    raise ValueError(f"unknown family {which!r}; choose from {CHOICES}")


def _in_rounds(rows: list[dict], seed: int) -> list[dict]:
    """Round order; shuffled within each round."""
    rng = random.Random(seed)
    out: list[dict] = []
    for r in sorted({x["round"] for x in rows}):
        rnd = [x for x in rows if x["round"] == r]
        rng.shuffle(rnd)
        out += rnd
    return out


def cells(which: str = "all", seed: int = 7, **include) -> list[dict]:
    """Manifest rows for family `which` (A, A-input, B, C, D, soak, ref, all) or the pilot."""
    if which == "pilot":
        return pilot(seed, **include)
    rows = []
    if which != "ref":
        rows = [row(c, rep) for c in configs(which, **include) for rep in range(c.repeats)]
    if which in ("ref", "all"):
        rows += [ref_row(k, mhz, rep) for k in REF_KEYS for mhz in REF_MHZ
                 for rep in range(REF_REPEATS)]
    return _in_rounds(rows, seed)


# ---- pilot -----------------------------------------------------------------------------------
def pilot(seed: int = 7, *, include_fp16=False, include_optional=False, include_concurrent=True,
          include_int8d=False, include_prune=False) -> list[dict]:
    """One cell per distinct configuration group (two DVFS repeats), a 5-minute soak, and one v1ref
    cell per key at 2400 MHz.

    `include_optional` is accepted for CLI symmetry; the optional C cells add no pilot group."""
    del include_optional
    groups = [Config("A", m, 1, bench=True) for m in _BIG]
    groups += [Config("A", "yolo11s", 1, mhz=mhz, bench=True) for mhz in (1500, 2000)]
    groups += [replace(c, repeats=1) for c in configs_a_input()]
    groups += [replace(c, repeats=1) for c in configs_b(include_fp16, include_int8d, include_prune)
               if not (c.variant == "int8s" and c.mhz == 2000)]
    groups += [Config("C", "yolo11n", 1, imgsz=s, bench=True) for s in (320, 960)]
    groups += [Config("C", "yolo11m", 1, imgsz=960, bench=True)]
    groups += [Config("C", "yolo11n", 1, batch=b, bench=True) for b in (2, 8)]
    groups += [Config("C", "yolo11n", 1, threads=t, bench=True) for t in (1, 2)]
    groups += [Config("C", "yolo11n", 1, threads=1, imgsz=960, bench=True)]
    groups += [replace(c, repeats=2 if c.dvfs else 1) for c in configs_d(include_concurrent)
               if (c.model, c.condition) not in (("yolo11s", ""), ("yolo11n", "serve5"))]
    groups.append(Config("soak", "yolo11n", 1, condition="soak5", fan="auto", soak_minutes=5))
    rows = [row(c, rep, prefix="pilot_") for c in groups for rep in range(c.repeats)]
    rows += [ref_row(k, MAIN_MHZ, 0, prefix="pilot_") for k in REF_KEYS]
    return _in_rounds(rows, seed)

