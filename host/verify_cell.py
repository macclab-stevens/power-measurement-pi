#!/usr/bin/env python3
"""Check one aligned Monsoon run (schema 4) and write its phase_energy.csv.

Reads run_meta.json, alignment.json and power_trace.parquet from the run directory (align.py
must have run). Fatal checks make `ok` false; soft ones go to `warnings`.

Concurrent cells (run_meta["concurrent"], align.py's `tag` / `step_<tag>` columns) keep the
primary's step checks (the `step` column is the primary workload's) and add one check per tag:
it completed iterations and left a step column; each non-primary tag's steps are present and
cover its active window; and the primary was inside a step during >= 50 % of every other tag's window (the workloads really overlapped).

v2 cells (dataset_version 2) add: fan mode, DVFS clock range (governor ondemand), perf counter
coverage, thermal limits and a mAP-reference warning. Every cell also gets derived.json
(energy and latency per frame, measured iterations only).

    uv run python verify_cell.py ../runs/<cell> [--cpu-mhz 2400]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

P_MIN_W, P_MAX_W = 1.0, 25.0
MAX_NAN_FRACTION = 0.01
STEP_COVERAGE_FAIL, STEP_COVERAGE_WARN = 0.5, 0.9
ENERGY_CLOSURE = 0.01
INFER_STEPS = ("preprocess", "forward", "postprocess")
TRAIN_STEPS = ("dataload", "zero_grad", "forward", "loss", "backward", "optimizer_step")
THERMAL_ABORT_C = 85.0                 # same limit run.py enforces on the Pi
COUNTER_COVERAGE_MIN = 0.95
DVFS_RANGE_MHZ, DVFS_TOL_MHZ = (1500, 2400), 25
FAN_MODES = ("full", "auto", "off")
OVERLAP_FAIL, OVERLAP_WARN = 0.5, 0.9
PRIMARY_TAG = "m0"
SPIN_STEPS = ("spin",)                 # tag "spin"; "m1" is a second image model: INFER_STEPS


def expected_phases(meta: dict) -> list[str]:
    modes = list(meta.get("modes") or [])
    ph = ["load_weights", *[f"first:{m}" for m in modes], "sync:start", "baseline_A",
          *modes, "baseline_B", "sync:end"]
    if meta.get("warmup_passes"):
        ph.append("warmup")
    if meta.get("serve_requests"):
        ph.append("serve")
    return ph


def expected_steps(meta: dict) -> tuple[str, ...]:
    return TRAIN_STEPS if meta.get("task") == "train" else INFER_STEPS


def phase_energy(df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Energy per (phase, iter, step); ctx_read samples counted and flagged, not dropped.

    Missing (NaN) samples count toward the duration but add no energy (same policy as the
    total), so E_mJ is the sum of the observed samples x dt and mean_P_W their mean."""
    g = df.groupby(["phase", "iter", "step"], sort=False, dropna=False)
    out = g.agg(n_samples=("P_W", "size"), n_finite=("P_W", "count"),
                P_sum=("P_W", "sum"), mean_P_W=("P_W", "mean"),
                ctx_read_frac=("ctx_read", "mean"),
                t_start_s=("t_s", "min"), t_end_s=("t_s", "max")).reset_index()
    out["dur_ms"] = out["n_samples"] * dt_s * 1000.0
    out["E_mJ"] = out.pop("P_sum") * dt_s * 1000.0
    return out


def _frame_table(pe: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per measured iteration (frame >= 1) of one mode: energy E_mJ and duration dur_ms.

    Frame 0 is the first call of the mode (cold caches, lazy init) and is excluded."""
    rows = pe[pe["phase"] == mode]
    frame = pd.to_numeric(rows["iter"], errors="coerce")
    rows = rows[frame >= 1].assign(frame=frame[frame >= 1])
    return rows.groupby("frame")[["E_mJ", "dur_ms"]].sum()


def _concurrent_summary(conc: dict) -> dict:
    """What derived.json says about a concurrent cell: energy_per_frame_mJ is the whole board's
    draw per primary frame (both workloads), not the primary's own."""
    return {"mode": conc.get("mode"), "model": conc.get("model"), "tags": conc.get("tags"),
            "threads_split": conc.get("threads_split"), "iterations": conc.get("iterations"),
            "latency_ms": conc.get("latency_ms"),
            "note": "energy per frame covers both workloads running together"}


def derived_metrics(pe: pd.DataFrame, meta: dict, baseline_P_W: float) -> dict:
    """Energy and latency per frame from the per-(phase, iter, step) energy table.

    dyn_energy subtracts baseline_P_W x iteration duration (W x ms = mJ). Single-mode cells
    (image, train) also get the headline scalars; by_mode always has every mode's values."""
    batch = int(meta.get("batch") or 1)
    modes = list(meta.get("modes") or [])
    by_mode = {}
    for m in modes:
        t = _frame_table(pe, m)
        if t.empty:
            by_mode[m] = {"n_frames": 0, "energy_per_frame_mJ": None,
                          "dyn_energy_per_frame_mJ": None, "latency_ms_per_frame": None}
            continue
        e = float(t["E_mJ"].mean())
        by_mode[m] = {"n_frames": int(len(t)), "energy_per_frame_mJ": e,
                      "dyn_energy_per_frame_mJ": float((t["E_mJ"] - baseline_P_W * t["dur_ms"]).mean()),
                      "latency_ms_per_frame": float(t["dur_ms"].mean())}
    head = by_mode[modes[0]] if len(modes) == 1 else {}
    e = head.get("energy_per_frame_mJ")
    out = {"energy_per_frame_mJ": e,
            "dyn_energy_per_frame_mJ": head.get("dyn_energy_per_frame_mJ"),
            "latency_ms_per_frame": head.get("latency_ms_per_frame"),
            "energy_per_image_mJ": e / batch if e is not None else None,
            "n_frames": head.get("n_frames"), "batch": batch, "baseline_mean_P_W": baseline_P_W,
            "by_mode": by_mode, "variant": meta.get("variant"), "holdout": meta.get("holdout"),
            "map_reference": meta.get("map_reference")}
    if meta.get("concurrent"):
        out["concurrent"] = _concurrent_summary(meta["concurrent"])
    return out


def _active_window(col: pd.Series) -> tuple[int, int]:
    """Row positions [lo, hi) from the first to the last labelled sample (rows are in time
    order); (0, 0) if the column is never labelled."""
    idx = np.flatnonzero((col != "").to_numpy())
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 0)


def concurrent_checks(in_modes: pd.DataFrame, conc: dict) -> tuple[list, list, dict]:
    """Per-tag checks of a --concurrent cell: (errors, warnings, info). Every tag needs
    iterations and a step_<tag> column; the other tags also need their steps, coverage of their
    active window and overlap with the primary (the primary's own steps and coverage are checked
    by verify() on the `step` column, so they are not reported twice)."""
    errs: list[str] = []
    warns: list[str] = []
    info: dict = {"coverage": {}, "overlap": {}}
    tags = list(conc.get("tags") or [])
    if "tag" not in in_modes.columns:
        return ["concurrent cell without a tag column (events.csv had no tagged steps)"], warns, info
    iters = conc.get("iterations") or {}
    for t in tags:
        if not iters.get(t):
            errs.append(f"concurrent workload {t} completed no iterations")
        col = in_modes.get(f"step_{t}")
        if col is None:
            errs.append(f"no step_{t} column: workload {t} left no step events")
            continue
        if t == PRIMARY_TAG:
            continue
        seen = set(col.unique()) - {""}
        want = SPIN_STEPS if t == "spin" else INFER_STEPS
        missing = [x for x in want if x not in seen]
        if missing:
            errs.append(f"steps without samples for workload {t}: {missing}")
        lo, hi = _active_window(col)
        win = col.iloc[lo:hi]
        cov = float((win != "").mean()) if len(win) else 0.0
        info["coverage"][t] = cov
        if cov < STEP_COVERAGE_FAIL:
            errs.append(f"step coverage of workload {t} {cov:.2f} < {STEP_COVERAGE_FAIL}")
        elif cov < STEP_COVERAGE_WARN:
            warns.append(f"step coverage of workload {t} {cov:.2f} < {STEP_COVERAGE_WARN}")
        if PRIMARY_TAG in tags and f"step_{PRIMARY_TAG}" in in_modes:
            span = in_modes[f"step_{PRIMARY_TAG}"].iloc[lo:hi]
            ov = float((span != "").mean()) if len(span) else 0.0
            info["overlap"][t] = ov
            if ov < OVERLAP_FAIL:
                errs.append(f"workloads {PRIMARY_TAG} and {t} overlap only {ov:.2f} of {t}'s window")
            elif ov < OVERLAP_WARN:
                warns.append(f"workloads {PRIMARY_TAG} and {t} overlap {ov:.2f} < {OVERLAP_WARN}")
    return errs, warns, info


def _is_yolo(meta: dict) -> bool:
    return Path(str(meta.get("model") or "")).name.lower().startswith("yolo")


def _fan_errors(meta: dict) -> list[str]:
    """full: fan_full must be set. auto/off: the mode must be recorded. Anything else (incl. a
    v1-style meta without fan_full) is the old "fan_full not set" error."""
    mode = meta.get("fan_mode", "full" if meta.get("fan_full") else "none")
    if mode == "full":
        return [] if meta.get("fan_full") else ["fan_full not set (fan_mode full)"]
    if mode in FAN_MODES:
        return []
    return [f"fan_full not set (fan_mode {mode})"]


def _clock_range_errors(in_modes: pd.DataFrame) -> list[str]:
    cols = [c for c in in_modes.columns if c.startswith("sys_cpu") and c.endswith("_kHz")]
    vals = in_modes[cols].to_numpy(dtype=np.float64) if cols else np.empty((0, 0))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return ["no cpu frequency readings in mode phases"]
    lo, hi = (DVFS_RANGE_MHZ[0] - DVFS_TOL_MHZ) * 1000, (DVFS_RANGE_MHZ[1] + DVFS_TOL_MHZ) * 1000
    if vals.min() < lo or vals.max() > hi:
        return [f"cpu clock outside {DVFS_RANGE_MHZ[0]}-{DVFS_RANGE_MHZ[1]} MHz "
                f"(+/-{DVFS_TOL_MHZ}): min {vals.min() / 1000:.0f} max {vals.max() / 1000:.0f} MHz"]
    return []


def _thermal_errors(run_dir: Path, meta: dict | None) -> list[str]:
    errs = []
    t = None
    try:
        t = json.loads((run_dir / "thermal_abort.json").read_text())
    except OSError:
        pass
    except ValueError:
        t = {}
    if t is None and meta and meta.get("thermal_abort"):
        t = meta["thermal_abort"]
    if t is not None:
        temp = t.get("temp_C") if isinstance(t, dict) else None
        errs.append(f"thermal abort at {temp} C")
    max_t = (meta or {}).get("max_temp_C")
    if max_t is not None and max_t >= THERMAL_ABORT_C:
        errs.append(f"max_temp_C {max_t} >= {THERMAL_ABORT_C:g}")
    return errs


def verify(run_dir, cpu_mhz: int | None = None, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    res: dict = {"run_dir": str(run_dir), "errors": errors, "warnings": warnings}

    try:
        meta = json.loads((run_dir / "run_meta.json").read_text())
        ali = json.loads((run_dir / "alignment.json").read_text())
        df = pd.read_parquet(run_dir / "power_trace.parquet")
    except (OSError, ValueError) as e:
        errors.append(f"missing or unreadable output: {e}")
        errors.extend(_thermal_errors(run_dir, None))   # an aborted run leaves no run_meta.json
        res["ok"] = False
        return res

    if meta.get("schema_version") != 4:
        errors.append(f"schema_version {meta.get('schema_version')} != 4")
    if cpu_mhz is not None and meta.get("cpu_mhz_pinned") != cpu_mhz:
        errors.append(f"cpu_mhz_pinned {meta.get('cpu_mhz_pinned')} != {cpu_mhz}")
    errors.extend(_fan_errors(meta))
    if not ali.get("pass"):
        errors.append(f"alignment failed: {ali.get('fail_reasons')}")
    if ali.get("slip_warning"):
        warnings.append(f"slip_warning: {ali.get('slip_samples_total', 0)} samples lost")
    if ali.get("drift_warning"):
        warnings.append(f"drift_warning: {ali.get('drift_ppm'):.1f} ppm")

    n = len(df)
    if n == 0:
        errors.append("power_trace is empty")
        res["ok"] = False
        return res
    P = df["P_W"].to_numpy(dtype=np.float64)
    nan_frac = float(np.isnan(P).mean())
    if nan_frac > MAX_NAN_FRACTION:
        errors.append(f"NaN fraction {nan_frac:.3f} > {MAX_NAN_FRACTION}")
    n_inf = int(np.isinf(P).sum())
    if n_inf:
        errors.append(f"{n_inf} infinite power samples")
    fin = P[np.isfinite(P)]
    if fin.size == 0:
        errors.append("no finite power samples")
        res["ok"] = False
        return res
    lo, hi = float(np.percentile(fin, 0.1)), float(np.percentile(fin, 99.9))
    if lo < P_MIN_W or hi > P_MAX_W:
        errors.append(f"power out of range: p0.1={lo:.2f} W p99.9={hi:.2f} W "
                      f"(allowed {P_MIN_W}-{P_MAX_W})")

    counts = df["phase"].value_counts()
    missing = [p for p in expected_phases(meta) if counts.get(p, 0) == 0]
    if missing:
        errors.append(f"phases without samples: {missing}")
    benches_off = meta.get("bench_skipped") or meta.get("bench_forced_off")
    if not benches_off and not any(str(p).startswith("bench:") for p in counts.index):
        errors.append("no bench:* phase although benches were not skipped")

    modes = list(meta.get("modes") or [])
    in_modes = df[df["phase"].isin(modes)]
    seen_steps = set(in_modes["step"].unique())
    miss_steps = [s for s in expected_steps(meta) if s not in seen_steps]
    if miss_steps:
        errors.append(f"steps without samples in mode phases: {miss_steps}")
    cov = float((in_modes["step"] != "").mean()) if len(in_modes) else 0.0
    if cov < STEP_COVERAGE_FAIL:
        errors.append(f"step coverage in mode phases {cov:.2f} < {STEP_COVERAGE_FAIL}")
    elif cov < STEP_COVERAGE_WARN:
        warnings.append(f"step coverage in mode phases {cov:.2f} < {STEP_COVERAGE_WARN}")

    if meta.get("concurrent"):
        c_err, c_warn, res["concurrent"] = concurrent_checks(in_modes, meta["concurrent"])
        errors.extend(c_err)
        warnings.extend(c_warn)

    v2 = meta.get("dataset_version") == 2
    if meta.get("governor") == "ondemand":
        errors.extend(_clock_range_errors(in_modes))
    if v2:
        pc = meta.get("perf_counters") or {}
        if not pc.get("enabled"):
            errors.append("perf counters not enabled for a dataset_version 2 cell")
        else:
            ctr = (in_modes["ctr_instructions"].notna().mean()
                   if "ctr_instructions" in in_modes and len(in_modes) else 0.0)
            res["counter_coverage"] = float(ctr)
            if ctr < COUNTER_COVERAGE_MIN:
                errors.append(f"counter coverage {ctr:.3f} < {COUNTER_COVERAGE_MIN}")
    errors.extend(_thermal_errors(run_dir, meta))
    if _is_yolo(meta) and (v2 or "variant" in meta):
        ref = meta.get("map_reference")
        if not ref:
            warnings.append(f"mAP reference missing for {meta.get('variant')}@{meta.get('imgsz')}")
        elif (ref.get("measured_on") or {}).get("machine") != "aarch64":
            warnings.append(f"mAP reference not measured on the Pi "
                            f"({meta.get('variant')}@{meta.get('imgsz')})")

    dt_s = 1.0 / float(ali.get("slip_rate_hz") or 5000.0)
    pe = phase_energy(df, dt_s)
    e_total = float(fin.sum()) * dt_s * 1000.0
    e_sum = float(np.nansum(pe["E_mJ"]))
    closure = abs(e_sum - e_total) / e_total if e_total else float("nan")
    if not np.isfinite(closure) or closure > ENERGY_CLOSURE:
        errors.append(f"energy closure {closure:.4f} > {ENERGY_CLOSURE} (or undefined)")
    if write:
        pe.to_csv(run_dir / "phase_energy.csv", index=False)

    by_phase = df.groupby("phase")["P_W"].mean()
    by_step = in_modes.groupby("step")["P_W"].mean()
    res.update({
        "n_samples": n, "nan_fraction": nan_frac, "step_coverage": cov,
        "energy_total_J": e_total / 1000.0, "energy_closure": closure,
        "mean_P_W": {k: round(float(by_phase[k]), 4) for k in modes if k in by_phase},
        "baseline_P_W": round(float(by_phase.get("baseline_A", np.nan)), 4),
        "step_P_W": {k: round(float(v), 4) for k, v in by_step.items() if k},
        "max_residual_ms": ali.get("max_residual_ms"),
        "slip_samples_total": ali.get("slip_samples_total", 0),
    })
    if meta.get("max_temp_C") is not None:
        res["max_temp_C"] = meta["max_temp_C"]
    bl = [by_phase[k] for k in ("baseline_A", "baseline_B") if k in by_phase]
    base_w = float(np.mean(bl)) if bl else 0.0       # same definition as run_meta baseline_W
    res["derived"] = derived_metrics(pe, meta, base_w)
    if modes and not any(v["n_frames"] for v in res["derived"]["by_mode"].values()):
        warnings.append("no measured iterations (frame >= 1): derived metrics are empty")
    if write:
        (run_dir / "derived.json").write_text(json.dumps(res["derived"], indent=2))
    res["ok"] = not errors
    return res


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--cpu-mhz", type=int, default=None)
    a = ap.parse_args()
    r = verify(a.run_dir, a.cpu_mhz)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
