#!/usr/bin/env python3
"""Check one aligned Monsoon run (schema 4) and write its phase_energy.csv.

Reads run_meta.json, alignment.json and power_trace.parquet from the run directory (align.py
must have run). Fatal checks make `ok` false; soft ones go to `warnings`.

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
        res["ok"] = False
        return res

    if meta.get("schema_version") != 4:
        errors.append(f"schema_version {meta.get('schema_version')} != 4")
    if cpu_mhz is not None and meta.get("cpu_mhz_pinned") != cpu_mhz:
        errors.append(f"cpu_mhz_pinned {meta.get('cpu_mhz_pinned')} != {cpu_mhz}")
    if not meta.get("fan_full"):
        errors.append("fan_full not set")
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
    if not meta.get("bench_skipped") and not any(str(p).startswith("bench:") for p in counts.index):
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
