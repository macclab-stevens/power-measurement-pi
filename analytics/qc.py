"""Data-quality rules Q1-Q6 applied to every cell; decides which cells enter the analysis.

Rules (thresholds in config.toml [qc]):
  Q1flag  ctx_read_frac_scored > q1                       flag only
  Q2      alignment max_residual_ms >= q2                 exclude
  Q3flag  slip_scored_frac > q3                           flag only
  Q4      a current throttle bit (bits & 0xF) or uv_alarm kept, but not primary
  Q4unparsed  a throttle word that cannot be parsed flag only (the cell stays primary)
  Q5flag  median |P - V*I| / P (every 50th trace row) > q5 flag only
  Q6      t_pi_ns not strictly increasing, sample_ratio outside [lo, hi], or an unknown phase label
          quarantine (= exclude)
include = not (Q2 or Q6); primary = include and not Q4.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

import data
import derive
import settings

CURRENT_THROTTLE_MASK = 0xF            # under-voltage, freq-cap, throttled, soft temp limit (now)
PV_STRIDE = 50
TRACE_COLUMNS = ("t_pi_ns", "P_W", "V_V", "I_A")   # plus "phase", read separately as a dictionary
KNOWN_PHASES = frozenset([
    "", "idle", "load_weights", "warmup", "sync:start", "sync:end", "baseline_A", "baseline_B", "gap",
    "serve", "concurrent_drain", "forward", "prefill", "decode", "train"])
KNOWN_PREFIXES = ("first:", "bench:")
RULE_IDS = ("Q1flag", "Q2", "Q3flag", "Q4", "Q4unparsed", "Q5flag", "Q6")
COLUMNS = [
    "cell_id", "include", "primary", "rules", "max_residual_ms", "n_slips", "slip_samples",
    "slips_in_scored", "slip_frames_excluded", "slip_scored_frac", "ctx_read_frac_scored",
    "throttle_start", "throttle_end", "throttle_current", "throttle_unparsed", "pv_median_rel", "n_samples",
    "expected_samples", "sample_ratio", "monotonic", "dup_timestamps", "unknown_phases",
    "start_temp_C", "max_temp_C"]
NAN = float("nan")


def is_known_phase(label) -> bool:
    label = "" if label is None else str(label)
    return label in KNOWN_PHASES or label.startswith(KNOWN_PREFIXES)


def label_family(label: str) -> str:
    """first:<id> and bench:<id> labels carry a per-call hash; they are reported as first:* / bench:*."""
    for p in KNOWN_PREFIXES:
        if label.startswith(p):
            return p + "*"
    return label


def bounds_from_cfg(cfg: dict | None = None) -> dict:
    return dict((cfg or settings.cfg())["qc"])


def parse_bits(x) -> int:
    """Throttle word from an int, a float, a decimal string (zero-padded allowed: '05') or a hex string
    ('0x50005'); missing -> 0. Raises ValueError for a value that cannot be read."""
    if x is None:
        return 0
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0
        try:
            return int(s, 16) if s.lower().startswith("0x") else int(s, 10)
        except ValueError:
            f = float(s)                                  # '5.0'; garbage raises ValueError here
            if not np.isfinite(f) or f != int(f):
                raise ValueError(f"not a throttle word: {x!r}")
            return int(f)
    if isinstance(x, float) and np.isnan(x):
        return 0
    return int(x)


def _current(bits) -> bool:
    return bool(parse_bits(bits) & CURRENT_THROTTLE_MASK)


def _current_safe(bits, unparsed: list) -> bool:
    """_current, but an unreadable value counts as not current and is recorded in `unparsed`."""
    try:
        return _current(bits)
    except (ValueError, OverflowError):
        unparsed.append(bits)
        return False


def _measured_modes(meta: dict) -> list[str]:
    return list(dict.fromkeys(meta.get("modes") or []))


def slip_and_ctx_stats(pe, meta: dict, alignment: dict | None) -> dict:
    """slip_scored_frac and ctx_read_frac_scored from phase_energy rows of measured frames only."""
    out = {"slip_scored_frac": NAN, "ctx_read_frac_scored": NAN}
    if pe is None:
        return out
    slips = derive.slip_times_s(alignment, meta)
    tot = exc = 0.0
    ctx_w = ctx_d = 0.0
    for mode in _measured_modes(meta):
        t = derive.frame_table(pe, mode)
        drop = derive.slip_overlaps(t, slips)
        tot += float(t["dur_ms"].sum())
        exc += float(t.loc[drop, "dur_ms"].sum())
        rows = pe[pe["phase"] == mode]
        it = pd.to_numeric(rows["iter"], errors="coerce")
        rows = rows[it >= 1]
        ok = rows["ctx_read_frac"].notna() & rows["dur_ms"].notna()
        ctx_w += float((rows.loc[ok, "ctx_read_frac"] * rows.loc[ok, "dur_ms"]).sum())
        ctx_d += float(rows.loc[ok, "dur_ms"].sum())
    if tot > 0:
        out["slip_scored_frac"] = exc / tot
    if ctx_d > 0:
        out["ctx_read_frac_scored"] = ctx_w / ctx_d
    return out


def trace_stats(cell: Path, hz: float) -> tuple[dict, set]:
    """Q5 and Q6 measures from the power trace (columns t_pi_ns, P_W, V_V, I_A, phase only)."""
    out = {"pv_median_rel": NAN, "n_samples": 0, "expected_samples": NAN, "sample_ratio": NAN,
           "monotonic": False, "dup_timestamps": 0, "unknown_phases": ""}
    if not (cell / "power_trace.parquet").is_file():
        return out, set()
    tr = data.read_trace(cell, TRACE_COLUMNS)
    t = tr["t_pi_ns"].to_numpy()
    n = len(t)
    out["n_samples"] = n
    if n:
        d = np.diff(t)
        out["monotonic"] = bool((d > 0).all())
        out["dup_timestamps"] = int((d == 0).sum())
        span = (float(t[-1]) - float(t[0])) / 1e9
        out["expected_samples"] = span * hz if hz else NAN
        if out["expected_samples"] and out["expected_samples"] > 0:
            out["sample_ratio"] = n / out["expected_samples"]
        p, v, i = (tr[c].to_numpy()[::PV_STRIDE] for c in ("P_W", "V_V", "I_A"))
        ok = np.isfinite(p) & np.isfinite(v) & np.isfinite(i) & (p > 0)
        if ok.any():
            out["pv_median_rel"] = float(np.median(np.abs(p[ok] - v[ok] * i[ok]) / p[ok]))
    del tr
    ph = pq.read_table(cell / "power_trace.parquet", columns=["phase"], read_dictionary=["phase"])
    labels = {"" if x is None else str(x) for x in pc.unique(ph.column("phase")).to_pylist()}
    out["unknown_phases"] = ";".join(sorted(x for x in labels if not is_known_phase(x)))
    return out, labels


def throttle_stats(cell: Path, meta: dict) -> dict:
    """Q4 inputs: current bits at start/end (run_meta) and in-run (context.csv bits, sysfs uv_alarm)."""
    ctx = data.read_csv_opt(cell, "context.csv")
    sysfs = data.read_csv_opt(cell, "sysfs.csv")
    cur = False
    bad: list = []
    start_temp = NAN
    if ctx is not None and len(ctx):
        if "throttle_bits" in ctx:
            cur = any([_current_safe(b, bad) for b in ctx["throttle_bits"].dropna().unique()])
        if "temp_C" in ctx:
            start_temp = float(pd.to_numeric(ctx["temp_C"], errors="coerce").iloc[0])
    if sysfs is not None and "uv_alarm" in sysfs:
        cur = cur or bool((pd.to_numeric(sysfs["uv_alarm"], errors="coerce") == 1).any())
    mt = meta.get("max_temp_C")
    start, end = _current_safe(meta.get("throttled_start"), bad), _current_safe(meta.get("throttled_end"), bad)
    return {"throttle_start": start, "throttle_end": end, "throttle_current": cur,
            "throttle_unparsed": bool(bad),
            "start_temp_C": start_temp, "max_temp_C": NAN if mt is None else float(mt)}


def _analyse(row: pd.Series, drows: pd.DataFrame | None, bounds: dict) -> tuple[dict, set]:
    cell = Path(row["path"])
    meta = data.read_meta(cell)
    al = data.read_json_opt(cell, "alignment.json") or {}
    pe = data.read_csv_opt(cell, "phase_energy.csv")
    hz = meta.get("achieved_sample_hz") or al.get("slip_rate_hz") or NAN
    res = al.get("max_residual_ms")
    d = drows if drows is not None else pd.DataFrame()
    r = {"cell_id": row["cell_id"],
         "max_residual_ms": NAN if res is None else float(res),
         "n_slips": len(al.get("slips") or []),
         "slip_samples": int(al.get("slip_samples_total") or 0),
         "slips_in_scored": int(d["slips_in_scored"].iloc[0]) if len(d) else 0,
         "slip_frames_excluded": int(d["n_frames_slip_excluded"].sum()) if len(d) else 0}
    r.update(slip_and_ctx_stats(pe, meta, al))
    r.update(throttle_stats(cell, meta))
    tr, labels = trace_stats(cell, float(hz))
    r.update(tr)
    rules = []
    if r["ctx_read_frac_scored"] > bounds["q1_ctx_frac_flag"]:
        rules.append("Q1flag")
    if r["max_residual_ms"] >= bounds["q2_max_residual_ms"]:
        rules.append("Q2")
    if r["slip_scored_frac"] > bounds["q3_slip_frac_flag"]:
        rules.append("Q3flag")
    if r["throttle_start"] or r["throttle_end"] or r["throttle_current"]:
        rules.append("Q4")
    if r["throttle_unparsed"]:
        rules.append("Q4unparsed")
    if r["pv_median_rel"] > bounds["q5_pv_rel_flag"]:
        rules.append("Q5flag")
    if q6_reasons(r, bounds):
        rules.append("Q6")
    r["rules"] = ";".join(sorted(rules))
    r["include"] = not ("Q2" in rules or "Q6" in rules)
    r["primary"] = r["include"] and "Q4" not in rules
    return {k: r[k] for k in COLUMNS}, labels


def q6_reasons(r: dict, bounds: dict) -> list[str]:
    """Why a cell is quarantined (empty if Q6 does not apply). NaN ratio (no trace) counts."""
    why = []
    if r["n_samples"] == 0:
        why.append("no trace samples")
    if not r["monotonic"] and r["n_samples"]:
        why.append(f"t_pi_ns not strictly increasing ({r['dup_timestamps']} duplicates)")
    elif r["dup_timestamps"]:
        why.append(f"{r['dup_timestamps']} duplicate timestamps")
    ratio = r["sample_ratio"]
    if r["n_samples"] and not (bounds["q6_sample_ratio_lo"] <= ratio <= bounds["q6_sample_ratio_hi"]):
        why.append(f"sample_ratio {ratio:.4f} outside [{bounds['q6_sample_ratio_lo']}, "
                   f"{bounds['q6_sample_ratio_hi']}]")
    if r["unknown_phases"]:
        why.append(f"unknown phase labels: {r['unknown_phases']}")
    return why


def qc_cell(row: pd.Series, drows: pd.DataFrame, bounds: dict) -> dict:
    """QC one catalogue row; drows are its derive rows. Returns a dict with exactly COLUMNS."""
    return _analyse(row, drows, bounds)[0]


def run_qc(cat: pd.DataFrame, der: pd.DataFrame, bounds: dict | None = None) -> pd.DataFrame:
    """qc_cell over the catalogue, one cell at a time. result.attrs['phase_labels'] maps each phase
    label (first:<id> and bench:<id> collapsed to first:* / bench:*) found in any trace to the number
    of cells carrying it; result.attrs['phase_distinct'] maps the same keys to the number of distinct
    raw labels."""
    bounds = bounds or bounds_from_cfg()
    by_cell = {k: g for k, g in der.groupby("cell_id")}
    rows, counts, raw = [], {}, {}
    for _, row in cat.iterrows():
        r, labels = _analyse(row, by_cell.get(row["cell_id"]), bounds)
        rows.append(r)
        for fam in {label_family(x) for x in labels}:
            counts[fam] = counts.get(fam, 0) + 1
        raw.setdefault("", set()).update(labels)
    df = pd.DataFrame(rows, columns=COLUMNS)
    fam_raw: dict = {}
    for lab in raw.get("", ()):
        fam_raw.setdefault(label_family(lab), set()).add(lab)
    df.attrs["phase_labels"] = counts
    df.attrs["phase_distinct"] = {k: len(v) for k, v in fam_raw.items()}
    return df


MEASURES = ("max_residual_ms", "slip_scored_frac", "ctx_read_frac_scored", "pv_median_rel", "sample_ratio")


def quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """min, p50, p95, max (NaN ignored) of the five QC measures, plus the count of finite values."""
    rows = {}
    for m in MEASURES:
        v = df[m].dropna().to_numpy(dtype=float)
        rows[m] = {"n": len(v), **({"min": v.min(), "p50": np.quantile(v, 0.5),
                                   "p95": np.quantile(v, 0.95), "max": v.max()} if len(v) else {})}
    return pd.DataFrame(rows).T[["n", "min", "p50", "p95", "max"]]


def rule_counts(df: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Cells triggering each rule, by source, plus included / primary totals."""
    src = df["cell_id"].map(dict(zip(cat["cell_id"], cat["source"])))
    rows = {}
    for rid in RULE_IDS:
        hit = df["rules"].str.split(";").apply(lambda x, r=rid: r in x)
        rows[rid] = hit.groupby(src).sum()
    rows["include"] = df["include"].groupby(src).sum()
    rows["primary"] = df["primary"].groupby(src).sum()
    rows["cells"] = df.groupby(src).size()
    out = pd.DataFrame(rows).fillna(0).astype(int).T
    out["all"] = out.sum(axis=1)
    return out


def excluded_reasons(df: pd.DataFrame, bounds: dict) -> list[tuple[str, str]]:
    """(cell_id, reason text) for every cell with Q2 or Q6."""
    out = []
    for _, r in df.iterrows():
        rules = r["rules"].split(";")
        why = []
        if "Q2" in rules:
            why.append(f"Q2: max_residual_ms {r['max_residual_ms']:.3f} >= {bounds['q2_max_residual_ms']}")
        if "Q6" in rules:
            why.append("Q6: " + "; ".join(q6_reasons(r.to_dict(), bounds)))
        if why:
            out.append((r["cell_id"], " | ".join(why)))
    return out


def _table_md(df: pd.DataFrame, fmt="{:.6g}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def report_md(df: pd.DataFrame, cat: pd.DataFrame, bounds: dict) -> str:
    """The qc.md deliverable."""
    ex = excluded_reasons(df, bounds)
    labels = df.attrs.get("phase_labels", {})
    distinct = df.attrs.get("phase_distinct", {})
    unknown = sorted(x for x in labels if not is_known_phase(x))
    L = ["# QC report", "",
         f"{len(df)} cells; included {int(df['include'].sum())}, primary {int(df['primary'].sum())}, "
         f"excluded {int((~df['include']).sum())}. Rules are applied by `analytics/qc.py` with the "
         "bounds below (`analytics/config.toml`, `[qc]`). Per-cell values: `qc_cells.csv`.", "",
         "## Bounds", ""]
    L += [f"- `{k}` = {v}" for k, v in bounds.items()]
    L += ["", "## Rule counts (cells triggering each rule)", "",
          "Q1flag, Q3flag, Q4unparsed, Q5flag only annotate. Q2 and Q6 exclude (`include` False). Q4 keeps the cell "
          "but it is not `primary`.", "", _table_md(rule_counts(df, cat)), "",
          "## Excluded cells (Q2 or Q6)", ""]
    L += [f"- `{c}`: {w}" for c, w in ex] or ["None."]
    q4 = df.loc[df["rules"].str.split(";").apply(lambda x: "Q4" in x), "cell_id"]
    L += ["", f"## Q4 cells (kept, not primary): {len(q4)}", ""]
    L += [", ".join(f"`{c}`" for c in q4)] if len(q4) else ["None."]
    L += ["", "## Distribution of the QC measures", "", "Finite values only; `n` counts them.", "",
          _table_md(quantiles(df)), "", "## Phase labels in the traces", "",
          f"{sum(distinct.values())} distinct raw labels in {len(labels)} families. Outside the known vocabulary: "
          + (", ".join(f"`{x}`" for x in unknown) if unknown else "none."), "",
          _table_md(pd.DataFrame({"cells": pd.Series(labels), "distinct raw labels": pd.Series(distinct)})
                    .sort_index().rename_axis("label"), "{}")]
    return "\n".join(L) + "\n"
