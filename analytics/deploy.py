"""Deployment calculator (spec 6.6): total energy of a whole deployment from a per-image energy and
latency, and its validation against measured serving, ondemand, fan and soak cells.

predict() is pure arithmetic on a predicted (e_img_mJ, t_img_ms). validate() feeds it only held-out
(S-LOMO AUTO) Mode A predictions, never an in-sample fit. Everything measured here (idle table, fan
offsets, soak factor) is a lookup read from the cells, stated as an assumption in the output."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import runlog
import settings
import tables

SOAK_RAMP_S = 1800.0       # soak factor is reached after 30 minutes and ramps linearly up to there
ONDEMAND_MHZ = 2400        # ondemand active work is predicted at the pinned 2400 MHz configuration
DEFAULT_RIG = "nov2026"
FAN_ASSUMPTION = "assumption: measured on yolo11n only"
SOAK_ASSUMPTION = ("assumption: soak factor = mean energy-per-image ratio over the available soaks, ramped "
                   "as 1 + (f - 1) * min(1, duration_s / 1800); rests on one soak per model")
ONDEMAND_ASSUMPTION = "ondemand: active energy taken at 2400 MHz, idle power from the ondemand baseline"
VALIDATION_COLS = ["case", "cell_id", "phase", "measured_J", "predicted_J", "ape", "source", "notes",
                   "predicted_J_no_soak", "lo90_J", "hi90_J", "flagged", "saturated"]
_ABSENT = ("serve_fps", "soak_minutes", "concurrent_mode")
KNOWN_CASE = "serve-default (duration known)"
KNOWN_NOTE = "duration known (leaks measured duration)"


# ------------------------------------------------------------------ keys and measured lookups

def mode_a_key_for_cell(row) -> str:
    """Mode A configuration key of any catalogue row: governor `ondemand` maps to cpu_mhz 2400;
    serve, soak, fan, concurrency and input fields are ignored."""
    r = {c: row[c] for c in tables.CONFIG_COLS}
    if row.get("governor") == "ondemand":
        r["cpu_mhz"] = ONDEMAND_MHZ
    return tables.config_key(r)


def cell_config_map(cat: pd.DataFrame) -> pd.Series:
    """cell_id -> Mode A configuration key (`mode_a_key_for_cell`) of every row of `cat`; empty for an empty table."""
    return pd.Series({r["cell_id"]: mode_a_key_for_cell(r) for _, r in cat.iterrows()}, dtype=object)


def cpu_key(row) -> str:
    """`ondemand` for ondemand cells, else str(cpu_mhz)."""
    return "ondemand" if row["governor"] == "ondemand" else str(int(row["cpu_mhz"]))


def _primary(cat: pd.DataFrame, qc: pd.DataFrame | None) -> pd.DataFrame:
    """Training cells (never lockbox), QC-primary ones only if qc is given."""
    cat = data.training_cells(cat)
    if qc is None:
        return cat
    return cat[cat["cell_id"].isin(qc.loc[qc["primary"].astype(bool), "cell_id"])]


def cell_table(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Catalogue rows (QC-primary only if qc is given) joined with their `forward` derive row."""
    d = der[der["mode"] == tables.MODE].drop(columns=["batch"], errors="ignore")
    return _primary(cat, qc).merge(d, on="cell_id", how="inner").reset_index(drop=True)


def _idle_cells(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """QC-primary catalogue rows with a finite baseline_W and their cpu_key (one row per cell)."""
    base = der.drop_duplicates("cell_id")[["cell_id", "baseline_W"]]
    c = _primary(cat, qc).merge(base, on="cell_id")
    c = c[np.isfinite(c["baseline_W"].astype(float))].copy()
    c["cpu_key"] = [cpu_key(r) for _, r in c.iterrows()]
    return c


def _idle_aggregate(c: pd.DataFrame, exclude_cell: str | None = None) -> pd.DataFrame:
    if exclude_cell is not None:
        c = c[c["cell_id"] != exclude_cell]
    g = c.groupby(["rig", "cpu_key", "fan_mode"])["baseline_W"]
    return g.agg(P_idle_W="median", n="size",
                 iqr_W=lambda x: float(x.quantile(0.75) - x.quantile(0.25))).reset_index()


def idle_table(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame, exclude_cell: str | None = None) -> pd.DataFrame:
    """Idle power per (rig, cpu_key, fan_mode): median baseline_W over QC-primary cells of both rigs
    (every cell contributes one baseline, whatever its workload). Columns: rig, cpu_key, fan_mode,
    P_idle_W, n, iqr_W. `exclude_cell` leaves that cell out (leave-one-cell-out, used by `validate`
    so a validated cell's own baseline never enters the idle power that predicts it)."""
    return _idle_aggregate(_idle_cells(cat, der, qc), exclude_cell)


def _is_plain(t: pd.DataFrame) -> pd.DataFrame:
    """Rows with no serve, soak or concurrency and default input (the plain forward configuration)."""
    m = pd.Series(True, index=t.index)
    for k in _ABSENT:
        if k in t:
            m &= t[k].isna()
    if "input_source" in t:
        m &= t["input_source"] == "repeat"
    return t[m]


def _ref_cells(t: pd.DataFrame, base_model: str) -> pd.DataFrame:
    """nov2026 fp32 640 px, 4 thread, batch 1, pinned 2400 MHz forward cells of `base_model`."""
    t = _is_plain(t)
    m = ((t["rig"] == DEFAULT_RIG) & (t["base_model"] == base_model) & (t["variant"] == base_model)
         & (t["imgsz"] == 640) & (t["threads"] == 4) & (t["cpu_mhz"] == ONDEMAND_MHZ)
         & (t["governor"] == "pinned") & (t["batch"] == 1))
    return t[m]


def fan_offsets(t_cells: pd.DataFrame) -> dict:
    """{"auto": {"active_W", "idle_W"}, "off": {...}, "assumption": str}.

    active_W = mean mean_P_W(fan mode) - mean mean_P_W(full); idle_W = median baseline_W(fan mode) -
    median baseline_W(full), on yolo11n 640 px, 4 threads, 2400 MHz nov2026 cells. `t_cells` is a
    cell_table (catalogue joined with forward derive rows). A mode with no cells gives NaN offsets."""
    r = _ref_cells(t_cells, "yolo11n")
    full = r[r["fan_mode"] == "full"]
    out = {"assumption": FAN_ASSUMPTION}
    for fan in ("auto", "off"):
        f = r[r["fan_mode"] == fan]
        if f.empty or full.empty:
            out[fan] = {"active_W": float("nan"), "idle_W": float("nan"), "n": len(f)}
            continue
        out[fan] = {"active_W": float(f["mean_P_W"].mean() - full["mean_P_W"].mean()),
                    "idle_W": float(f["baseline_W"].median() - full["baseline_W"].median()), "n": len(f)}
    return out


def soak_factor(der: pd.DataFrame, cat: pd.DataFrame, exclude_cell: str | None = None,
                qc: pd.DataFrame | None = None) -> float:
    """Mean over the soak cells (minus `exclude_cell`) of soak energy per image / energy per image of
    the matching plain configuration of the same base model (640 px, 4 threads, 2400 MHz, fan full,
    pinned, nov2026). NaN if no soak is available."""
    t = cell_table(cat, der, qc)
    t = t[t["rig"] == DEFAULT_RIG]
    ratios = []
    for _, s in t[t["soak_minutes"].notna()].iterrows():
        if s["cell_id"] == exclude_cell:
            continue
        ref = _ref_cells(t, s["base_model"])
        ref = ref[(ref["fan_mode"] == "full") & np.isfinite(ref["energy_mJ_per_image"].astype(float))]
        if ref.empty or not np.isfinite(s["energy_mJ_per_image"]):
            continue
        ratios.append(float(s["energy_mJ_per_image"] / ref["energy_mJ_per_image"].mean()))
    return float(np.mean(ratios)) if ratios else float("nan")


# ------------------------------------------------------------------ the calculator

def _idle_row(idle: pd.DataFrame, rig: str, ck: str, fan: str) -> float:
    m = idle[(idle["rig"] == rig) & (idle["cpu_key"] == ck) & (idle["fan_mode"] == fan)]
    if m.empty:
        raise KeyError(f"no idle power for (rig, cpu_key, fan_mode) = ({rig!r}, {ck!r}, {fan!r})")
    return float(m["P_idle_W"].iloc[0])


def predict(scenario: dict, e_img_mJ: float, t_img_ms: float, idle: pd.DataFrame,
            offsets: dict | None, soak: float | None) -> dict:
    """Total energy of a deployment.

    scenario: cpu_key ("1500" / "2000" / "2400" / "ondemand"), fan_mode, rig (default "nov2026"), and
    exactly one of two duration kinds (anything else is a ValueError):
      - `duration_s`: the wall time is known a priori (open loop, a fixed fps, a wall-clock deployment), with
        either rate_fps (float, or "max") or n_images;
      - `gap_s`: a closed loop (request, then sleep `gap_s`, repeated), whose wall time depends on the latency
        being predicted. Needs `n_images`, forbids `rate_fps`; optional `overhead_s` (default 0.0, per-request time
        outside the forward pass, see `request_overhead_s`) and `batch` (default 1, images per request). With
        R = n_images / batch requests, duration = R * (batch * t_img + gap_s + overhead_s); busy = n_images * t_img.
        The measured serve duration is never an input of this kind.
    e_img_mJ and t_img_ms are per image;
    for `ondemand` the caller passes the 2400 MHz prediction and P_idle comes from the ondemand row.
    Idle power is the table row at fan `full`, plus offsets[fan]["idle_W"] for fan auto / off; the
    active offset adds offsets[fan]["active_W"] * busy_s. `soak` is the energy-per-image ratio f
    (None: no soak factor); the total is multiplied by 1 + (f - 1) * min(1, duration_s / 1800).

    busy time is clipped to the duration (idle time never negative) and `saturated` is set when the
    rate is capped by the latency (including rate "max") or the busy time had to be clipped.

    Returns E_J, E_J_no_soak, mean_P_W, n_images, busy_s, idle_s, saturated, assumptions."""
    fan, ck, rig = scenario.get("fan_mode", "full"), str(scenario["cpu_key"]), scenario.get("rig", DEFAULT_RIG)
    t_s = t_img_ms / 1000.0
    cap = 1.0 / t_s
    assumptions, saturated = [], False
    has_dur, has_gap = scenario.get("duration_s") is not None, scenario.get("gap_s") is not None
    if has_dur == has_gap:
        raise ValueError("scenario needs exactly one of duration_s (known a priori) and gap_s (closed loop), got "
                         + ("both" if has_dur else "neither"))
    if has_gap:
        if "n_images" not in scenario or "rate_fps" in scenario:
            raise ValueError("a gap_s scenario needs n_images and takes no rate_fps")
        n_req = float(scenario["n_images"]) / float(scenario.get("batch", 1))
        dur = n_req * (float(scenario.get("batch", 1)) * t_s + float(scenario["gap_s"])
                       + float(scenario.get("overhead_s", 0.0)))
        assumptions.append(f"closed loop: duration = requests * (batch * latency + gap {float(scenario['gap_s']):g} s "
                           f"+ overhead {float(scenario.get('overhead_s', 0.0)):.4g} s)")
    else:
        dur = float(scenario["duration_s"])
    if "n_images" in scenario:
        n = float(scenario["n_images"])
    else:
        rate = scenario["rate_fps"]
        rate = cap if rate == "max" else float(rate)
        saturated = rate >= cap
        n = min(rate, cap) * dur
    busy = n * t_s
    if busy > dur * (1 + 1e-9):
        saturated = True
    busy = min(busy, dur)
    if saturated:
        assumptions.append("saturated: rate capped by latency (or busy time clipped to the duration); "
                           "no idle time")
    p_idle = _idle_row(idle, rig, ck, "full")
    active_w = 0.0
    if fan != "full":
        if not offsets or fan not in offsets:
            raise ValueError(f"fan_mode {fan!r} needs fan offsets for it")
        p_idle += offsets[fan]["idle_W"]
        active_w = offsets[fan]["active_W"]
        assumptions.append(f"fan {fan}: {offsets.get('assumption', FAN_ASSUMPTION)}")
    if ck == "ondemand":
        assumptions.append(ONDEMAND_ASSUMPTION)
    e_no = n * e_img_mJ / 1000.0 + active_w * busy + (dur - busy) * p_idle
    factor = 1.0
    if soak is not None and np.isfinite(soak):
        factor = 1.0 + (soak - 1.0) * min(1.0, dur / SOAK_RAMP_S)
        assumptions.append(SOAK_ASSUMPTION)
    e = e_no * factor
    return {"E_J": e, "E_J_no_soak": e_no, "mean_P_W": e / dur if dur > 0 else float("nan"),
            "n_images": n, "busy_s": busy, "idle_s": dur - busy, "saturated": saturated,
            "assumptions": assumptions}


# ------------------------------------------------------------------ closed-loop request overhead

DEFAULT_SERVE_GAP_MS = 200.0     # src/run.py --serve-gap-ms default, for a run_meta without the key


def family_of(base_model: str) -> str:
    """Request-path family of a model: "yolo" (detector post-processing) or "other"."""
    return "yolo" if str(base_model).startswith("yolo") else "other"


def with_serve_gap(cells: pd.DataFrame) -> pd.DataFrame:
    """`cells` with a `serve_gap_ms` column: kept if present, else read from each row's `path`/run_meta.json
    (`serve_gap_ms`, default 200 ms as in src/run.py). A cell that cannot be read, or a frame with no `path`,
    gets the default."""
    if "serve_gap_ms" in cells.columns:
        return cells
    gaps = []
    for pth in (cells["path"] if "path" in cells.columns else [None] * len(cells)):
        try:
            gaps.append(float(data.read_meta(pth).get("serve_gap_ms", DEFAULT_SERVE_GAP_MS)))
        except (OSError, ValueError, TypeError):    # includes pth None
            gaps.append(DEFAULT_SERVE_GAP_MS)
    return cells.assign(serve_gap_ms=gaps)


def request_overhead_s(cells: pd.DataFrame, exclude_cell: str | None = None) -> dict:
    """Per-request time outside the forward pass and the idle gap, by family: {"yolo": s, "other": s, "all": s}.

    `cells` are TRAINING cells carrying serve_n_requests, batch, serve_dur_ms, latency_ms_per_image,
    serve_fps and serve_gap_ms (or a `path` to read it from; see `with_serve_gap`) and base_model, cell_id.
    Only closed-loop cells count (serve_fps NaN, finite duration, latency and a positive request count). For a
    cell, with R requests of `batch` images each, the overhead is
        (serve_dur - R * batch * latency - R * gap) / R,
    where serve_dur is the summed duration of the serve-phase samples (derive._serve) and every request, the last
    included, is followed by the gap (src/run.py). It uses the cell's own measured latency: this is a property of
    the request path, estimated from training cells only. The result per family is the median; a family with no
    cell gets the all-cells median ("all"). `exclude_cell` drops one cell (leave-one-cell-out validation).
    Raises ValueError when no usable cell remains."""
    c = with_serve_gap(cells)
    c = c[c["serve_fps"].isna() & (c["serve_n_requests"] > 0) & np.isfinite(c["serve_dur_ms"].astype(float))
          & np.isfinite(c["latency_ms_per_image"].astype(float))]
    if exclude_cell is not None:
        c = c[c["cell_id"] != exclude_cell]
    if c.empty:
        raise ValueError("no closed-loop serve cell to estimate the request overhead from")
    r = c["serve_n_requests"].astype(float)
    oh = (c["serve_dur_ms"] / 1000.0 - r * c["batch"].astype(float) * c["latency_ms_per_image"] / 1000.0
          - r * c["serve_gap_ms"] / 1000.0) / r
    fam = c["base_model"].map(family_of)
    allm = float(oh.median())
    return {f: (float(oh[fam == f].median()) if (fam == f).any() else allm) for f in ("yolo", "other")} | {"all": allm}


# ------------------------------------------------------------------ Mode A predictions

def preds_from_mode_a(energy_df: pd.DataFrame, latency_df: pd.DataFrame) -> pd.DataFrame:
    """Held-out predictions per configuration: the S-LOMO `AUTO` rows of the energy and latency Mode A
    parquets. Columns config_key, e_img_mJ, t_img_ms, lo90_mJ, hi90_mJ, flagged."""
    def auto(df):
        d = df[(df["split"] == "S-LOMO") & (df["candidate"] == "AUTO")]
        return d.groupby("config_key", sort=True).first()
    e, t = auto(energy_df), auto(latency_df)
    out = pd.DataFrame({"e_img_mJ": e["y_pred"], "lo90_mJ": e["lo90"], "hi90_mJ": e["hi90"],
                        "flagged": e["flagged"]}).join(t["y_pred"].rename("t_img_ms"), how="inner")
    return out.reset_index()[["config_key", "e_img_mJ", "t_img_ms", "lo90_mJ", "hi90_mJ", "flagged"]]


def _latest_parquet(split: str, target: str, log_path=None, mode_a_dir=None, seed=None) -> Path | None:
    """Parquet of the latest Mode A run of (split, target) in runs.jsonl; with `seed`, the latest run
    with that seed is preferred and the latest of any seed is the fallback."""
    log = Path(log_path) if log_path else runlog.DEFAULT_LOG
    if not log.is_file():
        return None
    out = settings.derived_dir() / "mode_a" if mode_a_dir is None else Path(mode_a_dir)
    best = best_seed = None
    for line in log.read_text().splitlines():
        rec = json.loads(line) if line.strip() else {}
        if rec.get("command") == "mode-a" and rec.get("split") == split and rec.get("target") == target:
            if (out / f"{rec['run_id']}.parquet").is_file():
                best = out / f"{rec['run_id']}.parquet"
                if seed is not None and rec.get("seed") == seed:
                    best_seed = best
    return best_seed or best


def load_mode_a_preds(log_path=None, mode_a_dir=None, seed=None) -> pd.DataFrame:
    """preds_from_mode_a over the S-LOMO runs of targets energy_mJ and latency_ms in runs.jsonl: the
    latest run with `seed` if there is one, else the latest run. The result's attrs record
    `run_ids` ({target: run_id}) and `seed_matched` ({target: bool}, False when `seed` is None or no
    run had that seed). Raises FileNotFoundError naming the missing run."""
    got, ids, matched = {}, {}, {}
    for target in ("energy_mJ", "latency_ms"):
        p = _latest_parquet("S-LOMO", target, log_path, mode_a_dir, seed)
        if p is None:
            raise FileNotFoundError(f"no S-LOMO Mode A run for target {target} in runs.jsonl with a parquet "
                                    f"under derived/mode_a; run `uv run python cli.py mode-a --split S-LOMO`")
        got[target] = pd.read_parquet(p)
        ids[target] = p.stem
        matched[target] = seed is not None and p.stem in _run_ids_with_seed(target, seed, log_path)
    out = preds_from_mode_a(got["energy_mJ"], got["latency_ms"])
    out.attrs["run_ids"], out.attrs["seed_matched"] = ids, matched
    return out


def _run_ids_with_seed(target: str, seed: int, log_path=None) -> set:
    log = Path(log_path) if log_path else runlog.DEFAULT_LOG
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    return {r["run_id"] for r in recs if r.get("command") == "mode-a" and r.get("split") == "S-LOMO"
            and r.get("target") == target and r.get("seed") == seed}


# ------------------------------------------------------------------ validation

def _scenario(row: pd.Series, **kw) -> dict:
    return {"cpu_key": cpu_key(row), "fan_mode": row["fan_mode"], "rig": row["rig"], **kw}


def _val_row(case, row, phase, measured_J, p, source, notes, p_no=None, lo=None, hi=None, flagged=None):
    meas = float(measured_J)
    return {"case": case, "cell_id": row["cell_id"], "phase": phase, "measured_J": meas,
            "predicted_J": p["E_J"], "ape": abs(p["E_J"] - meas) / abs(meas) if meas else float("nan"),
            "source": source, "notes": notes, "predicted_J_no_soak": p["E_J_no_soak"],
            "lo90_J": lo, "hi90_J": hi, "flagged": flagged, "saturated": p["saturated"]}


def validate(cat: pd.DataFrame, der: pd.DataFrame, qc: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Compare the calculator with measured whole-phase energy, using only `preds` (held-out Mode A
    predictions: config_key, e_img_mJ, t_img_ms, optional lo90_mJ, hi90_mJ, flagged).

    Cases: serve-default (duration known) (every Mode A eligible primary cell's serve phase, with the MEASURED
    serve duration: this leaks the latency being predicted, because a closed-loop serve phase lasts
    n * (latency + gap + overhead); kept for comparison and labelled so), serve-gap (the same cells, honest:
    gap scenario with the request overhead estimated from the other training serve cells, leave-one-cell-out),
    serve-fps (yolo11n serve 1/5/15 fps; duration known a priori as n / fps), ondemand (yolo11s ondemand,
    forward per image and serve; the serve rows are duration known, ondemand-gap are the honest ones), soak
    (forward phase with the soak factor of the other soak only), fan (offsets measured on these cells: not a
    validation; fan-gap holds the serve rows without the measured duration).
    Rows whose configuration has no prediction are skipped; counts are in df.attrs["skipped"] (case ->
    n) and the fan offsets, soak factors and idle table in attrs.

    Idle power is leave-one-cell-out: every validated row uses the idle table built without that
    row's own cell (global constraint 4); a cell with no other cell in its (rig, cpu_key, fan_mode)
    row has no leak-free idle power and is skipped (counted in `skipped`). `attrs["idle"]` is the full
    table, for reference only."""
    t = cell_table(cat, der, qc)
    idle_c = _idle_cells(cat, der, qc)
    idle = _idle_aggregate(idle_c)
    idle_loco: dict = {}
    offsets = fan_offsets(cell_table(cat, der, qc))
    p_by = preds.drop_duplicates("config_key").set_index("config_key")
    rows, skipped, soaks = [], {}, {}
    extra = [c for c in ("lo90_mJ", "hi90_mJ", "flagged") if c in p_by.columns]

    def run(case, row, phase, scen, measured_J, notes, soak=None, source="lomo"):
        key = mode_a_key_for_cell(row)
        if key not in p_by.index:
            skipped[case] = skipped.get(case, 0) + 1
            return
        pr = p_by.loc[key]
        e, tm = float(pr["e_img_mJ"]), float(pr["t_img_ms"])
        cid = row["cell_id"]
        if cid not in idle_loco:
            idle_loco[cid] = _idle_aggregate(idle_c, exclude_cell=cid)
        own_idle = idle_loco[cid]
        try:
            p = predict(scen, e, tm, own_idle, offsets, soak)
        except KeyError:                       # no other cell gives this row's idle power
            skipped[case] = skipped.get(case, 0) + 1
            return
        lo = hi = None
        if "lo90_mJ" in extra and np.isfinite(pr["lo90_mJ"]):
            lo = predict(scen, float(pr["lo90_mJ"]), tm, own_idle, offsets, soak)["E_J"]
            hi = predict(scen, float(pr["hi90_mJ"]), tm, own_idle, offsets, soak)["E_J"]
        fl = pr["flagged"] if "flagged" in extra else None
        rows.append(_val_row(case, row, phase, measured_J, p, source, notes, lo=lo, hi=hi, flagged=fl))

    def serve(case, row, notes=""):
        """Serve phase with the MEASURED duration (leaks the latency for a closed-loop cell)."""
        n = float(row["serve_n_requests"]) * float(row["batch"])
        if not n > 0 or not np.isfinite(row["serve_dur_ms"]):
            skipped[case] = skipped.get(case, 0) + 1
            return
        scen = _scenario(row, n_images=n, duration_s=row["serve_dur_ms"] / 1000.0)
        run(case, row, "serve", scen, row["serve_E_mJ"] / 1000.0, notes)

    def serve_fps(case, row, notes=""):
        """Fixed-rate serve phase: the duration is n / fps, known a priori (never the measured duration)."""
        n = float(row["serve_n_requests"]) * float(row["batch"])
        if not n > 0 or not np.isfinite(row["serve_dur_ms"]):
            skipped[case] = skipped.get(case, 0) + 1
            return
        scen = _scenario(row, n_images=n, duration_s=n / float(row["serve_fps"]))
        run(case, row, "serve", scen, row["serve_E_mJ"] / 1000.0, notes)

    t_gap = with_serve_gap(t)

    def serve_gap(case, row, notes=""):
        """Closed-loop serve phase without the measured duration: gap scenario, the request overhead from the
        other training serve cells (leave-one-cell-out; family of this cell's model)."""
        n = float(row["serve_n_requests"]) * float(row["batch"])
        if not n > 0 or not np.isfinite(row["serve_dur_ms"]):
            skipped[case] = skipped.get(case, 0) + 1
            return
        try:
            oh = request_overhead_s(t_gap, exclude_cell=row["cell_id"])[family_of(row["base_model"])]
        except ValueError:                         # no other closed-loop cell: no leak-free overhead
            skipped[case] = skipped.get(case, 0) + 1
            return
        gap = float(t_gap.loc[t_gap["cell_id"] == row["cell_id"], "serve_gap_ms"].iloc[0]) / 1000.0
        scen = _scenario(row, n_images=n, gap_s=gap, overhead_s=oh, batch=float(row["batch"]))
        run(case, row, "serve", scen, row["serve_E_mJ"] / 1000.0,
            f"{notes + '; ' if notes else ''}gap {gap * 1000:g} ms, request overhead {oh * 1000:.2f} ms "
            f"(median of the other training serve cells)")

    # forward phase has no idle time: duration is the busy time of the predicted latency
    def forward(case, row, n, notes, soak=None):
        key = mode_a_key_for_cell(row)
        if key not in p_by.index:
            skipped[case] = skipped.get(case, 0) + 1
            return
        dur = n * float(p_by.loc[key, "t_img_ms"]) / 1000.0
        run(case, row, "forward", _scenario(row, n_images=n, duration_s=dur),
            row["energy_mJ_per_image"] * n / 1000.0, notes, soak=soak)

    plain = t[np.isfinite(t["serve_dur_ms"].astype(float)) & (t["serve_n_requests"] > 0)]
    for _, r in plain[plain["mode_a_eligible"].astype(bool)].iterrows():
        serve(KNOWN_CASE, r, KNOWN_NOTE)
        serve_gap("serve-gap", r)
    for _, r in plain[plain["serve_fps"].notna() & (plain["base_model"] == "yolo11n")].iterrows():
        serve_fps("serve-fps", r, f"serve_fps={r['serve_fps']:g}; duration n / fps")
    for _, r in t[(t["governor"] == "ondemand") & (t["base_model"] == "yolo11s")].iterrows():
        forward("ondemand", r, 1, "forward phase, per image (J per image); active energy at 2400 MHz")
        serve("ondemand", r, f"serve phase; idle power from the ondemand baseline; {KNOWN_NOTE}")
        if np.isnan(r["serve_fps"]):
            serve_gap("ondemand-gap", r, "serve phase; idle power from the ondemand baseline")
    for _, r in t[t["soak_minutes"].notna()].iterrows():
        f = soak_factor(der, cat, exclude_cell=r["cell_id"], qc=qc)
        soaks[r["cell_id"]] = f
        if not np.isfinite(f):
            skipped["soak"] = skipped.get("soak", 0) + 1
            continue
        n = float(r["n_images"])
        forward("soak", r, n, f"forward phase; soak factor {f:.4f} from the other soak only; "
                              f"E_J_no_soak is the prediction without it", soak=f)
    for _, r in t[t["fan_mode"].isin(["auto", "off"]) & t["serve_fps"].isna() & t["soak_minutes"].isna()
                  & t["concurrent_mode"].isna()].iterrows():
        note = "offsets measured on these cells: not a validation"
        forward("fan", r, 1, f"forward phase, per image (J per image); {note}")
        serve("fan", r, f"serve phase; {note}; {KNOWN_NOTE}")
        serve_gap("fan-gap", r, f"serve phase; {note}")
    out = pd.DataFrame(rows, columns=VALIDATION_COLS)
    out.attrs.update(skipped=skipped, fan_offsets=offsets, soak_factors=soaks, idle=idle)
    return out
