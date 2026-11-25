"""R2 against R1 comparison of the Mode A battery -> reports/r2/mode_a_r2.md.

R2 is the run set of the current configuration table (`reports/mode_a_configs.csv`, with the R2 columns);
R1 is, per split and target, the latest run of the same seed whose recorded `table_sha256` is a different
table (the frozen-protocol runs, still in derived/mode_a/ and runs.jsonl). Pure formatting in `build`;
`main` only reads runs.jsonl and the prediction parquets. Usage: `uv run python mode_a_r2_report.py`."""
import json

import numpy as np
import pandas as pd

import evaluate
import mode_a
import report_inputs as ri
import runlog
import settings

OUT = settings.HERE / "reports" / "r2" / "mode_a_r2.md"
CANDS = ["B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear", "M1Ridge", "M2Trees", "M3Factored",
         "M4Physics", "AUTO"]


def _pct(x, d=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def _num(x, d=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def candidate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate of a prediction frame: n and the headline metrics, the interval width
    (median hi90/lo90) and the fraction of rows flagged."""
    rows = []
    for c, g in df.groupby("candidate", sort=False):
        m = evaluate.metrics(g["y_true"], g["y_pred"])
        rows.append({"candidate": c, **m, "width": evaluate.interval_width(g["lo90"], g["hi90"]),
                     "coverage90": evaluate.coverage(g["y_true"], g["lo90"], g["hi90"])})
    out = pd.DataFrame(rows).set_index("candidate")
    return out.reindex([c for c in CANDS if c in out.index])


def per_model(df: pd.DataFrame, cand: str) -> pd.Series:
    """MdAPE of `cand` per held-out base model."""
    g = df[df["candidate"] == cand]
    return g.groupby("base_model").apply(lambda x: evaluate.metrics(x["y_true"], x["y_pred"])["mdape"], include_groups=False)


def auto_choices(df: pd.DataFrame) -> dict:
    a = df[df["candidate"] == mode_a.AUTO].drop_duplicates("fold")
    return {k: int(v) for k, v in pd.Series([json.loads(p)["candidate"] for p in a["params"]]).value_counts().items()}


def _table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def split_rows(r1: dict, r2: dict, r2_prev: dict | None, target: str = "energy_mJ") -> list:
    """Rows [split, candidate, revision, MdAPE, MAPE, MaxAPE, cov90, width] of AUTO and M4Physics per split."""
    rows = []
    for split in ri.SPLITS:
        for cand in ("AUTO", "M4Physics"):
            for label, d in (("R2", r2), ("R2 round 0 (LOMO inner folds)", r2_prev or {}), ("R1", r1)):
                df = d.get((split, target))
                if df is None or cand not in set(df["candidate"]):
                    continue
                r = candidate_stats(df).loc[cand]
                rows.append([split, cand, label, _pct(r["mdape"]), _pct(r["mape"]), _pct(r["maxape"], 0),
                             _num(r["coverage90"]), _num(r["width"])])
    return rows


def build(r1: dict, r2: dict, seeds: dict | None = None, r2_prev: dict | None = None) -> str:
    """Markdown from prediction frames keyed by (split, target) for R1 and R2 (a missing key is skipped);
    `seeds` maps seed -> R2 S-LOMO energy frame; `r2_prev` the R2 round-0 runs."""
    L = ["# Mode A, R2 against R1 (balanced selection on inner folds that mimic the outer split; R2 features; M4Physics)\n",
         "Exploratory R2 revision; the lockbox has been seen, so nothing here is a pre-registered result. Percent "
         "columns are percent. MdAPE is the median, MAPE the mean, MaxAPE the maximum and P90 the 90th percentile of "
         "|pred - y| / y; R2log is 1 - SSE/SST of log y on log pred; width is the median hi90 / lo90 of the 90 % "
         "interval. R1 is the frozen protocol (pooled-MdAPE selection, R1 features, no M4).\n"]
    key = ("S-LOMO", "energy_mJ")
    if key in r2:
        s2 = candidate_stats(r2[key])
        s1 = candidate_stats(r1[key]) if key in r1 else None
        L += ["## S-LOMO, energy per image: every candidate (58 held-out configurations)\n"]
        hdr = ["candidate", "MdAPE", "MAPE", "MaxAPE", "P90", "R2log", "width", "cov90", "R1 MdAPE", "R1 MAPE", "R1 MaxAPE", "R1 width"]
        rows = []
        for c, r in s2.iterrows():
            o = s1.loc[c] if s1 is not None and c in s1.index else None
            rows.append([c, _pct(r["mdape"]), _pct(r["mape"]), _pct(r["maxape"], 0), _pct(r["p90ape"]), _num(r["r2_log"], 3),
                         _num(r["width"]), _num(r["coverage90"]),
                         *(["-"] * 4 if o is None else [_pct(o["mdape"]), _pct(o["mape"]), _pct(o["maxape"], 0), _num(o["width"])])])
        L += _table(hdr, rows) + [""]
        L += ["AUTO's choice per outer fold (R2): " + ", ".join(f"{k} x{v}" for k, v in auto_choices(r2[key]).items())
              + (". R1: " + ", ".join(f"{k} x{v}" for k, v in auto_choices(r1[key]).items()) if key in r1 else "") + ".\n"]
        a2, a1 = per_model(r2[key], "AUTO"), per_model(r1[key], "AUTO") if key in r1 else None
        m4 = per_model(r2[key], "M4Physics")
        L += ["## S-LOMO, energy: MdAPE per held-out base model (%)\n"]
        hdr = ["base model", "AUTO R2", "M4Physics R2"] + (["AUTO R1"] if a1 is not None else [])
        rows = [[m, _pct(a2.get(m)), _pct(m4.get(m))] + ([_pct(a1.get(m))] if a1 is not None else []) for m in a2.index]
        L += _table(hdr, rows) + [""]
    if seeds:
        L += ["## S-LOMO, energy, AUTO across seeds (R2)\n"]
        rows = []
        for sd, df in sorted(seeds.items()):
            m = evaluate.metrics(df[df["candidate"] == "AUTO"]["y_true"], df[df["candidate"] == "AUTO"]["y_pred"])
            rows.append([str(sd), _pct(m["mdape"]), _pct(m["mape"]), _pct(m["maxape"], 0)])
        L += _table(["seed", "MdAPE", "MAPE", "MaxAPE"], rows) + [""]
    rows = split_rows(r1, r2, r2_prev)
    if rows:
        L += ["## Energy per image: AUTO and M4Physics on every split\n"]
        L += _table(["split", "candidate", "revision", "MdAPE", "MAPE", "MaxAPE", "cov90", "width"], rows) + [""]
    L += ["## AUTO MdAPE (%) by split and target, R1 and R2\n"]
    rows = []
    for (split, target) in sorted(set(r1) | set(r2)):
        def v(d):
            x = d.get((split, target))
            if x is None:
                return "-"
            a = x[x["candidate"] == "AUTO"]
            return _pct(evaluate.metrics(a["y_true"], a["y_pred"])["mdape"])
        rows.append([split, target, v(r1), v(r2)])
    L += _table(["split", "target", "R1", "R2"], rows) + [""]
    return "\n".join(L)


def _runs(paths, recs, sha_now: str, seed: int) -> tuple:
    """(R1, R2, R2 before the inner-fold fix): per (split, target) the latest run of a different table, the
    latest of this table and the one before it."""
    r1, r2, prev = {}, {}, {}
    for split in ri.SPLITS:
        for target in ri.TARGETS:
            hits = [r for r in recs if r.get("command") == "mode-a" and r.get("split") == split
                    and r.get("target") == target and r.get("seed") == seed
                    and (paths.mode_a_dir / f"{r['run_id']}.parquet").is_file()]
            now = [r for r in hits if r.get("table_sha256") == sha_now]
            old = [r for r in hits if r.get("table_sha256") != sha_now]
            for out, rec in ((r2, now[-1] if now else None), (prev, now[0] if len(now) > 1 else None),
                             (r1, old[-1] if old else None)):
                if rec is not None:
                    out[(split, target)] = ri.ref_of(rec, paths.mode_a_dir).df
    return r1, r2, prev


def main() -> int:
    paths = ri.Paths.default()
    recs = ri.read_records(paths.runs_log)
    sha = runlog.file_hash(paths.configs_csv)
    seed = settings.seed()
    r1, r2, prev = _runs(paths, recs, sha, seed)
    seeds = {}   # the extra seeds gave identical results and predate the inner-fold fix: not shown
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(r1, r2, seeds, prev))
    print(f"wrote {OUT}: R1 runs {len(r1)}, R2 runs {len(r2)}, seeds {sorted(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
