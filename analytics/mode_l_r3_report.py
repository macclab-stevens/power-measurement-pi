"""R3 Mode L leave-one-model-out summary -> reports/r3/mode_l_r3.md: per rung, and per family of the held-out model
(YOLO v8/11, attention incl. yolo12, classifier/CNN). Run from analytics/: uv run python mode_l_r3_report.py"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import mode_l_r2_report as mlr, report_inputs as ri, settings
paths = ri.Paths.default(); recs = ri.read_records(paths.runs_log)
rec = mlr._latest_r2(recs, paths, settings.seed(), None)
df = pd.read_parquet(paths.mode_l_dir / f"{rec['run_id']}.parquet")
st = mlr.rung_stats(df)
ATT = {"swint", "vitb16", "maxvitt", "yolo12n", "yolo12s"}
CLS = {"mbv3", "effb0", "resnet18", "convnextt"}
fam = lambda m: "attention" if m in ATT else ("classifier/CNN" if m in CLS else "YOLO v8/11")
df["family"] = df["fold"].map(fam)
by = (df.groupby(["rung", "family"])[["ape", "ape_meas"]].median() * 100).round(2).unstack()
L = [f"# Mode L, R3: leave one base model out over every model with counters\n",
     f"Run `{rec['run_id']}` ({df['fold'].nunique()} held-out base models: {', '.join(sorted(df['fold'].unique()))}; "
     f"{df['cell_id'].nunique()} cells). Whole = whole-cell energy APE, inference = the forward segment, median over held-out cells.\n",
     "## Per rung\n", "| rung | whole MdAPE % | inference MdAPE % | 1 s block % | forward MAE W | forward R² |", "|---|---:|---:|---:|---:|---:|"]
for r, s in st.iterrows():
    L.append(f"| {r} | {100*s['whole']:.2f} | {100*s['meas']:.2f} | {100*s['block_1s']:.2f} | {s['mae_fwd_W']:.3f} | {s['r2_fwd']:.3f} |")
L += ["\n## Per family of the held-out model (median %, whole / inference)\n", "| rung | " + " | ".join(f"{f} whole | {f} inference" for f in by['ape'].columns) + " |",
      "|---|" + "---:|" * (2 * len(by['ape'].columns))]
for r in st.index:
    if r in by.index:
        L.append(f"| {r} | " + " | ".join(f"{by.loc[r, ('ape', f)]:.2f} | {by.loc[r, ('ape_meas', f)]:.2f}" for f in by['ape'].columns) + " |")
open("reports/r3/mode_l_r3.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
