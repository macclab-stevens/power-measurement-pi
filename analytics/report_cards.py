"""The two model cards (spec section 11 contents): target and mode, intended and out-of-scope use, data and
split, features, headline metrics with intervals, coverage, limitations, worst subgroups, hashes. Filled
from the analysis; the use and limitation text is fixed."""
import json

import pandas as pd

import eda_write as ew
import evaluate
import mode_a
import mode_a_models as mm
import report_l
import report_tables as rt
import report_tables_l as tl

OUT_OF_SCOPE_COMMON = [
    "Other boards than the one Raspberry Pi 5 measured here, other meters than the Monsoon rig, other runtimes or kernels.",
    "Claims about the SoC alone: every energy figure is gross board energy at the Monsoon input (SoC plus the rest of the board).",
    "Workloads other than single-model image inference: language-model, training and concurrent multi-workload energy are described only.",
]
LIMITS_COMMON = [
    "A single device and a single measurement rig pair (the oct2026 and nov2026 rigs): between-board and between-meter variation is not measured.",
    "The lockbox (about 15 new cells collected after the `analytics-frozen` tag) is pending: nothing here is a pristine out-of-sample test, "
    "and the protocol note discloses that configuration-level summaries were viewed before any model was fitted.",
    "Mode A and Mode L fold counts are small (9 and 5 base models); intervals and coverage estimates are coarse.",
]


def _head(title: str, a, what: str) -> list:
    r = a.master
    return [f"# {title}\n", f"{what}\n",
            f"Built {r.rec.get('ts', '')[:10]} from commit `{str(r.rec.get('commit', ''))[:10]}`"
            f"{' (dirty working tree)' if r.rec.get('dirty') else ''}, data hash `{str(r.rec.get('data_hash', ''))[:16]}`, run seed {a.seed}. "
            "Full results: `results.md`.\n"]


def _list(title: str, items) -> list:
    return [f"## {title}\n", *[f"- {x}" for x in items], "\n"]


def _worst(df: pd.DataFrame, by: str, k: int = 3) -> pd.DataFrame:
    rows = []
    for key, g in df[df["candidate"] == mode_a.AUTO].groupby(by, sort=True):
        m = evaluate.metrics(g["y_true"], g["y_pred"])
        rows.append({by: key, "n": m["n"], "AUTO MdAPE %": m["mdape"], "bias %": m["bias_pct"]})
    t = pd.DataFrame(rows).sort_values("AUTO MdAPE %", ascending=False).head(k)
    t["AUTO MdAPE %"], t["bias %"] = t["AUTO MdAPE %"].map(rt.pct), t["bias %"].map(rt.pct)
    return t


def mode_a_card(a) -> str:
    cap = lambda target, extra="": rt.caption(split="S-LOMO", target=target, prov=rt.version_text(a.configs, a.ma[("S-LOMO", target)].df["config_key"].unique()),
                                              run_ids=a.ma[("S-LOMO", target)].run_id, extra=extra)
    L = _head("Model card: Mode A (a priori per-image predictor)", a,
              "Predicts energy per image (mJ), latency per image (ms) and mean power (W) of an image-inference configuration from the model "
              "file's static features and the run configuration, before anything is run. Mode: a priori; nothing measured during the run is an input.")
    L += _list("Intended use", [
        "Planning the energy and latency of a vision model (YOLO family and classifiers like those in the data) on this Raspberry Pi 5 at a "
        "given image size, thread count, clock and precision, through `analytics.deploy.predict` and its 90 % interval.",
        "Reading the `no-similar-model` applicability flag: predictions flagged by novelty are outside what the validation supports."])
    L += _list("Out of scope", [*OUT_OF_SCOPE_COMMON, "Predicting latency or energy of models with no similar training model without the flag.",
                                "Governor and fan effects other than through the deployment calculator's measured offsets (rest on yolo11n)."])
    c = a.configs
    cells = int(c["n_cells"].sum())
    L += ["## Data and split\n",
          f"Training data: the Mode A configuration table, {ew.plural(len(c), 'configuration')} from {cells} QC-primary cells; "
          f"{rt.version_text(c, c['config_key'])}. Headline split S-LOMO (leave one base model out, 9 folds); other splits S-LSO, S-LVO, S-LCO, RANDOM "
          "are in `results.md`. Hyperparameters and AUTO are chosen by inner leave-one-model-out on each outer fold's training models only.\n"]
    L += ["## Features\n", "Known before running anything: " + ", ".join(f"`{f}`" for f in mm.FEATURES)
          + ". Never used: latency, temperature, baseline power or counters of the run being predicted.\n"]
    rows = []
    for t in ri_targets():
        raw = a.pooled[("S-LOMO", t)].set_index("candidate")
        base = a.base[("S-LOMO", t)]
        for cand in ("AUTO", base):
            r = raw.loc[cand]
            rows.append({"target": t, "predictor": rt.SHORT[cand], "n": int(r["n"]), "MdAPE % [95 % CI]": rt.pct_ci(r["mdape"], r["mdape_lo"], r["mdape_hi"]),
                         "bias % [95 % CI]": rt.pct_ci(r["bias_pct"], r["bias_pct_lo"], r["bias_pct_hi"]), "coverage90": r["coverage90"],
                         "run_id": a.ma[("S-LOMO", t)].run_id})
    L += ["## Headline metrics\n", rt.caption(split="S-LOMO", target="energy_mJ, latency_ms, mean_P_W",
                                             prov=rt.version_text(c, a.ma[("S-LOMO", "energy_mJ")].df["config_key"].unique()),
                                             run_ids=[a.ma[("S-LOMO", t)].run_id for t in ri_targets()],
                                             extra="Pooled over test configurations; intervals are cluster bootstrap over base models."),
          ew.md_table(pd.DataFrame(rows)) + "\n"]
    seeds = a.seed_runs
    if len(seeds) > 1:
        auto = [f"seed {s}: {rt.pct(evaluate.metrics(*[r.df.query('candidate == \"AUTO\"')[c] for c in ('y_true', 'y_pred')])['mdape'])} %"
                for s, r in sorted(seeds.items())]
        L += [f"Energy headline (AUTO MdAPE, S-LOMO) over the protocol seeds: {'; '.join(auto)}; see `results.md`, section 3.\n"]
    ref = a.master
    L += ["## Coverage\n", "The 90 % prediction interval comes from inner leave-one-model-out residual quantiles. Coverage of AUTO on S-LOMO energy by "
          "similarity class (coarse with 9 models):\n", cap("energy_mJ", "AUTO rows only."),
          ew.md_table(rt.similarity_table(ref.df).query("candidate == 'AUTO'").drop(columns="candidate")) + "\n"]
    L += ["## Worst subgroups\n", "AUTO on S-LOMO energy, the three base models and the similarity classes with the largest MdAPE:\n",
          cap("energy_mJ", "By base model."), ew.md_table(_worst(ref.df, "base_model")) + "\n",
          cap("energy_mJ", "By similarity class."), ew.md_table(_worst(ref.df, "similarity_class")) + "\n"]
    L += _list("Limitations", [*LIMITS_COMMON, "Mode A trains on pinned-clock, fan-full cells; the deployment calculator adds governor, fan and soak "
                               "effects from single measured offsets (one soak per model, fan offsets from yolo11n).",
                               "Static features of INT8 and FP16 variants come from the FP32 base model; FP16 (one configuration) is unsupported."])
    fits = [{"target": t, "candidate": f.get("candidate"), "params": json.dumps(f.get("params"), sort_keys=True),
             "AUTO chose (folds)": json.dumps(f.get("auto_counts"), sort_keys=True), "sha256": f.get("sha256"),
             "run_id": f.get("run_id")} for t, f in a.extras["fits"].items()]
    L += ["## Hashes and runs\n", rt.caption(split="all data", target="energy_mJ, latency_ms, mean_P_W", prov=f"n = {len(fits)} final models trained on {len(c)} configurations",
                                           run_ids=[f["run_id"] for f in fits], extra="Final models are refitted on every configuration, grid chosen by leave-one-model-out; "
                                                                                    "binaries are not committed (rebuilt by `cli.py mode-a-extras`)."),
          ew.md_table(pd.DataFrame(fits)) + "\n",
          f"Table sha256 `{a.master.rec.get('table_sha256')}`. Split sha256: "
          + ", ".join(f"{s} `{r.rec.get('split_sha256')}`" for s, r in ((k[0], v) for k, v in a.ma.items() if k[1] == "energy_mJ")) + ".\n"]
    return "\n".join(L).rstrip() + "\n"


def ri_targets() -> list:
    return ["energy_mJ", "latency_ms", "mean_P_W"]


def mode_l_card(a) -> str:
    lo, ml = a.ml["lomo"], a.ml
    raw = report_l.ladder(a)
    L = _head("Model card: Mode L (live 100 ms power model)", a,
              "Predicts board power (W) for each 100 ms window from the process's perf counters, sysfs state and the PMIC read, and accumulates it to "
              "energy. Mode: live, causal (features available at the window's end only); a ladder of rungs from a lookup (L0) to counters plus PMIC (L4).")
    L += _list("Intended use", [
        "Live energy accounting of a dedicated single-process image-inference deployment on this Raspberry Pi 5, with the same perf event set "
        "and sampling interval as the training cells.",
        "Probe mode: a short unmetered run of a new model, scored against Mode A on the same configurations (`results.md`, section 11)."])
    L += _list("Out of scope", [*OUT_OF_SCOPE_COMMON,
                                "System-wide or multi-process loads: counters cover the measured process's user-space code only (`perf_event_paranoid=2`), "
                                "other processes are invisible.",
                                "Windows shorter than 100 ms, per-step or per-layer attribution, a live scheduler.",
                                "Non-YOLO models: counter data is YOLO-only until the lockbox."])
    cells, folds = lo["cell_id"].nunique(), sorted(lo["fold"].unique())
    L += ["## Data and split\n",
          f"v2 counter cells of the YOLO models ({cells} held-out cells over folds {', '.join(folds)}; dataset v2, nov2026 rig). Split: leave one YOLO "
          "base model out; the grid point of each rung is chosen by inner leave-one-model-out on the training models; every window is weighted 1 / "
          "(windows in its cell). Windows before a cell's first available PMIC read have no PMIC feature; L1 predicts them with the fold's L0 lookup.\n",
          tl.EXPLORATORY_NOTE + "\n"]
    L += ["## Features\n", "L0: clock, governor, fan and busy / idle (task-clock share over 0.5). L1: PMIC rail power with zero-order hold. L2: task-clock "
          "and cycle rates. L3: all counter rates (IPC, cache and branch miss rates, backend stall share, LLC misses, context switches, page faults), "
          "sysfs and settings. L4: L3 plus the PMIC features. All features use data available at the window's end.\n"]
    sh = tl.show_ladder(raw)[["rung", "n", "whole-cell MdAPE % [95 % CI]", "whole-cell bias % [95 % CI]", "1 s block APE %", "10 s block APE %"]]
    L += ["## Headline metrics\n", rt.caption(split="leave one YOLO base model out", target="whole-cell energy", prov=report_l._prov(a), run_ids=ml["run_id"],
                                             extra="Interval: cluster bootstrap over the 5 held-out base models."), ew.md_table(sh) + "\n",
          report_l.overhead_text(a) + "\n"]
    L += ["## Coverage\n", "Mode L produces point predictions only: no prediction intervals, so there is no coverage figure. The stated uncertainty is the "
          "bootstrap interval of the metric above.\n"]
    ph = tl.phase_table(lo)
    L += ["## Worst subgroups\n", rt.caption(split="leave one YOLO base model out", target="whole-cell energy", prov=report_l._prov(a), run_ids=ml["run_id"],
                                             extra="Per held-out base model, median whole-cell APE of the rung with the lowest overall MdAPE among the non-exploratory rungs."),
          ew.md_table(_worst_folds(lo, raw)) + "\n",
          rt.caption(split="leave one YOLO base model out", target="phase energy", prov=report_l._prov(a), run_ids=ml["run_id"], extra="Median APE per phase."),
          ew.md_table(ph) + "\n"]
    L += _list("Limitations", [*LIMITS_COMMON, *report_l.LIMITS,
                               "No Mode L model file is frozen yet: the final rung is refitted on all data when the lockbox is frozen."])
    L += ["## Hashes and runs\n", f"Mode L run `{ml['run_id']}` (parts lomo, pmic_v1, probe), seed {ml['rec'].get('seed')}, commit `{str(ml['rec'].get('commit', ''))[:10]}`, "
          f"data hash `{str(ml['rec'].get('data_hash', ''))[:16]}`. Probe mode used Mode A run(s) "
          f"{', '.join('`' + str(x) + '`' for x in sorted({str(x) for x in ml['probe']['mode_a_run_id'].dropna()})) or 'none'}.\n"]
    return "\n".join(L).rstrip() + "\n"


def _worst_folds(lo: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    pre = raw[~raw["exploratory"]]
    rung = str(pre.loc[pre["mdape"].idxmin(), "rung"]) if len(pre) else str(raw["rung"].iloc[0])
    g = lo[lo["rung"] == rung].groupby("fold")["ape"].agg(["median", "size"]).sort_values("median", ascending=False)
    return pd.DataFrame({"rung": rung, "held-out base model": g.index, "n": g["size"].to_numpy(), "MdAPE %": g["median"].map(rt.pct).to_numpy()})

