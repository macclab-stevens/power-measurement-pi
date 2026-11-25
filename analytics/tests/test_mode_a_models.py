"""Mode A candidates and the novelty distance, on synthetic frames."""
import itertools

import numpy as np
import pandas as pd
import pytest

import mode_a_models as mm
import settings
import tables

DESC_COLS = ["log_params", "log_macs_per_px", "depth", "frac_macs_conv", "frac_macs_linear",
             "frac_macs_norm", "frac_macs_act", "frac_macs_pool", "frac_macs_other"]


def make_desc(n=6, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, len(DESC_COLS))), columns=DESC_COLS,
                     index=pd.Index([f"m{i}" for i in range(n)], name="base_model"))
    return d


def make_frame(n_models=5, seed=1, clocks=(1500, 2000, 2400), precisions=("fp32", "int8"), threads=(1, 4),
               batches=(1,), imgsz=(640,), rig="oct2026"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_models):
        info = {k: float(rng.uniform(1, 3)) for k in ("log_weight_bytes", "log_act_bytes", "depth")}
        fr = rng.dirichlet(np.ones(6))
        macs0 = float(np.exp(rng.uniform(18, 21)))
        for prec, th, mhz, b, sz in itertools.product(precisions, threads, clocks, batches, imgsz):
            macs = macs0 * (sz / 640) ** 2
            r = {"config_key": f"k{len(rows)}", "base_model": f"m{i}", "variant": prec, "precision": prec,
                 "imgsz": sz, "batch": b, "threads": th, "cpu_mhz": mhz, "rig": rig,
                 "n_quantized_modules": 0 if prec == "fp32" else 50, "macs": macs,
                 "log_macs": np.log(macs), "log_params": np.log(macs0) - 3 + 0.1 * i, "cpu_ghz": mhz / 1000,
                 "is_int8": int(prec == "int8"), "is_fp16": int(prec == "fp16"), **info,
                 "act_bytes": float(np.exp(info["log_act_bytes"])), "n_module_calls": 50.0 + 10 * i,
                 "log_n_calls": float(np.log(50.0 + 10 * i)),
                 "v2f": float(tables.vcore_v(mhz) ** 2 * mhz / 1000)}
            r.update({f"frac_macs_{g}": f for g, f in zip(["conv", "linear", "norm", "act", "pool", "other"], fr)})
            rows.append(r)
    df = pd.DataFrame(rows)
    # noiseless log-linear truth
    z = (0.9 * df["log_macs"] + 0.2 * df["log_params"] - 0.3 * df["cpu_ghz"] - 0.1 * df["threads"]
         + 0.05 * df["batch"] - 0.4 * df["is_int8"] + 0.1 * df["is_fp16"])
    df["energy_mJ"] = np.exp(z - 10)
    df["latency_ms"] = np.exp(0.8 * df["log_macs"] - 0.5 * df["cpu_ghz"] - 0.2 * df["threads"] - 8)
    df["mean_P_W"] = df["energy_mJ"] / df["latency_ms"]
    return df


# ---- novelty ----------------------------------------------------------------------------------
def test_novelty_zero_for_training_model_and_grows_with_distance():
    d = make_desc()
    train = ["m0", "m1", "m2"]
    assert mm.novelty_distance(d, "m1", train) == 0.0
    far = d.copy()
    far.loc["m5"] = far.loc["m0"] + 0.5
    far.loc["m4"] = far.loc["m0"] + 5.0
    near_d = mm.novelty_distance(far, "m5", train)
    far_d = mm.novelty_distance(far, "m4", train)
    assert 0 < near_d < far_d


def test_novelty_standardises_over_training_models_only():
    d = pd.DataFrame({"a": [0.0, 1.0, 100.0], "b": [0.0, 2.0, 0.0]}, index=["x", "y", "z"])
    got = mm.novelty_distance(d, "z", ["x", "y"])      # mean/SD from x, y only (population SD)
    mu, sd = np.array([0.5, 1.0]), np.array([0.5, 1.0])
    exp = min(np.linalg.norm((np.array(d.loc[m]) - np.array(d.loc["z"])) / sd) for m in ["x", "y"])
    assert got == pytest.approx(exp)
    assert mu is not None


def test_novelty_sd_floor_for_constant_column():
    d = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 5.0]}, index=["x", "y", "z"])
    v = mm.novelty_distance(d, "z", ["x", "y"])
    assert np.isfinite(v)


# ---- baselines --------------------------------------------------------------------------------
def test_b0_median():
    df = make_frame()
    m = mm.B0Median().fit(df, "energy_mJ", {})
    assert np.all(m.predict(df) == np.median(df["energy_mJ"]))


@pytest.mark.parametrize("target", ["energy_mJ", "latency_ms"])
def test_b1_exact_when_proportional_to_macs(target):
    df = make_frame()
    c = {1500: 3e-9, 2000: 2e-9, 2400: 1.5e-9}
    df[target] = df["cpu_mhz"].map(c) * df["macs"]
    m = mm.B1MacsProportional().fit(df, target, {})
    np.testing.assert_allclose(m.predict(df), df[target], rtol=1e-12)


def test_b1_nearest_clock_and_tie_lower():
    df = make_frame(clocks=(1000, 2000))
    df["energy_mJ"] = df["cpu_mhz"].map({1000: 1e-9, 2000: 5e-9}) * df["macs"]
    m = mm.B1MacsProportional().fit(df, "energy_mJ", {})
    t = df.iloc[:3].copy()
    t["cpu_mhz"] = [1400, 1500, 1600]                     # 1500 is a tie -> lower clock (1000)
    np.testing.assert_allclose(m.predict(t), np.array([1e-9, 1e-9, 5e-9]) * t["macs"])


def test_b1_power_is_per_clock_median_without_macs():
    df = make_frame()
    m = mm.B1MacsProportional().fit(df, "mean_P_W", {})
    med = df.groupby("cpu_mhz")["mean_P_W"].median()
    np.testing.assert_allclose(m.predict(df), df["cpu_mhz"].map(med))
    t = df.copy()
    t["macs"] = t["macs"] * 10
    np.testing.assert_allclose(m.predict(t), m.predict(df))


def test_b3_recovers_loglinear_coefficients():
    df = make_frame(precisions=("fp32", "int8", "fp16"), batches=(1, 4))
    m = mm.B3LogLinear().fit(df, "energy_mJ", {})
    np.testing.assert_allclose(m.coef_, [0.9, 0.2, -0.3, -0.1, 0.05, -0.4, 0.1], atol=1e-6)
    np.testing.assert_allclose(m.predict(df), df["energy_mJ"], rtol=1e-6)


@pytest.mark.parametrize("cls,params", [(mm.M1Ridge, {"alpha": 0.1}), (mm.M2Trees, {"max_depth": 3})])
@pytest.mark.parametrize("target", ["energy_mJ", "latency_ms", "mean_P_W"])
def test_learned_fit_predict_positive(cls, params, target):
    df = make_frame()
    p = cls().fit(df, target, params).predict(df)
    assert p.shape == (len(df),) and np.all(p > 0) and np.all(np.isfinite(p))


def test_m1_fits_scaler_on_training_rows_only():
    df = make_frame()
    m = mm.M1Ridge().fit(df, "energy_mJ", {"alpha": 1.0})
    sc = m.model_.named_steps["scale"]
    np.testing.assert_allclose(sc.mean_[mm.ALL_FEATURES.index("log_macs")], df["log_macs"].mean())


def test_rig_nov_is_derived_from_rig():
    df = make_frame()
    df2 = df.copy()
    df2["rig"] = "nov2026"
    X = mm.feature_matrix(df2, mm.ALL_FEATURES)
    assert np.all(X[:, -1] == 1) and np.all(mm.feature_matrix(df, mm.ALL_FEATURES)[:, -1] == 0)


def test_m2_default_seed_and_override():
    df = make_frame()
    m = mm.M2Trees().fit(df, "energy_mJ", {})
    assert m.model_.random_state == settings.seed()
    m = mm.M2Trees().fit(df, "energy_mJ", {"random_state": 7})
    assert m.model_.random_state == 7


def test_m3_is_product_of_parts():
    df = make_frame()
    spec = {"latency": {"name": "M1Ridge", "params": {"alpha": 0.1}},
            "power": {"name": "M2Trees", "params": {"max_depth": 2, "max_iter": 50}}}
    m = mm.M3Factored().fit(df, "energy_mJ", spec)
    lat = mm.M1Ridge().fit(df, "latency_ms", {"alpha": 0.1}).predict(df)
    pw = mm.M2Trees().fit(df, "mean_P_W", {"max_depth": 2, "max_iter": 50}).predict(df)
    np.testing.assert_allclose(m.predict(df), lat * pw)


def test_m3_only_energy_and_needs_parts():
    df = make_frame()
    spec = {"latency": {"name": "M1Ridge", "params": {}}, "power": {"name": "M1Ridge", "params": {}}}
    with pytest.raises(ValueError):
        mm.M3Factored().fit(df, "latency_ms", spec)
    with pytest.raises(ValueError):
        mm.M3Factored().fit(df, "energy_mJ", {"latency": spec["latency"]})
    assert mm.GRIDS["M3Factored"] is None


# ---- B2 ---------------------------------------------------------------------------------------
def b2_frame():
    """Two training models; m0 is the nearest to the test model m2 in descriptor space."""
    rows = []
    for mdl, base in (("m0", 10.0), ("m1", 1000.0)):
        for prec, th, b, mhz in itertools.product(["fp32", "int8"], [1, 4], [1, 4], [1500, 2000]):
            e = base + 100 * (prec == "int8") + 10 * th + 3 * b + mhz / 100
            rows.append({"base_model": mdl, "imgsz": 640, "batch": b, "threads": th, "cpu_mhz": mhz,
                         "precision": prec, "macs": 1e9 if mdl == "m0" else 4e9, "energy_mJ": e,
                         "latency_ms": e * 2, "mean_P_W": 5.0 + (mdl == "m1") + mhz / 1000})
    return pd.DataFrame(rows)


B2_DESC = pd.DataFrame({"a": [0.0, 10.0, 1.0], "b": [0.0, 10.0, 1.0]}, index=pd.Index(["m0", "m1", "m2"], name="base_model"))


def b2_test_row(**kw):
    r = {"base_model": "m2", "imgsz": 640, "batch": 1, "threads": 4, "cpu_mhz": 2000, "precision": "int8", "macs": 2e9}
    r.update(kw)
    return pd.DataFrame([r])


def b2_expected(train, mdl, macs_test=2e9, **match):
    q = train[train["base_model"] == mdl]
    for k, v in match.items():
        q = q[q[k] == v]
    assert len(q) == 1
    return q["energy_mJ"].iloc[0] * macs_test / q["macs"].iloc[0]


def test_b2_exact_match_nearest_model_and_mac_ratio():
    tr = b2_frame()
    m = mm.B2NearestModel(B2_DESC).fit(tr, "energy_mJ", {})
    got = m.predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(tr, "m0", batch=1, threads=4, cpu_mhz=2000, precision="int8"))


def test_b2_power_unscaled_and_latency_scaled():
    tr = b2_frame()
    p = mm.B2NearestModel(B2_DESC).fit(tr, "mean_P_W", {}).predict(b2_test_row())[0]
    assert p == pytest.approx(5.0 + 2.0)       # m0, 2000 MHz, no MAC scaling
    lat = mm.B2NearestModel(B2_DESC).fit(tr, "latency_ms", {}).predict(b2_test_row())[0]
    row = tr[(tr.base_model == "m0") & (tr.batch == 1) & (tr.threads == 4) & (tr.cpu_mhz == 2000) & (tr.precision == "int8")]
    assert lat == pytest.approx(row["latency_ms"].iloc[0] * 2.0)


def test_b2_relaxation_order_threads_then_batch_then_precision_then_clock():
    tr = b2_frame()
    m = mm.B2NearestModel(B2_DESC)
    # drop threads: threads 4 absent; batch/precision/clock match
    t1 = tr[tr["threads"] != 4]
    got = m.fit(t1, "energy_mJ", {}).predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(t1, "m0", batch=1, threads=1, cpu_mhz=2000, precision="int8"))
    # then batch: no row has batch 1 with int8 at 2000 and threads 4 -> drop batch (threads dropped too)
    t2 = tr[(tr["threads"] != 4) & (tr["batch"] != 1)]
    got = m.fit(t2, "energy_mJ", {}).predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(t2, "m0", batch=4, threads=1, cpu_mhz=2000, precision="int8"))
    # then precision: only fp32 left
    t3 = t2[t2["precision"] == "fp32"]
    got = m.fit(t3, "energy_mJ", {}).predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(t3, "m0", batch=4, threads=1, cpu_mhz=2000, precision="fp32"))
    # then nearest clock: only 1500 left; the test row asks for 2400 -> nearest is 2000 (absent) -> 1500
    t4 = t3[t3["cpu_mhz"] == 1500]
    got = m.fit(t4, "energy_mJ", {}).predict(b2_test_row(cpu_mhz=2400))[0]
    assert got == pytest.approx(b2_expected(t4, "m0", batch=4, threads=1, cpu_mhz=1500, precision="fp32"))


def test_b2_relaxation_keeps_batch_before_precision():
    tr = b2_frame()
    # remove int8 at batch 1/threads 4; int8 exists at batch 4 and fp32 exists at batch 1
    t = tr[~((tr["precision"] == "int8") & (tr["batch"] == 1))]
    t = t[t["threads"] != 4]
    got = mm.B2NearestModel(B2_DESC).fit(t, "energy_mJ", {}).predict(b2_test_row())[0]
    # threads dropped, then batch dropped (precision int8 kept): picks the int8 batch-4 row
    assert got == pytest.approx(b2_expected(t, "m0", batch=4, threads=1, cpu_mhz=2000, precision="int8"))


def test_b2_nearest_clock_tie_goes_to_lower():
    tr = b2_frame()
    tr = tr[(tr["batch"] == 1) & (tr["threads"] == 1) & (tr["precision"] == "fp32")]
    got = mm.B2NearestModel(B2_DESC).fit(tr, "energy_mJ", {}).predict(b2_test_row(cpu_mhz=1750))[0]
    assert got == pytest.approx(b2_expected(tr, "m0", cpu_mhz=1500))


def test_b2_standardisation_uses_training_models_only():
    # m2 is far from m0 in raw units but the training-model scale decides the nearest model
    d = pd.DataFrame({"a": [0.0, 1.0, 0.4], "b": [0.0, 100.0, 50.0]}, index=pd.Index(["m0", "m1", "m2"], name="base_model"))
    tr = b2_frame()
    got = mm.B2NearestModel(d).fit(tr, "energy_mJ", {}).predict(b2_test_row())[0]
    # standardised over {m0, m1}: m0 at (-1,-1), m1 at (1,1), m2 at (-0.2, 0): m0 (1.28) nearer than m1 (1.56)
    assert got == pytest.approx(b2_expected(tr, "m0", batch=1, threads=4, cpu_mhz=2000, precision="int8"))
    d2 = d.copy()
    d2.loc["m2"] = [0.9, 60.0]
    got = mm.B2NearestModel(d2).fit(tr, "energy_mJ", {}).predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(tr, "m1", batch=1, threads=4, cpu_mhz=2000, precision="int8"))


def test_b2_nearest_model_only_among_candidate_rows():
    tr = b2_frame()
    tr = tr[~((tr["base_model"] == "m0") & (tr["precision"] == "int8"))]   # m0 has no int8 rows
    got = mm.B2NearestModel(B2_DESC).fit(tr, "energy_mJ", {}).predict(b2_test_row())[0]
    assert got == pytest.approx(b2_expected(tr, "m1", batch=1, threads=4, cpu_mhz=2000, precision="int8"))


def test_b2_no_training_row_at_imgsz_falls_back_to_nearest_clock():
    tr = b2_frame()
    got = mm.B2NearestModel(B2_DESC).fit(tr, "energy_mJ", {}).predict(b2_test_row(imgsz=224))
    assert np.all(np.isfinite(got)) and np.all(got > 0)


# ---- contract ---------------------------------------------------------------------------------
def build(name, desc):
    return mm.B2NearestModel(desc) if name == "B2NearestModel" else mm.CANDIDATES[name]()


M3_SPEC = {"latency": {"name": "M1Ridge", "params": {"alpha": 1.0}},
           "power": {"name": "M2Trees", "params": {"max_iter": 30}}}


def all_runs():
    for name in mm.CANDIDATES:
        yield name, (M3_SPEC if name == "M3Factored" else {})


@pytest.mark.parametrize("name,params", list(all_runs()))
def test_predict_before_fit_raises(name, params):
    df = make_frame(n_models=3)
    with pytest.raises(ValueError):
        build(name, make_desc()).predict(df)


@pytest.mark.parametrize("name,params", list(all_runs()))
def test_fit_does_not_mutate_input_and_returns_self(name, params):
    df = make_frame(n_models=4)
    before = df.copy(deep=True)
    m = build(name, make_desc())
    assert m.fit(df, "energy_mJ", params) is m
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("name,params", list(all_runs()))
@pytest.mark.parametrize("target", mm.TARGETS)
def test_no_candidate_reads_columns_outside_allowed_set(name, params, target):
    """Frames carry exactly FEATURES (rig_nov derived from rig) + CONFIG_COLS + target + macs;
    everything else (other measures, dyn energy, CVs, cell ids, family, ...) is dropped."""
    if name == "M3Factored" and target != "energy_mJ":
        pytest.skip("M3 is energy only")
    df = make_frame(n_models=4)
    df["dyn_energy_mJ"] = 1.0
    df["energy_cv"] = 0.01
    allowed = set(mm.FEATURES) | set(mm.FEATURES_R2) | set(tables.CONFIG_COLS) | {"macs", "act_bytes", "n_module_calls"}
    factored = name in ("M3Factored", "M4Physics") and target == "energy_mJ"
    extra = {"latency_ms", "mean_P_W"} if factored else {target}   # M3 and M4 fit their two parts
    train = df[sorted(allowed | extra)]
    test = df[sorted(allowed)]
    desc = make_desc()
    m = build(name, desc).fit(train, target, params)
    p = m.predict(test)
    assert len(p) == len(test) and np.all(p > 0)
    # predictions are unchanged when the unreadable columns hold garbage
    junk = df.copy()
    for c in set(df.columns) - allowed - extra:
        if df[c].dtype != object:
            junk[c] = np.random.default_rng(0).normal(size=len(df))
    if factored:
        junk["energy_mJ"] = -1.0
    p2 = build(name, desc).fit(junk, target, params).predict(junk)
    np.testing.assert_allclose(p, p2)


def test_registry_and_grids():
    assert set(mm.CANDIDATES) == set(mm.GRIDS)
    for n in ("B0Median", "B1MacsProportional", "B2NearestModel", "B3LogLinear"):
        assert mm.GRIDS[n] == [{}]
    np.testing.assert_allclose([g["alpha"] for g in mm.GRIDS["M1Ridge"]], np.logspace(-2, 4, 7))
    g = mm.GRIDS["M2Trees"]
    assert len(g) == 24 and {x["max_depth"] for x in g} == {2, 3, 4}
    assert {x["min_samples_leaf"] for x in g} == {3, 5} and {x["max_iter"] for x in g} == {100, 300}
    assert {x["learning_rate"] for x in g} == {0.05, 0.1}


def test_nonpositive_target_rejected():
    df = make_frame(n_models=3)
    df.loc[0, "energy_mJ"] = 0.0
    with pytest.raises(ValueError):
        mm.M1Ridge().fit(df, "energy_mJ", {})
    with pytest.raises(ValueError):
        mm.M1Ridge().fit(df, "bogus", {})


def test_real_table_smoke():
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "reports" / "mode_a_configs.csv"
    df = pd.read_csv(p)
    for name, params in all_runs():
        desc = make_desc_from_table(df)
        p_ = build(name, desc).fit(df, "energy_mJ", params).predict(df)
        assert np.all(p_ > 0)


def make_desc_from_table(df):
    g = df.groupby("base_model").first()
    out = pd.DataFrame({"log_params": g["log_params"], "log_macs_per_px": np.log(g["macs"] / g["imgsz"] ** 2),
                        "depth": g["depth"], **{c: g[c] for c in g.columns if c.startswith("frac_macs_")}})
    return out


# ---- R2: features and M4Physics -----------------------------------------------------------------
def test_features_r2_drops_two_adds_two_and_models_use_it():
    assert set(mm.FEATURES) - set(mm.FEATURES_R2) == {"log_weight_bytes", "frac_macs_other"}
    assert set(mm.FEATURES_R2) - set(mm.FEATURES) == {"v2f", "log_n_calls"}
    assert mm.ALL_FEATURES == mm.FEATURES_R2 + ["rig_nov"]


def m4_frame(n_models=5, seed=2):
    """Latency exactly additive in the M4 cost terms (beta 0.75), power affine in the M4 design."""
    df = make_frame(n_models=n_models, seed=seed, precisions=("fp32", "int8", "fp16"))
    u = 1.0 / df["cpu_ghz"].to_numpy()
    s = df["threads"].to_numpy(float) ** 0.75
    q = df["macs"].to_numpy() * u / s
    df["latency_ms"] = (2e-9 * q * (1 - df["is_int8"] - df["is_fp16"]) + 1e-9 * q * df["is_int8"]
                        + 3e-9 * q * df["is_fp16"] + 5e-8 * df["act_bytes"] * u / s + 0.01 * df["n_module_calls"] * u + 4.0)
    df["mean_P_W"] = 2.0 + 1.5 * df["v2f"] + 0.1 * df["v2f"] * df["threads"]
    df["energy_mJ"] = df["latency_ms"] * df["mean_P_W"]
    return df


@pytest.mark.parametrize("target", mm.TARGETS)
def test_m4_fit_predict_shapes_and_positive(target):
    df = make_frame(n_models=4)
    m = mm.M4Physics().fit(df, target, {})
    p = m.predict(df.iloc[:7])
    assert p.shape == (7,) and np.all(np.isfinite(p)) and np.all(p > 0)


def test_m4_nnls_coefficients_are_non_negative_even_when_the_truth_wants_a_negative_term():
    df = m4_frame()
    df["latency_ms"] = df["latency_ms"] - 3.9 * 1.0   # a negative intercept is not representable
    df["latency_ms"] = df["latency_ms"].clip(lower=0.2)
    for beta in (0.5, 0.75, 1.0):
        m = mm.M4Physics().fit(df, "latency_ms", {"beta": beta})
        assert m.lat_coef_.shape == (6,) and np.all(m.lat_coef_ >= 0) and m.beta_ == beta
        assert np.all(m.predict(df) > 0)


def test_m4_recovers_an_exactly_additive_latency_and_affine_power():
    df = m4_frame(n_models=6)
    lat = mm.M4Physics().fit(df, "latency_ms", {"beta": 0.75}).predict(df)
    np.testing.assert_allclose(lat, df["latency_ms"], rtol=1e-4)
    pw = mm.M4Physics().fit(df, "mean_P_W", {"alpha": 1e-6})
    np.testing.assert_allclose(pw.predict(df), df["mean_P_W"], rtol=1e-3)


def test_m4_energy_is_latency_times_power_of_the_chosen_parts():
    df = m4_frame()
    spec = {"latency": {"beta": 0.5}, "power": {"alpha": 10.0}}
    e = mm.M4Physics().fit(df, "energy_mJ", spec).predict(df)
    lat = mm.M4Physics().fit(df, "latency_ms", {"beta": 0.5}).predict(df)
    pw = mm.M4Physics().fit(df, "mean_P_W", {"alpha": 10.0}).predict(df)
    np.testing.assert_allclose(e, lat * pw)


def test_m4_rows_are_balanced_per_base_model():
    """A base model with 10x the rows must not outweigh a small one: the weights give each model equal total weight."""
    df = m4_frame(n_models=3)
    big = pd.concat([df[df["base_model"] == "m0"]] * 4 + [df[df["base_model"] != "m0"]], ignore_index=True)
    w = mm._row_weights(big, big["latency_ms"].to_numpy(float))
    tot = pd.Series(w ** 2 * big["latency_ms"].to_numpy(float) ** 2).groupby(big["base_model"].to_numpy()).sum()
    np.testing.assert_allclose(tot.to_numpy(), 1.0)   # (w*y)^2 = 1/n_model per row


def test_m4_does_not_mutate_input_and_has_per_target_grids():
    df = m4_frame()
    before = df.copy(deep=True)
    mm.M4Physics().fit(df, "energy_mJ", {})
    pd.testing.assert_frame_equal(df, before)
    assert [g["beta"] for g in mm.GRIDS["M4Physics"]["latency_ms"]] == [0.5, 0.75, 1.0]
    assert [g["alpha"] for g in mm.GRIDS["M4Physics"]["mean_P_W"]] == [0.01, 0.1, 1, 10, 100]
    assert mm.grid_for(mm.GRIDS, "M1Ridge", "energy_mJ") == mm.GRIDS["M1Ridge"]
    assert mm.grid_for(mm.GRIDS, "M4Physics", "mean_P_W") == mm.GRIDS["M4Physics"]["mean_P_W"]
    with pytest.raises(ValueError):
        mm.grid_for(mm.GRIDS, "M4Physics", "energy_mJ")


def test_model_without_features_attribute_predicts_on_the_r1_feature_list():
    """A model pickled before R2 has no `features_`; it must keep using the columns it was fitted on."""
    df = make_frame(n_models=4)
    m = mm.M1Ridge().fit(df, "energy_mJ", {"alpha": 1.0})
    assert m.features_ == mm.ALL_FEATURES
    old = mm.M1Ridge()
    old.model_ = Pipeline_fit(df, mm.R1_ALL_FEATURES, np.log(df["energy_mJ"]))
    old._fitted = True
    np.testing.assert_allclose(old.predict(df), np.exp(old.model_.predict(mm.feature_matrix(df, mm.R1_ALL_FEATURES))))


def Pipeline_fit(df, cols, y):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(mm.feature_matrix(df, cols), y)


def test_hook_features_enter_only_when_hooks_vary():
    import pandas as pd
    t = pd.DataFrame({"hooks_on": [1, 1, 1]})
    assert not mm.hooks_vary(t) and mm.features_for(t) == mm.ALL_FEATURES
    t2 = pd.DataFrame({"hooks_on": [1, 0, 1]})
    assert mm.hooks_vary(t2) and mm.features_for(t2) == mm.ALL_FEATURES + mm.HOOK_FEATURES


def test_m4_hook_term_only_with_hooks():
    import pandas as pd
    df = pd.DataFrame({"cpu_ghz": [2.4, 2.4], "threads": [4, 4], "macs": [1e9, 2e9], "act_bytes": [1e6, 2e6],
                       "is_int8": [0, 0], "is_fp16": [0, 0], "n_module_calls": [100, 200], "hooks_on": [1, 0]})
    a, b = mm.M4Physics.latency_design(df, 1.0), mm.M4Physics.latency_design(df, 1.0, hooks=True)
    assert a.shape == (2, 6) and b.shape == (2, 7)
    assert b[0, 5] == b[0, 4] and b[1, 5] == 0.0          # the hook term is the call term on hooks-on rows only


def test_attention_term_only_with_attention_models():
    import pandas as pd
    df = pd.DataFrame({"cpu_ghz": [2.4], "threads": [4], "macs": [1e9], "act_bytes": [1e6], "is_int8": [0], "is_fp16": [0],
                       "n_module_calls": [100], "macs_attn": [5e8]})
    assert mm.M4Physics.latency_design(df, 1.0, attn=True).shape == (1, 7)
    assert not mm.attn_present(pd.DataFrame({"frac_attn": [0.0, 0.01]}))
    t = pd.DataFrame({"frac_attn": [0.0, 0.3], "hooks_on": [1, 1]})
    assert mm.attn_present(t) and mm.features_for(t)[-1] == "frac_attn"
