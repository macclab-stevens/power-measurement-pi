"""Seam contract: the Pi's real EventLog/syncpulse output (committed fixture) through align()."""
from pathlib import Path

import numpy as np
import pandas as pd

from align import align
from recorder import window_stats
from store import SampleStore

FIXTURE = Path(__file__).parent / "fixtures" / "events_contract.csv"
COLUMNS = ["t_real_ns", "t_mono_ns", "kind", "name", "depth", "mode", "iter",
           "layer_path", "config_id", "window_id"]
B = 200_000.0 * (1 + 30e-6)
USB_MIN_MS = 2.5
BASE_W, ON_W = 3.2, 6.0


def _events():
    return pd.read_csv(FIXTURE, keep_default_na=False,
                       dtype={"kind": str, "name": str, "mode": str, "iter": str,
                              "layer_path": str, "config_id": str})


def _setup(tmp_path):
    ev = _events()
    rec, pi = tmp_path / "rec", tmp_path / "pi"
    pi.mkdir()
    ev.to_csv(pi / "events.csv", index=False)
    t_first, t_last = int(ev["t_real_ns"].iloc[0]), int(ev["t_real_ns"].iloc[-1])
    t_start = t_first - 2_000_000_000
    n = int((t_last + 2_000_000_000 - t_start) / B) + 1
    true_ns = t_start + B * np.arange(n)
    P = np.full(n, BASE_W)
    on = {}
    for _, r in ev[ev["kind"].isin(["pulse_on", "pulse_off"])].iterrows():
        if r["kind"] == "pulse_on":
            on[r["name"]] = int(r["t_real_ns"])
        elif r["name"] in on:
            P[(true_ns >= on.pop(r["name"])) & (true_ns < r["t_real_ns"])] = ON_W
    P += np.random.default_rng(1).normal(0, 0.05, n)
    s = SampleStore(rec)
    blk, rng = 50, np.random.default_rng(2)
    for i0 in range(0, n, blk):
        last = min(i0 + blk, n) - 1
        host = true_ns[last] + USB_MIN_MS * 1e6 + rng.exponential(2e6)
        m = last - i0 + 1
        s.append(P[i0:i0 + m] / 5.0, np.full(m, 5.0), int(host), 0)
    return s, rec, pi, ev


def _marker_t(ev, name):
    return int(ev[(ev["kind"] == "marker") & (ev["name"] == name)]["t_real_ns"].iloc[0])


def test_fixture_columns_match_pi_header():
    with open(FIXTURE) as f:
        assert f.readline().strip().split(",") == COLUMNS


def test_contract_align_and_labels(tmp_path):
    s, rec, pi, ev = _setup(tmp_path)
    s.close()
    res = align(rec, pi, pi)
    assert res["pass"] is True and res["n_edges"] == 12
    df = pd.read_parquet(pi / "power_trace.parquet")

    def at(t):
        return df.iloc[int(np.argmin(np.abs(df["t_pi_ns"].to_numpy() - t)))]

    sp = ev[(ev["layer_path"] == "0.0") & (ev["iter"] == "1")]
    mid = (int(sp[sp["kind"] == "begin"]["t_real_ns"].iloc[0]) +
           int(sp[sp["kind"] == "end"]["t_real_ns"].iloc[0])) // 2
    r = at(mid)
    assert r["layer_path"] == "0.0" and r["mode"] == "forward" and r["iter"] == "1"

    b0 = _marker_t(ev, "bench:cfg1")
    nxt = int(ev[(ev["kind"] == "marker") & (ev["t_real_ns"] > b0)]["t_real_ns"].iloc[0])
    r = at((b0 + nxt) // 2)
    assert r["phase"] == "bench:cfg1" and r["mode"] == "" and r["layer_path"] == ""


def test_window_stats_matches_offline_phase_means(tmp_path):
    s, rec, pi, ev = _setup(tmp_path)
    t0 = _marker_t(ev, "baseline_A")
    t1 = int(ev[(ev["kind"] == "marker") & (ev["t_real_ns"] > t0)]["t_real_ns"].iloc[0])
    w = window_stats(s, t0, t1)
    s.close()
    assert w["complete"] is True
    align(rec, pi, pi)
    df = pd.read_parquet(pi / "power_trace.parquet")
    offline = float(df[df["phase"] == "baseline_A"]["P_W"].mean())
    assert abs(w["mean_P"] - offline) < 0.02
