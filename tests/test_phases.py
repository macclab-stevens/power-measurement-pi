# tests/test_phases.py
def test_no_idle_contamination():
    # contract: inter-bench gaps must be 'gap', never 'idle'; forward work must exist.
    # Points at the ABAB smoke run (runs/fix_abab_smoke). NOTE: the lm task's
    # forward modes are named prefill/decode (not "forward" like the image task),
    # so accept any forward-work phase.
    import csv
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "runs",
                     "fix_abab_smoke", "samples.csv")
    rows = list(csv.DictReader(open(p)))
    phases = {r["phase"].split(":")[0] for r in rows}
    assert phases & {"forward", "prefill", "decode"}, f"missing forward work, got {phases}"
    assert "gap" in phases, "inter-bench gaps must be labeled gap"
    assert "idle" not in phases, f"idle phase leaked into samples: {sorted(phases)}"
    assert "baseline_A" in phases and "baseline_B" in phases, (
        f"missing ABAB baselines, got {phases}")
