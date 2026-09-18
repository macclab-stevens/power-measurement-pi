# trace_forecast/stitch.py
import math
def stitch(plan, t_fwd_prior_s, base_fn, delta_fn, dur_rel_std=0.1, step=0.1):
    # R5: durations are PLANNED (frames x train prior), never test-measured.
    # Bands combine model rmse and duration uncertainty in quadrature (log scale).
    out = []; t = 0.0; e_cum = 0.0
    for seg in plan:
        dur = seg["frames"]*t_fwd_prior_s if seg.get("frames") else seg.get("dur_s", 0.0)
        dP, rmse = delta_fn(seg["phase"])
        sig = math.sqrt(rmse*rmse + dur_rel_std*dur_rel_std)
        n = max(1, int(round(dur/step)))
        for _ in range(n):
            pb = base_fn(t); p = pb + dP
            lp = math.log(max(p, 1e-6))
            lo, hi = math.exp(lp - 1.96*sig), math.exp(lp + 1.96*sig)
            e_cum += p*step
            out.append({"t_s": round(t,1), "phase": seg["phase"], "P_mean": p, "P_lo": lo, "P_hi": hi,
                        "E_cum": e_cum, "E_lo": math.exp(math.log(max(e_cum,1e-6)) - 1.96*sig),
                        "E_hi": math.exp(math.log(max(e_cum,1e-6)) + 1.96*sig)})
            t += step
    return out
