"""Supply-harness loss: the part of the Monsoon's measured energy that is I^2 R heat in the leads and connectors
between the meter and the board, so board energy = Monsoon energy - R * integral of I^2 dt.

The round-trip resistance is fitted per rig from the Monsoon voltage against the Pi's own EXT5V rail
(reports/measurement_audit.md section 3, audit/cable.py): it is 6-7 % of the measured power at the active
current. `data.rig_of` names a cell's rig."""
import numpy as np

# ohm, round trip (leads + connectors); oct2026 = v1 data at 5.25 V, nov2026 = v2, v3, lockbox and later at 5.40 V
R_OHM = {"oct2026": 0.179, "nov2026": 0.212}


def harness_mJ(t_s, I_A, t_start_s, t_end_s, r_ohm) -> np.ndarray:
    """R * integral of I^2 dt over each window [t_start_s[k], t_end_s[k]], in mJ.

    The integral of I^2 over the samples (`t_s` seconds since trace start, ascending) is the cumulative
    trapezoid, read at the window edges by linear interpolation. NaN for a window that is not inside
    [t_s[0], t_s[-1]] (or has a NaN edge), for a NaN `r_ohm`, and for every window when the trace has fewer
    than two samples or a non-finite current."""
    a = np.atleast_1d(np.asarray(t_start_s, dtype=float))
    b = np.atleast_1d(np.asarray(t_end_s, dtype=float))
    t = np.asarray(t_s, dtype=float)
    i2 = np.asarray(I_A, dtype=float) ** 2
    out = np.full(a.shape, np.nan)
    if len(t) < 2 or not np.isfinite(r_ohm) or not np.isfinite(i2).all():
        return out
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (i2[1:] + i2[:-1]) * np.diff(t))])
    inside = (a >= t[0]) & (b <= t[-1]) & (b >= a)           # NaN edges compare False
    out[inside] = r_ohm * (np.interp(b[inside], t, cum) - np.interp(a[inside], t, cum)) * 1000.0
    return out
