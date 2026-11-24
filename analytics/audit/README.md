# Audit levers (2026-11-24)

Small scripts that re-check the claims of the external audit (`docs/review/`) against the data in `~/pi-power-data`.
Run them from `analytics/` (they import `data`): `uv run python audit/<script>.py <out>`.

| script | checks | output |
|---|---|---|
| `slip_shear.py OUT.csv` | timestamp error of the sync-line fit after sample-loss slips; effect on energy per image and 100 ms windows | `slip_shear_2026-11-24.csv` |
| `cable.py OUT.parquet` | harness resistance per rig from Monsoon V − PMIC EXT5V vs current | printed |
| `rig_test.py R_OCT R_NOV` | rig gap on identical configurations, raw and with I²R removed | printed |
| `ctx_vcore.py` | effect of masking the vcgencmd (`ctx_read`) samples; VDD_CORE per clock | printed |
| `hookbench.py` (on the Pi) | frame latency with LayerProfiler hooks + layer events on vs off | `hookbench_2026-11-24.csv` |
