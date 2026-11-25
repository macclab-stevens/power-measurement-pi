# Legacy (superseded by `analytics/`)

This folder is an earlier modelling generation, kept for history. The current predictors, their evaluation and their
reports are in `analytics/`.

Known issues found by the 2026-11 external audit (`docs/review/`), not fixed:
- `modeling/train_scaling.py` reports in-sample OLS error, and summed isolated-operator replays miss cross-layer cache
  and dispatch effects.
- `trace_forecast/evaluate.py` feeds the family index `fam_idx` as one numeric column (an arbitrary linear order over
  families) and uses the live `temp0` reading, which is not known a priori.
