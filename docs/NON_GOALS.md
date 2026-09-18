# Non-goals (v2) — what we explicitly do NOT claim
1. No fused backends (TFLite/ONNX/inductor Conv+BN+SiLU fusion changes op table; v2 is eager PyTorch CPU only).
2. No wall-power claim without Monsoon (VDD_CORE only; wall = k*core + c calibrated on subset only).
3. No NPU/GPU, no threads!=4, no governor!=ondemand claims without re-collection.
4. No extrapolation beyond 2x trained T (T=256 from T≤128 is exploratory, report bound).
5. Old v2 runs are tainted — never mixed into v3 training.
Professor Q&A: why isolation? (200Hz vs ms ops — physical limit); why sum≠measured? (contention bound, reported); why power≈constant? (saturation, ablation proves correction value).
