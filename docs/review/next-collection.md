# Next collection: changes recommended by the 2026-11 audit

What to change before the next Monsoon campaign, in priority order. Each item names what it fixes and the evidence
(`docs/review/2026-11-24-audit-response.md`, `analytics/audit/`). Items marked **(approval)** need lab time, root on
the Pi, or physical rewiring. None of them has been done.

## 1. Run power cells with the layer hooks off

`run.py --layer-hooks off`. Static features come from `tools/static_features.py` on the NUC; its output matches the
Pi's layer tables exactly.

- **Why:** every measured cell so far ran with hooks and layer events on, which slows frames by 10–12 % for YOLO, 21 %
  for EfficientNet-B0 and 43 % for MobileNetV3 (`audit/hookbench_2026-11-24.csv`). A deployment has no hooks.
- **First batch (approval):** a hooks-off calibration subset that pairs with existing cells. Suggested: yolo11n,
  yolov8n, yolo11s, yolo12n, mbv3, effb0 and resnet18 at their existing configurations, 2400 MHz, 3 repeats each.
  That is 21 cells, about 30 minutes. It measures the instrumented-to-plain energy ratio, so the existing dataset can
  be mapped to deployment energy.

## 2. A fresh lockbox for R2 (approval)

The R2 models were designed after the lockbox was seen, so their lockbox scores are post hoc. A second lockbox gives
R2 a confirmatory test:

- 3–5 configurations, including at least two base models never collected;
- hooks off;
- predictions committed before measurement.

## 3. Keep the sync-pulse fix and add mid-run ticks

`host/align.py` now inserts the detected sample-loss steps before fitting the clock line. For cells longer than a
few minutes (soak), also add a short single-core sync tick every 60 s, so that the fit has interior points.

## 4. Supply harness (approval, physical)

The round-trip resistance from the Monsoon to the PMIC input is 0.18–0.21 Ω, which puts 6–7 % of the measured power
in the leads. 43 % of cells dipped below 4.90 V at the PMIC.

- Use short (< 20 cm), thick (14–16 AWG) leads to the 5 V and GND header pins, aiming for < 0.03 Ω.
- Keep Monsoon Vout fixed (5.40 V) across all campaigns.
- Clear or record the sticky throttle bits before each batch.

## 5. Kernel-visible counters (approval, root on the Pi)

`perf_event_paranoid=2` forces user-only counting, so `context-switches` reads 0 in every window, and kernel and I/O
time are invisible.

- Option: `sysctl kernel.perf_event_paranoid=-1`, or `CAP_PERFMON` for the perf binary, added to the whitelisted lab
  image.
- This is a security trade-off on a lab-only device, so it is your decision.

## 6. Faster sysfs polling

`SysfsSampler(period_s=...)` defaults to 1 s; under `ondemand` the clock changes every 10–20 ms. Poll at 20 Hz
(file reads, < 0.05 ms each). Mode L now also derives a 10 Hz effective clock from the counters.

## 7. Pause the vcgencmd context reads during scored phases

They cover about 1 % of scored time and shift power by about 0.15 W while they run. That is negligible for energy
per image (0.02 %), but easy to avoid: read PMIC and temperature only in the baselines and between phases.

## 8. More architectures (approval)

Mode A has 9 base models, 85 % of the rows YOLO; Mode L has counter data for 5 base models, all YOLO. Counter models
did not transfer to classifiers on the lockbox.

- Next diversity, with counters on: ConvNeXt-Tiny, ViT-Tiny or Swin-T (attention), MobileNetV3-Large, ResNet-34/50,
  EfficientNet-B2, RT-DETR.
- Each at 2–3 clocks, 3 repeats.
