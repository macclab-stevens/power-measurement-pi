"""Model-agnostic YOLO/object-detector wrapper around ultralytics (PyTorch CPU).

Any ultralytics weights file works (yolo11n, yolov8n, yolov12n, ...). `infer()`
is split from `plot()` so a profiler can wrap only the model forward pass.
Acceptance is class-agnostic: `max_conf` reports the strongest detection across
ALL classes, so the harness never depends on a specific object class.
"""
from __future__ import annotations

from ultralytics import YOLO


class Detector:
    def __init__(self, weights: str = "yolo11n.pt", imgsz: int = 640,
                 conf: float = 0.25, task: str | None = None, device: str = "cpu") -> None:
        self.model = YOLO(weights, task=task)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def infer(self, frame):
        """Single-frame detection; returns the ultralytics Result."""
        return self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
        )[0]

    @staticmethod
    def plot(result):
        return result.plot()

    @staticmethod
    def max_conf(result) -> float:
        """Highest confidence across ALL classes — the pipeline acceptance signal.
        Returns 0.0 when nothing is detected (boxes empty)."""
        boxes = result.boxes
        if boxes is None or boxes.conf is None or boxes.conf.numel() == 0:
            return 0.0
        return float(boxes.conf.max())