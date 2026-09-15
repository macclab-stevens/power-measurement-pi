"""Picamera2 wrapper: 640x480 RGB888, warmup, AE/AWB lock, frames()."""
from __future__ import annotations

import time

from picamera2 import Picamera2


class Camera:
    def __init__(self, size=(640, 480), warmup_s: float = 2.0, drop_frames: int = 3) -> None:
        self._size = size
        self._warmup_s = warmup_s
        self._drop_frames = drop_frames
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.meta: dict = {}

    def start(self) -> dict:
        self.picam2.start()
        time.sleep(self._warmup_s)
        for _ in range(self._drop_frames):  # let AE/AWB settle before locking
            self.picam2.capture_array()
        try:
            self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})
        except Exception as e:  # noqa: BLE001 - lock is best-effort; determinism warning
            print(f"[warn] AE/AWB lock failed: {e}")
        m = self.picam2.capture_metadata()
        self.meta = {
            "exposure_us": (m.get("ExposureTime") or 0) / 1000.0,
            "analogue_gain": m.get("AnalogueGain"),
            "colour_gain": list(m["ColourGain"]) if m.get("ColourGain") else None,
            "ae_locked": True,
            "awb_locked": True,
        }
        return self.meta

    def frames(self):
        while True:
            yield self.picam2.capture_array()

    def stop(self) -> None:
        try:
            self.picam2.stop()
        finally:
            self.picam2.close()