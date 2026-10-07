#!/usr/bin/env python3
"""Switch the HVPM main-channel output (Vout) OFF.

The recorder is keep-on: stopping it leaves Vout at its last setting, so the Pi stays
powered. Run this only after a clean Pi shutdown and after stopping the recorder
(`systemctl --user stop monsoon-recorder`; the recorder holds the USB device), and before
touching wiring:

    uv run python vout_off.py

Prints a confirmation line with the device serial once the command has been sent.
"""
from __future__ import annotations

import time

from Monsoon import HVPM, pmapi
from Monsoon import Operations as op


def main() -> None:
    protocol = pmapi.USB_protocol()
    serials = protocol.enumerateDevices()
    if not serials:
        raise SystemExit("[FAIL] no Monsoon on USB (2ab9:0001)")
    protocol.Connect(op.HardwareModel.HVPM, serials[0])
    mon = HVPM.Monsoon()
    mon.setup_usb(serials[0], protocol)
    try:
        mon.stopSampling()
    except Exception:  # noqa: BLE001 - not sampling is fine
        pass
    mon.setVout(0)
    time.sleep(0.2)
    mon.fillStatusPacket()
    print(f"[ok] HVPM {int(mon.statusPacket.serialNumber)}: Vout set to 0 V (output off)")
    mon.closeDevice()


if __name__ == "__main__":
    main()
