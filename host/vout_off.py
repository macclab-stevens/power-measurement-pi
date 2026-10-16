#!/usr/bin/env python3
"""Switch the HVPM main-channel output (Vout) OFF. Manual safety tool.

Never called by the recorder or experiment code. The recorder is keep-on: stopping it leaves
Vout at its last setting, so the Pi stays powered. Required order:

  1. cleanly shut the Pi down (`sudo poweroff`) and confirm it is halted
  2. `systemctl --user stop monsoon-recorder` (it holds the USB device)
  3. run this script

    uv run python vout_off.py --confirm-pi-halted

Without the flag it explains this and exits with code 2 without touching USB. On success it
prints a confirmation line with the device serial.
"""
from __future__ import annotations

import argparse
import sys
import time

EXPLAIN = """vout_off.py switches the Monsoon output OFF and cuts power to the Pi.
Order: (1) cleanly shut the Pi down and confirm it is halted, (2) stop the recorder with
`systemctl --user stop monsoon-recorder`, (3) run this script with --confirm-pi-halted.
Refusing to run without --confirm-pi-halted (USB not touched)."""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-pi-halted", action="store_true",
                    help="assert the Pi is cleanly shut down and the recorder is stopped")
    a = ap.parse_args(argv)
    if not a.confirm_pi_halted:
        print(EXPLAIN, file=sys.stderr)
        raise SystemExit(2)

    from Monsoon import HVPM, pmapi          # imported only after the confirmation check
    from Monsoon import Operations as op

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
