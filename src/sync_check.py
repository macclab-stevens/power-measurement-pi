"""Emit N sync-pulse trains and write events.csv, for validating alignment.

    uv run python src/sync_check.py --trains 20 --spacing 2 --out runs/sync_check
"""
from __future__ import annotations

import argparse
import os
import time

from eventlog import EventLog
from syncpulse import emit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trains", type=int, default=20)
    ap.add_argument("--spacing", type=float, default=2.0, help="idle seconds between trains")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = EventLog()
    ev.marker("idle")
    for k in range(a.trains):
        emit(ev, f"k{k}")
        ev.marker("idle")
        time.sleep(a.spacing)
    ev.marker("done")
    ev.write_csv(os.path.join(a.out, "events.csv"))
    print(f"[ok] {a.trains} trains -> {a.out}/events.csv")


if __name__ == "__main__":
    main()
