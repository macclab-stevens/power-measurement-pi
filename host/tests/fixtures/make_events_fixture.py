"""Generate events_contract.csv with the real Pi-side EventLog + syncpulse code.

Run once on the NUC (about 3 s, spins 2 cores) and commit the output:
    python3 host/tests/fixtures/make_events_fixture.py
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / ".." / "src"))

from eventlog import EventLog  # noqa: E402
from syncpulse import emit  # noqa: E402


def main() -> None:
    ev = EventLog()
    ev.marker("warmup")
    emit(ev, "start", ncores=2)
    ev.marker("baseline_A")
    time.sleep(0.3)
    ev.labels["mode"] = "forward"
    for it in range(3):
        ev.labels["iter"] = it
        ev.marker("forward")
        ev.begin("layer", "0")
        ev.begin("layer", "0.0")
        time.sleep(0.02)
        ev.end("layer", "0.0")
        ev.end("layer", "0")
        time.sleep(0.05)
    ev.marker("gap")
    time.sleep(0.2)
    ev.marker("bench:cfg1")
    time.sleep(0.3)
    ev.marker("gap")
    emit(ev, "end", ncores=2)
    ev.marker("done")
    ev.write_csv(HERE / "events_contract.csv")


if __name__ == "__main__":
    main()
