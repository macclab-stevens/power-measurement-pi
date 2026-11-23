"""Paths and configuration shared by every analytics module."""
import os
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def data_root() -> Path:
    """Dataset root: env ANALYTICS_DATA, default ~/pi-power-data (read-only except derived/)."""
    return Path(os.environ.get("ANALYTICS_DATA") or "~/pi-power-data").expanduser()


def derived_dir() -> Path:
    """data_root()/derived, created on demand: the only place analytics writes under the data root."""
    d = data_root() / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cfg() -> dict:
    """Parsed analytics/config.toml."""
    with open(HERE / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def seed() -> int:
    """Master seed; every stochastic step derives from it."""
    return int(cfg()["seed"])
