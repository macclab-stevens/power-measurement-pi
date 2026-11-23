"""Append-only run log (analytics/runs.jsonl): who ran what, on which code and data."""
import hashlib
import json
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import settings

DEFAULT_LOG = settings.HERE / "runs.jsonl"
SOURCES = ("v1", "v2", "lockbox")


def file_hash(path) -> str:
    """sha256 hex of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_hash(root=None, sources=SOURCES) -> str:
    """sha256 over the bytes of every <root>/<source>/dataset_index.csv present (of `sources`), sorted by path."""
    root = Path(root) if root else settings.data_root()
    h = hashlib.sha256()
    for p in sorted(root / s / "dataset_index.csv" for s in sources):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=settings.HERE, capture_output=True, text=True,
                          check=True).stdout


def code_files() -> dict:
    """{`analytics/<name>.py`: sha256} of every git-tracked top-level analytics module, sorted by path, hashed from the
    working tree (so a dirty tree is identified by what actually ran). Falls back to every top-level *.py when git is not available."""
    try:
        names = [n for n in _git("ls-files", "--full-name", "-z", "--", ":(glob)*.py").split("\0") if n]
        root = settings.HERE
        paths = {n: root / Path(n).name for n in names}
    except (OSError, subprocess.CalledProcessError):
        paths = {f"analytics/{p.name}": p for p in settings.HERE.glob("*.py")}
    return {n: file_hash(p) for n, p in sorted(paths.items())}


def code_sha256(files: dict | None = None) -> str:
    """sha256 over the sorted `path  sha256` lines of `files` (default: `code_files()`): one hash for the code that ran."""
    files = code_files() if files is None else files
    return hashlib.sha256("\n".join(f"{k}  {v}" for k, v in sorted(files.items())).encode()).hexdigest()


def append(record: dict, path=None) -> dict:
    """Add run_id, ts, commit, dirty, data_hash, code_sha256, machine to record; append it as one JSON line."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    rec = {
        "run_id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "data_hash": data_hash(),
        "code_sha256": code_sha256(),
        "machine": socket.gethostname(),
        **record,
    }
    path = Path(path) if path else DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec
