"""Central configuration for the Dynamic Routing interactive interface.

Data locations are controlled by two environment variables so the raw NWB
sessions or the precomputed artifacts can be relocated without code changes:

    DR_DATA_DIR         directory containing *.nwb session files
                        (default: <repo>/nwb_sessions)
    DR_PRECOMPUTED_DIR  directory for precomputed artifacts consumed by the app
                        (default: <repo>/data/precomputed)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("DR_DATA_DIR", REPO_ROOT / "nwb_sessions"))
PRECOMPUTED_DIR = Path(
    os.environ.get("DR_PRECOMPUTED_DIR", REPO_ROOT / "data" / "precomputed")
)

# Value of intervals/epochs/script_name that marks the Dynamic Routing task epoch.
TASK_SCRIPT_NAME = "DynamicRouting1"


def list_sessions(data_dir: Path | None = None) -> list[Path]:
    """Return sorted paths of all .nwb session files in the data directory."""
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    return sorted(d.glob("*.nwb"))


def session_id_from_path(p: Path | str) -> str:
    """Session id is the file stem, e.g. '759434_2025-02-04'."""
    return Path(p).stem
