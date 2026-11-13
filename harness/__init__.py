"""VTAS evaluation harness: ground truth (CHAIR-GT), fixed splits, statistics, gates and
external dataset loaders. Not part of the ``vtas-metric`` package; run from the repo root.

Data lives outside git, under ``$VTAS_DATA`` (default: ``<repo>/data``).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = Path(__file__).resolve().parent
SPLITS_DIR = HARNESS_DIR / "splits"


def data_dir() -> Path:
    """Root folder for downloaded datasets (never committed)."""
    return Path(os.environ.get("VTAS_DATA", REPO_ROOT / "data"))
