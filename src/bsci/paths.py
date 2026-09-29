"""Where everything lives.

Raw datasets are *not* part of the repository (they are several GB and have their own
licences). Download them as described in chapters/1_data/README.md, or point the
environment variable BSCI_RAW_DIR at the folder that holds them.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW = Path(os.environ.get("BSCI_RAW_DIR", ROOT / "data" / "raw"))
CACHE = Path(os.environ.get("BSCI_CACHE_DIR", ROOT / "data" / "cache"))  # regenerable, git-ignored
DERIVED = ROOT / "data" / "derived"  # small, versioned research artefacts (labels, strategies)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Raw inputs
TRAVIS_RAW = RAW / "travistorrent" / "final-2017-01-25.csv"
GHALOGS_RUNS = RAW / "ghalogs" / "runs.json.gz"
GHALOGS_REPOS = RAW / "ghalogs" / "repositories.json.gz"

# Versioned derived data
TRAVIS_LABELS = DERIVED / "travis_project_labels.csv"
GHALOGS_LIFETIMES = DERIVED / "ghalogs_branch_lifetimes.csv.gz"
GHALOGS_STRATEGIES = DERIVED / "ghalogs_repo_strategies.csv"
GHALOGS_REPO_META = DERIVED / "ghalogs_repo_metadata.csv"

# Regenerable caches
TRAVIS_MODEL_TABLE = CACHE / "travis_model_table_smote.parquet"
GHALOGS_RUN_TABLE = CACHE / "ghalogs_runs.parquet"


def results_dir(chapter: str) -> Path:
    d = RESULTS / chapter
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure_path(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path
