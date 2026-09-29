"""GHALogs: raw runs -> run table, branch lifetimes, and the modelling table (Tables 4, 6, 9, Fig. 3)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import SEED
from .paths import (GHALOGS_RUNS, GHALOGS_RUN_TABLE, GHALOGS_LIFETIMES, GHALOGS_STRATEGIES,
                    GHALOGS_REPOS, GHALOGS_REPO_META, require)

RAW_HINT = "Download GHALogs runs.json.gz (see chapters/1_data/README.md) or set BSCI_RAW_DIR."
BRANCHING = ["long_lived_count", "feeder_count", "median_feeder_seconds",
             "avg_branch_life_sec", "max_branch_life_sec", "min_branch_life_sec", "std_branch_life_sec"]
STANDARD_BASE = ["duration_sec", "logs_size", "run_number", "is_pr", "run_attempt"]
TARGET = "build_is_success"


def _branch(meta: dict):
    b = meta.get("head_branch")
    if b:
        return b
    ref = meta.get("ref")
    if ref:
        return ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    return None


def iter_runs():
    require(GHALOGS_RUNS, RAW_HINT)
    with gzip.open(GHALOGS_RUNS, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_run_table() -> pd.DataFrame:
    """One row per completed run (success/failure): the unit of analysis for GHALogs."""
    rows = []
    for obj in iter_runs():
        meta = obj.get("metadata", {}) or {}
        conclusion = meta.get("conclusion")
        if conclusion not in ("success", "failure"):
            continue
        try:
            t0 = datetime.fromisoformat(meta["run_started_at"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(meta["updated_at"].replace("Z", "+00:00"))
            duration = max((t1 - t0).total_seconds(), 0)
        except Exception:
            duration = np.nan
        rows.append({
            "run_id": meta.get("id"),
            "repository_name": obj.get("repository_name", "").lower(),
            "run_number": meta.get("run_number", np.nan),
            "event": meta.get("event", "unknown"),
            "duration_sec": duration,
            "logs_size": obj.get("total_logs_size", np.nan),
            "is_pr": int(bool(meta.get("pull_requests"))),
            "run_attempt": meta.get("run_attempt", 1),
            TARGET: int(conclusion == "success"),
        })
    return pd.DataFrame(rows)


def run_table() -> pd.DataFrame:
    if not GHALOGS_RUN_TABLE.exists():
        GHALOGS_RUN_TABLE.parent.mkdir(parents=True, exist_ok=True)
        build_run_table().to_parquet(GHALOGS_RUN_TABLE, index=False)
    return pd.read_parquet(GHALOGS_RUN_TABLE)


def build_branch_lifetimes() -> pd.DataFrame:
    """First and last run of every (repository, branch): the observed branch lifetime."""
    acc: dict[tuple[str, str], list] = {}
    for obj in iter_runs():
        meta = obj.get("metadata")
        repo = obj.get("repository_name")
        if not repo or not isinstance(meta, dict):
            continue
        branch, created = _branch(meta), meta.get("created_at")
        if branch is None or created is None:
            continue
        key = (repo, branch)
        if key in acc:
            acc[key][0] = min(acc[key][0], created)
            acc[key][1] = max(acc[key][1], created)
        else:
            acc[key] = [created, created]
    df = pd.DataFrame([(r, b, lo, hi) for (r, b), (lo, hi) in acc.items()],
                      columns=["repository_name", "branch", "min", "max"])
    lo, hi = pd.to_datetime(df["min"], utc=True), pd.to_datetime(df["max"], utc=True)
    df["observed_duration_seconds"] = (hi - lo).dt.total_seconds()
    return df.sort_values(["repository_name", "observed_duration_seconds"], ascending=[True, False])


def read_lifetimes() -> pd.DataFrame:
    return pd.read_csv(GHALOGS_LIFETIMES)


def read_strategies() -> pd.DataFrame:
    return pd.read_csv(GHALOGS_STRATEGIES)


def model_data() -> tuple[pd.DataFrame, pd.Series]:
    """Run-level feature matrix exactly as published (Table 4)."""
    df_runs = run_table()
    strat = read_strategies()
    strat["repository_name"] = strat["repository_name"].str.lower()
    strat = pd.concat([strat[["repository_name", "long_lived_count", "feeder_count", "median_feeder_seconds"]],
                       pd.get_dummies(strat["strategy"], prefix="strategy")], axis=1)
    life = read_lifetimes()
    life["repository_name"] = life["repository_name"].str.lower()
    life_agg = life.groupby("repository_name")["observed_duration_seconds"].agg(
        avg_branch_life_sec="mean", max_branch_life_sec="max",
        min_branch_life_sec="min", std_branch_life_sec="std").reset_index()
    df = df_runs.merge(strat, on="repository_name", how="left").merge(life_agg, on="repository_name", how="left")
    df = pd.concat([df.drop(columns=["event"]), pd.get_dummies(df["event"], prefix="event")], axis=1)
    X = df[[c for c in df.columns if c not in ("run_id", "repository_name", TARGET)]].copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median()) if X[c].dtype in (float, "float64") else X[c].fillna(0)
    X.columns = [str(c).replace("[", "").replace("]", "").replace("<", "").replace(">", "")
                 .replace('"', "").replace(",", "_") for c in X.columns]
    return X, df[TARGET].copy()


def split(test_size: float = 0.20):
    X, y = model_data()
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def read_repo_metadata() -> pd.DataFrame:
    """Main language, contributor count and creation date per repository (for confounders)."""
    return pd.read_csv(GHALOGS_REPO_META)


def build_repo_metadata(source=None) -> pd.DataFrame:
    """Extract language / contributors / creation date from the GHALogs repository file."""
    import ast
    from pathlib import Path
    src = Path(source) if source else GHALOGS_REPOS
    require(src, RAW_HINT.replace("runs.json.gz", "repositories.json.gz"))
    wanted = set(read_strategies()["repository_name"].str.lower())
    rows = []

    def take(r):
        name = str(r.get("name", "")).lower()
        if name in wanted:
            rows.append({"repository_name": name, "main_language": r.get("mainLanguage"),
                         "contributors": r.get("contributors"), "created_at": r.get("createdAt")})

    if str(src).endswith(".csv"):
        for ch in pd.read_csv(src, usecols=["repo"], chunksize=5000):
            for s in ch["repo"].dropna():
                try:
                    take(ast.literal_eval(s))
                except Exception:
                    continue
    else:
        opener = gzip.open if str(src).endswith(".gz") else open
        with opener(src, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                take(rec.get("repo", rec))
    return pd.DataFrame(rows).drop_duplicates("repository_name")
