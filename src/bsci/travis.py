"""TravisTorrent: raw builds -> modelling table (the table behind Tables 3, 6, 9 and Fig. 2).

Faithful to the published pipeline:
  1. join every build to its project's strategy label (human label, 948 projects);
  2. keep 22 features, impute numeric gaps with the median (test counts with 0);
  3. one-hot encode the strategy label and the build language;
  4. balance the four strategies with SMOTE (random_state=42), which adds synthetic builds;
  5. models are trained on a 500,000-row stratified sample, split 80/20.

Note on step 4: SMOTE is applied before the train/test split, so the test set contains
synthetic builds interpolated from real ones. See docs/limitations.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import SEED
from .paths import TRAVIS_RAW, TRAVIS_LABELS, TRAVIS_MODEL_TABLE, require

FEATURES = [
    "gh_team_size", "gh_repo_age", "gh_sloc", "human_label", "num_long_lived",
    "median_feeder_lifetime_days", "num_feeder_branches", "git_diff_src_churn",
    "git_diff_test_churn", "gh_num_commits_on_files_touched", "gh_diff_files_added",
    "gh_diff_files_deleted", "gh_diff_files_modified", "tr_duration", "gh_is_pr",
    "gh_by_core_team_member", "tr_log_lan", "tr_log_num_tests_run", "tr_log_num_tests_ok",
    "tr_log_num_tests_failed", "tr_log_num_tests_skipped", "tr_log_testduration",
]
BRANCHING = ["num_long_lived", "median_feeder_lifetime_days", "num_feeder_branches"]
STRATEGY_DUMMIES = ["strategy_Feature Branching", "strategy_Git Flow", "strategy_One Branch", "strategy_Trunk-based"]
TARGET = "build_is_success"

RAW_HINT = "Download TravisTorrent (see chapters/1_data/README.md) or set BSCI_RAW_DIR."


def read_raw(columns: list[str] | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """Read the raw TravisTorrent release (3.88 M build jobs) in chunks."""
    require(TRAVIS_RAW, RAW_HINT)
    parts = [c for c in pd.read_csv(TRAVIS_RAW, usecols=columns, chunksize=chunksize, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def read_labels() -> pd.DataFrame:
    """Project-level strategy labels and branching metrics (948 projects)."""
    return pd.read_csv(TRAVIS_LABELS)


def builds_with_labels(columns: list[str]) -> pd.DataFrame:
    """Raw builds joined to their project's human strategy label (inner join keeps row order)."""
    lab = read_labels().rename(columns={"project": "gh_project_name", "strategy_human": "human_label"})
    b = read_raw(list(dict.fromkeys(["gh_project_name"] + columns)))
    return b.merge(lab, on="gh_project_name", how="inner")


def encoded_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix, success target and strategy label, encoded exactly as published."""
    y = (df["tr_status"] == "passed").astype(int)
    X = df[[f for f in FEATURES if f in df.columns]].copy()
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in [c for c in X.columns if "tr_log_num_tests" in c or "tr_log_testduration" in c]:
        X[col] = X[col].fillna(0)
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Column order is identical to the published table; only the language dummies get an
    # honest "lang_" prefix (they were mislabelled "strategy_" in the original files).
    X = pd.get_dummies(X, columns=cats, prefix={"human_label": "strategy", "tr_log_lan": "lang"}, dummy_na=False)
    return X, y, df["human_label"]


def build_model_table() -> pd.DataFrame:
    """Recreate the SMOTE-balanced modelling table from the raw release (~20-40 min)."""
    from imblearn.over_sampling import SMOTE
    raw_cols = [f for f in FEATURES if f != "human_label" and f not in BRANCHING]   # branching cols come from the labels
    df = builds_with_labels(["tr_status"] + raw_cols)
    X, y, strategy = encoded_features(df)
    Xc = X.copy()
    Xc[TARGET] = y
    X_res, labels = SMOTE(random_state=SEED).fit_resample(Xc, strategy)
    X_res["human_label"] = labels
    X_res.attrs["n_real_rows"] = len(Xc)
    return X_res


def model_table() -> pd.DataFrame:
    """The cached SMOTE table (built on first use)."""
    if not TRAVIS_MODEL_TABLE.exists():
        TRAVIS_MODEL_TABLE.parent.mkdir(parents=True, exist_ok=True)
        t = build_model_table()
        t.to_parquet(TRAVIS_MODEL_TABLE, index=False)
    return pd.read_parquet(TRAVIS_MODEL_TABLE)


def split(sample_size: int = 500_000, test_size: float = 0.20, table: pd.DataFrame | None = None):
    """Stratified sample of the balanced table, then a stratified 80/20 split (seed 42)."""
    t = model_table() if table is None else table
    X = t[[c for c in t.columns if c not in (TARGET, "human_label")]]
    y = t[TARGET]
    X_s, _, y_s, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=SEED)
    return train_test_split(X_s, y_s, test_size=test_size, stratify=y_s, random_state=SEED)
