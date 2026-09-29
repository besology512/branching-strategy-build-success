"""Chapter 2 · Which branches are long-lived and which are short-lived? (descriptive)

Five rules are applied in order; each only labels builds that no earlier rule labelled:

  0a  name pattern     master, main, develop, release-*, hotfix-*, ...        -> long-lived
  0b  lifetime         branch active for >= 30 days                            -> long-lived
  1   PR-focused       <= 5 commits and (>= 80% PR builds or last build is PR) -> short-lived
  2   merge lifecycle  lifetime <= 7 days and some builds carry a merge commit -> short-lived
  3   clustering       k-means (k=2) on lifetime, builds, commits, builds/week -> long / short

Note: as in the original analysis, branch statistics are aggregated by branch *name*
across all projects (so every "master" is pooled). The result is used descriptively
(the lifecycle distribution reported in the paper); the strategy labels in
travis_classify.py are computed per project and do not depend on it.

Paper: Section V-C ("Identifying long-lived and short-lived branches").
Input : raw TravisTorrent release
Output: results/2_labeling/travis_branch_lifecycle.csv
Time  : ~5-10 min
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from bsci import travis, SEED
from bsci.paths import results_dir

LONG_NAMES = ["master", "main", "develop", "development", "dev", "trunk", "stable", "release", "production",
              "prod", "staging", "stage", "devel", "master-*", "release-*", "hotfix-*"]
LIFETIME_DAYS, MERGE_DAYS, PR_COMMITS = 30, 7, 5
COLS = ["git_branch", "gh_first_commit_created_at", "gh_pushed_at", "gh_build_started_at",
        "gh_num_commits_in_push", "gh_is_pr", "git_merged_with"]


def long_name(b) -> bool:
    if pd.isna(b):
        return False
    b = str(b).lower()
    return any(re.match("^" + p.replace("*", ".*") + "$", b) if "*" in p else b == p for p in LONG_NAMES)


def main():
    df = travis.read_raw(COLS)
    for c in ["gh_first_commit_created_at", "gh_pushed_at", "gh_build_started_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    df["event_timestamp"] = df[["gh_first_commit_created_at", "gh_pushed_at", "gh_build_started_at"]].min(axis=1, skipna=True)
    df["label"] = pd.NA

    # branch-level statistics (by branch name, as in the original)
    g = df.groupby("git_branch")
    br = g["event_timestamp"].agg(first="min", last="max")
    br["lifetime_days"] = (br["last"] - br["first"]).dt.days.clip(lower=0).fillna(0)
    br["num_builds"] = g.size()
    br["total_commits"] = g["gh_num_commits_in_push"].apply(lambda s: s.fillna(0).sum())
    br["pr_ratio"] = g["gh_is_pr"].mean()
    br["last_is_pr"] = df.sort_values("event_timestamp").groupby("git_branch")["gh_is_pr"].last()
    br["merge_ratio"] = df["git_merged_with"].notna().groupby(df["git_branch"]).mean()
    wk = df[["git_branch", "event_timestamp"]].dropna()
    br["active_weeks"] = wk.assign(w=wk.event_timestamp.dt.strftime("%G-%V")).groupby("git_branch")["w"].nunique()
    br["active_weeks"] = br["active_weeks"].fillna(0)
    br["builds_per_week"] = br["num_builds"] / np.maximum(br["active_weeks"], 1)
    f = df["git_branch"].map
    life, commits, pr, last_pr, merge = (f(br[c]) for c in ["lifetime_days", "total_commits", "pr_ratio", "last_is_pr", "merge_ratio"])

    unl = lambda: df["label"].isna()
    df.loc[df["git_branch"].apply(long_name), "label"] = "long_lived_name_pattern"
    df.loc[unl() & (life > 0) & (life >= LIFETIME_DAYS), "label"] = "long_lived_lifetime_threshold"
    df.loc[unl() & (commits <= PR_COMMITS) & ((pr >= 0.8) | (last_pr == True)), "label"] = "short_lived_pr_focused"
    df.loc[unl() & (life <= MERGE_DAYS) & (merge > 0), "label"] = "short_lived_merge_lifecycle"

    rest = sorted(df.loc[unl(), "git_branch"].dropna().unique())  # groupby order, as in the original
    if len(rest) >= 2:
        feats = br.loc[rest, ["lifetime_days", "num_builds", "total_commits", "builds_per_week"]].fillna(0)
        cl = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(StandardScaler().fit_transform(feats))
        long_cluster = int(feats.groupby(cl)["lifetime_days"].mean().idxmax())
        long_set = set(feats.index[cl == long_cluster])
        rest_mask = unl()
        df.loc[rest_mask & df["git_branch"].isin(long_set), "label"] = "long_lived_cluster"
        df.loc[rest_mask & ~df["git_branch"].isin(long_set), "label"] = "short_lived_cluster"

    out = (df.groupby("label", dropna=False)
           .agg(builds=("git_branch", "size"), branches=("git_branch", "nunique")).reset_index())
    out["share_of_builds_pct"] = (100 * out["builds"] / len(df)).round(1)
    out = out.sort_values("builds", ascending=False)
    path = results_dir("2_labeling") / "travis_branch_lifecycle.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
