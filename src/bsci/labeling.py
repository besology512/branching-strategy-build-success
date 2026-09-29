"""From branches to branching strategies.

Four strategies are assigned from three project-level metrics:

    long-lived branches   how many foundational lines of development exist
    feeder branches       branches whose work is merged into the main line
    median feeder life    how long a typical unit of parallel work lives

    feeders == 0                          -> One Branch
    long-lived >= 3 (and feeders > 0)     -> Git Flow
    median feeder lifetime < 7 days       -> Trunk-based
    otherwise                             -> Feature Branching

The *rules* are identical on both datasets. What differs is how the metrics can be
measured, because the two datasets record different things:

* TravisTorrent records the commits built by every build, so feeder branches are found
  by tracing merges into the main branch through the commit graph.
* GHALogs records runs, not commit parents, so every branch that is not long-lived is
  treated as a feeder branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

STRATEGIES = ["One Branch", "Trunk-based", "Feature Branching", "Git Flow"]
THRESHOLD_DAYS = 7.0


# --------------------------------------------------------------------------- TravisTorrent
def travis_significance_threshold(builds: pd.DataFrame) -> float:
    """Global 'top-quartile' threshold on the ratio of the 2nd- to the 1st-longest branch.

    Among projects with at least two branches, take the ratios L2/L1 that lie above their
    75th percentile, and return the 25th percentile of those. A branch counts as long-lived
    if its lifetime exceeds this fraction of the project's longest branch.
    """
    ratios = []
    for _, p in builds.groupby("gh_project_name", sort=False):
        life = p.groupby("git_branch")["gh_build_started_at"].agg(["min", "max"])
        dur = (life["max"] - life["min"]).dt.total_seconds()
        if len(dur) >= 2:
            top2 = dur.nlargest(2)
            if top2.iloc[0] > 0:
                ratios.append(top2.iloc[1] / top2.iloc[0])
    if not ratios:
        return 0.5
    q3 = np.quantile(ratios, 0.75)
    upper = [r for r in ratios if r > q3]
    return float(np.quantile(upper, 0.25)) if upper else float(q3)


def _commit_graph(project: pd.DataFrame) -> nx.DiGraph:
    """Commit graph from the commits built by each build (`git_all_built_commits`)."""
    df = project.sort_values("gh_build_started_at")
    g, info = nx.DiGraph(), {}

    def parents(s):
        if not isinstance(s, str):
            return []
        s = s.replace("#", "")
        return [s[i:i + 40] for i in range(0, len(s), 40)] if len(s) % 40 == 0 else []

    for trig, branch, t, built in zip(df["git_trigger_commit"], df["git_branch"],
                                      df["gh_build_started_at"], df["git_all_built_commits"]):
        info[trig] = {"branch": branch, "time": t}
        for par in parents(built):
            g.add_edge(par, trig)
            info.setdefault(par, {"branch": branch})
    for sha, attrs in info.items():
        g.add_node(sha)
        g.nodes[sha].update(attrs)
    return g


def travis_project_metrics(project: pd.DataFrame, threshold: float) -> dict:
    """Long-lived count, feeder branches and median feeder lifetime for one project."""
    life = project.groupby("git_branch")["gh_build_started_at"].agg(["min", "max"])
    life["days"] = (life["max"] - life["min"]).dt.total_seconds() / 86400
    life = life.sort_values("days", ascending=False)
    out = {"num_long_lived": 0, "l1_lifetime_days": np.nan, "l2_l1_ratio": np.nan,
           "num_feeder_branches": 0, "median_feeder_lifetime_days": np.nan}
    if life.empty:
        return out
    top = life.head(5)["days"]
    l1 = top.iloc[0]
    out["l1_lifetime_days"] = l1
    n_long = 1
    for i in range(1, len(top)):
        ratio = top.iloc[i] / l1 if l1 > 0 else 0
        if i == 1:
            out["l2_l1_ratio"] = ratio
        if ratio > threshold:
            n_long += 1
    out["num_long_lived"] = n_long

    main = life.index[0]
    g = _commit_graph(project)
    feeders = {g.nodes[u]["branch"] for u, v in g.edges()
               if "branch" in g.nodes[u] and "branch" in g.nodes[v]
               and g.nodes[v]["branch"] == main and g.nodes[u]["branch"] != main}
    out["num_feeder_branches"] = len(feeders)
    if feeders:
        out["median_feeder_lifetime_days"] = life.loc[life.index.intersection(feeders), "days"].median()
    return out


def travis_rule(num_long_lived, num_feeder_branches, median_feeder_days, threshold_days=THRESHOLD_DAYS) -> str:
    """Rule used by the automated TravisTorrent labeller (and shown to the human reviewer)."""
    if num_feeder_branches == 0:
        return "One Branch"
    if num_long_lived > 2:
        return "Git Flow"
    long_feeders = pd.notna(median_feeder_days) and median_feeder_days > threshold_days
    return "Feature Branching" if long_feeders else "Trunk-based"


# --------------------------------------------------------------------------- GHALogs
def ghalogs_strategies(lifetimes: pd.DataFrame) -> pd.DataFrame:
    """Branch-based classification of GHALogs repositories.

    `lifetimes` has one row per (repository_name, branch) with `observed_duration_seconds`.
    The longest branch (L1) is the main line; a branch is long-lived if it is L1 or lives
    longer than q * L1, where q is the 75th percentile of the L2/L1 ratio across all
    repositories. All other branches are feeder branches.
    """
    df = lifetimes.dropna(subset=["branch"]).copy()
    df["observed_duration_seconds"] = pd.to_numeric(df["observed_duration_seconds"], errors="coerce").fillna(0)
    df = df.sort_values(["repository_name", "observed_duration_seconds"], ascending=[True, False])
    l1 = (df.groupby("repository_name").head(1)[["repository_name", "branch", "observed_duration_seconds"]]
          .rename(columns={"branch": "l1_branch", "observed_duration_seconds": "l1_duration"}))
    l2 = df.groupby("repository_name").nth(1).reset_index()
    if l2.empty:
        q = 0.59
    else:
        r = l2[["repository_name", "observed_duration_seconds"]].merge(l1, on="repository_name")
        r = r[r["l1_duration"] > 0]
        q = (r["observed_duration_seconds"] / r["l1_duration"]).quantile(0.75)

    m = df.merge(l1, on="repository_name", how="left")
    m["l1_duration"] = m["l1_duration"].replace(0, 1)
    long_lived = (m["branch"] == m["l1_branch"]) | (m["observed_duration_seconds"] > q * m["l1_duration"])
    m["type"] = np.where(long_lived, "Long-Lived", "Feeder")

    out = pd.DataFrame({"repository_name": df["repository_name"].unique()})
    ll = m[m.type == "Long-Lived"].groupby("repository_name").size().rename("long_lived_count")
    fd = m[m.type == "Feeder"].groupby("repository_name").size().rename("feeder_count")
    md = m[m.type == "Feeder"].groupby("repository_name")["observed_duration_seconds"].median().rename("median_feeder_seconds")
    out = out.merge(ll, on="repository_name", how="left").fillna(0)
    out = out.merge(fd, on="repository_name", how="left").fillna(0)
    out = out.merge(md, on="repository_name", how="left")

    def rule(row):
        days = 0 if pd.isna(row.median_feeder_seconds) else row.median_feeder_seconds / 86400
        if row.feeder_count == 0:
            return "One Branch"
        if row.long_lived_count >= 3:
            return "Git Flow"
        return "Trunk-based" if days < THRESHOLD_DAYS else "Feature Branching"

    out["strategy"] = out.apply(rule, axis=1)
    out.attrs["long_lived_ratio_threshold"] = float(q)
    return out
