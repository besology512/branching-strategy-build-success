"""Chapter 2 · Label every TravisTorrent project with a branching strategy.

For each of the 948 projects: measure branch lifetimes from build timestamps, count the
long-lived branches, trace feeder branches that merge into the main branch through the
commit graph, and apply the four-strategy rule (src/bsci/labeling.py).

Paper: Section V-C ("Metrics for determining branching strategy", "Feeder branch
identification", "Branching strategy classification").

Input : raw TravisTorrent release
Output: results/2_labeling/travis_rule_labels.csv
Check : the metrics and rule labels must equal data/derived/travis_project_labels.csv
Time  : ~15-30 min (one commit graph per project)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from bsci import travis, labeling
from bsci.paths import results_dir

COLS = ["gh_project_name", "git_branch", "gh_build_started_at", "git_all_built_commits", "git_trigger_commit"]


def main():
    builds = travis.read_raw(COLS)
    builds["gh_build_started_at"] = pd.to_datetime(builds["gh_build_started_at"], errors="coerce")
    threshold = labeling.travis_significance_threshold(builds)
    print(f"long-lived significance threshold (L_i / L1): {threshold:.4f}")

    rows = []
    for i, (name, p) in enumerate(builds.groupby("gh_project_name", sort=True), 1):
        m = labeling.travis_project_metrics(p, threshold)
        m["project"] = name
        m["strategy_rule"] = labeling.travis_rule(m["num_long_lived"], m["num_feeder_branches"],
                                                  m["median_feeder_lifetime_days"])
        rows.append(m)
        if i % 100 == 0:
            print(f"  {i} projects", flush=True)
    out = pd.DataFrame(rows)[["project", "strategy_rule", "num_long_lived", "l1_lifetime_days", "l2_l1_ratio",
                              "num_feeder_branches", "median_feeder_lifetime_days"]]
    out.attrs["threshold"] = threshold
    path = results_dir("2_labeling") / "travis_rule_labels.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} projects -> {path}")

    # Reproduction check against the published labels
    ref = travis.read_labels().set_index("project")
    new = out.set_index("project").loc[ref.index]
    same_label = (new["strategy_rule"] == ref["strategy_rule"]).mean()
    same_ll = (new["num_long_lived"] == ref["num_long_lived"]).mean()
    same_fd = (new["num_feeder_branches"] == ref["num_feeder_branches"]).mean()
    same_md = np.isclose(new["median_feeder_lifetime_days"], ref["median_feeder_lifetime_days"], equal_nan=True).mean()
    print(f"reproduction vs published: rule label {same_label:.1%} | long-lived {same_ll:.1%} | "
          f"feeders {same_fd:.1%} | median feeder lifetime {same_md:.1%}")
    print(out["strategy_rule"].value_counts().to_string())


if __name__ == "__main__":
    main()
