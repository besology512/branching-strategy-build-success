"""Chapter 2 · Label every GHALogs repository with a branching strategy.

GHALogs records workflow runs but not the parent commits of each build, so merges into the
main branch cannot be traced as they are on TravisTorrent. Instead, branch lifetimes are
measured from the first and last run on each branch; the longest branch is the main line,
branches living longer than q x main line are long-lived (q = 75th percentile of the
second-to-first branch ratio), and every other branch is a feeder branch. The same
four-strategy rule is then applied.

Paper: Section V-C; strategy counts in Table 7.

Input : raw GHALogs runs.json.gz  (or, with --from-derived, the versioned lifetimes)
Output: data/derived/ghalogs_branch_lifetimes.csv.gz (when rebuilt from raw)
        results/2_labeling/ghalogs_rule_labels.csv
Check : labels must equal data/derived/ghalogs_repo_strategies.csv
Time  : ~10-20 min from raw (streams 1 GB of JSON); seconds from derived
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from bsci import ghalogs, labeling
from bsci.paths import results_dir


def main(from_derived: bool):
    if from_derived:
        life = ghalogs.read_lifetimes()
    else:
        life = ghalogs.build_branch_lifetimes()
        ref_life = ghalogs.read_lifetimes()
        key = ["repository_name", "branch"]
        cmp = life.merge(ref_life, on=key, suffixes=("", "_ref"))
        print(f"branch lifetimes rebuilt: {len(life):,} (published {len(ref_life):,}); "
              f"identical durations for {np.isclose(cmp.observed_duration_seconds, cmp.observed_duration_seconds_ref).mean():.2%} of matched branches")

    labels = ghalogs_labels = labeling.ghalogs_strategies(life)
    print(f"long-lived ratio threshold q = {labels.attrs['long_lived_ratio_threshold']:.4f}")
    path = results_dir("2_labeling") / "ghalogs_rule_labels.csv"
    labels.to_csv(path, index=False)

    ref = ghalogs.read_strategies().set_index("repository_name")
    new = ghalogs_labels.set_index("repository_name").loc[ref.index]
    print(f"{len(labels):,} repositories -> {path}")
    print(f"reproduction vs published: strategy {(new.strategy == ref.strategy).mean():.2%} | "
          f"feeders {(new.feeder_count == ref.feeder_count).mean():.2%} | "
          f"long-lived {(new.long_lived_count == ref.long_lived_count).mean():.2%}")
    print(labels["strategy"].value_counts().to_string())


if __name__ == "__main__":
    main(from_derived="--from-derived" in sys.argv)
