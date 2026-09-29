"""Chapter 5 · How often do builds pass under each branching strategy?

Raw (unbalanced) builds and runs are joined to their project's strategy, and the pass
rate is computed per strategy. These are build-weighted rates: large projects contribute
more builds. Chapter 5's confounders.py shows how much that weighting matters.

Paper: Tables 7 (distribution) and 8 (success rates); RQ3.
Output: results/5_strategy_and_success/success_rates.csv, figures/success_rates.png
Time  : ~3 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from bsci import travis, ghalogs, style
from bsci.paths import results_dir, figure_path


def main():
    t = travis.builds_with_labels(["tr_status"])
    t["success"] = (t["tr_status"] == "passed").astype(int)
    g = ghalogs.run_table().merge(ghalogs.read_strategies().assign(
        repository_name=lambda d: d.repository_name.str.lower())[["repository_name", "strategy"]], on="repository_name")
    rows = []
    for ds, df, col, succ, proj in [("TravisTorrent", t, "human_label", "success", "gh_project_name"),
                                    ("GHALogs", g, "strategy", "build_is_success", "repository_name")]:
        agg = df.groupby(col).agg(builds=(succ, "size"), projects=(proj, "nunique"), success_rate=(succ, "mean"))
        agg["share_of_builds"] = agg["builds"] / agg["builds"].sum()
        for s, r in agg.iterrows():
            rows.append({"dataset": ds, "strategy": s, **r.to_dict()})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir("5_strategy_and_success") / "success_rates.csv", index=False)
    print(df.round(4).to_string(index=False))
    figure(df)


def figure(df):
    style.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, ds in zip(axes, ["TravisTorrent", "GHALogs"]):
        d = df[df.dataset == ds].set_index("strategy").loc[style.STRATEGY_ORDER[::-1]]
        for i, (s, r) in enumerate(d.iterrows()):
            ax.barh(i, r.success_rate * 100, color=style.STRATEGY_COLORS[s], height=0.6)
            ax.text(r.success_rate * 100 + 1, i, f"{r.success_rate:.1%}", va="center", fontsize=9)
            ax.text(1.5, i, f"{int(r.projects):,} projects · {int(r.builds):,} builds", va="center",
                    fontsize=7.5, color="white")
        ax.set_yticks(range(4), d.index)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Builds that pass (%)")
        ax.set_title(ds, loc="left")
        ax.tick_params(axis="y", length=0)
        style.grid(ax)
    fig.suptitle("Pass rate by branching strategy (build-weighted)", x=0.02, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout()
    style.save(fig, figure_path("success_rates.png"))


if __name__ == "__main__":
    main()
