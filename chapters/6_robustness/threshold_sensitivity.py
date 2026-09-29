"""Chapter 6 · Does the 7-day boundary between trunk-based and feature branching matter?

Projects are relabelled with the boundary set to 1, 3, 7, 14 and 30 days (One-Branch and
Git Flow are unaffected), and the pass rates of trunk-based and feature-branching
projects are recomputed with the automated rule. On TravisTorrent trunk-based projects
pass more builds than feature-branching projects for every boundary from 1 to 14 days
(the gap is largest, 8.75 points, at 1 day); at 30 days the two converge. 7 days follows
the trunk-based-development literature. On GHALogs the feature-branching group is too
small (5-607 repositories depending on the boundary) for a stable comparison.

Paper: Section V-C (seven-day threshold paragraph).
Output: results/6_robustness/threshold_sensitivity.csv, figures/threshold_sensitivity.png
Time  : ~3 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from bsci import travis, ghalogs, style
from bsci.paths import results_dir, figure_path

THRESHOLDS = [1, 3, 7, 14, 30]


def relabel(n_long, n_feed, med_days, t):
    if n_long >= 3 and n_feed > 0:
        return "Git Flow"
    if n_feed == 0 or pd.isna(n_feed):
        return "One Branch"
    if n_long <= 2:
        return "Trunk-based" if (pd.isna(med_days) or med_days < t) else "Feature Branching"
    return "One Branch"


def rates(projects, builds, key, dataset):
    rows = []
    for t in THRESHOLDS:
        lab = projects.assign(strategy=[relabel(a, b, c, t) for a, b, c in
                                        zip(projects.num_long_lived, projects.num_feeder_branches,
                                            projects.median_feeder_lifetime_days)])
        m = builds.merge(lab[[key, "strategy"]], on=key)
        for s, g in m.groupby("strategy"):
            rows.append({"dataset": dataset, "threshold_days": t, "strategy": s,
                         "builds": len(g), "projects": g[key].nunique(), "success_rate": g["success"].mean()})
    return rows


def main():
    tp = travis.read_labels().rename(columns={"project": "gh_project_name"})
    tb = travis.read_raw(["gh_project_name", "tr_status"])
    tb["success"] = (tb["tr_status"] == "passed").astype(int)
    gp = ghalogs.read_strategies().rename(columns={"long_lived_count": "num_long_lived",
                                                   "feeder_count": "num_feeder_branches"})
    gp["median_feeder_lifetime_days"] = gp["median_feeder_seconds"].fillna(0) / 86400
    gp["repository_name"] = gp["repository_name"].str.lower()
    gb = ghalogs.run_table()[["repository_name", "build_is_success"]].rename(columns={"build_is_success": "success"})
    df = pd.DataFrame(rates(tp, tb, "gh_project_name", "TravisTorrent") + rates(gp, gb, "repository_name", "GHALogs"))
    df.to_csv(results_dir("6_robustness") / "threshold_sensitivity.csv", index=False)
    print(df.round(4).to_string(index=False))
    figure(df)


def figure(df):
    style.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for ax, ds in zip(axes, ["TravisTorrent", "GHALogs"]):
        d = df[df.dataset == ds]
        for s, mk in [("Trunk-based", "o"), ("Feature Branching", "s")]:
            r = d[d.strategy == s].sort_values("threshold_days")
            ax.plot(r.threshold_days, r.success_rate * 100, marker=mk, lw=2, ms=6,
                    color=style.STRATEGY_COLORS[s], mec="white", label=s)
            if s == "Feature Branching":      # show how many projects each point rests on
                trunk = d[d.strategy == "Trunk-based"].set_index("threshold_days").success_rate * 100
                for x, y, n in zip(r.threshold_days, r.success_rate * 100, r.projects):
                    above = y > trunk.get(x, float("inf"))   # label on the side away from the other line
                    ax.annotate(f"{n} proj.", (x, y), textcoords="offset points",
                                xytext=(-6, 7) if above else (6, -11), ha="right" if above else "left",
                                fontsize=7.5, color=style.MUTED)
        ax.margins(y=0.2)
        ax.axvline(7, color=style.MUTED, lw=1, ls="--")
        ax.text(7.3, ax.get_ylim()[1], " 7 days (used)", va="top", fontsize=8.5, color=style.MUTED)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(THRESHOLDS, [str(t) for t in THRESHOLDS])
        ax.set_xlabel("Boundary between trunk-based and feature branching (days)")
        ax.set_title(ds, loc="left")
        style.grid(ax, "y")
    axes[0].set_ylabel("Builds that pass (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("On TravisTorrent, trunk-based passes more builds than feature branching for every boundary "
                 "from 1 to 14 days", x=0.02, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    style.save(fig, figure_path("threshold_sensitivity.png"))


if __name__ == "__main__":
    main()
