"""Chapter 4 · Do branching features add anything on top of standard build metrics?

LightGBM is trained on three feature sets and evaluated on the same held-out builds:
  standard only   the build, test, change and repository metrics every CI study uses
  branching only  the engineered branch-structure features (+ strategy label)
  all combined    both

Two groupings are reported for TravisTorrent:
  published  — as in Table 9, where the build-language indicators were (because of a
               column-naming slip) grouped with the branching block rather than standard;
  corrected  — language counted as a standard feature. This is the fairer baseline.
The corrected grouping keeps the conclusion: branching features add AUC-ROC on top of
every standard feature.

Paper: Table 9.
Output: results/4_what_matters/ablation.csv, figures/ablation.png
Time  : ~10 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from bsci import travis, ghalogs, style
from bsci.models import lightgbm
from bsci.paths import results_dir, figure_path


def evaluate(dataset, grouping, feature_set, cols, X_tr, X_te, y_tr, y_te):
    m = lightgbm().fit(X_tr[cols], y_tr)
    prob = m.predict_proba(X_te[cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    r = {"dataset": dataset, "grouping": grouping, "feature_set": feature_set, "n_features": len(cols),
         "accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred, average="weighted"),
         "auc_roc": roc_auc_score(y_te, prob)}
    print(f"  {dataset:<13} {grouping:<9} {feature_set:<15} AUC={r['auc_roc']:.4f}", flush=True)
    return r


def main():
    rows = []
    X_tr, X_te, y_tr, y_te = travis.split()
    cols = list(X_tr.columns)
    lang = [c for c in cols if c.startswith("lang_")]
    strat = [c for c in cols if c.startswith("strategy_")]
    std = [c for c in cols if c.startswith(("tr_", "gh_", "git_"))]
    br = travis.BRANCHING
    for grouping, standard, branching in [("published", std, br + strat + lang),
                                          ("corrected", std + lang, br + strat)]:
        for fs, c in [("standard only", standard), ("branching only", branching), ("all combined", cols)]:
            if grouping == "published" and fs == "branching only":
                c = br          # Table 9 used the three engineered features alone
            rows.append(evaluate("TravisTorrent", grouping, fs, c, X_tr, X_te, y_tr, y_te))

    X_tr, X_te, y_tr, y_te = ghalogs.split()
    cols = list(X_tr.columns)
    std = ghalogs.STANDARD_BASE + [c for c in cols if c.startswith("event_")]
    brg = ghalogs.BRANCHING
    for fs, c in [("standard only", std), ("branching only", brg), ("all combined", cols)]:
        rows.append(evaluate("GHALogs", "published", fs, c, X_tr, X_te, y_tr, y_te))

    df = pd.DataFrame(rows)
    df.to_csv(results_dir("4_what_matters") / "ablation.csv", index=False)
    figure(df)


def figure(df):
    style.use()
    import matplotlib.pyplot as plt
    panels = [("TravisTorrent", "corrected", "TravisTorrent"), ("GHALogs", "published", "GHALogs")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    colors = {"standard only": style.BLUE, "branching only": style.ORANGE, "all combined": style.INK}
    for ax, (ds, grouping, title) in zip(axes, panels):
        d = df[(df.dataset == ds) & (df.grouping == grouping)].set_index("feature_set")
        order = ["branching only", "standard only", "all combined"]
        for i, fs in enumerate(order):
            v = d.loc[fs, "auc_roc"]
            ax.barh(i, v - 0.5, left=0.5, color=colors[fs], height=0.55)
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_yticks(range(3), ["Branching only", "Standard only", "Standard + branching"])
        ax.set_xlim(0.5, 1.0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUC-ROC (LightGBM, held-out builds)")
        style.grid(ax)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Branching features are weak alone but add signal on top of standard metrics",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    style.save(fig, figure_path("ablation.png"))


if __name__ == "__main__":
    main()
