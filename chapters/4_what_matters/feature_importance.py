"""Chapter 4 · Which features does each classifier actually rely on?

Built-in importance scores cannot be compared across algorithms (impurity reduction,
split gain and prediction-value change measure different things), so we use held-out
permutation importance: the drop in test AUC-ROC when one feature is shuffled. It is
model-agnostic and has the same unit for every classifier. One-hot variables are
permuted as a single group. Each model ranks the features; the consensus ranking orders
them by mean rank, and Kendall's W measures how much the six models agree.

The same trained models and held-out sets as in chapters/3_prediction/performance.py.

Paper: Figures 2-3, Section V-D (feature-importance method) and Section VI-A/B.
Output: results/4_what_matters/importance_{travis,ghalogs}_{per_model,consensus}.csv
        figures/importance_travis.png, figures/importance_ghalogs.png
Time  : ~45 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from bsci import travis, ghalogs, style, SEED
from bsci.models import evaluation_models
from bsci.names import label
from bsci.paths import results_dir, figure_path

N_REPEATS, MAX_ROWS = 5, 100_000


def travis_groups(cols):
    strat = [c for c in cols if c.startswith("strategy_")]
    lang = [c for c in cols if c.startswith("lang_")]
    g = {c: [c] for c in cols if c not in strat + lang}
    g["strategy_label"], g["build_language"] = strat, lang
    return g, set(travis.BRANCHING) | {"strategy_label"}


def ghalogs_groups(cols):
    strat = [c for c in cols if c.startswith("strategy_")]
    event = [c for c in cols if c.startswith("event_")]
    g = {c: [c] for c in cols if c not in strat + event}
    g["strategy_label"], g["trigger_event"] = strat, event
    return g, set(ghalogs.BRANCHING) | {"strategy_label"}


def permutation_importance(model, X, y, groups, rng):
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    Xp, out = X.copy(), {}
    for g, cols in groups.items():
        orig, drops = X[cols].to_numpy(copy=True), []
        for _ in range(N_REPEATS):
            Xp[cols] = orig[rng.permutation(len(X))]      # rows permuted jointly for one-hot groups
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        Xp[cols] = orig
        out[g] = float(np.mean(drops))
    return out


def kendalls_w(ranks: np.ndarray):
    m, n = ranks.shape
    s = ((ranks.sum(0) - ranks.sum(0).mean()) ** 2).sum()
    w = 12 * s / (m ** 2 * (n ** 3 - n))
    return w, stats.chi2.sf(m * (n - 1) * w, n - 1)


def run(name, X_tr, X_te, y_tr, y_te, grouping):
    groups, branching = grouping(list(X_tr.columns))
    X_pi, _, y_pi, _ = (train_test_split(X_te, y_te, train_size=MAX_ROWS, stratify=y_te, random_state=SEED)
                        if len(X_te) > MAX_ROWS else (X_te, None, y_te, None))
    per_model = {}
    for m, model in evaluation_models().items():
        model.fit(X_tr, y_tr)
        per_model[m] = permutation_importance(model, X_pi, y_pi, groups, np.random.default_rng(SEED))
        print(f"  {name} {m} done", flush=True)
    pi = pd.DataFrame(per_model)
    ranks = pi.rank(ascending=False, method="average")
    w, p = kendalls_w(ranks.T.values)
    cons = pd.DataFrame({"feature": pi.index, "branching": pi.index.isin(branching),
                         "mean_rank": ranks.mean(1).values, "best_rank": ranks.min(1).values,
                         "worst_rank": ranks.max(1).values, "mean_auc_drop": pi.mean(1).values,
                         "models_ranking_top10": (ranks <= 10).sum(1).values}).sort_values("mean_rank")
    cons["consensus_rank"] = range(1, len(cons) + 1)
    cons.attrs.update(kendall_w=w, kendall_p=p)
    out = results_dir("4_what_matters")
    pi.rename_axis("feature").to_csv(out / f"importance_{name}_per_model.csv")
    cons.to_csv(out / f"importance_{name}_consensus.csv", index=False)
    pd.DataFrame([{"dataset": name, "kendall_w": w, "p_value": p, "n_features": len(cons)}]).to_csv(
        out / f"importance_{name}_agreement.csv", index=False)
    print(f"{name}: Kendall's W = {w:.2f} (p = {p:.1g})")
    print(cons.head(12).round(4).to_string(index=False))
    figure(name, cons, w)


def figure(name, cons, w):
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = cons.iloc[::-1].reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * n + 1.3))
    for i, r in d.iterrows():
        c = style.ORANGE if r.branching else style.BLUE
        ax.plot([r.best_rank, r.worst_rank], [i, i], color=c, lw=2.2, alpha=0.35, solid_capstyle="round")
        ax.plot(r.mean_rank, i, "D" if r.branching else "o", color=c, ms=6.5, mec="white", mew=0.8, zorder=3)
    ax.set_yticks(range(n), [label(f) for f in d.feature])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks([t for t in (1, 5, 10, 15, 20) if t <= n - 3] + [n])
    ax.set_xlabel("Rank across the six classifiers (1 = most important)")
    style.grid(ax)
    ax.spines["left"].set_visible(False)
    title = {"travis": "TravisTorrent", "ghalogs": "GHALogs"}[name]
    ax.set_title(f"{title}: what the six classifiers rely on  (Kendall's W = {w:.2f})", loc="left")
    ax.legend(handles=[Line2D([], [], marker="D", color=style.ORANGE, lw=2.2, mec="white", label="Branching feature"),
                       Line2D([], [], marker="o", color=style.BLUE, lw=2.2, mec="white", label="Standard feature")],
              loc="upper right", fontsize=9)
    style.save(fig, figure_path(f"importance_{name}.png"))


def main():
    run("travis", *travis.split(), travis_groups)
    run("ghalogs", *ghalogs.split(), ghalogs_groups)


if __name__ == "__main__":
    main()
