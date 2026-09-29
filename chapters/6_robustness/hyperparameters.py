"""Chapter 6 (supplementary) · Why these hyperparameters? (Table 5)

On the TravisTorrent split used for Table 3:
  A  AUC-ROC vs number of trees (50-300) for all six models
  B  AdaBoost learning rate (0.1-1.0)
  C  training-sample size (100k-500k) for LightGBM
  D  Random Forest maximum depth (5-20, unlimited)

Caveat: these curves are measured on the same held-out split that Table 3 reports (no
separate validation split), and the balanced table contains SMOTE-generated builds on
both sides of the split; unlimited-depth Random Forests exploit that (see
docs/limitations.md), which is why the depth was capped at 15.

Not reported in the paper; documents the choices in Table 5.
Output: results/6_robustness/hyperparameters.csv, figures/hyperparameters.png
Time  : ~2-3 h
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from bsci import travis, style
from bsci.models import evaluation_models, MODEL_ORDER
from bsci.paths import results_dir, figure_path

OUT = results_dir("6_robustness") / "hyperparameters.csv"
TREES_PARAM = {"Random Forest": "n_estimators", "XGBoost": "n_estimators", "LightGBM": "n_estimators",
               "CatBoost": "iterations", "AdaBoost": "n_estimators", "GBDT": "max_iter"}


def auc(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    return roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])


def main():
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["experiment", "model", "value", "auc_roc"])
    seen = set(zip(done.experiment, done.model, done.value.astype(str)))
    table = travis.model_table()
    X_tr, X_te, y_tr, y_te = travis.split(table=table)
    base = evaluation_models()

    def record(exp, model, value, fn):
        if (exp, model, str(value)) in seen:
            return
        a = fn()
        pd.DataFrame([{"experiment": exp, "model": model, "value": value, "auc_roc": a}]).to_csv(
            OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {exp:<16} {model:<13} {value!s:>8}  AUC={a:.4f}", flush=True)

    for name, est in base.items():
        for n in [50, 100, 150, 200, 250, 300]:
            record("trees", name, n, lambda: auc(clone(est).set_params(**{TREES_PARAM[name]: n}), X_tr, X_te, y_tr, y_te))
    for lr in [0.1, 0.3, 0.5, 0.7, 1.0]:
        record("adaboost_lr", "AdaBoost", lr, lambda: auc(clone(base["AdaBoost"]).set_params(learning_rate=lr),
                                                            X_tr, X_te, y_tr, y_te))
    for n in [100_000, 200_000, 300_000, 400_000, 500_000]:
        record("sample_size", "LightGBM", n, lambda: auc(clone(base["LightGBM"]), *travis.split(sample_size=n, table=table)))
    for d in [5, 8, 10, 12, 15, 20, None]:
        record("rf_depth", "Random Forest", "unlimited" if d is None else d,
               lambda: auc(clone(base["Random Forest"]).set_params(max_depth=d), X_tr, X_te, y_tr, y_te))
    figure(pd.read_csv(OUT))


def figure(df):
    style.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    t = df[df.experiment == "trees"].copy()
    t["value"] = t["value"].astype(int)
    colors = dict(zip(MODEL_ORDER, [style.MAGENTA, style.INK, style.YELLOW, style.ORANGE, style.BLUE, style.AQUA]))
    ends = []
    for m in MODEL_ORDER:
        g = t[t.model == m].sort_values("value")
        axes[0].plot(g.value, g.auc_roc, marker="o", ms=4, lw=1.6, color=colors[m])
        ends.append([g.auc_roc.iloc[-1], m])
    ends.sort()                                  # spread end labels so they never overlap
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.009)
    for y, m in ends:
        axes[0].text(308, y, m, va="center", fontsize=8, color=style.INK)
    axes[0].axvline(200, color=style.MUTED, ls="--", lw=1)
    axes[0].set(title="Number of trees (200 used)", xlabel="Trees", ylabel="AUC-ROC", xlim=(40, 390))
    s = df[df.experiment == "sample_size"].copy()
    s["value"] = s["value"].astype(int) / 1000
    axes[1].plot(s.value, s.auc_roc, marker="o", color=style.BLUE, lw=1.8)
    axes[1].plot(500, s.auc_roc.iloc[-1], "o", ms=9, mfc="none", mec=style.ORANGE, mew=2)
    axes[1].text(0.97, 0.04, "narrow scale: all within 0.012", transform=axes[1].transAxes, ha="right",
                 fontsize=8, color=style.MUTED)
    axes[1].set(title="LightGBM training sample (500k used)", xlabel="Rows (thousands)")
    d = df[df.experiment == "rf_depth"].reset_index(drop=True)
    axes[2].plot(range(len(d)), d.auc_roc, marker="o", color=style.BLUE, lw=1.8)
    used = d.index[d.value.astype(str) == "15"][0]
    axes[2].plot(used, d.auc_roc[used], "o", ms=9, mfc="none", mec=style.ORANGE, mew=2)
    axes[2].set_xticks(range(len(d)), d.value.astype(str))
    axes[2].set(title="Random Forest depth (15 used)", xlabel="Maximum depth")
    for ax in axes:
        style.grid(ax, "y")
    fig.tight_layout()
    style.save(fig, figure_path("hyperparameters.png"))


if __name__ == "__main__":
    main()
