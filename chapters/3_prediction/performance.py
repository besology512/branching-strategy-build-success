"""Chapter 3 · How well do six classifiers predict whether a build passes?

Each dataset: stratified 80/20 split (TravisTorrent: from a 500,000-row sample of the
SMOTE-balanced table; GHALogs: all 573,993 runs), the six models of Table 5, and
accuracy, weighted precision/recall/F1, and AUC-ROC on the held-out 20 %.

Paper: Tables 3 (TravisTorrent), 4 (GHALogs) and 5 (hyperparameters).
Note : models run on CPU here. The published tables used GPU builds of XGBoost and
       CatBoost, so those two can differ in the third decimal.
Output: results/3_prediction/performance.csv, figures/model_performance.png
Time  : ~30 min
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from bsci import travis, ghalogs, style
from bsci.models import evaluation_models, MODEL_ORDER
from bsci.paths import results_dir, figure_path


def evaluate(dataset, X_tr, X_te, y_tr, y_te):
    rows = []
    for name, model in evaluation_models().items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        pred, prob = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        rows.append({"dataset": dataset, "model": name,
                     "accuracy": accuracy_score(y_te, pred),
                     "precision": precision_score(y_te, pred, average="weighted", zero_division=0),
                     "recall": recall_score(y_te, pred, average="weighted", zero_division=0),
                     "f1": f1_score(y_te, pred, average="weighted", zero_division=0),
                     "auc_roc": roc_auc_score(y_te, prob)})
        print(f"  {dataset:<13} {name:<13} AUC={rows[-1]['auc_roc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def figure(df):
    style.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ds in zip(axes, ["TravisTorrent", "GHALogs"]):
        d = df[df.dataset == ds].set_index("model").loc[MODEL_ORDER[::-1]]
        best = d["auc_roc"].idxmax()
        for i, (m, r) in enumerate(d.iterrows()):
            c = style.BLUE if m == best else style.MUTED
            ax.plot([0.5, r.auc_roc], [i, i], color=style.GRID, lw=1.2, zorder=1)
            ax.plot(r.auc_roc, i, "o", color=c, ms=7, zorder=2)
            ax.text(r.auc_roc + 0.014, i, f"{r.auc_roc:.3f}", va="center", fontsize=9,
                    color=style.INK, fontweight="semibold" if m == best else "normal")
        ax.set_yticks(range(len(d)), d.index)
        ax.set_xlim(0.5, 1.0)
        ax.set_title(ds, loc="left")
        ax.set_xlabel("AUC-ROC on held-out builds (0.5 = chance)")
        style.grid(ax)
    fig.suptitle("Predicting build success: LightGBM is best on both datasets", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    style.save(fig, figure_path("model_performance.png"))


def main():
    rows = evaluate("TravisTorrent", *travis.split())
    rows += evaluate("GHALogs", *ghalogs.split())
    df = pd.DataFrame(rows)
    df.to_csv(results_dir("3_prediction") / "performance.csv", index=False)
    print(df.round(4).to_string(index=False))
    figure(df)


if __name__ == "__main__":
    main()
