"""Chapter 3 · Are the differences between the six classifiers real?

Five-fold stratified cross-validation (TravisTorrent: 300,000-row sample of the balanced
table; GHALogs: all runs), a Friedman test on the fold AUC-ROC scores, and Nemenyi
post-hoc comparisons (Demsar, 2006).

This uses the lighter model configuration that produced Table 6 (100 trees; Random
Forest depth 12; AdaBoost learning rate 0.5; see src/bsci/models.py: cv_models).

Paper: Table 6 and the significance paragraph of Section VI-A.
Output: results/3_prediction/cv_fold_auc.csv, cv_summary.csv, nemenyi_pvalues.csv
Time  : ~1-2 h
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bsci import travis, ghalogs, SEED
from bsci.models import cv_models, MODEL_ORDER
from bsci.paths import results_dir


def cross_validate(dataset, X, y):
    rows = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), 1):
        for name, model in cv_models().items():
            model.fit(X.iloc[tr], y.iloc[tr])
            rows.append({"dataset": dataset, "fold": fold, "model": name,
                         "auc_roc": roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1])})
        print(f"  {dataset} fold {fold} done", flush=True)
    return rows


def main():
    t = travis.model_table()
    X = t[[c for c in t.columns if c not in (travis.TARGET, "human_label")]]
    X_t, _, y_t, _ = train_test_split(X, t[travis.TARGET], train_size=300_000, stratify=t[travis.TARGET], random_state=SEED)
    X_g, y_g = ghalogs.model_data()

    out = results_dir("3_prediction")
    folds = pd.DataFrame(cross_validate("TravisTorrent", X_t, y_t) + cross_validate("GHALogs", X_g, y_g))
    folds.to_csv(out / "cv_fold_auc.csv", index=False)

    summary, nemenyi = [], []
    for ds, d in folds.groupby("dataset", sort=False):
        wide = d.pivot(index="fold", columns="model", values="auc_roc")[MODEL_ORDER]
        stat, p = friedmanchisquare(*[wide[m] for m in MODEL_ORDER])
        for m in MODEL_ORDER:
            summary.append({"dataset": ds, "model": m, "mean_auc": wide[m].mean(), "std_auc": wide[m].std(ddof=0),
                            "friedman_chi2": stat, "friedman_p": p})
        pv = sp.posthoc_nemenyi_friedman(wide.values)
        pv.index = pv.columns = MODEL_ORDER
        nemenyi.append(pv.stack().rename("p_value").reset_index().rename(
            columns={"level_0": "model_a", "level_1": "model_b"}).assign(dataset=ds))
        print(f"{ds}: Friedman chi2 = {stat:.1f}, p = {p:.2g}")
    pd.DataFrame(summary).to_csv(out / "cv_summary.csv", index=False)
    pd.concat(nemenyi).to_csv(out / "nemenyi_pvalues.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
