"""Chapter 4 (supplementary) · Do branching features survive when the model is pruned?

LightGBM is retrained on only its top-K features (K = 5, 10, 15, ...), ranked by its own
importance, to see where AUC-ROC plateaus and whether branching features stay in.

Not reported in the paper; supports the claim that branching features carry signal that
is not redundant with the standard metrics.
Output: results/4_what_matters/feature_selection.csv
Time  : ~10 min
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from sklearn.metrics import roc_auc_score

from bsci import travis, ghalogs
from bsci.models import lightgbm
from bsci.paths import results_dir


def prune(dataset, X_tr, X_te, y_tr, y_te, branching, ks):
    full = lightgbm().fit(X_tr, y_tr)
    order = pd.Series(full.feature_importances_, index=X_tr.columns).sort_values(ascending=False).index
    rows = [{"dataset": dataset, "top_k": len(order),
             "auc_roc": roc_auc_score(y_te, full.predict_proba(X_te)[:, 1]),
             "branching_kept": sum(f in branching for f in order)}]
    for k in [k for k in ks if k < len(order)]:
        cols = list(order[:k])
        m = lightgbm().fit(X_tr[cols], y_tr)
        rows.append({"dataset": dataset, "top_k": k, "auc_roc": roc_auc_score(y_te, m.predict_proba(X_te[cols])[:, 1]),
                     "branching_kept": sum(f in branching for f in cols),
                     "branching_names": ", ".join(f for f in cols if f in branching)})
        print(f"  {dataset} top-{k}: AUC={rows[-1]['auc_roc']:.4f}, branching kept {rows[-1]['branching_kept']}", flush=True)
    return rows


def main():
    rows = prune("TravisTorrent", *travis.split(), set(travis.BRANCHING), [5, 10, 15, 20, 25, 30, 35])
    rows += prune("GHALogs", *ghalogs.split(), set(ghalogs.BRANCHING), [5, 10, 15, 20, 25, 30])
    pd.DataFrame(rows).sort_values(["dataset", "top_k"]).to_csv(
        results_dir("4_what_matters") / "feature_selection.csv", index=False)


if __name__ == "__main__":
    main()
