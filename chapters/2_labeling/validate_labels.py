"""Chapter 2 · How trustworthy are the automated labels?

Every project was also labelled by a human who inspected its branch-lifetime timeline
*without seeing the automated label* (the review tool reveals the rule's guess only after
the human has recorded a choice). Agreement between the two is the validation.

Paper: Section V-C ("Labeling validation"); 97.5% agreement, Cohen's kappa = 0.96.

Input : data/derived/travis_project_labels.csv
Output: results/2_labeling/label_agreement.csv, label_confusion.csv
        figures/labeling_agreement.png
Time  : seconds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from bsci import travis, style
from bsci.paths import results_dir, figure_path

ORDER = style.STRATEGY_ORDER


def main():
    lab = travis.read_labels()
    human, rule = lab["strategy_human"], lab["strategy_rule"]
    acc = (human == rule).mean()
    kappa = cohen_kappa_score(human, rule)
    p, r, f, n = precision_recall_fscore_support(human, rule, labels=ORDER, zero_division=0)
    per_class = pd.DataFrame({"strategy": ORDER, "projects": n, "precision": p, "recall": r, "f1": f}).round(3)
    summary = pd.DataFrame([{"projects": len(lab), "agreement": round(acc, 4), "cohen_kappa": round(kappa, 4),
                             "disagreements": int((human != rule).sum())}])
    out = results_dir("2_labeling")
    summary.to_csv(out / "label_agreement.csv", index=False)
    per_class.to_csv(out / "label_agreement_per_strategy.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(human, rule, labels=ORDER), index=ORDER, columns=ORDER)
    cm.to_csv(out / "label_confusion.csv")
    print(summary.to_string(index=False)); print(per_class.to_string(index=False))

    # Figure: row-normalised confusion matrix with counts
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    norm = cm.div(cm.sum(axis=1), axis=0)
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v, c = norm.values[i, j], cm.values[i, j]
            pct = f"{v:.0%}" if v >= 0.005 else "<1%"
            ax.text(j, i, f"{c}\n{pct}" if c else "·", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else style.INK)
    ax.set_xticks(range(4), [s.replace(" ", "\n") for s in ORDER])
    ax.set_yticks(range(4), ORDER)
    ax.set_xlabel("Automated rule-based label")
    ax.set_ylabel("Blind human label")
    ax.set_title(f"Automated vs. blind human labels ({len(lab)} projects)\n"
                 f"agreement {acc:.1%}, Cohen's κ = {kappa:.2f}", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    style.save(fig, figure_path("labeling_agreement.png"))


if __name__ == "__main__":
    main()
