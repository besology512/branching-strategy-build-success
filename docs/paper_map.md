# From the paper to the code

Every table, figure and quoted statistic in the paper, with the script that produces it
and the file that holds the numbers. "Exact" means the repository reproduces the
published value to the last reported digit.

| In the paper | What it reports | Script | Result file | Reproduces |
|---|---|---|---|---|
| §IV | dataset sizes, languages, periods | [`1_data/README.md`](../chapters/1_data/README.md) | — | exact |
| §V-C, lifecycle paragraph | 78.8 % / 10.9 % / 7.3 % / 2.9 % of builds by branch class | [`2_labeling/travis_branch_lifecycle.py`](../chapters/2_labeling/travis_branch_lifecycle.py) | `results/2_labeling/travis_branch_lifecycle.csv` | exact |
| §V-C, classification rules | strategy per project | [`2_labeling/travis_classify.py`](../chapters/2_labeling/travis_classify.py), [`2_labeling/ghalogs_classify.py`](../chapters/2_labeling/ghalogs_classify.py) | `data/derived/*` | exact (948/948, 28,658/28,658) |
| §V-C, labeling validation | 97.5 % agreement, κ = 0.96, per-strategy F1 | [`2_labeling/validate_labels.py`](../chapters/2_labeling/validate_labels.py) | `results/2_labeling/label_agreement*.csv` | exact |
| §V-C, seven-day threshold | 1-14 day sensitivity, 8.75-point gap at 1 day | [`6_robustness/threshold_sensitivity.py`](../chapters/6_robustness/threshold_sensitivity.py) | `results/6_robustness/threshold_sensitivity.csv` | exact |
| Table 3 | TravisTorrent: six models | [`3_prediction/performance.py`](../chapters/3_prediction/performance.py) | `results/3_prediction/performance.csv` | RF, GBDT, AdaBoost, LightGBM exact; XGBoost, CatBoost within 0.0011 AUC |
| Table 4 | GHALogs: six models | [`3_prediction/performance.py`](../chapters/3_prediction/performance.py) | `results/3_prediction/performance.csv` | RF, GBDT, AdaBoost, LightGBM exact; XGBoost, CatBoost within 0.001 AUC |
| Table 5 | hyperparameters | [`src/bsci/models.py`](../src/bsci/models.py); sensitivity in [`6_robustness/hyperparameters.py`](../chapters/6_robustness/hyperparameters.py) | `results/6_robustness/hyperparameters.csv` | settings as published; tuning curves match the original run (see chapter 6) |
| Table 6, Friedman / Nemenyi | five-fold CV AUC, significance | [`3_prediction/significance.py`](../chapters/3_prediction/significance.py) | `results/3_prediction/cv_*.csv`, `nemenyi_pvalues.csv` | Friedman, Nemenyi, RF, GBDT exact; others within 0.002 except AdaBoost (scikit-learn version, see chapter 3) |
| Figures 2-3 | consensus permutation-importance ranks, Kendall's W | [`4_what_matters/feature_importance.py`](../chapters/4_what_matters/feature_importance.py) | `results/4_what_matters/importance_*.csv` | exact (every mean rank; W = 0.91, 0.87) |
| Table 7 | strategy distribution | [`5_strategy_and_success/success_rates.py`](../chapters/5_strategy_and_success/success_rates.py) | `results/5_strategy_and_success/success_rates.csv` | exact |
| Table 8 | pass rate per strategy | [`5_strategy_and_success/success_rates.py`](../chapters/5_strategy_and_success/success_rates.py) | `results/5_strategy_and_success/success_rates.csv` | exact |
| Table 9 | ablation | [`4_what_matters/ablation.py`](../chapters/4_what_matters/ablation.py) | `results/4_what_matters/ablation.csv` | exact (published grouping); corrected grouping also reported |
| Table 10 | confounders: odds ratios, interaction, project level | [`5_strategy_and_success/confounders.py`](../chapters/5_strategy_and_success/confounders.py) | `results/5_strategy_and_success/confounders_*.csv` | exact |

Figure 1 (the framework diagram) is drawn, not computed; the pipeline is also shown in
the [README](../README.md#how-the-study-works).
