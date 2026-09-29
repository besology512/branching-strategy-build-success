"""The six classifiers, defined once.

`evaluation_models()` is the configuration behind Tables 3-5, 9 and Figures 2-3.
`cv_models()` is the lighter configuration that was used for the five-fold
cross-validation and Friedman/Nemenyi tests behind Table 6 (kept separate so that
Table 6 is reproduced exactly as it was computed).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from . import SEED

MODEL_ORDER = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "AdaBoost", "GBDT"]


def evaluation_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, n_jobs=-1,
                                 random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=-1,
                                   random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, random_seed=SEED,
                                       verbose=0, allow_writing_files=False),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=SEED),
        "GBDT": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=SEED),
    }


def cv_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1,
                                 random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=63, n_jobs=-1,
                                   random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_seed=SEED,
                                       verbose=0, allow_writing_files=False),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED),
        "GBDT": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=SEED),
    }


def lightgbm() -> LGBMClassifier:
    """The best model, used alone for the ablation and feature-selection analyses."""
    return LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, n_jobs=-1,
                          random_state=SEED, verbose=-1)
