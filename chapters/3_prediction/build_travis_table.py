"""Chapter 3 · Build the TravisTorrent modelling table from the raw release.

Joins every build to its project's strategy label, encodes the 22 features, and balances
the four strategies with SMOTE (random_state=42). The result (~9.9 M rows, 43 features)
is cached as data/cache/travis_model_table_smote.parquet and reused by chapters 3-6.

Paper: Section V ("Evaluation metrics") and the captions of Tables 3 and 9.
Input : raw TravisTorrent release + data/derived/travis_project_labels.csv
Output: data/cache/travis_model_table_smote.parquet
Time  : ~20-40 min, ~16 GB RAM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bsci import travis
from bsci.paths import TRAVIS_MODEL_TABLE


def main():
    if TRAVIS_MODEL_TABLE.exists():
        print(f"already built: {TRAVIS_MODEL_TABLE}")
    t = travis.model_table()
    print(f"{len(t):,} rows x {t.shape[1] - 2} features")
    print(t["human_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
