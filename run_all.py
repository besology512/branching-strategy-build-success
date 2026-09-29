"""Run the whole study, chapter by chapter.

    python run_all.py                  # everything (several hours on a laptop)
    python run_all.py --fast           # only the steps that take minutes
    python run_all.py --only 5 6       # selected chapters
    python run_all.py --list           # show the steps and expected run times

Each step is an ordinary script in chapters/ and can also be run on its own.
Output goes to results/ and figures/; logs to logs/.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    # chapter, script, minutes (approx., 8-core laptop), fast?
    ("2", "2_labeling/validate_labels.py", 0.1, True),
    ("2", "2_labeling/ghalogs_classify.py", 15, False),
    ("2", "2_labeling/travis_classify.py", 25, False),
    ("2", "2_labeling/travis_branch_lifecycle.py", 10, False),
    ("3", "3_prediction/build_travis_table.py", 35, False),
    ("3", "3_prediction/performance.py", 30, False),
    ("5", "5_strategy_and_success/success_rates.py", 3, True),
    ("5", "5_strategy_and_success/confounders.py", 20, True),
    ("6", "6_robustness/threshold_sensitivity.py", 3, True),
    ("4", "4_what_matters/feature_importance.py", 45, False),
    ("4", "4_what_matters/ablation.py", 10, False),
    ("4", "4_what_matters/feature_selection.py", 10, False),
    ("3", "3_prediction/significance.py", 90, False),
    ("6", "6_robustness/hyperparameters.py", 150, False),
]


def main(argv):
    if "--list" in argv:
        for ch, s, mins, fast in STEPS:
            print(f"  ch.{ch}  {s:<45} ~{mins:>5.0f} min{'   (fast)' if fast else ''}")
        return
    only = set(argv[argv.index("--only") + 1:]) if "--only" in argv else None
    steps = [s for s in STEPS if (only is None or s[0] in only) and ("--fast" not in argv or s[3])]
    (ROOT / "logs").mkdir(exist_ok=True)
    for ch, script, mins, _ in steps:
        log = ROOT / "logs" / (Path(script).stem + ".log")
        print(f"[ch.{ch}] {script}  (~{mins:.0f} min) ... ", end="", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run([sys.executable, "-u", str(ROOT / "chapters" / script)], stdout=f,
                                stderr=subprocess.STDOUT).returncode
        print(f"{'ok' if rc == 0 else 'FAILED'} in {(time.time() - t0) / 60:.1f} min  (log: {log.name})", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1:])
