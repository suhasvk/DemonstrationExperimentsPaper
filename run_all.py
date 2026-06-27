"""
Run all four simulation studies and regenerate the paper figures/table.

Estimated total runtime: 3–4 hours (1000 reps per cell, full grids).
Pass --quick for a ~2-minute pipeline check (smaller grids, 100 reps).

Usage
-----
    python run_all.py            # full paper run
    python run_all.py --quick    # quick sanity-check
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPRODUCTION_ROOT = Path(__file__).parent

STUDIES = [
    ("Type-I error grid",  "studies/type_i_error_heatmap.py"),
    ("Spike-model power",  "studies/spike_model_power.py"),
    ("Multi-scale power",  "studies/multiscale_power.py"),
    ("UCT application",    "experiments/setting_uct_application.py"),
]

COMBINERS = [
    ("Combine paper figures", "make_paper_figures.py"),
    ("Render Type-I table",   "make_type1_table.py"),
]


def _run(label: str, script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(REPRODUCTION_ROOT / script), *args]
    print(f"\n{'=' * 70}")
    print(f"[{time.strftime('%H:%M:%S')}] {label}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=REPRODUCTION_ROOT)
    print(f"  done in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced grids/reps (~1–2 min total) for pipeline checks.",
    )
    args = parser.parse_args()
    extra_args = ["--quick"] if args.quick else []

    overall_t0 = time.time()
    for label, script in STUDIES:
        _run(label, script, extra_args)

    for label, script in COMBINERS:
        _run(label, script, [])

    elapsed = time.time() - overall_t0
    print(f"\n{'=' * 70}")
    print(f"All studies + figures complete in {elapsed/60:.1f} min.")
    print(f"Outputs:")
    print(f"  CSVs    : {REPRODUCTION_ROOT / 'outputs' / 'csvs'}")
    print(f"  Figures : {REPRODUCTION_ROOT / 'outputs' / 'figures'}")
    print(f"  Tables  : {REPRODUCTION_ROOT / 'outputs' / 'tables'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
