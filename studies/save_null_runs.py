"""
Precompute and save (data, history) pairs from null-hypothesis runs.

For each cell (k, T) on the Type-I grid we generate `n_reps` runs:
  • inverse-gamma per-arm variances (matches type_i_error_heatmap.py),
  • Gaussian observations with mean 0,
  • UCB1 allocation history.

The saved tuples can be replayed against any test parameter combination
without re-running the (slow) UCB step. See `replay_type_i.py`.

Run:
    python studies/save_null_runs.py            # full grid, 1000 reps each
    python studies/save_null_runs.py --quick    # smaller grid, 100 reps each

Outputs land in `outputs/null_runs/k{k}_T{T}.npz`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Path setup so we can import baselines + the type-I study's variance helper.
PKG_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PKG_ROOT))

from algorithms.baselines import standard_ucb_algorithm  # noqa: E402
from studies.type_i_error_heatmap import TypeIErrorHeatmapStudy  # noqa: E402


def generate_one_run(rng: np.random.Generator, k: int, T: int,
                     study: TypeIErrorHeatmapStudy):
    """One null replication: variances → data → UCB history."""
    # The IG sampler in TypeIErrorHeatmapStudy uses np.random's global state,
    # so we temporarily seed it from `rng` to keep things deterministic.
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)
    variances = study.generate_inverse_gamma_variances(k)

    data = np.zeros((T, k), dtype=np.float32)
    for g in range(k):
        data[:, g] = np.random.normal(0.0, np.sqrt(variances[g]), T)
    history = standard_ucb_algorithm(
        data.astype(np.float64), T, k, c_param=2.0, burn_in=2
    ).astype(np.int8)
    return data, history, variances.astype(np.float32)


def save_cell(out_dir: Path, k: int, T: int, n_reps: int, seed: int):
    rng = np.random.default_rng(seed)
    study = TypeIErrorHeatmapStudy()

    all_data    = np.empty((n_reps, T, k), dtype=np.float32)
    all_history = np.empty((n_reps, T, k), dtype=np.int8)
    all_var     = np.empty((n_reps, k),    dtype=np.float32)

    desc = f"k={k}, T={T}, n_reps={n_reps}"
    for rep in tqdm(range(n_reps), desc=desc, leave=False):
        data, history, variances = generate_one_run(rng, k, T, study)
        all_data[rep]    = data
        all_history[rep] = history
        all_var[rep]     = variances

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"k{k}_T{T}.npz"
    np.savez_compressed(
        out_path,
        data=all_data,
        history=all_history,
        variances=all_var,
        k=k, T=T, n_reps=n_reps, seed=seed,
        algorithm="standard_ucb_algorithm",
        c_param=2.0, burn_in=2,
    )
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  saved {out_path.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid (k∈{5,10}, T∈{200,500}) and 100 reps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", type=Path,
        default=PKG_ROOT / "outputs" / "null_runs",
    )
    args = parser.parse_args()

    if args.quick:
        ks = [5, 10]
        Ts = [200, 500]
        n_reps = 100
    else:
        ks = [5, 10, 20, 50]
        Ts = [200, 500, 1000, 2000]
        n_reps = 1000

    print(f"Saving null runs: k×T grid {ks} × {Ts}, n_reps={n_reps}")
    print(f"  Output dir: {args.out_dir}")
    print()

    for k in ks:
        for T in Ts:
            save_cell(args.out_dir, k, T, n_reps,
                      seed=args.seed * 1000 + k * 10000 + T)

    print("\nDone.")


if __name__ == "__main__":
    main()
