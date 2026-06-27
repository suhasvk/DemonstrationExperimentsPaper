"""
Replay saved null runs against arbitrary test-parameter combinations.

This is the cheap "iterate on test settings" companion to
`save_null_runs.py`: load 1000 (data, history) pairs per (k, T) once,
then evaluate any test configuration (statistic, ζ, α, regularization
choice, …) against them in seconds.

Example
-------
Compute Type-I rates on the full grid for the current "auto" rule:

    from studies.replay_type_i import compute_type_i_grid

    df = compute_type_i_grid(
        runs_dir="outputs/null_runs",
        statistics=["pooled", "pooled-padded", "max-linear", "max-log"],
        alpha=0.05,
    )
    print(df.pivot(index="k", columns=["test_statistic", "T"], values="rate"))

Or sweep a parameter:

    for q0 in (15, 20, 25, 30):
        df = compute_type_i_grid(... , min_samples_q0=q0)
        ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Load test_statistics module by file path (mirrors how the studies do it).
_PKG_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "test_statistics", str(_PKG_ROOT / "statistics" / "test_statistics.py")
)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)


def _running_studentized_per_arm(data: np.ndarray, history: np.ndarray) -> np.ndarray:
    """
    For each (t, g), return Ẑ_g(N_g(t)) = ΣX / √SS computed using only
    samples assigned to arm g up to time t.

    Vectorized per arm: O(T·k) vs the naive O(T²·k) inner loop.

    Returns (T, k) array.
    """
    T_, k = data.shape
    mask = history.astype(np.float64)
    sampled_x  = data * mask
    sampled_x2 = (data ** 2) * mask

    cum_n  = np.cumsum(mask, axis=0)              # N_g(t)
    cum_s  = np.cumsum(sampled_x, axis=0)         # ΣX
    cum_s2 = np.cumsum(sampled_x2, axis=0)        # ΣX²

    n_safe = np.maximum(cum_n, 1.0)
    centered_ss = cum_s2 - (cum_s ** 2) / n_safe
    centered_ss = np.maximum(centered_ss, 1e-12)

    z = cum_s / np.sqrt(centered_ss)
    z[cum_n <= 1] = 0.0
    return z


def _running_pooled(data: np.ndarray, history: np.ndarray, *,
                    method: str, lambda_reg: float | None,
                    rho_reg: float = 2.0) -> float:
    """
    Pooled statistic at the final time T using either threshold or padding
    regularization. Mirrors test_statistics.pooled_statistic_test but
    operates on a single (data, history) pair without going through scipy.
    """
    T_, k = data.shape
    if method == "regularized":
        if lambda_reg is None:
            lambda_reg = float(np.sqrt(np.log(k * T_)))
    total = 0.0
    inv_root_T = 1.0 / np.sqrt(T_)
    for arm in range(k):
        mask = history[:, arm] == 1
        n = mask.sum()
        if n <= 1:
            continue
        x = data[mask, arm]
        mean = x.mean()
        ss = ((x - mean) ** 2).sum()
        sigma_hat = np.sqrt(ss / n)
        if method == "threshold":
            if n < rho_reg or sigma_hat < 1e-10:
                continue
            sigma = sigma_hat
        else:  # padding
            sigma = sigma_hat + lambda_reg / np.sqrt(n)
            if sigma < 1e-10:
                continue
        total += np.sum(x) / sigma * inv_root_T
    return total


def _load_cell(runs_dir: Path, k: int, T: int) -> dict:
    """Load saved arrays for cell (k, T)."""
    path = runs_dir / f"k{k}_T{T}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Saved null runs missing for (k={k}, T={T}): expected {path}. "
            f"Run save_null_runs.py first."
        )
    blob = np.load(path)
    return {
        "data":      blob["data"].astype(np.float64),     # (n_reps, T, k)
        "history":   blob["history"].astype(np.int32),    # (n_reps, T, k)
        "variances": blob["variances"],
        "n_reps":    int(blob["n_reps"]),
    }


def _resolve_zeta(rule, k: int, T: int) -> float:
    """Translate a ζ rule (numeric or string) to a float."""
    if isinstance(rule, (int, float)):
        return float(rule)
    if rule == "auto" or rule is None:
        return max(1.0, 25.0 * k / T)
    if isinstance(rule, str) and rule.startswith("q0:"):
        q0 = float(rule[3:])
        return max(1.0, q0 * k / T)
    if rule == "auto20":
        return max(1.0, 20.0 * k / T)
    if rule == "auto30":
        return max(1.0, 30.0 * k / T)
    if rule == "log_kT":
        return max(1.0, (np.log(k) + np.log(T) ** 2) * k / T)
    raise ValueError(f"Unknown ζ rule: {rule!r}")


def _evaluate_rep(
    data: np.ndarray, history: np.ndarray, *,
    statistic: str, alpha: float, zeta: float,
    pooled_method: str, correction: str,
) -> bool:
    """Apply one statistic to one replication. Vectorized; pure-numpy."""
    T_, k = data.shape

    if statistic in ("pooled", "pooled-padded"):
        method = "regularized" if statistic == "pooled-padded" else pooled_method
        stat = _running_pooled(data, history, method=method, lambda_reg=None)
        crit = float(_ts.stats.norm.ppf(1 - alpha))
        return bool(stat > crit)

    # max-statistic path: shared running Ẑ + boundary
    z_run = _running_studentized_per_arm(data, history)        # (T, k)
    n_run = np.cumsum(history, axis=0)                          # (T, k)
    min_samples = int(zeta * T_ / k)
    qual = n_run >= min_samples
    boundary_scale = n_run * k / T_  # uses T/k regardless of ζ

    if statistic == "max-linear":
        z_alpha = (_ts.solve_max_critical_value_linear(k, alpha) if correction == "bonferroni"
                   else _ts.solve_max_critical_value_linear_independent(k, alpha))
        boundary = z_alpha * np.sqrt(np.maximum(boundary_scale, 0.0))
        cross = (z_run > boundary) & qual
        return bool(cross.any())

    if statistic == "max-log":
        w_alpha = (_ts.solve_max_critical_value_log(k, alpha) if correction == "bonferroni"
                   else _ts.solve_max_critical_value_log_independent(k, alpha))
        h = lambda x: x ** 2 + 2 * np.log(_ts.stats.norm.cdf(x))
        boundary = h(w_alpha) + np.log(np.maximum(boundary_scale, 1e-12))
        cross = (h(z_run) > boundary) & qual
        return bool(cross.any())

    raise ValueError(f"Unknown statistic: {statistic!r}")


def evaluate_cell(
    cell: dict,
    *,
    statistic: str,
    alpha: float = 0.05,
    zeta_rule="auto",
    pooled_method: str = "threshold",
    correction: str = "bonferroni",
) -> tuple[float, float]:
    """Apply `statistic` to every saved replication and return (rate, se)."""
    n = cell["n_reps"]
    T_, k = cell["data"].shape[1:]
    zeta = _resolve_zeta(zeta_rule, k, T_)

    rejects = 0
    for rep in range(n):
        if _evaluate_rep(
            cell["data"][rep], cell["history"][rep],
            statistic=statistic, alpha=alpha, zeta=zeta,
            pooled_method=pooled_method, correction=correction,
        ):
            rejects += 1

    rate = rejects / n
    se = (rate * (1 - rate) / n) ** 0.5
    return rate, se


def compute_type_i_grid(
    runs_dir,
    *,
    ks: Sequence[int] = (5, 10, 20, 50),
    Ts: Sequence[int] = (200, 500, 1000, 2000),
    statistics: Iterable[str] = ("pooled", "max-linear", "max-log"),
    alpha: float = 0.05,
    zeta_rule="auto",
    pooled_method: str = "threshold",
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """
    Apply each `statistics` entry to every (k, T) cell. Returns a long
    DataFrame with columns [k, T, test_statistic, rate, se, n_reps, zeta].
    """
    runs_dir = Path(runs_dir)
    rows = []
    for k in ks:
        for T in Ts:
            cell = _load_cell(runs_dir, k, T)
            zeta = _resolve_zeta(zeta_rule, k, T)
            for stat in statistics:
                rate, se = evaluate_cell(
                    cell, statistic=stat, alpha=alpha, zeta_rule=zeta_rule,
                    pooled_method=pooled_method, correction=correction,
                )
                rows.append(dict(
                    k=k, T=T, test_statistic=stat,
                    rate=rate, se=se, n_reps=cell["n_reps"], zeta=zeta,
                ))
    return pd.DataFrame(rows)


def main():
    """CLI: print a Type-I grid for the current auto rule."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir", type=Path,
        default=_PKG_ROOT / "outputs" / "null_runs",
    )
    parser.add_argument("--zeta_rule", default="auto",
                        help="'auto' | 'auto20' | 'log_kT' | a float")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    # Try to interpret numeric rules.
    try:
        zeta_rule = float(args.zeta_rule)
    except ValueError:
        zeta_rule = args.zeta_rule

    statistics = ["pooled", "pooled-padded", "max-linear", "max-log"]
    df = compute_type_i_grid(
        args.runs_dir,
        statistics=statistics,
        alpha=args.alpha,
        zeta_rule=zeta_rule,
    )
    pivot = df.pivot(index="k", columns=["test_statistic", "T"], values="rate")
    print(f"Type-I rates (α={args.alpha}, ζ rule = {args.zeta_rule}):\n")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
