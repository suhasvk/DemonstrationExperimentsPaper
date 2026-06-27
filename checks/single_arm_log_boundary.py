"""
Single-arm sanity check for the delayed-start log boundary.

Lemma 1 (Robbins; paper §3.2) states:

    lim_{q0 → ∞} P{ max_{q ≥ q0}  S_q > sqrt(q) h^{-1}(log(q/q0) + h(a)) } = Ψ^+(a)

with h(x) = x^2 + 2 log Φ(x) and Ψ^+(a) = 1 - Φ(a) + φ(a)[a + φ(a)/Φ(a)].

This script Monte-Carlos a *single* Gaussian random walk, applies the
boundary, and checks whether the empirical crossing rate matches the
nominal α = Ψ^+(w_α). It also compares the studentized-sum form
Ẑ(q) = ΣX/√SS used in the paper's max test against the
"true" Brownian form S_q/√q, since they differ at finite q.

Crossing rate should converge to α as q0 → ∞.

Run:
    python checks/single_arm_log_boundary.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def h(x: np.ndarray) -> np.ndarray:
    """h(x) = x^2 + 2 log Φ(x)."""
    return x ** 2 + 2.0 * np.log(stats.norm.cdf(x))


def h_inv_positive(y: float) -> float:
    """Numerical inverse of h on x ≥ 0 (h is strictly increasing there)."""
    if y <= h(0.0):
        return 0.0
    # h(x) ~ x^2 - 2/x · φ(x)/... for large x; certainly h(10) > 100, fine
    return brentq(lambda x: h(x) - y, 0.0, 50.0)


def psi_plus(a: float) -> float:
    """Ψ^+(a) = 1 - Φ(a) + φ(a)·[a + φ(a)/Φ(a)]."""
    Phi  = stats.norm.cdf(a)
    phi  = stats.norm.pdf(a)
    return 1 - Phi + phi * (a + phi / Phi)


def solve_w(alpha: float) -> float:
    """w_α such that Ψ^+(w_α) = α."""
    # Ψ^+(0) = 1 - 0.5 + φ(0)·(0 + φ(0)/0.5) ≈ 0.5 + 2φ(0)^2 ≈ 0.818
    # Ψ^+(∞) → 0, so root exists for α ∈ (0, ~0.818)
    return brentq(lambda w: psi_plus(w) - alpha, -3.0, 10.0)


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def _check_replication(rng: np.random.Generator, T: int, q0: int,
                       w_alpha: float, mode: str) -> bool:
    """
    Returns True if the boundary is crossed at any q ∈ [q0, T].

    mode:
      'brownian'    — compare S_q/√q to h^{-1}(log(q/q0) + h(w))
      'studentized' — compare Ẑ(q) = ΣX/√SS to the same boundary
    """
    X = rng.standard_normal(T)
    cumX = np.cumsum(X)

    # boundary at each q ∈ [q0, T]
    qs = np.arange(q0, T + 1)
    log_term = np.log(qs / q0)
    # Vectorize h_inv via brentq is slow; precompute on a grid then interp.
    # Range of log_term: [0, log(T/q0)]. h(w_alpha) is fixed.
    # Total argument range: [h(w_alpha), h(w_alpha) + log(T/q0)].
    arg_lo = h(np.array([w_alpha]))[0]
    arg_hi = arg_lo + log_term[-1] + 1e-9
    grid_x = np.linspace(0, max(50.0, np.sqrt(arg_hi) + 5.0), 2000)
    grid_h = h(grid_x)
    boundary = np.interp(arg_lo + log_term, grid_h, grid_x)

    # statistic at each q
    if mode == "brownian":
        stat = cumX[q0 - 1:T] / np.sqrt(qs)
    else:  # studentized
        # Ẑ(q) = sum X / sqrt(SS_q),   SS_q = Σ (X_i - X̄_q)^2 = ΣX_i^2 - q X̄_q^2
        cumX2 = np.cumsum(X ** 2)
        # at index q-1 we have q observations; mean = cumX[q-1]/q
        means_q = cumX[q0 - 1:T] / qs
        ss_q = cumX2[q0 - 1:T] - qs * means_q ** 2
        stat = cumX[q0 - 1:T] / np.sqrt(np.maximum(ss_q, 1e-12))

    return bool(np.any(stat > boundary))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    seed   = 42
    n_reps = 5000

    # ------------------------------------------------------------------
    # Block A — α = 0.05; sweep ratio T/q0 to diagnose finite-horizon truncation.
    # ------------------------------------------------------------------
    alpha = 0.05
    w_alpha = solve_w(alpha)
    print(f"α = {alpha},   w_α = {w_alpha:.4f},   Ψ^+(w_α) = {psi_plus(w_alpha):.4f}")
    print(f"\n{'q0':>6} {'T':>9} {'T/q0':>6} {'mode':>12} {'rate':>7} {'se':>7}")
    print("-" * 55)

    for q0 in (100, 500):
        for ratio in (5, 20, 100, 500):
            T = q0 * ratio
            for mode in ("brownian", "studentized"):
                rng = np.random.default_rng(seed)
                crossings = sum(
                    _check_replication(rng, T, q0, w_alpha, mode)
                    for _ in range(n_reps)
                )
                rate = crossings / n_reps
                se   = (rate * (1 - rate) / n_reps) ** 0.5
                print(f"{q0:>6} {T:>9} {ratio:>6} {mode:>12} {rate:>7.4f} {se:>7.4f}")
            print()

    # ------------------------------------------------------------------
    # Block B — Bonferroni-stringent levels (α/k for k=10, 50) at small q0.
    # This mirrors what the multi-arm max-log test asks of the per-arm boundary.
    # ------------------------------------------------------------------
    print("\n=== Stringent α (per-arm Bonferroni levels) ===")
    print(f"{'α':>8} {'w_α':>7} {'q0':>5} {'T':>9} {'T/q0':>6} {'mode':>12} {'rate':>7} {'se':>7}")
    print("-" * 70)

    for alpha in (0.005, 0.001):  # α/k for k=10 and k=50 at level 0.05
        w_alpha = solve_w(alpha)
        for q0, T in [(8, 200), (40, 1000), (40, 10_000), (40, 100_000)]:
            for mode in ("brownian", "studentized"):
                rng = np.random.default_rng(seed)
                crossings = sum(
                    _check_replication(rng, T, q0, w_alpha, mode)
                    for _ in range(n_reps)
                )
                rate = crossings / n_reps
                se   = (rate * (1 - rate) / n_reps) ** 0.5
                ratio = T // q0
                print(f"{alpha:>8.4f} {w_alpha:>7.4f} {q0:>5} {T:>9} {ratio:>6} "
                      f"{mode:>12} {rate:>7.4f} {se:>7.4f}")
            print()


if __name__ == "__main__":
    main()
