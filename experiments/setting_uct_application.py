"""
UCT Application Simulation
Section 5 of "Demonstration Experiments" paper

Power curves and allocation paths for a stylized unconditional cash transfer
(UCT) experiment with 14 program variants: 8 core arms (4 targeting rules
× 2 transfer amounts) plus 6 straw-man arms (CBT, self-targeting, elderly
× 2 transfer amounts) calibrated per the updated UCT design doc §3.5.
Observations are shifted by the arm-specific cost-effectiveness threshold
before algorithms and test statistics are applied, so the standard null
H_0: max_g tilde_mu_g <= 0 maps directly to "no variant is cost-effective
at c = 0.60 dollars per dollar transferred."

The key structural feature is a multi-scale signal: the arm with the
largest *net* mean (Geographic-High, arm 3, tilde_mu = $10) is *not* the
arm with the best SNR (PMT-Low, arm 4, SNR = 0.50).

Algorithm comparison
--------------------
* SN-UCB    – self-normalised UCB; targets SNR → converges to PMT-Low
* UCB       – vanilla UCB1; targets mean with fixed exploration bonus
* UCB-V     – variance-aware UCB; scales bonus by sigma_hat
* Thompson  – Gaussian posterior approximation
* Uniform   – equal allocation paired with Bonferroni-corrected t-test

UCB and UCB-V both target the largest-mean arm (Geographic-High); UCB-V's
sigma-scaled bonus over-explores it, which is why it underperforms even
the variance-blind UCB1 in this calibrated setting.

Figures produced
----------------
  uct_power_curves.png      – power vs T for each (algorithm, statistic)
  uct_allocation_paths.png  – N_g(t)/t vs t for SN-UCB vs UCB-V
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats as sp_stats
from tqdm import tqdm
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup (mirrors other study files)
# ---------------------------------------------------------------------------
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from algorithms.baselines import (
    equal_allocation_algorithm,
    variance_aware_ucb_algorithm,
    standard_ucb_algorithm,
    thompson_sampling_algorithm,
)
from algorithms.sn_ucb import sn_ucb_algorithm

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "test_statistics",
    str(Path(__file__).parent.parent / "statistics" / "test_statistics.py"),
)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)
pooled_statistic_test = _ts.pooled_statistic_test
max_statistic_test = _ts.max_statistic_test


# ---------------------------------------------------------------------------
# UCT arm parameters (design doc Table, Section 3)
# ---------------------------------------------------------------------------
UCT_ARMS = {
    # Eight core arms (Sec. 3 of design doc) plus six straw-man arms
    # (Sec. 3.5) calibrated to specific UCT-targeting studies. The straw-man
    # arms have marginally positive net means (+$1 to +$3) and low SNR
    # (≤ 0.13), enriching the menu of plausible-but-statistically-weak
    # alternatives that an adaptive demonstration experiment must learn to
    # deallocate from. The Female-headed-HH arms in the design doc are
    # omitted because their calibration overlaps with the existing
    # Demographic (children-under-5) arms — both are "demographic category
    # correlated with poverty, intermediate gain rate ~0.65". PMT-Low
    # (idx 4) remains best-SNR; Geographic-High (idx 3) remains
    # largest-mean.
    "labels": [
        # core 8
        "Universal-Low", "Universal-High",
        "Geographic-Low", "Geographic-High",
        "PMT-Low", "PMT-High",
        "Demographic-Low", "Demographic-High",
        # straw-man 6
        "CBT-Low", "CBT-High",
        "Self-Low", "Self-High",
        "Elderly-Low", "Elderly-High",
    ],
    "gross_means": np.array([
        8.0, 20.0, 16.0, 40.0, 16.0, 36.0, 13.0, 32.0,
        13.0, 33.0, 14.0, 33.0, 13.0, 33.0,
    ]),
    "sigmas": np.array([
        20.0, 45.0, 16.0, 45.0,  8.0, 22.0, 14.0, 35.0,
        15.0, 32.0, 18.0, 40.0, 20.0, 42.0,
    ]),
    "transfers": np.array([
        20.0, 50.0, 20.0, 50.0, 20.0, 50.0, 20.0, 50.0,
        20.0, 50.0, 20.0, 50.0, 20.0, 50.0,
    ]),
    "thresholds": np.array([
        12.0, 30.0, 12.0, 30.0, 12.0, 30.0, 12.0, 30.0,
        12.0, 30.0, 12.0, 30.0, 12.0, 30.0,
    ]),
}
UCT_ARMS["net_means"] = UCT_ARMS["gross_means"] - UCT_ARMS["thresholds"]
UCT_ARMS["snrs"]      = UCT_ARMS["net_means"] / UCT_ARMS["sigmas"]

N_ARMS = len(UCT_ARMS["labels"])


# ---------------------------------------------------------------------------
# Study class
# ---------------------------------------------------------------------------
class UCTApplicationStudy:
    """
    Power curves and allocation paths for the UCT application simulation.

    Data-generating process
    -----------------------
    X_g(t) ~ N(mu_g, sigma_g^2)  (gross consumption gain)
    tilde_X_g(t) = X_g(t) - u_g  (shifted; mean = tilde_mu_g = mu_g - u_g)

    Shifting is applied at data-generation time so that algorithms and test
    statistics see tilde_X transparently — no code changes needed elsewhere.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.power_results: Optional[pd.DataFrame] = None
        self.alloc_results: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def generate_data(self, T: int) -> np.ndarray:
        """Return a (T, 8) array of shifted observations tilde_X_g(t)."""
        data = np.zeros((T, N_ARMS))
        for g in range(N_ARMS):
            data[:, g] = np.random.normal(
                loc=UCT_ARMS["net_means"][g],
                scale=UCT_ARMS["sigmas"][g],
                size=T,
            )
        return data

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------
    def _run_algorithm(self, algorithm: str, data: np.ndarray, T: int) -> np.ndarray:
        if algorithm == "sn_ucb":
            return sn_ucb_algorithm(data, T, N_ARMS, burn_in=2)
        elif algorithm == "ucb":
            return standard_ucb_algorithm(data, T, N_ARMS, c_param=2.0, burn_in=2)
        elif algorithm == "ucb_v":
            return variance_aware_ucb_algorithm(data, T, N_ARMS, c_param=2.0, burn_in=2)
        elif algorithm == "thompson":
            return thompson_sampling_algorithm(data, T, N_ARMS, burn_in=2)
        elif algorithm == "equal":
            return equal_allocation_algorithm(data, T, N_ARMS)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

    @staticmethod
    def _naive_bonferroni_test(
        data: np.ndarray, history: np.ndarray, alpha: float = 0.05
    ) -> bool:
        """
        One-sided Bonferroni-corrected t-tests for equal allocation.

        Per Section 5.1 of the paper, uniform allocation uses standard
        per-arm t-tests with a Bonferroni correction for multiplicity —
        no adjustment for adaptive allocation is required.  The critical
        value is z = Phi^{-1}(1 - alpha/k), corresponding to k one-sided
        tests at family-wise level alpha.
        """
        _, n_arms = data.shape
        z_crit = sp_stats.norm.ppf(1 - alpha / n_arms)

        for arm in range(n_arms):
            arm_mask = history[:, arm] == 1
            n_samples = int(np.sum(arm_mask))
            if n_samples <= 1:
                continue
            arm_data = data[arm_mask, arm]
            sample_mean = np.mean(arm_data)
            sample_std  = np.std(arm_data, ddof=1)
            if sample_std < 1e-10:
                continue
            t_stat = sample_mean * np.sqrt(n_samples) / sample_std
            if t_stat > z_crit:
                return True
        return False

    def run_single_experiment(
        self,
        T: int,
        algorithm: str,
        test_statistic: str,
        alpha: float = 0.05,
    ) -> bool:
        """Run one replication; return True if null is rejected."""
        data    = self.generate_data(T)
        history = self._run_algorithm(algorithm, data, T)

        # Equal allocation uses naive Bonferroni t-test (Section 5.1 of paper)
        if algorithm == "equal":
            return self._naive_bonferroni_test(data, history, alpha)

        try:
            if test_statistic == "pooled":
                _, _, reject = pooled_statistic_test(
                    data, history, alpha=alpha, method="threshold"
                )
            elif test_statistic == "max-linear":
                _, _, reject = max_statistic_test(
                    data, history, alpha=alpha, shape="linear", correction="bonferroni"
                )
            elif test_statistic == "max-log":
                _, _, reject = max_statistic_test(
                    data, history, alpha=alpha, shape="log", correction="bonferroni"
                )
            else:
                raise ValueError(f"Unknown test statistic: {test_statistic!r}")
        except Exception as exc:
            print(f"Warning: test failed ({test_statistic}): {exc}")
            reject = False

        return bool(reject)

    # ------------------------------------------------------------------
    # Power curves
    # ------------------------------------------------------------------
    def run_power_curves(
        self,
        T_values: Optional[List[int]] = None,
        algorithms: Optional[List[str]] = None,
        test_statistics: Optional[List[str]] = None,
        n_reps: int = 1000,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Estimate power at each (T, algorithm, test_statistic) cell.

        Returns DataFrame with columns [T, algorithm, test_statistic, power, se].
        """
        if T_values is None:
            T_values = [250, 500, 750, 1000]
        if algorithms is None:
            algorithms = ["sn_ucb", "ucb", "ucb_v", "thompson", "equal"]
        if test_statistics is None:
            test_statistics = ["pooled", "max-linear", "max-log"]

        n_cells = len(T_values) * len(algorithms) * len(test_statistics)
        if verbose:
            print("UCT Power Curve Analysis")
            print(f"  T values:        {T_values}")
            print(f"  Algorithms:      {algorithms}")
            print(f"  Test statistics: {test_statistics}")
            print(f"  Replications:    {n_reps}")
            print(f"  Total cells:     {n_cells}  ({n_cells * n_reps:,} experiments)")
            print()

        rows = []
        with tqdm(total=n_cells, desc="Cells", disable=not verbose) as pbar:
            for T in T_values:
                for alg in algorithms:
                    for ts in test_statistics:
                        pbar.set_description(f"T={T}, {alg}, {ts}")
                        rejects = [
                            self.run_single_experiment(T, alg, ts, alpha)
                            for _ in range(n_reps)
                        ]
                        power = float(np.mean(rejects))
                        rows.append({
                            "T": T,
                            "algorithm": alg,
                            "test_statistic": ts,
                            "n_reps": n_reps,
                            "power": power,
                            "se": float(np.sqrt(power * (1 - power) / n_reps)),
                        })
                        pbar.update(1)

        self.power_results = pd.DataFrame(rows)
        return self.power_results

    # ------------------------------------------------------------------
    # Allocation paths
    # ------------------------------------------------------------------
    def run_allocation_paths(
        self,
        T: int = 1000,
        n_reps: int = 1000,
        verbose: bool = True,
    ) -> Dict:
        """
        Average N_g(t)/t over replications for SN-UCB and UCB-V (var-aware).

        Returns dict with keys 'sn_ucb' and 'ucb_v', each a (T, N_ARMS)
        array of cumulative allocation fractions.
        """
        alg_names  = ["sn_ucb", "ucb_v"]
        cum_counts = {a: np.zeros((T, N_ARMS)) for a in alg_names}

        with tqdm(total=n_reps, desc="Allocation paths", disable=not verbose) as pbar:
            for _ in range(n_reps):
                data = self.generate_data(T)
                for alg in alg_names:
                    history = self._run_algorithm(alg, data, T)
                    cum_counts[alg] += np.cumsum(history, axis=0)
                pbar.update(1)

        t_index = np.arange(1, T + 1).reshape(-1, 1)
        self.alloc_results = {
            "T": T,
            "n_reps": n_reps,
            "fracs": {alg: cum_counts[alg] / n_reps / t_index for alg in alg_names},
        }
        return self.alloc_results

    # ------------------------------------------------------------------
    # Figure 1: power curves
    # ------------------------------------------------------------------
    def create_power_curve_figure(
        self,
        results_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Three-panel power-vs-T figure (one panel per test statistic).
        Y-axis is linear in [0, 1].
        """
        if results_df is None:
            results_df = self.power_results
        if results_df is None:
            raise ValueError("No power results. Run run_power_curves() first.")

        output_dir = _resolve_figures_dir(output_dir)

        # --- style -----------------------------------------------------
        test_stats_ordered = ["pooled", "max-linear", "max-log"]
        panel_titles = {
            "pooled":     "Pooled",
            "max-linear": "Max (linear)",
            "max-log":    "Max (log)",
        }
        alg_style = {
            "sn_ucb":   dict(label="SN-UCB",   color="#d62728", marker="s", lw=2.5),
            "ucb":      dict(label="UCB",      color="#1f77b4", marker="^", lw=2.5),
            "ucb_v":    dict(label="UCB-V",    color="#ff7f0e", marker="P", lw=2.5),
            "thompson": dict(label="Thompson", color="#2ca02c", marker="v", lw=2.5),
            "equal":    dict(label="Uniform",  color="#7f7f7f", marker="o", lw=2.5),
        }

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

        for ax, ts in zip(axes, test_stats_ordered):
            df_ts = results_df[results_df["test_statistic"] == ts]
            for alg, style in alg_style.items():
                df_alg = df_ts[df_ts["algorithm"] == alg].sort_values("T")
                if df_alg.empty:
                    continue
                T_vals = df_alg["T"].values
                power  = df_alg["power"].values
                se     = df_alg["se"].values
                ax.plot(T_vals, power, **style)
                ax.fill_between(T_vals,
                                np.clip(power - se, 0, 1),
                                np.clip(power + se, 0, 1),
                                alpha=0.15, color=style["color"])

            ax.axhline(0.05, color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_title(panel_titles[ts], fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([100, 200, 300, 400, 500])
            ax.tick_params(axis="x", labelsize=14)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(True, alpha=0.3, linestyle="--")

        axes[0].set_ylabel("Power", fontsize=14)
        fig.supxlabel("Horizon $T$", fontsize=14, y=0.02)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, -0.10))
        fig.suptitle(
            f"UCT Application: Power vs. Horizon with {N_ARMS} arms",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        filepath = output_dir / "uct_power_curves.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Figure 2: allocation paths
    # ------------------------------------------------------------------
    def create_allocation_path_figure(
        self,
        alloc_data: Optional[Dict] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Two-panel N_g(t)/t vs t: SN-UCB (left) and UCB var-aware (right).

        PMT-Low (best SNR) and Geographic-High (largest mean) are drawn
        thick and annotated; other arms are thin background lines.
        """
        if alloc_data is None:
            alloc_data = self.alloc_results
        if alloc_data is None:
            raise ValueError("No allocation data. Run run_allocation_paths() first.")

        output_dir = _resolve_figures_dir(output_dir)

        T      = alloc_data["T"]
        fracs  = alloc_data["fracs"]
        t_vals = np.arange(1, T + 1)

        arm_meta = [
            dict(label="Universal-Low",    color="#aec7e8", lw=1.2, ls="--", alpha=0.7),
            dict(label="Universal-High",   color="#aec7e8", lw=1.2, ls="-.", alpha=0.7),
            dict(label="Geographic-Low",   color="#1f77b4", lw=1.5, ls="--", alpha=0.8),
            dict(label="Geographic-High",  color="#1f77b4", lw=2.5, ls="-",  alpha=1.0),
            dict(label="PMT-Low",          color="#d62728", lw=2.5, ls="-",  alpha=1.0),
            dict(label="PMT-High",         color="#d62728", lw=1.5, ls="--", alpha=0.8),
            dict(label="Demographic-Low",  color="#2ca02c", lw=1.2, ls="--", alpha=0.7),
            dict(label="Demographic-High", color="#2ca02c", lw=1.2, ls="-.", alpha=0.7),
            # Straw-man arms (drawn as muted background lines).
            dict(label="CBT-Low",          color="#9467bd", lw=1.0, ls="--", alpha=0.5),
            dict(label="CBT-High",         color="#9467bd", lw=1.0, ls="-.", alpha=0.5),
            dict(label="Self-Low",         color="#8c564b", lw=1.0, ls="--", alpha=0.5),
            dict(label="Self-High",        color="#8c564b", lw=1.0, ls="-.", alpha=0.5),
            dict(label="Elderly-Low",      color="#e377c2", lw=1.0, ls="--", alpha=0.5),
            dict(label="Elderly-High",     color="#e377c2", lw=1.0, ls="-.", alpha=0.5),
        ]
        panel_titles = {
            "sn_ucb": "SN-UCB",
            "ucb_v":  "UCB-V (variance-aware)",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, alg in zip(axes, ["sn_ucb", "ucb_v"]):
            frac = fracs[alg]
            for g, meta in enumerate(arm_meta):
                ax.plot(t_vals, frac[:, g],
                        **{k: v for k, v in meta.items() if k != "label"},
                        label=meta["label"])
            ax.axhline(1 / N_ARMS, color="gray", linestyle=":", linewidth=1,
                       label="Uniform (1/k)")
            ax.set_title(panel_titles[alg], fontsize=12, fontweight="bold")
            ax.set_xlabel("Round $t$", fontsize=11)
            ax.set_xlim(1, T)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, linestyle="--")

        axes[0].set_ylabel("Fraction of samples $N_g(t)/t$", fontsize=11)

        for ax, alg in zip(axes, ["sn_ucb", "ucb_v"]):
            frac = fracs[alg]
            fp = frac[-1, 4]   # PMT-Low
            fg = frac[-1, 3]   # Geographic-High
            ax.annotate(
                f"PMT-Low\n(best SNR)\n{fp:.2f}",
                xy=(T, fp), xytext=(T * 0.70, fp + 0.08),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
            )
            ax.annotate(
                f"Geographic-High\n(largest net mean)\n{fg:.2f}",
                xy=(T, fg), xytext=(T * 0.55, fg - 0.12),
                fontsize=8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1),
            )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5,
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(
            f"UCT Application: Allocation Paths"
            f" (T={T:,}, averaged over {alloc_data['n_reps']:,} replications)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        filepath = output_dir / "uct_allocation_paths.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Top-level runner
    # ------------------------------------------------------------------
    def run_complete_study(
        self, quick_mode: bool = False, verbose: bool = True
    ) -> Dict:
        """Run the full study and save both figures."""
        if quick_mode:
            T_values, n_reps_power, T_alloc, n_reps_alloc = [50, 100], 200, 100, 200
            if verbose:
                print("=" * 60)
                print("QUICK MODE – reduced parameters for testing")
                print("=" * 60)
        else:
            T_values, n_reps_power, T_alloc, n_reps_alloc = list(range(50, 501, 50)), 1000, 500, 1000
            if verbose:
                print("=" * 60)
                print("FULL MODE – UCT application study")
                print("=" * 60)

        if verbose:
            print("\n[1/2] Power curves")
        power_df = self.run_power_curves(
            T_values=T_values,
            algorithms=["sn_ucb", "ucb", "ucb_v", "thompson", "equal"],
            test_statistics=["pooled", "max-linear", "max-log"],
            n_reps=n_reps_power,
            verbose=verbose,
        )

        results_dir = Path(__file__).parent.parent / "outputs" / "csvs"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "uct_power_results.csv"
        power_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\nSaved raw results: {csv_path}")

        if verbose:
            print("\n[2/2] Allocation paths")
        alloc_data = self.run_allocation_paths(T=T_alloc, n_reps=n_reps_alloc,
                                               verbose=verbose)

        if verbose:
            print("\nGenerating figures…")
        figures_dir = Path(__file__).parent.parent / "outputs" / "figures"
        fig1 = self.create_power_curve_figure(output_dir=str(figures_dir))
        fig2 = self.create_allocation_path_figure(output_dir=str(figures_dir))

        if verbose:
            print("\n" + "=" * 60)
            print("UCT STUDY COMPLETE")
            print("=" * 60)
            print(f"  Power results CSV : {csv_path}")
            print(f"  Power curve figure: {fig1}")
            print(f"  Alloc path figure : {fig2}")

        return {"power_results": power_df, "alloc_results": alloc_data,
                "figures": [fig1, fig2]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_figures_dir(output_dir: Optional[str]) -> Path:
    p = Path(output_dir) if output_dir else Path(__file__).parent.parent / "outputs" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UCT application study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer T values and replications")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("UCT Application Simulation")
    print("Unconditional Cash Transfer – 8 variants (targeting × transfer amount)")
    print()
    study = UCTApplicationStudy(seed=args.seed)
    study.run_complete_study(quick_mode=args.quick, verbose=True)
