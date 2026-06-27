"""
Multi-Scale Power Study

Power analysis with heteroscedastic variances and varying mean effects.
This study examines performance when both means and variances scale with arm index:
- Means: μ_g = δ·g
- Variances: σ²_g = 2·g³

The factor-of-2 in the variance widens UCB1's exploration loss (its bonus
is variance-blind) and weakens uniform allocation's per-arm t-tests, so
SN-UCB's signal-to-noise targeting shows up more cleanly on the pooled
statistic. The asymmetry between best mean (g=k) and best SNR (g=1) is
preserved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
from typing import Dict, List, Optional

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from algorithms.baselines import (
    standard_ucb_algorithm,
    thompson_sampling_algorithm,
    equal_allocation_algorithm,
    oracle_allocation_algorithm,
    variance_aware_ucb_algorithm,
)
from algorithms.sn_ucb import sn_ucb_algorithm

# Import test statistics
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_statistics",
    str(Path(__file__).parent.parent / "statistics" / "test_statistics.py")
)
test_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_stats)

pooled_statistic_test = test_stats.pooled_statistic_test
max_statistic_test = test_stats.max_statistic_test
naive_test = test_stats.naive_test
oracle_test = test_stats.oracle_test


class MultiScalePowerStudy:
    """
    Power analysis with multi-scale DGP: μ_g = δ·g, σ²_g = 2·g³
    
    This challenging scenario has arm 10 with highest mean (10δ) but also
    highest variance (100), testing whether algorithms can identify and
    allocate appropriately to the best arm despite high noise.
    """
    
    def __init__(self, seed=None):
        """
        Initialize study with optional random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.results = None
    
    def generate_multiscale_dgp(self, delta, k=10):
        """
        Generate multi-scale means and variances.

        Args:
            delta: Effect size scaling parameter
            k: Number of arms (default 10)

        Returns:
            means: Array of length k with μ_g = δ·g
            variances: Array of length k with σ²_g = 2·g³
        """
        arm_indices = np.arange(1, k + 1)
        means = delta * arm_indices            # [δ, 2δ, …, kδ]
        variances = 2.0 * arm_indices ** 3     # [2, 16, 54, …, 2k³]

        return means, variances
    
    def run_single_experiment(self, delta, k, T, algorithm, test_statistic, alpha=0.05):
        """
        Run single experiment with specified configuration.
        
        Args:
            delta: Effect size parameter
            k: Number of arms
            T: Sample size
            algorithm: Algorithm name ('equal', 'ucb', 'thompson', 'oracle', 'sn_ucb')
            test_statistic: Test to use ('max-log', 'max-linear', 'pooled')
            alpha: Significance level
            
        Returns:
            reject: Boolean indicating whether null was rejected
        """
        # Generate DGP
        means, variances = self.generate_multiscale_dgp(delta, k)
        
        # Generate data: shape (T, k)
        data = np.zeros((T, k))
        for g in range(k):
            data[:, g] = np.random.normal(
                loc=means[g],
                scale=np.sqrt(variances[g]),
                size=T
            )
        
        # Run algorithm
        if algorithm == 'equal':
            history = equal_allocation_algorithm(data, T, k)
        elif algorithm == 'ucb':
            history = standard_ucb_algorithm(data, T, k, c_param=2.0, burn_in=2)
        elif algorithm == 'ucb_v':
            history = variance_aware_ucb_algorithm(data, T, k, c_param=2.0, burn_in=2)
        elif algorithm == 'thompson':
            history = thompson_sampling_algorithm(data, T, k, burn_in=2)
        elif algorithm == 'oracle':
            # Oracle knows best arm is arm with index k-1 (arm 10)
            true_snrs = means / np.sqrt(variances)
            history = oracle_allocation_algorithm(data, T, k, true_snrs, burn_in=2)
        elif algorithm == 'sn_ucb':
            # Use actual SN-UCB algorithm
            history = sn_ucb_algorithm(data, T, k, burn_in=2)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Apply test statistic
        try:
            if algorithm == 'equal':
                _, _, reject = naive_test(data, history, alpha=alpha)
            elif algorithm == 'oracle':
                true_snrs = means / np.sqrt(variances)
                _, _, reject = oracle_test(data, history, true_snrs, alpha=alpha)
            elif test_statistic == 'pooled':
                _, _, reject = pooled_statistic_test(
                    data, history, alpha=alpha, method='threshold'
                )
            elif test_statistic == 'max-linear':
                _, _, reject = max_statistic_test(
                    data, history, alpha=alpha,
                    shape='linear', correction='bonferroni'
                )
            elif test_statistic == 'max-log':
                _, _, reject = max_statistic_test(
                    data, history, alpha=alpha,
                    shape='log', correction='bonferroni'
                )
            else:
                raise ValueError(f"Unknown test statistic: {test_statistic}")
        except Exception as e:
            print(f"Warning: Test failed for {test_statistic}: {e}")
            reject = False
        
        return reject
    
    def run_single_configuration(self, delta, algorithm, test_statistic, 
                                k=10, T=250, n_reps=500, alpha=0.05):
        """
        Run experiments for single (δ, algorithm, test) configuration.
        
        Args:
            delta: Effect size parameter
            algorithm: Algorithm name
            test_statistic: Test statistic name
            k: Number of arms
            T: Sample size
            n_reps: Number of replications
            alpha: Significance level
            
        Returns:
            DataFrame with power estimate
        """
        rejections = []
        
        for rep in range(n_reps):
            reject = self.run_single_experiment(
                delta, k, T, algorithm, test_statistic, alpha
            )
            rejections.append(reject)
        
        power = np.mean(rejections)
        
        return pd.DataFrame([{
            'delta': delta,
            'algorithm': algorithm,
            'test_statistic': test_statistic,
            'k': k,
            'T': T,
            'n_reps': n_reps,
            'power': power,
            'se': np.sqrt(power * (1 - power) / n_reps)
        }])
    
    def run_full_power_curves(self,
                             delta_values=None,
                             algorithms=None,
                             test_statistics=None,
                             k=10, T=250, n_reps=500,
                             alpha=0.05, verbose=True):
        """
        Run complete power curve analysis.
        
        Args:
            delta_values: Array of effect sizes to test
            algorithms: List of algorithm names
            test_statistics: List of test statistics
            k: Number of arms
            T: Sample size
            n_reps: Replications per configuration
            alpha: Significance level
            verbose: Whether to show progress
            
        Returns:
            DataFrame with all results
        """
        # Default parameters
        if delta_values is None:
            delta_values = np.linspace(0, 1.0, 11)
        if algorithms is None:
            algorithms = ['equal', 'sn_ucb', 'ucb', 'ucb_v', 'thompson', 'oracle']
        if test_statistics is None:
            test_statistics = ['max-log', 'max-linear', 'pooled']
        
        # Calculate total configurations
        total_configs = len(delta_values) * len(algorithms) * len(test_statistics)
        
        if verbose:
            print(f"Running Multi-Scale Power Analysis:")
            print(f"  Delta values: {len(delta_values)} ({delta_values[0]:.2f} to {delta_values[-1]:.2f})")
            print(f"  Algorithms: {algorithms}")
            print(f"  Test statistics: {test_statistics}")
            print(f"  k={k}, T={T}, n_reps={n_reps}")
            print(f"  Total configurations: {total_configs}")
            print(f"  Total experiments: {total_configs * n_reps}")
            print()
        
        all_results = []
        
        with tqdm(total=total_configs, desc="Configurations", disable=not verbose) as pbar:
            for delta in delta_values:
                for algorithm in algorithms:
                    for test_stat in test_statistics:
                        if verbose:
                            pbar.set_description(f"δ={delta:.2f}, {algorithm}, {test_stat}")
                        
                        result_df = self.run_single_configuration(
                            delta, algorithm, test_stat, k, T, n_reps, alpha
                        )
                        all_results.append(result_df)
                        
                        pbar.update(1)
        
        self.results = pd.concat(all_results, ignore_index=True)
        return self.results
    
    def create_power_curve_plots(self, results_df=None, output_dir=None):
        """
        Create 3 power curve plots (one per test statistic).
        
        Each plot shows power vs δ for all algorithms.
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            output_dir: Directory to save figures
            
        Returns:
            List of saved figure paths
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_power_curves() first.")
        
        # Use default output directory if not provided
        if output_dir is None:
            package_root = Path(__file__).parent.parent
            output_dir = package_root / 'outputs' / 'figures'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        test_statistics = results_df['test_statistic'].unique()
        
        # Algorithm display settings
        algorithm_labels = {
            'equal': 'Equal',
            'sn_ucb': 'SN-UCB',
            'ucb': 'UCB',
            'ucb_v': 'UCB-V',
            'thompson': 'Thompson',
            'oracle': 'Oracle',
        }
        colors = {
            'equal': 'gray',
            'sn_ucb': 'red',
            'ucb': 'blue',
            'ucb_v': '#ff7f0e',
            'thompson': 'green',
            'oracle': 'black',
        }
        markers = {
            'equal': 'o',
            'sn_ucb': 's',
            'ucb': '^',
            'ucb_v': 'P',
            'thompson': 'v',
            'oracle': 'D',
        }
        
        for test_stat in test_statistics:
            # Filter data for this test statistic
            df_test = results_df[results_df['test_statistic'] == test_stat]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all six algorithms (UCB-V included).
            algorithms = [
                a for a in ['equal', 'sn_ucb', 'ucb', 'ucb_v', 'thompson', 'oracle']
                if a in df_test['algorithm'].values
            ]
            for alg in algorithms:
                df_alg = df_test[df_test['algorithm'] == alg].sort_values('delta')
                
                label = algorithm_labels.get(alg, alg.upper())
                color = colors.get(alg, 'gray')
                marker = markers.get(alg, 'o')
                
                ax.plot(
                    df_alg['delta'], df_alg['power'],
                    label=label, color=color, marker=marker,
                    linewidth=2, markersize=8, alpha=0.8
                )
            
            # Format plot
            test_label = test_stat.replace('-', ' ').replace('_', ' ').upper()
            ax.set_xlabel('Effect Size (δ)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Power', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Multi-Scale Power Curves: {test_label}\n'
                f'(μ_g = δ·g, σ²_g = 2·g³, k=10, T=250)',
                fontsize=14, fontweight='bold'
            )
            ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1.05])
            ax.set_xlim([df_test['delta'].min() - 0.02, df_test['delta'].max() + 0.02])
            
            # Add horizontal line at α = 0.05
            ax.axhline(y=0.05, color='red', linestyle=':', linewidth=1, alpha=0.5, label='α = 0.05')
            
            plt.tight_layout()
            
            # Save figure
            filename = f'multiscale_power_{test_stat.replace("-", "_")}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_files.append(str(filepath))
            print(f"Saved: {filepath}")
        
        return saved_files
    
    def generate_summary_table(self, results_df=None, output_dir=None):
        """
        Generate LaTeX summary table of key power values.
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            output_dir: Directory to save table
            
        Returns:
            Path to saved LaTeX file
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_power_curves() first.")
        
        # Use default output directory if not provided
        if output_dir is None:
            package_root = Path(__file__).parent.parent
            output_dir = package_root / 'outputs' / 'tables'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select key delta values for table
        delta_values = sorted(results_df['delta'].unique())
        key_deltas = [delta_values[0], delta_values[len(delta_values)//2], delta_values[-1]]
        
        # Create LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Multi-Scale Power Analysis: Key Results}")
        latex_lines.append("\\label{tab:multiscale_power}")
        latex_lines.append("\\begin{tabular}{llrrr}")
        latex_lines.append("\\toprule")
        latex_lines.append(f"Algorithm & Test & $\\delta={key_deltas[0]:.2f}$ & $\\delta={key_deltas[1]:.2f}$ & $\\delta={key_deltas[2]:.2f}$ \\\\")
        latex_lines.append("\\midrule")
        
        algorithms = ['equal', 'sn_ucb', 'ucb', 'ucb_v', 'thompson', 'oracle']
        test_statistics = ['max-linear', 'max-log', 'pooled']
        
        for alg in algorithms:
            for test_stat in test_statistics:
                df_subset = results_df[
                    (results_df['algorithm'] == alg) &
                    (results_df['test_statistic'] == test_stat) &
                    (results_df['delta'].isin(key_deltas))
                ].sort_values('delta')
                
                if len(df_subset) == 3:
                    powers = df_subset['power'].values
                    latex_lines.append(
                        f"{alg.upper()} & {test_stat} & "
                        f"{powers[0]:.3f} & {powers[1]:.3f} & {powers[2]:.3f} \\\\"
                    )
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Save to file
        filepath = output_dir / 'multiscale_power_summary.tex'
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex_lines))
        
        print(f"Saved LaTeX table: {filepath}")
        return str(filepath)
    
    def run_complete_study(self, quick_mode=False, verbose=True):
        """
        Convenience method to run entire study and generate all outputs.
        
        Args:
            quick_mode: If True, uses fewer delta values and replications
            verbose: Whether to show progress
            
        Returns:
            Dictionary with all results and outputs
        """
        if quick_mode:
            if verbose:
                print("=" * 60)
                print("QUICK MODE - Using reduced parameters for testing")
                print("=" * 60)
            delta_values = np.array([0.0, 0.25, 0.5])
            n_reps = 100
        else:
            if verbose:
                print("=" * 60)
                print("FULL MODE - Running comprehensive power analysis")
                print("=" * 60)
            delta_values = np.linspace(0, 1., 11)
            n_reps = 1000
        
        # Run full grid
        results_df = self.run_full_power_curves(
            delta_values=delta_values,
            algorithms=['equal', 'sn_ucb', 'ucb', 'ucb_v', 'thompson', 'oracle'],
            test_statistics=['max-log', 'max-linear', 'pooled'],
            k=10, T=250, n_reps=n_reps,
            verbose=verbose
        )
        
        # Save raw results
        package_root = Path(__file__).parent.parent
        output_dir = package_root / 'outputs' / 'csvs'
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / 'multiscale_power_results.csv'
        results_df.to_csv(results_path, index=False)
        if verbose:
            print(f"\nSaved raw results: {results_path}")
        
        # Display summary statistics
        if verbose:
            print("\n" + "=" * 60)
            print("POWER SUMMARY")
            print("=" * 60)
            summary = results_df.groupby(['test_statistic', 'algorithm'])['power'].agg(['mean', 'min', 'max'])
            print(summary.to_string())
        
        # Create power curve plots
        if verbose:
            print("\n" + "=" * 60)
            print("GENERATING POWER CURVES")
            print("=" * 60)
        figures_dir = package_root / 'outputs' / 'figures'
        power_curve_files = self.create_power_curve_plots(results_df, output_dir=str(figures_dir))

        # Generate LaTeX table
        tables_dir = package_root / 'outputs' / 'tables'
        latex_file = self.generate_summary_table(results_df, output_dir=str(tables_dir))
        
        if verbose:
            print("\n" + "=" * 60)
            print("STUDY COMPLETE!")
            print("=" * 60)
            print(f"Results: {results_path}")
            print(f"Power curves: {len(power_curve_files)} figures saved")
            print(f"LaTeX table: {latex_file}")
        
        return {
            'results_df': results_df,
            'power_curve_files': power_curve_files,
            'latex_file': latex_file
        }


# CLI: full mode by default; --quick for a ~30s sanity check.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-scale power study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (3 deltas, 100 reps).")
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()

    print("Multi-Scale Power Study")
    print("=" * 60)
    MultiScalePowerStudy(seed=cli_args.seed).run_complete_study(
        quick_mode=cli_args.quick, verbose=True
    )
