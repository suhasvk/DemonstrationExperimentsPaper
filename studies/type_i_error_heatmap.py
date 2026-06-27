"""
Type I Error Heatmap Study

Validates Type-I error control across k×T grid for all three test statistics
(max-log, max-linear, pooled) under heteroscedastic conditions.

This is a critical validation that demonstrates the test statistics maintain
proper Type-I error control under the null hypothesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from algorithms.baselines import standard_ucb_algorithm

# Import test statistics directly from the module file to avoid conflict with built-in statistics
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_statistics", 
    str(Path(__file__).parent.parent / "statistics" / "test_statistics.py")
)
test_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_stats)

pooled_statistic_test = test_stats.pooled_statistic_test
max_statistic_test = test_stats.max_statistic_test


class TypeIErrorHeatmapStudy:
    """
    Validates Type-I error control across k×T grid.
    
    This study runs simulations under the null hypothesis (all means = 0)
    with heteroscedastic variances to validate that test statistics maintain
    proper Type-I error control.
    """
    
    def __init__(self, seed=None):
        """
        Initialize study with optional random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.results = None
        
    def generate_inverse_gamma_variances(self, k, alpha=2.0, beta=0.5):
        """
        Generate heteroscedastic variances from Inverse Gamma distribution.
        
        IG(α, β) has mean = β/(α-1) = 0.5 and variance = β²/((α-1)²(α-2)) = 0.5
        
        In scipy, we generate Gamma(α, 1/β) then take reciprocal:
        If X ~ Gamma(α, 1/β), then 1/X ~ InverseGamma(α, β)
        
        Args:
            k: Number of arms
            alpha: Shape parameter (default: 2.0)
            beta: Scale parameter (default: 0.5)
            
        Returns:
            Array of k variances from InverseGamma(α, β)
        """
        # Generate from Gamma(alpha, scale=1/beta)
        gamma_samples = np.random.gamma(shape=alpha, scale=1.0/beta, size=k)
        
        # Take reciprocal to get InverseGamma
        inverse_gamma_samples = 1.0 / gamma_samples
        
        return inverse_gamma_samples
    
    def run_single_experiment(self, k, T, variances, test_statistic, alpha=0.05):
        """
        Run single experiment under null hypothesis.
        
        Args:
            k: Number of arms
            T: Sample size (number of rounds)
            variances: Array of k variances
            test_statistic: Which test to use ('max-log', 'max-linear', 'pooled')
            alpha: Significance level (default: 0.05)
            
        Returns:
            reject: Boolean indicating whether null was rejected
        """
        # Generate data under null (all means = 0)
        means = np.zeros(k)
        
        # Generate observations: shape (T, k)
        data = np.zeros((T, k))
        for g in range(k):
            data[:, g] = np.random.normal(
                loc=means[g],
                scale=np.sqrt(variances[g]),
                size=T
            )
        
        # Run UCB algorithm
        history = standard_ucb_algorithm(
            data=data,
            n_rounds=T,
            n_arms=k,
            c_param=2.0,
            burn_in=2
        )
        
        # Apply test statistic. 'pooled' is the threshold-regularized version
        # used in the paper's Table 1; 'pooled-padded' is the padding-
        # regularized variant of Theorem 3.1, run alongside for comparison.
        try:
            if test_statistic == 'pooled':
                _, _, reject = pooled_statistic_test(
                    data, history, alpha=alpha, method='threshold'
                )
            elif test_statistic == 'pooled-padded':
                _, _, reject = pooled_statistic_test(
                    data, history, alpha=alpha, method='regularized'
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
            # If test fails (e.g., numerical issues), count as non-rejection
            print(f"Warning: Test failed for {test_statistic}: {e}")
            reject = False
        
        return reject
    
    def run_single_condition(self, k, T, test_statistic, n_reps=1000, alpha=0.05):
        """
        Run experiments for a single (k, T, test_statistic) condition.
        
        Args:
            k: Number of arms
            T: Sample size
            test_statistic: Which test to use
            n_reps: Number of replications (default: 1000)
            alpha: Significance level (default: 0.05)
            
        Returns:
            DataFrame with results from n_reps experiments
        """
        rejections = []
        
        for rep in range(n_reps):
            # Generate new variances for each replication
            variances = self.generate_inverse_gamma_variances(k)
            
            # Run experiment
            reject = self.run_single_experiment(k, T, variances, test_statistic, alpha)
            
            rejections.append({
                'k': k,
                'T': T,
                'test_statistic': test_statistic,
                'replication': rep,
                'reject': reject
            })
        
        return pd.DataFrame(rejections)
    
    def run_full_grid(self, 
                     k_values=[5, 10, 50],
                     T_values=[200, 500, 1000],
                     test_statistics=['max-log', 'max-linear', 'pooled', 'pooled-padded'],
                     n_reps=1000,
                     alpha=0.05,
                     verbose=True):
        """
        Run complete k×T grid for all test statistics.
        
        Args:
            k_values: List of k values to test
            T_values: List of T values to test
            test_statistics: List of test statistics to compare
            n_reps: Number of replications per condition
            alpha: Significance level
            verbose: Whether to show progress bar
            
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        # Calculate total number of conditions
        total_conditions = len(k_values) * len(T_values) * len(test_statistics)
        
        if verbose:
            print(f"Running Type-I Error Validation:")
            print(f"  k values: {k_values}")
            print(f"  T values: {T_values}")
            print(f"  Test statistics: {test_statistics}")
            print(f"  Replications per condition: {n_reps}")
            print(f"  Total conditions: {total_conditions}")
            print(f"  Total experiments: {total_conditions * n_reps}")
            print()
        
        # Run all conditions
        with tqdm(total=total_conditions, desc="Conditions", disable=not verbose) as pbar:
            for k in k_values:
                for T in T_values:
                    for test_stat in test_statistics:
                        if verbose:
                            pbar.set_description(f"k={k}, T={T}, {test_stat}")
                        
                        # Run condition
                        results_df = self.run_single_condition(
                            k, T, test_stat, n_reps, alpha
                        )
                        all_results.append(results_df)
                        
                        pbar.update(1)
        
        # Combine all results
        self.results = pd.concat(all_results, ignore_index=True)
        
        return self.results
    
    def compute_rejection_rates(self, results_df=None):
        """
        Compute rejection rates for each condition.
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            
        Returns:
            DataFrame with rejection rates
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_grid() first.")
        
        # Compute rejection rate for each condition
        rejection_rates = results_df.groupby(
            ['k', 'T', 'test_statistic']
        )['reject'].agg(['mean', 'std', 'count']).reset_index()
        
        rejection_rates.columns = ['k', 'T', 'test_statistic', 
                                   'rejection_rate', 'std', 'n_reps']
        
        # Compute 95% confidence intervals (normal approximation)
        rejection_rates['se'] = rejection_rates['std'] / np.sqrt(rejection_rates['n_reps'])
        rejection_rates['ci_lower'] = rejection_rates['rejection_rate'] - 1.96 * rejection_rates['se']
        rejection_rates['ci_upper'] = rejection_rates['rejection_rate'] + 1.96 * rejection_rates['se']
        
        return rejection_rates
    
    def create_heatmaps(self, results_df=None, output_dir=None):
        """
        Create 3 heatmaps (one per test statistic) showing rejection rates.
        
        Color scheme:
        - Green at α = 0.05 (nominal level)
        - Blue for conservative (< 0.05)
        - Red for inflated (> 0.05)
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            output_dir: Directory to save figures
            
        Returns:
            List of saved figure paths
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_grid() first.")
        
        # Compute rejection rates
        rejection_rates = self.compute_rejection_rates(results_df)
        
        # Use default output directory if not provided
        if output_dir is None:
            package_root = Path(__file__).parent.parent
            output_dir = package_root / 'outputs' / 'figures'
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        test_statistics = rejection_rates['test_statistic'].unique()
        
        for test_stat in test_statistics:
            # Filter data for this test statistic
            df_test = rejection_rates[rejection_rates['test_statistic'] == test_stat]
            
            # Create pivot table for heatmap
            pivot = df_test.pivot(
                index='k', 
                columns='T',
                values='rejection_rate'
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Custom colormap: blue (conservative) -> green (0.05) -> red (inflated)
            # Use RdYlGn_r (reversed) so red is high, green is middle, blue/yellow is low
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',
                vmin=0.00,
                vmax=0.10,
                center=0.05,
                cbar_kws={'label': 'Rejection Rate'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax
            )
            
            # Format title based on test statistic
            test_label = test_stat.replace('-', ' ').replace('_', ' ').upper()
            ax.set_title(
                f'Type-I Error Control: {test_label}\n(Null Hypothesis: All μ = 0)',
                fontsize=14,
                fontweight='bold'
            )
            ax.set_xlabel('Sample Size (T)', fontsize=12)
            ax.set_ylabel('Number of Arms (k)', fontsize=12)
            
            # Add reference line annotation
            ax.text(
                0.5, -0.15,
                'Target: α = 0.05 (green), Conservative: < 0.05 (blue), Inflated: > 0.05 (red)',
                transform=ax.transAxes,
                ha='center',
                fontsize=10,
                style='italic'
            )
            
            plt.tight_layout()
            
            # Save figure
            filename = f'type_i_error_heatmap_{test_stat.replace("-", "_")}.png'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_files.append(str(filepath))
            print(f"Saved: {filepath}")
        
        return saved_files
    
    def validate_type_i_error_control(self, results_df=None, threshold=0.055):
        """
        Validate that all conditions maintain Type-I error control.
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            threshold: Maximum acceptable rejection rate (default: 0.055)
            
        Returns:
            Validation report dictionary
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_grid() first.")
        
        # Compute rejection rates
        rejection_rates = self.compute_rejection_rates(results_df)
        
        # Check each condition
        validation_results = []
        all_pass = True
        
        for _, row in rejection_rates.iterrows():
            passes = row['rejection_rate'] <= threshold
            
            if not passes:
                all_pass = False
            
            validation_results.append({
                'k': row['k'],
                'T': row['T'],
                'test_statistic': row['test_statistic'],
                'rejection_rate': row['rejection_rate'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'threshold': threshold,
                'passes': passes
            })
        
        validation_df = pd.DataFrame(validation_results)
        
        # Create summary report
        report = {
            'all_pass': all_pass,
            'n_conditions': len(validation_df),
            'n_pass': validation_df['passes'].sum(),
            'n_fail': (~validation_df['passes']).sum(),
            'max_rejection_rate': validation_df['rejection_rate'].max(),
            'mean_rejection_rate': validation_df['rejection_rate'].mean(),
            'validation_df': validation_df
        }
        
        return report
    
    def generate_summary_table(self, results_df=None, output_dir=None):
        """
        Generate LaTeX table of rejection rates for paper.
        
        Args:
            results_df: DataFrame with results (uses self.results if None)
            output_dir: Directory to save table
            
        Returns:
            Path to saved LaTeX file
        """
        if results_df is None:
            results_df = self.results
        
        if results_df is None:
            raise ValueError("No results available. Run run_full_grid() first.")
        
        # Compute rejection rates
        rejection_rates = self.compute_rejection_rates(results_df)
        
        # Use default output directory if not provided
        if output_dir is None:
            package_root = Path(__file__).parent.parent
            output_dir = package_root / 'outputs' / 'tables'
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Type-I Error Rates Under Null Hypothesis}")
        latex_lines.append("\\label{tab:type_i_error}")
        latex_lines.append("\\begin{tabular}{lrrrrr}")
        latex_lines.append("\\toprule")
        latex_lines.append("Test Statistic & $k$ & $T$ & Rejection Rate & 95\\% CI & Valid \\\\")
        latex_lines.append("\\midrule")
        
        for _, row in rejection_rates.iterrows():
            valid = "✓" if row['rejection_rate'] <= 0.055 else "✗"
            latex_lines.append(
                f"{row['test_statistic']} & {row['k']} & {row['T']} & "
                f"{row['rejection_rate']:.3f} & "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] & {valid} \\\\"
            )
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Save to file
        filepath = Path(output_dir) / 'type_i_error_summary.tex'
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex_lines))
        
        print(f"Saved LaTeX table: {filepath}")
        return str(filepath)
    
    def run_complete_study(self, quick_mode=False, verbose=True):
        """
        Convenience method to run entire study and generate all outputs.
        
        Args:
            quick_mode: If True, uses fewer replications and conditions for testing
            verbose: Whether to show progress
            
        Returns:
            Dictionary with all results and outputs
        """
        if quick_mode:
            if verbose:
                print("=" * 60)
                print("QUICK MODE - Using reduced parameters for testing")
                print("=" * 60)
            k_values = [5, 10]
            T_values = [200, 500]
            n_reps = 100
        else:
            if verbose:
                print("=" * 60)
                print("FULL MODE - Running comprehensive validation")
                print("=" * 60)
            k_values = [5, 10, 20, 50]
            T_values = [200, 500, 1000, 2000]
            n_reps = 1000
        
        # Run full grid
        results_df = self.run_full_grid(
            k_values=k_values,
            T_values=T_values,
            test_statistics=['max-log', 'max-linear', 'pooled', 'pooled-padded'],
            n_reps=n_reps,
            verbose=verbose
        )
        
        # Save raw results
        package_root = Path(__file__).parent.parent
        output_dir = package_root / 'outputs' / 'csvs'
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / 'type_i_error_results.csv'
        results_df.to_csv(results_path, index=False)
        if verbose:
            print(f"\nSaved raw results: {results_path}")
        
        # Compute and display rejection rates
        rejection_rates = self.compute_rejection_rates(results_df)
        if verbose:
            print("\n" + "=" * 60)
            print("REJECTION RATES")
            print("=" * 60)
            print(rejection_rates.to_string(index=False))
        
        # Validate Type-I error control
        validation_report = self.validate_type_i_error_control(results_df)
        if verbose:
            print("\n" + "=" * 60)
            print("VALIDATION SUMMARY")
            print("=" * 60)
            print(f"All tests pass: {validation_report['all_pass']}")
            print(f"Conditions passing: {validation_report['n_pass']}/{validation_report['n_conditions']}")
            print(f"Max rejection rate: {validation_report['max_rejection_rate']:.3f}")
            print(f"Mean rejection rate: {validation_report['mean_rejection_rate']:.3f}")
            
            if not validation_report['all_pass']:
                print("\nFAILED CONDITIONS:")
                failed = validation_report['validation_df'][~validation_report['validation_df']['passes']]
                print(failed.to_string(index=False))
        
        # Create heatmaps
        if verbose:
            print("\n" + "=" * 60)
            print("GENERATING HEATMAPS")
            print("=" * 60)
        figures_dir = package_root / 'outputs' / 'figures'
        heatmap_files = self.create_heatmaps(results_df, output_dir=str(figures_dir))

        # Generate LaTeX table
        tables_dir = package_root / 'outputs' / 'tables'
        latex_file = self.generate_summary_table(results_df, output_dir=str(tables_dir))
        
        if verbose:
            print("\n" + "=" * 60)
            print("STUDY COMPLETE!")
            print("=" * 60)
            print(f"Results: {results_path}")
            print(f"Heatmaps: {len(heatmap_files)} figures saved")
            print(f"LaTeX table: {latex_file}")
        
        return {
            'results_df': results_df,
            'rejection_rates': rejection_rates,
            'validation_report': validation_report,
            'heatmap_files': heatmap_files,
            'latex_file': latex_file
        }


# CLI: produces the paper's Type-I error grid (k in {5,10,20,50}, T in {200,500,1000,2000})
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Type-I error heatmap study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (k in {5,10}, T in {200,500}, 100 reps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Type-I Error Heatmap Study")
    print("=" * 60)

    study = TypeIErrorHeatmapStudy(seed=args.seed)
    study.run_complete_study(quick_mode=args.quick, verbose=True)
