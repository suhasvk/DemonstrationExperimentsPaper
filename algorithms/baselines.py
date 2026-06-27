"""
Baseline Algorithms for "Demonstration Experiments" paper

This module implements baseline algorithms for comparison with SN-UCB:
- Standard UCB
- Thompson Sampling  
- Equal allocation (naive)
- Oracle allocation (knowing best arm)
- Fixed allocation with Bonferroni correction
"""

import numpy as np
from numba import njit
from typing import Dict, Any, Optional, List
from scipy import stats


@njit
def standard_ucb_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    c_param: float = 2.0,
    burn_in: int = 2
) -> np.ndarray:
    """
    Standard Upper Confidence Bound algorithm.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        c_param: exploration parameter (typically sqrt(2))
        burn_in: number of initial samples per arm
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    history = np.zeros((n_rounds, n_arms), dtype=np.int32)
    t = 0
    
    # Burn-in phase: sample each arm 'burn_in' times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        arm = round_idx % n_arms
        history[t, arm] = 1
        t += 1
    
    # UCB selection phase
    while t < n_rounds:
        ucb_values = np.zeros(n_arms)
        
        for arm in range(n_arms):
            n_samples = np.sum(history[:t, arm])
            
            if n_samples == 0:
                ucb_values[arm] = np.inf
                continue
            
            # Compute empirical mean
            arm_mask = history[:t, arm] == 1
            arm_data = data[:t][arm_mask, arm]
            empirical_mean = np.mean(arm_data)
            
            # UCB formula: mean + sqrt(c * log(t) / n)
            confidence_width = np.sqrt(c_param * np.log(t) / n_samples)
            ucb_values[arm] = empirical_mean + confidence_width
        
        # Select arm with highest UCB value
        selected_arm = np.argmax(ucb_values)
        history[t, selected_arm] = 1
        t += 1
    
    return history


def thompson_sampling_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    burn_in: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Thompson Sampling algorithm for Gaussian bandits.
    Uses a simple normal posterior approximation for unknown mean and variance.

    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        prior_alpha: prior precision parameter
        prior_beta: prior variance parameter
        burn_in: number of initial samples per arm
        seed: optional explicit seed. When None (the call sites in the
            paper studies pass nothing), draws come from numpy's *global*
            RNG, which the studies seed once at construction. This keeps
            Thompson reproducible alongside the rest of the pipeline.

    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    # Use the global numpy RNG (which the studies seed at __init__) unless
    # the caller passes an explicit seed. Passing `seed=` forks a fresh
    # generator and decouples Thompson's draws from the global state — this
    # is occasionally useful for ad-hoc tests but breaks reproducibility
    # of the whole rep loop, so the studies leave it None.
    rng = np.random.default_rng(seed) if seed is not None else np.random
    history = np.zeros((n_rounds, n_arms), dtype=int)
    t = 0

    # Burn-in phase: sample each arm 'burn_in' times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        arm = round_idx % n_arms
        history[t, arm] = 1
        t += 1

    # Thompson sampling phase
    while t < n_rounds:
        sampled_means = np.zeros(n_arms)

        for arm in range(n_arms):
            n_samples = np.sum(history[:t, arm])

            if n_samples == 0:
                # Sample from prior
                sampled_means[arm] = rng.normal(0, 1)
                continue

            # Get arm data
            arm_mask = history[:t, arm] == 1
            arm_data = data[:t][arm_mask, arm]

            # Posterior parameters (normal-inverse-gamma)
            sample_mean = np.mean(arm_data)
            sample_var = np.var(arm_data, ddof=1) if n_samples > 1 else 1.0

            # Simple normal approximation for Thompson sampling
            posterior_mean = sample_mean
            posterior_std = np.sqrt(sample_var / n_samples)

            # Sample from posterior
            sampled_means[arm] = rng.normal(posterior_mean, posterior_std)
        
        # Select arm with highest sampled mean
        selected_arm = np.argmax(sampled_means)
        history[t, selected_arm] = 1
        t += 1
    
    return history


@njit
def variance_aware_ucb_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    c_param: float = 2.0,
    burn_in: int = 2,
) -> np.ndarray:
    """
    UCB with exploration bonus scaled by the empirical standard deviation
    (UCB-V, Audibert et al. 2009):

        UCB_g(t) = mean_hat_g + sigma_hat_g * sqrt(c * log(t) / N_g(t))

    Gaussian analogue of UCB1 — targets the arm with the largest *mean* while
    accounting for arm-specific variance. UCB1's fixed bonus is too narrow
    when sigma is large; scaling by sigma_hat restores its intended behavior
    and provides the natural contrast with SN-UCB (which targets SNR).
    """
    history = np.zeros((n_rounds, n_arms), dtype=np.int32)
    t = 0

    # Burn-in: sample each arm burn_in times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        history[t, round_idx % n_arms] = 1
        t += 1

    while t < n_rounds:
        ucb_values = np.zeros(n_arms)

        for arm in range(n_arms):
            n_samples = np.sum(history[:t, arm])
            if n_samples == 0:
                ucb_values[arm] = np.inf
                continue

            arm_mask = history[:t, arm] == 1
            arm_data = data[:t][arm_mask, arm]
            emp_mean = np.mean(arm_data)

            if n_samples > 1:
                emp_var = np.sum((arm_data - emp_mean) ** 2) / (n_samples - 1)
                emp_std = np.sqrt(emp_var)
            else:
                emp_std = 1.0

            bonus = emp_std * np.sqrt(c_param * np.log(t) / n_samples)
            ucb_values[arm] = emp_mean + bonus

        history[t, np.argmax(ucb_values)] = 1
        t += 1

    return history


@njit
def equal_allocation_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int
) -> np.ndarray:
    """
    Equal allocation (round-robin) algorithm.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    history = np.zeros((n_rounds, n_arms), dtype=np.int32)
    
    for t in range(n_rounds):
        arm = t % n_arms
        history[t, arm] = 1
    
    return history


@njit
def oracle_allocation_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    true_SNRs: np.ndarray,
    burn_in: int = 2
) -> np.ndarray:
    """
    Oracle allocation algorithm (knows true means).
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        true_SNRs: true means of each arm
        burn_in: number of initial samples per arm
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    history = np.zeros((n_rounds, n_arms), dtype=np.int32)
    best_arm = np.argmax(true_SNRs)
    t = 0
    
    # Burn-in phase: sample each arm 'burn_in' times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        arm = round_idx % n_arms
        history[t, arm] = 1
        t += 1
    
    # Oracle phase: always choose best arm
    while t < n_rounds:
        history[t, best_arm] = 1
        t += 1
    
    return history


def fixed_allocation_bonferroni_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Fixed allocation with Bonferroni correction.
    Allocates samples equally and applies Bonferroni correction for multiple testing.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        alpha: significance level
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    # Equal allocation
    history = equal_allocation_algorithm(data, n_rounds, n_arms)
    
    # Note: The Bonferroni correction is applied in the testing phase,
    # not in the allocation phase. This function just does equal allocation.
    return history


def epsilon_greedy_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    epsilon: float = 0.1,
    burn_in: int = 2,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Epsilon-greedy algorithm.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        epsilon: exploration probability
        burn_in: number of initial samples per arm
        seed: random seed
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    rng = np.random.default_rng(seed)
    history = np.zeros((n_rounds, n_arms), dtype=int)
    t = 0
    
    # Burn-in phase: sample each arm 'burn_in' times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        arm = round_idx % n_arms
        history[t, arm] = 1
        t += 1
    
    # Epsilon-greedy phase
    while t < n_rounds:
        if rng.random() < epsilon:
            # Explore: choose random arm
            selected_arm = rng.integers(0, n_arms)
        else:
            # Exploit: choose arm with highest empirical mean
            empirical_means = np.zeros(n_arms)
            
            for arm in range(n_arms):
                n_samples = np.sum(history[:t, arm])
                if n_samples > 0:
                    arm_mask = history[:t, arm] == 1
                    arm_data = data[:t][arm_mask, arm]
                    empirical_means[arm] = np.mean(arm_data)
                else:
                    empirical_means[arm] = 0.0
            
            selected_arm = np.argmax(empirical_means)
        
        history[t, selected_arm] = 1
        t += 1
    
    return history


class BaselineAlgorithms:
    """
    Class for running baseline algorithms with consistent interface.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize baseline algorithms runner.
        
        Args:
            seed: random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def run_algorithm(
        self,
        algorithm_name: str,
        data: np.ndarray,
        algorithm_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Run a specified baseline algorithm.
        
        Args:
            algorithm_name: name of the algorithm
            data: (n_rounds, n_arms) array of potential observations
            algorithm_params: parameters for the algorithm
        
        Returns:
            history: (n_rounds, n_arms) binary allocation matrix
        """
        n_rounds, n_arms = data.shape
        
        if algorithm_name == 'ucb':
            return standard_ucb_algorithm(
                data, n_rounds, n_arms,
                c_param=algorithm_params.get('c_param', 2.0),
                burn_in=algorithm_params.get('burn_in', 2)
            )
        
        elif algorithm_name == 'thompson':
            return thompson_sampling_algorithm(
                data, n_rounds, n_arms,
                prior_alpha=algorithm_params.get('prior_alpha', 1.0),
                prior_beta=algorithm_params.get('prior_beta', 1.0),
                burn_in=algorithm_params.get('burn_in', 2),
                seed=algorithm_params.get('seed', self.seed)
            )
        
        elif algorithm_name == 'equal':
            return equal_allocation_algorithm(data, n_rounds, n_arms)
        
        elif algorithm_name == 'oracle':
            true_SNRs = algorithm_params.get('true_SNRs')
            if true_SNRs is None:
                raise ValueError("Oracle algorithm requires 'true_SNRs' parameter")
            return oracle_allocation_algorithm(
                data, n_rounds, n_arms, true_SNRs,
                burn_in=algorithm_params.get('burn_in', 2)
            )
        
        elif algorithm_name == 'bonferroni':
            return fixed_allocation_bonferroni_algorithm(
                data, n_rounds, n_arms,
                alpha=algorithm_params.get('alpha', 0.05)
            )
        
        elif algorithm_name == 'epsilon_greedy':
            return epsilon_greedy_algorithm(
                data, n_rounds, n_arms,
                epsilon=algorithm_params.get('epsilon', 0.1),
                burn_in=algorithm_params.get('burn_in', 2),
                seed=algorithm_params.get('seed', self.seed)
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def compare_algorithms(
        self,
        algorithms: List[str],
        data: np.ndarray,
        algorithm_params: Dict[str, Dict[str, Any]],
        true_SNRs: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple algorithms on the same data.
        
        Args:
            algorithms: list of algorithm names
            data: (n_rounds, n_arms) array of potential observations
            algorithm_params: parameters for each algorithm
            true_SNRs: true means for oracle and regret computation
        
        Returns:
            Dictionary with results for each algorithm
        """
        results = {}
        
        for algorithm in algorithms:
            params = algorithm_params.get(algorithm, {})
            
            # Add true means for oracle
            if algorithm == 'oracle' and true_SNRs is not None:
                params['true_SNRs'] = true_SNRs
            
            # Run algorithm
            history = self.run_algorithm(algorithm, data, params)
            
            # Compute metrics
            total_reward = np.sum(data * history)
            arm_allocations = np.sum(history, axis=0)
            
            # Compute regret if true SNRs are known
            regret = 0.0
            if true_SNRs is not None:
                best_SNR = np.max(true_SNRs)
                n_rounds, n_arms = data.shape
                for t in range(n_rounds):
                    for arm in range(n_arms):
                        if history[t, arm] == 1:
                            regret += best_SNR - true_SNRs[arm]
                            break
            
            results[algorithm] = {
                'history': history,
                'total_reward': total_reward,
                'arm_allocations': arm_allocations,
                'cumulative_regret': regret,
                'best_arm_fraction': np.max(arm_allocations) / np.sum(arm_allocations)
            }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Baseline Algorithms ===")
    
    # Set up test problem
    np.random.seed(42)
    n_arms = 3
    n_rounds = 1000
    true_SNRs = np.array([0.3, 0.0, -0.1])
    variances = np.array([1.0, 1.0, 1.0])
    
    # Generate data
    data = np.zeros((n_rounds, n_arms))
    for arm in range(n_arms):
        data[:, arm] = np.random.normal(true_SNRs[arm], np.sqrt(variances[arm]), n_rounds)
    
    # Test individual algorithms
    baseline_runner = BaselineAlgorithms(seed=42)
    
    algorithms_to_test = ['ucb', 'thompson', 'equal', 'oracle', 'epsilon_greedy']
    algorithm_params = {
        'ucb': {'c_param': 2.0},
        'thompson': {'prior_alpha': 1.0, 'prior_beta': 1.0},
        'equal': {},
        'oracle': {'true_SNRs': true_SNRs},
        'epsilon_greedy': {'epsilon': 0.1}
    }
    
    print("Testing individual algorithms:")
    for algorithm in algorithms_to_test:
        params = algorithm_params[algorithm]
        history = baseline_runner.run_algorithm(algorithm, data, params)
        arm_allocations = np.sum(history, axis=0)
        total_reward = np.sum(data * history)
        
        print(f"{algorithm:15s}: allocations={arm_allocations}, reward={total_reward:.2f}")
    
    # Test comparison function
    print("\nTesting algorithm comparison:")
    comparison_results = baseline_runner.compare_algorithms(
        algorithms_to_test, data, algorithm_params, true_SNRs
    )
    
    print("\nComparison Results:")
    for algorithm, results in comparison_results.items():
        print(f"{algorithm:15s}: reward={results['total_reward']:8.2f}, "
              f"regret={results['cumulative_regret']:8.2f}, "
              f"best_arm_frac={results['best_arm_fraction']:.3f}")
    
    print("\n=== Baseline Algorithms Test Complete ===")
