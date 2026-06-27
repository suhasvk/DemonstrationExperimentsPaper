"""
Self-Normalized Upper Confidence Bound (SN-UCB) Algorithm
Implementation for "Demonstration Experiments" paper

This module implements the SN-UCB algorithm from Section 4 of the paper,
which uses self-normalized statistics and proper exploration bounds.
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional


@njit
def compute_self_normalized_statistic(data: np.ndarray, history: np.ndarray, arm: int) -> float:
    """
    Compute the self-normalized statistic for a given arm.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        history: (n_rounds, n_arms) binary array indicating which arms were pulled
        arm: arm index to compute statistic for
    
    Returns:
        Self-normalized statistic Z_g = sum(X_g) / sqrt(sum((X_g - mu_hat_g)^2))
    """
    # Get observations for this arm
    arm_mask = history[:, arm] == 1
    if np.sum(arm_mask) == 0:
        return 0.0
    
    arm_observations = data[arm_mask, arm]
    n_samples = len(arm_observations)
    
    if n_samples <= 1:
        return 0.0
    
    # Compute running mean
    running_mean = np.mean(arm_observations)
    
    # Compute quadratic variation
    quadratic_variation = np.sum((arm_observations - running_mean) ** 2)
    
    if quadratic_variation <= 1e-10:  # Avoid division by zero
        return 0.0
    
    # Self-normalized statistic
    return np.sum(arm_observations) / np.sqrt(quadratic_variation)


@njit
def compute_exploration_bound(n_samples: int, t: int, scale_param: float = 4.5, tail_param: float = 2.01) -> float:
    """
    Compute exploration bound τ(n, t; β) = scale_param · sqrt(tail_param · log(t) / n).

    Paper §4 originally quoted τ(n, t; β) = 5ν² · sqrt(β log(4t)/n) with β > 2
    and ν the sub-Gaussian parameter from Assumption 2 (ν = 1 for standard
    Gaussians). A tighter analysis lets the leading constant be 4.5 instead
    of 5 and drops the factor of 4 inside the log (so just log(t)); both
    refinements come from the same union-bound improvement. β = 2.01 is the
    smallest value strictly above 2 we use in practice.
    """
    if n_samples <= 0:
        return np.inf

    return scale_param * np.sqrt(tail_param * np.log(t) / n_samples)


@njit
def compute_sn_ucb(z_normalized: float, n_samples: int, exploration_bound: float) -> float:
    """
    SN-UCB upper bound (paper §4, eq. for Û_g):

        Û_g(t; β) = Ẑ_g/√n + [1 + |Ẑ_g|/√n] · τ(n, t; β).
    """
    if n_samples <= 0:
        return np.inf

    return z_normalized + (1 + abs(z_normalized)) * exploration_bound


@njit
def sn_ucb_algorithm(
    data: np.ndarray,
    n_rounds: int,
    n_arms: int,
    burn_in: int = 2,
    scale_param: float = 4.5,
    tail_param: float = 2.01,
) -> np.ndarray:
    """
    Run the Self-Normalized UCB algorithm.
    
    Args:
        data: (n_rounds, n_arms) array of potential observations
        n_rounds: total number of rounds
        n_arms: number of arms
        burn_in: number of initial samples per arm (default 2 from Assumption 1)
        scale_param: exploration scale parameter
        tail_param: exploration tail parameter
    
    Returns:
        history: (n_rounds, n_arms) binary allocation matrix
    """
    history = np.zeros((n_rounds, n_arms), dtype=np.int32)
    t = 0
    
    # Step 1: Burn-in phase - sample each arm 'burn_in' times
    for round_idx in range(min(n_arms * burn_in, n_rounds)):
        arm = round_idx % n_arms
        history[t, arm] = 1
        t += 1
    
    # Step 2: SN-UCB selection phase
    while t < n_rounds:
        ucb_values = np.zeros(n_arms)
        
        for arm in range(n_arms):
            # Count samples for this arm
            n_samples = np.sum(history[:t, arm])
            
            if n_samples == 0:
                ucb_values[arm] = np.inf
                continue
            
            # Compute self-normalized statistic
            z_stat = compute_self_normalized_statistic(data[:t], history[:t], arm)
            z_normalized = z_stat / np.sqrt(n_samples)
            
            # Compute exploration bound
            exploration_bound = compute_exploration_bound(n_samples, t, scale_param, tail_param)
            
            # Compute SN-UCB value
            ucb_values[arm] = compute_sn_ucb(z_normalized, n_samples, exploration_bound)
        
        # Select arm with highest UCB value
        selected_arm = np.argmax(ucb_values)
        history[t, selected_arm] = 1
        t += 1
    
    return history


class SNUCBExperiment:
    """
    Class for running SN-UCB experiments with different configurations.
    """
    
    def __init__(self, n_arms: int, n_rounds: int, seed: Optional[int] = None):
        """
        Initialize SN-UCB experiment.
        
        Args:
            n_arms: number of arms
            n_rounds: number of rounds
            seed: random seed for reproducibility
        """
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.rng = np.random.default_rng(seed)
    
    def generate_gaussian_data(self, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian data for multi-armed bandit.
        
        Args:
            means: array of arm means
            variances: array of arm variances
        
        Returns:
            data: (n_rounds, n_arms) array of potential observations
        """
        assert len(means) == self.n_arms
        assert len(variances) == self.n_arms
        
        data = np.zeros((self.n_rounds, self.n_arms))
        for arm in range(self.n_arms):
            data[:, arm] = self.rng.normal(means[arm], np.sqrt(variances[arm]), self.n_rounds)
        
        return data
    
    def run_experiment(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        burn_in: int = 2,
        scale_param: float = 5.0,
        tail_param: float = 2.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single SN-UCB experiment.
        
        Args:
            means: array of arm means
            variances: array of arm variances
            burn_in: burn-in samples per arm
            scale_param: exploration scale parameter
            tail_param: exploration tail parameter
        
        Returns:
            data: generated data
            history: allocation history
        """
        data = self.generate_gaussian_data(means, variances)
        history = sn_ucb_algorithm(data, self.n_rounds, self.n_arms, burn_in, scale_param, tail_param)
        return data, history
    
    def compute_cumulative_reward(self, data: np.ndarray, history: np.ndarray) -> float:
        """
        Compute cumulative reward (sum of all observations).
        
        Args:
            data: generated data
            history: allocation history
        
        Returns:
            Total cumulative reward
        """
        return np.sum(data * history)
    
    def compute_arm_statistics(self, data: np.ndarray, history: np.ndarray) -> dict:
        """
        Compute statistics for each arm.
        
        Args:
            data: generated data
            history: allocation history
        
        Returns:
            Dictionary with arm statistics
        """
        stats = {}
        for arm in range(self.n_arms):
            arm_mask = history[:, arm] == 1
            if np.sum(arm_mask) > 0:
                arm_data = data[arm_mask, arm]
                stats[f'arm_{arm}'] = {
                    'n_samples': len(arm_data),
                    'mean': np.mean(arm_data),
                    'std': np.std(arm_data),
                    'total_reward': np.sum(arm_data)
                }
            else:
                stats[f'arm_{arm}'] = {
                    'n_samples': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'total_reward': 0.0
                }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test with simple 2-arm bandit
    n_arms = 2
    n_rounds = 1000
    means = np.array([0.3, 0.0])  # First arm has positive mean
    variances = np.array([1.0, 1.0])
    
    experiment = SNUCBExperiment(n_arms, n_rounds, seed=42)
    data, history = experiment.run_experiment(means, variances)
    
    print("SN-UCB Algorithm Test Results:")
    print(f"Total rounds: {n_rounds}")
    print(f"Arm allocation counts: {np.sum(history, axis=0)}")
    print(f"Cumulative reward: {experiment.compute_cumulative_reward(data, history):.3f}")
    
    stats = experiment.compute_arm_statistics(data, history)
    for arm_name, arm_stats in stats.items():
        print(f"{arm_name}: {arm_stats}")
