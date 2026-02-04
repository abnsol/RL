"""
Multi-objective reward wrapper for The Last Signal.
Enables configurable weighted scalarization of reward components.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple
from the_last_signal_env import TheLastSignalEnv
from game_config import GameConfig


class MultiObjectiveEnv(gym.Wrapper):
    """
    Wrapper that applies weighted scalarization to reward components.
    
    Args:
        env: Base environment (TheLastSignalEnv)
        weights: Dictionary mapping reward component names to weights
                 Default uses equal weighting
    
    Example:
        weights = {
            "signal_collection": 2.0,   # Prioritize signal collection
            "hazard_damage": 1.0,       # Standard penalty
            "time_cost": 0.5,           # Less concern for time
            "stabilization": 1.0,
            "exploration": 0.5,
            "energy_cost": 1.0,
        }
    """
    
    def __init__(self, env: TheLastSignalEnv, weights: Dict[str, float] = None):
        super().__init__(env)
        
        # Default weights: uniform (all equally important)
        self.default_weights = {
            "signal_collection": 50.0,   # ⬆️ Increase signal reward
            "exploration": 5.0,          # ⬇️ Reduce exploration (was 40.0)
            "hazard_damage": 10.0,       # ⬆️ Increase damage penalty (was 0.5)
            "stabilization": 2.0,        # ⬆️ Increase survival reward (was 0.05)
            "time_cost": 1.0,
            "energy_cost": 0.5,
        }

        
        self.weights = weights if weights is not None else self.default_weights
        
        # Logging
        self.episode_rewards = []
        self.episode_components = {key: [] for key in self.default_weights.keys()}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step with weighted reward scalarization.
        """
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract reward components
        reward_vector = info["reward_vector"]
        
        # Apply weighted scalarization
        weighted_reward = sum(
            self.weights.get(component, 1.0) * value
            for component, value in reward_vector.items()
        )
        
        # Store components for analysis
        for component, value in reward_vector.items():
            self.episode_components[component].append(value)
        
        # Augment info with weighted reward
        info["weighted_reward"] = weighted_reward
        info["weights"] = self.weights
        
        return obs, weighted_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset and clear episode tracking."""
        self.episode_rewards = []
        self.episode_components = {key: [] for key in self.default_weights.keys()}
        return self.env.reset(**kwargs)
    
    def get_episode_statistics(self) -> Dict:
        """Return statistics for the last episode."""
        stats = {
            "total_reward": sum(self.episode_rewards),
            "component_totals": {
                component: sum(values) 
                for component, values in self.episode_components.items()
            },
            "component_means": {
                component: np.mean(values) if values else 0.0
                for component, values in self.episode_components.items()
            },
        }
        return stats