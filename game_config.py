"""
Configuration parameters for The Last Signal environment.
Allows for easy difficulty scaling and customization.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class GameConfig:
    """Configuration for The Last Signal environment."""
    
    # World parameters
    grid_width: int = 30
    grid_height: int = 30
    
    # Agent parameters
    initial_health: int = 100
    max_health: int = 100
    time_budget: int = 200
    
    # Signal nodes
    num_signals: int = 5
    signal_value: float = 10.0
    
    # Hazards
    base_hazard_probability: float = 0.1
    max_hazard_probability: float = 0.8
    hazard_damage_range: Tuple[int, int] = (5, 20)
    
    # Stability mechanics
    hazard_degradation_per_step: float = 0.02  # Hazard increases per step if not stabilized
    stabilize_reduction: float = 0.3  # How much STABILIZE action reduces hazard
    
    # Rewards
    signal_collection_reward: float = 10.0
    hazard_damage_penalty: float = -1.0  # Per point of damage
    time_penalty_per_step: float = -0.01
    stabilization_reward: float = 0.5  # Reward for proactive stabilization
    exploration_bonus: float = 0.1  # Small bonus for visiting new cells
    
    # Observation
    observation_radius: int = 3  # 3x3 grid around agent
    
    # Seed
    seed: int = None
