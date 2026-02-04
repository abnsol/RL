"""
Gymnasium-compatible environment wrapper for The Last Signal.
Enables seamless integration with standard RL libraries like Stable-Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Dict

from game_engine import GameEngine, Action, RewardVector
from game_config import GameConfig


class TheLastSignalEnv(gym.Env):
    """
    Gymnasium environment for The Last Signal.
    
    Observation Space:
        - Local map (7x7x3): signal presence, hazard probability, agent position
        - Agent state (4): normalized x, y, health, time
        - Total: 151-dimensional vector
    
    Action Space:
        - 0: MOVE_UP
        - 1: MOVE_DOWN
        - 2: MOVE_LEFT
        - 3: MOVE_RIGHT
        - 4: STABILIZE
        - 5: WAIT
    
    Reward:
        - Multi-objective reward vector with components:
          - signal_collection: +10 for collecting signals
          - hazard_damage: negative based on damage taken
          - time_cost: -0.01 per step
          - stabilization: +0.5 for proactive hazard reduction
          - exploration: +0.1 for visiting new cells
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(self, config: GameConfig = None, render_mode: str = None):
        """
        Initialize The Last Signal environment.
        
        Args:
            config: GameConfig object. Uses defaults if None.
            render_mode: How to render ('human' for text, None for headless)
        """
        if config is None:
            config = GameConfig()
        
        self.config = config
        self.render_mode = render_mode
        self.game = GameEngine(config)
        
        # Define action space
        self.action_space = spaces.Discrete(6)  # 6 discrete actions
        
        # Calculate observation space size
        radius = config.observation_radius
        obs_size = 2 * radius + 1
        local_map_size = obs_size * obs_size * 3  # 7x7x3 = 147
        agent_state_size = 4  # x, y, health, time
        total_obs_size = local_map_size + agent_state_size
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        self.reward_history = []
        self.last_reward_vector = None
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
            self.game.rng = np.random.RandomState(seed)
            self.game._generate_world()
        
        obs = self.game.reset()
        self.reward_history = []
        self.last_reward_vector = None
        
        info = {
            "episode_info": {
                "health": self.game.health,
                "time_remaining": self.game.time_remaining,
                "signals_available": sum(1 for y in range(self.config.grid_height) 
                                         for x in range(self.config.grid_width) 
                                         if self.game.grid[y][x].has_signal),
            }
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-5)
        
        Returns:
            observation: Current observation
            reward: Scalar reward (sum of reward vector)
            terminated: Whether episode ended
            truncated: Whether episode was truncated (always False here)
            info: Additional information including reward vector
        """
        if action not in [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid action: {action}")
        
        obs, reward_vector, terminated, info = self.game.step(Action(action))
        
        # Store reward vector for analysis
        self.last_reward_vector = reward_vector
        self.reward_history.append({
            "step": self.game.episode_step,
            "signal_collection": reward_vector.signal_collection,
            "hazard_damage": reward_vector.hazard_damage,
            "time_cost": reward_vector.time_cost,
            "stabilization": reward_vector.stabilization,
            "exploration": reward_vector.exploration,
            "total": reward_vector.total,
        })
        
        # Scalar reward for training
        scalar_reward = reward_vector.total
        
        # Augment info with reward breakdown
        info["reward_vector"] = {
            "signal_collection": float(reward_vector.signal_collection),
            "hazard_damage": float(reward_vector.hazard_damage),
            "time_cost": float(reward_vector.time_cost),
            "stabilization": float(reward_vector.stabilization),
            "exploration": float(reward_vector.exploration),
        }
        
        if self.render_mode == "human":
            self.render()
        
        return obs, scalar_reward, terminated, False, info
    
    def render(self) -> None:
        """Render the game state."""
        if self.render_mode == "human":
            self.game.render()
    
    def get_reward_history(self):
        """Return the history of rewards for this episode."""
        return self.reward_history
    
    def get_world_map(self) -> np.ndarray:
        """
        Get full world state for offline analysis (not exposed to agent).
        Useful for debugging and visualization.
        """
        world_map = np.zeros((self.config.grid_height, self.config.grid_width, 3), 
                            dtype=np.float32)
        
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                cell = self.game.grid[y][x]
                world_map[y, x, 0] = float(cell.has_signal and not cell.signal_collected)
                world_map[y, x, 1] = cell.hazard_probability
                world_map[y, x, 2] = 1.0 if (x, y) == (self.game.agent_x, self.game.agent_y) else 0.0
        
        return world_map


# Register the environment with Gymnasium
gym.register(
    id="TheLastSignal-v0",
    entry_point="the_last_signal_env:TheLastSignalEnv",
    max_episode_steps=200,
)
