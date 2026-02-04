"""
Core game logic for The Last Signal.
Manages world state, agent dynamics, hazards, and reward calculation.
"""

import numpy as np
from enum import IntEnum
from typing import Tuple, Dict, List
from dataclasses import dataclass, field
from game_config import GameConfig


class CellType(IntEnum):
    """Types of cells in the game world."""
    EMPTY = 0
    SIGNAL_NODE = 1
    HAZARD = 2


class Action(IntEnum):
    """Available actions for the agent."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STABILIZE = 4
    WAIT = 5


@dataclass
class Cell:
    """Represents a single cell in the grid."""
    cell_type: CellType = CellType.EMPTY
    hazard_probability: float = 0.0  # Current hazard level
    has_signal: bool = False
    signal_collected: bool = False
    visited: bool = False


@dataclass
class RewardVector:
    """Multi-reward vector for RL training."""
    signal_collection: float = 0.0
    hazard_damage: float = 0.0
    time_cost: float = 0.0
    stabilization: float = 0.0
    exploration: float = 0.0
    
    @property
    def total(self) -> float:
        """Compute scalar reward (can be overridden for different objectives)."""
        return (self.signal_collection + self.hazard_damage + self.time_cost + 
                self.stabilization + self.exploration)


class GameEngine:
    """Core engine managing The Last Signal game logic."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Game state
        self.grid: List[List[Cell]] = []
        self.agent_x = 0
        self.agent_y = 0
        self.health = config.initial_health
        self.time_remaining = config.time_budget
        self.episode_step = 0
        self.visited_cells = set()
        
        # Initialize world
        self._generate_world()
    
    def _generate_world(self):
        """Procedurally generate the game world."""
        self.grid = [[Cell() for _ in range(self.config.grid_width)] 
                     for _ in range(self.config.grid_height)]
        
        # Place agent
        self.agent_x = self.config.grid_width // 2
        self.agent_y = self.config.grid_height // 2
        
        # Place signal nodes randomly
        signal_count = 0
        while signal_count < self.config.num_signals:
            x = self.rng.randint(0, self.config.grid_width)
            y = self.rng.randint(0, self.config.grid_height)
            if not self.grid[y][x].has_signal:
                self.grid[y][x].has_signal = True
                self.grid[y][x].cell_type = CellType.SIGNAL_NODE
                signal_count += 1
        
        # Initialize hazard levels across map
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                if self.grid[y][x].cell_type != CellType.SIGNAL_NODE:
                    # Random initial hazard level
                    self.grid[y][x].hazard_probability = self.rng.uniform(
                        0, self.config.base_hazard_probability
                    )
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height
    
    def _apply_hazard(self) -> float:
        """Apply hazard damage at current position if triggered."""
        cell = self.grid[self.agent_y][self.agent_x]
        damage = 0
        
        # Hazard triggers with cell's probability
        if self.rng.rand() < cell.hazard_probability:
            damage = self.rng.randint(*self.config.hazard_damage_range)
            self.health = max(0, self.health - damage)
        
        return float(damage)
    
    def step(self, action: Action) -> Tuple[np.ndarray, RewardVector, bool, Dict]:
        """
        Execute one step in the environment.
        
        Returns:
            observation: Current observation
            reward: RewardVector with breakdown
            terminated: Whether episode ended
            info: Additional information
        """
        self.episode_step += 1
        self.time_remaining -= 1
        reward = RewardVector()
        
        # Handle movement actions
        new_x, new_y = self.agent_x, self.agent_y
        
        if action == Action.MOVE_UP:
            new_y = max(0, self.agent_y - 1)
        elif action == Action.MOVE_DOWN:
            new_y = min(self.config.grid_height - 1, self.agent_y + 1)
        elif action == Action.MOVE_LEFT:
            new_x = max(0, self.agent_x - 1)
        elif action == Action.MOVE_RIGHT:
            new_x = min(self.config.grid_width - 1, self.agent_x + 1)
        elif action == Action.STABILIZE:
            # Reduce hazard at current cell
            cell = self.grid[self.agent_y][self.agent_x]
            reduction = self.config.stabilize_reduction * cell.hazard_probability
            cell.hazard_probability = max(0, cell.hazard_probability - reduction)
            reward.stabilization = self.config.stabilization_reward
        elif action == Action.WAIT:
            # No movement
            pass
        
        # Update position if movement happened
        if (new_x, new_y) != (self.agent_x, self.agent_y):
            self.agent_x, self.agent_y = new_x, new_y
            
            # Check for signal collection
            cell = self.grid[self.agent_y][self.agent_x]
            if cell.has_signal and not cell.signal_collected:
                cell.signal_collected = True
                reward.signal_collection = self.config.signal_collection_reward
            
            # Track exploration
            if (self.agent_x, self.agent_y) not in self.visited_cells:
                self.visited_cells.add((self.agent_x, self.agent_y))
                reward.exploration = self.config.exploration_bonus
        
        # Apply hazard damage
        damage = self._apply_hazard()
        reward.hazard_damage = damage * self.config.hazard_damage_penalty
        
        # Apply time penalty
        reward.time_cost = self.config.time_penalty_per_step
        
        # Degrade hazards in all cells (increases danger over time)
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                cell = self.grid[y][x]
                if cell.hazard_probability < self.config.max_hazard_probability:
                    cell.hazard_probability = min(
                        self.config.max_hazard_probability,
                        cell.hazard_probability + self.config.hazard_degradation_per_step
                    )
        
        # Check termination
        terminated = self.health <= 0 or self.time_remaining <= 0
        
        # Generate observation
        observation = self._get_observation()
        
        info = {
            "step": self.episode_step,
            "health": self.health,
            "time_remaining": self.time_remaining,
            "signals_collected": sum(1 for y in range(self.config.grid_height) 
                                     for x in range(self.config.grid_width) 
                                     if self.grid[y][x].signal_collected),
            "damage_taken": damage,
        }
        
        return observation, reward, terminated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate agent's observation.
        Returns partial observability: local neighborhood + agent state.
        """
        radius = self.config.observation_radius
        obs_size = 2 * radius + 1
        
        # Local map (3x3 grid around agent)
        local_map = np.zeros((obs_size, obs_size, 3), dtype=np.float32)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = self.agent_y + dy
                x = self.agent_x + dx
                
                if self._is_valid_position(x, y):
                    cell = self.grid[y][x]
                    local_map[dy + radius, dx + radius, 0] = float(cell.has_signal and not cell.signal_collected)
                    local_map[dy + radius, dx + radius, 1] = cell.hazard_probability
                    local_map[dy + radius, dx + radius, 2] = 1.0 if (x, y) == (self.agent_x, self.agent_y) else 0.0
        
        # Agent state vector
        agent_state = np.array([
            self.agent_x / self.config.grid_width,
            self.agent_y / self.config.grid_height,
            self.health / self.config.max_health,
            self.time_remaining / self.config.time_budget,
        ], dtype=np.float32)
        
        # Combine observations
        obs = np.concatenate([local_map.flatten(), agent_state])
        return obs
    
    def reset(self):
        """Reset game to initial state."""
        self.health = self.config.initial_health
        self.time_remaining = self.config.time_budget
        self.episode_step = 0
        self.visited_cells = set()
        self._generate_world()
        return self._get_observation()
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the game state to console (text-based).
        """
        print("\n" + "=" * 60)
        print(f"Step: {self.episode_step} | Health: {self.health} | Time: {self.time_remaining}")
        print("=" * 60)
        
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                if (x, y) == (self.agent_x, self.agent_y):
                    print("A", end=" ")
                elif self.grid[y][x].has_signal and not self.grid[y][x].signal_collected:
                    print("S", end=" ")
                elif self.grid[y][x].hazard_probability > 0.5:
                    print("X", end=" ")
                elif self.grid[y][x].hazard_probability > 0.2:
                    print("~", end=" ")
                else:
                    print(".", end=" ")
            print()
        print("=" * 60)
