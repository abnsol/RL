# The Last Signal - RL Environment

A sophisticated 2D grid-based reinforcement learning environment designed for strategic decision-making research. The game prioritizes RL-relevant complexity over reflex mechanics, featuring multi-objective rewards, stochastic hazards, and long-horizon planning challenges.

## Features

- **Grid-Based World**: Procedurally generated 2D grid with configurable size (10×10 to 20×20)
- **Rich Agent Dynamics**: Health system, time budget, and strategic decision-making
- **Hazard System**: Probabilistic damage with stability mechanics
- **Multi-Objective Rewards**: Separate reward components for signal collection, hazard avoidance, stabilization, and exploration
- **Partial Observability**: Agent observes local neighborhood (3×3 grid) + internal state
- **Gymnasium Compatible**: Seamless integration with Stable-Baselines3 and other standard RL libraries
- **Optional Pygame Rendering**: Beautiful 2D visualization (optional)
- **Fully Modular**: Clean separation between game logic, environment wrapper, and visualization

## Installation with uv
```bash

#uv initialization
uv init

#create a venv for your project
uv venv --python 3.12

source .venv/bin/activate
```


### With Stable-Baselines3 (for training)
```bash
uv pip install gymnasium numpy pygame stable-baselines3
```

## Quick Start

### 1. Run the Interactive Menu
```bash
python main.py
```

Choose from:
- Random baseline comparison
- Multi-objective reward analysis
- Interactive play with Pygame
- Train with PPO
- Custom environment testing

### 2. Use as Gymnasium Environment
```python
import gymnasium as gym
from the_last_signal_env import TheLastSignalEnv

# Create environment
env = TheLastSignalEnv()

# Standard Gymnasium API
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 3. Train with Stable-Baselines3
```python
from stable_baselines3 import PPO
from the_last_signal_env import TheLastSignalEnv

env = TheLastSignalEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
```

## Game Mechanics

### Action Space (Discrete, 6 actions)
- **MOVE_UP** (0): Move up one cell
- **MOVE_DOWN** (1): Move down one cell
- **MOVE_LEFT** (2): Move left one cell
- **MOVE_RIGHT** (3): Move right one cell
- **STABILIZE** (4): Reduce hazard probability in current cell
- **WAIT** (5): No movement, just advance time

### Observation Space
- **Local Map** (7×7×3): Neighborhood around agent showing:
  - Signal node presence (channel 0)
  - Hazard probability (channel 1)
  - Agent position (channel 2)
- **Agent State** (4D vector):
  - Normalized X position
  - Normalized Y position
  - Health / Max Health
  - Time Remaining / Time Budget

Total: 151-dimensional continuous observation

### Reward Vector (Multi-Objective)
The environment emits rewards broken into 5 components:

1. **Signal Collection** (+10 per signal): Collected when visiting signal nodes
2. **Hazard Damage** (-1 per damage): Proportional to damage taken
3. **Time Cost** (-0.01 per step): Encourages efficiency
4. **Stabilization** (+0.5): Reward for proactive hazard reduction
5. **Exploration** (+0.1): Bonus for visiting new cells

Access individual components via `info["reward_vector"]`:
```python
_, reward, _, _, info = env.step(action)
print(info["reward_vector"])
# {
#     "signal_collection": 10.0,
#     "hazard_damage": -15.0,
#     "time_cost": -0.01,
#     "stabilization": 0.0,
#     "exploration": 0.1
# }
```

### World Dynamics

**Hazard Progression**:
- Hazards increase over time (+0.02 per step) if not stabilized
- **STABILIZE** action reduces hazard by 30% in current cell
- Hazards trigger damage with their current probability

**Signal Nodes**:
- Fixed at episode start, procedurally positioned
- Non-renewable (once collected, unavailable)
- No explicit "win" condition; collecting all signals is optional

**Degradation**:
- Environment becomes increasingly dangerous as time passes
- Encourages strategic risk-taking and exploration

## Configuration

Edit `game_config.py` to customize difficulty:

```python
from game_config import GameConfig

config = GameConfig(
    grid_width=15,
    grid_height=15,
    initial_health=150,
    time_budget=300,
    num_signals=8,
    base_hazard_probability=0.15,
    signal_collection_reward=15.0,
)

env = TheLastSignalEnv(config=config)
```

### Key Parameters
- `grid_width`, `grid_height`: World size
- `initial_health`: Starting health
- `time_budget`: Steps per episode
- `num_signals`: Signal nodes to place
- `base_hazard_probability`: Initial hazard level
- `hazard_degradation_per_step`: How fast hazards increase
- `stabilize_reduction`: Effectiveness of stabilization
- Various reward scales

## Learning Challenges

The environment naturally induces:

1. **Long-Horizon Planning**: Must balance exploration with time constraints
2. **Exploration vs Exploitation**: Finding signals vs avoiding hazards
3. **Risk Sensitivity**: When to stabilize vs move on
4. **Multi-Objective Trade-offs**: No single "correct" strategy
5. **Stochasticity Robustness**: Handling probabilistic hazards

## Visualization

### Interactive Play
```python
from game_renderer import interactive_play
interactive_play()
```

**Controls**:
- Arrow Keys: Move
- S: Stabilize
- W: Wait
- ESC: Exit

### Headless Mode (No Rendering)
Set `render_mode=None` in `TheLastSignalEnv` for faster training.

## Project Structure

```
.
├── game_config.py          # Configuration parameters
├── game_engine.py          # Core game logic (MDP dynamics)
├── the_last_signal_env.py  # Gymnasium wrapper
├── game_renderer.py        # Pygame visualization
├── train_example.py        # Training examples with SB3
├── main.py                 # Interactive menu
└── README.md               # This file
```

## API Reference

### TheLastSignalEnv

```python
class TheLastSignalEnv(gym.Env):
    def __init__(self, config: GameConfig = None, render_mode: str = None)
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]
    def render(self) -> None
    def get_reward_history(self) -> List[Dict]
    def get_world_map(self) -> np.ndarray  # Full world state (debugging)
```

### GameEngine

```python
class GameEngine:
    def __init__(self, config: GameConfig)
    def step(self, action: Action) -> Tuple[np.ndarray, RewardVector, bool, Dict]
    def reset(self) -> np.ndarray
    def render(self, mode: str = 'human') -> None
```

## Example Workflows

### 1. Baseline Comparison
```bash
python -c "from train_example import random_baseline, heuristic_agent; random_baseline(10); heuristic_agent(10)"
```

### 2. Multi-Objective Analysis
```bash
python -c "from train_example import multi_objective_analysis; multi_objective_analysis(5)"
```

### 3. Custom Training Loop
```python
from the_last_signal_env import TheLastSignalEnv
import numpy as np

env = TheLastSignalEnv()
obs, _ = env.reset()

for step in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Log multi-objective rewards
    print(f"Reward components: {info['reward_vector']}")
    
    if terminated or truncated:
        obs, _ = env.reset()
```

## Extending the Environment

### Add Custom Reward
Modify `RewardVector` in `game_engine.py` and `step()` method.

### Change World Generation
Override `_generate_world()` in `GameEngine` for custom procedural generation.

### Implement Custom Observation
Modify `_get_observation()` to change partial observability.

### Add New Actions
Extend `Action` enum and add handling in `GameEngine.step()`.

## Performance Notes

- **Headless Mode** (~50-100 episodes/second on CPU)
- **Pygame Rendering** (~4-10 FPS at 30px cells)
- Observation space is small (~150D), suitable for any RL algorithm

## Troubleshooting

**ImportError: No module named 'pygame'**
```bash
pip install pygame
```

**ImportError: No module named 'gymnasium'**
```bash
pip install gymnasium
```

**ImportError: No module named 'stable_baselines3'**
```bash
pip install stable-baselines3[extra]
```

## References

The environment design supports:
- **Standard RL**: DQN, PPO, A2C, etc.
- **Multi-Objective RL**: Pareto optimization algorithms
- **Curriculum Learning**: Gradual difficulty scaling via GameConfig
- **Model-Based RL**: Clean MDP structure enables planning

## License

Public domain - Use freely for research and education.

## Citation

If using The Last Signal in research, please cite:
```
The Last Signal - An RL-Focused Game Environment
A 2D strategic decision-making environment for reinforcement learning research.
```

---

**Happy researching!** 🚀
