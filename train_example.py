"""
Example training script for The Last Signal using Stable-Baselines3.
Demonstrates how to use the environment with standard RL algorithms.
"""

import gymnasium as gym
import numpy as np
from the_last_signal_env import TheLastSignalEnv
from game_config import GameConfig

# Optional: Install with `pip install stable-baselines3[extra]`
try:
    from stable_baselines3 import PPO, DQN, A2C
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Note: Stable-Baselines3 not installed. Run: pip install stable-baselines3[extra]")


def create_env(config: GameConfig = None):
    """Create a The Last Signal environment."""
    return TheLastSignalEnv(config=config, render_mode=None)


def random_baseline(num_episodes: int = 10):
    """
    Run random policy baseline for comparison.
    """
    print("\n" + "="*60)
    print("Random Baseline")
    print("="*60)
    
    env = create_env()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    env.close()
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    return episode_rewards


def heuristic_agent(num_episodes: int = 10):
    """
    Run a simple heuristic agent that tries to:
    1. Collect nearby signals
    2. Avoid high-hazard cells
    3. Stabilize dangerous cells
    """
    print("\n" + "="*60)
    print("Heuristic Agent (Simple Strategy)")
    print("="*60)
    
    env = create_env()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Simple heuristic: look for signals or stabilize hazards
            # (This is a minimal example - a real heuristic would be more sophisticated)
            action = env.action_space.sample()  # Replace with actual heuristic
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    env.close()
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    return episode_rewards


def train_with_sb3(total_timesteps: int = 10000, algorithm: str = "PPO"):
    """
    Train an RL agent using Stable-Baselines3.
    
    Args:
        total_timesteps: Total number of environment steps
        algorithm: Which algorithm to use ('PPO', 'DQN', or 'A2C')
    """
    if not HAS_SB3:
        print("Stable-Baselines3 not installed. Install with:")
        print("  pip install stable-baselines3[extra]")
        return
    
    print("\n" + "="*60)
    print(f"Training with {algorithm}")
    print("="*60)
    
    env = create_env()
    
    # Create agent
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    elif algorithm == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4)
    elif algorithm == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    eval_episodes = 5
    episode_rewards = []
    
    for episode in range(eval_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Eval Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    print(f"\nAverage Eval Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    # Save model
    model.save(f"the_last_signal_{algorithm}")
    print(f"Model saved as 'the_last_signal_{algorithm}'")
    
    env.close()
    return episode_rewards


def multi_objective_analysis(num_episodes: int = 5):
    """
    Analyze the multi-objective reward structure.
    """
    print("\n" + "="*60)
    print("Multi-Objective Reward Analysis")
    print("="*60)
    
    env = create_env()
    
    reward_components = {
        "signal_collection": [],
        "hazard_damage": [],
        "time_cost": [],
        "stabilization": [],
        "exploration": [],
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        episode_components = {k: [] for k in reward_components.keys()}
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if "reward_vector" in info:
                for component in reward_components.keys():
                    episode_components[component].append(info["reward_vector"].get(component, 0))
            
            done = terminated or truncated
        
        # Aggregate episode results
        for component in reward_components.keys():
            total = sum(episode_components[component])
            reward_components[component].append(total)
            print(f"Episode {episode + 1} - {component}: {total:.2f}")
        print()
    
    # Print statistics
    print("Summary Statistics:")
    for component, values in reward_components.items():
        print(f"{component}:")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Std:  {np.std(values):.2f}")
        print(f"  Min:  {np.min(values):.2f}")
        print(f"  Max:  {np.max(values):.2f}")
    
    env.close()


if __name__ == "__main__":
    # Run baselines
    random_baseline(num_episodes=5)
    heuristic_agent(num_episodes=5)
    
    # Run multi-objective analysis
    multi_objective_analysis(num_episodes=3)
    
    # Train with Stable-Baselines3 (if available)
    if HAS_SB3:
        train_with_sb3(total_timesteps=100000, algorithm="PPO")
    else:
        print("\nTo train with Stable-Baselines3, install it with:")
        print("  pip install stable-baselines3[extra]")
        print("\nThen you can use:")
        print("  python train_example.py")
