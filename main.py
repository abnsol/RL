"""
The Last Signal - Main entry point
A 2D RL-focused game environment for strategic decision-making research.
"""

import sys
from game_config import GameConfig
from the_last_signal_env import TheLastSignalEnv
import gymnasium as gym


def main():
    """Main entry point with menu system."""
    
    print("\n" + "="*60)
    print("THE LAST SIGNAL - RL Environment")
    print("="*60)
    print("\nSelect mode:")
    print("1. Run random baseline (5 episodes)")
    print("2. Multi-objective reward analysis")
    print("3. Interactive play (requires Pygame)")
    print("4. Play saved model (Pygame)")
    print("4. Train with Stable-Baselines3 (PPO)")
    print("5. Custom environment test")
    print("0. Exit")
    print("-"*60)
    
    choice = input("Enter choice (0-5): ").strip()
    
    if choice == "1":
        run_random_baseline()
    elif choice == "2":
        run_multi_objective_analysis()
    elif choice == "3":
        run_interactive_play()
    elif choice == "4":
        run_play_model()
    elif choice == "4":
        run_sb3_training()
    elif choice == "5":
        run_custom_test()
    elif choice == "0":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")


def run_random_baseline():
    """Run a simple random baseline."""
    print("\nRunning random baseline (5 episodes)...\n")
    
    env = TheLastSignalEnv()
    total_rewards = []
    
    for ep in range(5):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print(f"\nAverage Reward: {sum(total_rewards)/len(total_rewards):.2f}")


def run_multi_objective_analysis():
    """Analyze multi-objective reward structure."""
    print("\nAnalyzing multi-objective rewards (3 episodes)...\n")
    
    env = TheLastSignalEnv()
    
    reward_stats = {
        "signal_collection": {"values": []},
        "hazard_damage": {"values": []},
        "time_cost": {"values": []},
        "stabilization": {"values": []},
        "exploration": {"values": []},
    }
    
    for ep in range(3):
        obs, info = env.reset()
        done = False
        episode_rewards = {k: 0 for k in reward_stats.keys()}
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if "reward_vector" in info:
                for component in reward_stats.keys():
                    episode_rewards[component] += info["reward_vector"].get(component, 0)
            
            done = terminated or truncated
        
        print(f"Episode {ep+1}:")
        for component, value in episode_rewards.items():
            print(f"  {component}: {value:.2f}")
            reward_stats[component]["values"].append(value)
        print()


def run_interactive_play():
    """Run interactive gameplay with Pygame."""
    try:
        import pygame
        from game_renderer import interactive_play
        print("\nStarting interactive game...")
        print("Controls: Arrow Keys = Move, S = Stabilize, W = Wait, ESC = Exit\n")
        interactive_play()
    except ImportError:
        print("Pygame not installed. Install with: pip install pygame")


def run_play_model():
    """Load a saved Stable-Baselines3 model and run it with the visualizer."""
    try:
        import pygame
        from play_model import play_model

        model = input("Model path (default the_last_signal_PPO): ").strip() or "the_last_signal_PPO"
        algo = input("Algorithm (PPO/DQN/A2C, default PPO): ").strip() or "PPO"
        fs = input("Fullscreen? (y/N): ").strip().lower() == 'y'

        print("Starting model playback...")
        play_model(model, algo=algo, fullscreen=fs)
    except ImportError:
        print("Required packages missing. Install with: pip install stable-baselines3 pygame")
    except Exception as e:
        print(f"Error launching model playback: {e}")


def run_sb3_training():
    """Train with Stable-Baselines3."""
    try:
        from stable_baselines3 import PPO
        
        print("\nTraining PPO agent on The Last Signal...")
        print("This will take a moment...\n")
        
        env = TheLastSignalEnv()
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        model.learn(total_timesteps=5000)
        
        print("\nEvaluating trained agent (3 episodes)...\n")
        
        for ep in range(3):
            obs, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
        
        # Save model
        model.save("the_last_signal_ppo")
        print("\nModel saved as 'the_last_signal_ppo'")
        
        env.close()
        
    except ImportError:
        print("Stable-Baselines3 not installed. Install with:")
        print("  pip install stable-baselines3[extra]")


def run_custom_test():
    """Test with custom configuration."""
    print("\nCustom Environment Test")
    print("-"*40)
    
    # Create custom config
    config = GameConfig()
    
    print("Enter custom parameters (or press Enter to use defaults):")
    
    try:
        grid_size = input("Grid size (default 10): ").strip()
        if grid_size:
            config.grid_width = config.grid_height = int(grid_size)
        
        num_signals = input("Number of signals (default 5): ").strip()
        if num_signals:
            config.num_signals = int(num_signals)
        
        time_budget = input("Time budget (default 200): ").strip()
        if time_budget:
            config.time_budget = int(time_budget)
        
        initial_health = input("Initial health (default 100): ").strip()
        if initial_health:
            config.initial_health = int(initial_health)
        
    except ValueError:
        print("Invalid input, using defaults")
    
    print("\nRunning test with custom config...\n")
    
    env = TheLastSignalEnv(config=config)
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 50:  # Max 50 steps for quick test
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    print(f"Test completed!")
    print(f"Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Health: {info.get('health', 'N/A')}")
    print(f"Time Remaining: {info.get('time_remaining', 'N/A')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
