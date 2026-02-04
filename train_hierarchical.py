import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from the_last_signal_env import TheLastSignalEnv
from hierarchical_env import HierarchicalRewardEnv
from game_config import GameConfig
import os

class HierarchicalLogger(BaseCallback):
    """Custom logger to track hierarchical performance metrics."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.history = {"signals": [], "health": [], "energy": [], "steps": [], "rewards": []}

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            info = self.locals["infos"][0]
            self.history["signals"].append(info.get("signals_collected", 0))
            self.history["health"].append(info.get("health", 0))
            self.history["energy"].append(info.get("energy", 0))
            self.history["steps"].append(info.get("step", 0))
            self.history["rewards"].append(self.locals["rewards"][0])
        return True

def train():
    # Setup configuration
    config = GameConfig()
    config.num_signals = 15
    
    # Initialize Hierarchical Environment
    base_env = TheLastSignalEnv(config=config)
    env = HierarchicalRewardEnv(base_env)
    
    logger = HierarchicalLogger()

    # Train PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        ent_coef=0.05, 
        learning_rate=3e-4,
        tensorboard_log="./logs/hrl_training/"
    )

    print("\n" + "="*50)
    print("STARTING HIERARCHICAL TRAINING")
    print("="*50)
    
    model.learn(total_timesteps=100000, callback=logger)
    
    # Final Report
    print("\n" + "="*50)
    print("FINAL HIERARCHICAL TRAINING REPORT")
    print("="*50)
    print(f"Mean Signals:    {np.mean(logger.history['signals']):.2f}")
    print(f"Max Signals:     {np.max(logger.history['signals'])}")
    print(f"Mean Final Health: {np.mean(logger.history['health']):.1f}")
    print(f"Mean Final Energy: {np.mean(logger.history['energy']):.1f}")
    print(f"Mean Ep Length:  {np.mean(logger.history['steps']):.1f}")
    print("="*50 + "\n")

    os.makedirs("models", exist_ok=True)
    model.save("models/hrl_agent_v1")

if __name__ == "__main__":
    train()