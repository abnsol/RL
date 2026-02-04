import numpy as np
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from the_last_signal_env import TheLastSignalEnv
from multi_objective_env import MultiObjectiveEnv
from game_config import GameConfig

class ComprehensiveLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.history = {
            "signals": [], "rewards": [], "steps": [],
            "health": [], "energy": [], "exploration": []
        }

    def _on_step(self) -> bool:
        # Check if any episodes finished in this step
        for info in self.locals.get("infos", []):
            if "episode_info" in info or "signals_collected" in info:
                pass
        
        
        dones = self.locals.get("dones")
        if dones[0]:
            info = self.locals["infos"][0]
            # Extract metrics from the 'info' dict provided by step()
            self.history["signals"].append(info.get("signals_collected", 0))
            self.history["health"].append(info.get("health", 0))
            self.history["energy"].append(info.get("energy", 0))
            self.history["steps"].append(info.get("step", 0))
            # We get the reward from the rollout buffer
            self.history["rewards"].append(self.locals["rewards"][0])

        return True

    def print_final_report(self):
        if not self.history["signals"]:
            print("No episodes completed.")
            return

        print("\n" + "="*50)
        print("FINAL TRAINING REPORT FOR AI ANALYSIS")
        print("="*50)
        print(f"Total Episodes:  {len(self.history['signals'])}")
        print(f"Mean Signals:    {np.mean(self.history['signals']):.2f}")
        print(f"Max Signals:     {np.max(self.history['signals'])}")
        print(f"Mean Reward:     {np.mean(self.history['rewards']):.2f}")
        print(f"Mean Ep Length:  {np.mean(self.history['steps']):.1f}")
        print(f"Mean Final Health: {np.mean(self.history['health']):.1f}")
        print(f"Mean Final Energy: {np.mean(self.history['energy']):.1f}")
        print("-" * 50)
        
        # Save to JSON
        with open("detailed_logs.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f)

def run_step3_experiment():
    config = GameConfig()
    config.num_signals = 15 
    
    weights = {
            "signal_collection": 50.0,   
            "exploration": 5.0,          
            "hazard_damage": 10.0,       
            "stabilization": 2.0,        
            "time_cost": 1.0,
            "energy_cost": 0.5,
        }

    env = MultiObjectiveEnv(TheLastSignalEnv(config=config), weights=weights)
    logger = ComprehensiveLogger()

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05)
    
    print("Starting training (100,000 steps)...")
    model.learn(total_timesteps=100000, callback=logger)
    
    logger.print_final_report()
    model.save("models/mo_agent_fixed")

if __name__ == "__main__":
    run_step3_experiment()