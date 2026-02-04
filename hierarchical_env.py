import gymnasium as gym
from the_last_signal_env import TheLastSignalEnv

class HierarchicalRewardEnv(gym.Wrapper):
    """
    A Hierarchical Reward Manager that switches objectives
    based on the agent's current health and energy levels.
    """
    def __init__(self, env: TheLastSignalEnv):
        super().__init__(env)
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        game = self.env.unwrapped.game
        reward_vec = info["reward_vector"]
        
        # SURVIVAL (Priority: Don't die)
        if game.health < 40:
            weights = {
                "signal_collection": 2.0,
                "exploration": 2.0,
                "hazard_damage": 15.0,    # High penalty for taking hits
                "stabilization": 30.0,    # High reward for proactive safety
                "time_cost": 1.0,
                "energy_cost": 1.0
            }
            mode = "SURVIVAL"

        # RECHARGE (Priority: Conserve energy)
        elif game.energy < 60:
            weights = {
                "signal_collection": 2.0,
                "exploration": 0.0,
                "hazard_damage": 1.0,
                "stabilization": 1.0,
                "time_cost": 1.0,
                "energy_cost": 25.0       # Movement becomes very expensive
            }
            mode = "RECHARGE"

        # Mode C: MISSION (Priority: Find signals)
        else:
            weights = {
                "signal_collection": 25.0, # Massive reward for signals
                "exploration": 45.0,      # Massive reward for moving to new tiles
                "hazard_damage": 0.5,      # Ignore minor damage
                "stabilization": 0.05,     # Nerf farming
                "time_cost": 1.0,
                "energy_cost": 0.5
            }
            mode = "MISSION"

        # 3. Calculate the hierarchical scalar reward
        hierarchical_reward = sum(weights.get(k, 1.0) * v for k, v in reward_vec.items())
        
        # Add metadata for logging/debugging
        info["active_mode"] = mode
        info["hierarchical_weights"] = weights
        
        return obs, hierarchical_reward, terminated, truncated, info