"""
Play a saved Stable-Baselines3 model with the interactive renderer.

Usage:
    python play_model.py --model the_last_signal_PPO --algo PPO --fullscreen

Controls:
    P = Pause/Resume, N = Step (when paused), ESC = Exit
"""
import argparse
import pygame

try:
    from stable_baselines3 import PPO, DQN, A2C
except Exception:
    PPO = DQN = A2C = None

from the_last_signal_env import TheLastSignalEnv
from game_renderer import GameRenderer


def _load_model(path: str, algo: str):
    algo = (algo or "PPO").upper()
    if algo == "PPO":
        if PPO is None:
            raise ImportError("stable-baselines3 not available")
        return PPO.load(path)
    if algo == "DQN":
        if DQN is None:
            raise ImportError("stable-baselines3 not available")
        return DQN.load(path)
    if algo == "A2C":
        if A2C is None:
            raise ImportError("stable-baselines3 not available")
        return A2C.load(path)
    # Default attempt: try PPO
    if PPO is not None:
        return PPO.load(path)
    raise ImportError("Unsupported algorithm or SB3 not installed")


def play_model(model_path: str, algo: str = "PPO", fullscreen: bool = True):
    env = TheLastSignalEnv(render_mode=None)
    obs, info = env.reset()

    renderer = GameRenderer(env.game, fullscreen=fullscreen)

    # Load model
    model = _load_model(model_path, algo)

    total_reward = 0.0
    try:
        while renderer.running:
            # Let renderer process events and draw (handles pause/step)
            renderer.render()

            if renderer.paused and not renderer.step:
                continue

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)

            # consume single-step
            renderer.step = False

            if terminated or truncated:
                print("Episode ended. Total reward:", total_reward)
                break

    finally:
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="the_last_signal_ppo", help="Path to saved model (SB3)")
    parser.add_argument("--algo", "-a", default="PPO", help="Algorithm used to train the model (PPO/DQN/A2C)")
    parser.add_argument("--fullscreen", action="store_true", help="Open renderer in fullscreen")
    args = parser.parse_args()

    try:
        play_model(args.model, algo=args.algo, fullscreen=args.fullscreen)
    except Exception as e:
        print("Error running model:", e)


if __name__ == "__main__":
    main()
