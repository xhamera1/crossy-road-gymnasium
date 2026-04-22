import gymnasium as gym
import time

import gymnasium_env
from crossy_agent import StrategicCrossyAgent


RENDER_MODE = "ansi"  # "human" opens pygame window, "ansi" prints to terminal
MAX_STEPS = 500


if __name__ == "__main__":
    env = gym.make(
        "gymnasium_env/CrossyRoad-v0",
        render_mode=RENDER_MODE,
        observation_mode="large_discrete",
    )
    obs, info = env.reset(seed=42)
    agent = StrategicCrossyAgent(width=8, height=14)

    print(f"Starting episode (render_mode={RENDER_MODE})")
    if RENDER_MODE == "ansi":
        print(env.render())
        print("-" * 24)
    else:
        print("Pygame window should open now.")

    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated) and info["steps"] < MAX_STEPS:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if RENDER_MODE == "ansi":
            print(f"action={action} reward={reward}")
            print(env.render())
            print("-" * 24)
        else:
            # Keep pygame window responsive while episode runs.
            time.sleep(1.0 / 8.0)

    print("Episode finished")
    print(f"score={info['score']} total_reward={total_reward} steps={info['steps']}")
    env.close()
