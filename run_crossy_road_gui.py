import gymnasium as gym

import gymnasium_env
from crossy_agent import StrategicCrossyAgent


if __name__ == "__main__":
    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GUI mode requires pygame. Install it first and run again."
        ) from exc

    env = gym.make(
        "gymnasium_env/CrossyRoad-v0",
        render_mode="human",
        observation_mode="large_discrete",
    )
    agent = StrategicCrossyAgent(width=8, height=50)

    print("GUI mode started.")
    print("Controls:")
    print("  SPACE - execute exactly one agent move")
    print("  R     - reset current episode manually")
    print("  ESC   - quit")

    obs, info = env.reset(seed=42)
    episode_idx = 1
    episode_reward = 0
    previous_space = False
    previous_r = False
    previous_esc = False
    running = True

    try:
        while running:
            env.render()

            keys = pygame.key.get_pressed()
            space_pressed = bool(keys[pygame.K_SPACE])
            r_pressed = bool(keys[pygame.K_r])
            esc_pressed = bool(keys[pygame.K_ESCAPE])

            if esc_pressed and not previous_esc:
                running = False
            elif r_pressed and not previous_r:
                print(f"[Episode {episode_idx}] manual reset.")
                obs, info = env.reset()
                episode_reward = 0
            elif space_pressed and not previous_space:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                print(
                    f"[Episode {episode_idx}] step={info['steps']} action={action} "
                    f"reward={reward} score={info['score']}"
                )

                if terminated or truncated:
                    print(
                        f"[Episode {episode_idx}] finished: "
                        f"score={info['score']} total_reward={episode_reward} steps={info['steps']}"
                    )
                    episode_idx += 1
                    obs, info = env.reset()
                    episode_reward = 0
                    print(f"[Episode {episode_idx}] auto-restart.")

            previous_space = space_pressed
            previous_r = r_pressed
            previous_esc = esc_pressed
    finally:
        env.close()
