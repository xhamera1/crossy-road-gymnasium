from gymnasium.envs.registration import register

from gymnasium_env.envs import CrossyRoadEnv

register(
    id="gymnasium_env/CrossyRoad-v0",
    entry_point="gymnasium_env.envs:CrossyRoadEnv",
    max_episode_steps=200,
)

__all__ = ["CrossyRoadEnv"]
