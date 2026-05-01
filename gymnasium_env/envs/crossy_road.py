from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium_env.core import CellID, CrossyRoadEngine, GameConfig
from gymnasium_env.renderers import AnsiRenderer, PygameRenderer


class CrossyRoadEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 8}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        width: int = 8,
        height: int = 50,
        observation_mode: str = "large_discrete",
        window_size: int = 600,
        view_height: int = 10,
    ):
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.view_height = min(view_height, height)
        self.config = GameConfig(width=width, height=height, window_size=window_size)

        if observation_mode == "grid":
            self.observation_space = spaces.Box(
                low=0,
                high=int(CellID.AGENT),
                shape=(height, width),
                dtype=np.int32,
            )
        elif observation_mode == "large_discrete":
            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=int(CellID.AGENT),
                        shape=(height, width),
                        dtype=np.int32,
                    ),
                    "lane_directions": spaces.MultiDiscrete([3] * height),
                    "lane_speeds": spaces.MultiDiscrete([4] * height),
                    "agent_position": spaces.MultiDiscrete([height, width]),
                }
            )
        elif observation_mode == "local":
            # Cropped: tylko view_height najnizszych rzedow + dynamika pasow.
            # ~5x mniejsza obserwacja niz "large_discrete" (160 vs 808 cech po one-hot),
            # agent skupia sie na otoczeniu.
            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(
                        low=0,
                        high=int(CellID.AGENT),
                        shape=(self.view_height, width),
                        dtype=np.int32,
                    ),
                    "lane_directions": spaces.MultiDiscrete([3] * self.view_height),
                    "lane_speeds": spaces.MultiDiscrete([4] * self.view_height),
                    "agent_position": spaces.MultiDiscrete([2, width]),
                }
            )
        else:
            raise ValueError("observation_mode must be 'grid', 'large_discrete' or 'local'")

        self.action_space = spaces.Discrete(4)
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.engine = CrossyRoadEngine(config=self.config)
        self.ansi_renderer = AnsiRenderer()
        self.pygame_renderer: Optional[PygameRenderer] = None
        self._last_obs: Optional[Any] = None

    def _build_observation(self):
        if self.observation_mode == "grid":
            return self.engine.grid_observation()
        if self.observation_mode == "local":
            return self.engine.local_observation(self.view_height)
        grid = self.engine.grid_observation()
        return {
            "grid": grid,
            "lane_directions": self.engine.lane_directions(),
            "lane_speeds": self.engine.lane_speeds(),
            "agent_position": np.array([self.engine.agent_y, self.engine.agent_x], dtype=np.int32),
        }

    def _info(self) -> dict:
        return {"score": self.engine.score, "steps": self.engine.steps}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.engine.reset(self.np_random)
        self._last_obs = self._build_observation()
        if self.render_mode == "human":
            self._render_human()
        return self._last_obs, self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action={action}")
        reward, terminated = self.engine.step(action=action, rng=self.np_random)
        self._last_obs = self._build_observation()
        if self.render_mode == "human":
            self._render_human()
        return self._last_obs, reward, terminated, False, self._info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def _render_ansi(self) -> str:
        if self._last_obs is None:
            return ""
        grid = self._last_obs if self.observation_mode == "grid" else self._last_obs["grid"]
        return self.ansi_renderer.render(grid=grid, score=self.engine.score, steps=self.engine.steps)

    def _render_human(self):
        if self._last_obs is None:
            return None
        try:
            if self.pygame_renderer is None:
                self.pygame_renderer = PygameRenderer(config=self.config, fps=self.metadata["render_fps"])
            # Renderer chce zawsze pelny grid (50 rzedow) niezaleznie od observation_mode --
            # cropped obs jest wylacznie dla agenta.
            grid = self.engine.grid_observation()
            self.pygame_renderer.render(
                grid=grid,
                score=self.engine.score,
                background_grid=self.engine.base_grid(),
                lane_directions=self.engine.lane_directions(),
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pygame is required for human rendering. Install with `pip install pygame`."
            ) from exc
        return None

    def close(self):
        if self.pygame_renderer is not None:
            self.pygame_renderer.close()
        self.pygame_renderer = None
