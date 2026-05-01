from __future__ import annotations

import numpy as np

from gymnasium_env.core.config import GameConfig
from gymnasium_env.core.constants import ACTION_DELTAS, ActionID, CellID, TerrainType
from gymnasium_env.core.entities import Lane
from gymnasium_env.core.generator import LaneFactory


class CrossyRoadEngine:
    def __init__(self, config: GameConfig):
        self.config = config
        self.factory = LaneFactory(config=config)
        self.lanes: list[Lane] = []
        self.agent_x = config.width // 2
        self.agent_y = self._bottom_buffer_row()
        self.score = 0
        self.steps = 0
        self.terminated = False

    def _bottom_buffer_row(self) -> int:
        return min(1, self.config.height - 1)

    def reset(self, rng: np.random.Generator) -> None:
        safe_rows = min(self.config.safe_start_rows, self.config.height)
        self.lanes = [Lane(terrain=TerrainType.GRASS) for _ in range(safe_rows)]
        for _ in range(self.config.height - safe_rows):
            self.lanes.append(self.factory.create_lane(rng))
        self.agent_x = self.config.width // 2
        self.agent_y = self._bottom_buffer_row()
        self.score = 0
        self.steps = 0
        self.terminated = False

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _scroll_world(self, rng: np.random.Generator) -> None:
        self.lanes.pop(0)
        self.lanes.append(self.factory.create_lane(rng))
        self.agent_y = max(self._bottom_buffer_row(), self.agent_y - 1)

    def step(self, action: int, rng: np.random.Generator) -> tuple[float, bool]:
        if self.terminated:
            raise RuntimeError("Episode already terminated. Call reset().")

        self.steps += 1
        # Per-step baseline penalty, zeby stanie w miejscu nie bylo zerowo-rewardowe.
        reward = float(self.config.reward_step)
        action_id = ActionID(action)
        dx, dy = ACTION_DELTAS[action_id]

        next_x = self.agent_x + dx
        next_y = self.agent_y + dy
        moved_forward = False
        if self._in_bounds(next_x, next_y):
            self.agent_x = next_x
            self.agent_y = next_y
            moved_forward = dy > 0 and self.agent_y > self._bottom_buffer_row()

        if moved_forward:
            reward += float(self.config.reward_forward)
            self.score += 1
            self._scroll_world(rng)

        lane_shift = 0
        for lane_index, lane in enumerate(self.lanes):
            shift = lane.step(self.config.width)
            if lane_index == self.agent_y:
                lane_shift = shift

        current = self.lanes[self.agent_y]
        if current.terrain == TerrainType.RIVER and lane_shift != 0:
            self.agent_x += lane_shift
            if not (0 <= self.agent_x < self.config.width):
                self.agent_x = min(max(self.agent_x, 0), self.config.width - 1)
                self.terminated = True
                return float(self.config.reward_death), True

        actors = current.actors
        hit_car = current.terrain == TerrainType.ROAD and self.agent_x in actors
        drown = current.terrain == TerrainType.RIVER and self.agent_x not in actors
        if hit_car or drown:
            self.terminated = True
            return float(self.config.reward_death), True

        # Survival bonus: agent zostal na niebezpiecznym kafelku i przezyl.
        # Czesciowo rekompensuje step penalty -- "czekanie az auto przejedzie" stoi sie
        # opcja realnie 0-rewardowa zamiast -0.1.
        if current.terrain in {TerrainType.ROAD, TerrainType.RIVER}:
            reward += float(self.config.reward_survival)

        return reward, False

    def base_grid(self) -> np.ndarray:
        grid = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        for y, lane in enumerate(self.lanes):
            if lane.terrain == TerrainType.GRASS:
                grid[y, :] = CellID.GRASS
            elif lane.terrain == TerrainType.ROAD:
                grid[y, :] = CellID.ROAD
                for x in lane.actors:
                    grid[y, x] = CellID.CAR
            else:
                grid[y, :] = CellID.RIVER
                for x in lane.actors:
                    grid[y, x] = CellID.LILY_PAD
        return grid

    def grid_observation(self) -> np.ndarray:
        grid = self.base_grid()
        grid[self.agent_y, self.agent_x] = CellID.AGENT
        return grid

    def lane_directions(self) -> np.ndarray:
        out = np.zeros(self.config.height, dtype=np.int32)
        for y, lane in enumerate(self.lanes):
            out[y] = 0 if lane.direction == -1 else 1 if lane.direction == 0 else 2
        return out

    def lane_speeds(self) -> np.ndarray:
        out = np.zeros(self.config.height, dtype=np.int32)
        for y, lane in enumerate(self.lanes):
            out[y] = 0 if lane.terrain == TerrainType.GRASS else lane.move_interval
        return out

    def local_observation(self, view_height: int) -> dict:
        """Cropped obserwacja: tylko `view_height` najnizszych rzedow + dynamika pasow.

        Drastycznie zmniejsza wymiar obserwacji (z ~808 do ~160 cech po one-hot)
        i sprawia, ze siec uczy sie tylko tego co istotne -- co jest w okolicy
        agenta. Reszta planszy (rzedy odlegle) i tak nie wplywa na decyzje.
        """
        h = min(view_height, self.config.height)
        grid = np.zeros((h, self.config.width), dtype=np.int32)
        lane_dirs = np.zeros(h, dtype=np.int32)
        lane_speeds = np.zeros(h, dtype=np.int32)
        for y in range(h):
            lane = self.lanes[y]
            if lane.terrain == TerrainType.GRASS:
                grid[y, :] = CellID.GRASS
            elif lane.terrain == TerrainType.ROAD:
                grid[y, :] = CellID.ROAD
                for x in lane.actors:
                    grid[y, x] = CellID.CAR
            else:
                grid[y, :] = CellID.RIVER
                for x in lane.actors:
                    grid[y, x] = CellID.LILY_PAD
            lane_dirs[y] = 0 if lane.direction == -1 else 1 if lane.direction == 0 else 2
            lane_speeds[y] = 0 if lane.terrain == TerrainType.GRASS else lane.move_interval

        if 0 <= self.agent_y < h:
            grid[self.agent_y, self.agent_x] = CellID.AGENT

        # agent_y trzymamy w {0, 1} (bottom-buffer ustala minimum, scroll resetuje na 1).
        clamped_y = max(0, min(1, self.agent_y))
        return {
            "grid": grid,
            "lane_directions": lane_dirs,
            "lane_speeds": lane_speeds,
            "agent_position": np.array([clamped_y, self.agent_x], dtype=np.int32),
        }
