from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StrategicCrossyAgent:
    """Rule-based agent with deterministic strategy (no random actions)."""

    width: int = 8
    height: int = 50

    def act(self, observation: Any) -> int:
        grid = observation["grid"] if isinstance(observation, dict) else observation
        lane_dirs = observation.get("lane_directions") if isinstance(observation, dict) else None
        lane_speeds = observation.get("lane_speeds") if isinstance(observation, dict) else None
        y, x = self._find_agent(grid)

        # Candidate priority reflects strategy:
        # 1) advance safely, 2) move sideways to avoid imminent hazards, 3) fallback down.
        candidates = [0, 2, 3, 1]  # Up, Left, Right, Down

        best_action = 1
        best_score = -10**9
        for action in candidates:
            score = self._score_action(grid, y, x, action, lane_dirs, lane_speeds)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    @staticmethod
    def _find_agent(grid: np.ndarray) -> tuple[int, int]:
        positions = np.argwhere(grid == 5)
        if len(positions) == 0:
            return 0, 0
        y, x = positions[0]
        return int(y), int(x)

    def _score_action(
        self,
        grid: np.ndarray,
        y: int,
        x: int,
        action: int,
        lane_dirs: np.ndarray | None,
        lane_speeds: np.ndarray | None,
    ) -> int:
        deltas = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = deltas[action]
        ny, nx = y + dy, x + dx
        if not (0 <= ny < self.height and 0 <= nx < self.width):
            return -10_000

        cell = int(grid[ny, nx])
        # 2 = car, 3 = water: immediate death-risk tiles
        if cell in {2, 3}:
            return -5_000

        score = 0
        if action == 0:
            score += 300  # prefer forward progress
        elif action in {2, 3}:
            score += 80
        else:
            score += 10

        # Prefer stable tiles over moving hazards.
        if cell == 0:
            score += 80   # grass
        elif cell == 1:
            score += 30   # road
        elif cell == 4:
            score += 40   # lily pad

        # Local risk estimate in Moore neighborhood.
        risk = 0
        for yy in range(max(0, ny - 1), min(self.height, ny + 2)):
            for xx in range(max(0, nx - 1), min(self.width, nx + 2)):
                neighbor = int(grid[yy, xx])
                if neighbor == 2:
                    risk += 25
                elif neighbor == 3:
                    risk += 15

        # Look one tick ahead using lane dynamics from large_discrete observation.
        if lane_dirs is not None and lane_speeds is not None:
            risk += self._predict_next_tick_risk(grid, ny, nx, lane_dirs, lane_speeds)

        score -= risk
        return score

    def _predict_next_tick_risk(
        self,
        grid: np.ndarray,
        ny: int,
        nx: int,
        lane_dirs: np.ndarray,
        lane_speeds: np.ndarray,
    ) -> int:
        direction_enc = int(lane_dirs[ny])  # 0 left, 1 static, 2 right
        speed = int(lane_speeds[ny])        # 0 grass, 1+ moving lane
        if speed <= 0:
            return 0

        direction = -1 if direction_enc == 0 else 1 if direction_enc == 2 else 0
        if direction == 0:
            return 0

        lane = grid[ny]
        predicted_x = (nx - direction) % self.width
        current_tile = int(lane[nx])
        incoming_tile = int(lane[predicted_x])

        danger = 0
        if current_tile == 1 and incoming_tile == 2:
            danger += 120  # car may enter our tile on road lane
        if current_tile == 4 and incoming_tile == 3:
            danger += 140  # lily may move away and expose water
        if current_tile == 3 and incoming_tile == 4:
            danger -= 20   # a lily may arrive to rescue route
        return danger
