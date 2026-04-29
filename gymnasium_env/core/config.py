from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    width: int = 8
    height: int = 50
    window_size: int = 600
    reward_forward: int = 1
    reward_death: int = -100
    terrain_probs: tuple[float, float, float] = (0.45, 0.33, 0.22)  # grass, road, river
    road_density_range: tuple[float, float] = (0.12, 0.26)
    river_density_range: tuple[float, float] = (0.45, 0.72)
    lane_speed_values: tuple[int, ...] = (2, 3)
    safe_start_rows: int = 3
