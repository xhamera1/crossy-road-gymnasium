from __future__ import annotations

import numpy as np

from gymnasium_env.core.config import GameConfig
from gymnasium_env.core.constants import TerrainType
from gymnasium_env.core.entities import Lane


class LaneFactory:
    def __init__(self, config: GameConfig):
        self.config = config

    def create_lane(self, rng: np.random.Generator) -> Lane:
        terrain = rng.choice(
            [TerrainType.GRASS, TerrainType.ROAD, TerrainType.RIVER],
            p=self.config.terrain_probs,
        )
        if terrain == TerrainType.GRASS:
            return Lane(terrain=TerrainType.GRASS)

        direction = int(rng.choice([-1, 1]))
        move_interval = 1 if terrain == TerrainType.ROAD else int(rng.choice(self.config.lane_speed_values))
        density = self._density_for_terrain(terrain, rng)
        actors = {x for x in range(self.config.width) if rng.random() < density}
        actors = self._normalize_actor_count(terrain=terrain, actors=actors, rng=rng)
        return Lane(
            terrain=terrain,
            direction=direction,
            move_interval=move_interval,
            actors=actors,
        )

    def _density_for_terrain(self, terrain: TerrainType, rng: np.random.Generator) -> float:
        if terrain == TerrainType.ROAD:
            lo, hi = self.config.road_density_range
            return float(rng.uniform(lo, hi))
        lo, hi = self.config.river_density_range
        return float(rng.uniform(lo, hi))

    def _normalize_actor_count(
        self, terrain: TerrainType, actors: set[int], rng: np.random.Generator
    ) -> set[int]:
        width = self.config.width
        if terrain == TerrainType.ROAD:
            min_count, max_count = 1, max(1, width - 2)
        elif terrain == TerrainType.RIVER:
            min_count, max_count = 2, max(2, width - 1)
        else:
            return actors

        if len(actors) < min_count:
            actors = set(rng.choice(width, size=min_count, replace=False).tolist())
        elif len(actors) > max_count:
            chosen = rng.choice(list(actors), size=max_count, replace=False)
            actors = set(chosen.tolist())
        return actors
