from __future__ import annotations

from dataclasses import dataclass, field

from gymnasium_env.core.constants import TerrainType


@dataclass
class Lane:
    terrain: TerrainType
    direction: int = 0
    move_interval: int = 1
    actors: set[int] = field(default_factory=set)
    tick: int = 0

    def step(self, width: int) -> int:
        if self.terrain == TerrainType.GRASS:
            return 0
        self.tick += 1
        if self.tick % self.move_interval != 0 or not self.actors:
            return 0
        self.actors = {((x + self.direction) % width) for x in self.actors}
        return self.direction
