from __future__ import annotations

from enum import IntEnum


class CellID(IntEnum):
    GRASS = 0
    ROAD = 1
    CAR = 2
    RIVER = 3
    LILY_PAD = 4
    AGENT = 5


class TerrainType(IntEnum):
    GRASS = CellID.GRASS
    ROAD = CellID.ROAD
    RIVER = CellID.RIVER


class ActionID(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS = {
    ActionID.UP: (0, 1),
    ActionID.DOWN: (0, -1),
    ActionID.LEFT: (-1, 0),
    ActionID.RIGHT: (1, 0),
}


ANSI_SYMBOLS = {
    CellID.GRASS: ".",
    CellID.ROAD: "=",
    CellID.CAR: "C",
    CellID.RIVER: "~",
    CellID.LILY_PAD: "O",
    CellID.AGENT: "A",
}

