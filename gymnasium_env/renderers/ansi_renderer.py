from __future__ import annotations

import numpy as np

from gymnasium_env.core.constants import ANSI_SYMBOLS, CellID


class AnsiRenderer:
    def render(self, grid: np.ndarray, score: int, steps: int) -> str:
        height, width = grid.shape
        lines: list[str] = []
        for y in range(height - 1, -1, -1):
            row = "".join(ANSI_SYMBOLS[CellID(int(grid[y, x]))] for x in range(width))
            lines.append(row)
        lines.append(f"score={score} steps={steps}")
        return "\n".join(lines)
