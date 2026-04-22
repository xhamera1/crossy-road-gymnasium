from __future__ import annotations

import numpy as np

from gymnasium_env.core.config import GameConfig
from gymnasium_env.core.constants import CellID


class PygameRenderer:
    def __init__(self, config: GameConfig, fps: int):
        self.config = config
        self.fps = fps
        self.window = None
        self.clock = None
        self.window_closed = False

    def _ensure_ready(self):
        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.config.window_size, self.config.window_size))
            pygame.display.set_caption("Crossy Road Gymnasium")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        return pygame

    def render(self, grid: np.ndarray) -> None:
        if self.window_closed:
            return
        pygame = self._ensure_ready()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                self.close()
                return

        canvas = pygame.Surface((self.config.window_size, self.config.window_size))
        canvas.fill((36, 40, 48))

        h, w = grid.shape
        cell_w = self.config.window_size / w
        cell_h = self.config.window_size / h

        palette = {
            CellID.GRASS: (126, 200, 80),
            CellID.ROAD: (64, 67, 74),
            CellID.CAR: (237, 99, 99),
            CellID.RIVER: (80, 153, 222),
            CellID.LILY_PAD: (140, 220, 130),
            CellID.AGENT: (255, 211, 92),
        }

        for y in range(h):
            for x in range(w):
                tile = CellID(int(grid[y, x]))
                draw_y = h - 1 - y
                rect = pygame.Rect(int(x * cell_w), int(draw_y * cell_h), int(cell_w), int(cell_h))
                pygame.draw.rect(canvas, palette[tile], rect)
                if tile in {CellID.CAR, CellID.AGENT, CellID.LILY_PAD}:
                    inset = pygame.Rect(
                        int(x * cell_w + cell_w * 0.15),
                        int(draw_y * cell_h + cell_h * 0.15),
                        int(cell_w * 0.7),
                        int(cell_h * 0.7),
                    )
                    pygame.draw.rect(canvas, (245, 245, 245), inset, border_radius=6)

        for gx in range(w + 1):
            pygame.draw.line(
                canvas,
                (30, 30, 30),
                (int(gx * cell_w), 0),
                (int(gx * cell_w), self.config.window_size),
                width=1,
            )
        for gy in range(h + 1):
            pygame.draw.line(
                canvas,
                (30, 30, 30),
                (0, int(gy * cell_h)),
                (self.config.window_size, int(gy * cell_h)),
                width=1,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
