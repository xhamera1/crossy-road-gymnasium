from __future__ import annotations

from pathlib import Path

import numpy as np

from gymnasium_env.core.config import GameConfig
from gymnasium_env.core.constants import CellID


SPRITE_TILE_SIZE = 15
SPRITE_SHEET_PATH = Path(__file__).resolve().parents[2] / "sprite-sheet.png"
LILY_PAD_PATH = Path(__file__).resolve().parents[2] / "lily-pad.png"
AGENT_SPRITE_COORDS = (7, 1)
CAR_SPRITE_COORDS = ((2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (4, 3))


class PygameRenderer:
    def __init__(self, config: GameConfig, fps: int):
        self.config = config
        self.fps = fps
        self.hud_height = max(42, config.window_size // 12)
        self.viewport_size = 8
        self.window = None
        self.clock = None
        self.window_closed = False
        self.sprites_loaded = False
        self.font = None
        self.agent_sprite = None
        self.lily_pad_sprite = None
        self.car_sprites = []
        self.car_sprite_indices: dict[tuple[int, int], int] = {}
        self.next_car_sprite_index = 0

    def _ensure_ready(self):
        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.config.window_size, self.config.window_size + self.hud_height)
            )
            pygame.display.set_caption("Crossy Road Gymnasium")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, max(28, self.hud_height - 14))
        if not self.sprites_loaded:
            self._load_sprites(pygame)
        return pygame

    def _load_sprites(self, pygame):
        self.sprites_loaded = True
        self.agent_sprite = None
        self.lily_pad_sprite = None
        self.car_sprites = []
        if SPRITE_SHEET_PATH.exists():
            sheet = pygame.image.load(str(SPRITE_SHEET_PATH)).convert_alpha()

            def crop(coord: tuple[int, int]):
                col, row = coord
                rect = pygame.Rect(
                    col * SPRITE_TILE_SIZE,
                    row * SPRITE_TILE_SIZE,
                    SPRITE_TILE_SIZE,
                    SPRITE_TILE_SIZE,
                )
                return sheet.subsurface(rect).copy()

            self.agent_sprite = crop(AGENT_SPRITE_COORDS)
            self.car_sprites = [crop(coord) for coord in CAR_SPRITE_COORDS]

        if LILY_PAD_PATH.exists():
            self.lily_pad_sprite = pygame.image.load(str(LILY_PAD_PATH)).convert_alpha()

    def _car_sprite_index(self, x: int, y: int, width: int, used_previous: set[tuple[int, int]]) -> int:
        for prev_y in (y, y + 1):
            for prev_x in (x, (x - 1) % width, (x + 1) % width):
                previous = (prev_y, prev_x)
                if previous in self.car_sprite_indices and previous not in used_previous:
                    used_previous.add(previous)
                    return self.car_sprite_indices[previous]

        sprite_index = self.next_car_sprite_index % len(self.car_sprites)
        self.next_car_sprite_index += 1
        return sprite_index

    def _sprite_for(
        self,
        tile: CellID,
        x: int,
        y: int,
        width: int,
        used_previous: set[tuple[int, int]],
    ):
        if tile == CellID.CAR and self.car_sprites:
            sprite_index = self._car_sprite_index(x, y, width, used_previous)
            return self.car_sprites[sprite_index], sprite_index
        if tile == CellID.AGENT:
            return self.agent_sprite, None
        if tile == CellID.LILY_PAD:
            return self.lily_pad_sprite, None
        return None, None

    def _viewport_bounds(self, grid: np.ndarray) -> tuple[int, int, int, int]:
        height, width = grid.shape
        viewport_h = min(self.viewport_size, height)
        viewport_w = min(self.viewport_size, width)
        agent_positions = np.argwhere(grid == CellID.AGENT)

        if len(agent_positions) == 0:
            agent_y, agent_x = 0, 0
        else:
            agent_y, agent_x = (int(v) for v in agent_positions[0])

        y_start = min(max(agent_y - viewport_h // 3, 0), height - viewport_h)
        x_start = min(max(agent_x - viewport_w // 2, 0), width - viewport_w)
        return y_start, y_start + viewport_h, x_start, x_start + viewport_w

    def render(
        self,
        grid: np.ndarray,
        score: int,
        background_grid: np.ndarray | None = None,
        lane_directions: np.ndarray | None = None,
    ) -> None:
        if self.window_closed:
            return
        pygame = self._ensure_ready()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                self.close()
                return

        y_start, y_end, x_start, x_end = self._viewport_bounds(grid)
        viewport_h = y_end - y_start
        viewport_w = x_end - x_start
        cell_w = self.config.window_size / viewport_w
        cell_h = self.config.window_size / viewport_h

        canvas = pygame.Surface((self.config.window_size, self.config.window_size + self.hud_height))
        canvas.fill((36, 40, 48))
        self._draw_hud(pygame, canvas, score, y_start, viewport_h)

        board = pygame.Surface((self.config.window_size, self.config.window_size))
        board.fill((36, 40, 48))

        palette = {
            CellID.GRASS: (126, 200, 80),
            CellID.ROAD: (64, 67, 74),
            CellID.CAR: (237, 99, 99),
            CellID.RIVER: (80, 153, 222),
            CellID.LILY_PAD: (140, 220, 130),
            CellID.AGENT: (255, 211, 92),
        }
        car_sprite_indices: dict[tuple[int, int], int] = {}
        used_previous: set[tuple[int, int]] = set()

        for local_y, y in enumerate(range(y_start, y_end)):
            for local_x, x in enumerate(range(x_start, x_end)):
                tile = CellID(int(grid[y, x]))
                draw_y = viewport_h - 1 - local_y
                left = int(local_x * cell_w)
                right = int((local_x + 1) * cell_w)
                top = int(draw_y * cell_h)
                bottom = int((draw_y + 1) * cell_h)
                rect = pygame.Rect(left, top, right - left, bottom - top)
                sprite, sprite_index = self._sprite_for(tile, x, y, grid.shape[1], used_previous)
                if tile == CellID.CAR and sprite_index is not None:
                    car_sprite_indices[(y, x)] = sprite_index
                background_tile = CellID(int(background_grid[y, x])) if background_grid is not None else tile
                if tile == CellID.CAR:
                    background_tile = CellID.ROAD
                elif tile == CellID.LILY_PAD or background_tile == CellID.LILY_PAD:
                    background_tile = CellID.RIVER
                pygame.draw.rect(board, palette[background_tile], rect)
                if tile == CellID.AGENT and background_grid is not None:
                    underlay_tile = CellID(int(background_grid[y, x]))
                    underlay_sprite, _ = self._sprite_for(
                        underlay_tile,
                        x,
                        y,
                        grid.shape[1],
                        used_previous,
                    )
                    if underlay_sprite is not None:
                        board.blit(pygame.transform.scale(underlay_sprite, rect.size), rect)
                if sprite is not None:
                    scaled = pygame.transform.scale(sprite, rect.size)
                    if tile == CellID.CAR and lane_directions is not None and int(lane_directions[y]) == 2:
                        scaled = pygame.transform.flip(scaled, True, False)
                    board.blit(scaled, rect)
                elif tile in {CellID.CAR, CellID.AGENT, CellID.LILY_PAD}:
                    inset = pygame.Rect(
                        int(local_x * cell_w + cell_w * 0.15),
                        int(draw_y * cell_h + cell_h * 0.15),
                        int(cell_w * 0.7),
                        int(cell_h * 0.7),
                    )
                    pygame.draw.rect(board, (245, 245, 245), inset, border_radius=6)

        self.car_sprite_indices = car_sprite_indices

        canvas.blit(board, (0, self.hud_height))
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.fps)

    def _draw_hud(self, pygame, canvas, score: int, row_start: int, visible_rows: int) -> None:
        pygame.draw.rect(canvas, (23, 26, 32), pygame.Rect(0, 0, self.config.window_size, self.hud_height))
        pygame.draw.line(
            canvas,
            (73, 79, 91),
            (0, self.hud_height - 1),
            (self.config.window_size, self.hud_height - 1),
            1,
        )
        text = self.font.render(f"Distance: {score}", True, (245, 247, 250))
        text_rect = text.get_rect(midleft=(16, self.hud_height // 2))
        canvas.blit(text, text_rect)
        rows = self.font.render(f"Rows {row_start + 1}-{row_start + visible_rows}", True, (185, 193, 204))
        rows_rect = rows.get_rect(midright=(self.config.window_size - 16, self.hud_height // 2))
        canvas.blit(rows, rows_rect)

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
        self.font = None
