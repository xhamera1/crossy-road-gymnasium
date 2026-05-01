from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    width: int = 8
    height: int = 50
    window_size: int = 600
    reward_forward: float = 1.0
    # Smierc kosztowna -- ma motywowac uniki (LEFT/RIGHT) zamiast kamikadze.
    # EV(forward przy 90% live) = 0.9*0.9 + 0.1*(-5) = +0.31 -- wciaz oplaca sie isc.
    # EV(forward przy 80% live) = 0.8*0.9 + 0.2*(-5) = -0.28 -- agent woli zrobic unik.
    reward_death: float = -5.0
    # Stanie 200 krokow = -20 (najgorsza strategia, gorsza niz natychmiastowa smierc).
    reward_step: float = -0.1
    # Bonus za przezycie kroku na DRODZE / RZECE -- czesciowo rekompensuje step penalty.
    # Sprawia, ze "stoj na drodze i czekaj az auto przejedzie" staje sie 0-rewardowe
    # zamiast -0.1, czyli agent moze realnie planowac uniki czasowe.
    reward_survival: float = 0.05
    terrain_probs: tuple[float, float, float] = (0.45, 0.33, 0.22)  # grass, road, river
    road_density_range: tuple[float, float] = (0.12, 0.26)
    river_density_range: tuple[float, float] = (0.45, 0.72)
    lane_speed_values: tuple[int, ...] = (2, 3)
    safe_start_rows: int = 3
