"""Uruchamia wytrenowanego agenta PPO w trybie graficznym (Pygame).

Wymaga zapisanego modelu (domyslnie models/ppo_crossy_final.zip albo
models/best/best_model.zip). Najpierw odpal trening: `python train_ppo.py`.

Sterowanie w oknie Pygame:
    SPACE - jeden krok agenta (krokowy podglad)
    A     - tryb auto-play (agent gra ciagle)
    R     - reset epizodu
    ESC   - wyjscie

Uruchomienie:
    python play_trained_gui.py
    python play_trained_gui.py --model models/best/best_model.zip
    python play_trained_gui.py --auto         # od razu w trybie auto-play
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym

import gymnasium_env  # noqa: F401  -- rejestruje srodowisko


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ppo_crossy_final.zip"
FALLBACK_MODEL = PROJECT_ROOT / "models" / "best" / "best_model.zip"


def _resolve_model_path(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise SystemExit(f"[BLAD] Nie znaleziono modelu: {path}")
        return path
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    if FALLBACK_MODEL.exists():
        print(f"[INFO] {DEFAULT_MODEL} nie istnieje - uzywam {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    raise SystemExit(
        f"[BLAD] Brak modelu. Przewidywane sciezki:\n"
        f"  {DEFAULT_MODEL}\n  {FALLBACK_MODEL}\n"
        f"Najpierw uruchom: python train_ppo.py"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent in Pygame GUI")
    parser.add_argument("--model", type=str, default=None, help="Sciezka do modelu .zip")
    parser.add_argument("--seed", type=int, default=42, help="Seed startowy srodowiska")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Tryb auto-play od startu (bez czekania na SPACE)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Greedy policy (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Probkuj akcje stochastycznie zamiast greedy.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Predkosc auto-play")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise SystemExit("Tryb GUI wymaga pygame. Zainstaluj `pip install pygame`.") from exc

    if sys.version_info >= (3, 13):
        print(
            "[BLAD] stable-baselines3 nie obsluguje Pythona >= 3.13. "
            f"Wykryto {sys.version_info.major}.{sys.version_info.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)

    from stable_baselines3 import PPO

    model_path = _resolve_model_path(args.model)
    print(f"[INFO] Laduje model: {model_path}")
    model = PPO.load(str(model_path))

    env = gym.make(
        "gymnasium_env/CrossyRoad-v0",
        render_mode="human",
        observation_mode="local",
    )
    obs, info = env.reset(seed=args.seed)

    auto = bool(args.auto)
    episode_idx = 1
    episode_reward = 0.0
    previous_space = False
    previous_r = False
    previous_a = False
    previous_esc = False
    running = True
    auto_period = 1.0 / max(args.fps, 1)
    last_auto_step = time.time()

    print("Sterowanie: SPACE=krok, A=auto-play, R=reset, ESC=wyjscie")

    try:
        while running:
            env.render()

            keys = pygame.key.get_pressed()
            space_pressed = bool(keys[pygame.K_SPACE])
            r_pressed = bool(keys[pygame.K_r])
            a_pressed = bool(keys[pygame.K_a])
            esc_pressed = bool(keys[pygame.K_ESCAPE])

            do_step = False
            if esc_pressed and not previous_esc:
                running = False
            elif r_pressed and not previous_r:
                print(f"[Episode {episode_idx}] manual reset.")
                obs, info = env.reset()
                episode_reward = 0.0
            elif a_pressed and not previous_a:
                auto = not auto
                print(f"[Episode {episode_idx}] auto-play = {auto}")
            elif space_pressed and not previous_space:
                do_step = True

            if auto:
                now = time.time()
                if now - last_auto_step >= auto_period:
                    do_step = True
                    last_auto_step = now

            if do_step:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                action_int = int(action)
                obs, reward, terminated, truncated, info = env.step(action_int)
                episode_reward += float(reward)
                print(
                    f"[Episode {episode_idx}] step={info['steps']} action={action_int} "
                    f"reward={reward} score={info['score']}"
                )
                if terminated or truncated:
                    print(
                        f"[Episode {episode_idx}] finished: score={info['score']} "
                        f"total_reward={episode_reward:.1f} steps={info['steps']}"
                    )
                    episode_idx += 1
                    obs, info = env.reset()
                    episode_reward = 0.0

            previous_space = space_pressed
            previous_r = r_pressed
            previous_a = a_pressed
            previous_esc = esc_pressed
    finally:
        env.close()


if __name__ == "__main__":
    main()
