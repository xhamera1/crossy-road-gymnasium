"""Trening PPO na srodowisku CrossyRoad-v0.

Wynikiem treningu sa:
  - models/ppo_crossy_final.zip            -- zapisany ostateczny model
  - models/best/best_model.zip             -- najlepszy model wg EvalCallback
  - training_logs/monitor_*.csv            -- per-epizodowe statystyki (Monitor)
  - training_logs/tensorboard/PPO_*/       -- logi do TensorBoard
  - training_logs/learning_curves.png      -- wykresy z monitor CSV (rysowane na koncu)

Uruchomienie:
    python train_ppo.py
    python train_ppo.py --timesteps 500000 --n-envs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym

import gymnasium_env  # noqa: F401  -- rejestruje gymnasium_env/CrossyRoad-v0


# Wstepna walidacja srodowiska Pythona: SB3 nie wspiera Pythona >= 3.13.
if sys.version_info >= (3, 13):
    print(
        "[BLAD] stable-baselines3 nie obsluguje Pythona >= 3.13. "
        f"Wykryto {sys.version_info.major}.{sys.version_info.minor}. "
        "Zaloz venv z Pythonem 3.10-3.12 (np. `py -3.12 -m venv .venv`).",
        file=sys.stderr,
    )
    sys.exit(1)


from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "training_logs"
MODELS_DIR = PROJECT_ROOT / "models"
TB_DIR = LOG_DIR / "tensorboard"
EVAL_LOG_DIR = LOG_DIR / "eval"


def _ensure_dirs() -> None:
    for path in (LOG_DIR, MODELS_DIR, TB_DIR, EVAL_LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def make_env(seed: int, monitor_dir: Path):
    """Fabryka srodowiska. Owija je w Monitor (CSV per-epizod).

    Uzywa observation_mode='local' -- 10 najbliszych rzedow (~160 cech po one-hot)
    zamiast pelnej 50-rzedowej obserwacji (~808 cech). Drastycznie przyspiesza
    uczenie i pozwala sieci skupic sie na otoczeniu agenta.
    """

    def _init():
        env = gym.make(
            "gymnasium_env/CrossyRoad-v0",
            observation_mode="local",
        )
        monitor_path = str(monitor_dir / f"monitor_{seed}.csv")
        env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed)
        return env

    return _init


def build_vec_env(n_envs: int, base_seed: int, monitor_dir: Path) -> DummyVecEnv:
    fns = [make_env(seed=base_seed + i, monitor_dir=monitor_dir) for i in range(n_envs)]
    return DummyVecEnv(fns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on CrossyRoad-v0")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs (DummyVecEnv)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Adam learning rate",
    )
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length per env")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Pomin generowanie learning_curves.png po treningu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs()

    print(f"[INFO] timesteps={args.timesteps} n_envs={args.n_envs} seed={args.seed}")
    print(f"[INFO] logs    -> {LOG_DIR}")
    print(f"[INFO] models  -> {MODELS_DIR}")
    print(f"[INFO] tboard  -> {TB_DIR}  (otworz: tensorboard --logdir {TB_DIR})")

    train_env = build_vec_env(n_envs=args.n_envs, base_seed=args.seed, monitor_dir=LOG_DIR)
    eval_env = build_vec_env(n_envs=1, base_seed=args.seed + 9999, monitor_dir=EVAL_LOG_DIR)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(EVAL_LOG_DIR),
        eval_freq=max(2_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="ppo_crossy",
    )

    # Linear LR schedule: 1.0 -> 0.0 (mnoznik), zeby pod koniec treningu polityka
    # stabilizowala sie zamiast oscylowac.
    def linear_schedule(progress_remaining: float) -> float:
        return progress_remaining * args.learning_rate

    # Wieksza siec -- 808-wymiarowa obserwacja (50 rzedow x 8 + onehoty) wymaga
    # wiekszej pojemnosci niz domyslne [64, 64].
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        learning_rate=linear_schedule,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        # Wysoki ent_coef -> mocna eksploracja, zeby polityka nie zacementowala
        # sie na zachowawczym "stoje na trawie" przed nauczeniem przeskakiwac drogi.
        ent_coef=0.1,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device="auto",
    )

    print(f"[INFO] device = {model.device}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
    )

    final_path = MODELS_DIR / "ppo_crossy_final"
    model.save(str(final_path))
    print(f"[OK] Zapisano model: {final_path}.zip")

    train_env.close()
    eval_env.close()

    if not args.no_plot:
        try:
            from plot_learning_curves import plot_learning_curves

            plot_learning_curves()
        except Exception as exc:
            print(f"[WARN] Nie udalo sie wygenerowac wykresow: {exc}")


if __name__ == "__main__":
    main()
