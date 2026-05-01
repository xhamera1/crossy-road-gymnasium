"""Ewaluacja wytrenowanego agenta PPO na N epizodach.

Uruchamia N=50 epizodow z roznymi seedami, zbiera statystyki score / reward /
length i rysuje histogramy do `training_logs/evaluation.png`.

Uruchomienie:
    python evaluate.py
    python evaluate.py --episodes 100 --model models/best/best_model.zip
    python evaluate.py --baseline   # rownolegle ewaluuje agenta regulowego dla porownania
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

import gymnasium_env  # noqa: F401  -- rejestruje srodowisko


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ppo_crossy_final.zip"
FALLBACK_MODEL = PROJECT_ROOT / "models" / "best" / "best_model.zip"
OUTPUT_PNG = PROJECT_ROOT / "training_logs" / "evaluation.png"


def _resolve_model_path(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise SystemExit(f"[BLAD] Nie znaleziono modelu: {path}")
        return path
    if FALLBACK_MODEL.exists():
        return FALLBACK_MODEL
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    raise SystemExit(
        "[BLAD] Brak modelu PPO. Uruchom najpierw: python train_ppo.py"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent on N episodes")
    parser.add_argument("--episodes", type=int, default=50, help="Liczba epizodow")
    parser.add_argument("--model", type=str, default=None, help="Sciezka do modelu .zip")
    parser.add_argument("--seed", type=int, default=1000, help="Seed startowy (kolejne: seed+i)")
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
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Ewaluuj rownolegle StrategicCrossyAgent (rule-based) dla porownania",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help="Sciezka wyjsciowa PNG z histogramami",
    )
    return parser.parse_args()


def _print_stats(name: str, scores: list[int], rewards: list[float], lengths: list[int]) -> None:
    print(f"\n=== {name} (N={len(scores)}) ===")
    print(f"  score:  mean={statistics.mean(scores):6.2f}  median={statistics.median(scores):6.1f}  "
          f"max={max(scores):3d}  min={min(scores):3d}  std={statistics.stdev(scores) if len(scores) > 1 else 0:5.2f}")
    print(f"  reward: mean={statistics.mean(rewards):7.3f}  median={statistics.median(rewards):7.3f}  "
          f"max={max(rewards):7.3f}  min={min(rewards):7.3f}")
    print(f"  length: mean={statistics.mean(lengths):6.2f}  median={statistics.median(lengths):6.1f}  "
          f"max={max(lengths):3d}  min={min(lengths):3d}")


def evaluate_ppo(model, env, episodes: int, base_seed: int, deterministic: bool):
    scores: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []
    for i in range(episodes):
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_steps = int(info["steps"])
            ep_score = int(info["score"])
            done = terminated or truncated
        scores.append(ep_score)
        rewards.append(ep_reward)
        lengths.append(ep_steps)
    return scores, rewards, lengths


def evaluate_rule_based(env, episodes: int, base_seed: int):
    from crossy_agent import StrategicCrossyAgent

    agent = StrategicCrossyAgent(width=env.unwrapped.config.width, height=env.unwrapped.config.height)
    scores: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []
    for i in range(episodes):
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_score = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_steps = int(info["steps"])
            ep_score = int(info["score"])
            done = terminated or truncated
        scores.append(ep_score)
        rewards.append(ep_reward)
        lengths.append(ep_steps)
    return scores, rewards, lengths


def plot_histograms(
    output: Path,
    ppo_scores: list[int],
    ppo_rewards: list[float],
    ppo_lengths: list[int],
    baseline_scores: list[int] | None = None,
):
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    has_baseline = baseline_scores is not None

    if has_baseline:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes = axes.reshape(1, 3)

    bins_score = np.arange(0, max(ppo_scores + (baseline_scores or [0])) + 2)

    axes[0, 0].hist(ppo_scores, bins=bins_score, alpha=0.7, color="tab:blue", label="PPO")
    if has_baseline:
        axes[0, 0].hist(baseline_scores, bins=bins_score, alpha=0.5, color="tab:orange", label="rule-based")
        axes[0, 0].legend()
    axes[0, 0].set_xlabel("score (forward steps)")
    axes[0, 0].set_ylabel("liczba epizodow")
    axes[0, 0].set_title("Histogram score")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(ppo_rewards, bins=20, alpha=0.7, color="tab:blue")
    axes[0, 1].set_xlabel("total reward")
    axes[0, 1].set_ylabel("liczba epizodow")
    axes[0, 1].set_title("Histogram reward (PPO)")
    axes[0, 1].grid(True, alpha=0.3)

    if has_baseline:
        # 2x2 layout
        axes[1, 0].hist(ppo_lengths, bins=20, alpha=0.7, color="tab:blue", label="PPO")
        axes[1, 0].hist([b for b in baseline_scores], bins=20, alpha=0.0)  # placeholder for symmetry
        axes[1, 0].set_xlabel("length (steps)")
        axes[1, 0].set_ylabel("liczba epizodow")
        axes[1, 0].set_title("Histogram dlugosc epizodu")
        axes[1, 0].grid(True, alpha=0.3)

        x = np.arange(2)
        ppo_mean = np.mean(ppo_scores)
        rule_mean = np.mean(baseline_scores)
        axes[1, 1].bar(x, [ppo_mean, rule_mean], color=["tab:blue", "tab:orange"])
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(["PPO", "rule-based"])
        axes[1, 1].set_ylabel("Sredni score")
        axes[1, 1].set_title("Porownanie srednich score")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    else:
        axes[0, 2].hist(ppo_lengths, bins=20, alpha=0.7, color="tab:blue")
        axes[0, 2].set_xlabel("length (steps)")
        axes[0, 2].set_ylabel("liczba epizodow")
        axes[0, 2].set_title("Histogram dlugosc epizodu")
        axes[0, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close(fig)
    print(f"[OK] Zapisano histogram -> {output}")


def main() -> None:
    args = parse_args()

    if sys.version_info >= (3, 13):
        print(
            "[BLAD] stable-baselines3 nie obsluguje Pythona >= 3.13.",
            file=sys.stderr,
        )
        sys.exit(1)

    from stable_baselines3 import PPO

    model_path = _resolve_model_path(args.model)
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Epizody: {args.episodes}  seed: {args.seed}-{args.seed + args.episodes - 1}")
    print(f"[INFO] deterministic: {args.deterministic}")

    model = PPO.load(str(model_path))

    env = gym.make("gymnasium_env/CrossyRoad-v0", observation_mode="local")
    print("\n[RUN] PPO ...")
    ppo_scores, ppo_rewards, ppo_lengths = evaluate_ppo(
        model=model,
        env=env,
        episodes=args.episodes,
        base_seed=args.seed,
        deterministic=args.deterministic,
    )
    env.close()
    _print_stats("PPO (trained)", ppo_scores, ppo_rewards, ppo_lengths)

    baseline_scores = None
    if args.baseline:
        # Rule-based agent czyta `grid` z observation -- uzywamy tu trybu large_discrete,
        # zeby agent mial dostep do pelnej planszy jak w `run_crossy_road.py`.
        env_rb = gym.make("gymnasium_env/CrossyRoad-v0", observation_mode="large_discrete")
        print("\n[RUN] rule-based StrategicCrossyAgent ...")
        rb_scores, rb_rewards, rb_lengths = evaluate_rule_based(
            env=env_rb,
            episodes=args.episodes,
            base_seed=args.seed,
        )
        env_rb.close()
        _print_stats("rule-based StrategicCrossyAgent", rb_scores, rb_rewards, rb_lengths)
        baseline_scores = rb_scores

    plot_histograms(
        output=args.output,
        ppo_scores=ppo_scores,
        ppo_rewards=ppo_rewards,
        ppo_lengths=ppo_lengths,
        baseline_scores=baseline_scores,
    )


if __name__ == "__main__":
    main()
