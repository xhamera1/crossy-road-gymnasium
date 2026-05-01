"""Generuje krzywe uczenia z plikow Monitor (training_logs/monitor_*.csv).

Tworzy wykresy:
  - reward (suma nagrod) per epizod (z wygladzaniem rolling mean)
  - dlugosc epizodu (steps) per epizod (z wygladzaniem rolling mean)
  - reward jako funkcja skumulowanych krokow srodowiska (timesteps)

Wynik zapisuje do training_logs/learning_curves.png. Skrypt mozna uruchamiac
samodzielnie lub jest wywolywany na koncu train_ppo.py.

Uruchomienie:
    python plot_learning_curves.py
    python plot_learning_curves.py --window 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "training_logs"
OUTPUT_PNG = LOG_DIR / "learning_curves.png"


def _read_monitor(path: Path) -> pd.DataFrame:
    """Wczytuje plik Monitor CSV (pierwsza linia to JSON-comment z metadana)."""
    df = pd.read_csv(path, skiprows=1)
    # Kolumny: r (reward), l (episode length), t (wall-clock time od startu)
    df = df.rename(columns={"r": "reward", "l": "length", "t": "time"})
    df["episode"] = df.index
    df["timesteps"] = df["length"].cumsum()
    return df


def plot_learning_curves(window: int = 50, output: Path = OUTPUT_PNG) -> Path:
    monitors = sorted(LOG_DIR.glob("monitor_*.csv"))
    if not monitors:
        raise FileNotFoundError(
            f"Nie znaleziono plikow monitor_*.csv w {LOG_DIR}. "
            "Najpierw uruchom: python train_ppo.py"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    all_rewards: list[pd.Series] = []
    all_lengths: list[pd.Series] = []

    for path in monitors:
        df = _read_monitor(path)
        if df.empty:
            continue
        smoothed_r = df["reward"].rolling(window=window, min_periods=1).mean()
        smoothed_l = df["length"].rolling(window=window, min_periods=1).mean()
        label = path.stem
        axes[0].plot(df["episode"], smoothed_r, label=label, alpha=0.7)
        axes[1].plot(df["episode"], smoothed_l, label=label, alpha=0.7)
        axes[2].plot(df["timesteps"], smoothed_r, label=label, alpha=0.7)
        all_rewards.append(df["reward"])
        all_lengths.append(df["length"])

    if all_rewards:
        combined_r = pd.concat(all_rewards, ignore_index=True)
        combined_l = pd.concat(all_lengths, ignore_index=True)
        mean_r = combined_r.rolling(window=window, min_periods=1).mean()
        mean_l = combined_l.rolling(window=window, min_periods=1).mean()
        axes[0].plot(mean_r.index, mean_r, label="srednia (wszystkie envy)", color="black", linewidth=2)
        axes[1].plot(mean_l.index, mean_l, label="srednia (wszystkie envy)", color="black", linewidth=2)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Reward (rolling mean, window={window})")
    axes[0].set_title("Krzywa uczenia - reward per epizod")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Episode length (rolling mean, window={window})")
    axes[1].set_title("Krzywa uczenia - dlugosc epizodu")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Timesteps (skumulowane)")
    axes[2].set_ylabel(f"Reward (rolling mean, window={window})")
    axes[2].set_title("Krzywa uczenia - reward vs timesteps")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120)
    plt.close(fig)
    print(f"[OK] Zapisano krzywe uczenia -> {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves from Monitor CSVs")
    parser.add_argument("--window", type=int, default=50, help="Rolling-mean window size")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help="Sciezka wyjsciowa PNG",
    )
    parser.add_argument("--show", action="store_true", help="Pokaz okno matplotlib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_learning_curves(window=args.window, output=args.output)
    if args.show:
        img = plt.imread(output)
        plt.figure(figsize=(18, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
