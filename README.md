# crossy-road-gymnasium

Custom Gymnasium environment for a simplified Crossy Road game.

## Status (requirements 6/8)

- [x] Custom Gymnasium environment (`CrossyRoadEnv`)
- [x] Large discrete observation space (`observation_mode="large_discrete"`)
- [x] Agent playing the game (`StrategicCrossyAgent` in `crossy_agent.py`)
- [x] Agent uses deterministic strategy (not random)
- [x] Graphical mode (`render_mode="human"`) with Pygame
- [x] Trained ML agent (PPO from Stable-Baselines3) with learning curves

## Implemented Features

- Environment package structure:
  - `gymnasium_env/envs/crossy_road.py`
  - registration in `gymnasium_env/__init__.py`
  - env ID: `gymnasium_env/CrossyRoad-v0`
- Core game mechanics:
  - grid `8 x 50`
  - terrains: Grass, Road, River
  - dynamic entities: Cars and Lily pads
  - scrolling world on forward move
  - rewards: `+1` forward, `0` lateral/back, `-100` on death
- Action space:
  - `Discrete(4)` with mapping `0=Up`, `1=Down`, `2=Left`, `3=Right`
- Observation modes:
  - `grid`: basic matrix `(50, 8)` with IDs `0..5`
  - `large_discrete` (default, used by the rule-based agent):
    - `grid` (full board),
    - `lane_directions` (`MultiDiscrete([3] * height)`),
    - `lane_speeds` (`MultiDiscrete([4] * height)`),
    - `agent_position` (`MultiDiscrete([height, width])`)
  - `local` (used by the PPO trained agent): cropped to `view_height=10`
    nearest rows plus their dynamics. ~160-dim observation instead of ~808,
    forces the policy to focus on the agent's neighborhood.
- Rendering:
  - `ansi`: text mode in terminal
  - `human`: graphical window (Pygame, richer tile renderer)
- Agent:
  - deterministic risk-aware policy,
  - prioritizes safe forward progress and avoids hazards
- Modular architecture (professional structure):
  - `gymnasium_env/core/` -> config, constants, entities, lane factory, engine
  - `gymnasium_env/renderers/` -> ANSI and Pygame render backends
  - `gymnasium_env/envs/` -> Gymnasium wrapper API layer

## Quick Start

### 1) Create virtual environment

```bash
python -m venv .venv
```

### 2) Activate virtual environment

Git Bash (MINGW64):

```bash
source .venv/Scripts/activate
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

Option A (recommended, package mode):

```bash
pip install -e .
```

Option B (requirements file):

```bash
pip install -r requirements.txt
```

Recommended Python version for GUI: `3.10-3.12` (Pygame support is more stable than on `3.14`).

### 4) Run demo (agent + environment)

```bash
python run_crossy_road.py
```

### 5) Run graphical mode (Pygame window)

```bash
python run_crossy_road_gui.py
```

This command opens a game window.  
If no window appears, verify that `pygame` is installed and that you are not using Python 3.14.

GUI controls (`run_crossy_road_gui.py`):
- `SPACE` - one agent move (step-by-step observation)
- `R` - reset current episode
- `ESC` - quit the app
- After death, episode restarts automatically (auto-restart)

## Machine learning (PPO trained agent)

The repository ships with a full reinforcement learning pipeline based on
[Stable-Baselines3](https://stable-baselines3.readthedocs.io). The trained
agent learns from scratch — there is no hand-written strategy involved — and
the training process produces learning curves you can inspect.

### Important: Python version

`stable-baselines3` requires **Python 3.10 - 3.12** (it does not support
Python 3.13 or 3.14). If `python --version` reports `3.13` or `3.14`, create
a fresh venv with the right interpreter, e.g.:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate     # Git Bash
# .\.venv\Scripts\Activate.ps1    # PowerShell
pip install -e .                  # installs SB3, matplotlib, tensorboard, ...
```

### Train the agent

```bash
python train_ppo.py                              # 200_000 timesteps, 4 envs
python train_ppo.py --timesteps 500000 --n-envs 8
```

Training produces:

- `models/ppo_crossy_final.zip` - final PPO model
- `models/best/best_model.zip`  - best model selected by `EvalCallback`
- `models/checkpoints/`         - periodic checkpoints
- `training_logs/monitor_*.csv` - per-episode reward / length / time
- `training_logs/tensorboard/`  - TensorBoard logs
- `training_logs/learning_curves.png` - learning curves rendered with matplotlib

Live training metrics (mean reward, episode length, value loss, etc.):

```bash
tensorboard --logdir training_logs/tensorboard
```

### Re-render the learning curves any time

```bash
python plot_learning_curves.py
python plot_learning_curves.py --window 100 --show
```

### Watch the trained agent play

```bash
python play_trained_gui.py                   # opens Pygame window
python play_trained_gui.py --auto            # auto-play from start
python play_trained_gui.py --model models/best/best_model.zip
```

### Quantitative evaluation (N episodes)

```bash
python evaluate.py --episodes 50              # PPO only
python evaluate.py --episodes 50 --baseline   # PPO vs rule-based comparison
```

Generates per-episode statistics (mean / median / max / std for score, reward,
length) and saves histograms to `training_logs/evaluation.png`.

GUI controls (`play_trained_gui.py`):
- `SPACE` - one step of the trained agent
- `A`     - toggle auto-play
- `R`     - reset current episode
- `ESC`   - quit

The rule-based `StrategicCrossyAgent` (in `run_crossy_road.py` /
`run_crossy_road_gui.py`) is kept as a deterministic baseline for comparison.

## Files

- `gymnasium_env/core/config.py` - game constants and tuning
- `gymnasium_env/core/engine.py` - main game simulation engine
- `gymnasium_env/core/generator.py` - lane and obstacle generation
- `gymnasium_env/renderers/pygame_renderer.py` - graphical renderer
- `gymnasium_env/renderers/ansi_renderer.py` - terminal renderer
- `gymnasium_env/envs/crossy_road.py` - Gymnasium API integration layer
- `crossy_agent.py` - strategic agent
- `run_crossy_road.py` - ANSI simulation script (rule-based baseline)
- `run_crossy_road_gui.py` - GUI simulation script (rule-based baseline)
- `train_ppo.py` - PPO training pipeline (Stable-Baselines3)
- `plot_learning_curves.py` - generates `training_logs/learning_curves.png`
- `play_trained_gui.py` - run the trained PPO agent in the Pygame window
- `evaluate.py` - N-episode evaluation with histograms (`evaluation.png`)
- `pyproject.toml` / `requirements.txt` - dependencies

```python
import gymnasium as gym
import gymnasium_env

env = gym.make(
    "gymnasium_env/CrossyRoad-v0",
    render_mode="ansi",
    observation_mode="large_discrete",
)
obs, info = env.reset(seed=42)
```
