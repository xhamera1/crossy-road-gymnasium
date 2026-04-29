# crossy-road-gymnasium

Custom Gymnasium environment for a simplified Crossy Road game.

## Status (requirements 6/8)

- [x] Custom Gymnasium environment (`CrossyRoadEnv`)
- [x] Large discrete observation space (`observation_mode="large_discrete"`)
- [x] Agent playing the game (`StrategicCrossyAgent` in `crossy_agent.py`)
- [x] Agent uses deterministic strategy (not random)
- [x] Graphical mode (`render_mode="human"`) with Pygame

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
  - `large_discrete` (default, recommended):
    - `grid` (full board),
    - `lane_directions` (`MultiDiscrete([3] * height)`),
    - `lane_speeds` (`MultiDiscrete([4] * height)`),
    - `agent_position` (`MultiDiscrete([height, width])`)
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

## Files

- `gymnasium_env/core/config.py` - game constants and tuning
- `gymnasium_env/core/engine.py` - main game simulation engine
- `gymnasium_env/core/generator.py` - lane and obstacle generation
- `gymnasium_env/renderers/pygame_renderer.py` - graphical renderer
- `gymnasium_env/renderers/ansi_renderer.py` - terminal renderer
- `gymnasium_env/envs/crossy_road.py` - Gymnasium API integration layer
- `crossy_agent.py` - strategic agent
- `run_crossy_road.py` - ANSI simulation script
- `run_crossy_road_gui.py` - GUI simulation script
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
