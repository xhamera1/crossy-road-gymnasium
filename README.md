# crossy-road-gymnasium

A custom Gymnasium environment imitating a simplified Crossy Road game, together with a **trained PPO agent** (Stable-Baselines3) and a deterministic rule-based agent as a baseline.

## Demo gameplay — trained agent in action (YouTube)

 A short **demo video** — a trained PPO agent (the `best_model.zip` model from `EvalCallback`) plays in the Pygame GUI with real graphics (chicken, cars, river, lily pads). The recording shows the agent's strategy: when it moves forward, when it waits on the road, and when it dodges sideways.

> **Video link**: [https://youtu.be/M1ANcz5gUqU](https://youtu.be/M1ANcz5gUqU)

## Results

Training **PPO for 300,000 timesteps** (~17.5 min on CPU) with a best-checkpoint selection mechanism from 17 saved models gives results **competitive with the rule-based agent** despite using a 5× smaller observation (`local`, ~160 features vs ~808):

### Deterministic evaluation, N=50 episodes


| metric          | PPO (trained) | StrategicCrossyAgent | Delta (PPO - rule)          |
| --------------- | ------------- | -------------------- | --------------------------- |
| **mean score**  | 9.66          | 10.24                | -0.58 *(94% of rule level)* |
| median score    | 6.0           | 7.5                  | -1.5                        |
| **max score**   | **45**        | 40                   | **+5** ✓                    |
| std score       | 8.66          | 7.87                 | +0.79                       |
| **mean reward** | +2.657        | +3.317               | -0.66                       |
| **max reward**  | **+34.05**    | +30.40               | **+3.65** ✓                 |
| min reward      | **-4.30**     | -4.60                | +0.30 ✓                     |
| mean length     | 13.00         | 14.84                | -1.84                       |


PPO **beats the rule-based baseline on peaks** (max score 45, max reward +34.05), while also being **more stable in the lower tail** (min reward -4.30 vs -4.60), and its average reaches 94% of the deterministic heuristic level. Full analysis is in `trening.ipynb` (section 9).

### Final training metrics (300k timesteps)


| Metric                     | Value              |
| -------------------------- | ------------------ |
| `eval/mean_reward`         | **+4.62 +/- 7.07** |
| `eval/mean_ep_length`      | 15.1 +/- 10.9      |
| `train/explained_variance` | 0.26               |
| `train/entropy_loss`       | -0.90              |
| Training time              | 1057 s ~ 17.5 min  |


## Notebook — `trening.ipynb`

The **complete project delivery** is the `trening.ipynb` notebook, which includes:

- an inline Gymnasium environment definition (self-contained, works in Colab),
- a full PPO training pipeline,
- learning curves from `Monitor` CSV,
- quantitative evaluation N=50 vs the rule-based agent,
- automatic selection of the best checkpoint (section 5.3),
- the history of 4 reward-shaping iterations,
- analysis and final conclusions.

The notebook is most convenient to open with the `.venv (Python 3.12)` kernel — installing `ipykernel` may be required on first open.

## Quick start

### 1. Virtual environment (Python 3.10-3.12)

`stable-baselines3` **does not support Python 3.13/3.14**. If `python --version` shows `3.13`+, install 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # PowerShell
# source .venv/Scripts/activate         # Git Bash
```

### 2. Install dependencies

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Dependency list: `gymnasium`, `numpy`, `pygame`, `stable-baselines3`, `matplotlib`, `pandas`, `tensorboard`.

### 3. Run the trained agent in GUI

The repository already contains trained models in `models/`:

```bash
python play_trained_gui.py --auto
```

**Controls**: `SPACE` = one step, `A` = toggle auto-play, `R` = reset episode, `ESC` = quit.

### 4. (Optional) Open the training notebook

```bash
jupyter notebook trening.ipynb
# or open the file in Cursor / VS Code
```

## Running — all scripts

### Rule-based agent (baseline)

```bash
python run_crossy_road.py          # ANSI mode in terminal
python run_crossy_road_gui.py      # Pygame GUI
```

### PPO training (CLI script, alternative to notebook)

```bash
python train_ppo.py                                # 200k timesteps (default), 4 envs
python train_ppo.py --timesteps 300000             # recommended (~17 min CPU)
python train_ppo.py --timesteps 500000 --n-envs 8  # more exploration
```

Training produces:

- `models/ppo_crossy_final.zip` — final model (last iteration)
- `models/best/best_model.zip` — best model selected by `EvalCallback`
- `models/checkpoints/ppo_crossy_*_steps.zip` — checkpoints every 20,000 steps
- `training_logs/monitor_*.csv` — per-episode (reward / length / time)
- `training_logs/tensorboard/` — TensorBoard logs
- `training_logs/learning_curves.png` — matplotlib plots

### TensorBoard (live monitoring during training)

```bash
tensorboard --logdir training_logs/tensorboard
```

### Learning-curve plots (if monitor CSV files exist)

```bash
python plot_learning_curves.py
python plot_learning_curves.py --window 100 --show
```

### Trained agent in Pygame GUI

```bash
python play_trained_gui.py                                                      # default model + step mode
python play_trained_gui.py --auto                                               # auto-play immediately
python play_trained_gui.py --auto --model models/best/best_model.zip            # specific file
python play_trained_gui.py --auto --stochastic                                  # sampling instead of greedy
python play_trained_gui.py --auto --fps 4                                       # slower
```

### Quantitative evaluation (N episodes + histograms)

```bash
python evaluate.py --episodes 50              # PPO only
python evaluate.py --episodes 50 --baseline   # PPO vs rule-based + comparative bar chart
```

Output is saved to `training_logs/evaluation.png` plus statistics in stdout.

## Crossy Road environment

### Board and tiles

- **Dimensions**: 50 rows x 8 columns (world scrolls forward on UP move).
- **Tiles**: grass (safe), road (with cars, fatal collision), river (fatal if not standing on a lily pad).
- **Bottom-buffer**: the bottom row is a "trail" — the agent always sees where it came from.

### Actions

`Discrete(4)` — `0=UP`, `1=DOWN`, `2=LEFT`, `3=RIGHT`.

### Observation modes


| mode             | shape               | usage                                    |
| ---------------- | ------------------- | ---------------------------------------- |
| `grid`           | `(50, 8)`           | simple matrix, for convolutional agents  |
| `large_discrete` | dict, ~808 features | `**StrategicCrossyAgent`** (rule-based)  |
| `local`          | dict, ~160 features | **PPO** — cropped to the 10 nearest rows |


### Reward shape (after 4 iteration rounds)


| reward            | value   | role                                                                           |
| ----------------- | ------- | ------------------------------------------------------------------------------ |
| `reward_forward`  | `+1.0`  | bonus for a successful UP move (with scrolling)                                |
| `reward_step`     | `-0.1`  | per-step penalty — discourages standing still (200 standing steps = -20)       |
| `reward_death`    | `-5.0`  | death (car/drowning/pushed off-board by current)                               |
| `reward_survival` | `+0.05` | bonus for surviving a step on road/river — allows actually *waiting* for a gap |


Full history of the 4 reward-shaping iterations (from `+1/0/-100`, which taught standing still, to the current shape) — in the notebook, section 8.

### PPO hyperparameters

```
algorithm:        PPO (Stable-Baselines3 v2.8)
policy:           MultiInputPolicy + net_arch=[256, 256]
n_envs:           4 (DummyVecEnv)
n_steps:          512
batch_size:       128
n_epochs:         10
learning_rate:    linear schedule 3e-4 -> 0
gamma:            0.99
gae_lambda:       0.95
clip_range:       0.2
ent_coef:         0.1   (strong exploration)
total_timesteps:  300_000
max_episode_steps: 200
seed:             0
```

## Project architecture

```
crossy-road-gymnasium/
├── gymnasium_env/
│   ├── core/                    # game logic (env-agnostic)
│   │   ├── config.py            # GameConfig (dimensions, reward shape, probabilities)
│   │   ├── constants.py         # CellID, ActionID, ACTION_DELTAS, ANSI_SYMBOLS
│   │   ├── entities.py          # Lane (board lane)
│   │   ├── generator.py         # LaneFactory (random lane generation)
│   │   └── engine.py            # CrossyRoadEngine (game step, observations, collisions)
│   ├── renderers/
│   │   ├── ansi_renderer.py     # terminal rendering
│   │   └── pygame_renderer.py   # Pygame GUI with graphics (chicken, cars, river)
│   ├── envs/
│   │   └── crossy_road.py       # CrossyRoadEnv (Gymnasium API wrapper)
│   └── __init__.py              # registration `gymnasium_env/CrossyRoad-v0`
├── crossy_agent.py              # StrategicCrossyAgent (rule-based baseline)
├── run_crossy_road.py           # ANSI demo with rule-based agent
├── run_crossy_road_gui.py       # GUI demo with rule-based agent
├── train_ppo.py                 # PPO training (CLI)
├── play_trained_gui.py          # GUI with trained agent
├── evaluate.py                  # evaluation N episodes + histograms
├── plot_learning_curves.py      # plots from monitor CSV
├── trening.ipynb                # ★ PROJECT DELIVERY ★ — everything in one notebook
├── models/                      # trained models
│   ├── ppo_crossy_final.zip     # default loaded by play_trained_gui.py
│   ├── best/best_model.zip      # best model from EvalCallback
│   └── checkpoints/             # snapshot every 20k steps
├── training_logs/               # monitor CSV + TensorBoard + PNG
├── pyproject.toml
├── requirements.txt
└── README.md
```

