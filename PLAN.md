# PLAN — kolejne kroki implementacyjne

Dokument planuje rozwój projektu `crossy-road-gymnasium`. Część pierwsza opisuje
iterację realizowaną w ramach bieżącego zlecenia (29.04.2026), część druga to
backlog zadań na później.

## Iteracja w trakcie

### A. Sprite kurczaka zamiast żółtego pola agenta

Cel: w trybie `render_mode="human"` zamiast jednolitego żółtego prostokąta z
białym wkładem ma być rysowany obrazek `chicken-model.jpeg` (znajduje się
w katalogu głównym projektu).

Kroki:

1. W `gymnasium_env/renderers/pygame_renderer.py` dodać ścieżkę do pliku
   (`Path(__file__).resolve().parents[2] / "chicken-model.jpeg"`), żeby
   działała niezależnie od bieżącego katalogu uruchomieniowego.
2. Sprite ładowany leniwie w `_ensure_ready` (po zainicjalizowaniu okna
   Pygame, bo `convert_alpha` wymaga aktywnego trybu wideo).
3. JPEG nie ma kanału alfa, więc po `convert_alpha()` należy oznaczyć
   piksele zbliżone do białego (R/G/B > 235) jako przezroczyste przy pomocy
   `pygame.surfarray`. Dzięki temu kurczak nie ma białego prostokąta tła.
4. Sprite skalować do ok. 90% rozmiaru kafelka (`smoothscale`) i wyśrodkować
   w komórce.
5. Aby kurczak nie przykrywał terenu pod sobą (bo standardowy `grid` ma
   `CellID.AGENT` zamiast właściwego terenu), w silniku wprowadzić nową
   metodę `engine.terrain_grid()` zwracającą siatkę BEZ markera agenta.
   Renderer dostaje siatkę terenu + osobno pozycję agenta i blituje sprite
   kurczaka na samym końcu.
6. Fallback: jeżeli plik z kurczakiem nie istnieje albo `pygame.image.load`
   rzuci wyjątkiem, renderer cofa się do dotychczasowego żółtego pola
   (zachowanie wsteczne, brak crashu).

### B. Szybsze samochody

Cel: samochody na drodze poruszają się szybciej niż lilie na rzece, żeby gra
była bardziej dynamiczna.

Kroki:

1. W `GameConfig` rozdzielić pojedyncze `lane_speed_values` na dwa pola:
   - `road_speed_values: tuple[int, ...] = (1, 2)` — `move_interval=1` oznacza
     ruch co tick (najszybszy możliwy), `2` co drugi tick.
   - `river_speed_values: tuple[int, ...] = (2, 3)` — lilie zostają wolniejsze.
2. `LaneFactory.create_lane` musi wybrać odpowiedni zestaw zależnie od
   `TerrainType`.
3. Sprawdzić, że obserwacja `lane_speeds` (`MultiDiscrete([4] * height)`)
   nadal mieści maksimum 3 — tak, mieści.

### C. Wiersz „śladu" pod kurczakiem

Cel: kurczak nigdy nie stoi na ostatnim widocznym wierszu — pod nim cały czas
widać jeden wiersz pokazujący, skąd właśnie przyszedł.

Kroki:

1. Dodać `bottom_padding_rows: int = 1` do `GameConfig`.
2. W `engine.reset` startowa pozycja: `agent_y = bottom_padding_rows`
   (zamiast 0).
3. W `engine._scroll_world`: `agent_y = max(bottom_padding_rows, agent_y - 1)`.
4. W `engine._in_bounds`: dolna granica `y` to `bottom_padding_rows`. Akcja
   `Down` z minimalnego wiersza zostaje odrzucona (brak ruchu, brak nagrody),
   tak samo jak `Up` z górnego wiersza w wersji bazowej.
5. Skutek wizualny: kurczak siedzi „o jedno wyżej" niż dotąd, a wiersz pod
   nim pokazuje teren, na którym był poprzedni krok temu.

## Refaktor towarzyszący

- Dodać `engine.terrain_grid()` (czysty teren + samochody + lilie, bez
  markera agenta) — używana wyłącznie przez Pygame renderer.
- `engine.grid_observation()` zostaje bez zmian, żeby nie psuć kontraktu
  `observation_space` dla agentów RL.
- Renderer Pygame przyjmuje teraz `(grid_terrain, agent_pos)` — drobna
  zmiana sygnatury, lokalna do `_render_human`.

## Plan testowy

1. `python run_crossy_road.py` (tryb ANSI) — agent powinien wciąż grać,
   `score` rośnie, brak wyjątków. ANSI ignoruje sprite kurczaka.
2. `python run_crossy_road_gui.py` — okno Pygame, kurczak widoczny w polu
   agenta, samochody przesuwają się co klatkę / co dwie klatki, pod kurczakiem
   stale jest jeden wiersz „śladu".
3. Wymusić brak pliku `chicken-model.jpeg` (chwilowa zmiana nazwy) — gra
   działa, fallback na żółty kwadrat.
4. `seed=42` startowy: agent wstaje od `y=1` zamiast `y=0`.

## Iteracja: cropped obs + survival bonus (29.04.2026, runda 4)

Po rundzie 3 (`death=-5`, `[256,256]`, 300k timesteps + linear LR) agent
osiagnal score 18 w najlepszym epizodzie i robil pierwsze uniki, ale srednia
zostala oscylujaca w pasie -2 do +1 (plateau). Identyfikowane bottleneck-i:

- **808-wymiarowa obserwacja** (50 rzedow x 8 + onehoty) -- wiekszosc to noise
  (rzedy odlegle nie wplywaja na decyzje), siec tonie w danych.
- **`explained_variance = 0.19`** -- value-fn slabo przewiduje reward.
- **Brak mechanizmu zachecajacego do "stania na drodze i czekania"** -- step
  penalty -0.1 sprawia, ze kazdy ruch boczny na drodze kosztuje, wiec agent
  woli charge.

### Naprawa nr 4

#### A. Cropped observation (`observation_mode="local"`)

Nowy tryb obserwacji w `CrossyRoadEnv`:

- `view_height=10` (parametr konstruktora, default 10).
- Obserwacja zawiera tylko 10 najnizszych rzedow planszy (te ktore sa wokol
  agenta i tuz nad nim), nie cale 50.
- Wymiar po one-hot: `10*8 (grid) + 10*3 (dirs) + 10*4 (speeds) + 2+8 (pos)`
  = **160 cech**, vs ~808 w `large_discrete`. **5x mniej wymiarow.**
- `observation_space.agent_position` to `MultiDiscrete([2, width])` -- agent_y
  tylko {0, 1} dzieki bottom-buffer.

Zachowane wsteczna kompatybilnosc:
- `large_discrete` i `grid` dzialaja jak dotad (uzywane przez agenta regulowego).
- `local` jest opcjonalne, wybierane przez `train_ppo.py` i `play_trained_gui.py`.

#### B. Survival bonus (`reward_survival = +0.05`)

`engine.step` po przezyciu kroku na ROAD/RIVER dodaje `+0.05` do rewardu:

| Sytuacja | Per-step reward |
|---|---:|
| Ruch lateral na trawie | -0.1 |
| Ruch lateral na drodze (przezyl) | **-0.1 + 0.05 = -0.05** |
| Ruch forward na trawie | +0.9 |
| Ruch forward na drodze (przezyl) | **+0.9 + 0.05 = +0.95** |
| Smierc | -5.0 (override) |

Efekt: "stoj na drodze sekunde i czekaj az auto przejedzie" kosztuje teraz
tylko -0.05 zamiast -0.1, czyli 50% taniej. Czasowy unik staje sie realna
strategia, nie tylko boczny krok.

#### C. Skrypt `evaluate.py`

`evaluate.py` -- ewaluacja N epizodow z roznymi seedami, statystyki
score / reward / length, histogramy w `training_logs/evaluation.png`. Z opcja
`--baseline` rownolegle ewaluuje rule-based agenta i tworzy wykres porownawczy.

### Spodziewana matematyka

EV(forward) na grass: `0.9 * 0.9 + 0.1 * (-5) = +0.31`.
EV(forward) na road, dodging: `0.95 * 0.95 + 0.05 * (-5) = +0.65` (znacznie lepiej!).

Przy wiekszej obserwowalnej "objetosci" pasow przed soba (10 rzedow w obserwacji
zamiast wszystkich 50 ale rozproszonych), siec [256, 256] powinna nauczyc sie
mapowania "zobacz auto -> dodge" znacznie szybciej.

### Plan testowy

1. `rm training_logs/monitor_*.csv && rm -rf training_logs/eval`
   (czysta krzywa).
2. `mv models models_v3_kamikaze` (backup poprzedniego).
3. `python train_ppo.py` -- 300k timesteps.
4. `python play_trained_gui.py --auto`.
5. **`python evaluate.py --episodes 50 --baseline`** -- liczby do sprawozdania,
   bezposrednie porownanie PPO vs rule-based.

Cel: srednia score >= 15, max score >= 30.

## Iteracja: kamikadze fix (29.04.2026, runda 3)

Po rundzie 2 (`reward_step=-0.1`, `reward_death=-1`, `ent_coef=0.1`) agent
zaczal isc do przodu, ale wpadl w **tryb kamikadze**: `UP, UP, UP, smierc po
3-7 krokach`. W GUI: `score = 3-7`, max ~15. Krzywa uczenia urosla z -8 do +3
i tam zaplateauala.

### Diagnoza

- `eval/mean_reward = -3.40 +/- 10.18` -- ogromna wariancja (czasem 200 length,
  czasem 5).
- `explained_variance = 0.05` -- value-fn nie ogarnia rewardu, bo zachowanie
  jest chaotyczne.
- Matematyka: `EV(forward) = 0.9*0.9 + 0.1*(-1) = +0.71` -- "zawsze do przodu"
  jest dodatnie EV, bo smierc (-1) tania. Agent nie ma motywacji unikow.
- Lateral move kosztuje `-0.1` (step penalty) i nie ma natychmiastowej nagrody.
  Greedy agent wybiera `+0.9 forward` zamiast `-0.1 lateral`.

### Naprawa nr 3

| Parametr | poprzednio | teraz |
|---|---:|---:|
| `reward_death` | -1.0 | **-5.0** |
| `learning_rate` | const 3e-4 | **linear schedule** 3e-4 -> 0 |
| `policy_kwargs.net_arch` | domyslne `[64, 64]` | **`[256, 256]`** (pi i vf) |
| `--timesteps` default | 200_000 | **300_000** |

### Spodziewana matematyka

| Strategia | Reward |
|---|---:|
| Stac caly epizod | -20 |
| Zginac od razu | -5 |
| Kamikadze 1 forward | -4.1 |
| Kamikadze 5 forward | -0.5 |
| Smart play 10 forward + smierc | +4 |
| Smart play 20 forward + smierc | +13 |

EV(forward) przy 90% live = `+0.31` (oplaca sie isc).
EV(forward) przy 80% live = `-0.28` (woli zrobic unik!).

Czyli polityka powinna nauczyc sie odczytywac obecnosc samochodu przed soba
(z `grid` + `lane_directions` + `lane_speeds`) i robic uniki bocznymi ruchami.

### Wieksza siec -- po co

50-rzedowa plansza + onehot encoding = **808 wymiarow obserwacji**. Domyslny
MLP `[64, 64]` ma za malo pojemnosci, zeby z tego wyciagnac dynamike pasow.
`[256, 256]` zwieksza pojemnosc ~16x. Zostaje to samo PPO + Adam, wiec
trening dalej stabilny.

### Linear LR schedule -- po co

W rundzie 2 widzielismy: szybki wzrost do +3 w 50k krokow, potem chaotyczne
oscylacje 0-4 przez kolejne 150k. To klasyczny objaw: `lr` za wysoki w fazie
fine-tuningu. Linear schedule `lr -> 0` sprawia, ze pod koniec treningu
polityka sie konsoliduje.

### Plan testowy

1. `rm training_logs/monitor_*.csv` -- czyste krzywe (rozne reward shape).
2. `mv models models_v2_kamikaze` -- zachowanie poprzedniego modelu na backup.
3. `python train_ppo.py` -- 300k timesteps, ~25-30 min na CPU.
4. `python play_trained_gui.py --auto` -- agent powinien teraz robic uniki
   gdy widzi samochod, score docelowy 20+.

## Iteracja: agresywny reward shaping (29.04.2026, runda 2)

Po pelnym treningu 200_000 timesteps z reward shape `+1 / -0.05 / -10` agent
nadal utknal -- `eval/mean_reward = -8.30`, `mean_ep_length = 200.00 ± 0.00`.
Zachowanie obserwowane w `play_trained_gui.py`: 1-2 ruchy w gore, potem stale
ruchy w bok / odrzucone "Down" -- agent siedzi na trawie do limitu.

Powod: nawet po podniesieniu kary za "stanie", optimum lokalne `2 forward + sit`
(= `-8`) bylo lepsze niz `10 forward + die` (= `-9`).

### Naprawa nr 2

| Parametr | poprzednio | teraz |
|---|---:|---:|
| `reward_step` | -0.05 | **-0.1** |
| `reward_death` | -10 | **-1.0** |
| `reward_forward` | +1 | +1 (bez zmian) |
| `ent_coef` (PPO) | 0.05 | **0.1** |

### Spodziewana matematyka epizodu

| Strategia | Reward |
|---|---:|
| Stac caly epizod (200 krokow) | **-20** (najgorsze!) |
| Zginac od razu | -1 |
| 1 forward + smierc | -0.1 |
| 5 forward + smierc | **+3.5** |
| 20 forward + smierc | +18 |

Roznica miedzy "stac" a "1 forward i smierc" wynosi teraz **+19.9**. Tego
sygnalu polityka nie da sie zignorowac.

### Plan testowy

1. (Opcjonalnie) Zachowac stary trening: `mv models models_v1` zeby porownac.
2. Skasowac CSV-ki z poprzedniego runa: `rm training_logs/monitor_*.csv` --
   inaczej `learning_curves.png` zmiesza dwie rozne reward shape.
3. `python train_ppo.py` -- pelny trening 200k. Spodziewany czas ~30 min.
4. `python play_trained_gui.py --auto` -- agent powinien systematycznie isc
   w gore zamiast siedziec.
5. Krzywa uczenia powinna teraz **wyraznie rosnac** zamiast plateau.

## Iteracja: reward shaping (29.04.2026, po pierwszym uruchomieniu PPO)

Pierwszy trening PPO z reward shape `+1 forward / 0 lateral / -100 death`
+ `max_episode_steps=500` utknal w klasycznym lokalnym minimum:

- `eval/mean_reward = 0.0`
- `eval/mean_ep_length = 500` (czyli zawsze max)
- `explained_variance = 0.92` (siec wie, ze reward = 0)

Czyli polityka deterministycznie nauczyla sie **stac w miejscu / lazic w bok**,
bo kazdy ruch w przod = ryzyko `-100`, a stanie = `0`.

### Naprawa

1. `GameConfig`:
   - `reward_forward: float = 1.0` (typ z int->float)
   - `reward_death: float = -10.0` (zamiast -100; stosunek 10:1 zamiast 100:1)
   - `reward_step: float = -0.05` (NOWE) -- per-step penalty.
2. `CrossyRoadEngine.step` -- baseline rewardu = `reward_step` na kazdym kroku,
   `+= reward_forward` przy ruchu w przod, smierc nadpisuje (zwracamy
   `reward_death` bez sumowania, zeby kara byla stala niezaleznie od kroku).
3. `gymnasium_env/__init__.py` -- `max_episode_steps: 500 -> 200` (krotsze
   epizody, "stanie" dobija do limitu szybciej i z wieksza kara skumulowana).
4. `train_ppo.py` -- `ent_coef: 0.01 -> 0.05` (wieksza eksploracja).

### Spodziewany efekt

Po treningu (200_000 timesteps):

- `ep_rew_mean` powinien rosnac w czasie (na poczatku ~`-10`, docelowo
  dodatni gdy agent bezpiecznie idzie w przod).
- `ep_len_mean` na poczatku duzy (stoi), pozniej powinien spasc i rosnac
  w funkcji `score`, az osiagnie limit 200 przy bezpiecznej grze.
- `eval/mean_reward` > 0 dla agenta, ktory robi >=10 forward i przezywa
  do konca epizodu (`10 * 0.95 = 9.5` netto za poruszanie, minus drobne kary
  za kroki w bok).

### Plan testowy (kontynuacja)

1. `python train_ppo.py --timesteps 100000` -- szybsza weryfikacja, czy reward
   rosnie. Krzywa uczenia w `training_logs/learning_curves.png`.
2. Jezeli mimo tego utknie -- nastepny krok: `reward_step = -0.1`,
   `reward_death = -5`, lub reward shaping na nowy max-score.

## Iteracja: ML / krzywe uczenia (29.04.2026)

Cel: agent przestaje być regułowy, jest TRENOWANY, a jakość uczenia da się
zobrazować krzywymi uczenia.

### Wybór algorytmu

- PPO (Proximal Policy Optimization) z `stable-baselines3`.
- `MultiInputPolicy` -- nasza obserwacja to `Dict` (`grid` + 3 x `MultiDiscrete`),
  którą SB3 obsługuje natywnie przez `CombinedExtractor`. Bez ręcznego
  flattenowania.

### Wymagania środowiska

- `stable-baselines3` nie wspiera Pythona >= 3.13 -- trening musi iść z venv
  3.10 / 3.11 / 3.12.
- Dorzucone zależności: `stable-baselines3>=2.3`, `matplotlib>=3.7`,
  `pandas>=2.0`, `tensorboard>=2.13`.
- W `pyproject.toml` zacieśnione `requires-python = ">=3.10,<3.13"`.

### Pliki

- `train_ppo.py` -- pełen pipeline: `DummyVecEnv` (4 procesy), `Monitor` per
  środowisko, `EvalCallback` (best model), `CheckpointCallback`, TensorBoard,
  argparse na hiperparametry.
- `plot_learning_curves.py` -- czyta `training_logs/monitor_*.csv` i rysuje
  3 wykresy: reward / epizod, długość epizodu, reward / timesteps.
- `play_trained_gui.py` -- ładuje zapisany model `.zip`, gra w trybie GUI
  Pygame (SPACE = krok, A = auto-play, R = reset, ESC = wyjście).

### Krzywe uczenia

- Per-epizodowe statystyki idą do `training_logs/monitor_<seed>.csv` przez
  `Monitor` z SB3.
- Po treningu `train_ppo.py` automatycznie woła `plot_learning_curves` i
  zapisuje `training_logs/learning_curves.png`.
- Online: `tensorboard --logdir training_logs/tensorboard`.

### Plan testowy (RL)

1. `python train_ppo.py --timesteps 50000 --n-envs 2` -- smoke test, kilka minut.
2. Sprawdzenie `training_logs/learning_curves.png` -- średni reward rośnie w
   czasie.
3. `python play_trained_gui.py` -- wytrenowany agent gra w GUI, reaguje na
   nowe lane'y.
4. Porównanie z `StrategicCrossyAgent` (`run_crossy_road.py`) na N=20 epizodów
   (do dorobienia jako `evaluate.py` w kolejnej iteracji).

## Backlog (kolejne iteracje)

- [ ] `evaluate.py` -- N epizodów per agent (regułowy vs PPO), tabela mean /
      std / max score.
- [ ] Reward shaping: drobny bonus za skok na kolejną lilię, kara za bezruch
      (jeśli plain reward nie wystarcza do nauczenia rzeki).
- [ ] Profile trudności (`easy`, `normal`, `hard`) modyfikujące prędkości i
      gęstości pasów.
- [ ] Tryb sterowania klawiaturą (gracz człowiek).
- [ ] Testy jednostkowe `pytest` dla `engine` (kolizje, scroll, utonięcia).
- [ ] Replay seedów (zapis / odtworzenie trajektorii).
- [ ] Inne algorytmy RL do porównania (DQN, A2C) -- ten sam pipeline,
      podmieniony obiekt w `train_ppo.py`.
- [ ] Przeniesienie zasobów graficznych do `gymnasium_env/assets/` i
      dystrybucja przez `package_data` w `pyproject.toml`.
