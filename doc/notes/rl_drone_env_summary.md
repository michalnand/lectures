# Drone Racing Environment — `drone_env`

Single-drone FPV gate racing on top of **MuJoCo-drones-gym** (`BaseAviary`, Crazyflie CF2X).
The agent drives the four motors directly and must fly through square gates **in order**,
as many as possible before the episode ends, without touching any gate frame.

---

## 1. Task

| Item | Value |
|---|---|
| Goal | pass as many gates as possible, in order, within `EPISODE_LEN_SEC` (60 s) |
| Agent | one Crazyflie CF2X, X configuration, mass 0.027 kg, arm 0.0397 m, collision radius 0.06 m |
| Physics / control rate | 250 Hz / 50 Hz (5 physics steps per action) |
| Episode length | 60 s = 3000 control steps (if no crash) |
| Start | position (0, 0, 0.5) m, level, at rest |
| Course | random every `reset()` (`GatesRandomizer`), or fixed (`gates=` / `DEFAULT_GATES`) |
| Laps | gates wrap: after the last gate the first one is the target again |

---

## 2. Scene

### World (from `BaseAviary`)

| Element | Description |
|---|---|
| Floor | 20 x 20 m plane, checkerboard RGB (0.2, 0.3, 0.4) / (0.1, 0.2, 0.3), light grid lines (0.8, 0.8, 0.8), reflectance 0.2 |
| Sky | vertical gradient (0.3, 0.5, 0.7) to black, haze (0.15, 0.25, 0.35) |
| Light | headlight + directional light from above, no shadows |
| Drone | CF2X mesh: dark frame, light-blue propellers |

### Gates

Gates are **not** MuJoCo bodies: they are drawn into the rendered scene and their collision is
computed geometrically in `RaceAviary._update_gates()`.

| Property | Value | Constant |
|---|---|---|
| Shape | vertical square frame with a square hole, 4 box bars | |
| Outer edge | 0.70 m | `GATE_SIZE` |
| Hole edge | 0.50 m | `GATE_HOLE` |
| Bar width | 0.10 m | `(GATE_SIZE - GATE_HOLE) / 2` |
| Frame depth | 0.05 m | `GATE_DEPTH` |
| Usable hole for the drone centre | 0.38 m | hole minus collision radius on each side |
| Centre height | 0.6 - 2.0 m (randomizer) | `height_range` |

**Gate orientation (normal):**

| Gate | Normal |
|---|---|
| single | bisector of the arrival direction (from previous gate) and departure direction (to next gate) |
| stacked pair, first | bisector of arrival direction and reversed departure direction |
| stacked pair, second | same plane, opposite normal |

**Stacked pair (reverser):** two touching gates, the upper exactly `GATE_SIZE` above the lower,
consecutive in the order, lower or upper first at random (50 / 50). The drone passes the first,
turns back through the second, and the course continues back the way it came.

### Rendered overlays (`drone_render.py`)

| Element | Look |
|---|---|
| Target gate | top/bottom bars white, side bars sky blue |
| Future gates (in observation) | top/bottom sky blue, sides royal blue |
| Other gates | alternating: royal blue + black, violet + black |
| Gate label | flight order number (1-based) on the top bar |
| Course path | white lines gate to gate, alpha 0.15, 1 px |
| Flight trace (`trace=True`) | last 300 positions (6 s), 3 px line, colour sky blue (0 m/s) to violet (>= 4 m/s), alpha `0.95 * (k / n)^1.5` (older = fainter) |
| HUD (human mode) | speed (m/s) and g-force, peak-hold with 1 % decay per frame; gates passed, camera; key help. White text with black outline |

**Colours (RGB 0-1 / 0-255 / hex):**

| Name | RGB (0-1) | RGB (0-255) | Hex |
|---|---|---|---|
| `SKY_BLUE` | 0.53, 0.81, 0.98 | 135, 207, 250 | `#87CFFA` |
| `ROYAL_BLUE` | 0.25, 0.41, 0.88 | 64, 105, 224 | `#4069E0` |
| `VIOLET` | 0.54, 0.17, 0.89 | 138, 43, 227 | `#8A2BE3` |
| `BLACK` | 0.05, 0.05, 0.08 | 13, 13, 20 | `#0D0D14` |
| `WHITE` | 1.0, 1.0, 1.0 | 255, 255, 255 | `#FFFFFF` |

**Cameras** (`camera=`, key `c` cycles):

| Camera | View |
|---|---|
| `third_person` (default) | behind the drone, facing the target gate, smoothed turning (10 % per frame) |
| `track` | follows the drone from a fixed angle: 2 m, azimuth -45 deg, elevation -20 deg |
| `course` | static overview framed on all gates: azimuth -60 deg, elevation -40 deg |

Frame size 640 x 480. Human mode runs in real time in an OpenCV window;
`q` / Esc closes the window (simulation continues).

---

## 3. Actions

Shape `(4,)`, range `[-1, 1]`, one value per motor, clipped. Direct motor control, no inner controller.

| Value | Motor speed (CF2X) |
|---|---|
| -1 | 0 RPM (off) |
| 0 | hover, 14 476 RPM |
| +1 | maximum, 21 714 RPM (thrust-to-weight 2.25) |

Mapping (`BaseAviary._normalizedActionToRPM`):

```
rpm = HOVER_RPM * (1 + a)                     for a <= 0
rpm = HOVER_RPM + a * (MAX_RPM - HOVER_RPM)   for a > 0
```

Climb: all motors together. Roll / pitch: left-right / front-back difference. Yaw: clockwise vs
counter-clockwise propeller pairs.

---

## 4. Observation

Shape `(35 + 12 * num_future_gates,)` = **59** for the default `num_future_gates = 2`. float32.
Everything is drone-centric (as from onboard sensors); no world x, y and no world yaw.

| Index (K = 2) | Size | Content | Frame | Typical range |
|---|---|---|---|---|
| 0 - 3 | 4 | sin(roll), cos(roll), sin(pitch), cos(pitch) | horizon | [-1, 1] |
| 4 - 6 | 3 | angular rates wx, wy, wz | body | ~[-10, 10] rad/s |
| 7 - 8 | 2 | velocity over ground: forward, left | heading (yaw-aligned, horizontal) | ~[-5, 5] m/s |
| 9 | 1 | altitude z | world | 0.1 - 3 m |
| 10 | 1 | vertical velocity vz | world | ~[-3, 3] m/s |
| 11 - 22 | 12 | **target gate**: 4 hole corners (x, y, z each) | body | ~[-9, 9] m |
| 23 - 34 | 12 | future gate 1 | body | ~[-9, 9] m |
| 35 - 46 | 12 | future gate 2 | body | ~[-9, 9] m |
| 47 - 58 | 12 | **nearest gate** (obstacle): any gate whose centre is closest to the drone | body | ~[-2, 2] m |

Body frame: x forward, y left, z up. Corners are relative positions `corner - drone`, rotated into
the body frame. Ranges are typical, not limits: the space is declared `Box(-inf, inf)`.

**Corner order** (fixed in each gate's own frame, so the gate's facing is unambiguous):

| Corner | Gate-frame position |
|---|---|
| 0 | (+y, +z) upper left |
| 1 | (+y, -z) lower left |
| 2 | (-y, +z) upper right |
| 3 | (-y, -z) lower right |

Hole corners are at +-0.25 m from the gate centre. The nearest-gate block can repeat the target or a
future gate; it can also be a gate already passed. Its purpose is obstacle awareness.

---

## 5. Reward

Per control step:

```
alignment = min(d0, d1, d2, d3) / max(d0, d1, d2, d3)       d_i = |hole corner i of target gate - drone|
Phi       = -PROGRESS * |target gate centre - drone| + ALIGN * alignment

r = GATE_PASSED                      if the target gate was passed this step
r = Phi(t) - Phi(t-1)                otherwise                       (shaping)
r = r - CRASH_PENALTY                if the episode terminates (crash)
```

After a gate pass, Phi restarts from the new target gate (no jump reward).

| Term | Constant | Default | Typical size |
|---|---|---|---|
| Gate passed | `GATE_PASSED` | 1.0 | +1 per gate |
| Progress toward target centre | `PROGRESS` | 1.0 | 1.0 x metres gained; +0.08 per step at 4 m/s |
| Alignment change | `ALIGN` | 0.0 (off) | +-0.01 per step near the gate when enabled |
| Crash | `CRASH_PENALTY` | 2.0 | -2 once |

**Shaping is potential-based:** only changes of distance and alignment pay. Hovering earns 0, any
closed loop sums to 0, so shaping cannot be farmed and does not change the optimal policy.

**Alignment values** (1 = on the gate axis):

| Distance | on axis | 30 deg off | 60 deg off | 85 deg off |
|---|---|---|---|---|
| 0.5 m | 1.00 | 0.71 | 0.52 | 0.45 |
| 1.0 m | 1.00 | 0.80 | 0.67 | 0.62 |
| 2.0 m | 1.00 | 0.89 | 0.81 | 0.78 |

**Gate pass:** the drone centre crosses the target gate plane with its whole body inside the hole
(`|y|, |z| < GATE_HOLE/2 - 0.06` at the crossing point). Either direction, unless
`GATE_FORWARD_ONLY = True` (front to back along the normal only).

**Gate collision** (any gate): crossing a gate plane inside the outer square but not inside the
usable hole, or touching the frame slab (`|x| < GATE_DEPTH/2 + 0.06`) outside the usable hole.

### Episode end

| Type | Condition | Constant | Log flag |
|---|---|---|---|
| terminated | touched any gate frame | | `episode_done_crash = 1` |
| terminated | altitude below 0.1 m | `LOW_ALTITUDE` | `episode_done_crash = 1` |
| terminated | abs(roll) or abs(pitch) above 2.0 rad | `MAX_TILT` | `episode_done_crash = 1` |
| truncated | 60 s elapsed | `EPISODE_LEN_SEC` | `episode_done_time = 1` |
| truncated | lap limit (off) | `MAX_LAPS = None` | `episode_done_time = 1` |

---

## 6. Course generation (`GatesRandomizer`)

Random path from (0, 0) in a random direction; each gate position is a random turn plus a step ahead.
Gates are in flight order; sharp turns are allowed. Up to 30 candidate turns per position are tried to
stay inside the arena and away from earlier gates; if none fits, the best candidate is used.

| Parameter | Default | Meaning |
|---|---|---|
| `num_gates` | 8 | gate positions (each reverser adds one gate) |
| `step_range` | (1.5, 3.0) m | horizontal distance between consecutive positions |
| `max_turn_deg` | 110 | largest heading change at a single gate |
| `reverser_turn_deg` | 45 | largest heading change right after a reverser |
| `height_range` | (0.6, 2.0) m | gate centre height |
| `arena_radius` | 4.0 m | course stays inside this circle around (0, 0) |
| `min_separation` | 1.2 m | minimum horizontal distance between gate positions (best effort) |
| `stacked_prob` | 0.0 | probability of a reverser at each position |

| Statistic (2000 courses) | `stacked_prob = 0` | `stacked_prob = 0.3` |
|---|---|---|
| gates per course | 8 | 8 - 16, mean 10.4 |
| reversers per course | 0 | mean 2.4 |

Without a randomizer, `DEFAULT_GATES` is used: a fixed oval of 6 gates, 0.8 - 1.5 m high.

---

## 7. Logging

`env.get_log()` returns stats of the **last finished episode** (all zero until one finishes):

| Key | Meaning |
|---|---|
| `num_episodes` | finished episodes |
| `num_steps` | length of the last episode (control steps) |
| `reward_episode` | return of the last episode |
| `gates_passed_ratio` | gates passed / gates in course (> 1 = more than one lap) |
| `speed_mean`, `speed_max` | drone speed, m/s |
| `g_force_mean`, `g_force_max` | specific force in g (1.0 = hover): `|dv/dt + (0, 0, g)| / g` |
| `episode_done_time` | 1 if it ended by timeout / lap limit, else 0 |
| `episode_done_crash` | 1 if it ended by crash / collision, else 0 |

`info` from `step()`: `gates_passed`, `num_gates`.

---

## 8. Files and API

```
drone_env/
    __init__.py          from .race_aviary import *, .drone_utils import *, .gates_randomizer import *
    drone_constants.py   reward weights, episode limits, gate dimensions, start position, default course
    race_aviary.py       RaceAviary: the gymnasium environment
    gates_randomizer.py  GatesRandomizer: random course generator
    drone_render.py      DroneRenderer: offscreen MuJoCo rendering + OpenCV window
    drone_utils.py       math helpers (rotations, gate orientation and corners, rpm <-> action)
    README.md            this file
```

Imports: `numpy`, `mujoco`, `gymnasium`, `multi_drone_mujoco` (BaseAviary, enums);
`opencv-python` only for `render_mode="human"`.
Package exports: `RaceAviary`, `GatesRandomizer` and the `drone_utils` functions.

### `RaceAviary` (`race_aviary.py`)

```python
RaceAviary(drone_model=DroneModel.CF2X, physics=Physics.MJC, sim_freq=250, ctrl_freq=50,
           gates=None,                 # (N, 3) fixed gate centres; default DEFAULT_GATES
           gates_randomizer=None,      # GatesRandomizer -> new course every reset
           num_future_gates=2,         # gates after the target in the observation
           render_mode=None,           # None | "rgb_array" | "human"
           camera="third_person",      # "third_person" | "track" | "course"
           trace=False)                # draw the fading flight trace
```

| Method | Returns | Description |
|---|---|---|
| `reset(seed=None, options=None)` | `obs, info` | new course (if randomizer), drone to start; `seed` makes the course reproducible |
| `step(action)` | `obs, reward, terminated, truncated, info` | one control step (1/50 s) |
| `render(camera=None)` | frame / `None` / `-1` | `rgb_array`: (480, 640, 3) uint8; `human`: shows window, `-1` when closed with `q` |
| `get_log()` | `dict` | last finished episode stats (section 7) |
| `close()` | | closes renderer and window |

Attributes: `observation_shape`, `action_shape`, `gates` (N, 3), `gate_yaw` (N,), `gate_corners`
(N, 4, 3), `gates_passed`, `speed`, `g_force`.

### `GatesRandomizer` (`gates_randomizer.py`)

```python
GatesRandomizer(num_gates=8, step_range=(1.5, 3.0), max_turn_deg=110.0, reverser_turn_deg=45.0,
                height_range=(0.6, 2.0), arena_radius=4.0, min_separation=1.2, stacked_prob=0.0)
.sample(rng: np.random.Generator) -> np.ndarray      # (N, 3) gate centres in flight order
```

### `DroneRenderer` (`drone_render.py`)

```python
DroneRenderer(model, width=640, height=480, window="drone race", trace=False)
.render(data, pos, speed, gates, yaws, highlight, camera="third_person") -> (H, W, 3) uint8
.show(frame, sim_time, lines=()) -> str              # pressed key or ""
.reset()                                             # clear trace, camera smoothing, clock
.close()
```

### `drone_utils.py`

| Function | Description |
|---|---|
| `body_rotation(quat_wxyz)` | 3x3 rotation body -> world |
| `rpm_to_action(rpm, hover_rpm, max_rpm)` | motor RPMs -> normalized action |
| `yaw_matrix(yaw)` | 3x3 rotation about z |
| `gate_yaws(gates)` | normal yaw of every gate (section 2) |
| `gate_corners(gates, yaws, half)` | (N, 4, 3) corners at +-`half` in each gate plane |
| `to_gate_frame(point, gates, yaws)` | point in every gate's frame: x normal, y side, z up |

---

## 9. Default constants (`drone_constants.py`)

| Constant | Default | Unit |
|---|---|---|
| `GATE_PASSED` | 1.0 | reward |
| `PROGRESS` | 1.0 | reward per m |
| `ALIGN` | 0.0 | reward per alignment unit |
| `CRASH_PENALTY` | 2.0 | reward |
| `EPISODE_LEN_SEC` | 60.0 | s |
| `MAX_LAPS` | None | laps |
| `LOW_ALTITUDE` | 0.1 | m |
| `MAX_TILT` | 2.0 | rad |
| `GATE_SIZE` | 0.7 | m |
| `GATE_HOLE` | 0.5 | m |
| `GATE_DEPTH` | 0.05 | m |
| `GATE_FORWARD_ONLY` | False | |
| `INITIAL_XYZ` | (0, 0, 0.5) | m |
| `DEFAULT_GATES` | 6 gates, oval | m |