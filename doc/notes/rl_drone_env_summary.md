# Drone Racing Environment (`drone_env.RaceAviary`)

Single-drone FPV gate racing built on MuJoCo-drones-gym (`BaseAviary`, Crazyflie CF2X model).
The agent controls the four motors directly and must fly through square gates, in order,
as many as possible within the episode time.

| Property | Value |
|---|---|
| Drone | Crazyflie CF2X (X configuration), collision radius 0.06 m |
| Physics / control rate | 250 Hz / 50 Hz (5 physics steps per action) |
| Episode length | 30 s = 1500 control steps |
| Action | 4 normalized motor speeds |
| Observation | `11 + 12 * (1 + num_future_gates)` values (47 for `num_future_gates=2`) |
| Course | randomized every reset (`GatesRandomizer`) or fixed (`gates=`) |

---

## Actions

Shape `(4,)`, range `[-1, 1]`, one value per motor. Values outside the range are clipped.

| Action value | Motor speed (CF2X) |
|---|---|
| -1 | 0 RPM (motor off) |
| 0 | hover speed, ~14 500 RPM |
| +1 | maximum, ~21 700 RPM (~2.25x weight in thrust) |

This is direct motor control: there is no inner velocity or attitude controller.
Climb comes from all motors together, roll / pitch from left-right / front-back differences,
yaw from the clockwise vs counter-clockwise propeller pairs. The mapping is asymmetric:
the range below 0 spans the whole hover speed, the range above 0 only adds half again.

---

## Observation

Everything is drone-centric, as from onboard sensors: no world position and no world yaw.

| Index | Size | Content | Frame | Real sensor | Typical range |
|---|---|---|---|---|---|
| 0-3 | 4 | sin(roll), cos(roll), sin(pitch), cos(pitch) | horizon | IMU attitude | [-1, 1] |
| 4-6 | 3 | angular velocity wx, wy, wz | body | gyro | ~[-10, 10] rad/s |
| 7-9 | 3 | velocity: forward, left, up | heading (yaw-aligned, horizontal) | optical flow + ToF rate | ~[-5, 5] m/s |
| 10 | 1 | height above floor | world z | ToF | 0.1 - 3 m |
| 11-22 | 12 | next gate: 4 hole corners (x, y, z each) | body | gate detector + PnP | ~[-8, 8] m |
| 23-34 | 12 | gate after next (if `num_future_gates >= 1`) | body | | |
| ... | 12 | further gates, one block each | body | | |

The observation space is declared as `Box(-inf, inf)`; the ranges above are typical values,
not hard limits (random actions can spin the drone well above 10 rad/s).

**Body frame:** x forward, y left, z up.

**Corner order** (fixed in the gate's own frame, so the policy can tell which corner is which
and therefore the gate's facing direction):

| Corner | Gate-frame position |
|---|---|
| 0 | (+y, +z), upper left |
| 1 | (+y, -z), lower left |
| 2 | (-y, +z), upper right |
| 3 | (-y, -z), lower right |

Corners are the *inner* (hole) corners, at +-0.25 m from the gate centre.
The 12 numbers fully determine the gate's position and orientation relative to the drone.

---

## Gates

Square frame with a square hole, standing vertically.

| Constant | Value | Meaning |
|---|---|---|
| `GATE_SIZE` | 0.7 m | outer edge |
| `GATE_HOLE` | 0.5 m | hole edge |
| `GATE_DEPTH` | 0.05 m | frame thickness |
| effective hole | 0.38 m | hole minus drone radius on each side |

**Passing:** a gate counts when the drone centre crosses the gate plane with its whole body inside
the hole (checked at the exact crossing point, so fast crossings are not missed).
Only the *next* gate in order can be passed. Direction: either way by default; only
along the normal if `GATE_FORWARD_ONLY = True`.

**Collision:** touching any gate frame (any gate, not only the next) ends the episode as a crash.
Collision is computed geometrically; gates are not MuJoCo bodies.

**Orientation (normal):**

| Gate type | Normal direction |
|---|---|
| single gate | bisector of arrival and departure direction (turn split before / after the gate) |
| stacked pair, first gate | bisector of arrival and reversed departure direction |
| stacked pair, second gate | same plane, opposite normal |

---

## Reward

Per control step:

```
if next gate passed:  r = GATE_PASSED
else:                 r = Phi(s_t) - Phi(s_t-1)              # shaping
if crash:             r -= CRASH_PENALTY

Phi = -PROGRESS * distance_to_next_gate_centre + ALIGN * alignment
alignment = min(d0, d1, d2, d3) / max(d0, d1, d2, d3)   # d_i = distance to hole corner i
```

| Term | Constant | Default | When | Typical size |
|---|---|---|---|---|
| Gate passed | `GATE_PASSED` | 1.0 | step the next gate is passed | +1 |
| Progress | `PROGRESS` | 1.0 | every step, x metres gained toward the next gate centre | +-0.08 per step at 4 m/s |
| Alignment | `ALIGN` | 1.0 | every step, x change of alignment | small, mostly in the last metre |
| Crash | `CRASH_PENALTY` | 1.0 | once, on the terminating step | -1 |

**Shaping is potential-based:** only *changes* of distance and alignment are rewarded.
Hovering earns 0, any closed loop sums to exactly 0, so the shaping cannot be farmed and does
not change which policy is optimal; it only speeds up learning. After a gate pass the potential
restarts from the new next gate (no jump reward).

**Alignment** is 1 on the gate's axis and drops when approaching at an angle.
It is strong close to the gate and fades with distance:

| Distance | On axis | 30 deg off | 60 deg off | 85 deg off |
|---|---|---|---|---|
| 0.5 m | 1.00 | 0.71 | 0.52 | 0.45 |
| 1.0 m | 1.00 | 0.80 | 0.67 | 0.62 |
| 2.0 m | 1.00 | 0.89 | 0.81 | 0.78 |

Episode return is dominated by gates passed: roughly `gates_passed * GATE_PASSED`
plus the shaping toward the gate the episode ended on, minus `CRASH_PENALTY` if it crashed.

---

## Episode end

| Type | Condition | Constant |
|---|---|---|
| terminated (crash) | touched a gate frame | |
| terminated (crash) | height below 0.1 m | `LOW_ALTITUDE` |
| terminated (crash) | abs(roll) or abs(pitch) above 2.0 rad | `MAX_TILT` |
| truncated | 30 s elapsed | `EPISODE_LEN_SEC` |
| truncated | lap limit reached (off by default) | `MAX_LAPS = None` |

The drone starts at (0, 0, 0.5) m, level, at rest (`INITIAL_XYZ`).
Gates wrap around: after the last gate the next one is the first again.

---

## Course generation (`GatesRandomizer`)

A random path through the arena, sampled on every `reset()` (reproducible with `reset(seed=...)`).
Gates come in flight order; sharp turns are allowed.

| Parameter | Default | Meaning |
|---|---|---|
| `num_gates` | 8 | gate positions along the path (each reverser adds one gate) |
| `step_range` | (1.5, 3.0) m | horizontal distance between consecutive positions |
| `max_turn_deg` | 110 | sharpest heading change at a single gate |
| `reverser_turn_deg` | 45 | sharpest heading change right after a reverser |
| `height_range` | (0.6, 2.0) m | gate centre height |
| `arena_radius` | 4.0 m | course stays inside this circle around (0, 0) |
| `min_separation` | 1.2 m | minimum distance between different gate positions (best effort) |
| `stacked_prob` | 0.0 | probability of a reverser at each position |

**Reverser (stacked pair):** two touching gates, one exactly above the other, consecutive in the
order; lower or upper first at random. The drone flies through the first, turns back through the
second, and the course continues back the way it came. A course can contain several.

| Statistic (`stacked_prob=0.25`, 2000 courses) | Value |
|---|---|
| gates per course | 8 - 14, mean 10 |
| reversers per course | mean 2.0, 64% have two or more |
| turn at single gates | median 69 deg, p90 103 deg |
| lower gate first in a pair | 50% |
| last -> first gate leg | median 3.6 m (open path, not a designed loop) |

---

## Logging (`env.get_log()`)

| Key | Meaning |
|---|---|
| `num_steps` | steps in the current episode |
| `num_episodes` | episodes started |
| `reward_episode` | return of the last finished episode |
| `gates_passed_ratio` | gates passed / gates in course, last finished episode (> 1 = more than a lap) |
| `speed_mean`, `speed_max` | drone speed in m/s, last finished episode |
| `g_force_mean`, `g_force_max` | specific force in g (1.0 = hover), last finished episode |

`info` from `step()` contains `gates_passed` and `num_gates`.

---

## Rendering

| Option | Values |
|---|---|
| `render_mode` | `None`, `"rgb_array"` (returns 480x640x3 frame), `"human"` (OpenCV window, real-time) |
| `camera` | `"third_person"` (default, behind drone facing next gate), `"track"`, `"course"` |
| `trace` | `True` draws a fading 3 s flight trail, sky blue (slow) to violet (fast) |
| keys (human) | `c` switch camera, `q` / Esc close window (simulation continues) |

The window shows speed (m/s), g-force, gates passed and the current camera.
Gate colours: next gate white / sky blue, gates in the observation sky blue / royal blue,
others royal blue or violet with black. Numbers on the gates give the flight order.

---

## Constructor

```python
RaceAviary(drone_model=DroneModel.CF2X, physics=Physics.MJC, sim_freq=250, ctrl_freq=50,
           gates=None,               # fixed course, (N, 3) gate centres; default DEFAULT_GATES
           gates_randomizer=None,    # GatesRandomizer(...) -> new course every reset
           num_future_gates=1,       # gates in the observation after the next one
           render_mode=None, camera="third_person", trace=False)
```

All reward weights and task settings live in `drone_constants.py`.