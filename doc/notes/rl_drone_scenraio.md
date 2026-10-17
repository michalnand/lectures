

# environment intro

- drone_example-environment.png : overall drone racing image screenshot, 
inside arrow descriptions (what is FPV gate, drone passing is scoring, and gates order matters)

- we use Mujoco as fast simulator
- custom class inherited from : class RaceAviary(BaseAviary)

- for full environment details, see file rl_drone_env_summary.md

**TASKS**
- describe environemnt, task
- describe state 
- describe reward function
- we always test env, with dummy random actions


actions notes : 
- 4 actions, raw motor control, range -1, +1
- value -1 : motor off
- value 0 : hover (14,500 RPM)
- value +1 : max speed (21,700 RPM)

state notes : 
- we used only from real sensors observable
- angles from IMU, decomposed into sine cosine terms to make them continuous
- angular rates
- velocity against surface (can be obtained from down facing camera of optical flow)
- altitude - z - from TOF sensor
- corner relative positions of 3 next gates (12 numbers per gate)
- multiple gates (1 current + 2 next) allows planning future path

reward notes : 
- GATE_PASSED, strong signal, but very sparse
- CRASH_PENALTY, punish crash or collisions
- dense reward from potential, two potentials
- PROGRESS : if given step makes distance to gate shorter, difference current distance vs prev distance, without this we have only very sparse rewards, challenging exploration
- ALIGN : reward perpendicular aligment to gate, without this reward we observer spiralling entry into gate, or shallow entry angles




- drone main class **race_aviary.py** : 
```python
"""Single-drone gate racing on top of MuJoCo-drones-gym's BaseAviary.

Action (4,) in [-1, 1]: motor RPMs (-1 = off, 0 = hover, 1 = max).
Observation (11 + 12 * (1 + num_future_gates),), all drone-centric (no world position / yaw):
    IMU
    [0:4]   sin(roll), cos(roll), sin(pitch), cos(pitch)
    [4:7]   angular velocity, body frame (gyro)
    optical flow + height sensor
    [7:10]  velocity in the heading frame: forward, left, up (yaw-aligned, horizontal)
    [10]    height above the floor
    gates, next gate first
    [11:]   4 hole corners per gate, relative to the drone in the body frame (x forward, y left, z up),
            corner order (+y, +z), (+y, -z), (-y, +z), (-y, -z) in the gate's own frame

A gate counts when the drone crosses its plane fully inside the hole (either direction, or only
forward along the normal if GATE_FORWARD_ONLY). Touching a frame is a crash.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType

from .drone_constants import *
from .drone_render import DroneRenderer, CAMERAS
from .drone_utils import body_rotation, gate_yaws, gate_corners, to_gate_frame


class RaceAviary(BaseAviary):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, drone_model=DroneModel.CF2X, physics=Physics.MJC, sim_freq=250, ctrl_freq=50,
                 gates=None, gates_randomizer=None, num_future_gates=1,
                 render_mode=None, camera="third_person", trace=False):
        self.randomizer = gates_randomizer
        self._set_gates(gates_randomizer.sample(np.random.default_rng()) if gates_randomizer
                        else DEFAULT_GATES if gates is None else gates)
        self.num_future_gates = num_future_gates
        self.camera, self.trace, self.viewer = camera, trace, None

        super().__init__(drone_model=drone_model, num_drones=1, physics=physics,
                         sim_freq=sim_freq, ctrl_freq=ctrl_freq, initial_xyzs=np.array([INITIAL_XYZ]),
                         obs_type=ObservationType.KIN, act_type=ActionType.RPM, render_mode=render_mode)

        self.observation_shape = self.observation_space.shape
        self.action_shape      = self.action_space.shape
        self.num_episodes = 0
        self._reset_state()
        self.last_episode = self.episode

    # --- gym API ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        gymnasium.Env.reset(self, seed=seed)  # seed self.np_random -> reproducible layouts
        if self.randomizer:
            self._set_gates(self.randomizer.sample(self.np_random))
        if self.viewer:
            self.viewer.reset()
        if self.episode["num_steps"] > 0:
            self.last_episode = self.episode
        self.num_episodes += 1
        self.gates_passed = 0  # before super().reset(), which builds the first obs
        obs, info = super().reset(options=options)
        self._reset_state()

        self.speed_filtered = 0
        self.g_force_filtered = 0
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # speed and g-force (acceleration minus gravity: 1 g when hovering)
        vel = self.vel[0].copy()
        acc = (vel - self.prev_vel) / self.CTRL_TIMESTEP
        self.prev_vel = vel
        self.speed = float(np.linalg.norm(vel))
        self.g_force = float(np.linalg.norm(acc + [0.0, 0.0, self.G]) / self.G)

        ep = self.episode
        ep["num_steps"] += 1
        ep["reward"] += reward
        ep["speed_sum"] += self.speed
        ep["g_force_sum"] += self.g_force
        ep["speed_max"] = max(ep["speed_max"], self.speed)
        ep["g_force_max"] = max(ep["g_force_max"], self.g_force)
        ep["gates_passed_ratio"] = self.gates_passed / len(self.gates)

        # BaseAviary builds obs before the reward advances gates_passed; rebuild it.
        return self._computeObs(), reward, terminated, truncated, info

    def render(self, camera=None):
        """rgb_array: return the frame. human: OpenCV window, keys c = switch camera, q / Esc = quit."""
        if self.render_mode is None:
            return None
        if self.viewer is None:
            self.viewer = DroneRenderer(self.model, trace=self.trace)
        frame = self.viewer.render(self.data, self.pos[0], self.speed, self.gates, self.gate_yaw,
                                   self._upcoming(), camera or self.camera)
        if self.render_mode == "rgb_array":
            return frame


        self.speed_filtered = max(0.99*self.speed_filtered, self.speed)
        self.g_force_filtered = max(0.99*self.g_force_filtered, self.g_force)

        key = self.viewer.show(frame, self.step_counter / self.SIM_FREQ, [
            (f"{self.speed_filtered:4.1f} m/s   {self.g_force_filtered:3.1f} g", 0.8),
            (f"gates {self.gates_passed}   {self.camera}", 0.55),
            ("[c] camera  [q] quit", 0.45)])
        if key == "c":
            self.camera = CAMERAS[(CAMERAS.index(self.camera) + 1) % len(CAMERAS)]
        elif key in ("q", "\x1b"):  # stop rendering, simulation keeps running
            self.viewer.close()
            self.viewer, self.render_mode = None, None
            return -1

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        super().close()

    def get_log(self):
        """Current episode length plus stats of the last finished episode."""
        ep, n = self.last_episode, max(self.last_episode["num_steps"], 1)
        return {"num_steps": self.episode["num_steps"], "num_episodes": self.num_episodes,
                "reward_episode": ep["reward"], "gates_passed_ratio": ep["gates_passed_ratio"],
                "speed_mean": ep["speed_sum"] / n, "speed_max": ep["speed_max"],
                "g_force_mean": ep["g_force_sum"] / n, "g_force_max": ep["g_force_max"]}

    # --- state -----------------------------------------------------------

    def _set_gates(self, gates):
        self.gates = np.array(gates, dtype=float)
        self.gate_yaw = gate_yaws(self.gates)
        self.gate_corners = gate_corners(self.gates, self.gate_yaw, GATE_HOLE / 2)   # inner corners

    def _reset_state(self):
        self.gates_passed, self.hit_gate = 0, False
        self.prev_pos, self.prev_vel = self.pos[0].copy(), self.vel[0].copy()
        self.prev_potential = self._potential()
        self.speed, self.g_force = 0.0, 1.0
        self.episode = dict(num_steps=0, reward=0.0, gates_passed_ratio=0.0,
                            speed_sum=0.0, speed_max=0.0, g_force_sum=0.0, g_force_max=0.0)

    def _update_gates(self):
        """Check all gates for frame contact (sets hit_gate); return True if the next gate was passed."""
        r = self.COLLISION_R
        a = to_gate_frame(self.prev_pos, self.gates, self.gate_yaw)
        b = to_gate_frame(self.pos[0], self.gates, self.gate_yaw)
        self.prev_pos = self.pos[0].copy()

        def inside(q, half):
            return (np.abs(q[:, 1]) < half) & (np.abs(q[:, 2]) < half)

        crossed = (a[:, 0] > 0) != (b[:, 0] > 0)                   # plane crossed this step
        t = a[:, 0] / np.where(crossed, a[:, 0] - b[:, 0], 1.0)
        x = a + (t * crossed)[:, None] * (b - a)                    # crossing point
        hole, outer = GATE_HOLE / 2 - r, GATE_SIZE / 2 + r
        in_slab = np.abs(b[:, 0]) < GATE_DEPTH / 2 + r
        hit = (crossed & inside(x, outer) & ~inside(x, hole)) | (in_slab & inside(b, outer) & ~inside(b, hole))
        self.hit_gate = bool(hit.any())

        nxt = self._upcoming()[0]
        forward = a[nxt, 0] < 0 <= b[nxt, 0]                       # from the front side to the back side
        return bool(crossed[nxt] and inside(x, hole)[nxt] and (forward or not GATE_FORWARD_ONLY))

    # --- spaces / obs ----------------------------------------------------

    def _actionSpace(self):
        return spaces.Box(-1, 1, (4,), np.float32)

    def _observationSpace(self):
        return spaces.Box(-np.inf, np.inf, (11 + 12 * (1 + self.num_future_gates),), np.float32)

    def _preprocessAction(self, action):
        return self._normalizedActionToRPM(np.clip(action, -1, 1)).reshape(1, 4)

    def _upcoming(self):
        """Indices of the next gate and the following num_future_gates (wrapping laps)."""
        return (self.gates_passed + np.arange(1 + self.num_future_gates)) % len(self.gates)

    def _computeObs(self):
        pos, rot = self.pos[0], body_rotation(self.quat[0])
        roll, pitch, yaw = self.rpy[0]
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy, vz = self.vel[0]
        corners = (self.gate_corners[self._upcoming()] - pos) @ rot     # world -> body, (K, 4, 3)
        return np.concatenate([
            [np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)],
            self.ang_v[0],                              # gyro (MuJoCo free-joint rates are body-frame)
            [c * vx + s * vy, -s * vx + c * vy, vz],    # heading-frame velocity
            [pos[2]],                                   # height
            corners.ravel(),
        ]).astype(np.float32)

    # --- task ------------------------------------------------------------

    def _alignment(self):
        """0..1, 1 = drone on the next gate's axis: min/max of the distances to the 4 hole corners."""
        d = np.linalg.norm(self.gate_corners[self._upcoming()[0]] - self.pos[0], axis=1)
        return d.min() / d.max()

    def _potential(self):
        """Shaping potential for the next gate: closer and better aligned = higher."""
        dist = np.linalg.norm(self.gates[self._upcoming()[0]] - self.pos[0])
        return -PROGRESS * dist + ALIGN * self._alignment()

    def _computeReward(self):
        passed = self._update_gates()
        self.gates_passed += passed

        # Shaping is the change of the potential, so hovering on the axis earns nothing:
        # only getting closer / better aligned does. After a pass, restart from the new gate.
        potential = self._potential()
        reward = GATE_PASSED if passed else potential - self.prev_potential
        self.prev_potential = potential

        if self._computeTerminated():
            reward -= CRASH_PENALTY
        return float(reward)

    def _computeTerminated(self):
        return bool(self.hit_gate or self.pos[0, 2] < LOW_ALTITUDE
                    or np.any(np.abs(self.rpy[0, :2]) > MAX_TILT))

    def _computeTruncated(self):
        laps_done = MAX_LAPS is not None and self.gates_passed >= MAX_LAPS * len(self.gates)
        return bool(laps_done or self.step_counter / self.SIM_FREQ >= EPISODE_LEN_SEC)

    def _computeInfo(self):
        return {"gates_passed": self.gates_passed, "num_gates": len(self.gates)}
```

- drone params **drone_constants.py** : 
```python
"""Reward weights and task settings."""

# Reward
GATE_PASSED     = 1.0       # per gate
PROGRESS        = 1.0       # x metres moved toward the next gate centre, every step
ALIGN           = 0*1.0       # x change of gate alignment (0..1, 1 = on the gate's axis), every step
CRASH_PENALTY   = 2.0       # once, on crash (floor, flip or gate frame)

# Episode
EPISODE_LEN_SEC = 60.0
MAX_LAPS        = None      # None = only the time limit ends the episode
LOW_ALTITUDE    = 0.1       # m
MAX_TILT        = 2.0       # rad, |roll| or |pitch| above this = crash

# Gates: square frame with a square hole, normal along the flight path
GATE_SIZE       = 0.7       # m, outer edge
GATE_HOLE       = 0.5       # m, hole edge
GATE_DEPTH      = 0.05      # m, frame thickness
GATE_FORWARD_ONLY = False   # True: a gate counts only when crossed along its normal (front -> back)

# Drone
INITIAL_XYZ = (0.0, 0.0, 0.5)
DEFAULT_GATES = [
    [1.0, 0.0, 1.0], [2.0, 1.0, 1.2], [1.0, 2.0, 1.5],
    [0.0, 2.0, 1.3], [-1.0, 1.0, 1.0], [0.0, 0.0, 0.8],
]
```


- environment test
```python
import time
import numpy
import drone_env

import rl_robot_libs

if __name__ == "__main__":
    num_envs    = 128
    envs        = rl_robot_libs.vector_env.VectorEnv(num_envs, drone_env.RaceAviary, gates_randomizer=drone_env.GatesRandomizer(stacked_prob=0.3), num_future_gates=2)

    print("num envs     : ", num_envs)
    print("state shape  : ", envs.observation_shape)
    print("action shape : ", envs.action_shape)


    for i in range(num_envs):
        envs.reset(i)

    num_steps = 1000

    time_start = time.time()
    for step in range(num_steps):    
        actions = numpy.random.randn(num_envs, envs.action_shape[0])
        obs, reward, terminated, truncated, info = envs.step(actions)
        for i in numpy.flatnonzero(terminated | truncated):
            obs[i], _ = envs.reset(i)
    time_stop = time.time()

    print("obs shape        : ", obs.shape)
    print("reward shape     : ", reward.shape)
    print("terminated shape : ", terminated.shape)
    print("truncated shape  : ", truncated.shape)


    print("sps/env = ", round(num_envs*num_steps/(time_stop - time_start), 3))
    print("sps     = ", round(num_steps/(time_stop - time_start), 3))

    logger = rl_robot_libs.training_logger.TrainingLogger("result/")
    print("\n\n")
    print(envs.get_logs())

    #logger.add()
```


result : 
```bash
num envs     :  128
state shape  :  (47,)
action shape :  (4,)
obs shape        :  (128, 47)
reward shape     :  (128,)
terminated shape :  (128,)
truncated shape  :  (128,)
sps/env =  6156.286
sps     =  48.096



{'num_steps_mean': 8.28906, 'num_steps_std': 7.25232, 'num_steps_p68': 9.0, 'num_steps_p95': 22.0, 'num_episodes_mean': 69.125, 'num_episodes_std': 4.06586, 'num_episodes_p68': 72.0, 'num_episodes_p95': 75.0, 'reward_episode_mean': -2.06854, 'reward_episode_std': 0.17559, 'reward_episode_p68': -1.99546, 'reward_episode_p95': -1.8934, 'gates_passed_ratio_mean': 0.0, 'gates_passed_ratio_std': 0.0, 'gates_passed_ratio_p68': 0.0, 'gates_passed_ratio_p95': 0.0, 'speed_mean_mean': 0.66854, 'speed_mean_std': 0.37483, 'speed_mean_p68': 0.69078, 'speed_mean_p95': 1.4924, 'speed_max_mean': 1.62062, 'speed_max_std': 0.78113, 'speed_max_p68': 1.69883, 'speed_max_p95': 3.06447, 'g_force_mean_mean': 1.01864, 'g_force_mean_std': 0.12064, 'g_force_mean_p68': 1.08032, 'g_force_mean_p95': 1.21294, 'g_force_max_mean': 1.70384, 'g_force_max_std': 0.25726, 'g_force_max_p68': 1.83752, 'g_force_max_p95': 2.0933}
```



# agent 

- simple PPO, continuous actions
- see file ppo_continuous_notes.md