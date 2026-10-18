# PPO Drone Racing Experiment Setup Summary

### 1. Environment & Simulation Specifications

| Parameter | Value / Configuration | Description |
| :--- | :--- | :--- |
| **Task Domain** | FPV Drone Gate Racing | Autonomous navigation through randomized sequential gates |
| **Physics Engine** | MuJoCo | Implemented via `RaceAviary` extending `BaseAviary` |
| **Vectorization** | `VectorEnv` | Parallel environment execution framework |
| **Number of Environments** | `128` | Number of simultaneous environment instances (`num_envs`) |
| **Gate Randomization** | `GatesRandomizer(stacked_prob=0.3)` | Probability configuration for randomized gate layout generation |
| **Lookahead Horizon** | `2` future gates | Number of upcoming gates provided in the observation (`num_future_gates`) |
| **Simulation Frequencies** | Sim: `250 Hz` \| Ctrl: `50 Hz` | Physics integration step vs. control output step frequency |

---

### 2. State & Action Spaces

| Space | Type / Shape | Components & Details |
| :--- | :--- | :--- |
| **Action Space** | Continuous Box `(4,)` | Bounded in $[-1, 1]$, representing normalized motor RPMs |
| **Observation Space** | Continuous Box `(11 + 12 * (1 + num_future_gates),)` | **Kinematics (11):** IMU roll/pitch ($\sin, \cos$), body rates (gyro), heading-frame velocity, altitude<br>**Gate Relative Geometry:** 3D body-frame coordinates for 4 inner corners per upcoming gate |

---

### 3. Neural Network Architecture (`MLPActorCritic`)

| Layer / Head | Architecture | Activation & Initialization Details |
| :--- | :--- | :--- |
| **Base Backbone** | 2 Hidden Layers (256 units each) | **Activation:** SiLU (Swish)<br>**Weights:** Orthogonal (`gain = 0.5`), **Biases:** Zero |
| **Actor Mean Head ($\mu$)** | Linear `(256 -> num_outputs)` | **Activation:** Tanh<br>**Weights:** Orthogonal (`gain = 0.01`), **Biases:** Zero |
| **Actor Variance Head ($\sigma^2$)** | Linear `(256 -> num_outputs)` | **Activation:** Softplus<br>**Weights:** Orthogonal (`gain = 0.01`), **Biases:** Zero |
| **Critic Head ($V$)** | Linear `(256 -> 1)` | **Activation:** Linear<br>**Weights:** Orthogonal (`gain = 0.01`), **Biases:** Zero |

---

### 4. PPO Hyperparameters & Optimization

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Learning rate $\eta = 0.0002$, Adam $\epsilon = 1\times 10^{-5}$ |
| **Total Training Steps** | `100,000` | Target total environment steps for training (`n_training_steps`) |
| **Rollout Buffer Steps** | `128` | Steps collected per environment per update cycle (`n_ppo_steps`) |
| **Optimization Epochs** | `4` | Number of passes over the buffer per update (`n_ppo_episodes`) |
| **Mini-batch Size** | `1024` | Batch size used during gradient descent updates (`batch_size`) |
| **Discount Factor ($\gamma$)** | `0.99` | Reward discount factor |
| **GAE Lambda ($\lambda$)** | `0.95` | Generalized Advantage Estimation parameter |
| **PPO Clip ($\epsilon_{\text{clip}}$)** | `0.2` | Surrogate objective clipping parameter |
| **Advantage Coeff (`adv_coeff`)** | `1.0` | Policy surrogate loss scaling coefficient |
| **Value Coeff (`val_coeff`)** | `0.5` | Critic MSE loss scaling coefficient |
| **Entropy Coeff (`entropy_beta`)** | `0.001` | Gaussian policy entropy maximization coefficient |
| **Max Gradient Norm** | `0.5` | Maximum threshold for gradient clipping (`max_grad_norm`) |

---

### 5. Reward Function & Episode Dynamics

| Component | Mechanism | Details |
| :--- | :--- | :--- |
| **Potential-Based Shaping** | $\Delta \text{Potential}$ | Derived from Euclidean distance and alignment relative to the target gate axis |
| **Gate Pass Reward** | Sparse Bonus | Awarded upon successfully traversing inside the gate hole frame |
| **Crash Penalty** | Negative Terminal Reward | Applied on collision with gate structure, ground hit, or max tilt breach |
| **Termination Conditions** | True on Crash / Limits | Triggered by gate contact (`hit_gate`), minimum altitude (`LOW_ALTITUDE`), or excessive tilt (`MAX_TILT`) |
| **Truncation Conditions** | True on Time / Max Laps | Triggered by maximum lap completion (`MAX_LAPS`) or episode duration limit (`EPISODE_LEN_SEC`) |