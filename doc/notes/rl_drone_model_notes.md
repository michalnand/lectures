# MLP Actor-Critic (`MLPActorCritic`)

Shared-trunk actor-critic for continuous control, used by the continuous PPO agent.
One MLP body feeds three linear heads: action mean, action variance and state value.

| Property | Value |
|---|---|
| Type | MLP, shared trunk, 3 output heads |
| Hidden layers | 2 x `num_hidden` (default 256), SiLU |
| Policy | diagonal Gaussian, mean in [-1, 1], variance > 0 |
| Critic | scalar state value V(s) |
| Parameters | 83 465 (59 inputs, 4 actions, 256 hidden) |

---

## Architecture

```
x (input_shape[0])
 -> Linear(input, 256) -> SiLU
 -> Linear(256, 256)   -> SiLU            = z (shared features)
     |-> actor_mu  : Linear(256, A) -> Tanh      -> pi_mean  (A)
     |-> actor_var : Linear(256, A) -> Softplus  -> pi_var   (A)
     |-> critic    : Linear(256, 1)              -> value    (1)
```

| Layer | Shape | Activation | Init | Parameters (59 -> 4) |
|---|---|---|---|---|
| `lin_0` | input -> 256 | SiLU | orthogonal, gain 0.5, bias 0 | 15 360 |
| `lin_1` | 256 -> 256 | SiLU | orthogonal, gain 0.5, bias 0 | 65 792 |
| `actor_mu` | 256 -> A | Tanh | orthogonal, gain `actor_mu_gain` (0.01), bias 0 | 1 028 |
| `actor_var` | 256 -> A | Softplus | orthogonal, gain `actor_var_gain` (0.01), bias 0 | 1 028 |
| `critic` | 256 -> 1 | none | orthogonal, gain `critic_gain` (0.01), bias 0 | 257 |
| **Total** | | | | **83 465** |

---

## Constructor

```python
MLPActorCritic(input_shape, num_outputs, num_hidden=256,
               actor_mu_gain=0.01, actor_var_gain=0.01, critic_gain=0.01)
```

| Argument | Default | Meaning |
|---|---|---|
| `input_shape` | | observation shape, tuple; only `input_shape[0]` is used (flat input) |
| `num_outputs` | | action dimension A |
| `num_hidden` | 256 | width of both hidden layers |
| `actor_mu_gain` | 0.01 | init gain of the mean head |
| `actor_var_gain` | 0.01 | init gain of the variance head |
| `critic_gain` | 0.01 | init gain of the value head |

Hidden-layer gain (0.5) is fixed in the code.

---

## Inputs and outputs

`forward(x)` returns `(pi_mean, pi_var, value)`.

| Tensor | Shape | Range | Meaning |
|---|---|---|---|
| `x` | (B, input) | any (float32) | observation batch |
| `pi_mean` | (B, A) | (-1, 1) | mean of each action dimension (tanh) |
| `pi_var` | (B, A) | (0, inf) | variance of each action dimension (softplus) |
| `value` | (B, 1) | any | critic estimate V(s) |

### Policy distribution

Independent Gaussian per action dimension:

```
a_i ~ N(pi_mean_i, pi_var_i)            sampled as  a = pi_mean + sqrt(pi_var) * eps,  eps ~ N(0, 1)
log pi(a|s) = sum_i [ -(a_i - mu_i)^2 / (2 var_i) - 0.5 * log(2 pi var_i) ]
entropy     = sum_i 0.5 * log(2 pi e var_i)
```

Only the mean is squashed by tanh; sampled actions are not, so they can leave [-1, 1].
The environment clips them, and the log-probability is computed on the unclipped sample.

---

## Initial behaviour

Measured at initialization (59 inputs, 4 actions, inputs ~ N(0, 2^2)):

| Output | Value at init | Why |
|---|---|---|
| `pi_mean` | ~0 (abs < 0.003) | head gain 0.01, zero bias; for the drone env 0 = hover speed on all motors |
| `pi_var` | 0.693 (std 0.83) | softplus(0) = ln 2, since the head output is ~0 |
| `value` | ~0 (abs < 0.003) | head gain 0.01, zero bias |
| hidden features `z` | std ~0.06 | gain 0.5 per layer shrinks activations |

Small output-head gains make every state start from the same neutral policy and a flat value
estimate, so early learning is not dominated by random initial differences between states.

---

## Design notes

| Aspect | Current choice | Effect / option |
|---|---|---|
| Shared trunk | actor and critic use the same features | fewer parameters; critic gradients also shape actor features. Works well when rewards are scaled to ~1. Separate trunks are the alternative if the value loss starts to dominate. |
| State-dependent variance | variance comes from a head on `z` | exploration can differ per state. The common alternative is one learned log-std parameter per action, independent of the state. |
| Initial std 0.83 | from softplus(0) | wide for direct motor control (random motor noise of +-0.83 flips the drone quickly). A negative variance-head bias lowers it, e.g. bias -2.8 gives std ~0.25. |
| No variance floor | softplus can approach 0 | if the variance collapses, log-probabilities and PPO ratios can blow up. Adding a small constant (e.g. `+ 1e-4`) to `pi_var` prevents it. |
| Hidden gain 0.5 | activations shrink layer by layer (std ~0.06 at init) | trains fine at this depth; sqrt(2) is the standard gain for ReLU-like activations if the network is made deeper. |
| Tanh on the mean | mean stays inside the action range | gradients vanish when the mean saturates near +-1; worth monitoring (share of abs(mu) > 0.95). |