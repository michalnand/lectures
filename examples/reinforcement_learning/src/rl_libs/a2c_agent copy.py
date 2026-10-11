import torch
from torch.distributions import Categorical


class A2CAgent:

    def __init__(
        self,
        state_shape,
        n_actions,
        Model,
        learning_rate,
        n_steps,
        gamma,
        entropy_beta,
        device="cpu"
    ):
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.entropy_beta = entropy_beta
        self.device       = device
        self.state_shape  = state_shape

        self.model = Model(state_shape, n_actions)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self._reset_buffer()

        # current observation, owned by the agent
        self.state = None

        print(self.model)
        print("n_steps      = ", self.n_steps)
        print("gamma        = ", self.gamma)
        print("entropy_beta = ", self.entropy_beta)
        print("device       = ", self.device)

    def step(self, env, training_enabled):
        # lazy init / after construction
        if self.state is None:
            self.state = env.reset()

        state_t = torch.from_numpy(self.state).float().to(self.device)

        with torch.no_grad():
            logits, _ = self.model(state_t.unsqueeze(0))

        action = self._sample_action(logits.squeeze(0))

        # agent drives the environment itself
        next_state, reward, done = env.step(action)

        if training_enabled:
            # aligned transition: (s_t, a_t, r_{t+1}, done_{t+1})
            self.states[self.step_idx]  = torch.from_numpy(self.state).float()
            self.actions[self.step_idx] = action
            self.rewards[self.step_idx] = reward
            self.dones[self.step_idx]   = done

            self.step_idx += 1

            if self.step_idx >= self.n_steps:
                # next_state really is the state after the last stored step
                self._train_model(next_state, done)
                self._reset_buffer()

        # handle reset here, so no stale action crosses episodes
        if done:
            self.state = env.reset()
        else:
            self.state = next_state

        return reward, done

    def _sample_action(self, logits):
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def _train_model(self, next_state, next_done):

        self.states  = self.states.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.dones   = self.dones.to(self.device)
        self.actions = self.actions.to(self.device)

        # ---------------------------------------------------------
        # Bootstrap value of the state after the rollout.
        # If the episode ended, V(next_state) = 0.
        # ---------------------------------------------------------
        with torch.no_grad():
            next_state_t = torch.from_numpy(next_state).float().to(self.device)

            _, next_value = self.model(next_state_t.unsqueeze(0))
            next_value = next_value.squeeze()

            if next_done:
                next_value = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=self.device
                )

        # compute returns
        returns = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)

        R = next_value

        for n in reversed(range(self.n_steps)):
            R = self.rewards[n] + self.gamma * R * (1.0 - self.dones[n].float())
            returns[n] = R

        # model forward pass
        logits, values = self.model(self.states)

        values = values.squeeze(-1)

        # advantages and normalization
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # actor loss
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(self.actions)

        actor_loss = -(log_probs * advantages).mean()

        # Critic MSE loss
        critic_loss = (returns - values).pow(2).mean()

        # Entropy loss
        entropy = dist.entropy().mean()

        # Final loss
        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy

        # optimization
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def _reset_buffer(self):
        self.states = torch.zeros(
            (self.n_steps,) + self.state_shape,
            dtype=torch.float32,
            device="cpu"
        )

        self.rewards = torch.zeros(
            (self.n_steps,),
            dtype=torch.float32,
            device="cpu"
        )

        self.dones = torch.zeros(
            (self.n_steps,),
            dtype=torch.bool,
            device="cpu"
        )

        self.actions = torch.zeros(
            (self.n_steps,),
            dtype=torch.long,
            device="cpu"
        )

        self.step_idx = 0