import torch
from torch.distributions import Categorical


class PPOAgent:

    def __init__(
        self,
        state_shape,
        n_actions,
        Model,
        learning_rate,
        n_steps,
        gamma,
        entropy_beta,
        ppo_epochs=4,
        batch_size=64,
        clip_eps=0.2,
        device="cpu"
    ):
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.entropy_beta = entropy_beta
        self.ppo_epochs   = ppo_epochs
        self.batch_size   = batch_size
        self.clip_eps     = clip_eps
        self.device       = device
        self.state_shape  = state_shape

        self.model = Model(state_shape, n_actions)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self._reset_buffer()

        print(self.model)
        print("n_steps      = ", self.n_steps)
        print("gamma        = ", self.gamma)
        print("entropy_beta = ", self.entropy_beta)
        print("ppo_epochs   = ", self.ppo_epochs)
        print("batch_size   = ", self.batch_size)
        print("clip_eps     = ", self.clip_eps)
        print("device       = ", self.device)

    def step(self, state, reward, done, training_enabled):
        state_t = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            logits, _ = self.model(state_t.unsqueeze(0))

        logits = logits.squeeze(0)

        dist     = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        action   = action_t.item()

        if training_enabled:
            self.states[self.step_idx]    = state_t.detach().cpu()
            self.actions[self.step_idx]   = action
            self.rewards[self.step_idx]   = reward
            self.dones[self.step_idx]     = done
            self.log_probs[self.step_idx] = log_prob.detach().cpu()

            self.step_idx += 1

            if self.step_idx >= self.n_steps:
                self._train_model()
                self._reset_buffer()

        return action

    def _sample_action(self, logits):
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def _train_model(self):

        self.states    = self.states.to(self.device)
        self.rewards   = self.rewards.to(self.device)
        self.dones     = self.dones.to(self.device)
        self.actions   = self.actions.to(self.device)
        self.log_probs = self.log_probs.to(self.device)

        with torch.no_grad():
            _, next_value = self.model(self.states[-1].unsqueeze(0))
            next_value = next_value.squeeze()

        # compute returns
        returns = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)

        R = next_value

        for n in reversed(range(self.n_steps)):
            R = self.rewards[n] + self.gamma * R * (1.0 - self.dones[n].float())
            returns[n] = R

        # advantages, computed ONCE with the old policy's values,
        # then kept fixed across all PPO epochs
        with torch.no_grad():
            _, values = self.model(self.states)
            values = values.squeeze(-1)

        advantages = returns - values

        # ---------------------------------------------------------
        # PPO : several epochs over the rollout, in minibatches
        # ---------------------------------------------------------
        for epoch in range(self.ppo_epochs):

            perm = torch.randperm(self.n_steps, device=self.device)

            for start in range(0, self.n_steps, self.batch_size):
                idx = perm[start:start + self.batch_size]

                # minibatch forward pass
                logits_b, values_b = self.model(self.states[idx])
                values_b = values_b.squeeze(-1)

                dist_b = Categorical(logits=logits_b)

                new_log_probs = dist_b.log_prob(self.actions[idx])

                # advantage normalization, per minibatch
                adv_b = advantages[idx]
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # clipped surrogate actor loss
                ratio = torch.exp(new_log_probs - self.log_probs[idx])

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b

                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic MSE loss
                critic_loss = (returns[idx] - values_b).pow(2).mean()

                # Entropy loss
                entropy = dist_b.entropy().mean()

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

        self.log_probs = torch.zeros(
            (self.n_steps,),
            dtype=torch.float32,
            device="cpu"
        )

        self.step_idx = 0