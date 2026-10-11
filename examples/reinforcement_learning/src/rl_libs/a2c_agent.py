import torch
import numpy

class A2CAgent:
 
    def __init__(self, n_envs, state_shape, n_actions, Model, learning_rate, n_steps, gamma, entropy_beta, device="cpu"):
        self.n_envs       = n_envs
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.entropy_beta = entropy_beta
        self.device       = device
        self.state_shape  = state_shape
 
        self.model = Model(state_shape, n_actions)
        self.model.to(self.device)
 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
 
        self._reset_buffer()
 
        print(self.model)
        print("n_envs       = ", self.n_envs)
        print("n_steps      = ", self.n_steps)
        print("gamma        = ", self.gamma)
        print("entropy_beta = ", self.entropy_beta)
        print("device       = ", self.device)
 
    def step(self, state, reward, done, training_enabled):
        # state: (n_envs, obs_dim) numpy  -> whole batch goes through at once
        state_t = torch.from_numpy(state).float().to(self.device)
 
        # predict prob distribution, shape (n_envs, n_actions)
        logits, _ = self.model(state_t)         

        # sample discrete action, shape (n_envs, )
        actions = self._sample_action(logits.detach())

        # add to buffer
        if training_enabled:
            self.states[self.step_idx]  = state_t.cpu()
            self.actions[self.step_idx] = actions.cpu()
            self.rewards[self.step_idx] = torch.from_numpy(reward)
            self.dones[self.step_idx]   = torch.from_numpy(done)
 
            self.step_idx += 1  

            # buffer full, start training
            if self.step_idx >= self.n_steps:
                self._train_model()

                # clean buffer
                self._reset_buffer()
 
        # envpool wants a numpy array of actions, one per env
        return actions.cpu().numpy()
 
    def _sample_action(self, logits):
        # convert raw logits to probs
        probs = torch.softmax(logits, dim=-1)          # (n_envs, n_actions)

        # sample
        actions = torch.multinomial(probs, num_samples=1)  # (n_envs, 1)
        return actions.squeeze(-1)                         # (n_envs,)
 
    def _train_model(self):
        # buffers: (n_steps, n_envs, ...) : time step first, env ID second dim
        states  = self.states.to(self.device)
        rewards = self.rewards.to(self.device)
        dones   = self.dones.to(self.device)
        actions = self.actions.to(self.device)
 
        T, N = self.n_steps, self.n_envs
 
        # bootstrap: value of the LAST stored state, one per env
        with torch.no_grad():
            _, next_value = self.model(states[-1])   # (n_envs, 1)
            next_value = next_value.squeeze(-1)      # (n_envs,)
 
        # n-step discounted returns, computed for all envs in parallel:
        # each column (env) has its own running R, reset where done=True
        returns = torch.zeros((T, N), dtype=torch.float32, device=self.device)
 
        R = next_value                               # (n_envs,)
        for n in reversed(range(T)):
            R = rewards[n] + self.gamma * R * (1.0 - dones[n].float())
            returns[n] = R
 
        # single forward pass over all T*N states: flatten time and env dims, massive 16k batching
        logits, values = self.model(states.reshape(T * N, -1))

        values  = values.squeeze(-1)
        returns = returns.reshape(T * N)
        actions = actions.reshape(T * N)
 
        # advantages and normalization (mean and std), for more stable training
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        # log_softmax = log(softmax(logits)), computed in a numerically
        # stable way. Row r holds log p(a | state_r) for every action a
        # returns (T*N, n_actions)
        log_probs_all = torch.log_softmax(logits, dim=-1)  


        # from each row select action that was actually taken, log p(a_taken | state)
        # returns (T*N,)
        log_probs = log_probs_all[range(T*N), actions]
 
        # actor loss : policy gradient, push up log-prob of actions
        actor_loss = -(log_probs * advantages).mean()
 
        # critic : MSE loss
        critic_loss = (returns - values).pow(2).mean()
 
        # entropy loss
        # H = -sum_i p_i * log(p_i)   per state, then averaged
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
 
        # final loss, 0.5 is standart critic loss weight, entropy is maximized, added minus sign
        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy

        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
 
    def _reset_buffer(self):
        shape_tmp = (self.n_steps, self.n_envs)
        self.states  = torch.zeros(shape_tmp + self.state_shape, dtype=torch.float32)
        self.rewards = torch.zeros(shape_tmp, dtype=torch.float32)
        self.dones   = torch.zeros(shape_tmp, dtype=torch.bool)
        self.actions = torch.zeros(shape_tmp, dtype=torch.long)
        self.step_idx = 0