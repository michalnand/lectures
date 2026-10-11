import torch

class MLPRLModel(torch.nn.Module):
    def __init__(self, input_shape, num_actions, num_hidden = 128):
        super().__init__()

        n_inputs = input_shape[0]

        self.lin0 = torch.nn.Linear(n_inputs, num_hidden)
        self.act0 = torch.nn.SiLU()

        self.lin1 = torch.nn.Linear(num_hidden, num_hidden)
        self.act1 = torch.nn.SiLU()

        self.actor  = torch.nn.Linear(num_hidden, num_actions)
        self.critic = torch.nn.Linear(num_hidden, 1)

        torch.nn.init.orthogonal_(self.lin0.weight, 0.5)
        torch.nn.init.zeros_(self.lin0.bias)

        torch.nn.init.orthogonal_(self.lin1.weight, 0.5)
        torch.nn.init.zeros_(self.lin1.bias)

        torch.nn.init.orthogonal_(self.actor.weight, 0.1)
        torch.nn.init.zeros_(self.actor.bias)

        torch.nn.init.orthogonal_(self.critic.weight, 0.1)
        torch.nn.init.zeros_(self.critic.bias)

    def forward(self, state):

        z = self.lin0(state)
        z = self.act0(z) 

        z = self.lin1(z)
        z = self.act1(z) 

        return self.actor(z), self.critic(z)
            