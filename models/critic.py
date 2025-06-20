import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.q1 = nn.Linear(hidden_size, action_dim)
        self.q2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2