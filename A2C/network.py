import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr,  inputs, n_actions, fc1_dims, fc2_dims):
        super(ActorCriticNetwork, self).__init__()
        self.lr = lr
        self.inputs = inputs
        self.n_action = n_actions

        self.fc1 = nn.Linear(*self.inputs,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims,1)

        self.optim = optim.Adam(params=self.parameters(), lr=self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi,v)
