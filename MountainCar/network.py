import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, input_shape, fc1_dims, fc2_dims, n_actions):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.input_shape = input_shape

        self.fc1 = nn.Linear(*self.input_shape, fc1_dims)
        self.bn1 = nn.LayerNorm()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm()
        self.out = nn.Linear(fc2_dims, self.n_actions)
