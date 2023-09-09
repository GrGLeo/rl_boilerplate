import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, inputs, n_actions,hidden_dims, alpha):
        super(PolicyNetwork, self).__init__()
        self.inputs = inputs
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.alpha = alpha

        # Network
        self.fc1 = nn.Linear(*self.inputs, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, n_actions)

        # Optimizer
        self.optim = optim.Adam(params=self.parameters(), lr=self.alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
