import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os


class CriticNetwork(nn.Module):
    def __init__(self,critic_lr, state_dim, n_actions, name, chkpt_dir="tmp/models"):
        super(self,CriticNetwork).__init__()
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_td3')


        self.fc1 = nn.Linear(self.state_dim[0] + self.n_actions,400)
        self.fc2 = nn.Linear(400,300)
        self.q1 = nn.Linear(300,1)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optim = optim.Adam(params=self.parameters(), lr = self.critic_lr)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save(self):
        print("... saving models ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print("... loading models ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, actor_lr, state_dim, n_actions, name, chkpt_dir="tmp/models"):
        super(self, ActorNetwork).__init__()
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_td3')

        self.fc1 = nn.Linear(*self.state_dim,400)
        self.fc2 = nn.Linear(400,300)
        self.mu = (300, self.n_actions)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.optim = optim.Adam(params = self.parameters(), lr = self.action_lr)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.mu(x))

        return x

    def save(self):
        print("... saving models ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print("... loading models ...")
        self.load_state_dict(T.load(self.checkpoint_file))
