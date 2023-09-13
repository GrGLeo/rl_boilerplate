import torch as T
import torch.nn as nn
import torch.nn.functional as F
from network import Network
import numpy as np
from typing import Optional


class ActorCriticAgent():
    def __init__(self, lr: float, n_actions: int,
                 gamma: Optional[float] = 0.99,
                 epsilon: Optional[float] = 0.95,
                 decay: Optional[float] = 1e-5,
                 exploration: Optional[bool] = True):
        self.gamma = gamma
        self.network = Network(lr, n_actions)
        self.n_actions = n_actions
        self.lr = lr
        self.log_prob = None
        self.epsilon = epsilon
        self.step = 0
        self.decay = decay
        self.exploration = exploration


    def choose_action(self, state: np.array):
        if T.rand(1) < self.epsilon:
            random_values = T.rand(self.n_actions)
            probabilities = random_values / random_values.sum()
        else:

            state = T.tensor(np.array(state, dtype = np.float64), dtype=T.float).to(self.network.device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            probabilities, _ = self.network.forward(state)
        probabilities = F.softmax(probabilities, dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state: np.array, reward: int, state_: np.array, done: bool):
        self.network.optim.zero_grad()

        state = T.tensor(np.array(state, dtype = np.float64), dtype=T.float).to(self.network.device)
        state = state.unsqueeze(0).permute(0, 3, 1, 2)
        state_ = T.tensor(np.array(state_, dtype = np.float64), dtype=T.float).to(self.network.device)
        state_ = state_.unsqueeze(0).permute(0, 3, 1, 2)
        reward = T.tensor(reward, dtype=T.float).to(self.network.device)


        _,critic_value = self.network.forward(state)
        _,critic_value_ = self.network.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
        actor_loss = delta * -self.log_prob
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.network.optim.step()

    def update_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= 1 - self.decay

    def save_agent(self, path: str):
        print(".. Saving Model ..")
        T.save(self.network.state_dict(), path)

    def load_agent(self, path: str):
        print(".. Loading Model ..")
        self.network.load_state_dict(T.load(path))
