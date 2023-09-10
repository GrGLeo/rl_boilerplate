import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network import ActorCriticNetwork

class A2CAgent():
    def __init__(self, inputs, fc1_dims, fc2_dims, lr=5e-6, n_actions=4, gamma=0.99):
        self.gamma = gamma
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.network = ActorCriticNetwork(lr=lr, inputs=inputs,\
                    n_actions=n_actions, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.log_prob = None


    def choose_action(self, state):
        state = T.tensor(list(state)).to(self.network.device)
        probabilities, _ = self.network.forward(state)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.network.optim.zero_grad()

        state = T.tensor(list(state), dtype=T.float).to(self.network.device)
        state_ = T.tensor(list(state_), dtype=T.float).to(self.network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.network.device)

        _, critic_value = self.network.forward(state)
        _, critic_value_ = self.network(state_)

        delta = reward + self.gamma*critic_value_*(1-(int(done))) - critic_value
        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss+critic_loss).backward()
        self.network.optim.step()

    def save_agent(self, path):
        print(".. Saving model ..")
        T.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(T.load(path))
