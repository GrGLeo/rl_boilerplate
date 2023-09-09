import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import PolicyNetwork


class Agent():
    def __init__(self,inputs, alpha, gamma=0.99, n_actions=4):
        self.alpha = alpha
        self.policy = PolicyNetwork(inputs, n_actions, 128, alpha)
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []



    def choose_action(self, state):
        state = T.tensor(list(state)).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    # G_t = R_t+1 + gamma * R_t+2 + gamma**2 + R_t+3 ...
    # G_t = sum from k=0 to k=T {gamma**k + R_tk+1}
    def learn(self):
        self.policy.optim.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t,len(self.reward_memory)):
                G_sum += self.reward_memory[k]* discount
                discount *= self.gamma
            G[t] = G_sum
            G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G,self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optim.step()

        self.action_memory = []
        self.reward_memory = []

    def save_agent(self,path):
        T.save(self.policy.state_dict(), path)

    def load_agent(self,path):
        self.policy.load_state_dict(T.load(path))
