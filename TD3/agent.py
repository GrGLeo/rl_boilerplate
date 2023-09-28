from network import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer

import torch as T
import torch.nn as nn
import numpy as np


class Agent():
    def __init__(self, alpha, beta, state_dims, n_actions, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 max_size=1000000, batch_size = 100, noise=0.1):

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.learn_step_center = 0
        self.time_step = 0
        self.warmup = warmup

        self.actor = ActorNetwork(actor_lr=alpha, state_dim=state_dims, n_actions=n_actions,
                                  name="actor")
        self.critic_1 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                    name="critic_1")
        self.critic_2 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                      name="critic_2")
        self.target_actor = ActorNetwork(actor_lr=alpha, state_dim=state_dims, n_actions=n_actions,
                                         name="target_actor")
        self.target_critic_1 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                             name="target_critic_1")
        self.target_critic_2 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                             name="target_critic_2")
        self.replay = ReplayBuffer(max_len=max_size, input_shape=state_dims, actions_dims=n_actions)

        self.noise = noise
        self.update_netwok_parameter(tau=1)


    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            pass
