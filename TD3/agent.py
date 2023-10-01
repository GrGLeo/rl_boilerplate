from network import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent():
    def __init__(self, alpha, beta, state_dims, n_actions, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 max_size=1000000, batch_size = 100, noise=0.1, hardcore=False):

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.n_actions = n_actions
        self.learn_step_center = 0
        self.time_step = 0
        self.warmup = warmup
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(actor_lr=alpha, state_dim=state_dims, n_actions=n_actions,
                                  name="actor" if not hardcore else "hc_actor")
        self.critic_1 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                    name="critic_1" if not hardcore else "hc_critic_1")
        self.critic_2 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                      name="critic_2" if not hardcore else "hc_critic_2")
        self.target_actor = ActorNetwork(actor_lr=alpha, state_dim=state_dims, n_actions=n_actions,
                                         name="target_actor" if not hardcore else "hc_target_actor")
        self.target_critic_1 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                             name="target_critic_1" if not hardcore else "hc_target_critic_1")
        self.target_critic_2 = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                             name="target_critic_2" if not hardcore else "hc_target_critic_2")
        self.replay = ReplayBuffer(max_size=max_size, input_shape=state_dims, n_actions=n_actions)

        self.noise = noise
        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)),device=self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float, device=self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.replay.store_transition(state, new_state, reward, action, done)

    def learn(self):
        if self.replay.mem_cntr < self.batch_size:
            return
        state, new_state, reward, action, done = \
            self.replay.sample_memory(self.batch_size)

        reward = T.tensor(reward, dtype=T.float, device=self.critic_1.device)
        done = T.tensor(done, device=self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float, device = self.critic_1.device)
        state = T.tensor(state, dtype=T.float, device=self.critic_1.device)
        action = T.tensor(action, dtype=T.float, device=self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0
        q2_[done] = 0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optim.zero_grad()
        self.critic_2.optim.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optim.step()
        self.critic_2.optim.step()

        self.learn_step_center += 1

        if self.learn_step_center % self.update_actor_interval != 0:
            return

        self.actor.optim.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optim.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()


    def save_models(self):
        self.actor.save()
        self.target_actor.save()
        self.critic_1.save()
        self.target_critic_1.save()
        self.critic_2.save()
        self.target_critic_2.save()

    def load_models(self):
        self.actor.load()
        self.target_actor.load()
        self.critic_1.load()
        self.target_critic_1.load()
        self.critic_2.load()
        self.target_critic_2.load()
