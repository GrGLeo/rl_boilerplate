import gymnasium as gym
from agent import ActorCriticAgent
import numpy as np

env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
agent = ActorCriticAgent(lr=5e-3, n_actions=5, epsilon=0, decay=1e-5, exploration=True)
agent.load_agent("/home/leo/code/rl_boilerplate/CarRacing/models/CarRacing_A2C_lr_0.005_ep_451.h5")
EPISODE = 5

for ep in range(EPISODE):
    done = False
    truncated = False
    score = 0
    state, _ = env.reset()

    while not done and not truncated:
        action = agent.choose_action(state)
        state, reward, done, truncated, _ = env.step(action)
        score += reward

    print(score)
