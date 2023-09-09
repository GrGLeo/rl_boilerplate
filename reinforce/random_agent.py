import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2")

EPISODE = 10

for i in range(EPISODE):
    state,_ = env.reset()
    done = False
    total_reward = 0
    env.render()
    while not done:
        action = env.action_space.sample()
        state_, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print("{}, total reward {}".format(i, total_reward))
