from agent import Agent
import gymnasium as gym
import numpy as np


env = gym.make("LunarLander-v2", render_mode = 'human')
inputs = env.observation_space.shape
agent = Agent(inputs, 0.0005)
fname = 'REINFORCE_'+ "LunarLander_" + str(agent.alpha) + "_" + "3000_2500"
agent.load_agent(fname)

EPISODE = 5
scores = []

for ep in range(EPISODE):
    state, _ = env.reset()
    done = False
    truncated = False
    score = 0

    while not done and not truncated:
        action = agent.choose_action(state)
        state, reward, done, truncated, info = env.step(action)
        score += reward
    scores.append(score)
    print("reward: ",score)

print("Evaluation completed, average reward: ",round(np.mean(scores),2))
