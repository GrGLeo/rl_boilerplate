import gymnasium as gym
import numpy as np
from agent import Agent

def main():
    env = gym.make("BipedalWalker-v3", render_mode = "human")
    agent = Agent(alpha=0.001, beta=0.001, state_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=100, n_actions=env.action_space.shape[0])
    agent.load_models()
    agent.eval()
    n_games = 10

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        while not done and not truncated:
            action = agent.choose_action(observation)
            observation, reward, done, truncated, _ = env.step(action)
            score += reward

        print('episode: ',i, 'score: %.2f' %score)

if __name__ == "__main__":
    main()
