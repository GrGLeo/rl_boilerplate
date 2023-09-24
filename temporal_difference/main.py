import gymnasium as gym
import numpy as np

def simple_policy(state):
    action = 0 if state > 5 else 1
    return action

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    alpha = 0.1
    gamma = 0.99

    states = np.linspace(-0.2094,0.2094,10)
    V = {}
    for state in range(len(states)+1):
        V[state] = 0

    for i in range(5000):
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0

        while not done and not truncated:
            state = np.digitize(obs[2],states)
            action = simple_policy(state)
            obs_, reward, done, truncated, _ = env.step(action)
            state_ = np.digitize(obs_[2], states)
            V[state] = V[state] + alpha*(reward + gamma*V[state_] - V[state])
            obs = obs_

    for state in V:
        print(state, V[state])
