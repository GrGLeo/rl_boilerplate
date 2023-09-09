import gymnasium as gym
import numpy as np
from agent import A2CAgent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    inputs = env.observation_space.shape
    agent = A2CAgent(inputs=inputs, fc1_dims=2048, fc2_dims=1536)
    EPISODE = 3000
    scores = []
    fname = "ACTOR_CRITIC_" + "LunarLander_" + str(agent.fc1_dims) +\
                    "_" + str(agent.fc2_dims) + "_" + str(EPISODE)
    figure_file = fname + ".png"

    for ep in range(EPISODE):
        state,_ = env.reset()
        done = False
        truncated = False
        score = 0

        while not done and not truncated:
            action = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            score += reward
            agent.learn(state, reward ,state_, done)

            state = state_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print("episode: ",ep, "score: ", score, "avg_score: ", avg_score)

    x = [i+1 for i in range(EPISODE)]
    plot_learning_curve(score, x, figure_file)
