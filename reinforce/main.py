from agent import Agent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros_like(scores)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 games")
    plt.savefig(figure_file)


if __name__ == "__main__":
    EPISODE = 3000
    env = gym.make("LunarLander-v2")
    inputs = env.observation_space.shape
    agent = Agent(inputs, 0.0005)
    fname = 'REINFORCE_'+ "LunarLander_" + str(agent.alpha) + "_" + str(EPISODE)
    figure_file = fname + ".png"
    scores = []


    for ep in range(EPISODE):
        state,_ = env.reset()
        done = False
        total_reward = 0
        score = 0
        scores = []

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            score += reward
            state = state_

        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print("Episode: ", ep, "Score: ", score, "Avg score: ", avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
