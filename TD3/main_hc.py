from agent import Agent
import gymnasium as gym
import numpy as np


def main():
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode = "rgb_array")
    env = gym.wrappers.RecordVideo(env, "videos_hc", episode_trigger=lambda t: t%100 == 0)
    agent = Agent(alpha=0.001, beta=0.001, state_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=100, n_actions=env.action_space.shape[0], hardcore=True)
    agent.load_models()
    n_games = 1500
    filename = 'Walker2d_hardcore' + str(n_games) + ".png"

    best_scores = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        while not done and not truncated:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_scores:
            agent.save_models()


        print('episode: ',i, 'score: %.2f' %score, "trailing 100 games avg : %.3f" %avg_score)

if __name__ == "__main__":
    main()
