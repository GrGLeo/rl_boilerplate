import gymnasium as gym
from agent_mc import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    agent = Agent(eps=0.001)
    EPISODES = 200_000
    w_l_draw = {-1:0, 0:0, 1:0}
    win_rates = []

    for ep in range(EPISODES):
        if ep > 0 and ep % 1000 == 0:
            pct = w_l_draw[1] / ep
            win_rates.append(pct)

        if ep % 50_000 == 0:
            rates = win_rates[-1] if win_rates else 0.0
            print("starting episode: ",ep, " win rates: ", rates)

        state, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            agent.memory.append((state, action, reward))
            state = state_

        agent.update_Q()
        w_l_draw[reward] += 1

    plt.plot(win_rates)
    plt.show()
