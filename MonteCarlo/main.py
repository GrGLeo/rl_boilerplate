import gymnasium as gym
from agent import Agent


def main():
    env = gym.make("Blackjack-v1")
    agent = Agent()
    EPISODE = 500_000

    for ep in range(EPISODE):
        if ep%50000 == 0:
            print("starting episode: ",ep)
        done = False
        state, _ = env.reset()

        while not done:
            action = agent.policy(state)
            state_, reward, done, _, _ = env.step(action)
            agent.memory.append((state, reward))
            state = state_
        agent.update_v()
    print(agent.V[(21,3,True)])
    print(agent.V[(4,1,False)])

if __name__ == "__main__":
    main()


-3+2-2+3
