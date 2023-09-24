import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

class Agent():
    def __init__(self,n_actions, state_space, eps_decay, alpha=0.1, gamma=0.99, epsilon=1,
                 min_eps=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.n_actions = n_actions

        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]
        self.Q = {}

        self.init_Q()


    def init_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state,action)] = 0.0

    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.max_action(state)
        return action

    def update_eps(self):
        if self.epsilon > self.min_eps:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.min_eps


    def learn(self,state, action, state_, reward):
        a_max = self.max_action(state_)
        self.Q[(state,action)] = self.Q[(state,action)] + self.alpha*(reward + \
            self.gamma*self.Q[(state_, a_max)] - self.Q[(state,action)])


class CartPoleDigitizer():
    def __init__(self, bounds=(4.8, 3.7, 4.1, 3.7), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states = self.get_state_space()

    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i,j,k,l))
        return states

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x,self.position_space))
        cart_xdot = int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_thetadot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x, cart_xdot, pole_theta, pole_thetadot)

def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.show()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    n_games = 50_000
    digitizer = CartPoleDigitizer()
    eps_decay = 2 / n_games
    agent = Agent(n_actions=2, state_space=digitizer.states, eps_decay=eps_decay)


    scores = []
    for ep in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        state = digitizer.digitize(observation)

        while not done and not truncated:
            action = agent.choose_action(state)
            observation_, reward, done, truncated, _ = env.step(action)
            state_ = digitizer.digitize(observation_)
            agent.learn(state, action, state_, reward)
            state = state_
            score += reward
        agent.update_eps()
        scores.append(score)

        if ep % 1000 == 0:
            print(f"episode: {ep}, score: {round(score,2)} exploration rate: {round(agent.epsilon,2)}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores,x)
