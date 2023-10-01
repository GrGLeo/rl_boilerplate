import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state = np.zeros((self.mem_size, *input_shape))
        self.state_ = np.zeros((self.mem_size, *input_shape))
        self.reward = np.zeros(self.mem_size)
        self.actions = np.zeros((self.mem_size, n_actions))
        self.terminal = np.zeros((self.mem_size), dtype=np.bool_)


    def store_transition(self, state, state_, reward, action, done):
        i = self.mem_cntr % self.mem_size
        self.state[i] = state
        self.state_[i] = state_
        self.reward[i] = reward
        self.actions[i] = action
        self.terminal[i] = done

        self.mem_cntr += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state[batch]
        states_ = self.state_[batch]
        rewards = self.reward[batch]
        actions = self.actions[batch]
        dones = self.terminal[batch]

        return states, states_, rewards, actions, dones
