from network import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer


class Agent():
    def __init__(self, alpha, beta, state_dims, n_actions):
        self.actor = ActorNetwork(actor_lr=alpha, state_dim=state_dims, n_actions=n_actions,
                                  name="actor")
        self.critic = CriticNetwork(critic_lr=beta, state_dim=state_dims, n_actions=n_actions,
                                    name="critic")
        self.replay = ReplayBuffer(max_len=10_000, batch_size=64, state_dims=state_dims, actions_dims=n_actions)

    def choose_action(self, state):
        state = state.to(self.actor.device)
        prob = (self.)
