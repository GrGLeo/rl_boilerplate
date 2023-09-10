import gymnasium as gym
import numpy as np
from agent import A2CAgent

env = gym.make("LunarLander-v2", render_mode = "human")
EPISODE = 5
scores = []
inputs = env.observation_space.shape
agent = A2CAgent(inputs=inputs, fc1_dims=2048, fc2_dims=1536)
agent.load_model("ACTOR_CRITIC_LunarLander_2048_1536_3000")

for ep in range(EPISODE):
    done = False
    truncated = False
    score = 0

    state, _ = env.reset()

    while not done and not truncated:
       action = agent.choose_action(state)
       state, reward, done, truncated, _ = env.step(action)
       score += reward

    scores.append(score)
    print("Episode: ",ep, "score: ",score)
print("Average score: ",np.mean(scores))
