import gymnasium as gym
import numpy as np
import os
from agent import ActorCriticAgent
from utils import create_video, plot_learning_curve
from collections import deque


env = gym.make("CarRacing-v2", continuous=False)
name = "CarRacing_"
agent = ActorCriticAgent(lr=5e-3, n_actions=5, decay=1e-5, exploration=True)


EPISODE = 500
scores = deque(maxlen=100)
best_score = -np.inf
PATH = "CarRacing_A2C_lr_"+str(agent.lr)
MODEL_PATH = os.path.join("models", PATH)
FIGURE_PATH = os.path.join("tmp",PATH+".png")
VIDEO_PATH = os.path.join("video",PATH+".mp4")


for ep in range(EPISODE):
    actions = {0:0,1:0,2:0,3:0,4:0}
    state, _ = env.reset()
    score = 0
    done = False
    truncated = False
    images = []

    while not done and not truncated:
        action = agent.choose_action(state)
        state_, reward, done ,truncated, _ = env.step(action)
        score += reward
        agent.learn(state, reward, state_, done)
        state = state_
        agent.update_epsilon()
        actions[action] += 1
        images.append(state)
        scores.append(score)

    if score > best_score:
        v_path = VIDEO_PATH+"_ep_"+str(ep)
        m_path = MODEL_PATH+"_ep_"+str(ep)+".h5"
        agent.save_agent(m_path)
        create_video(images,v_path)
        best_score = score

    print("episode: ",ep,"exploration rate: ",round(agent.epsilon*100,2),"score: ",round(score,2)\
            ,"avg score: ",round(np.mean(scores),2))
    print(actions)


x = [i+1 for i in range(len(scores))]
plot_learning_curve(scores=scores,x=x,figure_file=FIGURE_PATH)
