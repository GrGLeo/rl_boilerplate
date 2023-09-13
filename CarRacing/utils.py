import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from typing import List, Optional
from agent import ActorCriticAgent
import torch as T


def create_video(images: List, path: str) -> None:
    fps = 30
    file_name = path+"_video.mp4"
    writer = imageio.get_writer(file_name, fps=fps)

    for image in images:
        writer.append_data(image)

    writer.close()

    print(".. Video created ..")


def plot_learning_curve(scores: List, x: List, figure_file: str) -> None:
    running_avg = np.zeros_like(scores)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 games")
    plt.savefig(figure_file)


def save_every(agent: ActorCriticAgent, ep: int, path: str, every: Optional[int] = 100):
    if ep % every == 0:
        agent.save_agent(path+"_"+str(ep)+".h5")


def grayscale_resize(state: np.array) -> np.array:
    """Return the a grayscale image, resize to 48*48, and normalize"""
    state = cv2.cvtColor(state,cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state,(48,48))
    state = np.expand_dims(state,-1)
    state = state / 255
    return state
