import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2


def create_video(images, path):
    fps = 30
    file_name = path+"_video.mp4"
    writer = imageio.get_writer(file_name, fps=fps)

    for image in images:
        writer.append_data(image)

    writer.close()

    print(".. Video created ..")


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros_like(scores)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 games")
    plt.savefig(figure_file)


def save_every(agent, ep, path, every=100):
    if ep % every == 0:
        agent.save_agent(path+"_"+str(ep)+".h5")


def grayscale_resize(state):
    state = cv2.cvtColor(state,cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state,(48,48))
    state = np.expand_dims(state,-1)
    return state
