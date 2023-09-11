import numpy as np
import imageio
import matplotlib.pyplot as plt


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
