import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import subprocess


class Visualizer:
    def __init__(self, eps=200, g_loss=[], magnitude=[], g_output=[]):
        self.g_loss = g_loss
        self.mag = magnitude
        self.g_output = g_output
        self.eps = eps

    def update(self, g_loss, magnitude, g_output):
        self.g_loss = g_loss
        self.magnitude = magnitude
        self.g_output = g_output

    def display(self, i):

        fig = plt.figure(figsize=(18, 15), tight_layout=True)
        fig.suptitle("Training evolution, epoch {}".format(i), fontsize=14)
        gs = gridspec.GridSpec(2, 2)

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            4, 4, subplot_spec=gs[:, 1])
        for j, img in enumerate(self.g_output):
            ax_temp = plt.subplot(gs_inner[j // 4, j % 4])
            ax_temp.imshow(img[0, :, :], cmap='gray')
            [s.set_visible(False) for s in ax_temp.spines.values()]

        ax_temp = plt.subplot(gs[0, 0])
        ax_temp.plot(self.g_loss)
        ax_temp.set_xticks(np.arange(0, self.eps, step=int(self.eps/10)))
        ax_temp.set_xlabel("Epochs")
        ax_temp.set_ylabel("Generator loss")

        ax_temp = plt.subplot(gs[1, 0])
        ax_temp.plot(self.magnitude)
        ax_temp.set_xticks(np.arange(0, self.eps, step=int(self.eps/10)))
        ax_temp.set_xlabel("Epochs")
        ax_temp.set_ylabel("Gradient magnitude")

        plt.savefig("frames/fig_{}.png".format(i))
        plt.show()

    def video(self):
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 5 -i frames/fig_%d.png -vcodec libx264 -y movie.mp4", shell=True)
        print("Done !")
