import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import subprocess


class Visualizer:
    """ 
    Visualizer class for GAN training.
    """

    def __init__(self, eps=200):
        """__init__ 
        Initializes Visualizer instance.
        Args:
            eps (int, optional): Number of epochs GAN was trained on. Defaults to 200.
            g_loss (list, optional): Generator loss list. Defaults to [].
            magnitude (list, optional):Gradient magnitude of generator loss so far. Defaults to [].
            g_output (list, optional): . Defaults to [].
        """
        self.g_loss = []
        self.mag = []
        self.g_output = []
        self.eps = eps

    def update(self, g_loss, magnitude, g_output):
        """update 
        Updates generator loss, gradient magitude and generator output for visualization.

        Args:
            g_loss (list): Current Generator loss.
            magnitude (list): Current magnitude.
            g_output (list): Current Generator output.
        """
        self.g_loss = g_loss
        self.magnitude = magnitude
        self.g_output = g_output

    def display(self, i):
        """display 
        Display current status of Generator output, Generator loss, and gradient magnitude.

        Args:
            i (int): Current epoch.
        """
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
        """video 
        Once training is done, turn frames into video.

        """
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 5 -i frames/fig_%d.png -vcodec -y movie.mp4", shell=True)
        print("Done !")
