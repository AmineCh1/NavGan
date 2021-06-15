import generator
import numpy as np
from helpers import GEN_PATH_DS, extract_directions
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from IPython import display
from scipy.interpolate import griddata
import subprocess


class Directions:
    def __init__(self, init_vec_size):

        self.generator = generator.Generator()
        self.generator.load_state_dict(torch.load(
            GEN_PATH_DS, map_location=torch.device("cpu")))

        self.eigen = extract_directions(self.generator)
        self.init_vec = torch.normal(
            mean=0, std=1, size=(init_vec_size, 2))
        self.init_output = self.generator(self.init_vec).detach()
        self.init_vec.requires_grad = False

    def compute_difference(self, alpha, eig):
        curr = self.generator(torch.tensor(
            self.init_vec + alpha*eig, requires_grad=False))
        diff = curr - self.init_output
        diff = diff.detach()
        return diff.numpy()

    def vis_difference(self):
        """vis_difference [summary]

        Plots eigenvector difference.
        """
        print(self.init_vec)

        init = self.init_output.numpy()

        alphas = np.linspace(0, 1, 20)
        for i, alpha in enumerate(alphas):

            display.clear_output(wait=True)
            norm = [torch.linalg.norm(torch.tensor(
                self.init_vec + alpha*self.eigen[i]), axis=1).detach().numpy() for i in range(2)]

            diff = np.array([self.compute_difference(
                alpha, self.eigen[i]) for i in range(2)])

            fig = plt.figure(figsize=(14, 12), tight_layout=True)
            fig.suptitle("Latent direction variation", fontsize=20)
            gs = gridspec.GridSpec(2, 2)

            ax_temp = plt.subplot(gs[0, :])
            ax_temp.scatter(
                init[:, 0], init[:, 1])
            ax_temp.set_title("Initial Dataset")
            ax_temp.set_xlim(-1, 1)
            ax_temp.set_ylim(-1, 1)
            [s.set_visible(False) for s in ax_temp.spines.values()]

            for j in range(2):
                ax_temp = plt.subplot(gs[1, j])
                sc = ax_temp.quiver(
                    init[:, 0], init[:, 1], diff[j, :, 0], diff[j, :, 1], norm[j])
                sc.set_clim(np.min(norm[j]), np.max(norm[j]))
                plt.colorbar(sc)
                ax_temp.set_title(
                    "Direction: {}, alpha: {}".format(j+1, alpha))
                ax_temp.set_xlim(-1, 1)
                ax_temp.set_ylim(-1, 1)
                [s.set_visible(False) for s in ax_temp.spines.values()]

            plt.savefig("frames_dir/fig_{}".format(i))
            plt.show()

    def video(self):
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 1 -i frames_dir/fig_%d.png -vcodec libx264 -y movie_dir.mp4", shell=True)
        print("Done !")

    def vis_latent_space_abs(self):
        """ Plots latent space difference visualisation 
        """
        gen_input = self.init_vec.detach().numpy()

        grid_x, grid_y = np.mgrid[-2:2:200j, -2:2:200j]

        gen_input = torch.tensor(
            np.stack([grid_x.reshape(-1,), grid_y.reshape(-1,)], axis=1), requires_grad=False).float()
        gen_output = self.generator(gen_input).detach().numpy()

        grid_color = griddata(gen_input, np.abs(gen_output[:, 1])-np.abs(gen_output[:, 0]),
                              (grid_x, grid_y), method='cubic')

        fig, ax = plt.subplots(figsize=(14, 12))
        col = ax.imshow(grid_color.T, extent=(-2, 2, -2, 2))
        plt.colorbar(col, label='|Y| - |X|')
        plt.title("Latent Space coloring (Cross)")
