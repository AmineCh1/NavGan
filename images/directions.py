# import generator
# import generator_fmnist
# import gan_mnist
from model_skeleton.BiGAN import BiGAN, VAE
import numpy as np
import helpers
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from IPython import display
from matplotlib import gridspec
import subprocess


LATENT_VECTOR_SEED = 69


class Directions:
    def __init__(self):
        # self.generator = generator_fmnist.Generator(
        #     helpers.LATENT_DIM_FMNIST, helpers.HIDDEN_DIM_FMNIST, helpers.OUTPUT_DIM_FMNIST)
        self.generator = BiGAN.Generator(latent_dim=helpers.LATENT_DIM_BIGAN)
        ckpt = torch.load(
            helpers.GEN_PATH_BIGAN, map_location=torch.device(helpers.DEVICE))
        self.generator.load_state_dict(ckpt)

        self.eigenvals, self.eigen = helpers.extract_directions(self.generator)

        torch.manual_seed(LATENT_VECTOR_SEED)
        self.init_vec = torch.normal(
            mean=0, std=1, size=(helpers.NB_SAMPLES, helpers.LATENT_DIM_BIGAN))

        self.init_output = self.generator(self.init_vec).detach().numpy()
        self.init_vec.requires_grad = False

    def compute_difference(self, alpha, eig):
        diff = self.generator(torch.tensor(
            self.init_vec + alpha*eig, requires_grad=False)) - self.generator(torch.tensor(self.init_vec, requires_grad=False))
        diff = diff.detach()
        return diff.numpy()

    def plot_eigen_vectors(self, alpha_=0.05, steps=15):

        alphas = np.linspace(0, alpha_, steps)
        for a, alpha in enumerate(alphas):
            fig = plt.figure(figsize=(20, 15), tight_layout=True)
            fig.suptitle("Eigenvector differences, alpha = {}".format(
                alpha), fontsize=14)
            gs_outer = gridspec.GridSpec(1, len(self.eigen))
            display.clear_output(wait=True)
            output_with_vars = []
            for eigen in self.eigen:
                eig = self.compute_difference(alpha, eigen)
                eig_reshaped = []
                for sample in eig:
                    sample = helpers.reshape_output(sample)
                    eig_reshaped.append(sample)
                output_with_vars.append(eig_reshaped)

            for eigen, eigen_vector in enumerate(output_with_vars):

                gs_inner = gridspec.GridSpecFromSubplotSpec(
                    helpers.NB_SAMPLES, 1, subplot_spec=gs_outer[eigen])

                for i_img, img in enumerate(self.init_output):
                    gs_innerr = gridspec.GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=gs_inner[i_img])

                    ax_temp = plt.subplot(gs_innerr[0])
                    ax_temp.imshow(helpers.reshape_output(img))
                    [s.set_visible(False) for s in ax_temp.spines.values()]

                    ax_temp = plt.subplot(gs_innerr[1])
                    ax_temp.imshow(
                        eigen_vector[i_img])
                    [s.set_visible(False) for s in ax_temp.spines.values()]

            plt.savefig("frames/figeigen_{}.png".format(a))
            plt.show()

    def video_eigen(self):
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 2 -i frames/figeigen_%d.png -vcodec libx264 -y movie_eigen.mp4", shell=True)
        print("Done !")

    def plot_gen_in_out(self):
        return None
        # def vis_difference(self):
        #     print(self.init_vec)

        #     init = self.init_output()

        #     fig = plt.figure(figsize=(20, 15), tight_layout=True)
        #     fig.suptitle("Latent space differences", fontsize=14)
        #     gs = gridspec.GridSpec(4, 4)

        #     for j, img in enumerate(init):
        #         gs_inner = Gridspec.GridSpecFromSubplotSpec(
        #             2, 1, supblot_spec=gs[j//4, j % 4])
        #         ax_temp = plt.subplot(gs_inner[0])
        #         ax_temp.imshow(img[0, :, :])
        #     alphas = np.linspace(0, 0.1, 20)
        #     for alpha in alphas:

        #         display.clear_output(wait=True)

        #         diff = np.array([self.compute_difference(
        #             init, alpha, self.eigen[i]) for i in range(2)])

        #         fig = plt.figure(figsize=(14, 12), tight_layout=True)
        #         fig.suptitle("Latent direction variation", fontsize=14)
        #         gs = gridspec.GridSpec(2, 2)

        #         ax_temp = plt.subplot(gs[0, :])
        #         ax_temp.scatter(
        #             init[:, 0], init[:, 1])
        #         ax_temp.set_title("Initial Dataset")
        #         ax_temp.set_xlim(-1, 1)
        #         ax_temp.set_ylim(-1, 1)
        #         [s.set_visible(False) for s in ax_temp.spines.values()]

        #         for j in range(2):
        #             ax_temp = plt.subplot(gs[1, j])

        #             ax_temp.quiver(
        #                 init[:, 0], init[:, 1], diff[j, :, 0], diff[j, :, 1])
        #             ax_temp.set_title(
        #                 "Direction: {}, alpha: {}".format(j+1, alpha))
        #             ax_temp.set_xlim(-1, 1)
        #             ax_temp.set_ylim(-1, 1)
        #             [s.set_visible(False) for s in ax_temp.spines.values()]

        #         plt.show()
