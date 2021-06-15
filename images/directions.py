# import generator
import model_skeleton.gan_mnist as gan_mnist
# import gan_mnist
# from model_skeleton.BiGAN import BiGAN, VAE
import numpy as np
import generator
import helpers
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from IPython import display
from matplotlib import gridspec
import matplotlib.patches as patches
import subprocess

# def to_tensor(turns it into float)
LATENT_VECTOR_SEED = 456


class Directions:
    """
    Extract direction from given generative model.
    """

    def __init__(self):
        # self.generator = gan_mnist.G
        self.generator = generator.Generator(latent_dim=helpers.LATENT_DIM_IMG)
        ckpt = torch.load(
            helpers.GEN_PATH_IMG_1, map_location=torch.device(helpers.DEVICE))
        self.generator.load_state_dict(ckpt)
        self.eigenvals, self.eigen = helpers.extract_directions(self.generator)

        torch.manual_seed(LATENT_VECTOR_SEED)

        self.init_vec = torch.normal(
            mean=0, std=1, size=(helpers.NB_SAMPLES, helpers.LATENT_DIM_IMG))

        self.init_output = self.generator(self.init_vec).detach().numpy()
        self.init_vec.requires_grad = False

    def compute_difference(self, alpha, eig):
        '''
        """compute_difference [summary]

        [extended_summary]

        Returns:
            [type]: [description]
        """        '''
        output = self.generator(torch.tensor(
            self.init_vec + alpha*eig, requires_grad=False))
        diff = self.generator(torch.tensor(
            self.init_vec + alpha*eig, requires_grad=False)) - self.generator(torch.tensor(self.init_vec, requires_grad=False))

        output = output.detach()
        diff = diff.detach()

        return output.numpy(), diff.numpy()

    def summary(self):
        # torchsummary.summary(self.generator, input_size=(
        #     helpers.LATENT_DIM_IMG,), device="cpu")
        print(self.generator)

    def plot_eigen_vectors(self, alpha_=20, steps=15):
        '''
        Plots evolution of difference in output space with respect to difference in latent space.
        Args:
            alpha: Magnitude of maximal difference.
            steps: Number of steps to show.
        Returns:
            (None) Plots difference using pyplot.
        '''

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
                    sample = [helpers.reshape_output(x) for x in sample]
                    eig_reshaped.append(sample)

                output_with_vars.append(eig_reshaped)
#                 print(np.shape(np.array(output_with_vars)))
            for eigen_idx, eigen_vector in enumerate(output_with_vars):

                gs_inner = gridspec.GridSpecFromSubplotSpec(
                    helpers.NB_SAMPLES, 1, subplot_spec=gs_outer[eigen_idx])

                for i_img, img in enumerate(eigen_vector):
                    gs_inner_inner = gridspec.GridSpecFromSubplotSpec(
                        3, 1, subplot_spec=gs_inner[i_img])

                    ax_temp = plt.subplot(gs_inner_inner[0])
                    ax_temp.imshow(helpers.reshape_output(
                        self.init_output[i_img]), cmap='gray')
                    ax_temp.set_title("Output")
                    [s.set_visible(False) for s in ax_temp.spines.values()]

                    ax_temp = plt.subplot(gs_inner_inner[1])
#                     print(eigen_vector[i_img].shape)
                    ax_temp.imshow(
                        eigen_vector[0][i_img], cmap='gray')
                    [s.set_visible(False) for s in ax_temp.spines.values()]
                    ax_temp.set_title(
                        'Output by varying direction {}'.format(eigen_idx))

                    ax_temp = plt.subplot(gs_inner_inner[2])
                    ax_temp.imshow(
                        eigen_vector[1][i_img], cmap='gray')
                    ax_temp.set_title("Difference")
                    [s.set_visible(False) for s in ax_temp.spines.values()]

            plt.savefig("frames/figeigen_{}.png".format(a))
            plt.tight_layout()
            plt.show()

    def video_plot_eigen_vectors(self):
        """video_plot_eigen_vectors
            Turns frames of plot_eigen_vectors into video.

        """
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 1 -i frames/figeigen_%d.png  -y movies/movie_eigen.mp4", shell=True)
        print("Done !")

    def plot_gen_in_out(self, location=None, rge=2, steps=10, seed=73, directions=None):
        """plot_gen_in_out [summary]

        [extended_summary]

        Args:
            location ((int,int), optional): Location of input point. Defaults to None and samples randomly in that case !
            rge (int, optional): Range of alpha. Defaults to 2.
            steps (int, optional): Alpha number of steps. Defaults to 10.
            seed (int, optional): Seed for consistency. Defaults to 73.

        """
        print(self.eigen)
        self.generator.eval()

        alphas = np.linspace(-rge, rge, steps)

        if location is None:
            np.random.seed(seed)
            input_ = torch.tensor(np.random.normal(
                0, 1, size=(2, helpers.LATENT_DIM_IMG)), requires_grad=False)
        else:
            input_ = torch.tensor(
                [location[0], location[1]], requires_grad=False)

        for a, alpha in enumerate(alphas):
            fig = plt.figure(figsize=(20, 15), tight_layout=True)
            fig.suptitle("Input and ouput of trained Generator, alpha={}".format(
                alpha), fontsize=14)
            gs = gridspec.GridSpec(2, 2)
            display.clear_output(wait=True)
            for i in range(2):
                curr = torch.tensor(
                    input_ + alpha*self.eigen[i], requires_grad=False)
                otpt = self.generator(
                    curr.float()).detach().numpy()

                ax_temp = plt.subplot(gs[i, 0])

                ax_temp.quiver(input_[0, 0], input_[0, 1],
                               curr[0, 0] - input_[0, 0], curr[0, 1] - input_[0, 1], angles='xy', scale_units='xy', scale=1)
                ax_temp.set_xlim(input_[0, 0]-2, input_[0, 0]+2)
                ax_temp.set_ylim(input_[0, 1]-2, input_[0, 1]+2)
                rect = patches.Rectangle(
                    (input_[0, 0]-1, input_[0, 1]-1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
                ax_temp.add_patch(rect)
                ax_temp.set_title("Direction : {}".format(i))

                ax_temp = plt.subplot(gs[i, 1])
                ax_temp.imshow(helpers.reshape_output(
                    otpt[0, :, :]), cmap="gray")
                if i == 0:
                    ax_temp.set_title("Output")

            plt.savefig("frames/figgen_{}.png".format(a))
            plt.tight_layout()
            plt.show()
        self.generator.train()

    def video_plot_gen_in_out(self):
        print("Preparing video ...")
        subprocess.call(
            "ffmpeg -r 1 -i frames/figgen_%d.png -y movies/movie_gen.mp4", shell=True)
        print("Done !")
