import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from IPython import display
import model_skeleton.BiGAN.VAE as VAE
import helpers

IMG_SIZE = 28


class VAEViz:
    """
    Class for various encoder/decoder visualizations, in the context of variational autoencoders.
    """

    def __init__(self):
        """
        Initialization of Encoder and Decoder (Pretrained Encoder and Decoders, with latent input of dimension n= 2)
        """
        self.encoder = VAE.Encoder(latent_dim=2)
        self.decoder = VAE.Decoder(latent_dim=2)

        ckpt_encoder = torch.load(
            helpers.ENCODER_PATH_VAE, map_location=torch.device(helpers.DEVICE))
        ckpt_decoder = torch.load(
            helpers.DECODER_PATH_VAE, map_location=torch.device(helpers.DEVICE))

        self.encoder.load_state_dict(ckpt_encoder)
        self.decoder.load_state_dict(ckpt_decoder)

        self.eigen_encoder = [np.reshape(x, (IMG_SIZE, IMG_SIZE)) for x in helpers.extract_directions(
            self.encoder, layer_level=0)[1]]
        self.eigen_encoder_values = helpers.extract_directions(
            self.encoder, layer_level=0)[0]

        self.eigen_decoder_values, self.eigen_decoder = helpers.extract_directions(
            self.decoder)

        self.encoder_data = datasets.MNIST(
            "data/mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(helpers.IMG_SIZE), transforms.ToTensor(
                ), transforms.Normalize([0.5], [0.5])]
            )
        )

    def encoder_output(self, eigen_noise=True, eigen_index=0, inputs=2):
        """encoder_output [summary]

        Samples some input images, and plots the output of a modified input by either applying random noise to the input for a few iterations, or with respect to a specific eigenvector of varying intensity.

        Args:
            eigen_noise (bool, optional): [description]. Defaults to True.
            eigen_index (int, optional): [description]. Defaults to 0.
            inputs (int, optional): [description]. Defaults to 2.
        """
        self.encoder.eval()

        imgs, _ = zip(*list(self.encoder_data)[0:inputs])
#
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(inputs, 4)
        alphas = np.linspace(0, 2, 20)

        for alpha in alphas:
            figure = plt.figure(figsize=(15, 15))
            display.clear_output(wait=True)
            for i in range(inputs):

                ax_temp = plt.subplot(gs[i, 0])
                ax_temp.imshow(imgs[i][0, :, :], cmap='gray')

                input_ = torch.tensor(imgs[i])

                mean, variance = self.encoder(input_)

                mean = np.squeeze(helpers.to_numpy(mean))
                variance = np.squeeze(helpers.to_numpy(variance))

                sample = np.random.multivariate_normal(
                    mean, np.exp(np.diag(variance)), size=50)

                ax_temp = plt.subplot(gs[i, 1])
                ax_temp.scatter(sample[:, 0], sample[:, 1])
                ax_temp.set_xlim(-3, 3)
                ax_temp.set_ylim(-3, 3)

                if eigen_noise:
                    noise = alpha * np.sqrt(784) * \
                        self.eigen_encoder[eigen_index]
                else:
                    noise = np.zeros_like(imgs[i])
                    indices = [np.random.randint(0, high=IMG_SIZE, size=25), np.random.randint(
                        0, high=IMG_SIZE, size=25)]
                    for pt_x, pt_y in zip(*indices):
                        noise[0, pt_x, pt_y] = 1

                img_noisy = imgs[i] + noise

                ax_temp = plt.subplot(gs[i, 2])
                ax_temp.imshow(img_noisy[0, :, :], cmap='gray')

                input_ = torch.tensor(img_noisy)
                mean, variance = self.encoder(input_)

                mean = np.squeeze(helpers.to_numpy(mean))
                variance = np.squeeze(helpers.to_numpy(variance))

                sample = np.random.multivariate_normal(
                    mean, np.exp(np.diag(variance)), size=50)

                ax_temp = plt.subplot(gs[i, 3])
                ax_temp.scatter(sample[:, 0], sample[:, 1])
                ax_temp.set_xlim(-3, 3)
                ax_temp.set_ylim(-3, 3)
                ax_temp.set_title(
                    "Variance X : {} / Variance Y : {}".format(np.exp(variance[0]), np.exp(variance[1])))
            plt.show()

    def decoder_encoder_spread(self, val_index=12, vectors=10, seed=0):
        """encoder_decoder_spread :
            Computes and plots "spread" of specified eigenvector of first layer of Encoder after performing Decoder(Encoder(eigenvector)).
            Computation of "spread":
            Spread(v_i) = [Decoder(Encoder(v_i))@v_0.T, Decoder(Decoder(v_i))@v_1.T, ...]

        Args:
            val_index (int, optional): Index of eigenvector. Defaults to 0.
            scale_vect (int, optional): Scale of eigenvector input. Defaults to 1.
            seed (int, optional): Seed for consistency.
        Returns
            (None): Plots (v_(val_index)), Decoder(Encoder(v_(val_index))), and barplot of Spread(v_i).
        """
        self.encoder.eval()
        self.decoder.eval()

        input_ = torch.tensor(self.eigen_encoder[val_index]).unsqueeze_(0)

        output_mean, output_var = self.encoder(input_)
        # output_mean, output_var = [np.squeeze(
        #     helpers.to_numpy(x)) for x in self.encoder(input_)]

        # sample = torch.tensor(np.random.multivariate_normal(output_mean, np.diag(
        #     np.exp(output_var)), 1)).float()

        first_pass_mean, first_pass_var = [np.squeeze(
            helpers.to_numpy(x)) for x in self.decoder(output_mean)]

        # first_pass = np.random.multivariate_normal(first_pass_mean.flatten(), np.diag(
        #     first_pass_var), 1)

        fig = plt.figure(figsize=(13, 15))
        # first_pass = np.reshape(first_pass, (IMG_SIZE, IMG_SIZE))
        gs = gridspec.GridSpec(2, 2)
        ax_temp = plt.subplot(gs[0, 0])
        ax_temp.imshow(self.eigen_encoder[val_index], cmap='gray')
        ax_temp.set_title("Eigenvector v_{}".format(val_index))

        ax_temp = plt.subplot(gs[0, 1])
        ax_temp.imshow(first_pass_mean, cmap='gray')
        ax_temp.set_title("Decoder(Encoder(v_{}))".format(val_index))

        flattened_eigenvecs = [x.flatten()
                               for x in self.eigen_encoder[0:vectors]]
        first_pass_mean = first_pass_mean.flatten()
        scalar_product = [
            first_pass_mean@x.T/(np.linalg.norm(first_pass_mean)*np.linalg.norm(x)) for x in flattened_eigenvecs]

        ax_temp = plt.subplot(gs[1, :])
        ax_temp.bar(range(len(scalar_product)), scalar_product)

    def encoder_decoder_spread(self, val_index=0, scale_vect=1):
        """encoder_decoder_spread :
            Computes and plots "spread" of specified eigenvector after performing Encoder(Decoder(eigenvector)).
            Computation of "spread":
            Spread(v_i) = [Encoder(Decoder(v_i))@v_0.T, Encoder(Decoder(v_i))@v_1.T, ...]

        Args:
            val_index (int, optional): Index of eigenvector. Defaults to 0.
            scale_vect (int, optional): Scale of eigenvector input. Defaults to 1.
        Returns
            (None): Plots barplot of Spread(v_i).
        """

        self.encoder.eval()
        self.decoder.eval()

        input_ = scale_vect * \
            torch.tensor(self.eigen_decoder[val_index]).unsqueeze_(0)
        output_mean, output_var = [np.squeeze(
            helpers.to_numpy(x)) for x in self.decoder(input_)]

        sample = torch.tensor(np.random.multivariate_normal(output_mean.flatten(), np.diag(
            np.exp(output_var)), 1)).float()

        first_pass_mean, first_pass_var = [np.squeeze(
            helpers.to_numpy(x)) for x in self.encoder(sample)]

        fig = plt.figure(figsize=(13, 15))
        gs = gridspec.GridSpec(2, 2)
        ax_temp = plt.subplot(gs[0, 0])
        ax_temp.quiver(0, 0, self.eigen_decoder[val_index][0],
                       self.eigen_decoder[val_index][1], scale=3.0, cmap='gray')
        ax_temp.set_xlim(-1, 1)
        ax_temp.set_ylim(-1, 1)

        ax_temp = plt.subplot(gs[0, 1])
        ax_temp.quiver(
            0, 0, first_pass_mean[0], first_pass_mean[1], scale=6.0, cmap='gray')
        ax_temp.set_xlim(-1, 1)
        ax_temp.set_ylim(-1, 1)

        scalar_product = [
            first_pass_mean@x.T/(np.linalg.norm(first_pass_mean)*np.linalg.norm(x)) for x in self.eigen_decoder]
        scalar_product = [np.squeeze(x) for x in scalar_product]
        print(scalar_product)
        ax_temp = plt.subplot(gs[1, :])
        ax_temp.bar(range(len(scalar_product)), scalar_product)

    def encoder_decoder_energy(self):
        """encoder_decoder_energy:
        Computes difference of "energy"  of eigenvectors v_i after performing Encoder(Decoder(v_i)).
        This allows to measure the preservation of a vector after mapping it to the output space and back to the input space.
        Energy formula: 
            Energy(v_i) = Sum_j(lambda_j *(v_i.T@v_j)).
        Difference Formula: 
            Difference(v_i) = Energy( Encoder( Decoder( v_i))) - Energy(v_i).
        Returns:
            diff: [Difference(v_0), Difference(v_1)]
        """

        self.encoder.eval()
        self.decoder.eval()

        energies_before = []
        energies_after = []

        for eigenvector in self.eigen_decoder:

            val = np.sum([u*eigenvector@x.T for u,
                         x in zip(self.eigen_decoder_values, self.eigen_decoder)])
            energies_before.append(val)

            input_ = torch.tensor(eigenvector)
            output_mean, output_var = self.decoder(input_.unsqueeze_(0))

            first_pass_mean, first_pass_var = [np.squeeze(
                helpers.to_numpy(x)) for x in self.encoder(output_mean)]

            energy = np.sum([u*(first_pass_mean@x.T)
                            for u, x in zip(self.eigen_decoder_values, self.eigen_decoder)])

            energies_after.append(energy)

        diff = [after - before for (after, before)
                in zip(energies_after, energies_before)]

        return diff
