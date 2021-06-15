import numpy as np
import torch.nn as nn
import torch
import visualizer
from IPython import display
import matplotlib.pyplot as plt
from helpers import *
import time


class Generator(nn.Module):

    def __init__(self, ranges=[]):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # * is used to unpack the layers iterable into arguments for nn.Sequential
        self.model = nn.Sequential(
            *block(2, 16, normalize=False),
            *block(16, 32, normalize=False),
            *block(32, 16, normalize=False),
            *block(16, 8, normalize=False),
            *block(8, 2, normalize=False),
            nn.Tanh()

        )

        self.ranges = ranges

    def map_to_range(self, points):
        new_points = torch.empty_like(points)
        new_points[:, 0] = self.ranges[0, 0] + \
            ((points[:, 0] + 1)*(self.ranges[0, 1]-self.ranges[0, 0]))/2
        new_points[:, 1] = self.ranges[1, 0] + \
            ((points[:, 1] + 1)*(self.ranges[1, 1]-self.ranges[1, 0]))/2
        return new_points

    def forward(self, z):
        points = self.model(z)
        if (len(self.ranges) > 0):
            return self.map_to_range(points)
        else:
            return points


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(2, 16, normalize=False),
            *block(16, 32, normalize=False),
            *block(32, 64, normalize=False),
            *block(64, 128, normalize=False),
            *block(128, 64, normalize=False),
            *block(64, 32, normalize=False),
            *block(32, 16, normalize=False),
            *block(16, 8, normalize=False),
            *block(8, 2, normalize=False),
            *block(2, 1, normalize=False)
        )

    def forward(self, dp):
        validity = self.model(dp)
        return validity


class Trainer:
    """ Class that encapsulates a generator and a discriminator, with the addition of a training method.
    """

    def __init__(self, eps=200, ds_size=20000, batch_size=100):
        """ Constructor.

        Args:
            eps (int, optional): Number of epochs to run in the training phase. Defaults to 200.

            ds_size (int, optional): Size of the dataset. Defaults to 20000.
            batch_size (int, optional): Size of batch. Defaults to 100.
        """

        dataset = generate_cross(ds_size, width=5)
        ranges = torch.tensor([[np.min(dataset[:, 0]), np.max(dataset[:, 0])], [
                              np.min(dataset[:, 1]), np.max(dataset[:, 1])]])

        # Misc and hyperparams
        self.epochs = eps
        self.vis = visualizer.VisualizerTraining(eps)
        self.batch_size = batch_size
        self.ds_size = ds_size
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        # NNs
        self.generator = Generator(ranges)
        self.discriminator = Discriminator()

        # Optimizers
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        # Losses
        self.adv_loss = torch.nn.BCEWithLogitsLoss()
        self.adv_loss_map = torch.nn.BCEWithLogitsLoss(reduction='none')

    def cuda(self):
        """Transfers parameters of generator, discriminator and loss to GPU memory.
        """
        if CUDA:
            print(CUDA)
            self.generator.to("cuda")
            self.discriminator.to("cuda")
            self.adv_loss.to("cuda")

    def train(self, fix_latent_seed=False, seed_size=100):
        """Performs training of the generator/discriminator couple.

        [extended_summary]
        """

        g_loss_evolution = []
        magnitude_evolution = []
        np.random.seed(44)
        gen_input_fixed = torch.normal(
            mean=0, std=1, size=(seed_size, 2)).to("cuda")
        for epoch in range(self.epochs):
            g_loss_avg = []
            magnitude_avg = []
            display.clear_output(wait=True)
            for batch_nb, pts in enumerate(self.dataloader):
                # for batch in dataloader:
                # gen input to "compare with" batch:
                # then perform the same steps as image training ...

                real_datapoints = torch.tensor(pts).float().to("cuda")

                # Avoid non-compatible batch_sizes
                batch_size_tmp = pts.shape[0]

                # Ground truths, no gradient necessary
                gt_valid, gt_fake = torch.ones(size=(batch_size_tmp, 1)).to(
                    "cuda"), torch.zeros(size=(batch_size_tmp, 1)).to("cuda")

                # -----------------
                # Train Generator
                # -----------------

                self.opt_G.zero_grad()
                # Generate input in latent dim for generator ( i.e sample noise ...)

                gen_input = torch.normal(
                    mean=0, std=1, size=(batch_size_tmp, 2)).to("cuda")

                gen_output = self.generator(gen_input)

                decision = self.discriminator(gen_output)

                g_loss = self.adv_loss(decision, gt_valid)

                g_loss.backward()

                self.opt_G.step()
                g_loss_avg.append(g_loss.item())

                # ---------------------
                # Train Discriminator
                # ---------------------

                self.opt_D.zero_grad()

                gen_out_detached = gen_output.detach().cuda()
                gen_out_detached.requires_grad = True

                decision_nograd_g = self.discriminator(gen_out_detached)
                real_loss = self.adv_loss(
                    self.discriminator(real_datapoints), gt_valid)

                fake_loss = self.adv_loss(decision_nograd_g, gt_fake)
                fake_loss_map = self.adv_loss_map(decision_nograd_g, gt_fake)

                d_loss = (real_loss+fake_loss)/2
                d_loss.backward()

                grad_mag = torch.mean(torch.norm(gen_out_detached.grad, dim=1))

                magnitude_avg.append(grad_mag.item())

                self.opt_D.step()

                fake_loss_detached = to_numpy(fake_loss_map)

            magnitude_evolution.append(np.mean(magnitude_avg))
            g_loss_evolution.append(np.mean(g_loss_avg))

            if (fix_latent_seed):

                gen_out_detached = self.generator(gen_input_fixed)

                decision_nograd_g = self.discriminator(gen_out_detached)

                gt_fake = torch.zeros(size=(seed_size, 1)).to("cuda")

                fake_loss_detached = to_numpy(
                    self.adv_loss_map(decision_nograd_g, gt_fake))

            self.vis.update(g_loss_evolution, magnitude_evolution,
                            to_numpy(gen_out_detached), fake_loss_detached)

            self.vis.display(epoch)

        # Test with large input
        final_in = torch.randn(self.ds_size, 2, device='cuda:0')
        final_out = self.generator(final_in)
        self.vis.disp_final(to_numpy(final_out))
        self.vis.video()

        # Print extracted eigenvectors
        vecs = extract_directions(self.generator)
        for i, vector in enumerate(vecs):
            print(" Direction %d:" % (i+1))
            print(vector)

    def save_generator(self, path=GEN_PATH_DS):
        """save_generator 
        Save Generator model checkpoint to specified path.

        Args:
            path (str, optional): Path to save Generator. Defaults to helpers.GEN_PATH_DS.
        """
        torch.save(self.generator.state_dict(), path)

    def print_state(self):
        return (self.generator.state_dict())
