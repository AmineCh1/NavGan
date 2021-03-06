import numpy as np
import torch.nn as nn
import torch
import visualizer
from IPython import display
import torchvision.transforms as transforms
import os
from torchvision import datasets
import helpers


class Generator(nn.Module):
    """Generator [summary]
    Generator class that takes input from latent random distribution and outputs image.
    """

    def __init__(self, latent_dim=helpers.LATENT_DIM_IMG, img_size=helpers.IMG_SIZE, channels=helpers.CHANNELS):
        """__init__ 
        Initializes Generator instance.

        Args:
            latent_dim ([type], optional): [description]. Defaults to helpers.LATENT_DIM_IMG.
            img_size ([type], optional): [description]. Defaults to helpers.IMG_SIZE.
            channels ([type], optional): [description]. Defaults to helpers.CHANNELS.

        Returns:
            [type]: [description]
        """
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.img_size = img_size
        self.channels = channels
        self.model = nn.Sequential(

            *block(latent_dim, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels*img_size*img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels,
                       self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    """
    Discriminator of GAN. Funnel-like structure to the classify the input either real or from Generator.
    """

    def __init__(self, img_size=helpers.IMG_SIZE, channels=helpers.CHANNELS):
        """__init__ 
        Initiates Discriminator instance.

        Args:
            img_size (int, optional): Size of image input. Defaults to helpers.IMG_SIZE.
            channels (int, optional): Number of images per channel. Defaults to helpers.CHANNELS.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(channels*img_size*img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Trainer:
    """ Class that encapsulates a generator and a discriminator, with the addition of a training method.
    """

    def __init__(self, eps=200, latent_dim=helpers.LATENT_DIM_IMG, img_size=helpers.IMG_SIZE, channels=helpers.CHANNELS, batch_size=helpers.BATCH_SIZE_IMG):
        """__init__ 
        Constructs Generator/ Discriminator couple with specified parameters.

        Args:
            eps (int, optional): Number of epochs of trining for GAN. Defaults to 200.
            latent_dim (int, optional): Latent input dimension of Generator. Defaults to helpers.LATENT_DIM_IMG.
            img_size (int, optional): Dimension of output image of Generator. Defaults to helpers.IMG_SIZE.
            channels (int, optional): Number of channels per image. Defaults to helpers.CHANNELS.
            batch_size (int, optional): Number of images per batch to train on. Defaults to helpers.BATCH_SIZE_IMG.
        """

        os.makedirs("data/mnist", exist_ok=True)

        #Misc and Hyperparams
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.epochs = eps
        self.vis = visualizer.Visualizer(eps)

        # NNs
        self.generator = Generator(
            self.latent_dim, self.img_size, self.channels)
        self.discriminator = Discriminator(self.img_size, self.channels)

        # Optimizers
        self.opt_G = torch.optim.Adam(self.generator.parameters())
        self.opt_D = torch.optim.Adam(self.discriminator.parameters())

        # Loss
        self.adv_loss = torch.nn.BCELoss()

        # Images to train on
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "data/mnist",
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(
                    ), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    def CUDA(self):
        """Transfers parameters of generator, discriminator and loss to GPU memory.
        """

        if helpers.CUDA:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adv_loss.cuda()

    def train(self):
        """Performs training of the generator/discriminator couple. 
        """

        g_loss_evolution = []
        magnitude_evolution = []
        np.random.seed(44)
        gen_input_fixed = torch.normal(
            mean=0, std=1, size=(16, self.latent_dim)).cuda()
        for epoch in range(self.epochs):
            g_loss_avg = []
            magnitude_avg = []
            display.clear_output(wait=True)
            for i, (imgs, _) in enumerate(self.dataloader):

                # Ground truths, no gradient necessary
                gt_valid, gt_fake = torch.ones(size=(imgs.size(0), 1)).cuda(
                ), torch.zeros(size=(imgs.size(0), 1)).cuda()

                real_imgs = torch.tensor(imgs).float().to("cuda")

                # -----------------
                # Train Generator
                # -----------------

                self.opt_G.zero_grad()
                # Generate input in latent dim for generator ( i.e sample noise ...)
                gen_input = torch.normal(mean=0, std=1, size=(
                    imgs.shape[0], self.latent_dim)).cuda()

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

                gen_out_detached = gen_output.detach()

                grad_mag = torch.norm(gen_out_detached)

                magnitude_avg.append(grad_mag.item())

                decision_nograd_g = self.discriminator(gen_out_detached)

                real_loss = self.adv_loss(
                    self.discriminator(real_imgs), gt_valid)

                fake_loss = self.adv_loss(decision_nograd_g, gt_fake)

                d_loss = (real_loss+fake_loss)/2
                d_loss.backward()

                self.opt_D.step()
            print(gen_input_fixed.device)
            gen_out_display = self.generator(
                gen_input_fixed).detach().cpu().clone().numpy()

            self.vis.update(g_loss_evolution, magnitude_evolution,
                            gen_out_display)

            g_loss_evolution.append(np.mean(g_loss_avg))
            magnitude_evolution.append(np.mean(magnitude_avg))

            self.vis.display(epoch)

        self.vis.video()

    def save_generator(self, path=helpers.GEN_PATH_IMG):
        """save_generator 
        Save Generator model checkpoint to specified path.

        Args:
            path (str, optional): Path to save Generator. Defaults to helpers.GEN_PATH_IMG.
        """
        torch.save(self.generator.state_dict(), path)
