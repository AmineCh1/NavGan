import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import visualizer
import os
from IPython import display


CUDA = True if torch.cuda.is_available() else False

BATCH_SIZE = 16
IMG_SIZE = 28
LATENT_DIM = 64
CHANNELS = 1 


class Generator(nn.Module):
    def __init__(self, latent_dim = LATENT_DIM, img_size = IMG_SIZE, channels = CHANNELS):
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
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels*img_size*img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img
    
    

class Discriminator(nn.Module):
    def __init__(self, img_size = IMG_SIZE, channels = CHANNELS):
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


    
    
    
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

class Trainer:
    """ Class that encapsulates a generator and a discriminator, with the addition of a training method.
    """    
    def __init__(self, eps = 200, latent_dim = LATENT_DIM, img_size = IMG_SIZE, channels = CHANNELS, batch_size = BATCH_SIZE):
        """ Constructor.

        Args:
            eps (int, optional): Number of epochs to run in the training phase. Defaults to 200.
            rd_state (int, optional): Seed for consistent randomness of the dataset to train on. Defaults to 42.
            ds_size (int, optional): Size of the dataset. Defaults to 100.
        """        
        os.makedirs("data/mnist", exist_ok=True)
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.generator = Generator(self.latent_dim, self.img_size, self.channels)
        self.discriminator = Discriminator(self.img_size, self.channels)
        
        self.opt_G =  torch.optim.Adam(self.generator.parameters())
        self.opt_D = torch.optim.Adam(self.discriminator.parameters())
        
        self.adv_loss = torch.nn.BCELoss()
        self.epochs = eps
        self.vis = visualizer.Visualizer(eps)
        
        self.dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)
        
        


    def cuda(self):
        """Transfers parameters of generator, discriminator and loss to GPU memory.
        """        
        if CUDA:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adv_loss.cuda()


    def train(self):
        """Performs training of the generator/discriminator couple. 

        [extended_summary]
        """    

        g_loss_evolution = []
        magnitude_evolution = []    
        for epoch in range(self.epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                display.clear_output(wait=True)
                
                #Ground truths, no gradient necessary 
                gt_valid, gt_fake =  torch.ones(size=(imgs.size(0), 1)).cuda(), torch.zeros(size=(imgs.size(0), 1)).cuda()
               
                
                real_imgs = imgs.type(Tensor)

                #-----------------
                # Train Generator
                #-----------------

                self.opt_G.zero_grad()
                # Generate input in latent dim for generator ( i.e sample noise ...)
                gen_input  = torch.normal(mean = 0, std = 1, size=(imgs.shape[0],self.latent_dim)).cuda()
               
                gen_output = self.generator(gen_input)
                
                decision = self.discriminator(gen_output)

                g_loss = self.adv_loss(decision,gt_valid)
                
                g_loss.backward()

                self.opt_G.step()
                g_loss_evolution.append(g_loss.item())

                #---------------------
                # Train Discriminator
                #---------------------
                
                self.opt_D.zero_grad()

                gen_out_detached = gen_output.detach()
                print(gen_output.shape)
                grad_mag = torch.norm(gen_out_detached)
                magnitude_evolution.append(grad_mag)

                decision_nograd_g = self.discriminator(gen_out_detached)

                real_loss = self.adv_loss(self.discriminator(real_imgs),gt_valid)

                fake_loss = self.adv_loss(decision_nograd_g, gt_fake)
                
                d_loss = (real_loss+fake_loss)/2
                d_loss.backward()

                self.opt_D.step()
                
                print(" [Batch %d/%d] "% ( i, len(self.dataloader)))


                display.clear_output(wait=True)
                self.vis.update(g_loss_evolution,magnitude_evolution, gen_out_detached.cpu().clone().numpy())
                self.vis.display(epoch)
            # self.vis.video()



