import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.datasets import make_moons
from torch.autograd import Variable
import visualizer
from IPython import display


CUDA = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        ## * is used to unpack the layers iterable into arguments for nn.Sequential
        self.model = nn.Sequential(
            *block(2, 2, normalize=True),
            nn.Tanh()
        )

    def forward(self, z):
        points = self.model(z)
        return points


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, dp):
        validity = self.model(dp)
        return validity


Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

class Trainer:
    """ Class that encapsulates a generator and a discriminator, with the addition of a training method.
    """    
    def __init__(self, eps = 200, rd_state=42,ds_size=100):
        """ Constructor.

        Args:
            eps (int, optional): Number of epochs to run in the training phase. Defaults to 200.
            rd_state (int, optional): Seed for consistent randomness of the dataset to train on. Defaults to 42.
            ds_size (int, optional): Size of the dataset. Defaults to 100.
        """        

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.opt_G =  torch.optim.Adam(self.generator.parameters())
        self.opt_D = torch.optim.Adam(self.discriminator.parameters())
        self.adv_loss = torch.nn.BCELoss()
        self.epochs = eps
        self.ds_size = ds_size
        self.rd_state = rd_state
        self.vis = visualizer.Visualizer()
        


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
            
            display.clear_output(wait=True)
            
            toy_dataset = torch.tensor(make_moons(n_samples=self.ds_size, noise=0.2)[0]).float()
            toy_dataset = toy_dataset #.to("cuda")

            gt_valid, gt_fake =  Variable(Tensor(self.ds_size, 1).fill_(1.0), requires_grad=False), Variable(Tensor(self.ds_size, 1).fill_(0.0), requires_grad=False)

            #-----------------
            # Train Generator
            #-----------------

            self.opt_G.zero_grad()
            # Generate input in latent dim for generator ( i.e sample noise ...)
            gen_input  = Variable(Tensor(np.random.normal(0, 1, (100,2))))
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
            
            decision_nograd_g = self.discriminator(gen_output.detach())

            real_loss = self.adv_loss(self.discriminator(toy_dataset),gt_valid)
            fake_loss = self.adv_loss(decision_nograd_g, gt_fake)
            
            d_loss = (real_loss+fake_loss)/2
            d_loss.backward()

            self.opt_D.step()

            display.clear_output(wait=True)
            self.vis.update(g_loss_evolution,[], gen_output.detach().cpu().clone().numpy())
            self.vis.display()



