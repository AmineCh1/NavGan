import numpy as np
import torch.nn as nn
import torch
import visualizer
from IPython import display
import matplotlib.pyplot as plt
from helpers import *


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
            *block(2, 16, normalize = False),
            *block(16,32, normalize = False), 
            *block(32,16, normalize = False), 
            *block(16,8,normalize = False),
            *block(8,2,normalize=False), 
            nn.Tanh()
            # *block(2, 2, normalize = False),
            # nn.Tanh()
        )

    def forward(self, z):
        points = self.model(z)
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
            *block(2, 16, normalize = False),
            *block(16,32, normalize = False), 
            *block(32,16, normalize = False), 
            *block(16,8,normalize = False),
            *block(8,2,normalize=False),
            *block(2,1,normalize =False) 
        )

    def forward(self, dp):
        validity = self.model(dp)
        return validity


class Trainer:
    """ Class that encapsulates a generator and a discriminator, with the addition of a training method.
    """    
    def __init__(self, eps = 200,ds_size = 20000,batch_size = 100):
        """ Constructor.

        Args:
            eps (int, optional): Number of epochs to run in the training phase. Defaults to 200.
            
            ds_size (int, optional): Size of the dataset. Defaults to 100.
        """        
        #NNs 
        self.generator = Generator()
        self.discriminator = Discriminator()

        #Optimizers
        self.opt_G =  torch.optim.Adam(self.generator.parameters(),lr=1e-3) 
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr= 1e-3)

        #Losses
        self.adv_loss = torch.nn.BCEWithLogitsLoss()
        self.adv_loss_map = torch.nn.BCEWithLogitsLoss(reduction='none')

        #Misc and hyperparams
        self.epochs = eps
        self.vis = visualizer.Visualizer(eps)
        self.ds_size = ds_size
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(generate_moons(self.ds_size), batch_size = batch_size, shuffle=True)

        
    
    
    def cuda(self):
        """Transfers parameters of generator, discriminator and loss to GPU memory.
        """        
        if CUDA:
            print(CUDA)
            self.generator.to("cuda")
            self.discriminator.to("cuda")
            self.adv_loss.to("cuda")
            # self.dataset = self.dataset.to("cuda")

    
        
    def train(self, fix_latent_seed = False, seed_size = 100):
        """Performs training of the generator/discriminator couple. 

        [extended_summary]
        """    

        # g_loss_evolution = []
        # magnitude_evolution = []
        # gen_otpt = np.empty_like((1,2))
        g_loss_evolution = []
        magnitude_evolution = []  
        np.random.seed(44)
        gen_input_fixed =  torch.normal(mean = 0, std = 1, size=(seed_size,2)).to("cuda")
        for epoch in range(self.epochs):
            g_loss_avg = []
            magnitude_avg = []
            display.clear_output(wait=True)
            for batch_nb, pts in enumerate(self.dataloader):
                    ##for batch in dataloader: 
                    # gen input to "compare with" batch: 
                    #then perofrm the same steps as image training ...

                real_datapoints = torch.tensor(pts).float().to("cuda")
                
                
                # Avoid non-compatible batch_sizes
                batch_size_tmp = pts.shape[0]
           

                #Ground truths, no gradient necessary 
                gt_valid, gt_fake =  torch.ones(size=(batch_size_tmp,1)).to("cuda"), torch.zeros(size=(batch_size_tmp,1)).to("cuda")
             
                

                #-----------------
                # Train Generator
                #-----------------

                self.opt_G.zero_grad()
                # Generate input in latent dim for generator ( i.e sample noise ...)

                gen_input  = torch.normal(mean = 0, std = 1, size=(batch_size_tmp,2)).to("cuda")
                
                gen_output = self.generator(gen_input)
                
                decision = self.discriminator(gen_output)

                g_loss = self.adv_loss(decision,gt_valid)
                
                g_loss.backward()

                self.opt_G.step()
                g_loss_avg.append(g_loss.item())

                #---------------------
                # Train Discriminator
                #---------------------
                
                self.opt_D.zero_grad()
                
                
                gen_out_detached = gen_output.detach().cuda()
                gen_out_detached.requires_grad=True
                

                decision_nograd_g = self.discriminator(gen_out_detached)
                real_loss = self.adv_loss(self.discriminator(real_datapoints),gt_valid)

                
                fake_loss = self.adv_loss(decision_nograd_g, gt_fake)
                fake_loss_map = self.adv_loss_map(decision_nograd_g,gt_fake)

                d_loss = (real_loss+fake_loss)/2
                d_loss.backward()

                grad_mag = torch.mean(torch.norm(gen_out_detached.grad,dim=1))
                
                magnitude_avg.append(grad_mag.item())

                self.opt_D.step()

                
                
                # print( gen_otpt.shape)
                # gen_otpt = np.concatenate((gen_otpt, gen_out_detached.cpu().clone().numpy()), axis = 1)
                fake_loss_detached = to_numpy(fake_loss_map)
                
           
            magnitude_evolution.append(np.mean(magnitude_avg))    
            g_loss_evolution.append(np.mean(g_loss_avg))

            if (fix_latent_seed):

                gen_out_detached = self.generator(gen_input_fixed)

                decision_nograd_g = self.discriminator(gen_out_detached)

                gt_fake = torch.zeros(size=(seed_size,1)).to("cuda")
    
                fake_loss_detached = to_numpy(self.adv_loss_map(decision_nograd_g,gt_fake))

            self.vis.update(g_loss_evolution,magnitude_evolution,
            to_numpy(gen_out_detached),fake_loss_detached)
                
            self.vis.display(epoch)

        # Test with large input
        final_in = torch.randn(self.ds_size, 2, device='cuda:0')
        final_out = self.generator(final_in)
        self.vis.disp_final(to_numpy(final_out))
        self.vis.video()

        # Print extracted eigenvectors
        vecs = extract_directions(self.generator)
        for i, vector in enumerate(vecs):
            print("The %dth most important direction is:" % (i+1))
            print(vector)

        
        



