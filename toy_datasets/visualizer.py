import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Colormap
import numpy as np 
import subprocess
class Visualizer:
    def __init__(self, eps = 200, g_loss=[],magnitude= [],g_output = [],fake_loss = []):
        self.g_loss = g_loss
        self.mag = magnitude
        self.g_output = g_output
        self.eps = eps
        self.fake_loss = fake_loss
        

    def update(self, g_loss,magnitude,g_output,fake_loss):
        self.g_loss = g_loss
        self.magnitude = magnitude
        self.g_output = g_output
        self.fake_loss = fake_loss

    def display(self, i):
        

        fig = plt.figure(figsize = (18,15), tight_layout = True)
        fig.suptitle("Training evolution", fontsize=14)
        gs = gridspec.GridSpec(2, 2)
        
        ax_temp = plt.subplot(gs[:,1])
        cmap_fake = Colormap(self.fake_loss)
        ax_temp.scatter(self.g_output[:,0],self.g_output[:,1],cmap = cmap_fake)
        ax_temp.set_title("Generated dataset")
        [s.set_visible(False) for s in ax_temp.spines.values()]

        ax_temp = plt.subplot(gs[0,0])
        ax_temp.plot(self.g_loss)
        ax_temp.set_xticks(np.arange(self.eps+1,step=int(self.eps/10)))
        ax_temp.set_xlabel("Epochs")
        ax_temp.set_ylabel("Generator loss")

        ax_temp = plt.subplot(gs[1,0])
        ax_temp.plot(self.magnitude)
        ax_temp.set_xticks(np.arange(self.eps+1,step=int(self.eps/10)))
        ax_temp.set_xlabel("Epochs")
        ax_temp.set_ylabel("Gradient magnitude")
        
        
        plt.savefig("frames/fig_{}.png".format(i))
        plt.show()
        
    def video(self):
        print("Preparing video ...")
        subprocess.call("ffmpeg -r 5 -i frames/fig_%d.png -vcodec libx264 -y movie.mp4",shell=True)
        print("Done !")

        
        
