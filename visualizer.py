import matplotlib.pyplot as plt
from matplotlib import gridspec

class Visualizer:
    def __init__(self, eps = 200, g_loss=[],magnitude= [],g_output = []):
        self.g_loss = g_loss
        self.mag = magnitude
        self.g_output = g_output
        self.eps = eps
        

    def update(self, g_loss,magnitude,g_output):
        self.g_loss = g_loss
        self.magnitude = magnitude
        self.g_output = g_output

    def display(self):
        ## do plotting with gridspec ... 

        fig = plt.figure(figsize = (15,13), tight_layout = True)
        gs = gridspec.GridSpec(2,2)
        
        ax_temp = plt.subplot(gs[1,:])
        ax_temp.scatter(self.g_output[:,0],self.g_output[:,1])

        ax_temp = plt.subplot(gs[0,0])
        ax_temp.plot(self.g_loss)

        ax_temp = plt.subplot(gs[0,1])
        ax_temp.plot(self.magnitude)
        
        

        
        
