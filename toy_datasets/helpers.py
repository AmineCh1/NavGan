import numpy as np
from sklearn.datasets import make_moons
import scipy.linalg as la 

def to_numpy(x):
    
    return x.detach().cpu().clone().numpy()

def generate_cross(size = 100):
    
    np.random.seed(42)
    cross  = np.array([np.random.permutation([0,np.random.uniform(-1,1)]) for i in np.arange(size)])
    return cross 
    
def generate_moons (size = 100):
    
    np.random.seed(42)
    return make_moons(n_samples=size, noise=0)[0]

def extract_directions(generator, nb_dir = 4 ):
   
    params = list(generator.parameters())
   
    weight_mat = to_numpy(params[0])
    gram = weight_mat.T@weight_mat
    eigen_vals, eigen_vecs = la.eig(gram)

    return eigen_vecs[:,np.argsort(eigen_vals)][0:nb_dir]



