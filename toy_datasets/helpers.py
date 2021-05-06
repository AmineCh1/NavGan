import numpy as np
from sklearn.datasets import make_moons
import scipy.linalg as la
import torch

CUDA = True if torch.cuda.is_available() else False

GEN_PATH_IMG = "../models/gen_img.pth"
GEN_PATH_DS = "../models/gen_ds.pth"


def to_numpy(x):

    return x.detach().cpu().clone().numpy()


def generate_cross(size=100, width=1, seed=44):

    np.random.seed(seed)
    cross = np.array([np.random.permutation(
        [0, np.random.uniform(-width, width)]) for i in np.arange(size)])
    return cross


def generate_moons(size=100, width_x=1, width_y=1):

    np.random.seed(42)
    moons = make_moons(n_samples=size, noise=0)[0]
    moons[:, 0] = width_x * moons[:, 0]
    moons[:, 1] = width_y * moons[:, 1]
    return moons


def extract_directions(generator, nb_dir=4):

    params = list(generator.parameters())

    weight_mat = to_numpy(params[0])
    gram = weight_mat.T@weight_mat
    eigen_vals, eigen_vecs = la.eig(gram)

    return eigen_vecs[:, np.argsort(eigen_vals)][0:nb_dir]


def load_ds_calmcode():
    dataset = np.genfromtxt("data.csv", skip_header=1,
                            delimiter=',', usecols=(0, 1))
    return dataset
