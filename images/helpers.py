import numpy as np
from sklearn.datasets import make_moons
import scipy.linalg as la
import torch

CUDA = True if torch.cuda.is_available() else False
GEN_PATH_IMG = "../models/gen_img.pth"
GEN_PATH_DS = "../models/gen_ds.pth"
GEN_PATH_MNIST = "../models/G--300.ckpt"
GEN_PATH_FMNIST = "../models/fashionmnist/gmodel-ckpt.pth"
GEN_PATH_BIGAN = '../models/trainedBiGAN_G.pth'
DECODER_PATH_VAE = '../models/trainedVAE_D.pth'


# ----------------------------HYPERPARAMETERS

DEVICE = "cpu"
# "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_IMG = 64
IMG_SIZE = 28
LATENT_DIM_IMG = 2
CHANNELS = 1

LATENT_DIM_BIGAN = 2

LATENT_DIM_MNIST = 64
HIDDEN_DIM_MNIST = 256
OUTPUT_DIM_MNIST = 784

LATENT_DIM_FMNIST = 100
HIDDEN_DIM_FMNIST = 32
OUTPUT_DIM_FMNIST = 784
NB_SAMPLES = 2

# ----------------------------HELPER FUNCTIONS


def reshape_output(x):
    return np.reshape(x, (28, 28))


def to_numpy(x):

    return x.detach().cpu().clone().numpy()


def generate_cross(size=100, width=1):

    np.random.seed(42)
    cross = np.array([np.random.permutation(
        [0, np.random.uniform(-width, width)]) for i in np.arange(size)])
    return cross


def generate_moons(size=100, width_x=1, width_y=1):

    np.random.seed(42)
    moons = make_moons(n_samples=size, noise=0)[0]
    moons[:, 0] = width_x * moons[:, 0]
    moons[:, 1] = width_y * moons[:, 1]
    return moons


def extract_directions(generator, nb_dir=3):

    params = list(generator.parameters())

    weight_mat = to_numpy(params[0])
    gram = weight_mat.T@weight_mat
    eigen_vals, eigen_vecs = la.eigh(gram)
    vals = np.flip(np.sort(eigen_vals))
    vectors = np.flip(eigen_vecs[:, np.argsort(eigen_vals)][-nb_dir:])

    return vals, vectors
