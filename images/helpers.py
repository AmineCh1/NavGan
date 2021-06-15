import numpy as np
from sklearn.datasets import make_moons
import scipy.linalg as la
import torch

CUDA = True if torch.cuda.is_available() else False
# Paths for saving
GEN_PATH_IMG = "../models/gen_img.pth"
GEN_PATH_DS = "../models/gen_ds.pth"

# Paths for loading pretrained models
GEN_PATH_IMG_1 = "../models/gen_img_1.pth"  # Custom model wth 2d input (2)
GEN_PATH_MNIST = "../models/G--300.ckpt"  # Pretrained MNIST (64)
# Pretrained FMNIST (64)
GEN_PATH_FMNIST = "../models/fashionmnist/gmodel-ckpt.pth"
GEN_PATH_BIGAN = '../models/trainedBiGAN_G.pth'  # Trained G (64)

# VAEs
# Trained VAE D (2D -> Image)
DECODER_PATH_VAE = '../models/trainedVAE_D_2.pth'
ENCODER_PATH_VAE = '../models/trainedVAE_E_2.pth'  # TRAINED VAE ( Image -> 2D)

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
    """reshape_output 
    Turns flattened (784,) vector into (28, 28) image.

    Args:
        x (array-like): Flattened (784,) vector.

    Returns:
        img: (28,28) image.
    """
    return np.reshape(x, (28, 28))


def to_numpy(x):
    """to_numpy 
    Helper function that turns pytorch tensors into numpy arrays.

    Args:
        x (torch.Tensor) : PyTorch Tensor to convert

    Returns:
        [np.array]: Converted tensor.
    """
    if x.device == "cuda:0":
        return x.detach().cpu().clone().numpy()
    else:
        return x.detach().clone().numpy()


def generate_cross(size=100, width=1, seed=42):
    """generate_cross [summary]

    Generates cross dataset of given size and range.

    Args:
        size (int, optional): Number of datapoints in the cross. Defaults to 100.
        width (int, optional): range of the [x,y] values taken by the cross. Defaults to 1.
        seed(int, optional): Sets seed for reproducibility. Defaults to 42.

    Returns:
        Returns a cross dataset with [size] points in the range [(-width, width), (-width, width)]
    """
    np.random.seed(seed)
    cross = np.array([np.random.permutation(
        [0, np.random.uniform(-width, width)]) for i in np.arange(size)])
    return cross


def generate_moons(size=100, width_x=1, width_y=1, seed=42):
    """generate_moons [summary]
        Generates two moons  dataset of given size and range.

    Args:
        size (int, optional): Number of datapoints in the two moons. Defaults to 100.
        width_x (int, optional): [description]. Defaults to 1.
        width_y (int, optional): [description]. Defaults to 1.
        seed (int, optional):Sets seed for reproducibility. Defaults to 42.


    Returns:
        Returns a two moons dataset with [size] points in the range [(-width_x, width_x), (-width_y, width_y)]
    """
    np.random.seed(seed)
    moons = make_moons(n_samples=size, noise=0)[0]
    moons[:, 0] = width_x * moons[:, 0]
    moons[:, 1] = width_y * moons[:, 1]
    return moons


def extract_directions(generator, nb_dir=None, layer_level=0):
    """extract_directions 
    Extracts directions from given layer of generative model. 

    Args:
        generator (generator): Generative model to extract directions from.
        nb_dir (int, optional): If not None, extract given number of directions from generative model., else, return all directions. Defaults to None.
        layer_level (int, optional): Layer of generative model to extract directions from.  Defaults to 0, i.e first layer. 

    Returns:
        Sorted eigenvalues, and eigenvectors sorted with respect to eigenvalues.
    """

    params = list(generator.parameters())
    weight_mat = to_numpy(params[layer_level])
    print(weight_mat.shape)
    gram = weight_mat.T@weight_mat
    eigen_vals, eigen_vecs = la.eigh(gram)
    vals = np.flip(np.sort(eigen_vals))
    if nb_dir is not None:
        vectors = eigen_vecs[:, np.flip(np.argsort(eigen_vals))][-nb_dir:]
    else:
        vectors = eigen_vecs[:, np.flip(np.argsort(eigen_vals))]

    return vals, vectors
