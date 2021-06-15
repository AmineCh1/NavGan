import os
import argparse
import numpy as np
from scipy import misc
import scipy.io as sio
from warpgan import WarpGAN
import scipy.linalg as la

STYLE_COLLECTION_INDEX = 35
WARP_COLLECTION_INDEX = 59
WARP_DIM = 1048576


parser = argparse.ArgumentParser()

parser.add_argument("--dir_warp", help="Number of principal directions to vary on warping",
                    type=int, default=1)

args = parser.parse_args()


if __name__ == '__main__':

    path_to_model = "pretrained/warpgan_pretrained"
    network = WarpGAN()
    network.load_model(path_to_model)

    # Find eigenvalues of (output,output) matrix  instead of (input, input) matrix: They're the same. From there, find left eigenvectors (i.e eigen vectors of (input,input) based on eigenvectors and diagonalization trick)
    warp_layer = network.sess.run(
        network.graph.get_collection('model_variables')[WARP_COLLECTION_INDEX])
    vals_warp, vecs_warp = la.eigh(warp_layer.T@warp_layer)
    inverse_vals = [1/x for x in vals_warp]
    left_vectors = (warp_layer@vecs_warp)@np.diag(inverse_vals)

    perturbation = left_vectors[:, np.arange(args.dir_warp)]

    np.save("perturbation.npy", perturbation)
