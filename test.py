import os
import argparse
import numpy as np
from scipy import misc
import scipy.io as sio
from warpgan import WarpGAN
import scipy.linalg as la

# STYLE_COLLECTION_INDEX = 35
# WARP_COLLECTION_INDEX = 59
# WARP_DIM = 1048576
# kappa = 0.05

# Parse aguements
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="The path to the pretrained model",
                    type=str)
parser.add_argument("input", help="The path to the aligned image",
                    type=str)
parser.add_argument("output", help="The prefix path to the output file, subfix will be added for different styles.",
                    type=str, default=None)
parser.add_argument("--num_styles", help="The number of images to generate with different styles",
                    type=int, default=5)
parser.add_argument("--scale", help="The path to the input directory",
                    type=float, default=1.0)
parser.add_argument("--aligned", help="Set true if the input face is already normalized",
                    action='store_true')
parser.add_argument("--nbdir", help="Number of principal directions to vary on",
                    type=int, default=2)

args = parser.parse_args()


if __name__ == '__main__':

    network = WarpGAN()
    # Direction Extraction -- Warp
    # perturbation = np.load("perturbation.npy")

    network.load_model(
        args.model_dir)
    img = misc.imread(args.input, mode='RGB')

    if not args.aligned:
        from align.detect_align import detect_align
        img = detect_align(img)

    img = (img - 127.5) / 128.0

    images = np.tile(img[None], [args.num_styles, 1, 1, 1])
    scales = args.scale * np.ones((args.num_styles))

    a = np.random.randint(0, 10000)
    # print(a)
    np.random.seed(4937)
    styles = np.random.normal(
        0., 1., (args.num_styles, network.input_style.shape[1].value))
    alphas = np.linspace(0, 1.5, 10)

    # Direction Extraction - - Style
    vecs_style = np.load("vecs_style.npy")[:args.nbdir]

    # output = network.generate_BA(
    #     images, scales, 16, styles=styles)
    # output = 0.5*output + 0.5

    # for i in range(args.num_styles):
    #     misc.imsave('result/'+args.output +
    #                 '_style={}pert.jpg'.format(i+1), output_perturbation[i])
    #     misc.imsave('result/'+args.output +
    #                 '_style={}nopert.jpg'.format(i+1), output_no_perturbation[i])

    output_eigs = []
    for alpha in alphas:
        output_alpha = []
        for direction in range(args.nbdir):
            images = np.tile(img[None], [args.num_styles, 1, 1, 1])
            output_ = network.generate_BA(
                images, scales, 16, styles=styles + alpha*vecs_style[direction])
            output_alpha.append(0.5*output_+0.5)
        output_eigs.append(output_alpha)

    for style in range(args.num_styles):
        for direction in range(args.nbdir):
            os.makedirs(
                'result/style{}/direction{}'.format(style+1, direction+1))

    # for style in range(args.num_styles):
    #     os.makedirs('result/style{}/combinations/'.format(style+1))
    #     output_ = network.generate_BA(
    #         images, scales, 16, styles=styles + 0.5*vecs_style[0] + 0.25*vecs_style[1])
    #     misc.imsave(
    #         'result/style{}/combinations/'.format(style+1) + args.output + '_alpha={}.jpg'.format(alpha+1), output_[style])

    for alpha, output_alpha in zip(range(len(alphas)), output_eigs):
        for direction, output in enumerate(output_alpha):
            for style in range(args.num_styles):
                misc.imsave(
                    'result/style{}/direction{}/'.format(style+1, direction+1) + args.output + '_alpha={}.jpg'.format(alpha+1), output[style])
