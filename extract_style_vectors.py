import numpy as np
from scipy import misc
import scipy.io as sio
from warpgan import WarpGAN
import scipy.linalg as la
import matplotlib.pyplot as plt
STYLE_COLLECTION_INDEX = 35

network = WarpGAN()
network.load_model('pretrained/warpgan_pretrained')


def gs(X, row_vecs=True, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag(
            (X[i, :].dot(Y.T)/np.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


# Saving Style
style_layer = network.sess.run(
    network.graph.get_collection('model_variables')[STYLE_COLLECTION_INDEX])
gram_style = style_layer@style_layer.T
print(style_layer.shape)
vals_style, vecs_style = la.eigh(gram_style)
vals_style_sorted = np.flip(np.sort(vals_style))
vals_style_idx = np.flip(np.argsort(vals_style))
vecs_style_sorted = vecs_style[:, vals_style_idx]
np.save("vecs_style.npy", vecs_style_sorted)

vecs_style_orthogonalized = gs(vecs_style_sorted, row_vecs=False)
np.save("vecs_style_orthogonalized.npy", vecs_style_orthogonalized)

plt.scatter(range(1, len(vals_style)+1), vals_style_sorted)
plt.title("Eigenvalues of S_in.T S_in")
plt.show()
print("Done")
