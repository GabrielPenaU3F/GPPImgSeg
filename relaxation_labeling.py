import numpy as np

from PIL import Image

from ml_labeling import ml_segmentation
from nmc_labeling import nmc
from utilities import format_image, label_image_from_probabilities

shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def rl_segmentation(probs, n_iter=10, return_type='img'):

    h, w, k = probs.shape # height, width and number of classes
    compatibilities = compatibility_matrix(k)

    for _ in range(n_iter):
        '''
        1) Build the stack of neighbor probabilities: this is a (h, w, 8, k) tensor
           It contains 8 layers of the probs matrix, each corresponding to the 'probs' of one
           of the neighbors
        '''
        neighbor_stack = get_neighbor_stack(probs) # (h, w, 8, k)

        '''2) sum over 8-neighbors -> vecs is a (h, w, k) tensor
        For each pixel, this contains the sum of each class probabilities (over its neighbors)
        '''
        vecs = neighbor_stack.sum(axis=2) # (h, w, k)

        '''
        3) For each pixel, we calculate S (the support) in a vectorized fashion:
            S[i,j,l] = sum_lp R[l,lp] * vecs[i,j,lp]
            Thus S is a (h, w, k) tensor.
            
            This takes into account, linearly, the compatibilities and the probabilities of
            belonging to the respective class.
            
            Equivalent to np.einsum('ijk,lk->ijl', vecs, compatibilities)
        '''
        S = vecs @ compatibilities.T # (h, w, k)

        ''' 4) Update probabilities and normalize'''
        probs_new = probs * (1.0 + S)

        probs_new = np.clip(probs_new, 1e-8, None)
        probs_new /= probs_new.sum(axis=2, keepdims=True)
        probs = probs_new

    if return_type == 'img':
        return label_image_from_probabilities(probs)

    elif return_type == 'probs':
        return probs

# Simple discrete metric-type compatibility
def compatibility_matrix(n_classes):
    R = -np.ones((n_classes, n_classes), dtype=np.float32)
    np.fill_diagonal(R, 1.0)
    return R

'''
This extracts the 8-neighbor probabilities. Each row in the stack corresponds to a neighbor
in the order defined by shifts
'''
def get_neighbor_stack(probs):
    # probs: (h, w, n_classes)
    h, w, k = probs.shape
    padded = np.pad(probs, ((1,1),(1,1),(0,0)), mode='reflect')  # (h+2, w+2, k)
    neighbor_list = [padded[1+dy:1+dy+h, 1+dx:1+dx+w, :] for dy,dx in shifts]
    # stack -> (h, w, 8, k)
    neighbor_stack = np.stack(neighbor_list, axis=2)
    return neighbor_stack

if __name__ == "__main__":
    image_path = 'resources/test_img.bmp'
    img = Image.open(image_path)
    X = np.array(img)
    init_labels = nmc(X, n_iter=10, n_classes=3, return_type='raw')
    init_probabilities = ml_segmentation(X, init_labels, return_type='probs')
    Y = rl_segmentation(init_probabilities, n_iter=10, return_type='img')
    Y = Image.fromarray(Y)
    Y.show()
