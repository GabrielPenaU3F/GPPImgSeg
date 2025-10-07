from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from segmentation.methods.ml_labeling import ml_labeling
from segmentation.methods.nmc_labeling import nmc
from segmentation.neighborhood import Neighborhood
from segmentation.utilities import get_neighbor_stack, label_image_from_probabilities


def initialize_urns(probs, n_balls=100):
    h, w, K = probs.shape
    flat = probs.reshape(-1, K)
    target = flat * n_balls
    floor = np.floor(target).astype(int)  # sum <= n_balls. This underestimates
    rem = (target - floor) # remainder
    deficit = n_balls - floor.sum(axis=1)  # shape (N,) - how many balls we have yet to assign

    # We assign remaining balls, starting by classes with larger remainders
    urns = floor.copy()
    if flat.shape[0] == 1:
        order = np.argsort(-rem[0])
        urns[0, order[:deficit[0]]] += 1
    else:
        for i, d in enumerate(deficit):
            if d <= 0:
                continue
            order = np.argsort(-rem[i])
            urns[i, order[:d]] += 1

    return urns.reshape(h, w, K)

def sample_class_from_probs(probs):
    h, w, k = probs.shape
    rng = np.random.default_rng()
    cdf = np.cumsum(probs, axis=2)  # (h, w, k)
    r = rng.random((h, w, 1))  # (h, w, 1)
    samples = (cdf > r).argmax(axis=2)
    return samples

def update_urns(urns, sampled_classes, delta):
    h, w, k = urns.shape
    urns_flat = urns.reshape(-1, k)
    classes_flat = sampled_classes.ravel()
    np.add.at(urns_flat, (np.arange(h * w), classes_flat), delta)
    return urns_flat.reshape(h, w, k)

def save_frame(urns, n_frame):
    probs = urns / urns.sum(axis=2, keepdims=True)
    img = label_image_from_probabilities(probs)
    frame = Image.fromarray(img)
    frame.save(f'../outputs/polya_test/frame_{n_frame:03d}.png')

def polya_labeling(probs, neighborhood, n_balls, delta, n_iter=10, return_type='img', watch_evolution=False):

    urns = initialize_urns(probs, n_balls)

    for n in range(n_iter):

        '''
            Flow of the algorithm is the same as the relaxation labeling implementation
        '''
        neighbor_urn_stack = get_neighbor_stack(urns, neighborhood)
        super_urns = neighbor_urn_stack.sum(axis=2)  # (h, w, k)
        super_urn_probs = super_urns / super_urns.sum(axis=2, keepdims=True)
        sampled_classes = sample_class_from_probs(super_urn_probs)
        urns = update_urns(urns, sampled_classes, delta)

        if watch_evolution:
            save_frame(urns, n+1)

    probs = urns / urns.sum(axis=2, keepdims=True)
    if return_type == 'img':
        return label_image_from_probabilities(probs)

    elif return_type == 'probs':
        return probs

    elif return_type == 'urns':
        return urns


if __name__ == '__main__':
    image_path = '../../resources/test_img.bmp'
    img = Image.open(image_path)
    X = np.array(img)
    init_labels = nmc(X, n_iter=10, n_classes=3, return_type='raw')
    ml_probs = ml_labeling(X, init_labels, return_type='probs')
    # neighborhood = Neighborhood('radius', radius=4)
    neighborhood = Neighborhood('8')
    Y = polya_labeling(ml_probs, neighborhood, n_balls=100, delta=10, n_iter=10, return_type='img', watch_evolution=False)
    Y = Image.fromarray(Y)
    Y.show()