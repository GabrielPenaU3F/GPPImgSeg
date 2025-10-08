import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from segmentation.methods.ml_labeling import ml_labeling
from segmentation.methods.nmc_labeling import nmc
from segmentation.neighborhood import Neighborhood
from segmentation.utilities import get_neighbor_stack, label_image_from_probabilities, initialize_urns, \
    sample_class_from_probs


def update_urns(urns, sampled_classes, delta):
    h, w, k = urns.shape
    urns_flat = urns.reshape(-1, k)
    classes_flat = sampled_classes.ravel()
    np.add.at(urns_flat, (np.arange(h * w), classes_flat), delta)
    return urns_flat.reshape(h, w, k)

def polya_labeling(probs, neighborhood, n_balls, delta, n_iter=10, return_type='img',
                   watch_evolution=False, save_directory=None):

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
            if save_directory is None:
                raise Exception('Save directory not specified')
            probs = urns / urns.sum(axis=2, keepdims=True)
            save_frame(probs, n+1, save_directory)

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
    neighborhood = Neighborhood('radius', radius=10)
    # neighborhood = Neighborhood('8')
    Y = polya_labeling(ml_probs, neighborhood, n_balls=100, delta=10, n_iter=20, return_type='img',
                       watch_evolution=True, save_directory='../../outputs/polya_test/')
    Y = Image.fromarray(Y)
    Y.show()