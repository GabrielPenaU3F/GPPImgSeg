from abc import ABC

from matplotlib import pyplot as plt
import numpy as np

from utilities.image_format_utilities import label_image_from_probabilities
from utilities.output_utilities import save_frame
from utilities.segmentation_utilities import initialize_urns, get_neighbor_stack, sample_class_from_probs


class UrnLabeler(ABC):

    def label(self, probs, neighborhood, initial_total_balls, R, n_iter=10,
              return_type='img', watch_evolution=False, save_directory=None):

        urns = initialize_urns(probs, initial_total_balls)

        for n in range(n_iter):
            print(f'Iteration Nº{n+1}')

            '''
                Flow of the algorithm is the same as the relaxation labeling implementation
            '''
            neighbor_urn_stack = get_neighbor_stack(urns, neighborhood)
            super_urns = neighbor_urn_stack.sum(axis=2)  # (h, w, k)
            super_urn_probs = super_urns / super_urns.sum(axis=2, keepdims=True)
            sampled_classes = sample_class_from_probs(super_urn_probs)
            urns = self.update_urns(urns, sampled_classes, R)

            if watch_evolution:
                if save_directory is None:
                    raise Exception('Save directory not specified')
                probs = urns / urns.sum(axis=2, keepdims=True)
                img = label_image_from_probabilities(probs)
                save_frame(img, n + 1, save_directory)

        probs = urns / urns.sum(axis=2, keepdims=True)
        if return_type == 'img':
            return label_image_from_probabilities(probs)

        elif return_type == 'probs':
            return probs

        elif return_type == 'urns':
            return urns

    def update_urns(self, urns, sampled_classes, delta):
        pass


class PolyaLabeler(UrnLabeler):

    # In Polya method, delta is the number of balls to add
    def update_urns(self, urns, sampled_classes, delta):
        h, w, k = urns.shape
        urns_flat = urns.reshape(-1, k)
        classes_flat = sampled_classes.ravel()
        np.add.at(urns_flat, (np.arange(h * w), classes_flat), delta)
        return urns_flat.reshape(h, w, k)


class GPPLabeler(UrnLabeler):

    def update_urns(self, urns, sampled_classes, R):
        """
        Update urns given the sampled classes and a reinforcement matrix R.

        Parameters
        ----------
        urns : np.ndarray
            Shape (H, W, K), current number of balls per class.
        sampled_classes : np.ndarray
            Shape (H, W), sampled class index for each pixel.
        R : np.ndarray
            Shape (K, K). Reinforcement matrix: R[i, j] = number of balls of class j
            added when the sampled class is i.

        Returns
        -------
        updated_urns : np.ndarray
            Updated urns with same shape as input.
        """

        self.validate_reinforcement_matrix(R)
        R = R.astype(np.int64)
        h, w, k = urns.shape
        urns_flat = urns.reshape(-1, k)
        classes_flat = sampled_classes.ravel()

        # --- Apply reinforcement ---
        # For each pixel i, add R[class_i] to urns_flat[i]
        balls_to_add = R[classes_flat]
        urns_flat += balls_to_add
        # --- Prevent negative counts, this could happen with negative reinforcement ---
        urns_flat = np.maximum(urns_flat, 0)

        return urns_flat.reshape(h, w, k)

    def validate_reinforcement_matrix(self, delta):

        # Check if it is a matrix
        if not delta.ndim == 2:
            raise ValueError('Delta must be a 2D array')

        # Check that it is square
        if not delta.shape[0] == delta.shape[1]:
            raise ValueError('Delta must be a square matrix')

        # Check that it only contains integers
        if not np.all(np.equal(np.mod(delta, 1), 0)):
            raise ValueError('Delta must contain only integers')

        # Check that all rows sum to the same integer value b
        sums = np.sum(delta, axis=1)
        if not np.all(sums == sums[0]):
            raise ValueError('All rows must sum to the same total')