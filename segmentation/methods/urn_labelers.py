from abc import ABC

from matplotlib import pyplot as plt
import numpy as np

from utilities.image_format_utilities import label_image_from_probabilities
from utilities.output_utilities import save_frame
from utilities.segmentation_utilities import initialize_urns, get_neighbor_stack, sample_class_from_probs
from scipy.ndimage import label as cc_label


class UrnLabeler(ABC):

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def label(self, input, neighborhood, initial_total_balls, R, n_iter=10,
              input_type='probs', return_type='img', watch_evolution=False, save_directory=None, verbose=True):

        if input_type == 'probs':
            urns = initialize_urns(input, initial_total_balls)
        elif input_type == 'urns':
            urns = input
        else:
            raise ValueError("input_type must be 'probs' or 'urns'")

        self.validate_R(R, urns.shape[-1])

        for n in range(n_iter):
            if verbose is True:
                print(f'Iteration Nº{n+1}')

            '''
                Flow of the algorithm is the same as the relaxation labeling implementation
            '''
            neighbor_urn_stack = get_neighbor_stack(urns, neighborhood)
            super_urns = neighbor_urn_stack.sum(axis=2)  # (h, w, k)
            super_urn_probs = super_urns / super_urns.sum(axis=2, keepdims=True)

            sampled_classes = sample_class_from_probs(super_urn_probs, rng=self.rng)

            if callable(R):
                R_val = R(n)
            else:
                R_val = R
            urns = self.update_urns(urns, sampled_classes, R_val)

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

        else:
            raise ValueError("return_type must be 'img', 'probs', or 'urns'")

    def update_urns(self, urns, sampled_classes, delta):
        pass

    def validate_R(self, R_arg, k):

        # Case 1: R is a function
        if callable(R_arg):
            R = R_arg(-1)
            R = np.array(R)

        # Case 2: R is a square matrix
        elif isinstance(R_arg, (list, tuple, np.ndarray)):
            R = np.array(R_arg)

        else:
            raise TypeError('R must be a function or a matrix-like')

        # Validate shape, type and extra conditions
        if not R.shape == (k, k):
            raise ValueError('Reinforcement matrix shape is not correct')

        if not np.array_equal(R, R.astype(int)):
            raise ValueError('Reinforcement matrix must only contain integers')

        row_sums = R.sum(axis=1)
        if not np.all(row_sums == row_sums[0]):
            raise ValueError('All rows must sum to the same total')


class PolyaLabeler(UrnLabeler):

    # In Polya method, delta is the number of balls to add
    def update_urns(self, urns, sampled_classes, delta):
        h, w, k = urns.shape
        urns_flat = urns.reshape(-1, k)
        classes_flat = sampled_classes.ravel()
        np.add.at(urns_flat, (np.arange(h * w), classes_flat), delta)
        return urns_flat.reshape(h, w, k)

    def validate_R(self, R_arg, k):
        if R_arg != int(R_arg) or int(R_arg) <= 0:
            raise ValueError('Reinforcement must be a positive integer')

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

        # Ensure no urn becomes fully empty
        row_sums = urns_flat.sum(axis=1)
        empty_mask = (row_sums == 0)
        if np.any(empty_mask):
            urns_flat[empty_mask] = 1

        return urns_flat.reshape(h, w, k)


class StatisticsDiagnoseGPPLabeler(GPPLabeler):

    stats = {
        "var": [],  # varianza espacial por clase
        "entropy": [],  # entropía promedio del mapa
        "components": [],  # #componentes por clase
        "mass": [],  # masa total promedio por pixel
        "d_hist": []  # histograma del contraste d=u0-u1
    }

    def spatial_variance(self, probs):
        """Varianza espacial para cada clase: Var_x p(x,k)"""
        return np.var(probs, axis=(0, 1))

    def entropy_map(self, probs):
        """Entropía por pixel: -sum p log p"""
        eps = 1e-12
        return -np.sum(probs * np.log(probs + eps), axis=2)

    def count_components(self, seg):
        """Cuenta componentes conectados por clase. seg es imagen discreta."""
        classes = np.unique(seg)
        comp_counts = []
        for c in classes:
            structure = np.ones((3, 3), dtype=int)
            labeled, n = cc_label(seg == c, structure=structure)
            comp_counts.append(n)
        return comp_counts

    def mean_mass(self, urns):
        """masa promedio por pixel = sum_k urns / (h*w)"""
        return np.mean(np.sum(urns, axis=2))

    def get_stats(self):
        return self.stats

    def label(self, input, neighborhood, initial_total_balls, R, n_iter=10,
              input_type='probs', return_type='img', watch_evolution=False, save_directory=None, verbose=True):

        urns = initialize_urns(input, initial_total_balls)

        for n in range(n_iter):
            # Levantar vecindarios → super-urns
            neighbor_urn_stack = get_neighbor_stack(urns, neighborhood)
            super_urns = neighbor_urn_stack.sum(axis=2)
            super_urn_probs = super_urns / super_urns.sum(axis=2, keepdims=True)

            sampled_classes = sample_class_from_probs(super_urn_probs, rng=self.rng)

            # Obtener R(n) si es callable
            R_val = R(n) if callable(R) else R

            # Update tradicional
            urns = self.update_urns(urns, sampled_classes, R_val)

            # ---- Estadísticas ----
            probs = urns / urns.sum(axis=2, keepdims=True)
            seg = np.argmax(probs, axis=2)

            # (1) Varianza espacial por clase
            self.stats["var"].append(self.spatial_variance(probs))

            # (2) Entropía promedio
            self.stats["entropy"].append(np.mean(self.entropy_map(probs)))

            # (3) Componentes conexas
            self.stats["components"].append(self.count_components(seg))

            # (4) Masa total promedio
            self.stats["mass"].append(self.mean_mass(urns))

            # (5) Distribución de d = u0 - u1
            d = urns[:, :, 0] - urns[:, :, 1]
            hist, bins = np.histogram(d, bins=50, range=(-500, 500), density=True)
            self.stats["d_hist"].append((hist, bins))

        probs = urns / urns.sum(axis=2, keepdims=True)
        return label_image_from_probabilities(probs)
