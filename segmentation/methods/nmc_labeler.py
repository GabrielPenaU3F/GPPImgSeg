import numpy as np

from segmentation.utilities import format_labeled_image, save_frame


class NMCLabeler:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def compute_centroids(self, X, labels, n_classes):
        centroids = []
        for k in range(0, n_classes):
            members = X[labels == k]
            if len(members) > 0:
                centroids.append(members.mean(axis=0))
            else:
                # reinit if a cluster is empty
                centroids.append(X[np.random.randint(0, X.shape[0])])
        return np.vstack(centroids)

    def label(self, X, n_iter, n_classes, return_type='img',
              watch_evolution=False, save_directory='outputs/polya_test'):
        channels = X.shape[-1] if len(X.shape) > 2 else 1
        x = X.reshape(-1, channels)
        labels = self.rng.integers(0, high=n_classes, size=x.shape[0])
        for n in range(n_iter):
            centroids = self.compute_centroids(x, labels, n_classes)
            dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            if watch_evolution:
                if save_directory is None:
                    raise Exception('Save directory not specified')
                img = labels.reshape(X.shape[:2])
                img = format_labeled_image(img, n_classes)
                save_frame(img, n + 1, save_directory)

        segmented_img = labels.reshape(X.shape[:2]) # We rebuild a single channel
        if return_type == 'img':
            segmented_img = format_labeled_image(segmented_img, n_classes)
        return segmented_img
