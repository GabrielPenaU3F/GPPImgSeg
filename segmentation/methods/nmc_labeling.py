import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from segmentation.utilities import format_image


def compute_centroids(X, labels, n_classes):
    centroids = []
    for k in range(0, n_classes):
        members = X[labels == k]
        if len(members) > 0:
            centroids.append(members.mean(axis=0))
        else:
            # reinit if a cluster is empty
            centroids.append(X[np.random.randint(0, X.shape[0])])
    return np.vstack(centroids)

def nmc(X, n_iter, n_classes, return_type='img'):
    channels = X.shape[-1] if len(X.shape) > 2 else 1
    x = X.reshape(-1, channels)
    labels = np.random.randint(0, high=n_classes, size=x.shape[0])
    for i in range(n_iter):
        centroids = compute_centroids(x, labels, n_classes)
        dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

    segmented_img = labels.reshape(X.shape[:2]) # We rebuild a single channel
    if return_type == 'img':
        segmented_img = format_image(segmented_img, n_classes)
    return segmented_img

if __name__ == "__main__":
    image_path = '../../resources/test_img.bmp'
    img = Image.open(image_path)
    X = np.array(img)
    segmented_img = nmc(X, n_iter=20, n_classes=10)
    Y = Image.fromarray(segmented_img)

    Y.show()
