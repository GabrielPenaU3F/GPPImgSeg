import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

from segmentation.methods.nmc_labeling import nmc
from segmentation.utilities import label_image_from_probabilities


def estimate_gaussian_distributions(x, labels, n_classes):
    means = []
    covs = []
    for k in range(n_classes):
        means.append(np.mean(x[labels == k], axis=0))
        covs.append(np.cov(x[labels == k], rowvar=False, bias=True))

    return means, covs

def ml_labeling(X, labels, return_type='img'):
    channels = X.shape[-1] if len(X.shape) > 2 else 1
    x = X.reshape(-1, channels)
    labels = labels.ravel()
    n_classes = len(np.unique(labels))
    means, covs = estimate_gaussian_distributions(x, labels, n_classes) # Mean and variance of each class
    inv_covs, logdets = compute_inverses_and_determinants(covs)
    log_probs = log_likelihood(x, channels, means, inv_covs, logdets)
    h, w = X.shape[:2]
    log_probs_img = log_probs.reshape(h, w, n_classes)

    if return_type == 'img':
        return label_image_from_probabilities(log_probs_img)

    elif return_type == 'probs':
        # Convert to probabilities and return
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.reshape(h, w, n_classes)

def compute_inverses_and_determinants(covs):
    inv_covs = []
    logdets = []
    for cov in covs:
        cov = np.atleast_2d(cov)  # fuerza a matriz (1,1) si es escalar
        cov = cov + 1e-6 * np.eye(cov.shape[0])  # Simple regularization to avoid errors
        inv_covs.append(np.linalg.inv(cov))
        logdets.append(np.log(np.linalg.det(cov)))
    inv_covs = np.stack(inv_covs)  # (n_classes, channels, channels)
    logdets = np.array(logdets)  # (n_classes,)
    return inv_covs, logdets


def log_likelihood(x, channels, means, inv_covs, logdets):
    # Expand: (N,d) - (K,d) → (N,K,d)
    shifted_xs = x[:, None, :] - np.array(means)[None, :, :]
    # Log-likelihood
    d_mahalanobis = np.einsum('nkd,kde,nke->nk', shifted_xs, inv_covs, shifted_xs)
    log_probs = -0.5 * (d_mahalanobis + logdets[None, :] + channels * np.log(2 * np.pi))
    return log_probs


if __name__ == '__main__':
    image_path = '../../resources/test_img_3.png'
    img = Image.open(image_path)
    X = np.array(img)
    init_labels = nmc(X, n_iter=10, n_classes=3, return_type='raw')
    segmented_img = ml_labeling(X, init_labels, return_type='img')

    Y = Image.fromarray(segmented_img)
    Y.show()
