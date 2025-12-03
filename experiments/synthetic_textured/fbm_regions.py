import numpy as np

from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import GPPLabeler
from segmentation.neighborhood import Neighborhood
from synthesizers.fbm_image_generator import FBMImageGenerator

from matplotlib import pyplot as plt

from utilities.image_format_utilities import align_labels, label_image_from_probabilities

k = 4
seed = 42
img, ground_truth = FBMImageGenerator().generate_fbm_image(
    size=(256, 256),
    n_regions=k,
    H=[0.2, 0.5, 0.8, 1.2],
    seed=seed
)
n_iter = 50

# --- Labeling ---

nmc_labels = NMCLabeler(seed).label(img, n_iter=10, n_classes=k, return_type='raw')
nmc_img = align_labels(nmc_labels, ground_truth)

ml_probs = MLLabeler().label(img, nmc_labels, return_type='probs')
ml_labels = label_image_from_probabilities(ml_probs)
ml_img = align_labels(ml_labels, ground_truth)

# --- Relaxation ---

neighborhood = Neighborhood('radius', radius=3)

rl_probs = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=n_iter, return_type='probs')
rl_labels = label_image_from_probabilities(rl_probs)
rl_img = align_labels(rl_labels, ground_truth)

R = -200 * np.ones((k, k)) + 400 * np.eye(k)

labeler_1 = GPPLabeler(seed=42)
labeler_2 = GPPLabeler(seed=42)

hyper_labels = labeler_1.label(ml_probs, neighborhood, initial_total_balls=500,
                               R=R, n_iter=n_iter, return_type='img',input_type='probs')
hyper_img = align_labels(hyper_labels, ground_truth)


fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original Image')

ax[0, 1].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('Ground Truth')

ax[1, 0].imshow(rl_img, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title(f'RL (n={n_iter})')

ax[1, 1].imshow(hyper_img, cmap='gray', vmin=0, vmax=255)
ax[1, 1].axis('off')
ax[1, 1].set_title(f'Hyper (n={n_iter})')

fig.tight_layout()
plt.show()
