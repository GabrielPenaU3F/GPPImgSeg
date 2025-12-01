import numpy as np

from matplotlib import pyplot as plt
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import GPPLabeler
from segmentation.neighborhood import Neighborhood
from synthesizers.disconnected_image_generator import DisconnectedRegionsImageGenerator
from utilities.image_format_utilities import label_image_from_probabilities, align_labels

k = 3 # Number of regions
seed = 42
generator = DisconnectedRegionsImageGenerator()
img, ground_truth = generator.generate_textured_image(size=(256, 256), n_regions=k, seed=seed,
                                                      smoothness=[0.4, 0.4, 0.4], intensity=[0.7, 0.8, 1.2])

# --- Labeling ---

nmc_labels = NMCLabeler(seed).label(img, n_iter=10, n_classes=k, return_type='raw')
nmc_img = align_labels(nmc_labels, ground_truth)

ml_probs = MLLabeler().label(img, nmc_labels, return_type='probs')
ml_labels = label_image_from_probabilities(ml_probs)
ml_img = align_labels(ml_labels, ground_truth)

# --- Relaxation ---

neighborhood = Neighborhood('radius', radius=2)

# 200 iterations

rl_probs = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=50, return_type='probs')
rl_labels = label_image_from_probabilities(rl_probs)
rl_img = align_labels(rl_labels, ground_truth)

# Negative reinforcement on the minoritarian neighbors works wonders
R_init = -200 * np.ones((3, 3)) + 201 * np.eye(3)
R_final = 100 * np.eye(3)
def adaptive_R(n):
    if n < 20:
        return R_init
    elif 20 <= n < 50:
        t = (n - 20) / 30
        R = R_init + t * (R_final - R_init)
        return np.round(R).astype(int)
    elif n >= 50:
        return R_final

hyper_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=500,
                                  R=R_init, n_iter=100, return_type='img',
                                  input_type='probs', seed=42)
hyper_img = align_labels(hyper_labels, ground_truth)

adaptive_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=500,
                                     R=lambda n: adaptive_R(n), n_iter=100, return_type='img', input_type='probs',
                                     seed=42)
adaptive_img = align_labels(adaptive_labels, ground_truth)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('Ground Truth')

ax[0, 1].imshow(rl_img, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('RL (n=50)')

ax[1, 0].imshow(hyper_img, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title('Hyper (n=100)')

ax[1, 1].imshow(adaptive_img, cmap='gray', vmin=0, vmax=255)
ax[1, 1].axis('off')
ax[1, 1].set_title('Adaptive (n=100)')

fig.tight_layout()
plt.show()
