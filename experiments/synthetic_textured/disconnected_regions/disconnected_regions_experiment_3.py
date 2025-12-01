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

neighborhood = Neighborhood('radius', radius=3)

# 200 iterations

rl_probs = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=50, return_type='probs')
rl_labels = label_image_from_probabilities(rl_probs)
rl_img = align_labels(rl_labels, ground_truth)

# Negative reinforcement on the minoritarian neighbors works wonders
R_1 = -200 * np.ones((3, 3)) + 200 * np.eye(3)
R_2 = -200 * np.ones((3, 3)) + 400 * np.eye(3)

labeler_1 = GPPLabeler(seed=42)
labeler_2 = GPPLabeler(seed=42)

hyper_labels_1 = labeler_1.label(ml_probs, neighborhood, initial_total_balls=500,
                                    R=R_1, n_iter=50, return_type='img',input_type='probs')
hyper_img_1 = align_labels(hyper_labels_1, ground_truth)

hyper_labels_2 = labeler_2.label(ml_probs, neighborhood, initial_total_balls=500,
                                    R=R_2, n_iter=50, return_type='img', input_type='probs')
hyper_img_2 = align_labels(hyper_labels_2, ground_truth)


fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('Ground Truth')

ax[0, 1].imshow(rl_img, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('RL (n=50)')

ax[1, 0].imshow(hyper_img_1, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title('Hyper - purely negative (n=50)')

ax[1, 1].imshow(hyper_img_2, cmap='gray', vmin=0, vmax=255)
ax[1, 1].axis('off')
ax[1, 1].set_title('Hyper - negative + positive (n=50)')

fig.tight_layout()
plt.show()
