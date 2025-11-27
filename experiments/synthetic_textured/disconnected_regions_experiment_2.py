import numpy as np

from matplotlib import pyplot as plt
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import PolyaLabeler, GPPLabeler
from segmentation.metrics import SegmentationComparator
from segmentation.neighborhood import Neighborhood
from synthesizers.disconnected_image_generator import DisconnectedRegionsImageGenerator
from utilities.image_format_utilities import label_image_from_probabilities, align_labels, normalize_labels
from utilities.output_utilities import plot_confusion_matrix, plot_regional_mse_bars

k = 3 # Number of regions
seed = 42
generator = DisconnectedRegionsImageGenerator()
img, ground_truth = generator.generate_textured_image(size=(256, 256), n_regions=k, seed=seed,
                                                      smoothness=[0.4, 0.4, 0.4], intensity=[0.7, 0.8, 1.2])

# --- Labeling ---

nmc_labels = NMCLabeler(seed).label(img, n_iter=10, n_classes=k, return_type='raw')
nmc_img = align_labels(nmc_labels, ground_truth)

fig1, ax = plt.subplots(1, 3, figsize=(12, 8))

ml_probs = MLLabeler().label(img, nmc_labels, return_type='probs')
ml_labels = label_image_from_probabilities(ml_probs)
ml_img = align_labels(ml_labels, ground_truth)

# --- Relaxation ---

neighborhood = Neighborhood('radius', radius=4)

# 20 iterations
rl_probs_20 = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=20, return_type='probs')
rl_labels_20 = label_image_from_probabilities(rl_probs_20)
rl_img_20 = align_labels(rl_labels_20, ground_truth)

# 50 iterations
rl_probs_50 = RelaxationLabeler().label(rl_probs_20, neighborhood, n_iter=30, return_type='probs')
rl_labels_50 = label_image_from_probabilities(rl_probs_50)
rl_img_50 = align_labels(rl_labels_50, ground_truth)

# 100 iterations
rl_probs_100 = RelaxationLabeler().label(rl_probs_50, neighborhood, n_iter=50, return_type='probs')
rl_labels_100 = label_image_from_probabilities(rl_probs_100)
rl_img_100 = align_labels(rl_labels_100, ground_truth)

# 200 iterations
rl_probs_200 = RelaxationLabeler().label(rl_probs_100, neighborhood, n_iter=100, return_type='probs')
rl_labels_200 = label_image_from_probabilities(rl_probs_200)
rl_img_200 = align_labels(rl_labels_200, ground_truth)

reinforcement_matrix_hyper = -5 * np.ones((3, 3)) + 25 * np.eye(3)
gpp_hyper_labels = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                      R=reinforcement_matrix_hyper, n_iter=20, return_type='img')
hyper_img_20 = align_labels(gpp_hyper_labels, ground_truth)

fig1, ax = plt.subplots(3, 2, figsize=(12, 8))

ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original')

ax[0, 1].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('Ground Truth')

ax[1, 0].imshow(rl_img_20, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title('RL (n=20)')

ax[2].imshow(hyper_img_20, cmap='gray', vmin=0, vmax=255)
ax[2].axis('off')
ax[2].set_title('Hyperballistic (n=20)')

fig1.tight_layout()
plt.show()