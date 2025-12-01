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


# --- Hyperballistic urn ---

reinforcement_matrix_hyper = -5 * np.ones((3, 3)) + 25 * np.eye(3)

# 20 iterations
labeler_20 = GPPLabeler(seed)
hyper_urns_20 = labeler_20.label(ml_probs, neighborhood, initial_total_balls=100,
                                   R=reinforcement_matrix_hyper, n_iter=20, return_type='urns', input_type='probs')
hyper_probs_20 = hyper_urns_20 / hyper_urns_20.sum(axis=2, keepdims=True)
hyper_labels_20 = label_image_from_probabilities(hyper_probs_20)
hyper_img_20 = align_labels(hyper_labels_20, ground_truth)

# 50 iterations
labeler_50 = GPPLabeler(seed)
hyper_urns_50 = labeler_50.label(hyper_urns_20, neighborhood, initial_total_balls=None,
                                   R=reinforcement_matrix_hyper, n_iter=30, return_type='urns', input_type='urns')
hyper_probs_50 = hyper_urns_50 / hyper_urns_50.sum(axis=2, keepdims=True)
hyper_labels_50 = label_image_from_probabilities(hyper_probs_50)
hyper_img_50 = align_labels(hyper_labels_50, ground_truth)

# 100 iterations
labeler_100 = GPPLabeler(seed)
hyper_urns_100 = labeler_100.label(hyper_urns_50, neighborhood, initial_total_balls=None,
                                   R=reinforcement_matrix_hyper, n_iter=50, return_type='urns', input_type='urns')
hyper_probs_100 = hyper_urns_100 / hyper_urns_100.sum(axis=2, keepdims=True)
hyper_labels_100 = label_image_from_probabilities(hyper_probs_100)
hyper_img_100 = align_labels(hyper_labels_100, ground_truth)

# 200 iterations
labeler_200 = GPPLabeler(seed)
hyper_urns_200 = labeler_200.label(hyper_urns_100, neighborhood, initial_total_balls=None,
                                   R=reinforcement_matrix_hyper, n_iter=100, return_type='urns', input_type='urns')
hyper_probs_200 = hyper_urns_200 / hyper_urns_200.sum(axis=2, keepdims=True)
hyper_labels_200 = label_image_from_probabilities(hyper_probs_200)
hyper_img_200 = align_labels(hyper_labels_200, ground_truth)


# --- Show results ---

fig1, ax = plt.subplots(5, 2, figsize=(12, 12))

ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original')

ax[0, 1].imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax[0, 1].axis('off')
ax[0, 1].set_title('Ground Truth')

ax[1, 0].imshow(rl_img_20, cmap='gray', vmin=0, vmax=255)
ax[1, 0].axis('off')
ax[1, 0].set_title('RL (n=20)')

ax[1, 1].imshow(hyper_img_20, cmap='gray', vmin=0, vmax=255)
ax[1, 1].axis('off')
ax[1, 1].set_title('Hyperballistic (n=20)')

ax[2, 0].imshow(rl_img_50, cmap='gray', vmin=0, vmax=255)
ax[2, 0].axis('off')
ax[2, 0].set_title('RL (n=50)')

ax[2, 1].imshow(hyper_img_50, cmap='gray', vmin=0, vmax=255)
ax[2, 1].axis('off')
ax[2, 1].set_title('Hyperballistic (n=50)')

ax[3, 0].imshow(rl_img_100, cmap='gray', vmin=0, vmax=255)
ax[3, 0].axis('off')
ax[3, 0].set_title('RL (n=100)')

ax[3, 1].imshow(hyper_img_100, cmap='gray', vmin=0, vmax=255)
ax[3, 1].axis('off')
ax[3, 1].set_title('Hyperballistic (n=100)')

ax[4, 0].imshow(rl_img_200, cmap='gray', vmin=0, vmax=255)
ax[4, 0].axis('off')
ax[4, 0].set_title('RL (n=200)')

ax[4, 1].imshow(hyper_img_200, cmap='gray', vmin=0, vmax=255)
ax[4, 1].axis('off')
ax[4, 1].set_title('Hyperballistic (n=200)')

fig1.tight_layout()
plt.show()