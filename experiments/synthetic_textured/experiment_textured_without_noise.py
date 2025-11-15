import numpy as np

from matplotlib import pyplot as plt
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import PolyaLabeler, GPPLabeler
from segmentation.neighborhood import Neighborhood
from segmentation.utilities import format_labeled_image, label_image_from_probabilities, align_labels
from synthesizers.textured_image_generator import generate_textured_image

k = 3 # Number of regions
seed = 42
img, ground_truth = generate_textured_image(size=(256, 256), n_regions=k, seed=seed,
                                 smoothness=[0.4, 0.4, 0.4], intensity=[1.0, 0.8, 1.2])

# --- Reference

fig1, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].imshow(img, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Original Image')

ax[1].imshow(ground_truth, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Ground Truth')

plt.show()

nmc_labels = NMCLabeler(seed).label(img, n_iter=10, n_classes=k, return_type='raw')
nmc_img = align_labels(nmc_labels, ground_truth)

plt.imshow(nmc_img, cmap='gray')
plt.show()

ml_probs = MLLabeler().label(img, nmc_labels, return_type='probs')
ml_img = label_image_from_probabilities(ml_probs)

neighborhood = Neighborhood('radius', radius=4)
rl_img = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=20, return_type='img')

polya_img = PolyaLabeler().label(ml_probs, neighborhood, initial_total_balls=100, R=10, n_iter=30,
                                 return_type='img')

reinforcement_matrix_super = np.ones((3, 3)) + 8 * np.eye(3)
gpp_superdif_img = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                      R=reinforcement_matrix_super, n_iter=20, return_type='img')

reinforcement_matrix_sub = np.ones((3, 3))
gpp_subdif_img = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                    R=reinforcement_matrix_sub, n_iter=20, return_type='img')

reinforcement_matrix_hyper = -5 * np.ones((3, 3)) + 25 * np.eye(3)
gpp_hyper_img = GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                   R=reinforcement_matrix_hyper, n_iter=20, return_type='img')

# ---- Plot ----

fig2, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0, 0].imshow(ml_img, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('NMC + ML labeling')

ax[0, 1].imshow(rl_img, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('Relaxation labeling')

ax[0, 2].imshow(gpp_subdif_img, cmap='gray')
ax[0, 2].axis('off')
ax[0, 2].set_title('Subdiffusive GPP labeling')

ax[1, 0].imshow(gpp_superdif_img, cmap='gray')
ax[1, 0].axis('off')
ax[1, 0].set_title('Superdiffusive GPP labeling')

ax[1, 1].imshow(polya_img, cmap='gray')
ax[1, 1].axis('off')
ax[1, 1].set_title('Polya labeling')

ax[1, 2].imshow(gpp_hyper_img, cmap='gray')
ax[1, 2].axis('off')
ax[1, 2].set_title('Hyperballistic GPP labeling')

fig1.tight_layout()
fig2.tight_layout()

plt.show()
