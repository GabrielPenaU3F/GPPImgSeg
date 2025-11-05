from PIL import Image

import numpy as np

from matplotlib import pyplot as plt
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import PolyaLabeler, GPPLabeler
from segmentation.neighborhood import Neighborhood
from segmentation.utilities import format_image, label_image_from_probabilities
from synthesizers.image_generators import generate_regions_voronoi
from synthesizers.noise_generators import add_salt_pepper_noise

k = 6 # Number of regions
img = generate_regions_voronoi(256, 256, n_regions=k, seed=123)
noisy_img = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02, seed=42)

X_img = Image.fromarray(img)
X_noisy = Image.fromarray(noisy_img)

nmc_labels = NMCLabeler().label(noisy_img, n_iter=20, n_classes=k, return_type='raw')
nmc_img = Image.fromarray(format_image(nmc_labels, k))

ml_probs = MLLabeler().label(noisy_img, nmc_labels, return_type='probs')
ml_img = Image.fromarray(label_image_from_probabilities(ml_probs))

neighborhood = Neighborhood('radius', radius=4)
rl_img = Image.fromarray(RelaxationLabeler().label(ml_probs, neighborhood, n_iter=20, return_type='img'))

polya_img = Image.fromarray(PolyaLabeler().label(ml_probs, neighborhood, initial_total_balls=100, R=10, n_iter=20,
                                                 return_type='img'))

reinforcement_matrix_super = np.ones((6, 6)) + 8 * np.eye(6)
gpp_superdif_img = Image.fromarray(GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                                      R=reinforcement_matrix_super, n_iter=20, return_type='img'))

reinforcement_matrix_sub = np.ones((6, 6))
gpp_subdif_img = Image.fromarray(GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                                    R=reinforcement_matrix_sub, n_iter=20, return_type='img'))

reinforcement_matrix_hyper = -np.ones((6, 6)) + 9 * np.eye(6)
gpp_hyper_img = Image.fromarray(GPPLabeler().label(ml_probs, neighborhood, initial_total_balls=100,
                                                    R=reinforcement_matrix_hyper, n_iter=20, return_type='img'))

# ---- Plot ----

fig, ax = plt.subplots(2, 4, figsize=(12, 8))

ax[0, 0].imshow(X_noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy Image')

ax[0, 1].imshow(nmc_img)
ax[0, 1].axis('off')
ax[0, 1].set_title('NMC labeling')

ax[0, 2].imshow(ml_img)
ax[0, 2].axis('off')
ax[0, 2].set_title('NMC + ML labeling')

ax[0, 3].imshow(rl_img)
ax[0, 3].axis('off')
ax[0, 3].set_title('Relaxation labeling')

ax[1, 0].imshow(gpp_subdif_img)
ax[1, 0].axis('off')
ax[1, 0].set_title('Subdiffusive GPP labeling')

ax[1, 1].imshow(gpp_superdif_img)
ax[1, 1].axis('off')
ax[1, 1].set_title('Superdiffusive GPP labeling')

ax[1, 2].imshow(polya_img)
ax[1, 2].axis('off')
ax[1, 2].set_title('Polya labeling')

ax[1, 3].imshow(gpp_hyper_img)
ax[1, 3].axis('off')
ax[1, 3].set_title('Hyperballistic GPP labeling')

fig.tight_layout()
plt.show()
