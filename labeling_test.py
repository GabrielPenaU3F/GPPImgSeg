from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

from segmentation.methods.ml_labeling import ml_labeling
from segmentation.methods.nmc_labeling import nmc
from segmentation.methods.relaxation_labeling import relaxation_labeling
from segmentation.utilities import format_image, label_image_from_probabilities

image_path = 'resources/test_img.bmp'
img = Image.open(image_path).convert('L')
X = np.array(img)
n_classes = 3

nmc_labels = nmc(X, n_iter=10, n_classes=n_classes, return_type='raw')
nmc_img = format_image(nmc_labels, n_classes)

ml_probs = ml_labeling(X, nmc_labels, return_type='probs')
ml_img = label_image_from_probabilities(ml_probs)

rl_img = relaxation_labeling(ml_probs, n_iter=10, return_type='img')

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(nmc_img, cmap='gray')
axes[0, 0].set_title('NMC Image')

axes[0, 1].imshow(ml_img, cmap='gray')
axes[0, 1].set_title('ML Image')

axes[1, 0].imshow(rl_img, cmap='gray')
axes[1, 0].set_title('RL Image')

fig.tight_layout()
plt.show()