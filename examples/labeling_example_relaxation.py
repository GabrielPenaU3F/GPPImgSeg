from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.neighborhood import Neighborhood
from utilities import format_labeled_image, label_image_from_probabilities

image_path = '../resources/test_img.bmp'
img = Image.open(image_path).convert('L')
X = np.array(img)
n_classes = 3

nmc_labels = NMCLabeler().label(X, n_iter=10, n_classes=n_classes, return_type='raw')
nmc_img = format_labeled_image(nmc_labels, n_classes)

ml_probs = MLLabeler().label(X, nmc_labels, return_type='probs')
ml_img = label_image_from_probabilities(ml_probs)

neighborhood = Neighborhood('8')
rl_img = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=10, return_type='img')

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(nmc_img, cmap='gray')
axes[0, 0].set_title('NMC Image')

axes[0, 1].imshow(ml_img, cmap='gray')
axes[0, 1].set_title('ML Image')

axes[1, 0].imshow(rl_img, cmap='gray')
axes[1, 0].set_title('RL Image')

fig.tight_layout()
plt.show()
