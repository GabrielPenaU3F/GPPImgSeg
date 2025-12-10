from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.methods.urn_labelers import GPPLabeler
from segmentation.neighborhood import Neighborhood
from utilities.image_format_utilities import format_labeled_image, label_image_from_probabilities

image_path = '../resources/test_img_4.bmp'
img = Image.open(image_path).convert('L')
X = np.array(img)
k = 3

nmc_labels = NMCLabeler().label(X, n_iter=10, n_classes=k, return_type='raw')
nmc_img = format_labeled_image(nmc_labels, k)

ml_probs = MLLabeler().label(X, nmc_labels, return_type='probs')
ml_img = label_image_from_probabilities(ml_probs)

neighborhood = Neighborhood('radius', radius=3)
rl_img = RelaxationLabeler().label(ml_probs, neighborhood, n_iter=20, return_type='img')

gpp = GPPLabeler()
R = -200 * np.ones((k, k)) + 400 * np.eye(k)
hyper_img = gpp.label(ml_probs, neighborhood, initial_total_balls=500,
                      R=R, n_iter=20, return_type='img',input_type='probs')

fig, axes = plt.subplots(1, 2)

axes[0].imshow(rl_img, cmap='gray')
axes[0].set_title('RL Image')

axes[1].imshow(hyper_img, cmap='gray')
axes[1].set_title('GPP Image')

fig.tight_layout()
plt.show()
