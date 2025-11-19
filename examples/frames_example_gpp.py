import numpy as np
from PIL import Image

from segmentation.methods.urn_labelers import GPPLabeler
from segmentation.methods.ml_labeler import MLLabeler
from segmentation.methods.nmc_labeler import NMCLabeler
from segmentation.neighborhood import Neighborhood

from matplotlib import pyplot as plt

image_path = '../resources/test_img_4.bmp'
img = Image.open(image_path)
X = np.array(img)
labeler = GPPLabeler()

init_labels = NMCLabeler().label(X, n_iter=10, n_classes=3, return_type='raw')
init_probabilities = MLLabeler().label(X, init_labels, return_type='probs')
neighborhood = Neighborhood('radius', radius=8)
reinforcement_matrix = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]])
Y = labeler.label(init_probabilities, neighborhood, initial_total_balls=100, R=reinforcement_matrix, n_iter=10,
                  return_type='img', watch_evolution=True, save_directory='outputs/gpp_test')

plt.imshow(Y, cmap='gray')
plt.show()
