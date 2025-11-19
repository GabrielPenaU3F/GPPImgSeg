import numpy as np
from PIL import Image

from segmentation.methods.nmc_labeler import NMCLabeler
from matplotlib import pyplot as plt

image_path = '../resources/test_img_3.png'
img = Image.open(image_path)
X = np.array(img)
segmented_img = NMCLabeler().label(X, n_iter=20, n_classes=3, return_type='img',
                                   watch_evolution=True, save_directory='outputs/nmc_test')

plt.imshow(segmented_img, cmap='gray')
plt.show()
