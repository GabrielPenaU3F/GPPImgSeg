from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from synthesizers.textured_image_generator import TexturedImageGenerator

"""
    Genera una imagen sintética con regiones texturadas y un mapa de etiquetas.
    Cada región puede tener diferente intensidad y suavidad de textura.

    Parámetros
    ----------
    h, w : int
        Dimensiones de la imagen.
    n_regions : int
        Número de regiones a generar.
    smoothness : float o array-like de longitud n_regions
        Grado de suavizado del ruido (sigma del filtro gaussiano).
    intensity : float o array-like de longitud n_regions
        Escala de intensidad del ruido para cada región.
    seed : int o None
        Semilla para reproducibilidad.

    Retorna
    -------
    image : np.ndarray
        Imagen sintética (h, w).
    labels : np.ndarray
        Mapa de etiquetas (h, w) con valores en [0, n_regions-1].
    """

class ConnectedRegionsImageGenerator(TexturedImageGenerator):

    def generate_mask(self, size=(256, 256), n_regions=3, seed=None):
        h, w = size
        y = np.arange(0, h)
        x = np.arange(0, w)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

        kmeans = KMeans(n_clusters=n_regions, n_init=10, random_state=seed)
        mask = kmeans.fit_predict(coords)

        return mask.astype(np.int32)

if __name__ == '__main__':

    img, ground_truth = ConnectedRegionsImageGenerator().generate_textured_image(
        size=(256, 256), n_regions=3, seed=42, smoothness=[0.5, 0.1, 0.3], intensity=[12.2, 1.1, 1.0])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(ground_truth, cmap='gray')
    plt.show()