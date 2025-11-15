from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from segmentation.utilities import format_regular_image, format_labeled_image, align_labels

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

def generate_textured_image( # Typing
        size: tuple[int, int] = (256, 256),
        n_regions: int = 3,
        base_intensities: np.ndarray = None,
        smoothness: Union[float, np.ndarray] = 1.0,
        intensity: Union[float, np.ndarray] = 1.0,
        seed: int | None = None
):

    h, w = size
    if base_intensities is None:
        base_intensities = np.linspace(20, 235, n_regions)
    else:
        base_intensities = np.array(base_intensities, dtype=int)

    mask = generate_connected_masks(size, n_regions, seed=seed).reshape(h, w)
    ground_truth = base_intensities[mask].reshape(h, w).astype(np.float32)
    image = np.copy(ground_truth)

    # --- Validation of parameters ---
    def ensure_per_region(param, name):
        if np.isscalar(param):
            return np.full(n_regions, param)
        param = np.asarray(param)
        if param.shape != (n_regions,):
            raise ValueError(f"'{name}' must be a scalar or an array of size {n_regions}")
        return param

    smoothness = ensure_per_region(smoothness, "smoothness")
    intensity = ensure_per_region(intensity, "intensity")

    for k in range(n_regions):
        region = (mask == k)
        texture = generate_textured_region(size, smoothness[k], seed)
        intensity_k = intensity[k] * base_intensities[k]
        image += texture * region * intensity_k

    return image, ground_truth

def generate_textured_region(size=(256, 256), smoothness=1.0, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.random(size)
    texture = gaussian_filter(noise, sigma=smoothness)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-7)
    return texture

def generate_connected_masks(size=(256, 256), n_regions=3, seed=None):
    h, w = size
    y = np.arange(0, h)
    x = np.arange(0, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

    kmeans = KMeans(n_clusters=n_regions, n_init=10, random_state=seed)
    mask = kmeans.fit_predict(coords)

    return mask.astype(np.int32)

if __name__ == '__main__':

    img, ground_truth = generate_textured_image(size=(256, 256), n_regions=3, seed=42,
                                                smoothness=[0.5, 0.1, 0.3], intensity=[12.2, 1.1, 1.0])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(ground_truth, cmap='gray')
    plt.show()