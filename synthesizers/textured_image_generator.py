import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

def generate_textured_image(size=(256, 256), n_regions=3, smoothness=1.0, intensity=1.0, seed=None):
    rng = np.random.default_rng(seed)
    h, w = size
    masks = generate_connected_masks(size, n_regions, seed=seed)
    image = np.zeros((h, w))

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
        region_texture = generate_textured_region(size, smoothness[k], intensity[k], None if seed is None else seed + k)
        image[masks == k] = region_texture[masks == k] + 0.3 * k  # diferenciar tonos base

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image, masks

def generate_textured_region(size=(256, 256), smoothness=1.0, intensity=0.5, seed=None):
    noise = np.random.rand(*size)
    texture = gaussian_filter(noise, sigma=smoothness)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    texture = texture * intensity
    return texture

def generate_connected_masks(size=(256, 256), n_regions=3, seed=None):
    h, w = size
    y = np.arange(0, h)
    x = np.arange(0, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)

    kmeans = KMeans(n_clusters=n_regions, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(coords)
    masks = labels.reshape(h, w)
    return masks

if __name__ == '__main__':

    img, _ = generate_textured_image(size=(256, 256), n_regions=3, smoothness=[0.5, 0.1, 0.8], intensity=1.5)
    plt.imshow(img, cmap='gray')
    plt.show()