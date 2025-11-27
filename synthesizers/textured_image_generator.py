from abc import ABC, abstractmethod
from typing import Union
from scipy.ndimage import gaussian_filter

import numpy as np


class TexturedImageGenerator(ABC):

    def generate_textured_image(self,  # Typing
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

        mask = self.generate_mask(size, n_regions, seed=seed).reshape(h, w)
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
            intensity_k = intensity[k] * base_intensities[k]
            texture = self.generate_textured_region(size, smoothness[k], intensity_k, seed)
            image += texture * region

        return image, ground_truth

    def generate_textured_region(self, size=(256, 256), smoothness=1.0, intensity=1.0, seed=None):
        rng = np.random.default_rng(seed)
        noise = rng.random(size)
        texture = gaussian_filter(noise, sigma=smoothness)
        texture = texture - texture.mean()
        texture = texture / np.max(np.abs(texture))
        texture = texture * intensity
        return texture

    @abstractmethod
    def generate_mask(self, size=(256, 256), n_regions=3, seed=None):
        pass
