import numpy as np
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

from synthesizers.textured_image_generator import TexturedImageGenerator


class DisconnectedRegionsImageGenerator(TexturedImageGenerator):


    def generate_mask(self, size=(256, 256), n_regions=3, seed=0):
        """
        Genera una máscara con muchas regiones desconectadas, usando ruido Perlin.
        """
        h, w = size
        rng = np.random.default_rng(seed)

        # --- Generar ruido Perlin suave ---
        noise = PerlinNoise(octaves=5, seed=seed)
        field = np.zeros(size)

        for i in range(h):
            for j in range(w):
                field[i, j] = noise([i/h, j/w])

        # Normalizar 0-1
        field = (field - field.min()) / (field.max() - field.min())

        # --- Cortar en múltiples parches desconectados ---
        # Usamos igual separación en cuantiles para asignar clases dispersas
        thresholds = np.linspace(0, 1, n_regions + 1)

        mask = np.zeros_like(field, dtype=np.int32)
        for k in range(n_regions):
            region = (field >= thresholds[k]) & (field < thresholds[k+1])
            mask[region] = k

        return mask


if __name__ == "__main__":
    img, gt = DisconnectedRegionsImageGenerator().generate_textured_image(size=(256, 256), n_regions=3, smoothness=1.0, intensity=1.2, seed=10)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(gt, cmap="gray")
    axs[0].set_title("Ground truth")
    axs[1].imshow(img, cmap="gray")
    axs[1].set_title("Imagen texturada")
    plt.show()
