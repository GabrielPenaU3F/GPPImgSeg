import numpy as np
import scipy.ndimage as ndi


class SoftDegradeImageGenerator:

    def generate_soft_texture(self, size, seed=None):
        h, w = size
        rng = np.random.default_rng(seed)

        # Ruido inicial
        noise = rng.normal(0, 1, (h, w))

        # Suavizado fuerte para obtener textura suave
        sigma = rng.uniform(5, 25)
        smooth_noise = ndi.gaussian_filter(noise, sigma=sigma)

        # Gradiente suave aleatorio
        gx = rng.uniform(-0.5, 0.5)
        gy = rng.uniform(-0.5, 0.5)
        y, x = np.mgrid[0:h, 0:w]
        gradient = gx * x + gy * y

        # Otra capa suave (tipo iluminación)
        illum = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), sigma=50)

        out = smooth_noise + gradient + 0.2 * illum
        return out

    def generate_soft_degrade_image(self, size=(300, 300), n_regions=2, seed=None):
        rng = np.random.default_rng(seed)
        h, w = size

        # === 1. Voronoi partition ===
        cx = rng.integers(0, w, size=n_regions)
        cy = rng.integers(0, h, size=n_regions)

        y, x = np.mgrid[0:h, 0:w]
        dist = np.stack([(x - cx[i]) ** 2 + (y - cy[i]) ** 2 for i in range(n_regions)])
        labels = np.argmin(dist, axis=0)

        # === 2. Generate texture for each region ===
        img = np.zeros((h, w))

        for i in range(n_regions):
            tex = self.generate_soft_texture(size, seed=rng.integers(1_000_000))
            img[labels == i] = tex[labels == i]

        # === 3. Blur border to create true overlap ===
        # Compute distance to region boundary
        boundary = ndi.binary_dilation(labels != ndi.maximum_filter(labels, size=3))
        dist_to_boundary = ndi.distance_transform_edt(~boundary)
        dist_to_boundary = dist_to_boundary / dist_to_boundary.max()

        # Heavy blur on the "uncertain band"
        blurred = ndi.gaussian_filter(img, sigma=8)

        alpha = np.clip(np.exp(-(dist_to_boundary * 4)), 0, 1)
        img = alpha * img + (1 - alpha) * blurred

        # === 4. Normalize to [0, 255] ===
        img -= img.min()
        img /= img.max()
        img *= 255.0

        # Format ground truth
        gray_levels = np.linspace(30, 225, n_regions).astype(np.float32)
        ground_truth = gray_levels[labels]

        return img, ground_truth

if __name__ == '__main__':
    img, ground_truth = SoftDegradeImageGenerator().generate_soft_degrade_image((300, 300), n_regions=4)

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(ground_truth, cmap='gray')
    ax[1].set_title('Ground Truth')
    plt.show()