import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from synthesizers.voronoi_regions_generator import generate_regions_voronoi


class BlendedBordersImageGenerator:


    def generate_blended_borders_image(self, size, n_regions=6, blend_width=20, seed=None):
        h, w = size
        img, labels = generate_regions_voronoi(size, n_regions, return_type='full', seed=seed)

        # Creamos una imagen suave por región: un plano con gradiente + ruido suave
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(seed)

        # ---- 1) Construimos una textura suave por región ----
        region_img = np.zeros((h, w, n_regions))
        base = np.linspace(50, 220, n_regions)
        for k in range(n_regions):
            a = rng.uniform(-0.002, 0.002)
            b = rng.uniform(-0.002, 0.002)
            plane = base[k] + a * xx + b * yy
            noise = gaussian_filter(rng.normal(0, 5, size=(h, w)), sigma=3)
            region_img[:, :, k] = plane + noise

        # ---- 2) Distancia real al borde ----
        borders = np.zeros((h, w), dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neigh = labels[
                np.clip(np.arange(h)[:, None] + dy, 0, h - 1),
                np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
            ]
            borders |= (neigh != labels)

        dist_to_border = distance_transform_edt(~borders)

        # ---- 3) Región más cercana distinta ----
        distances = np.zeros((h, w, n_regions))
        for j in range(n_regions):
            distances[:, :, j] = distance_transform_edt(labels != j)

        # ignorar la región original
        for j in range(n_regions):
            mask = (labels == j)
            distances[mask, j] = np.inf

        second_label = distances.argmin(axis=2)

        # ---- 4) Mezcla con crossfade de ancho blend_width ----
        base_img = region_img[np.arange(h)[:, None], np.arange(w)[None, :], labels]
        alt_img = region_img[np.arange(h)[:, None], np.arange(w)[None, :], second_label]

        t = np.clip(1 - dist_to_border / blend_width, 0, 1)
        final = (1 - t) * base_img + t * alt_img

        # Format ground truth
        gray_levels = np.linspace(30, 225, n_regions).astype(np.float32)
        ground_truth = gray_levels[labels]

        return final.astype(np.float32), ground_truth

if __name__ == '__main__':
    img, ground_truth = (BlendedBordersImageGenerator().
                         generate_blended_borders_image((300, 300), n_regions=6, blend_width=10, seed=0))

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(ground_truth, cmap='gray')
    ax[1].set_title('Ground Truth')
    plt.show()