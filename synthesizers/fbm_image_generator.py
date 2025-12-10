import numpy as np
import matplotlib.pyplot as plt

from synthesizers.voronoi_regions_generator import generate_regions_voronoi


class FBMImageGenerator:

    def generate_fbm_texture(self, H, size, seed=None):
        """
        Generate a fractional Brownian motion texture with exponent H.
        Fast implementation using frequency-domain filtering.
        """
        h, w = size
        rng = np.random.default_rng(seed)

        noise = rng.standard_normal((h, w))

        ky = np.fft.fftfreq(h).reshape(-1, 1)
        kx = np.fft.fftfreq(w).reshape(1, -1)
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0

        amp = k2 ** (-(H + 1) / 2)

        fft_noise = np.fft.fft2(noise)
        filtered = fft_noise * amp
        fbm = np.fft.ifft2(filtered).real

        fbm -= fbm.min()
        fbm /= fbm.max()
        return fbm

    def generate_fbm_image(self, size, n_regions, H, base_intensities=None, seed=None):
        """
        Generate an image composed of several vertical regions,
        each with an fBm of different H.

        Parameters
        ----------
        size : (h, w)
            Output image size.
        n_regions : int
            Number of vertical regions (must match len(H_list)).
        H : list of floats
            List of Hurst parameters for each region.
        seed : int or None

        Returns
        -------
        img : 2D numpy array
            Composite fBm image.
        label_map : 2D array
            Ground-truth segmentation map (region indices).
        """
        if len(H) != n_regions:
            raise ValueError('H must be a list of length n_regions')

        if base_intensities is None:
            base_intensities = np.linspace(20, 235, n_regions)
        else:
            base_intensities = np.array(base_intensities, dtype=int)

        h, w = size
        rng = np.random.default_rng(seed)
        # We use a different seed for each region
        seeds = rng.integers(0, 1_000_000, size=n_regions)

        # Voronoi partition
        _, labels = generate_regions_voronoi(size, n_regions, return_type='full', seed=seed)

        # fBm for each region
        img = np.zeros((h, w))
        for i in range(n_regions):
            mask = (labels == i)
            if mask.sum() == 0:
                continue

            # Fill the image in the corresponding region
            fbm_full = self.generate_fbm_texture(H[i], (h, w), seed=seeds[i])
            img[mask] = fbm_full[mask]

        # Normalize final output
        img -= img.min()
        img /= img.max()
        img *= 255.0
        ground_truth = base_intensities[labels].reshape(h, w).astype(np.float32)

        return img, ground_truth


if __name__ == '__main__':
    img, gt = FBMImageGenerator().generate_fbm_image(
        size=(256, 256),
        n_regions=4,
        H=[0.2, 0.5, 0.8, 1.2],
        seed=123
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Image')
    ax[1].imshow(gt, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Ground truth')
    plt.show()
