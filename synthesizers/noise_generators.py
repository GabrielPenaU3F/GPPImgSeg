import numpy as np

from matplotlib import pyplot as plt

from synthesizers.voronoi_regions_generator import generate_regions_voronoi


def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01, seed=None):

    if seed is not None:
        np.random.seed(seed)

    noisy = img.copy()

    # Image dynamic range - we make this robust to the range
    min_val, max_val = noisy.min(), noisy.max()

    # Uniform-random matrix
    rand = np.random.rand(*img.shape[:2])

    # Salt pixels
    salt_mask = rand < salt_prob
    # Pepper pìxels
    pepper_mask = (rand >= salt_prob) & (rand < salt_prob + pepper_prob)

    if img.ndim == 2:
        noisy[salt_mask] = max_val
        noisy[pepper_mask] = min_val
    else:
        noisy[salt_mask, :] = max_val
        noisy[pepper_mask, :] = min_val

    return noisy


# Usage example
if __name__ == "__main__":
    img = generate_regions_voronoi(size=(256, 256), n_regions=6, seed=123)
    noisy_img = add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02, seed=42)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Noisy")
    plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()