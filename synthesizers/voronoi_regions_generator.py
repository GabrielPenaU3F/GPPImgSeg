import numpy as np
import matplotlib.pyplot as plt


def generate_regions_voronoi(size, n_regions=6, return_type='img', seed=None):

    h, w = size
    if seed is not None:
        np.random.seed(seed)

    # 1) Sample random centers
    margin_y = max(1, h // 20)
    margin_x = max(1, w // 20)
    xs = np.random.randint(margin_x, w - margin_x, size=n_regions)
    ys = np.random.randint(margin_y, h - margin_y, size=n_regions)
    centers = np.stack([xs, ys], axis=1)  # (n_regions, 2) -> (x, y)

    # 2) Make grid
    yy = np.arange(h)[:, None]   # (H,1)
    xx = np.arange(w)[None, :]    # (1,W)
    # Calculate distances to centers:
    # distance_sq: (H, W, n_regions)
    # compute (yy - cy)^2 + (xx - cx)^2 for each center
    dists_sq = np.empty((h, w, n_regions), dtype=np.int64)
    for k, (cx, cy) in enumerate(centers):
        # note: centers stored as (x, y)
        dists_sq[:, :, k] = (yy - cy)**2 + (xx - cx)**2

    # 3) Assign label by minimum distance (Voronoi)
    labels = np.argmin(dists_sq, axis=2).astype(np.int32)  # (H, W)

    # 4) Build image
    if n_regions == 1:
        gray_levels = np.array([128], dtype=np.uint8)
    else:
        gray_levels = np.linspace(30, 225, n_regions).astype(np.uint8)
    img = gray_levels[labels]

    if return_type == 'img':
        return img
    elif return_type == 'full':
        return img, labels

# Usage example
if __name__ == "__main__":
    size = (768, 1024)
    n_regions = 10
    img = generate_regions_voronoi(size, n_regions=n_regions, seed=42)

    fig, ax = plt.subplots()
    ax.set_title(f"Synthetic image ({n_regions} regions)")
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    plt.tight_layout()
    plt.show()