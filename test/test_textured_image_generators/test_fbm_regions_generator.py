import numpy as np
import pytest

from synthesizers.fbm_image_generator import FBMImageGenerator

@pytest.fixture
def gen():
    return FBMImageGenerator()


class TestGenerateFBMRegion:

    def test_generate_fbm_basic_properties(self, gen):
        H = 0.7
        size = (64, 64)
        fbm = gen.generate_fbm_region(H, size, seed=123)

        assert fbm.shape == size
        assert np.isfinite(fbm).all(), "The FBM must contain only finite values"
        assert fbm.min() >= 0.0 and fbm.max() <= 1.0, "FBM must be normalized to [0,1]"
        assert fbm.std() > 0, "FBM should not be constant"

    def test_generate_fbm_seed_reproducibility(self, gen):
        size = (64, 64)
        H = 0.5
        fbm1 = gen.generate_fbm_region(H, size, seed=111)
        fbm2 = gen.generate_fbm_region(H, size, seed=111)

        assert np.allclose(fbm1, fbm2)

    def test_generate_fbm_seed_different_outputs(self, gen):
        size = (64, 64)
        H = 0.5
        fbm1 = gen.generate_fbm_region(H, size, seed=111)
        fbm2 = gen.generate_fbm_region(H, size, seed=222)

        assert not np.allclose(fbm1, fbm2)


class TestGenerateFBMImage:

    def test_generate_fbm_H_validation(self, gen):
        size = (64, 64)
        with pytest.raises(ValueError, match='H must be a list of length n_regions'):
            gen.generate_fbm_image(size, n_regions=3, H=[0.2, 0.8])  # Wrong length

    def test_generate_fbm_image_shapes(self, gen):
        size = (64, 64)
        n_regions = 4
        H = [0.2, 0.5, 0.8, 0.9]

        img, labels = gen.generate_fbm_image(size, n_regions, H, seed=42)

        assert img.shape == size
        assert labels.shape == size
        assert set(np.unique(labels)) <= set(range(n_regions))

    def test_generate_fbm_image_regions_have_different_statistics(self, gen):
        size = (128, 128)
        H = [0.1, 0.9]
        img, labels = gen.generate_fbm_image(size, n_regions=2, H=H, seed=10)
        grad_stats = []

        for i in range(2):
            mask = (labels == i)
            ys, xs = np.where(mask)

            # get bounding box
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            region = img[y0:y1 + 1, x0:x1 + 1]

            # compute gradient in 2D
            gy, gx = np.gradient(region.astype(float))
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            grad_stats.append(grad_mag.mean())

        assert grad_stats[0] > grad_stats[1]