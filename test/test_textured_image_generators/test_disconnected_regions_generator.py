import numpy as np
import pytest

from synthesizers.disconnected_image_generator import DisconnectedRegionsImageGenerator
from scipy.ndimage import label


@pytest.fixture
def generator():
    return DisconnectedRegionsImageGenerator()

class TestMaskCreation:

    def test_mask_encodes_classes(self, generator):
        classes = np.sort([0, 1, 2, 3, 4])
        mask = np.sort(np.unique(generator.generate_mask(size=(10, 10), n_regions=5)))
        np.testing.assert_array_equal(classes, mask)

    def test_spatial_variability(self, generator):
        mask = generator.generate_mask((100, 100))

        # Mask should have some spatial variability
        assert mask.std() > 0

    def test_perlin_regions_are_not_connected(self, generator):
        mask = generator.generate_mask((100,100), n_regions=2)
        unique_labels = np.unique(mask)
        # For each class, we should have at least three disconnected components
        for lbl in unique_labels:
            labeled, n = label(mask == lbl)
            assert n >= 3

    def test_perlin_region_distribution_not_collapsed(self, generator):
        mask = generator.generate_mask((200, 200), n_regions=2)
        counts = np.array([(mask == lbl).sum() for lbl in np.unique(mask)])

        # No region should fill over 80% of the image
        assert np.all(counts < 0.8 * mask.size)
