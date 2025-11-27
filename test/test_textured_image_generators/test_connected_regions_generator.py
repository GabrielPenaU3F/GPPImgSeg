import numpy as np
import pytest

from synthesizers.connected_image_generator import ConnectedRegionsImageGenerator


@pytest.fixture
def generator():
    return ConnectedRegionsImageGenerator()

class TestMaskCreation:

    def test_mask_encodes_classes(self, generator):
        classes = np.sort([0, 1, 2])
        mask = np.sort(np.unique(generator.generate_mask(size=(4, 4), n_regions=3)))
        np.testing.assert_array_equal(classes, mask)

    def test_spatial_variability(self, generator):
        mask = generator.generate_mask((100, 100), n_regions=2)

        # Mask should have some spatial variability (the regions have different intensities)
        assert mask.std() > 0

    def test_mask_is_constant_over_regions(self, generator):
        mask = generator.generate_mask(size=(100, 100), n_regions=4).reshape(100, 100)
        # Each quadrant should be constant
        q1, q2, q3, q4 = mask[:50,:50], mask[:50,50:], mask[50:,:50], mask[50:,50:]
        assert q1.std() == 0
        assert q2.std() == 0
        assert q3.std() == 0
        assert q4.std() == 0
