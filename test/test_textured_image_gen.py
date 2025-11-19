import numpy as np
import pytest

from synthesizers.textured_image_generator import generate_connected_masks, generate_textured_region, \
    generate_textured_image


@pytest.fixture
def define_testcase():
    return None


class TestMaskCreation:

    def test_mask_encodes_classes(self):
        classes = np.sort([0, 1, 2])
        mask = np.sort(np.unique(generate_connected_masks(size=(4, 4), n_regions=3)))
        np.testing.assert_array_equal(classes, mask)


class TestTextureCreation:

    def test_texture_shape_is_correct(self):
        tex = generate_textured_region(size=(32, 16), smoothness=1.0, seed=42)
        assert tex.shape == (32, 16)

    def test_texture_same_seed_generates_same_texture(self):
        t1 = generate_textured_region(size=(32, 32), smoothness=1.0, seed=123)
        t2 = generate_textured_region(size=(32, 32), smoothness=1.0, seed=123)
        np.testing.assert_allclose(t1, t2)

    def test_different_seeds_generate_different_textures(self):
        t1 = generate_textured_region(size=(32, 32), smoothness=1.0, seed=1)
        t2 = generate_textured_region(size=(32, 32), smoothness=1.0, seed=2)
        # They should differ in many pixels
        assert not np.allclose(t1, t2)

    def test_smoothness_effect(self):
        """Higher smoothness should produce a smoother (less variant) texture."""
        t_low = generate_textured_region(size=(32, 32), smoothness=0.2, seed=42)
        t_high = generate_textured_region(size=(32, 32), smoothness=5.0, seed=42)
        # High smoothness => lower standard deviation
        assert t_high.std() < t_low.std()


class TestTexturedImageGeneration:

    def test_intensity_must_be_doubled(self):
        img_1, _ = generate_textured_image(size=(32, 32), n_regions=1, smoothness=0.2,
                                           intensity=1.0, base_intensities = [127], seed=42)
        img_2, _ = generate_textured_image(size=(32, 32), n_regions=1, smoothness=0.2,
                                           intensity=2.0, base_intensities = [127], seed=42)
        ground_truth = 127 * np.ones_like(img_1)
        tex = 2 * (img_1 - ground_truth)
        tex_actual = img_2 - ground_truth
        np.testing.assert_allclose(tex_actual, tex, rtol=1e-3)

    def test_ground_truth_and_image_coincide_when_intensity_is_zero(self):
        img, ground_truth = generate_textured_image(size=(32, 32), n_regions=2, smoothness=0.2, intensity=0)
        np.testing.assert_array_equal(img, ground_truth)
