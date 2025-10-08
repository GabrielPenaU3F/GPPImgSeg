import numpy as np
import pytest

from segmentation.methods.urn_labeling import update_urns
from segmentation.utilities import initialize_urns


@pytest.fixture
def multipixel_probs():
    probs = np.array([
        [[0.5, 0.5],
         [0.8, 0.2]]
    ])  # Shape (1, 2, 2)
    return probs

class TestUrnInicialization:

    def test_initialize_urns_total_balls_preserved(self):
        """Check that each urn (pixel) sums to n_balls."""
        probs = np.array([
            [[0.2, 0.3, 0.5],
             [0.1, 0.7, 0.2]]
        ])  # Shape (1, 2, 3)
        n_balls = 100
        urns = initialize_urns(probs, n_balls)
        assert np.allclose(urns.sum(axis=2), n_balls)

    def test_initialize_urns_proportionally_allocated(self):
        """Urns should roughly reflect the probability proportions."""
        probs = np.array([[[0.2, 0.3, 0.5]]])  # Shape (1,1,3)
        n_balls = 10
        urns = initialize_urns(probs, n_balls)
        counts = urns[0, 0]
        # Since we have 10 balls, and perfect probabilities, we must have
        # 10 * [0.2, 0.3, 0.5] = [2, 3, 5]
        assert set(counts) == {2, 3, 5}

    def test_initialize_urns_multiple_pixels_each_has_10_balls(self, multipixel_probs):
        """Handle multiple pixels independently."""
        probs = multipixel_probs
        n_balls = 10
        urns = initialize_urns(probs, n_balls)
        assert np.all(urns.sum(axis=2) == 10)

    def test_initialize_urns_multiple_pixels_correct_proportions(self, multipixel_probs):
        """Handle multiple pixels independently."""
        probs = multipixel_probs
        n_balls = 10
        urns = initialize_urns(probs, n_balls)
        assert abs(urns[0, 0, 0] - urns[0, 0, 1]) <= 1  # Pixel 1 - approxiimately uniform
        assert urns[0, 1, 0] > urns[0, 1, 1] # Pixel 2 - biased towards first class

    def test_initialize_urns_single_pixel_deficit_assignment(self):
        """Edge case: single pixel should still distribute remaining balls properly."""
        probs = np.array([[[0.3333, 0.3333, 0.3333]]])  # (1,1,3)
        n_balls = 10
        urns = initialize_urns(probs, n_balls)
        # Should distribute all 10 balls approximately evenly
        assert urns.sum() == n_balls
        assert urns.max() - urns.min() <= 1


class TestPolyaUrnUpdate:

    def test_update_empty_urns(self):
        urns = np.zeros((2, 2, 3), dtype=int)
        sampled_classes = np.array([
            [0, 1],
            [2, 0]
        ])
        delta = 2
        updated = update_urns(urns.copy(), sampled_classes, delta)

        # In each position, 2 balls of the corresponding class have been added
        assert updated[0, 0, 0] == 2  # Class 0
        assert updated[0, 1, 1] == 2  # Class 1
        assert updated[1, 0, 2] == 2  # Class 2
        assert updated[1, 1, 0] == 2  # Class 0
        # Everything else is stll zero
        assert np.sum(updated) == 8

    def test_update_nonempty_urns(self):
        urns = np.array([
            [[1, 0], [0, 1]],
            [[2, 2], [3, 3]]
        ])
        sampled_classes = np.array([
            [1, 0],
            [0, 1]
        ])
        delta = 1
        updated = update_urns(urns.copy(), sampled_classes, delta)

        # We added +1 ball to the chosen class
        expected = np.array([
            [[1, 1], [1, 1]],
            [[3, 2], [3, 4]]
        ])
        np.testing.assert_array_equal(updated, expected)
