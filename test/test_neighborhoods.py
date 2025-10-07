import numpy as np
import pytest

from segmentation.neighborhood import Neighborhood
from segmentation.utilities import make_shifts_from_mode, get_neighbor_stack

@pytest.fixture
def simple_image():
    X = np.array([
        [[1], [2]],
        [[3], [4]]
    ], dtype=float)
    return X

class TestMakeShiftsFromMode:

    def test_mode_4(self):
        shifts = make_shifts_from_mode('4')
        expected = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        assert set(shifts) == set(expected)
        assert len(shifts) == 4

    def test_mode_8(self):
        shifts = make_shifts_from_mode('8')
        expected = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        assert set(shifts) == set(expected)
        assert len(shifts) == 8

    def test_mode_radius_simple(self):
        shifts = make_shifts_from_mode('radius', radius=1)
        # The 4 points at distance 1. The center is excluded, of course.
        expected = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)])
        assert all(tuple(s) in map(tuple, expected) for s in shifts)
        assert len(shifts) == 4

    def test_mode_radius_2(self):
        shifts = make_shifts_from_mode('radius', radius=2)
        assert all((dy ** 2 + dx ** 2) <= 4 for dy, dx in shifts)

    def test_mode_k(self):
        shifts = make_shifts_from_mode('k', radius=3, k=5)
        dists = [dy ** 2 + dx ** 2 for dy, dx in shifts]
        assert dists == sorted(dists)
        assert len(shifts) == 5

    def test_mode_k_without_k_raises(self):
        with pytest.raises(ValueError):
            make_shifts_from_mode('k', radius=2)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_shifts_from_mode('banana')


class TestGetNeighborhoodStack:

    def test_stack_of_4_neighbors_shape(self, simple_image):
        neighborhood = Neighborhood('4')
        stack = get_neighbor_stack(simple_image, neighborhood)
        assert stack.shape == (2, 2, 4, 1)

    def test_stack_of_8_neighbors_shape(self, simple_image):
        neighborhood = Neighborhood('8')
        stack = get_neighbor_stack(simple_image, neighborhood)
        assert stack.shape == (2, 2, 8, 1)

    def test_get_stack_of_4_neighbors(self, simple_image):
        neighborhood = Neighborhood('4')
        stack = get_neighbor_stack(simple_image, neighborhood)

        # Expected neighbors (in this order)
        # shifts = [(-1,0), (0,-1), (0,1), (1,0)]
        # Verify some specific positions

        # Pixel (0,0) → up(3), left(2), right(2), down(3)
        expected_neighbors_00 = [3, 2, 2, 3]
        np.testing.assert_allclose(stack[0, 0, :, 0], expected_neighbors_00)

        # Pixel (1,1) → up(2), left(3), right(3), down(2)
        expected_neighbors_11 = [2, 3, 3, 2]
        np.testing.assert_allclose(stack[1, 1, :, 0], expected_neighbors_11)
