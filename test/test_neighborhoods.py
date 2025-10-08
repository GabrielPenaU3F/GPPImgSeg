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

@pytest.fixture
def simple_image_3x3():
    # Simple 3x3 image
    # 1 2 3
    # 4 5 6
    # 7 8 9
    return np.arange(1, 10).reshape(3, 3, 1)

@pytest.fixture
def simple_image_5x5():
    return np.arange(1, 26).reshape(5, 5, 1)

def euclidean_distance(dx, dy):
    return np.sqrt(dx**2 + dy**2)


class TestMakeShiftsFromMode:

    def test_mode_4(self):
        shifts = make_shifts_from_mode('4', distance_fn=euclidean_distance)
        expected = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        assert set(shifts) == set(expected)
        assert len(shifts) == 4

    def test_mode_8(self):
        shifts = make_shifts_from_mode('8', distance_fn=euclidean_distance)
        expected = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        assert set(shifts) == set(expected)
        assert len(shifts) == 8

    def test_mode_radius_simple(self):
        shifts = make_shifts_from_mode('radius', radius=1, distance_fn=euclidean_distance)
        # The 4 points at distance 1. The center is excluded, of course.
        expected = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)])
        assert all(tuple(s) in map(tuple, expected) for s in shifts)
        assert len(shifts) == 4

    def test_mode_radius_2(self):
        shifts = make_shifts_from_mode('radius', radius=2, distance_fn=euclidean_distance)
        assert all((dy ** 2 + dx ** 2) <= 4 for dy, dx in shifts)

    def test_mode_k(self):
        shifts = make_shifts_from_mode('k', radius=3, k=5, distance_fn=euclidean_distance)
        dists = [dy ** 2 + dx ** 2 for dy, dx in shifts]
        assert dists == sorted(dists)
        assert len(shifts) == 5

    def test_mode_k_without_k_raises(self):
        with pytest.raises(ValueError):
            make_shifts_from_mode('k', radius=2, distance_fn=euclidean_distance)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_shifts_from_mode('banana')

    def test_make_shifts_radius_manhattan(self):
        d_manhattan = lambda dy, dx: abs(dy) + abs(dx)
        shifts = make_shifts_from_mode('radius', radius=2, distance_fn=d_manhattan)
        # |dy| + |dx| ≤ 2
        expected = [
            (-2, 0), (-1, -1), (-1, 0), (-1, 1),
            (0, -2), (0, -1), (0, 1), (0, 2),
            (1, -1), (1, 0), (1, 1),
            (2, 0)
        ]
        assert all(s in expected for s in shifts)
        assert len(shifts) == len(expected)


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

def test_get_neighbor_stack_radius_1_center(simple_image_3x3):
    neighborhood = Neighborhood('radius', radius=1)
    stack = get_neighbor_stack(simple_image_3x3, neighborhood)
    expected_center_neighbors = [2, 4, 6, 8]
    assert stack.shape == (3, 3, 4, 1)
    np.testing.assert_allclose(sorted(stack[1, 1, :, 0]), sorted(expected_center_neighbors))

def test_get_neighbor_stack_radius_1_corner(simple_image_3x3):
    neighborhood = Neighborhood('radius', radius=1)
    stack = get_neighbor_stack(simple_image_3x3, neighborhood)
    # We check the reflect on the (0,0) corner
    expected_corner_neighbors = [4, 2, 2, 4]
    np.testing.assert_allclose(sorted(stack[0, 0, :, 0]), sorted(expected_corner_neighbors))

def test_get_neighbor_stack_radius_2_shapes(simple_image_5x5):
    neighborhood = Neighborhood('radius', radius=2)
    shifts = make_shifts_from_mode('radius', radius=2, distance_fn=euclidean_distance)
    stack = get_neighbor_stack(simple_image_5x5, neighborhood)
    h, w, k = simple_image_5x5.shape
    assert stack.shape == (h, w, len(shifts), k)

def test_get_neighbor_stack_radius_2_center(simple_image_5x5):
    neighborhood = Neighborhood('radius', radius=2)
    shifts = make_shifts_from_mode('radius', radius=2, distance_fn=euclidean_distance)
    stack = get_neighbor_stack(simple_image_5x5, neighborhood)

    # Expected neighbors
    max_dy = max(abs(dy) for dy, dx in shifts)
    max_dx = max(abs(dx) for dy, dx in shifts)
    padded = np.pad(simple_image_5x5, ((max_dy, max_dy), (max_dx, max_dx), (0, 0)), mode='reflect')

    # Center of the (padded) image
    cy, cx = 2 + max_dy, 2 + max_dx
    expected_center_neighbors = [int(padded[cy + dy, cx + dx, 0]) for dy, dx in shifts]
    np.testing.assert_allclose(sorted(stack[2, 2, :, 0]), sorted(expected_center_neighbors))

def test_get_neighbor_stack_radius_2_corner(simple_image_5x5):
    neighborhood = Neighborhood('radius', radius=2)
    shifts = make_shifts_from_mode('radius', radius=2, distance_fn=euclidean_distance)
    stack = get_neighbor_stack(simple_image_5x5, neighborhood)

    # Expected neighbors
    max_dy = max(abs(dy) for dy, dx in shifts)
    max_dx = max(abs(dx) for dy, dx in shifts)
    padded = np.pad(simple_image_5x5, ((max_dy, max_dy), (max_dx, max_dx), (0, 0)), mode='reflect')

    # Top-left corner
    cy0, cx0 = 0 + max_dy, 0 + max_dx
    expected_corner_neighbors = [int(padded[cy0 + dy, cx0 + dx, 0]) for dy, dx in shifts]
    np.testing.assert_allclose(sorted(stack[0, 0, :, 0]), sorted(expected_corner_neighbors))
