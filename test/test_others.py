import numpy as np
import pytest

from utilities.image_format_utilities import align_labels


class TestAlignLabels:

    def test_align_identity(self):
        gt = np.array([
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 2]
        ])
        pred = gt.copy()
        aligned = align_labels(pred, gt, return_type='labels')

        np.testing.assert_array_equal(aligned, gt)

    def test_align_permutation(self):
        gt = np.array([
            [0, 1, 2],
            [0, 1, 2],
        ])
        pred = np.array([
            [2, 0, 1],  # cyclic rotation
            [2, 0, 1],
        ])
        aligned = align_labels(pred, gt, return_type='labels')

        np.testing.assert_array_equal(aligned, gt)

    def test_align_reverse_labels(self):
        gt = np.array([
            [0, 0, 1, 1],
        ])
        pred = 1 - gt  # Inverted labels
        aligned = align_labels(pred, gt, return_type='labels')

        np.testing.assert_array_equal(aligned, gt)

    def test_0_255_output(self):
        gt = np.array([
            [0, 1, 2]
        ])
        pred = np.array([
            [2, 0, 1]
        ])

        aligned = align_labels(pred, gt, return_type='img')
        expected = np.array([[0, 127, 254]])

        np.testing.assert_array_equal(aligned, expected)

    def test_align_single_class(self):
        gt = np.zeros((4, 4), dtype=int)
        pred = np.zeros_like(gt)

        aligned = align_labels(pred, gt, return_type='labels')
        np.testing.assert_array_equal(aligned, gt)

    def test_handles_missing_class(self):
        gt = np.array([
            [0, 1],
        ])
        pred = np.array([
            [0, 0],
        ])
        aligned = align_labels(pred, gt, return_type='labels')

        # Must map to the optimal class in ground-truth
        assert np.all(aligned == 0)