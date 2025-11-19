import numpy as np
import pytest

from segmentation.metrics import SegmentationComparator
from utilities.image_format_utilities import align_labels


@pytest.fixture
def comparator():
    return SegmentationComparator()

class TestRegionalMSE:

    def test_regional_mse_is_zero_when_identical_matrices(self, comparator):
        gt = np.array([[10, 10, 30, 30],
                       [10, 10, 30, 30]], dtype=float)

        pred = gt.copy()

        mse_region = comparator.regional_mse(gt, pred, return_type='region')
        mse_mean = comparator.regional_mse(gt, pred, return_type='mean')

        np.testing.assert_array_equal(mse_region, np.zeros_like(mse_region))
        np.testing.assert_array_equal(mse_mean, np.zeros_like(mse_mean))

    def test_a_three_region_image_must_return_three_region_mses(self, comparator):
        gt = np.array([
            [10, 10, 10, 10],
            [30, 30, 120, 120],
            [120, 120, 120, 120]
        ])
        pred = gt.copy() # Prediction here is irrelevant, we just check the shape

        mse_region = comparator.regional_mse(gt, pred, return_type='region')
        assert mse_region.shape == (3,)

    def test_regional_mse_affects_only_modified_region(self, comparator):
        gt = np.array([[0, 0, 100, 100],
                       [0, 0, 100, 100]], dtype=float)

        pred = gt.copy()
        pred[gt == 100] += 10

        mse_vals = comparator.regional_mse(gt, pred, return_type='region')

        assert mse_vals[0] == 0.0  # First region should have zero MSE
        assert mse_vals[1] == 100.0  # Second region must differ in 10^2

    def test_regional_mse_mean_is_the_mean_across_regions(self, comparator):
        gt = np.array([[0, 0, 100, 100]], dtype=float)
        pred = np.array([[0, 0, 110, 110]], dtype=float)

        region_vals = comparator.regional_mse(gt, pred, return_type='region')
        mean_val = comparator.regional_mse(gt, pred, return_type='mean')

        assert mean_val == region_vals.mean()


class TestBoundaryF1:

    def test_bf1_is_one_for_identical_matrices(self, comparator):
        gt = np.zeros((50, 50))
        gt[:, 25:] = 1
        pred = gt.copy()

        bf1 = comparator.boundary_f1(gt, pred)
        assert abs(bf1 - 1.0) < 1e-6

    def test_bf1_is_one_for_exactly_opposite_segmentation(self, comparator):
        gt = np.zeros((40, 40))
        gt[:, 20:] = 1
        pred = 1 - gt  # The exactly opposite binary matrix

        bf1 = comparator.boundary_f1(gt, pred)

        # Should not be sensitive to the value of the labels
        np.testing.assert_allclose(bf1, 1, rtol=1e-3)

    def test_bf1_improves_with_better_predictions(self, comparator):
        gt = np.zeros((60, 60))
        gt[:, 30:] = 1

        pred_bad = np.zeros_like(gt)
        pred_bad[:, 10:] = 1
        pred_good = np.zeros_like(gt)
        pred_good[:, 28:] = 1

        bf_bad = comparator.boundary_f1(gt, pred_bad)
        bf_good = comparator.boundary_f1(gt, pred_good)

        assert bf_good > bf_bad


class TestConfusionMatrix:

    def test_confusion_preserves_all_gt_classes_even_if_missing_in_pred(self, comparator):
        gt = np.array([0, 1, 2, 2])
        pred = np.array([0, 0, 0, 0])

        cm, labels = comparator.compute_confusion_matrix(gt, pred)

        expected = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ])

        assert np.array_equal(labels, np.array([0, 1, 2]))
        assert np.array_equal(cm, expected)

    def test_confusion_perfect_segmentation(self, comparator):
        gt = np.array([
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 2],
        ])
        pred = gt.copy()
        cm, labels = comparator.compute_confusion_matrix(gt, pred)

        expected = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [0, 0, 4],
        ])

        assert np.array_equal(labels, np.array([0, 1, 2]))
        assert np.array_equal(cm, expected)

    def test_confusion_inverted_labels(self, comparator):
        gt = np.array([
            [0, 0, 1],
            [1, 1, 0]
        ])
        pred = 1 - gt

        cm, labels = comparator.compute_confusion_matrix(gt, pred)
        expected = np.array([
            [3, 0],
            [0, 3],
        ])

        assert np.array_equal(labels, np.array([0, 1]))
        assert np.array_equal(cm, expected)

    def test_confusion_partial_mixing(self, comparator):
        # deliberately mix classes
        gt = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 1, 0, 1, 0, 1])

        cm, labels = comparator.compute_confusion_matrix(gt, pred)
        expected = np.array([
            [2, 1],  # gt=0 predicted as 0 twice, as 1 once
            [1, 2],  # gt=1 predicted as 1 twice, as 0 once
        ])

        assert np.array_equal(labels, np.array([0, 1]))
        assert np.array_equal(cm, expected)

    def test_confusion_missing_class_in_pred(self, comparator):
        # pred never predicts class 1
        gt = np.array([0, 1, 1, 0])
        pred = np.array([0, 0, 0, 0])

        cm, labels = comparator.compute_confusion_matrix(gt, pred)
        expected = np.array([
            [2, 0],  # gt=0 predicted as 0
            [2, 0],  # gt=1 predicted as 0
        ])

        assert np.array_equal(labels, np.array([0, 1]))
        assert np.array_equal(cm, expected)