import numpy as np
import pytest

from segmentation.methods.nmc_labeling import compute_centroids


@pytest.fixture
def define_testcase():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [10.0, 11.0]
    ])

    labels = np.array([0, 0, 1, 1])
    n_classes = len(np.unique(labels))

    return X, labels, n_classes


class TestComputeCentroids:

    def test_compute_centroids_shape(self, define_testcase):
        X, labels, n_classes = define_testcase
        # Shape: (n_classes, n_features)
        centroids = compute_centroids(X, labels, n_classes)
        assert centroids.shape == (n_classes, X.shape[1])

    def test_compute_centroids(self, define_testcase):
        X, labels, n_classes = define_testcase
        centroids = compute_centroids(X, labels, n_classes)
        expected = np.array([
            [0.0, 0.5],
            [10.0, 10.5]
        ])
        np.testing.assert_allclose(centroids, expected, rtol=1e-7, atol=1e-9)
