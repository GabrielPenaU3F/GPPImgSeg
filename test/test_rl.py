import numpy as np
import pytest

from segmentation.methods.rl_labeler import RelaxationLabeler
from segmentation.neighborhood import Neighborhood


@pytest.fixture
def labeler():
    return RelaxationLabeler()

class TestCompatibilityMatrix:

    def test_compatibility_matrix_with_correct_dimensions_is_valid(self, labeler):
        n_classes = 3
        X = 2 * np.eye(n_classes) - np.ones((n_classes, n_classes))
        assert labeler.validate_compatibility_matrix(X, n_classes) == True

    def test_3x2_matrix_is_not_valid(self, labeler):
        X = np.ones((3, 2))
        with pytest.raises(ValueError,
                           match='The provided compatibility matrix is not adequate for this number of classes'):
            labeler.validate_compatibility_matrix(X, 1)

    def test_square_but_not_kxk_matrix_is_not_valid(self, labeler):
        X = np.ones((3, 3))
        n_classes = 4
        with pytest.raises(ValueError,
                           match='The provided compatibility matrix is not adequate for this number of classes'):
            labeler.validate_compatibility_matrix(X, n_classes)

    def test_relaxation_labeling_iterative_equivalence(self, labeler):
        # Imagen pequeña random con 3 clases
        h, w, k = 20, 20, 3
        probs0 = np.random.rand(h, w, k)
        probs0 /= probs0.sum(axis=2, keepdims=True)
        neighborhood = Neighborhood('8')

        # A) 50 iterations in one run
        probs_50 = labeler.label(
            probs0.copy(),
            neighborhood=neighborhood,
            n_iter=50,
            return_type='probs'
        )

        # B) 20 iterations, then 30 iterations from this result
        probs_20 = labeler.label(
            probs0.copy(),
            neighborhood=neighborhood,
            n_iter=20,
            return_type='probs'
        )

        probs_20_30 = labeler.label(
            probs_20,
            neighborhood=neighborhood,
            n_iter=30,
            return_type='probs'
        )

        # Must be almost identical
        np.testing.assert_allclose(probs_50, probs_20_30, rtol=1e-12)