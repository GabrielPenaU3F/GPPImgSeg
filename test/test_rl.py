import numpy as np
import pytest

from segmentation.methods.rl_labeler import RelaxationLabeler


class TestCompatibilityMatrix:

    def test_compatibility_matrix_with_correct_dimensions_is_valid(self):
        n_classes = 3
        X = 2 * np.eye(n_classes) - np.ones((n_classes, n_classes))
        assert RelaxationLabeler().validate_compatibility_matrix(X, n_classes) == True

    def test_3x2_matrix_is_not_valid(self):
        X = np.ones((3, 2))
        with pytest.raises(ValueError,
                           match='The provided compatibility matrix is not adequate for this number of classes'):
            RelaxationLabeler().validate_compatibility_matrix(X, 1)

    def test_square_but_not_kxk_matrix_is_not_valid(self):
        X = np.ones((3, 3))
        n_classes = 4
        with pytest.raises(ValueError,
                           match='The provided compatibility matrix is not adequate for this number of classes'):
            RelaxationLabeler().validate_compatibility_matrix(X, n_classes)