import numpy as np
import pytest

from segmentation.methods.ml_labeling import estimate_gaussian_distributions, compute_inverses_and_determinants, \
    log_likelihood


def generate_unidimensional_testcase():
    X = np.array([0.0, 5.0, 5.0, 15.0])
    labels = np.array([0, 0, 1, 1])
    n_classes = 2
    return X, labels, n_classes

def generate_multidimensional_testcase():
    # This has non-singular covariance
    X = np.array([
        [10.2, 1.0],  # class 0
        [0.2, 0.5],  # clase 0
        [0.2, 15.4],  # class 0
        [120.0, 120.0],  # class 1
        [90.5, 120.1],  # class 1
        [100.2, 90.7]  # class 1
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    n_classes = 2
    return X, labels, n_classes

@pytest.fixture
def simple_testcase_mean_var():
    X, labels, n_classes = generate_unidimensional_testcase()
    means, covs = estimate_gaussian_distributions(X, labels, n_classes)
    return means, covs

@pytest.fixture
def simple_testcase_inverses_logdets():
    X, labels, n_classes = generate_unidimensional_testcase()
    _, vars = estimate_gaussian_distributions(X, labels, n_classes)
    inv_vars, logdets = compute_inverses_and_determinants(vars)
    return inv_vars, logdets

@pytest.fixture
def two_channel_matrix():
    X, labels, n_classes = generate_multidimensional_testcase()
    return X, labels

@pytest.fixture
def two_channel_testcase_means_covs():
    X, labels, n_classes = generate_multidimensional_testcase()
    means, covs = estimate_gaussian_distributions(X, labels, n_classes)
    return means, covs

@pytest.fixture
def two_channel_testcase_inverses_logdets():
    X, labels, n_classes = generate_multidimensional_testcase()
    _, covs = estimate_gaussian_distributions(X, labels, n_classes)
    inv_covs, logdets = compute_inverses_and_determinants(covs)
    return inv_covs, logdets


class TestEstimateUnivariateGaussian:

    def test_estimate_univariate_gaussian_distributions_shapes(self, simple_testcase_mean_var):
        means, var = simple_testcase_mean_var
        assert np.array(means).shape == (2,) # 2 classes, each mean in R -> Shape (2,)
        assert np.array(var).shape == (2,) # 2 classes, each variance in R -> Shape (2,)

    def test_estimate_univariate_gaussian_distributions_means(self, simple_testcase_mean_var):
        means, _ = simple_testcase_mean_var
        expected_means = np.array([2.5, 10.0])
        np.testing.assert_allclose(means, expected_means, atol=1e-7)

    def test_estimate_univariate_gaussian_distributions_variances(self, simple_testcase_mean_var):
        _, vars = simple_testcase_mean_var
        expected_vars = [6.25, 25.0]
        np.testing.assert_allclose(vars, expected_vars, atol=1e-7)



class TestEstimateMultivariateGaussian:

    def test_estimate_multivariate_gaussian_distributions_shapes(self, two_channel_testcase_means_covs):
        means, covs = two_channel_testcase_means_covs
        assert np.array(means).shape == (2, 2) # 2 classes, each mean in R^2 -> Shape (2, 2)
        assert np.array(covs).shape == (2, 2, 2) # 2 classes, covariance matrix 2x2 -> Shape (2, 2, 2)

    def test_estimate_multivariate_gaussian_distributions_means(self, two_channel_testcase_means_covs):
        means, _ = two_channel_testcase_means_covs
        expected_means = np.array([[3.5333, 5.6333], [103.5666, 110.2666]])
        np.testing.assert_allclose(means, expected_means, atol=1e-4)

    def test_estimate_multivariate_gaussian_distributions_covariances(self, two_channel_testcase_means_covs):
        _, covs = two_channel_testcase_means_covs
        expected_cov0 = np.array([[22.2222, -15.4444], [-15.4444, 47.7355]])
        expected_cov1 = np.array([[150.7088, 32.4455], [32.4455, 191.4288]])
        np.testing.assert_allclose(covs[0], expected_cov0, atol=1e-4)
        np.testing.assert_allclose(covs[1], expected_cov1, atol=1e-4)


class TestComputeInversesAndDeterminants:

    def test_univariate_inverses_and_dets_shapes(self, simple_testcase_inverses_logdets):
        inverse_vars, logdets = simple_testcase_inverses_logdets
        assert inverse_vars.shape == (2, 1, 1)
        assert logdets.shape == (2,)

    def test_multivariate_inverses_and_dets_shapes(self, two_channel_testcase_inverses_logdets):
        inverse_covs, logdets = two_channel_testcase_inverses_logdets
        assert inverse_covs.shape == (2, 2, 2)
        assert logdets.shape == (2,)

    def test_verify_inverse_is_the_true_inverse(self, two_channel_testcase_means_covs, two_channel_testcase_inverses_logdets):
        _, covs = two_channel_testcase_means_covs
        inverse_covs, _ = two_channel_testcase_inverses_logdets
        for k in range(2):
            prod = inverse_covs[k] @ covs[k]
            np.testing.assert_allclose(prod, np.eye(2), atol=1e-3)

    def test_verify_dets_are_finite(self, two_channel_testcase_inverses_logdets):
        _, logdets = two_channel_testcase_inverses_logdets
        assert np.all(np.isfinite(logdets))

    def test_log_likelihood(self, two_channel_matrix, two_channel_testcase_means_covs, two_channel_testcase_inverses_logdets):
        X, labels = two_channel_matrix
        means, _ = two_channel_testcase_means_covs
        inv_covs, logdets = two_channel_testcase_inverses_logdets
        log_probs = log_likelihood(X, 2, means, inv_covs, logdets)
        assigned = np.argmax(log_probs, axis=1)
        np.testing.assert_array_equal(assigned, labels)
