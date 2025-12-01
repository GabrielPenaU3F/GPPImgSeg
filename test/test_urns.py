import numpy as np
import pytest

from segmentation.methods.urn_labelers import GPPLabeler, PolyaLabeler
from segmentation.neighborhood import Neighborhood
from utilities.segmentation_utilities import initialize_urns


@pytest.fixture
def multipixel_probs():
    probs = np.array([
        [[0.5, 0.5],
         [0.8, 0.2]]
    ])  # Shape (1, 2, 2)
    return probs

@pytest.fixture
def test_urn_array_1():
    urns = np.array([
        [[2, 1],
         [0, 3]]
    ], dtype=int)  # (1,2,2)
    sampled_classes = np.array([[0, 1]])
    return urns, sampled_classes

@pytest.fixture
def simple_2x2_tracking():
    probs = np.array([
        [[0.8, 0.2], [0.7, 0.3]],
        [[0.3, 0.7], [0.1, 0.9]]
    ], dtype=float)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs

@pytest.fixture
def reinforcement_minimal_testcase():
    urn = np.array([[[10, 10]]], dtype=float)  # shape (1,1,2)
    sampled = np.array([[0]])                  # shape (1,1)
    R = np.array([[5, -3],
                  [-2, 7]], dtype=float)
    return urn, sampled, R

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


class TestUrnArguments:

    def test_input_type_is_not_correct(self):
        labeler = GPPLabeler()
        with pytest.raises(ValueError, match="input_type must be 'probs' or 'urns'"):
            labeler.label(np.zeros((2, 2, 2)),
                          Neighborhood('4'),
                          initial_total_balls=10,
                          R=np.eye(2, 2),
                          input_type=1)

    def test_output_type_is_not_correct(self):
        labeler = GPPLabeler()
        with pytest.raises(ValueError, match="return_type must be 'img', 'probs', or 'urns'"):
            labeler.label(np.zeros((2, 2, 2)),
                          Neighborhood('4'),
                          initial_total_balls=10,
                          R=np.eye(2, 2),
                          return_type=1)


class TestValidateReinforcementMatrix:

    def test_R_function_must_return_matrix(self):
        labeler = GPPLabeler()

        # The function must return a matrix
        with pytest.raises(ValueError, match='Reinforcement matrix shape is not correct'):
            labeler.validate_R(lambda x: 123, 3)

    def test_R_must_be_a_square_matrix(self):
        labeler = GPPLabeler()
        bad_R = np.array([[1, 2, 3], [1, 2, 3]])

        with pytest.raises(ValueError, match='Reinforcement matrix shape is not correct'):
            labeler.validate_R(bad_R, 3)

    def test_R_must_only_contain_integers(self):
        labeler = GPPLabeler()
        bad_R = 0.5 * np.eye(3)

        with pytest.raises(ValueError, match='Reinforcement matrix must only contain integers'):
            labeler.validate_R(bad_R, 3)

    def test_validate_reinforcement_matrix_raises_error_if_rows_dont_sum_equal(self):
        labeler = GPPLabeler()
        bad_R = np.array([
            [2, 0, 0],
            [1, 1, 0],
            [0, 0, 3]  # This one sums 3
        ])
        with pytest.raises(ValueError, match='All rows must sum to the same total'):
            labeler.validate_R(bad_R, 3)

class TestPolyaUrnUpdate:

    def test_update_empty_urns(self):
        urns = np.zeros((2, 2, 3), dtype=int)
        sampled_classes = np.array([
            [0, 1],
            [2, 0]
        ])
        delta = 2
        updated = PolyaLabeler().update_urns(urns.copy(), sampled_classes, delta)

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
        updated = PolyaLabeler().update_urns(urns.copy(), sampled_classes, delta)

        # We added +1 ball to the chosen class
        expected = np.array([
            [[1, 1], [1, 1]],
            [[3, 2], [3, 4]]
        ])
        np.testing.assert_array_equal(updated, expected)


class TestGPPUrnUpdate:

    def test_update_urns_vectorial_delta_uniform(self):
        urns = np.ones((2, 2, 3), dtype=int)
        sampled_classes = np.array([
            [0, 1],
            [2, 0]
        ])
        R = np.ones((3, 3)) # We add 1 ball of each color regardless of the result
        updated = GPPLabeler().update_urns(urns, sampled_classes, R)
        expected = 2 * np.ones((2, 2, 3), dtype=int)
        np.testing.assert_array_equal(updated, expected)

    def test_update_urns_vectorial_delta_nonuniform(self):
        urns = np.ones((2, 2, 3), dtype=int)
        sampled_classes = np.array([
            [0, 1],
            [2, 0]
        ])
        R = np.array([[1, 2, 1], [0, 0, 4], [2, 2, 0]])
        updated = GPPLabeler().update_urns(urns, sampled_classes, R)
        expected = np.array([ [ [2, 3, 2] ],
                              [ [1, 1, 5] ],
                              [ [3, 3, 1] ],
                              [ [2, 3, 2] ]
                            ], dtype=int).reshape(2, 2, 3)
        np.testing.assert_array_equal(updated, expected)

    def test_update_urns_vectorial_polya_type(self):
        urns = np.ones((2, 2, 3), dtype=int)
        sampled_classes = np.array([
            [0, 1],
            [2, 0]
        ])
        R = np.eye(3) # Polya reinforcement matrix is the identity
        updated = GPPLabeler().update_urns(urns, sampled_classes, R)
        expected = np.array([ [ [2, 1, 1] ],
                              [ [1, 2, 1] ],
                              [ [1, 1, 2] ],
                              [ [2, 1, 1] ]
                            ], dtype=int).reshape(2, 2, 3)
        np.testing.assert_array_equal(updated, expected)

    def test_update_urns_negative_reinforcement_yields_no_negatives(self, test_urn_array_1):
        urns, sampled_classes = test_urn_array_1
        R = np.array([[2, -6], [2, -6]])  # remove 6 of the other color
        updated = GPPLabeler().update_urns(urns, sampled_classes, R)
        assert np.all(updated >= 0)


    def test_update_urns_negative_reinforcement_works_correctly(self, test_urn_array_1):
        urns, sampled_classes = test_urn_array_1
        R = np.array([[2, -1], [2, -1]])  # remove 1 of the other color
        updated = GPPLabeler().update_urns(urns, sampled_classes, R)
        # [2,1] and class=0 → +2,-1 → [4,0]
        # [0,3] and class=1 → +2,-1 → [2,2]
        expected = np.array([[[4, 0], [2, 2]]])
        np.testing.assert_array_equal(updated, expected)

    def test_urns_iterative_equivalence(self):
        h, w, k = 20, 20, 3
        probs0 = np.random.rand(h, w, k)
        probs0 /= probs0.sum(axis=2, keepdims=True)

        neighborhood = Neighborhood('8')
        initial_total_balls = 50
        R = np.eye(k)
        labeler_1 = GPPLabeler(seed=42)
        labeler_2 = GPPLabeler(seed=42)

        # A) 50 iterations in one run
        urns_50 = labeler_1.label(
            input=probs0.copy(),
            input_type='probs',
            neighborhood=neighborhood,
            initial_total_balls=initial_total_balls,
            R=R,
            n_iter=50,
            return_type='urns',
            verbose=False
        )

        # B) 20 iterations, then 30, preserving urns
        urns_20 = labeler_2.label(
            input=probs0.copy(),
            input_type='probs',
            neighborhood=neighborhood,
            initial_total_balls=initial_total_balls,
            R=R,
            n_iter=20,
            return_type='urns',
            verbose=False
        )

        urns_20_30 = labeler_2.label(
            input=urns_20,  # << URNS, not probs
            input_type='urns',
            neighborhood=neighborhood,
            initial_total_balls=None,
            R=R,
            n_iter=30,
            return_type='urns',
            verbose=False
        )

        np.testing.assert_allclose(urns_50, urns_20_30, rtol=1e-12)

    def test_urns_reinitializing_from_probs_fails(self):
        h, w, k = 20, 20, 3
        probs0 = np.random.rand(h, w, k)
        probs0 /= probs0.sum(axis=2, keepdims=True)

        neighborhood = Neighborhood('8')
        initial_total_balls = 50
        R = np.eye(k)
        labeler_1 = GPPLabeler(seed=42)
        labeler_2 = GPPLabeler(seed=42)

        urns_50 = labeler_1.label(
            input=probs0.copy(),
            input_type='probs',
            neighborhood=neighborhood,
            initial_total_balls=initial_total_balls,
            R=R,
            n_iter=50,
            return_type='urns',
            verbose=False
        )

        # 20 iterations → probs
        probs_20 = labeler_2.label(
            input=probs0.copy(),
            input_type='probs',
            neighborhood=neighborhood,
            initial_total_balls=initial_total_balls,
            R=R,
            n_iter=20,
            return_type='probs',
            verbose=False
        )

        # Reinitialize from probs should produce a different state
        urns_reinit = labeler_2.label(
            input=probs_20,
            input_type='probs',  # << reinicio incorrecto
            neighborhood=neighborhood,
            initial_total_balls=initial_total_balls,
            R=R,
            n_iter=30,
            return_type='urns',
            verbose=False
        )

        # They should be significantly different
        diff = np.abs(urns_50 - urns_reinit).mean()
        assert diff > 1.0, "Reinitializing urns from probs should break equivalence"

    def test_urns_never_empty(self):
        """
        If reinforcement produces a fully empty urn, the algorithm must ensure
        that the urn contains at least 1 ball in total.
        """

        urns = np.array([[[1, 0, 0]]], dtype=int)
        sampled_classes = np.array([[1]])
        R = np.array([  # Extreme negative reinforcement
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1]
        ])

        labeler = GPPLabeler()
        updated = labeler.update_urns(urns, sampled_classes, R)

        # Urn should be empty, but the algorithm must prevent this
        # Our choice is to force a 'minimal uniform state' in this case: 1 ball in each class
        np.testing.assert_array_equal(updated, np.ones_like(urns))


class TestReinforcementScheme:

    def test_R_applied_correctly(self, reinforcement_minimal_testcase):
        urn, sampled, R = reinforcement_minimal_testcase
        labeler = GPPLabeler()
        updated = labeler.update_urns(urn, sampled, R)[0, 0]

        assert np.allclose(updated, np.array([15, 7]))

    def test_R_accumulates_over_iterations(self, reinforcement_minimal_testcase):
        urn, _, R = reinforcement_minimal_testcase
        sampled_classes = [0, 0]
        labeler = GPPLabeler()

        for s in sampled_classes:
            urn = labeler.update_urns(urn, np.array([[s]]), R)

        expected = np.array([[[20, 4]]])
        assert np.allclose(urn, expected)

    def test_R_row_selection(self, reinforcement_minimal_testcase):
        urn, _, _ = reinforcement_minimal_testcase
        R = np.array([[+10, -1],
                      [-20, +3]])
        labeler = GPPLabeler()

        # Si sampleamos clase 1, debe sumar R[1]
        updated = labeler.update_urns(urn, np.array([[1]]), R)[0, 0]

        expected = np.array([0, 10 + 3])  # [-10, 13]

        assert np.allclose(updated, expected), \
            f"Expected [-10,13], got {updated}"

    def test_fixed_R(self, simple_2x2_tracking):
        labeler = GPPLabeler()
        R = np.eye(2)

        urns = labeler.label(
            simple_2x2_tracking,
            neighborhood=Neighborhood('4'),
            initial_total_balls=10,
            R=R,
            n_iter=5,
            input_type='probs',
            return_type='urns',
        )

        # Exact expected urn
        expected_urns = np.array([
            [[8, 7], [9, 6]],
            [[5, 10], [5, 10]]
        ])

        assert np.array_equal(urns, expected_urns)

    def test_R_as_function_called_each_iteration(self, simple_2x2_tracking):

        # When R is a function, it should be called once per iteration
        labeler = GPPLabeler()
        n_iter = 15
        calls = []

        def R_func(n):
            calls.append(n)
            return np.eye(2)

        labeler.label(
            simple_2x2_tracking,
            neighborhood=Neighborhood('4'),
            initial_total_balls=5,
            R=R_func,
            n_iter=n_iter,
            input_type='probs',
            return_type='probs'
        )

        # plus a -1 for the validation procedure
        assert calls == [-1] + list(range(n_iter))

    def test_schedule_changes_results(self):

        seed = 1
        rng = np.random.default_rng(seed)
        h, w, k = 15, 15, 2
        probs = rng.random((h, w, k))
        probs /= probs.sum(axis=2, keepdims=True)
        labeler = GPPLabeler()

        def R_polya_then_soft(n):
            if n < 10:
                return np.eye(2)  # aggresive
            else:
                return np.array([[2, 1], [1, 2]])  # soft

        out_fixed = labeler.label(
            probs, Neighborhood('4'), 10, np.eye(2), 20,
            input_type='probs', return_type='probs'
        )

        out_sched = labeler.label(
            probs, Neighborhood('4'), 10, R_polya_then_soft, 20,
            input_type='probs', return_type='probs'
        )

        # Should not be equal
        assert not np.allclose(out_fixed, out_sched)