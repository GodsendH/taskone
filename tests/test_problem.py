import numpy as np

from noisy_opt.problem import (
    constraints,
    heteroscedastic_sigma,
    is_feasible,
    project_to_feasible,
    sample_feasible,
    true_objective,
)


def test_true_objective_known_value():
    assert np.isclose(true_objective([1.0, 1.0]), 0.0)


def test_sigma_bounds_on_feasible_domain():
    rng = np.random.default_rng(123)
    points = sample_feasible(rng, 200)
    sigma = heteroscedastic_sigma(points)
    assert np.all(sigma >= 0.1)
    assert np.all(sigma < 0.3)


def test_projection_returns_feasible_point():
    projected = project_to_feasible([-3.0, 4.0])
    assert is_feasible(projected)
    assert np.all(projected >= 0.0)
    assert np.linalg.norm(projected) <= 1.0 + 1e-12


def test_constraints_sign_convention():
    assert np.all(constraints([0.5, 0.5]) <= 0.0)
    assert not is_feasible([1.0, 1.0])
