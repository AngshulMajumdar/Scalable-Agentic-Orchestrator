from __future__ import annotations

import numpy as np
import pytest

from sparse_orchestrator.solvers import (
    fista_l1,
    irls_lp,
    matching_pursuit,
    orthogonal_least_squares,
    orthogonal_matching_pursuit,
)


def identity_problem() -> tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(6)
    target = np.array([1.0, 0.0, 2.0, 0.0, 0.5, 0.0])
    return matrix, target


@pytest.mark.parametrize(
    "solver",
    [
        lambda a, b: matching_pursuit(a, b, max_atoms=6, tolerance=1e-10),
        lambda a, b: orthogonal_matching_pursuit(a, b, max_atoms=6, tolerance=1e-10),
        lambda a, b: orthogonal_least_squares(a, b, max_atoms=6, tolerance=1e-10),
    ],
)
def test_greedy_solvers_recover_identity(solver) -> None:
    matrix, target = identity_problem()
    result = solver(matrix, target)
    np.testing.assert_allclose(matrix @ result.coefficients, target, atol=1e-8)
    assert set(result.support.tolist()) == {0, 2, 4}
    assert result.objective < 1e-12


def test_fista_recovers_sparse_nonnegative_signal() -> None:
    matrix, target = identity_problem()
    result = fista_l1(
        matrix,
        target,
        l1_lambda=1e-5,
        max_iterations=500,
        tolerance=1e-10,
        monotone_restart=True,
    )
    np.testing.assert_allclose(result.coefficients, target, atol=2e-4)
    assert set(result.support.tolist()) == {0, 2, 4}


def test_irls_recovers_sparse_nonnegative_signal() -> None:
    matrix, target = identity_problem()
    result = irls_lp(
        matrix,
        target,
        p=0.5,
        regularization=1e-6,
        epsilon=1e-6,
        outer_iterations=12,
        inner_iterations=64,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(result.coefficients, target, atol=1e-3)
    assert set(result.support.tolist()) == {0, 2, 4}


def test_omp_refits_correlated_atoms() -> None:
    matrix = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.0, 0.1, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    target = np.array([1.0, 1.0, 0.0])
    result = orthogonal_matching_pursuit(matrix, target, max_atoms=2, tolerance=1e-12)
    assert result.support.size == 2
    assert np.linalg.norm(result.residual) < 1e-7


def test_ols_minimizes_post_refit_loss() -> None:
    matrix = np.array([[1.0, 0.8, 0.0], [0.0, 0.6, 1.0]])
    target = np.array([1.0, 1.0])
    result = orthogonal_least_squares(matrix, target, max_atoms=2, tolerance=1e-12)
    assert np.linalg.norm(result.residual) < 1e-7
    assert result.support.size == 2
