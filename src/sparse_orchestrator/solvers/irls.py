"""Nonnegative IRLS for sparse Lp approximation."""
from __future__ import annotations

import numpy as np

from .common import prepare_problem, restore_coefficients
from ..math_utils import conjugate_gradient, relative_change
from ..model import FloatArray, SolverResult


def irls_lp(
    matrix: FloatArray,
    target: FloatArray,
    *,
    p: float = 0.5,
    regularization: float = 0.025,
    epsilon: float = 1e-3,
    outer_iterations: int = 8,
    inner_iterations: int = 32,
    tolerance: float = 1e-6,
    positive: bool = True,
    normalize: bool = True,
    initial: FloatArray | None = None,
) -> SolverResult:
    """Solve a smoothed Lp-regularised least-squares problem by IRLS.

    The linear systems are solved by inexact conjugate gradient, which is the
    implementation used by the distributed backend as well.
    """

    if not (0 < p <= 1):
        raise ValueError("p must lie in (0, 1]")
    if regularization < 0 or epsilon <= 0:
        raise ValueError("regularization and epsilon are invalid")
    problem = prepare_problem(matrix, target, normalize=normalize)
    a, b = problem.matrix, problem.target
    n_atoms = a.shape[1]
    rhs = a.T @ b
    if initial is None:
        x = np.maximum(rhs, 0.0) if positive else rhs.copy()
        scale = max(float(np.max(np.abs(x))), 1.0)
        x /= scale
    else:
        x = np.asarray(initial, dtype=np.float64).copy()
        if x.size != n_atoms:
            raise ValueError("initial vector has the wrong size")
    converged = False
    cg_iterations = 0
    cg_converged = 0

    for outer in range(1, outer_iterations + 1):
        continuation_epsilon = max(epsilon, 0.25 * (0.5 ** (outer - 1)))
        weights = np.power(x * x + continuation_epsilon**2, 0.5 * p - 1.0)
        diagonal = regularization * (p / 2.0) * weights

        def matvec(vector: np.ndarray) -> np.ndarray:
            return a.T @ (a @ vector) + diagonal * vector

        cap = min(inner_iterations, max(2, 2 ** outer))
        x_new, used, solved = conjugate_gradient(
            matvec,
            rhs,
            initial=x,
            max_iterations=cap,
            tolerance=tolerance * 0.1,
        )
        cg_iterations += used
        cg_converged += int(solved)
        if positive:
            x_new = np.maximum(x_new, 0.0)
        change = relative_change(x_new, x)
        x = x_new
        if change <= tolerance and outer >= 3:
            converged = True
            break

    restored = restore_coefficients(x, problem.scales)
    residual = problem.target - problem.original_matrix @ restored
    penalty = regularization * float(
        np.sum(np.power(restored * restored + epsilon * epsilon, 0.5 * p))
    )
    support = np.flatnonzero(np.abs(restored) > tolerance).astype(np.int64)
    return SolverResult(
        coefficients=restored,
        support=support,
        residual=residual,
        objective=0.5 * float(residual @ residual) + penalty,
        iterations=outer,
        converged=converged,
        diagnostics={
            "cg_iterations": cg_iterations,
            "cg_converged_systems": cg_converged,
            "p": p,
            "regularization": regularization,
        },
    )


__all__ = ["irls_lp"]
