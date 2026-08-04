"""Nonnegative FISTA for L1-regularised sparse approximation."""
from __future__ import annotations

import math

import numpy as np

from .common import prepare_problem, restore_coefficients
from ..math_utils import relative_change, spectral_norm_squared
from ..model import FloatArray, SolverResult


def fista_l1(
    matrix: FloatArray,
    target: FloatArray,
    *,
    l1_lambda: float = 0.025,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    positive: bool = True,
    normalize: bool = True,
    monotone_restart: bool = True,
    initial: FloatArray | None = None,
) -> SolverResult:
    """Minimise ``0.5 ||Ax-b||² + lambda ||x||_1`` with FISTA."""

    if l1_lambda < 0:
        raise ValueError("l1_lambda cannot be negative")
    problem = prepare_problem(matrix, target, normalize=normalize)
    a, b = problem.matrix, problem.target
    n_atoms = a.shape[1]
    lipschitz = spectral_norm_squared(a)
    step = 1.0 / lipschitz
    threshold = l1_lambda * step
    if initial is None:
        x = np.zeros(n_atoms, dtype=np.float64)
    else:
        x = np.asarray(initial, dtype=np.float64).copy()
        if x.size != n_atoms:
            raise ValueError("initial vector has the wrong size")
    if positive:
        x = np.maximum(x, 0.0)
    y = x.copy()
    momentum = 1.0
    previous_objective = np.inf
    converged = False
    restarts = 0

    def objective(z: np.ndarray) -> float:
        residual = a @ z - b
        return 0.5 * float(residual @ residual) + l1_lambda * float(np.sum(np.abs(z)))

    for iteration in range(1, max_iterations + 1):
        gradient = a.T @ (a @ y - b)
        proposal = y - step * gradient
        if positive:
            x_new = np.maximum(proposal - threshold, 0.0)
        else:
            x_new = np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0.0)
        current_objective = objective(x_new)
        if monotone_restart and current_objective > previous_objective + 1e-14:
            y = x.copy()
            momentum = 1.0
            gradient = a.T @ (a @ y - b)
            proposal = y - step * gradient
            if positive:
                x_new = np.maximum(proposal - threshold, 0.0)
            else:
                x_new = np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0.0)
            current_objective = objective(x_new)
            restarts += 1
        change = relative_change(x_new, x)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        y_new = x_new + ((momentum - 1.0) / next_momentum) * (x_new - x)
        # Gradient restart prevents momentum from moving against the proximal step.
        if monotone_restart and float((y_new - x_new) @ (x_new - x)) > 0:
            y_new = x_new.copy()
            next_momentum = 1.0
            restarts += 1
        x, y, momentum = x_new, y_new, next_momentum
        previous_objective = current_objective
        if change <= tolerance and iteration >= 3:
            converged = True
            break

    restored = restore_coefficients(x, problem.scales)
    residual = problem.target - problem.original_matrix @ restored
    support = np.flatnonzero(np.abs(restored) > tolerance).astype(np.int64)
    return SolverResult(
        coefficients=restored,
        support=support,
        residual=residual,
        objective=0.5 * float(residual @ residual) + l1_lambda * float(np.sum(np.abs(restored))),
        iterations=iteration,
        converged=converged,
        diagnostics={
            "restarts": restarts,
            "lipschitz": lipschitz,
            "l1_lambda": l1_lambda,
        },
    )


__all__ = ["fista_l1"]
