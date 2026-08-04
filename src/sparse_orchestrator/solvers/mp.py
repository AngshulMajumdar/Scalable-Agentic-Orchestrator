"""Reference Matching Pursuit."""
from __future__ import annotations

import numpy as np

from .common import prepare_problem, restore_coefficients
from ..model import FloatArray, SolverResult


def matching_pursuit(
    matrix: FloatArray,
    target: FloatArray,
    *,
    max_atoms: int | None = None,
    tolerance: float = 1e-6,
    positive: bool = True,
    normalize: bool = True,
    allow_reselection: bool = True,
) -> SolverResult:
    """Solve a sparse approximation problem by classical Matching Pursuit.

    Unlike OMP, MP updates the residual with the newly selected atom only and
    does not refit earlier coefficients.  Reselection is enabled by default,
    which is part of the classical algorithm and useful when the dictionary is
    coherent.
    """

    problem = prepare_problem(matrix, target, normalize=normalize)
    a, b = problem.matrix, problem.target
    n_atoms = a.shape[1]
    budget = n_atoms if max_atoms is None else min(max_atoms, n_atoms)
    coefficients = np.zeros(n_atoms, dtype=np.float64)
    residual = b.copy()
    support_order: list[int] = []
    initial_norm = max(float(np.linalg.norm(b)), 1e-15)
    converged = False

    for iteration in range(1, budget + 1):
        correlations = a.T @ residual
        if positive:
            correlations = np.maximum(correlations, 0.0)
        if not allow_reselection and support_order:
            correlations[np.asarray(support_order, dtype=np.int64)] = -np.inf
        index = int(np.argmax(correlations))
        value = float(correlations[index])
        if not np.isfinite(value) or value <= tolerance * initial_norm:
            converged = True
            break
        atom = a[:, index]
        denominator = max(float(atom @ atom), 1e-15)
        step = value / denominator
        if positive:
            step = max(step, 0.0)
        coefficients[index] += step
        residual -= step * atom
        if index not in support_order:
            support_order.append(index)
        if np.linalg.norm(residual) <= tolerance * initial_norm:
            converged = True
            break
    else:
        iteration = budget

    restored = restore_coefficients(coefficients, problem.scales)
    original_residual = problem.target - problem.original_matrix @ restored
    support = np.flatnonzero(np.abs(restored) > 0).astype(np.int64)
    return SolverResult(
        coefficients=restored,
        support=support,
        residual=original_residual,
        objective=0.5 * float(original_residual @ original_residual),
        iterations=int(iteration if budget else 0),
        converged=converged,
        diagnostics={"selected_events": len(support_order), "allow_reselection": allow_reselection},
    )


__all__ = ["matching_pursuit"]
