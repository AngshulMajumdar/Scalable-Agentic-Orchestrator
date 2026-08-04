"""Reference Orthogonal Matching Pursuit."""
from __future__ import annotations

import numpy as np

from .common import prepare_problem, restore_coefficients
from ..math_utils import nonnegative_lstsq
from ..model import FloatArray, SolverResult


def orthogonal_matching_pursuit(
    matrix: FloatArray,
    target: FloatArray,
    *,
    max_atoms: int | None = None,
    tolerance: float = 1e-6,
    positive: bool = True,
    normalize: bool = True,
    ridge: float = 1e-12,
) -> SolverResult:
    """Classical OMP with a full support refit at each iteration."""

    problem = prepare_problem(matrix, target, normalize=normalize)
    a, b = problem.matrix, problem.target
    n_atoms = a.shape[1]
    budget = min(a.shape[0], n_atoms) if max_atoms is None else min(max_atoms, n_atoms)
    residual = b.copy()
    support: list[int] = []
    active_coefficients = np.empty(0, dtype=np.float64)
    initial_norm = max(float(np.linalg.norm(b)), 1e-15)
    converged = False

    for iteration in range(1, budget + 1):
        correlations = a.T @ residual
        if positive:
            correlations = np.maximum(correlations, 0.0)
        if support:
            correlations[np.asarray(support, dtype=np.int64)] = -np.inf
        index = int(np.argmax(correlations))
        best = float(correlations[index])
        if not np.isfinite(best) or best <= tolerance * initial_norm:
            converged = True
            break
        support.append(index)
        subdictionary = a[:, support]
        if positive:
            active_coefficients = nonnegative_lstsq(subdictionary, b, ridge=ridge)
        else:
            active_coefficients = np.linalg.lstsq(subdictionary, b, rcond=None)[0]
        residual = b - subdictionary @ active_coefficients
        if np.linalg.norm(residual) <= tolerance * initial_norm:
            converged = True
            break
    else:
        iteration = budget

    coefficients = np.zeros(n_atoms, dtype=np.float64)
    if support:
        coefficients[np.asarray(support, dtype=np.int64)] = active_coefficients
    restored = restore_coefficients(coefficients, problem.scales)
    original_residual = problem.target - problem.original_matrix @ restored
    final_support = np.flatnonzero(np.abs(restored) > 0).astype(np.int64)
    return SolverResult(
        coefficients=restored,
        support=final_support,
        residual=original_residual,
        objective=0.5 * float(original_residual @ original_residual),
        iterations=int(iteration if budget else 0),
        converged=converged,
        diagnostics={"candidate_support_size": len(support), "positive": positive},
    )


__all__ = ["orthogonal_matching_pursuit"]
