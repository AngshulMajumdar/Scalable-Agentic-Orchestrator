"""Reference Orthogonal Least Squares."""
from __future__ import annotations

import numpy as np

from .common import prepare_problem, restore_coefficients
from ..math_utils import nonnegative_lstsq
from ..model import FloatArray, SolverResult


def _candidate_residuals_vectorized(
    dictionary: FloatArray,
    target: FloatArray,
    support: list[int],
    candidates: np.ndarray,
    positive: bool,
    ridge: float,
) -> tuple[int, np.ndarray, float]:
    """Evaluate exact post-refit residuals.

    Candidate pools in the production scheduler are intentionally bounded, so a
    clear exact implementation is preferable to a difficult-to-audit shortcut.
    """

    best_index = -1
    best_coefficients = np.empty(0, dtype=np.float64)
    best_loss = np.inf
    for candidate in candidates:
        trial_support = support + [int(candidate)]
        sub = dictionary[:, trial_support]
        if positive:
            coefficients = nonnegative_lstsq(sub, target, ridge=ridge)
        else:
            coefficients = np.linalg.lstsq(sub, target, rcond=None)[0]
        residual = target - sub @ coefficients
        loss = float(residual @ residual)
        if loss < best_loss - 1e-15 or (
            abs(loss - best_loss) <= 1e-15 and int(candidate) < best_index
        ):
            best_index = int(candidate)
            best_coefficients = coefficients
            best_loss = loss
    return best_index, best_coefficients, best_loss


def orthogonal_least_squares(
    matrix: FloatArray,
    target: FloatArray,
    *,
    max_atoms: int | None = None,
    tolerance: float = 1e-6,
    positive: bool = True,
    normalize: bool = True,
    ridge: float = 1e-12,
) -> SolverResult:
    """Select the atom that minimises the residual after orthogonal refitting."""

    problem = prepare_problem(matrix, target, normalize=normalize)
    a, b = problem.matrix, problem.target
    n_atoms = a.shape[1]
    budget = min(a.shape[0], n_atoms) if max_atoms is None else min(max_atoms, n_atoms)
    support: list[int] = []
    active_coefficients = np.empty(0, dtype=np.float64)
    residual = b.copy()
    initial_norm = max(float(np.linalg.norm(b)), 1e-15)
    converged = False

    for iteration in range(1, budget + 1):
        excluded = np.ones(n_atoms, dtype=np.bool_)
        if support:
            excluded[np.asarray(support, dtype=np.int64)] = False
        candidates = np.flatnonzero(excluded)
        index, trial_coefficients, loss = _candidate_residuals_vectorized(
            a, b, support, candidates, positive, ridge
        )
        if index < 0:
            converged = True
            break
        current_loss = float(residual @ residual)
        improvement = current_loss - loss
        if improvement <= tolerance * max(current_loss, 1.0):
            converged = True
            break
        support.append(index)
        active_coefficients = trial_coefficients
        residual = b - a[:, support] @ active_coefficients
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


__all__ = ["orthogonal_least_squares"]
