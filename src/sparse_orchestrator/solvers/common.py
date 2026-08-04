"""Shared sparse-solver preparation and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..math_utils import ColumnScaling, normalize_columns
from ..model import FloatArray, SolverResult, ValidationError


@dataclass(frozen=True, slots=True)
class PreparedProblem:
    matrix: FloatArray
    target: FloatArray
    original_matrix: FloatArray
    scales: FloatArray


def prepare_problem(
    matrix: FloatArray,
    target: FloatArray,
    *,
    normalize: bool,
) -> PreparedProblem:
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.ndim != 2:
        raise ValidationError("dictionary matrix must be two-dimensional")
    if b.ndim != 1:
        raise ValidationError("target must be one-dimensional")
    if a.shape[0] != b.size:
        raise ValidationError(
            f"dictionary rows {a.shape[0]} do not match target size {b.size}"
        )
    if a.shape[1] == 0:
        raise ValidationError("dictionary must contain at least one atom")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValidationError("dictionary and target must be finite")
    if normalize:
        scaled = normalize_columns(a)
        return PreparedProblem(scaled.matrix, b, a, scaled.scales)
    return PreparedProblem(a, b, a, np.ones(a.shape[1], dtype=np.float64))


def restore_coefficients(coefficients: FloatArray, scales: FloatArray) -> FloatArray:
    x = np.asarray(coefficients, dtype=np.float64)
    return x / scales


def objective_l2(matrix: FloatArray, target: FloatArray, coefficients: FloatArray) -> float:
    residual = target - matrix @ coefficients
    return 0.5 * float(residual @ residual)


def empty_result(target: FloatArray, n_atoms: int = 0) -> SolverResult:
    return SolverResult(
        coefficients=np.zeros(n_atoms, dtype=np.float64),
        support=np.empty(0, dtype=np.int64),
        residual=np.asarray(target, dtype=np.float64).copy(),
        objective=0.5 * float(np.asarray(target) @ np.asarray(target)),
        iterations=0,
        converged=True,
    )


__all__ = [
    "PreparedProblem",
    "empty_result",
    "objective_l2",
    "prepare_problem",
    "restore_coefficients",
]
