"""Numerical helpers shared by solvers and schedulers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .model import FloatArray, ValidationError


@dataclass(frozen=True, slots=True)
class ColumnScaling:
    matrix: FloatArray
    scales: FloatArray


def normalize_columns(matrix: FloatArray, epsilon: float = 1e-12) -> ColumnScaling:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValidationError("matrix must be two-dimensional")
    scales = np.linalg.norm(arr, axis=0)
    scales = np.maximum(scales, epsilon)
    return ColumnScaling(arr / scales[None, :], scales)


def stable_top_k(scores: FloatArray, indices: NDArray[np.integer], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic descending top-k scores and corresponding indices."""

    values = np.asarray(scores, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or idx.ndim != 1 or values.size != idx.size:
        raise ValidationError("scores and indices must be same-length vectors")
    if k <= 0 or values.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    k = min(k, values.size)
    if k == values.size:
        candidate = np.arange(values.size)
    else:
        candidate = np.argpartition(values, values.size - k)[-k:]
    order = np.lexsort((idx[candidate], -values[candidate]))
    chosen = candidate[order]
    return idx[chosen], values[chosen]


def nonnegative_lstsq(matrix: FloatArray, target: FloatArray, ridge: float = 0.0) -> FloatArray:
    """Small nonnegative least-squares refit without a mandatory SciPy call.

    The unconstrained solution is projected and active-set-refit until stable.
    For the tiny supports used by OMP/OLS this is reliable and avoids making the
    core package depend on a particular SciPy optimisation API.
    """

    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.size:
        raise ValidationError("incompatible least-squares dimensions")
    n = a.shape[1]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    gram = a.T @ a
    if ridge > 0:
        gram = gram + ridge * np.eye(n)
    rhs = a.T @ b
    try:
        x = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(a, b, rcond=None)[0]
    active = x > 0
    for _ in range(n + 2):
        x = np.maximum(x, 0.0)
        new_active = x > 0
        if np.array_equal(active, new_active):
            break
        active = new_active
        if not np.any(active):
            return np.zeros(n, dtype=np.float64)
        sub = a[:, active]
        g = sub.T @ sub
        if ridge > 0:
            g = g + ridge * np.eye(g.shape[0])
        r = sub.T @ b
        try:
            solution = np.linalg.solve(g, r)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(sub, b, rcond=None)[0]
        x = np.zeros(n, dtype=np.float64)
        x[active] = solution
    return np.maximum(x, 0.0)


def conjugate_gradient(
    matvec: Callable[[FloatArray], FloatArray],
    rhs: FloatArray,
    initial: FloatArray | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[FloatArray, int, bool]:
    b = np.asarray(rhs, dtype=np.float64)
    x = np.zeros_like(b) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    residual = b - matvec(x)
    direction = residual.copy()
    squared = float(residual @ residual)
    threshold = tolerance * max(float(b @ b), 1.0)
    if squared <= threshold:
        return x, 0, True
    for iteration in range(1, max_iterations + 1):
        image = matvec(direction)
        denominator = float(direction @ image)
        if not math.isfinite(denominator) or denominator <= 1e-30:
            return x, iteration - 1, False
        alpha = squared / denominator
        x += alpha * direction
        residual -= alpha * image
        next_squared = float(residual @ residual)
        if next_squared <= threshold:
            return x, iteration, True
        beta = next_squared / max(squared, 1e-30)
        direction = residual + beta * direction
        squared = next_squared
    return x, max_iterations, False


def projected_gradient_pack_order(weights: FloatArray, ids: np.ndarray) -> np.ndarray:
    """Deterministic descending order, positive weights first."""
    weights = np.asarray(weights, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if weights.size != ids.size:
        raise ValidationError("weights and ids must have the same size")
    positive = weights > 0
    return np.lexsort((ids, ~positive, -weights))


def relative_change(current: FloatArray, previous: FloatArray, epsilon: float = 1e-12) -> float:
    return float(np.linalg.norm(current - previous) / max(np.linalg.norm(previous), epsilon))


def spectral_norm_squared(matrix: FloatArray) -> float:
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValidationError("matrix must be two-dimensional")
    # Compute through the smaller Gram matrix.
    if a.shape[0] <= a.shape[1]:
        gram = a @ a.T
    else:
        gram = a.T @ a
    return max(float(np.linalg.eigvalsh(gram)[-1]), 1e-15)


__all__ = [
    "ColumnScaling",
    "conjugate_gradient",
    "nonnegative_lstsq",
    "normalize_columns",
    "projected_gradient_pack_order",
    "relative_change",
    "spectral_norm_squared",
    "stable_top_k",
]
