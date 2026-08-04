"""Chunked candidate scoring functions."""
from __future__ import annotations

import numpy as np

from .model import FloatArray, ValidationError


def _normalized(demands: FloatArray, capacity: FloatArray) -> np.ndarray:
    d = np.asarray(demands, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    if d.ndim != 2 or cap.ndim != 1 or d.shape[1] != cap.size:
        raise ValidationError("incompatible demand and capacity dimensions")
    return d / cap[None, :]


def feasible_mask(demands: FloatArray, remaining: FloatArray, tolerance: float = 1e-10) -> np.ndarray:
    return np.all(np.asarray(demands) <= np.asarray(remaining)[None, :] + tolerance, axis=1)


def score_candidates(
    method: str,
    demands: FloatArray,
    remaining: FloatArray,
    capacity: FloatArray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Fast map-stage proxy used to construct a bounded candidate pool.

    The exact sparse solver is run only after the reducer has formed the global
    pool.  These scores are therefore retrieval proxies, not substitutes for
    MP, OMP, OLS, FISTA, or IRLS.
    """

    a = _normalized(demands, capacity)
    target = np.asarray(remaining, dtype=np.float64) / np.asarray(capacity, dtype=np.float64)
    correlation = a @ target
    norm2 = np.einsum("ij,ij->i", a, a)
    total = np.sum(a, axis=1)
    spread = np.std(a, axis=1)

    if method == "mp":
        return correlation
    if method == "omp":
        return correlation / np.sqrt(norm2 + epsilon)
    if method == "ols":
        positive = np.maximum(correlation, 0.0)
        return positive * positive / (norm2 + epsilon)
    if method == "fista":
        # Correlation minus a mild concentration penalty gives FISTA a useful
        # warm pool without solving the proximal problem during map.
        return correlation - 0.02 * total - 0.02 * spread
    if method == "irls":
        return correlation / np.power(norm2 + epsilon, 0.25) - 0.015 * spread
    if method in {"fifo", "windowed_fifo", "kahn", "langchain_policy", "langgraph_policy"}:
        return -np.arange(a.shape[0], dtype=np.float64)
    raise ValidationError(f"unknown scoring method: {method}")


def direction_affinity(
    candidates: FloatArray,
    directions: FloatArray,
    direction_weights: FloatArray,
    capacity: FloatArray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Score all pool candidates by affinity to sparse selected directions."""

    if directions.size == 0:
        return np.zeros(np.asarray(candidates).shape[0], dtype=np.float64)
    c = _normalized(candidates, capacity)
    d = _normalized(directions, capacity)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), epsilon)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), epsilon)
    weights = np.asarray(direction_weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    weights /= np.sum(weights)
    similarities = c @ d.T
    return similarities @ weights


__all__ = ["direction_affinity", "feasible_mask", "score_candidates"]
