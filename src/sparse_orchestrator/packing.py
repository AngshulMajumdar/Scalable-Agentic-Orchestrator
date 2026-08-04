"""Deterministic multidimensional packing kernels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray, IntArray, ValidationError

try:  # pragma: no cover - import availability depends on installation extras
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


@dataclass(frozen=True, slots=True)
class PackResult:
    selected_positions: IntArray
    remaining: FloatArray
    used: FloatArray
    considered: int
    rejected: int


def _pack_python(
    demands: np.ndarray,
    order: np.ndarray,
    capacity: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    remaining = capacity.astype(np.float64, copy=True)
    selected: list[int] = []
    rejected = 0
    considered = 0
    for position in order:
        considered += 1
        row = demands[int(position)]
        if np.all(row <= remaining + tolerance):
            selected.append(int(position))
            remaining -= row
        else:
            rejected += 1
    return np.asarray(selected, dtype=np.int64), remaining, considered, rejected


if njit is not None:

    @njit(cache=True)
    def _pack_numba(
        demands: np.ndarray,
        order: np.ndarray,
        capacity: np.ndarray,
        tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        remaining = capacity.copy()
        selected = np.empty(order.size, dtype=np.int64)
        selected_count = 0
        rejected = 0
        considered = 0
        for q in range(order.size):
            position = int(order[q])
            considered += 1
            feasible = True
            for resource in range(demands.shape[1]):
                if demands[position, resource] > remaining[resource] + tolerance:
                    feasible = False
                    break
            if feasible:
                selected[selected_count] = position
                selected_count += 1
                for resource in range(demands.shape[1]):
                    remaining[resource] -= demands[position, resource]
            else:
                rejected += 1
        return selected[:selected_count], remaining, considered, rejected

else:
    _pack_numba = None


def greedy_pack(
    demands: FloatArray,
    order: IntArray,
    capacity: FloatArray,
    *,
    tolerance: float = 1e-10,
    use_numba: bool = True,
) -> PackResult:
    """Pack candidates in a supplied priority order.

    The routine never changes the order.  This is important because the sparse
    solver owns the policy while the packer owns only feasibility.
    """

    d = np.asarray(demands)
    order_arr = np.asarray(order, dtype=np.int64)
    cap = np.asarray(capacity, dtype=np.float64)
    if d.ndim != 2 or cap.ndim != 1 or d.shape[1] != cap.size:
        raise ValidationError("incompatible packing dimensions")
    if order_arr.ndim != 1:
        raise ValidationError("order must be one-dimensional")
    if order_arr.size and (
        order_arr.min(initial=0) < 0 or order_arr.max(initial=-1) >= d.shape[0]
    ):
        raise ValidationError("order contains an invalid candidate position")
    if tolerance < 0:
        raise ValidationError("tolerance cannot be negative")
    if _pack_numba is not None and use_numba:
        selected, remaining, considered, rejected = _pack_numba(
            np.asarray(d, dtype=np.float64), order_arr, cap, tolerance
        )
    else:
        selected, remaining, considered, rejected = _pack_python(
            np.asarray(d, dtype=np.float64), order_arr, cap, tolerance
        )
    used = cap - remaining
    return PackResult(selected, remaining, used, considered, rejected)


def first_fit_window(
    demands: FloatArray,
    positions: IntArray,
    capacity: FloatArray,
    *,
    stop_at_first_failure: bool,
    tolerance: float = 1e-10,
) -> PackResult:
    """FCFS packer with either strict or scan-ahead behaviour."""

    d = np.asarray(demands)
    positions = np.asarray(positions, dtype=np.int64)
    cap = np.asarray(capacity, dtype=np.float64)
    remaining = cap.copy()
    selected: list[int] = []
    rejected = 0
    considered = 0
    for position in positions:
        considered += 1
        row = d[int(position)]
        if np.all(row <= remaining + tolerance):
            selected.append(int(position))
            remaining -= row
        else:
            rejected += 1
            if stop_at_first_failure:
                break
    selected_arr = np.asarray(selected, dtype=np.int64)
    return PackResult(selected_arr, remaining, cap - remaining, considered, rejected)


def best_fit_refinement(
    demands: FloatArray,
    initial_order: IntArray,
    capacity: FloatArray,
    *,
    passes: int = 2,
    tolerance: float = 1e-10,
) -> PackResult:
    """Pack then revisit rejected candidates using residual-sensitive ordering."""

    d = np.asarray(demands, dtype=np.float64)
    order = np.asarray(initial_order, dtype=np.int64)
    cap = np.asarray(capacity, dtype=np.float64)
    if passes <= 0:
        raise ValidationError("passes must be positive")
    remaining = cap.copy()
    selected_mask = np.zeros(d.shape[0], dtype=np.bool_)
    considered = 0
    rejected_total = 0
    pending = order.copy()
    for _ in range(passes):
        if pending.size == 0:
            break
        scores = (d[pending] / np.maximum(remaining[None, :], tolerance)).sum(axis=1)
        # Prefer candidates that use scarce residual resources without exceeding them.
        pass_order = pending[np.argsort(-scores, kind="mergesort")]
        next_pending: list[int] = []
        for position in pass_order:
            considered += 1
            row = d[int(position)]
            if np.all(row <= remaining + tolerance):
                selected_mask[int(position)] = True
                remaining -= row
            else:
                rejected_total += 1
                next_pending.append(int(position))
        pending = np.asarray(next_pending, dtype=np.int64)
    selected = np.flatnonzero(selected_mask).astype(np.int64)
    # Preserve the original policy order in the returned selection.
    rank = np.full(d.shape[0], order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    selected = selected[np.argsort(rank[selected], kind="mergesort")]
    return PackResult(selected, remaining, cap - remaining, considered, rejected_total)


__all__ = ["PackResult", "best_fit_refinement", "first_fit_window", "greedy_pack"]
