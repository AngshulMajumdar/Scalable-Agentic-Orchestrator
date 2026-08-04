"""Lower bounds, utilisation summaries, and schedule checks."""
from __future__ import annotations

import math

import numpy as np

from .model import AgentSet, FloatArray, Provider, ValidationError


def resource_time_lower_bound(agents: AgentSet, provider: Provider) -> float:
    """Continuous lower bound on makespan under renewable resources."""

    work = np.sum(
        np.asarray(agents.demands, dtype=np.float64)
        * np.asarray(agents.durations, dtype=np.float64)[:, None],
        axis=0,
    )
    resource_bound = float(np.max(work / provider.capacity))
    longest_agent = float(np.max(agents.durations))
    return max(resource_bound, longest_agent)


def unit_wave_lower_bound(agents: AgentSet, provider: Provider) -> int:
    if not agents.unit_duration:
        raise ValidationError("unit_wave_lower_bound requires equal durations")
    work = np.sum(np.asarray(agents.demands, dtype=np.float64), axis=0)
    waves = int(math.ceil(float(np.max(work / provider.capacity)) - 1e-12))
    return max(waves, 1)


def utilization_summary(
    usage_integral: FloatArray,
    provider: Provider,
    makespan: float,
) -> tuple[float, float, float, np.ndarray]:
    if makespan <= 0:
        raise ValidationError("makespan must be positive")
    utilization = np.asarray(usage_integral, dtype=np.float64) / (
        provider.capacity * makespan
    )
    utilization = np.clip(utilization, 0.0, 1.0 + 1e-8)
    return (
        float(np.mean(utilization)),
        float(np.min(utilization)),
        float(np.quantile(utilization, 0.95)),
        utilization,
    )


def verify_dispatch(
    demands: FloatArray,
    indices: np.ndarray,
    available: FloatArray,
    tolerance: float = 1e-8,
) -> None:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return
    used = np.sum(np.asarray(demands)[idx], axis=0, dtype=np.float64)
    if np.any(used > np.asarray(available, dtype=np.float64) + tolerance):
        raise ValidationError(
            f"dispatch exceeds available capacity: used={used}, available={available}"
        )


__all__ = [
    "resource_time_lower_bound",
    "unit_wave_lower_bound",
    "utilization_summary",
    "verify_dispatch",
]
