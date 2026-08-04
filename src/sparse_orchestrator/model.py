"""Core data structures for sparse agent orchestration.

The package deliberately separates agent data, provider capacity, scheduling
policy, and simulation state.  The benchmark can therefore scale to one
million explicit agents without changing the mathematical objects used by the
small reference solvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


class ValidationError(ValueError):
    """Raised when an experiment object is internally inconsistent."""


def _as_float_matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be a two-dimensional array; got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError(f"{name} must have positive dimensions; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    return arr


def _as_float_vector(value: np.ndarray | Sequence[float], name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional; got shape {arr.shape}")
    if arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    return arr


@dataclass(frozen=True, slots=True)
class Provider:
    """Finite provider capacity shared by all running agents.

    Parameters
    ----------
    capacity:
        Positive capacity in each resource dimension.
    resource_names:
        Optional labels such as ``("cpu", "memory", "network", "gpu")``.
    name:
        Human-readable provider identifier used in reports.
    """

    capacity: FloatArray
    resource_names: tuple[str, ...] = ()
    name: str = "provider"

    def __post_init__(self) -> None:
        cap = _as_float_vector(self.capacity, "capacity")
        if np.any(cap <= 0):
            raise ValidationError("every provider capacity must be strictly positive")
        object.__setattr__(self, "capacity", cap)
        if self.resource_names:
            if len(self.resource_names) != cap.size:
                raise ValidationError(
                    "resource_names length must match capacity dimensions: "
                    f"{len(self.resource_names)} != {cap.size}"
                )
            if len(set(self.resource_names)) != len(self.resource_names):
                raise ValidationError("resource_names must be unique")
        else:
            object.__setattr__(
                self,
                "resource_names",
                tuple(f"resource_{i}" for i in range(cap.size)),
            )

    @property
    def n_resources(self) -> int:
        return int(self.capacity.size)

    def normalize(self, demand: FloatArray) -> FloatArray:
        """Normalize one demand vector or a demand matrix by provider capacity."""
        arr = np.asarray(demand, dtype=np.float64)
        if arr.shape[-1] != self.n_resources:
            raise ValidationError(
                f"last demand dimension {arr.shape[-1]} does not match provider "
                f"dimension {self.n_resources}"
            )
        return arr / self.capacity


@dataclass(slots=True)
class AgentSet:
    """Explicit agent records.

    One million agents means one million rows in ``demands``.  There is no
    profile-count compression in this object.  Arrays may be ordinary NumPy
    arrays or memory-mapped arrays.
    """

    demands: FloatArray
    durations: FloatArray
    ids: IntArray | None = None
    arrival_order: IntArray | None = None
    priorities: FloatArray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        demands = np.asanyarray(self.demands)
        if demands.ndim != 2:
            raise ValidationError(f"demands must be 2-D; got {demands.shape}")
        if demands.shape[0] == 0 or demands.shape[1] == 0:
            raise ValidationError("demands must contain at least one agent and one resource")
        if not np.issubdtype(demands.dtype, np.floating):
            demands = demands.astype(np.float32)
        if np.any(~np.isfinite(demands)) or np.any(demands <= 0):
            raise ValidationError("all demands must be finite and strictly positive")
        self.demands = demands

        durations = np.asanyarray(self.durations)
        if durations.ndim != 1 or durations.size != demands.shape[0]:
            raise ValidationError(
                "durations must be a vector with one entry per agent; "
                f"got {durations.shape} for {demands.shape[0]} agents"
            )
        if not np.issubdtype(durations.dtype, np.floating):
            durations = durations.astype(np.float32)
        if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
            raise ValidationError("all durations must be finite and strictly positive")
        self.durations = durations

        n = demands.shape[0]
        if self.ids is None:
            self.ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(self.ids)
            if ids.ndim != 1 or ids.size != n:
                raise ValidationError("ids must contain one value per agent")
            if len(np.unique(ids)) != n:
                raise ValidationError("agent ids must be unique")
            self.ids = ids.astype(np.int64, copy=False)

        if self.arrival_order is None:
            self.arrival_order = np.arange(n, dtype=np.int64)
        else:
            order = np.asarray(self.arrival_order)
            if order.ndim != 1 or order.size != n:
                raise ValidationError("arrival_order must be a permutation of all row indices")
            if order.min(initial=0) < 0 or order.max(initial=-1) >= n:
                raise ValidationError("arrival_order contains an out-of-range index")
            if np.unique(order).size != n:
                raise ValidationError("arrival_order must not repeat an index")
            self.arrival_order = order.astype(np.int64, copy=False)

        if self.priorities is not None:
            priorities = np.asarray(self.priorities, dtype=np.float64)
            if priorities.ndim != 1 or priorities.size != n:
                raise ValidationError("priorities must contain one value per agent")
            if np.any(~np.isfinite(priorities)):
                raise ValidationError("priorities contains NaN or infinite values")
            self.priorities = priorities

    @property
    def n_agents(self) -> int:
        return int(self.demands.shape[0])

    @property
    def n_resources(self) -> int:
        return int(self.demands.shape[1])

    @property
    def unit_duration(self) -> bool:
        first = float(self.durations[0])
        return bool(np.allclose(self.durations, first, rtol=0.0, atol=1e-12))

    def validate_against(self, provider: Provider) -> None:
        if self.n_resources != provider.n_resources:
            raise ValidationError(
                f"agents use {self.n_resources} resources but provider has "
                f"{provider.n_resources}"
            )
        oversize = np.any(self.demands > provider.capacity[None, :], axis=1)
        if np.any(oversize):
            rows = np.flatnonzero(oversize)[:10].tolist()
            raise ValidationError(
                "at least one agent cannot fit on an empty provider; "
                f"first offending rows: {rows}"
            )

    def subset(self, rows: IntArray | Sequence[int]) -> "AgentSet":
        idx = np.asarray(rows, dtype=np.int64)
        priorities = None if self.priorities is None else self.priorities[idx]
        return AgentSet(
            demands=self.demands[idx],
            durations=self.durations[idx],
            ids=self.ids[idx],
            arrival_order=np.arange(idx.size, dtype=np.int64),
            priorities=priorities,
            metadata=dict(self.metadata),
        )

    def iter_chunks(self, chunk_size: int, rows: IntArray | None = None) -> Iterator[tuple[IntArray, FloatArray]]:
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be positive")
        if rows is None:
            for start in range(0, self.n_agents, chunk_size):
                stop = min(start + chunk_size, self.n_agents)
                idx = np.arange(start, stop, dtype=np.int64)
                yield idx, self.demands[start:stop]
        else:
            rows = np.asarray(rows, dtype=np.int64)
            for start in range(0, rows.size, chunk_size):
                idx = rows[start : start + chunk_size]
                yield idx, self.demands[idx]


@dataclass(frozen=True, slots=True)
class DispatchBatch:
    """A feasible set of agents launched at the same simulation instant."""

    indices: IntArray
    used: FloatArray
    remaining: FloatArray
    selector_time_s: float
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = np.asarray(self.indices, dtype=np.int64)
        used = _as_float_vector(self.used, "used")
        remaining = _as_float_vector(self.remaining, "remaining")
        if used.shape != remaining.shape:
            raise ValidationError("used and remaining must have identical shapes")
        if np.any(used < -1e-10) or np.any(remaining < -1e-8):
            raise ValidationError("dispatch resources cannot be negative")
        object.__setattr__(self, "indices", idx)
        object.__setattr__(self, "used", used)
        object.__setattr__(self, "remaining", remaining)


@dataclass(slots=True)
class ScheduleTrace:
    """Optional per-agent trace.

    Full traces consume substantial memory at one million agents.  The
    simulator therefore allocates them only when requested.
    """

    start_times: FloatArray
    finish_times: FloatArray
    wave_ids: IntArray

    @classmethod
    def allocate(cls, n_agents: int) -> "ScheduleTrace":
        return cls(
            start_times=np.full(n_agents, np.nan, dtype=np.float64),
            finish_times=np.full(n_agents, np.nan, dtype=np.float64),
            wave_ids=np.full(n_agents, -1, dtype=np.int32),
        )

    def validate_complete(self) -> None:
        if np.any(~np.isfinite(self.start_times)):
            raise ValidationError("trace contains agents without start times")
        if np.any(~np.isfinite(self.finish_times)):
            raise ValidationError("trace contains agents without finish times")
        if np.any(self.finish_times < self.start_times):
            raise ValidationError("trace contains a finish before a start")
        if np.any(self.wave_ids < 0):
            raise ValidationError("trace contains agents without wave identifiers")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    method: str
    n_agents: int
    n_resources: int
    makespan: float
    lower_bound: float
    normalized_makespan: float
    scheduler_time_s: float
    wall_time_s: float
    dispatches: int
    mean_utilization: float
    min_utilization: float
    p95_utilization: float
    completed: int
    valid: bool
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)
    trace: ScheduleTrace | None = None

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "method": self.method,
            "n_agents": self.n_agents,
            "n_resources": self.n_resources,
            "makespan": self.makespan,
            "lower_bound": self.lower_bound,
            "normalized_makespan": self.normalized_makespan,
            "scheduler_time_s": self.scheduler_time_s,
            "wall_time_s": self.wall_time_s,
            "dispatches": self.dispatches,
            "mean_utilization": self.mean_utilization,
            "min_utilization": self.min_utilization,
            "p95_utilization": self.p95_utilization,
            "completed": self.completed,
            "valid": self.valid,
        }
        row.update({f"diagnostic_{k}": v for k, v in self.diagnostics.items()})
        return row


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Result returned by a sparse reference solver."""

    coefficients: FloatArray
    support: IntArray
    residual: FloatArray
    objective: float
    iterations: int
    converged: bool
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", np.asarray(self.coefficients, dtype=np.float64))
        object.__setattr__(self, "support", np.asarray(self.support, dtype=np.int64))
        object.__setattr__(self, "residual", np.asarray(self.residual, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    """Filesystem locations produced by a benchmark run."""

    root: Path
    raw_csv: Path
    summary_csv: Path
    metadata_json: Path
    report_md: Path


__all__ = [
    "AgentSet",
    "BoolArray",
    "DispatchBatch",
    "ExperimentArtifacts",
    "FloatArray",
    "IntArray",
    "Provider",
    "ScheduleTrace",
    "SimulationResult",
    "SolverResult",
    "ValidationError",
]
