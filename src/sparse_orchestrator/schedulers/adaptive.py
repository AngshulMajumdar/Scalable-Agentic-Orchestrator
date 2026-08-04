"""Adaptive gating between windowed FIFO and sparse MP scheduling."""
from __future__ import annotations

import numpy as np

from ..config import CandidateConfig, SolverConfig
from ..distributed import CandidateBackend, LocalBackend
from ..model import AgentSet, DispatchBatch, FloatArray, Provider, ValidationError
from ..storage import ActiveSet
from .base import BatchScheduler
from .fifo import FIFOScheduler
from .pursuit import MPScheduler


def _unit_directions(demands: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    normalized = np.asarray(demands, dtype=np.float64) / capacity[None, :]
    norms = np.linalg.norm(normalized, axis=1)
    norms = np.maximum(norms, np.finfo(np.float64).tiny)
    return normalized / norms[:, None]


def order_correlation_excess(
    agents: AgentSet,
    provider: Provider,
    *,
    sample_size: int = 50_000,
    seed: int = 0,
) -> float:
    """Estimate excess adjacent cosine similarity over random-pair similarity.

    The statistic is computed once per workload.  It is intentionally based on
    explicit rows and the declared arrival order.  No profile-count compression
    is used.
    """
    if sample_size < 2:
        raise ValidationError("adaptive sample_size must be at least two")
    n = agents.n_agents
    m = min(sample_size, n)
    order = np.asarray(agents.arrival_order[:m], dtype=np.int64)
    adjacent = _unit_directions(agents.demands[order], provider.capacity)
    c_adj = float(np.mean(np.einsum("ij,ij->i", adjacent[:-1], adjacent[1:])))

    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=m, dtype=np.int64)
    right = rng.integers(0, n, size=m, dtype=np.int64)
    y_left = _unit_directions(agents.demands[left], provider.capacity)
    y_right = _unit_directions(agents.demands[right], provider.capacity)
    c_random = float(np.mean(np.einsum("ij,ij->i", y_left, y_right)))
    return c_adj - c_random


class AdaptiveSparseScheduler(BatchScheduler):
    """Use MP only when queue order exhibits resource-direction correlation."""

    def __init__(
        self,
        *,
        candidate: CandidateConfig | None = None,
        solver: SolverConfig | None = None,
        backend: CandidateBackend | None = None,
        fifo_window: int = 65_536,
        sample_size: int = 50_000,
        threshold: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__("Adaptive-SPARSE")
        if sample_size < 2:
            raise ValidationError("adaptive sample_size must be at least two")
        if not np.isfinite(threshold):
            raise ValidationError("adaptive threshold must be finite")
        self.sample_size = int(sample_size)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self._mp = MPScheduler(
            candidate=candidate or CandidateConfig(),
            solver=solver or SolverConfig(),
            backend=backend or LocalBackend(),
        )
        self._fifo = FIFOScheduler(window=fifo_window, strict=False)
        self._selected: BatchScheduler = self._fifo
        self.gate_score = float("nan")
        self.uses_sparse = False

    def reset(self, agents: AgentSet, provider: Provider) -> None:
        self.gate_score = order_correlation_excess(
            agents,
            provider,
            sample_size=self.sample_size,
            seed=self.seed,
        )
        self.uses_sparse = self.gate_score > self.threshold
        self._selected = self._mp if self.uses_sparse else self._fifo
        self._selected.reset(agents, provider)
        self.stats = self._selected.stats
        self.stats.diagnostics.update(
            {
                "adaptive_gate_score": self.gate_score,
                "adaptive_threshold": self.threshold,
                "adaptive_branch": "mp" if self.uses_sparse else "windowed_fifo",
                "adaptive_sample_size": min(self.sample_size, agents.n_agents),
            }
        )

    def select(
        self,
        agents: AgentSet,
        provider: Provider,
        active: ActiveSet,
        remaining: FloatArray,
    ) -> DispatchBatch:
        batch = self._selected.select(agents, provider, active, remaining)
        diagnostics = dict(batch.diagnostics)
        diagnostics.update(
            {
                "adaptive_gate_score": self.gate_score,
                "adaptive_threshold": self.threshold,
                "adaptive_branch": "mp" if self.uses_sparse else "windowed_fifo",
            }
        )
        return DispatchBatch(
            indices=batch.indices,
            used=batch.used,
            remaining=batch.remaining,
            selector_time_s=batch.selector_time_s,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        self._mp.close()
        self._fifo.close()


__all__ = ["AdaptiveSparseScheduler", "order_correlation_excess"]
