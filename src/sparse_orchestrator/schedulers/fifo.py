"""FIFO and framework-policy baselines."""
from __future__ import annotations

import time

import numpy as np

from ..model import AgentSet, DispatchBatch, FloatArray, Provider, ValidationError
from ..packing import first_fit_window
from ..storage import ActiveSet
from .base import BatchScheduler


class FIFOScheduler(BatchScheduler):
    """Arrival-order scheduler.

    ``strict=True`` stops at the first head-of-line job that does not fit.
    ``strict=False`` scans a bounded window and launches later jobs that fit.
    Both variants are asynchronous when used by the event simulator.
    """

    def __init__(
        self,
        *,
        window: int = 65_536,
        strict: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name or ("FIFO-Strict" if strict else "FIFO-Windowed"))
        if window <= 0:
            raise ValidationError("FIFO window must be positive")
        self.window = window
        self.strict = strict
        self._cursor = 0
        self._rank: np.ndarray | None = None
        self._order: np.ndarray | None = None

    def reset(self, agents: AgentSet, provider: Provider) -> None:
        super().reset(agents, provider)
        self._cursor = 0
        self._order = np.asarray(agents.arrival_order, dtype=np.int64)
        self._rank = np.empty(agents.n_agents, dtype=np.int64)
        self._rank[self._order] = np.arange(agents.n_agents, dtype=np.int64)

    def _window_indices(self, active: ActiveSet) -> np.ndarray:
        assert self._order is not None
        n = self._order.size
        while self._cursor < n and not active.contains(int(self._order[self._cursor])):
            self._cursor += 1
        if self._cursor >= n:
            return np.empty(0, dtype=np.int64)
        if self.strict:
            stop = min(self._cursor + self.window, n)
            block = self._order[self._cursor:stop]
            # Strict order still omits agents already launched in earlier scan-ahead calls.
            return block[active.mask[block]]
        # Windowed FIFO selects the first active jobs by arrival rank.
        active_indices = active.indices()
        assert self._rank is not None
        ranks = self._rank[active_indices]
        if active_indices.size > self.window:
            take = np.argpartition(ranks, self.window - 1)[: self.window]
            active_indices = active_indices[take]
            ranks = ranks[take]
        return active_indices[np.argsort(ranks, kind="mergesort")]

    def select(
        self,
        agents: AgentSet,
        provider: Provider,
        active: ActiveSet,
        remaining: FloatArray,
    ) -> DispatchBatch:
        start = time.perf_counter()
        self.stats.calls += 1
        indices = self._window_indices(active)
        if indices.size == 0:
            self.stats.empty_calls += 1
            elapsed = time.perf_counter() - start
            return DispatchBatch(
                indices=np.empty(0, dtype=np.int64),
                used=np.zeros(provider.n_resources),
                remaining=np.asarray(remaining, dtype=np.float64).copy(),
                selector_time_s=elapsed,
                diagnostics={"considered": 0, "rejected": 0},
            )
        result = first_fit_window(
            agents.demands,
            indices,
            remaining,
            stop_at_first_failure=self.strict,
        )
        selected = result.selected_positions
        elapsed = time.perf_counter() - start
        self.stats.scanned_agents += result.considered
        self.stats.feasible_agents += int(selected.size)
        self.stats.selected_agents += int(selected.size)
        self.stats.packing_time_s += elapsed
        if selected.size == 0:
            self.stats.empty_calls += 1
        return DispatchBatch(
            indices=selected,
            used=result.used,
            remaining=result.remaining,
            selector_time_s=elapsed,
            diagnostics={
                "considered": result.considered,
                "rejected": result.rejected,
                "strict": int(self.strict),
            },
        )


class KahnFIFOScheduler(FIFOScheduler):
    """Kahn ready-queue policy for the independent-agent special case.

    With no precedence edges, Kahn's ready queue initially contains every
    agent.  Its dispatch policy is therefore a FIFO scan.  The separate class
    keeps benchmark labels precise and leaves room for a future DAG extension.
    """

    def __init__(self, window: int = 65_536) -> None:
        super().__init__(window=window, strict=False, name="Kahn-FIFO")


class LangChainPolicyBaseline(FIFOScheduler):
    """Policy-level baseline, not an installed-framework runtime measurement."""

    def __init__(self, window: int = 65_536) -> None:
        super().__init__(window=window, strict=False, name="LangChain-FCFS-policy")


class LangGraphPolicyBaseline(FIFOScheduler):
    """Policy-level baseline, not an installed-framework runtime measurement."""

    def __init__(self, window: int = 65_536) -> None:
        super().__init__(window=window, strict=False, name="LangGraph-FCFS-policy")


__all__ = [
    "FIFOScheduler",
    "KahnFIFOScheduler",
    "LangChainPolicyBaseline",
    "LangGraphPolicyBaseline",
]
