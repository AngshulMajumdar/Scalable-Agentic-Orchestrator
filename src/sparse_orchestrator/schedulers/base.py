"""Batch scheduler interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..model import AgentSet, DispatchBatch, FloatArray, Provider
from ..storage import ActiveSet


@dataclass(slots=True)
class SchedulerStats:
    calls: int = 0
    selected_agents: int = 0
    scanned_agents: int = 0
    feasible_agents: int = 0
    candidate_map_time_s: float = 0.0
    candidate_reduce_time_s: float = 0.0
    solver_time_s: float = 0.0
    packing_time_s: float = 0.0
    empty_calls: int = 0
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "calls": self.calls,
            "selected_agents": self.selected_agents,
            "scanned_agents": self.scanned_agents,
            "feasible_agents": self.feasible_agents,
            "candidate_map_time_s": self.candidate_map_time_s,
            "candidate_reduce_time_s": self.candidate_reduce_time_s,
            "solver_time_s": self.solver_time_s,
            "packing_time_s": self.packing_time_s,
            "empty_calls": self.empty_calls,
            **self.diagnostics,
        }


class BatchScheduler(ABC):
    """Select a feasible set from the currently active agents."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.stats = SchedulerStats()

    def reset(self, agents: AgentSet, provider: Provider) -> None:
        self.stats = SchedulerStats()

    @abstractmethod
    def select(
        self,
        agents: AgentSet,
        provider: Provider,
        active: ActiveSet,
        remaining: FloatArray,
    ) -> DispatchBatch:
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""


__all__ = ["BatchScheduler", "SchedulerStats"]
