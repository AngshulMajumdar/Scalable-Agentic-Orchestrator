"""Candidate-retrieval backend protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..model import AgentSet, FloatArray, IntArray, Provider
from ..storage import ActiveSet


@dataclass(frozen=True, slots=True)
class CandidatePool:
    indices: IntArray
    scores: FloatArray
    scanned: int
    feasible: int
    map_time_s: float
    reduce_time_s: float
    workers: int


class CandidateBackend(ABC):
    """MapReduce-like backend that returns a deterministic global top-k pool."""

    @abstractmethod
    def top_k(
        self,
        *,
        method: str,
        agents: AgentSet,
        provider: Provider,
        active: ActiveSet,
        remaining: FloatArray,
        pool_size: int,
        local_top_k: int,
        chunk_size: int,
        epsilon: float,
    ) -> CandidatePool:
        raise NotImplementedError

    def close(self) -> None:
        """Release worker resources.  Local implementations need no action."""


__all__ = ["CandidateBackend", "CandidatePool"]
