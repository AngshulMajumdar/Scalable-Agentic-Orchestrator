"""Single-process chunked MapReduce backend."""
from __future__ import annotations

import time

import numpy as np

from ..math_utils import stable_top_k
from ..model import AgentSet, FloatArray, Provider
from ..scoring import feasible_mask, score_candidates
from ..storage import ActiveSet
from .protocol import CandidateBackend, CandidatePool


class LocalBackend(CandidateBackend):
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
        map_start = time.perf_counter()
        shard_indices: list[np.ndarray] = []
        shard_scores: list[np.ndarray] = []
        scanned = 0
        feasible_total = 0
        for indices in active.iter_chunks(chunk_size):
            demand = agents.demands[indices]
            scanned += int(indices.size)
            fit = feasible_mask(demand, remaining)
            if not np.any(fit):
                continue
            feasible_indices = indices[fit]
            feasible_demands = demand[fit]
            feasible_total += int(feasible_indices.size)
            scores = score_candidates(
                method,
                feasible_demands,
                remaining,
                provider.capacity,
                epsilon=epsilon,
            )
            local_indices, local_scores = stable_top_k(
                scores,
                feasible_indices,
                min(local_top_k, feasible_indices.size),
            )
            shard_indices.append(local_indices)
            shard_scores.append(local_scores)
        map_time = time.perf_counter() - map_start

        reduce_start = time.perf_counter()
        if not shard_indices:
            return CandidatePool(
                indices=np.empty(0, dtype=np.int64),
                scores=np.empty(0, dtype=np.float64),
                scanned=scanned,
                feasible=feasible_total,
                map_time_s=map_time,
                reduce_time_s=time.perf_counter() - reduce_start,
                workers=1,
            )
        merged_indices = np.concatenate(shard_indices)
        merged_scores = np.concatenate(shard_scores)
        indices, scores = stable_top_k(
            merged_scores,
            merged_indices,
            min(pool_size, merged_indices.size),
        )
        return CandidatePool(
            indices=indices,
            scores=scores,
            scanned=scanned,
            feasible=feasible_total,
            map_time_s=map_time,
            reduce_time_s=time.perf_counter() - reduce_start,
            workers=1,
        )


__all__ = ["LocalBackend"]
