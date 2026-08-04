"""Multiprocess MapReduce backend for memory-mapped demand arrays."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..math_utils import stable_top_k
from ..model import AgentSet, FloatArray, Provider, ValidationError
from ..scoring import feasible_mask, score_candidates
from ..storage import ActiveSet
from .protocol import CandidateBackend, CandidatePool


@dataclass(frozen=True, slots=True)
class _ShardTask:
    method: str
    demands_path: str
    demands_dtype: str
    demands_shape: tuple[int, int]
    indices: np.ndarray
    remaining: np.ndarray
    capacity: np.ndarray
    local_top_k: int
    epsilon: float


def _map_shard(task: _ShardTask) -> tuple[np.ndarray, np.ndarray, int, int]:
    demands = np.memmap(
        task.demands_path,
        mode="r",
        dtype=np.dtype(task.demands_dtype),
        shape=task.demands_shape,
    )
    block = demands[task.indices]
    fit = feasible_mask(block, task.remaining)
    feasible_indices = task.indices[fit]
    if feasible_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            int(task.indices.size),
            0,
        )
    scores = score_candidates(
        task.method,
        block[fit],
        task.remaining,
        task.capacity,
        epsilon=task.epsilon,
    )
    idx, val = stable_top_k(scores, feasible_indices, min(task.local_top_k, scores.size))
    return idx, val, int(task.indices.size), int(feasible_indices.size)


class ProcessPoolBackend(CandidateBackend):
    """Map shards in worker processes and reduce top-k results centrally.

    This backend intentionally requires ``agents.demands`` to be a NumPy
    memmap.  Passing an ordinary million-row array to spawned processes would
    silently copy it and defeat the point of a distributed scheduler.
    """

    def __init__(self, workers: int, start_method: str = "spawn") -> None:
        if workers <= 0:
            raise ValidationError("workers must be positive")
        self.workers = workers
        context = mp.get_context(start_method)
        self._executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)

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
        if not isinstance(agents.demands, np.memmap):
            raise ValidationError(
                "ProcessPoolBackend requires a memory-mapped demand matrix; "
                "use save_memmap/open_memmap first"
            )
        filename = agents.demands.filename
        if not isinstance(filename, (str, bytes, os.PathLike)):
            raise ValidationError("could not resolve memmap filename")
        active_indices = active.indices()
        shards = [
            active_indices[start : start + chunk_size]
            for start in range(0, active_indices.size, chunk_size)
        ]
        tasks = [
            _ShardTask(
                method=method,
                demands_path=os.fspath(filename),
                demands_dtype=str(agents.demands.dtype),
                demands_shape=tuple(int(x) for x in agents.demands.shape),
                indices=indices,
                remaining=np.asarray(remaining, dtype=np.float64),
                capacity=np.asarray(provider.capacity, dtype=np.float64),
                local_top_k=local_top_k,
                epsilon=epsilon,
            )
            for indices in shards
        ]
        map_start = time.perf_counter()
        results = list(self._executor.map(_map_shard, tasks))
        map_time = time.perf_counter() - map_start

        reduce_start = time.perf_counter()
        nonempty = [(idx, val) for idx, val, _, _ in results if idx.size]
        scanned = sum(scanned for _, _, scanned, _ in results)
        feasible = sum(count for _, _, _, count in results)
        if not nonempty:
            return CandidatePool(
                indices=np.empty(0, dtype=np.int64),
                scores=np.empty(0, dtype=np.float64),
                scanned=scanned,
                feasible=feasible,
                map_time_s=map_time,
                reduce_time_s=time.perf_counter() - reduce_start,
                workers=self.workers,
            )
        merged_indices = np.concatenate([x[0] for x in nonempty])
        merged_scores = np.concatenate([x[1] for x in nonempty])
        indices, scores = stable_top_k(
            merged_scores,
            merged_indices,
            min(pool_size, merged_indices.size),
        )
        return CandidatePool(
            indices=indices,
            scores=scores,
            scanned=scanned,
            feasible=feasible,
            map_time_s=map_time,
            reduce_time_s=time.perf_counter() - reduce_start,
            workers=self.workers,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["ProcessPoolBackend"]
