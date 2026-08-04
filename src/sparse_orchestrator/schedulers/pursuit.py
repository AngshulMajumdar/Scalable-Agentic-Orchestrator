"""Scalable sharded MP, OMP, and OLS batch schedulers."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import CandidateConfig, SolverConfig
from ..distributed import CandidateBackend, LocalBackend
from ..math_utils import projected_gradient_pack_order
from ..model import AgentSet, DispatchBatch, FloatArray, Provider, ValidationError
from ..packing import best_fit_refinement, greedy_pack
from ..scoring import direction_affinity
from ..solvers import matching_pursuit, orthogonal_least_squares, orthogonal_matching_pursuit
from ..storage import ActiveSet
from .base import BatchScheduler


@dataclass(frozen=True, slots=True)
class _SparseRanking:
    weights: np.ndarray
    support_positions: np.ndarray
    iterations: int
    objective: float
    converged: bool


class PursuitScheduler(BatchScheduler):
    """Apply an exact sparse reference solver to a sharded candidate pool.

    This is not a claim that classical OMP or OLS scans one million atoms at
    every support update.  The map stage retrieves a deterministic global
    candidate pool, the exact reference solver selects sparse resource
    directions inside that pool, and a feasibility-preserving packer launches
    explicit agents ranked by their affinity to those directions.
    """

    def __init__(
        self,
        method: str,
        *,
        candidate: CandidateConfig | None = None,
        solver: SolverConfig | None = None,
        backend: CandidateBackend | None = None,
    ) -> None:
        if method not in {"mp", "omp", "ols"}:
            raise ValidationError("PursuitScheduler method must be mp, omp, or ols")
        super().__init__(method.upper())
        self.method = method
        self.candidate = candidate or CandidateConfig()
        self.solver = solver or SolverConfig()
        self.candidate.validate()
        self.solver.validate()
        self.backend = backend or LocalBackend()

    def _solve_directions(
        self,
        demands: np.ndarray,
        provider: Provider,
        remaining: np.ndarray,
        proxy_scores: np.ndarray,
    ) -> _SparseRanking:
        normalized = demands / provider.capacity[None, :]
        dictionary = normalized.T
        target = remaining / provider.capacity
        # OLS exact candidate testing is deliberately bounded.  Its selected
        # directions then rank the complete candidate pool by affinity.
        direction_pool = min(self.candidate.direction_pool_size, demands.shape[0])
        seed_positions = np.arange(direction_pool, dtype=np.int64)
        subdictionary = dictionary[:, seed_positions]
        budget = min(self.candidate.direction_budget, direction_pool)
        if self.method == "mp":
            result = matching_pursuit(
                subdictionary,
                target,
                max_atoms=budget,
                tolerance=self.solver.tolerance,
                positive=self.solver.positive,
                normalize=self.solver.normalize_columns,
                allow_reselection=False,
            )
        elif self.method == "omp":
            result = orthogonal_matching_pursuit(
                subdictionary,
                target,
                max_atoms=budget,
                tolerance=self.solver.tolerance,
                positive=self.solver.positive,
                normalize=self.solver.normalize_columns,
            )
        else:
            result = orthogonal_least_squares(
                subdictionary,
                target,
                max_atoms=budget,
                tolerance=self.solver.tolerance,
                positive=self.solver.positive,
                normalize=self.solver.normalize_columns,
            )
        support_positions = seed_positions[result.support]
        if support_positions.size:
            directions = demands[support_positions]
            direction_weights = np.abs(result.coefficients[result.support])
            affinity = direction_affinity(
                demands,
                directions,
                direction_weights,
                provider.capacity,
            )
        else:
            affinity = np.zeros(demands.shape[0], dtype=np.float64)
        proxy = np.asarray(proxy_scores, dtype=np.float64)
        proxy_range = float(np.ptp(proxy))
        proxy_scaled = (proxy - float(np.min(proxy))) / max(proxy_range, 1e-12)
        # Sparse support defines resource directions; the map score retains
        # target-specific discrimination among agents with similar directions.
        weights = 0.55 * proxy_scaled + 0.45 * affinity
        if support_positions.size:
            weights[support_positions] += 1.0 + np.linspace(
                1e-6, 0.0, support_positions.size, endpoint=False
            )
        return _SparseRanking(
            weights=weights,
            support_positions=support_positions,
            iterations=result.iterations,
            objective=result.objective,
            converged=result.converged,
        )

    def select(
        self,
        agents: AgentSet,
        provider: Provider,
        active: ActiveSet,
        remaining: FloatArray,
    ) -> DispatchBatch:
        total_start = time.perf_counter()
        self.stats.calls += 1
        pool = self.backend.top_k(
            method=self.method,
            agents=agents,
            provider=provider,
            active=active,
            remaining=remaining,
            pool_size=self.candidate.pool_size,
            local_top_k=self.candidate.local_top_k,
            chunk_size=self.candidate.chunk_size,
            epsilon=self.candidate.score_epsilon,
        )
        self.stats.scanned_agents += pool.scanned
        self.stats.feasible_agents += pool.feasible
        self.stats.candidate_map_time_s += pool.map_time_s
        self.stats.candidate_reduce_time_s += pool.reduce_time_s
        if pool.indices.size == 0:
            self.stats.empty_calls += 1
            return DispatchBatch(
                indices=np.empty(0, dtype=np.int64),
                used=np.zeros(provider.n_resources),
                remaining=np.asarray(remaining, dtype=np.float64).copy(),
                selector_time_s=time.perf_counter() - total_start,
                diagnostics={"pool_size": 0, "scanned": pool.scanned},
            )

        demands = np.asarray(agents.demands[pool.indices], dtype=np.float64)
        solve_start = time.perf_counter()
        ranking = self._solve_directions(demands, provider, np.asarray(remaining), pool.scores)
        solver_time = time.perf_counter() - solve_start
        self.stats.solver_time_s += solver_time

        order = projected_gradient_pack_order(ranking.weights, pool.indices)
        pack_start = time.perf_counter()
        packed = best_fit_refinement(
            demands,
            order,
            np.asarray(remaining, dtype=np.float64),
            passes=2,
        )
        packing_time = time.perf_counter() - pack_start
        self.stats.packing_time_s += packing_time
        selected = pool.indices[packed.selected_positions]
        if selected.size == 0:
            # Pool entries are individually feasible, so deterministic fallback
            # must make progress unless remaining capacity is effectively zero.
            fallback_order = np.lexsort((pool.indices, -pool.scores))
            fallback = greedy_pack(
                demands,
                fallback_order[:1],
                np.asarray(remaining, dtype=np.float64),
            )
            selected = pool.indices[fallback.selected_positions]
            packed = fallback
        self.stats.selected_agents += int(selected.size)
        if selected.size == 0:
            self.stats.empty_calls += 1
        elapsed = time.perf_counter() - total_start
        return DispatchBatch(
            indices=selected,
            used=packed.used,
            remaining=packed.remaining,
            selector_time_s=elapsed,
            diagnostics={
                "pool_size": int(pool.indices.size),
                "support_size": int(ranking.support_positions.size),
                "solver_iterations": ranking.iterations,
                "solver_objective": ranking.objective,
                "solver_converged": int(ranking.converged),
                "scanned": pool.scanned,
                "feasible": pool.feasible,
                "workers": pool.workers,
            },
        )

    def close(self) -> None:
        self.backend.close()


class MPScheduler(PursuitScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("mp", **kwargs)


class OMPScheduler(PursuitScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("omp", **kwargs)


class OLSScheduler(PursuitScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("ols", **kwargs)


__all__ = ["MPScheduler", "OLSScheduler", "OMPScheduler", "PursuitScheduler"]
