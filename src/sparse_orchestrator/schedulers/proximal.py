"""Scalable FISTA and IRLS schedulers."""
from __future__ import annotations

import time

import numpy as np

from ..config import CandidateConfig, SolverConfig
from ..distributed import CandidateBackend, LocalBackend
from ..math_utils import projected_gradient_pack_order
from ..model import AgentSet, DispatchBatch, FloatArray, Provider, ValidationError
from ..packing import best_fit_refinement, greedy_pack
from ..solvers import fista_l1, irls_lp
from ..storage import ActiveSet
from .base import BatchScheduler


class ProximalScheduler(BatchScheduler):
    """Solve a continuous sparse relaxation on the global candidate pool."""

    def __init__(
        self,
        method: str,
        *,
        candidate: CandidateConfig | None = None,
        solver: SolverConfig | None = None,
        backend: CandidateBackend | None = None,
    ) -> None:
        if method not in {"fista", "irls"}:
            raise ValidationError("ProximalScheduler method must be fista or irls")
        super().__init__(method.upper())
        self.method = method
        self.candidate = candidate or CandidateConfig()
        self.solver = solver or SolverConfig()
        self.candidate.validate()
        self.solver.validate()
        self.backend = backend or LocalBackend()

    def _solve(self, demands: np.ndarray, provider: Provider, remaining: np.ndarray):
        dictionary = (demands / provider.capacity[None, :]).T
        target = remaining / provider.capacity
        if self.method == "fista":
            return fista_l1(
                dictionary,
                target,
                l1_lambda=self.solver.l1_lambda,
                max_iterations=self.solver.max_iterations,
                tolerance=self.solver.tolerance,
                positive=self.solver.positive,
                normalize=self.solver.normalize_columns,
            )
        return irls_lp(
            dictionary,
            target,
            p=self.solver.irls_p,
            regularization=self.solver.l1_lambda,
            epsilon=self.solver.irls_epsilon,
            outer_iterations=self.solver.irls_outer_iterations,
            inner_iterations=self.solver.irls_inner_iterations,
            tolerance=self.solver.tolerance,
            positive=self.solver.positive,
            normalize=self.solver.normalize_columns,
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
        result = self._solve(demands, provider, np.asarray(remaining, dtype=np.float64))
        solver_time = time.perf_counter() - solve_start
        self.stats.solver_time_s += solver_time

        weights = np.asarray(result.coefficients, dtype=np.float64)
        # The map proxy resolves the many near-zero ties produced by L1/Lp
        # regularisation without changing the continuous solution itself.
        proxy = np.asarray(pool.scores, dtype=np.float64)
        proxy_scaled = (proxy - proxy.min(initial=0.0)) / max(float(np.ptp(proxy)), 1e-12)
        positive_scale = max(float(np.max(np.abs(weights))), 1e-12)
        ranking_weights = weights / positive_scale + 1e-4 * proxy_scaled
        order = projected_gradient_pack_order(ranking_weights, pool.indices)

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
        return DispatchBatch(
            indices=selected,
            used=packed.used,
            remaining=packed.remaining,
            selector_time_s=time.perf_counter() - total_start,
            diagnostics={
                "pool_size": int(pool.indices.size),
                "solver_iterations": result.iterations,
                "solver_objective": result.objective,
                "solver_converged": int(result.converged),
                "support_size": int(result.support.size),
                "scanned": pool.scanned,
                "feasible": pool.feasible,
                "workers": pool.workers,
            },
        )

    def close(self) -> None:
        self.backend.close()


class FISTAScheduler(ProximalScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("fista", **kwargs)


class IRLSScheduler(ProximalScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("irls", **kwargs)


__all__ = ["FISTAScheduler", "IRLSScheduler", "ProximalScheduler"]
