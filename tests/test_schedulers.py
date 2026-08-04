from __future__ import annotations

import numpy as np
import pytest

from sparse_orchestrator.config import CandidateConfig, SolverConfig
from sparse_orchestrator.generator import generate
from sparse_orchestrator.config import GeneratorConfig
from sparse_orchestrator.model import Provider
from sparse_orchestrator.schedulers import (
    FISTAScheduler,
    FIFOScheduler,
    IRLSScheduler,
    MPScheduler,
    OLSScheduler,
    OMPScheduler,
)
from sparse_orchestrator.storage import ActiveSet


def problem():
    provider = Provider(np.full(4, 25_000.0))
    cfg = GeneratorConfig(n_agents=3000, n_resources=4, n_clusters=8, seed=13)
    agents = generate(cfg, provider).agents
    active = ActiveSet(agents.n_agents)
    candidate = CandidateConfig(
        pool_size=1024,
        chunk_size=512,
        local_top_k=256,
        direction_budget=6,
        direction_pool_size=128,
    )
    solver = SolverConfig(max_iterations=24, irls_outer_iterations=5, irls_inner_iterations=16)
    return agents, provider, active, candidate, solver


@pytest.mark.parametrize("scheduler_type", [MPScheduler, OMPScheduler, OLSScheduler, FISTAScheduler, IRLSScheduler])
def test_sparse_scheduler_returns_feasible_batch(scheduler_type) -> None:
    agents, provider, active, candidate, solver = problem()
    scheduler = scheduler_type(candidate=candidate, solver=solver)
    scheduler.reset(agents, provider)
    batch = scheduler.select(agents, provider, active, provider.capacity.copy())
    scheduler.close()
    assert batch.indices.size > 0
    assert np.unique(batch.indices).size == batch.indices.size
    used = agents.demands[batch.indices].sum(axis=0, dtype=np.float64)
    assert np.all(used <= provider.capacity + 1e-8)
    np.testing.assert_allclose(batch.used, used, rtol=1e-7, atol=1e-6)


def test_fifo_batch_is_in_arrival_order() -> None:
    agents, provider, active, _, _ = problem()
    scheduler = FIFOScheduler(window=1000, strict=False)
    scheduler.reset(agents, provider)
    batch = scheduler.select(agents, provider, active, provider.capacity.copy())
    ranks = np.empty(agents.n_agents, dtype=np.int64)
    ranks[agents.arrival_order] = np.arange(agents.n_agents)
    selected_ranks = ranks[batch.indices]
    assert np.all(selected_ranks[:-1] <= selected_ranks[1:])
