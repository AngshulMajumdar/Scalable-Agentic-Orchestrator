from __future__ import annotations

import numpy as np
import pytest

from sparse_orchestrator.config import CandidateConfig, GeneratorConfig, SimulationConfig, SolverConfig
from sparse_orchestrator.generator import generate
from sparse_orchestrator.model import Provider
from sparse_orchestrator.schedulers import FIFOScheduler, MPScheduler
from sparse_orchestrator.simulator import Simulator


def test_wave_simulation_completes_all_agents() -> None:
    provider = Provider(np.full(4, 20_000.0))
    cfg = GeneratorConfig(
        n_agents=2000,
        n_resources=4,
        n_clusters=8,
        seed=8,
        duration_mode="unit",
    )
    agents = generate(cfg, provider).agents
    scheduler = MPScheduler(
        candidate=CandidateConfig(
            pool_size=1000,
            chunk_size=500,
            local_top_k=250,
            direction_budget=6,
            direction_pool_size=128,
        ),
        solver=SolverConfig(max_iterations=20),
    )
    result = Simulator(SimulationConfig(mode="waves", record_trace=True)).run(
        agents, provider, scheduler
    )
    assert result.valid
    assert result.completed == agents.n_agents
    assert result.trace is not None
    result.trace.validate_complete()
    assert result.normalized_makespan >= 1.0 - 1e-10
    assert 0 < result.mean_utilization <= 1.0 + 1e-8


def test_event_simulation_with_variable_durations() -> None:
    provider = Provider(np.full(4, 20_000.0))
    cfg = GeneratorConfig(
        n_agents=1500,
        n_resources=4,
        n_clusters=8,
        seed=6,
        duration_mode="lognormal",
        duration_sigma=0.2,
    )
    agents = generate(cfg, provider).agents
    scheduler = FIFOScheduler(window=512, strict=False)
    result = Simulator(SimulationConfig(mode="events", record_trace=True)).run(
        agents, provider, scheduler
    )
    assert result.valid
    assert result.completed == agents.n_agents
    assert result.makespan >= np.max(agents.durations)
    assert result.trace is not None
    result.trace.validate_complete()


def test_wave_mode_rejects_variable_durations() -> None:
    provider = Provider(np.full(4, 20_000.0))
    cfg = GeneratorConfig(
        n_agents=100,
        n_resources=4,
        n_clusters=8,
        seed=6,
        duration_mode="uniform",
    )
    agents = generate(cfg, provider).agents
    with pytest.raises(Exception, match="equal durations"):
        Simulator(SimulationConfig(mode="waves")).run(
            agents, provider, FIFOScheduler(window=100)
        )
