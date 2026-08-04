"""Schedule a custom explicit agent matrix."""
from __future__ import annotations

import numpy as np

from sparse_orchestrator.config import CandidateConfig, SolverConfig
from sparse_orchestrator.model import AgentSet, Provider
from sparse_orchestrator.schedulers import FISTAScheduler, MPScheduler
from sparse_orchestrator.simulator import Simulator

rng = np.random.default_rng(19)
n_agents = 50_000
n_resources = 4

demands = rng.lognormal(mean=2.0, sigma=0.6, size=(n_agents, n_resources)).astype(
    np.float32
)
durations = rng.lognormal(mean=0.0, sigma=0.2, size=n_agents).astype(np.float32)
agents = AgentSet(demands=demands, durations=durations)
provider = Provider(
    capacity=np.full(n_resources, 50_000.0),
    resource_names=("cpu", "memory", "network", "accelerator"),
)

candidate = CandidateConfig(
    pool_size=8192,
    chunk_size=16384,
    local_top_k=2048,
    direction_pool_size=512,
)
solver = SolverConfig(max_iterations=48, l1_lambda=0.02)

for scheduler in [
    MPScheduler(candidate=candidate, solver=solver),
    FISTAScheduler(candidate=candidate, solver=solver),
]:
    result = Simulator().run(agents, provider, scheduler)
    print(
        scheduler.name,
        f"makespan={result.makespan:.3f}",
        f"normalized={result.normalized_makespan:.4f}",
        f"scheduler_s={result.scheduler_time_s:.3f}",
    )
