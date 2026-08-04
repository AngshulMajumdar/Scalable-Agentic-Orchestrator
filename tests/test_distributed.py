from __future__ import annotations

import numpy as np

from sparse_orchestrator.config import GeneratorConfig
from sparse_orchestrator.distributed import LocalBackend, ProcessPoolBackend
from sparse_orchestrator.generator import generate
from sparse_orchestrator.model import Provider
from sparse_orchestrator.storage import ActiveSet, open_memmap, save_memmap


def _problem(n: int = 10_000):
    provider = Provider(np.full(4, 100_000.0))
    cfg = GeneratorConfig(n_agents=n, n_resources=4, n_clusters=8, seed=9)
    agents = generate(cfg, provider).agents
    active = ActiveSet(n)
    remaining = provider.capacity.copy()
    return agents, provider, active, remaining


def test_local_backend_is_deterministic() -> None:
    agents, provider, active, remaining = _problem()
    backend = LocalBackend()
    first = backend.top_k(
        method="mp",
        agents=agents,
        provider=provider,
        active=active,
        remaining=remaining,
        pool_size=512,
        local_top_k=128,
        chunk_size=1024,
        epsilon=1e-12,
    )
    second = backend.top_k(
        method="mp",
        agents=agents,
        provider=provider,
        active=active,
        remaining=remaining,
        pool_size=512,
        local_top_k=128,
        chunk_size=1024,
        epsilon=1e-12,
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.scores, second.scores)
    assert first.scanned == agents.n_agents
    assert first.feasible == agents.n_agents


def test_process_backend_matches_local_top_k(tmp_path) -> None:
    agents, provider, active, remaining = _problem(5000)
    save_memmap(agents, tmp_path / "dataset")
    mapped = open_memmap(tmp_path / "dataset")
    kwargs = dict(
        method="ols",
        agents=mapped,
        provider=provider,
        active=active,
        remaining=remaining,
        pool_size=256,
        local_top_k=256,
        chunk_size=1000,
        epsilon=1e-12,
    )
    local = LocalBackend().top_k(**kwargs)
    process = ProcessPoolBackend(workers=2, start_method="spawn")
    try:
        distributed = process.top_k(**kwargs)
    finally:
        process.close()
    np.testing.assert_array_equal(local.indices, distributed.indices)
    np.testing.assert_allclose(local.scores, distributed.scores)
