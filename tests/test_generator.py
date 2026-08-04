from __future__ import annotations

import numpy as np
import pytest

from sparse_orchestrator.config import GeneratorConfig
from sparse_orchestrator.generator import distinctness_check, generate
from sparse_orchestrator.model import Provider


def provider(resources: int = 4) -> Provider:
    return Provider(np.full(resources, 100_000.0), tuple(f"r{i}" for i in range(resources)))


@pytest.mark.parametrize("pattern", ["correlated_bursts", "iid", "complementary"])
def test_generators_return_explicit_agents(pattern: str) -> None:
    cfg = GeneratorConfig(
        n_agents=20_000,
        n_resources=4,
        n_clusters=8,
        seed=17,
        pattern=pattern,
    )
    workload = generate(cfg, provider())
    assert workload.agents.demands.shape == (20_000, 4)
    assert workload.agents.ids.shape == (20_000,)
    assert np.unique(workload.agents.ids).size == 20_000
    assert workload.agents.metadata["explicit_agents"] is True
    check = distinctness_check(workload.agents, sample=10_000)
    assert check["unique_fraction"] > 0.999


def test_correlated_bursts_are_contiguous() -> None:
    cfg = GeneratorConfig(
        n_agents=10_000,
        n_resources=4,
        n_clusters=8,
        seed=4,
        pattern="correlated_bursts",
    )
    workload = generate(cfg, provider())
    labels = workload.cluster_labels
    transitions = np.count_nonzero(labels[1:] != labels[:-1])
    assert transitions <= cfg.n_clusters - 1


def test_iid_labels_are_mixed() -> None:
    cfg = GeneratorConfig(
        n_agents=10_000,
        n_resources=4,
        n_clusters=8,
        seed=4,
        pattern="iid",
    )
    workload = generate(cfg, provider())
    labels = workload.cluster_labels
    transitions = np.count_nonzero(labels[1:] != labels[:-1])
    assert transitions > 1000


def test_duration_modes() -> None:
    for mode in ["unit", "uniform", "lognormal"]:
        cfg = GeneratorConfig(
            n_agents=1000,
            n_resources=4,
            n_clusters=8,
            seed=7,
            duration_mode=mode,
        )
        workload = generate(cfg, provider())
        assert np.all(workload.agents.durations > 0)
        assert workload.agents.unit_duration is (mode == "unit")
