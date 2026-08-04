from __future__ import annotations

from dataclasses import asdict

import pytest

from sparse_orchestrator.config import BenchmarkConfig, load_config, save_config
from sparse_orchestrator.model import ValidationError


def test_default_config_is_valid() -> None:
    config = BenchmarkConfig()
    config.validate()
    assert config.generator.n_agents == 1_000_000
    assert "mp" in config.scheduler.methods


def test_yaml_roundtrip(tmp_path) -> None:
    config = BenchmarkConfig(name="roundtrip", seeds=[3, 7])
    config.generator.n_agents = 1234
    config.scheduler.candidate.pool_size = 512
    config.scheduler.candidate.local_top_k = 128
    config.scheduler.candidate.direction_pool_size = 128
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert asdict(loaded) == asdict(config)


def test_json_roundtrip(tmp_path) -> None:
    config = BenchmarkConfig(name="json", seeds=[1])
    config.generator.n_resources = 3
    config.generator.n_clusters = 6
    config.provider.capacities = [100.0, 100.0, 100.0]
    config.provider.resource_names = ["a", "b", "c"]
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.generator.n_resources == 3
    assert loaded.provider.resource_names == ["a", "b", "c"]


def test_dimension_mismatch_rejected() -> None:
    config = BenchmarkConfig()
    config.provider.capacities = [1.0, 2.0]
    config.provider.resource_names = ["a", "b"]
    with pytest.raises(ValidationError, match="dimensions"):
        config.validate()


def test_candidate_pool_validation() -> None:
    config = BenchmarkConfig()
    config.scheduler.candidate.direction_pool_size = config.scheduler.candidate.pool_size + 1
    with pytest.raises(ValidationError, match="direction_pool_size"):
        config.validate()
