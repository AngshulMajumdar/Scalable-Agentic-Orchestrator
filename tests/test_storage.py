from __future__ import annotations

import numpy as np

from sparse_orchestrator.config import GeneratorConfig
from sparse_orchestrator.generator import generate
from sparse_orchestrator.model import Provider
from sparse_orchestrator.storage import ActiveSet, load_manifest, open_memmap, save_memmap


def test_memmap_roundtrip(tmp_path) -> None:
    provider = Provider(np.full(4, 100_000.0))
    cfg = GeneratorConfig(n_agents=5000, n_resources=4, n_clusters=8, seed=2)
    original = generate(cfg, provider).agents
    manifest = save_memmap(original, tmp_path / "dataset")
    loaded_manifest = load_manifest(tmp_path / "dataset")
    assert loaded_manifest.n_agents == 5000
    loaded = open_memmap(tmp_path / "dataset")
    assert isinstance(loaded.demands, np.memmap)
    np.testing.assert_allclose(loaded.demands, original.demands)
    np.testing.assert_allclose(loaded.durations, original.durations)
    np.testing.assert_array_equal(loaded.ids, original.ids)
    assert loaded.metadata == original.metadata


def test_active_set_remove_and_chunks() -> None:
    active = ActiveSet(10)
    active.remove(np.array([1, 3, 5]))
    assert active.remaining == 7
    assert not active.contains(3)
    assert active.contains(4)
    chunks = list(active.iter_chunks(3))
    np.testing.assert_array_equal(np.concatenate(chunks), np.array([0, 2, 4, 6, 7, 8, 9]))
    active.remove(np.array([1, 1, 3]))
    assert active.remaining == 7
