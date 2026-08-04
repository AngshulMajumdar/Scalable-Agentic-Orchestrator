import numpy as np

from sparse_orchestrator.model import AgentSet, Provider
from sparse_orchestrator.schedulers import AdaptiveSparseScheduler, order_correlation_excess


def _agents(demands: np.ndarray) -> AgentSet:
    n = demands.shape[0]
    return AgentSet(
        demands=demands.astype(np.float32),
        durations=np.ones(n, dtype=np.float32),
        ids=np.arange(n, dtype=np.int64),
        arrival_order=np.arange(n, dtype=np.int64),
    )


def test_order_correlation_detects_blocks() -> None:
    a = np.tile(np.array([[10.0, 1.0]], dtype=np.float32), (100, 1))
    b = np.tile(np.array([[1.0, 10.0]], dtype=np.float32), (100, 1))
    agents = _agents(np.vstack([a, b]))
    provider = Provider(np.array([100.0, 100.0]), ("cpu", "memory"))
    score = order_correlation_excess(agents, provider, sample_size=200, seed=3)
    assert score > 0.2


def test_adaptive_uses_fifo_for_complementary_order() -> None:
    demands = np.array([[10.0, 1.0], [1.0, 10.0]] * 100, dtype=np.float32)
    agents = _agents(demands)
    provider = Provider(np.array([100.0, 100.0]), ("cpu", "memory"))
    scheduler = AdaptiveSparseScheduler(sample_size=200, threshold=0.1, seed=4)
    scheduler.reset(agents, provider)
    assert not scheduler.uses_sparse
    assert scheduler.stats.diagnostics["adaptive_branch"] == "windowed_fifo"
