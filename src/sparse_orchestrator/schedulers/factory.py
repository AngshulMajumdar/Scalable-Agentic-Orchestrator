"""Scheduler construction from configuration."""
from __future__ import annotations

from ..config import DistributedConfig, SchedulerConfig
from ..distributed import LocalBackend, ProcessPoolBackend
from ..model import ValidationError
from .adaptive import AdaptiveSparseScheduler
from .base import BatchScheduler
from .fifo import (
    FIFOScheduler,
    KahnFIFOScheduler,
    LangChainPolicyBaseline,
    LangGraphPolicyBaseline,
)
from .proximal import FISTAScheduler, IRLSScheduler
from .pursuit import MPScheduler, OLSScheduler, OMPScheduler


def _backend(config: DistributedConfig):
    if config.backend == "local":
        return LocalBackend()
    if config.backend == "process":
        return ProcessPoolBackend(config.workers, config.start_method)
    raise ValidationError(f"unknown backend: {config.backend}")


def build_scheduler(
    method: str,
    scheduler_config: SchedulerConfig,
    distributed_config: DistributedConfig,
) -> BatchScheduler:
    if method == "fifo":
        return FIFOScheduler(
            window=scheduler_config.fifo_window,
            strict=scheduler_config.strict_fifo,
        )
    if method == "windowed_fifo":
        return FIFOScheduler(window=scheduler_config.fifo_window, strict=False)
    if method == "kahn":
        return KahnFIFOScheduler(window=scheduler_config.fifo_window)
    if method == "langchain_policy":
        return LangChainPolicyBaseline(window=scheduler_config.fifo_window)
    if method == "langgraph_policy":
        return LangGraphPolicyBaseline(window=scheduler_config.fifo_window)
    if method == "adaptive_sparse":
        return AdaptiveSparseScheduler(
            candidate=scheduler_config.candidate,
            solver=scheduler_config.solver,
            backend=_backend(distributed_config),
            fifo_window=scheduler_config.fifo_window,
            sample_size=scheduler_config.adaptive_sample_size,
            threshold=scheduler_config.adaptive_threshold,
            seed=scheduler_config.adaptive_seed,
        )
    kwargs = {
        "candidate": scheduler_config.candidate,
        "solver": scheduler_config.solver,
        "backend": _backend(distributed_config),
    }
    if method == "mp":
        return MPScheduler(**kwargs)
    if method == "omp":
        return OMPScheduler(**kwargs)
    if method == "ols":
        return OLSScheduler(**kwargs)
    if method == "fista":
        return FISTAScheduler(**kwargs)
    if method == "irls":
        return IRLSScheduler(**kwargs)
    raise ValidationError(f"unknown scheduler method: {method}")


__all__ = ["build_scheduler"]
