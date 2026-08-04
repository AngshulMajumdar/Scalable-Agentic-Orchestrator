"""Configuration objects and YAML/JSON loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from .model import ValidationError

T = TypeVar("T")


@dataclass(slots=True)
class GeneratorConfig:
    n_agents: int = 1_000_000
    n_resources: int = 4
    n_clusters: int = 8
    seed: int = 100
    pattern: str = "correlated_bursts"
    duration_mode: str = "unit"
    duration_mean: float = 1.0
    duration_sigma: float = 0.35
    dominant_low: float = 90.0
    dominant_high: float = 140.0
    background_low: float = 4.0
    background_high: float = 16.0
    jitter_sigma: float = 0.14
    burst_concentration: float = 60.0
    dtype: str = "float32"

    def validate(self) -> None:
        if self.n_agents <= 0:
            raise ValidationError("n_agents must be positive")
        if self.n_resources < 2:
            raise ValidationError("n_resources must be at least two")
        if self.n_clusters < self.n_resources:
            raise ValidationError("n_clusters must be at least n_resources")
        if self.pattern not in {"correlated_bursts", "iid", "complementary"}:
            raise ValidationError(f"unsupported generator pattern: {self.pattern}")
        if self.duration_mode not in {"unit", "lognormal", "uniform"}:
            raise ValidationError(f"unsupported duration_mode: {self.duration_mode}")
        if self.duration_mean <= 0 or self.duration_sigma < 0:
            raise ValidationError("duration parameters are invalid")
        if not (0 < self.background_low <= self.background_high):
            raise ValidationError("background demand range is invalid")
        if not (0 < self.dominant_low <= self.dominant_high):
            raise ValidationError("dominant demand range is invalid")
        if self.jitter_sigma < 0:
            raise ValidationError("jitter_sigma cannot be negative")
        if self.burst_concentration <= 0:
            raise ValidationError("burst_concentration must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValidationError("dtype must be float32 or float64")


@dataclass(slots=True)
class ProviderConfig:
    capacities: list[float] = field(default_factory=lambda: [3_000_000.0] * 4)
    resource_names: list[str] = field(
        default_factory=lambda: ["cpu", "memory", "network", "accelerator"]
    )
    name: str = "benchmark-provider"

    def validate(self) -> None:
        if not self.capacities or any(x <= 0 for x in self.capacities):
            raise ValidationError("provider capacities must be positive")
        if self.resource_names and len(self.resource_names) != len(self.capacities):
            raise ValidationError("resource_names must match capacities")


@dataclass(slots=True)
class CandidateConfig:
    pool_size: int = 65_536
    chunk_size: int = 131_072
    local_top_k: int = 16_384
    direction_budget: int = 12
    direction_pool_size: int = 2048
    refill_rounds: int = 4
    score_epsilon: float = 1e-12
    deterministic: bool = True

    def validate(self) -> None:
        if self.pool_size <= 0 or self.chunk_size <= 0 or self.local_top_k <= 0:
            raise ValidationError("candidate sizes must be positive")
        if self.local_top_k > self.pool_size:
            raise ValidationError("local_top_k cannot exceed pool_size")
        if self.direction_budget <= 0:
            raise ValidationError("direction_budget must be positive")
        if self.direction_pool_size <= 0:
            raise ValidationError("direction_pool_size must be positive")
        if self.direction_pool_size > self.pool_size:
            raise ValidationError("direction_pool_size cannot exceed pool_size")
        if self.refill_rounds <= 0:
            raise ValidationError("refill_rounds must be positive")


@dataclass(slots=True)
class SolverConfig:
    max_iterations: int = 64
    tolerance: float = 1e-6
    l1_lambda: float = 0.025
    irls_p: float = 0.5
    irls_epsilon: float = 1e-3
    irls_outer_iterations: int = 8
    irls_inner_iterations: int = 32
    positive: bool = True
    normalize_columns: bool = True

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise ValidationError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValidationError("tolerance must be positive")
        if self.l1_lambda < 0:
            raise ValidationError("l1_lambda cannot be negative")
        if not (0 < self.irls_p <= 1):
            raise ValidationError("irls_p must lie in (0, 1]")
        if self.irls_epsilon <= 0:
            raise ValidationError("irls_epsilon must be positive")
        if self.irls_outer_iterations <= 0 or self.irls_inner_iterations <= 0:
            raise ValidationError("IRLS iteration counts must be positive")


@dataclass(slots=True)
class SchedulerConfig:
    methods: list[str] = field(
        default_factory=lambda: [
            "adaptive_sparse",
            "mp",
            "omp",
            "ols",
            "fista",
            "irls",
            "fifo",
            "windowed_fifo",
            "kahn",
        ]
    )
    fifo_window: int = 65_536
    strict_fifo: bool = False
    objective: str = "makespan"
    adaptive_sample_size: int = 50_000
    adaptive_threshold: float = 0.1
    adaptive_seed: int = 0
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def validate(self) -> None:
        known = {
            "mp",
            "omp",
            "ols",
            "fista",
            "irls",
            "fifo",
            "windowed_fifo",
            "kahn",
            "langchain_policy",
            "langgraph_policy",
            "adaptive_sparse",
        }
        unknown = set(self.methods) - known
        if unknown:
            raise ValidationError(f"unknown scheduler methods: {sorted(unknown)}")
        if self.fifo_window <= 0:
            raise ValidationError("fifo_window must be positive")
        if self.objective not in {"makespan", "throughput"}:
            raise ValidationError("objective must be makespan or throughput")
        if self.adaptive_sample_size < 2:
            raise ValidationError("adaptive_sample_size must be at least two")
        if not isinstance(self.adaptive_seed, int):
            raise ValidationError("adaptive_seed must be an integer")
        if not isinstance(self.adaptive_threshold, (int, float)):
            raise ValidationError("adaptive_threshold must be numeric")
        self.candidate.validate()
        self.solver.validate()


@dataclass(slots=True)
class SimulationConfig:
    mode: str = "auto"
    record_trace: bool = False
    validate_every_dispatch: bool = True
    max_dispatches: int | None = None
    event_tolerance: float = 1e-10

    def validate(self) -> None:
        if self.mode not in {"auto", "waves", "events"}:
            raise ValidationError("simulation mode must be auto, waves, or events")
        if self.max_dispatches is not None and self.max_dispatches <= 0:
            raise ValidationError("max_dispatches must be positive when supplied")
        if self.event_tolerance <= 0:
            raise ValidationError("event_tolerance must be positive")


@dataclass(slots=True)
class DistributedConfig:
    backend: str = "local"
    workers: int = 1
    start_method: str = "spawn"
    shard_size: int = 131_072
    temp_dir: str | None = None

    def validate(self) -> None:
        if self.backend not in {"local", "process"}:
            raise ValidationError("backend must be local or process")
        if self.workers <= 0:
            raise ValidationError("workers must be positive")
        if self.start_method not in {"spawn", "fork", "forkserver"}:
            raise ValidationError("unsupported multiprocessing start method")
        if self.shard_size <= 0:
            raise ValidationError("shard_size must be positive")


@dataclass(slots=True)
class BenchmarkConfig:
    name: str = "million-distinct-agents"
    seeds: list[int] = field(default_factory=lambda: list(range(100, 110)))
    output_dir: str = "results/million_agents"
    warmup_agents: int = 20_000
    repetitions: int = 1
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValidationError("benchmark name cannot be blank")
        if not self.seeds:
            raise ValidationError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValidationError("benchmark seeds must be unique")
        if self.warmup_agents < 0:
            raise ValidationError("warmup_agents cannot be negative")
        if self.repetitions <= 0:
            raise ValidationError("repetitions must be positive")
        self.generator.validate()
        self.provider.validate()
        if len(self.provider.capacities) != self.generator.n_resources:
            raise ValidationError(
                "provider capacity dimensions must match generator n_resources"
            )
        self.scheduler.validate()
        self.simulation.validate()
        self.distributed.validate()


def _construct_dataclass(cls: type[T], raw: Mapping[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        target = f.type
        # Explicit dispatch is more reliable than introspecting postponed annotations.
        nested: dict[str, type[Any]] = {
            "generator": GeneratorConfig,
            "provider": ProviderConfig,
            "scheduler": SchedulerConfig,
            "candidate": CandidateConfig,
            "solver": SolverConfig,
            "simulation": SimulationConfig,
            "distributed": DistributedConfig,
        }
        if f.name in nested and isinstance(value, Mapping):
            kwargs[f.name] = _construct_dataclass(nested[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValidationError("configuration file must be YAML or JSON")
    if not isinstance(raw, Mapping):
        raise ValidationError("configuration root must be a mapping")
    config = _construct_dataclass(BenchmarkConfig, raw)
    config.validate()
    return config


def save_config(config: BenchmarkConfig, path: str | Path) -> None:
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        raise ValidationError("configuration file must be YAML or JSON")


__all__ = [
    "BenchmarkConfig",
    "CandidateConfig",
    "DistributedConfig",
    "GeneratorConfig",
    "ProviderConfig",
    "SchedulerConfig",
    "SimulationConfig",
    "SolverConfig",
    "load_config",
    "save_config",
]
