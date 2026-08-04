"""Reproducible benchmark runner."""
from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import BenchmarkConfig, GeneratorConfig
from ..generator import distinctness_check, generate
from ..model import ExperimentArtifacts, Provider
from ..reporting import results_frame, write_artifacts
from ..schedulers import build_scheduler
from ..simulator import Simulator
from ..storage import open_memmap, save_memmap


def provider_from_config(config: BenchmarkConfig) -> Provider:
    return Provider(
        capacity=np.asarray(config.provider.capacities, dtype=np.float64),
        resource_names=tuple(config.provider.resource_names),
        name=config.provider.name,
    )


def environment_metadata() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "timestamp_unix": time.time(),
    }


def _warmup(config: BenchmarkConfig, provider: Provider) -> None:
    if config.warmup_agents <= 0:
        return
    small_cfg = replace(
        config.generator,
        n_agents=min(config.warmup_agents, config.generator.n_agents),
        seed=config.seeds[0] - 1,
    )
    workload = generate(small_cfg, provider)
    # Warm only the packing path using the first method.  A full method warmup
    # can dominate small benchmark suites and is not part of measured timing.
    if config.scheduler.methods:
        method = config.scheduler.methods[0]
        local_distributed = replace(config.distributed, backend="local", workers=1)
        scheduler = build_scheduler(method, config.scheduler, local_distributed)
        Simulator(replace(config.simulation, record_trace=False)).run(
            workload.agents, provider, scheduler
        )


def run_benchmark(config: BenchmarkConfig) -> ExperimentArtifacts:
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provider = provider_from_config(config)
    _warmup(config, provider)
    records: list[dict[str, object]] = []

    for seed in config.seeds:
        generator_config = replace(config.generator, seed=seed)
        workload = generate(generator_config, provider)
        integrity = distinctness_check(workload.agents)
        agents = workload.agents
        memmap_root: Path | None = None
        if config.distributed.backend == "process":
            memmap_root = output / "memmap" / f"seed_{seed}"
            save_memmap(agents, memmap_root, overwrite=True)
            agents = open_memmap(memmap_root)

        for repetition in range(config.repetitions):
            for method in config.scheduler.methods:
                scheduler = build_scheduler(method, config.scheduler, config.distributed)
                result = Simulator(config.simulation).run(agents, provider, scheduler)
                row = result.as_dict()
                row.update(
                    {
                        "seed": seed,
                        "repetition": repetition,
                        "generator": config.generator.pattern,
                        "explicit_agents": bool(
                            workload.agents.metadata.get("explicit_agents", False)
                        ),
                        "unique_fraction": integrity["unique_fraction"],
                        "n_clusters": config.generator.n_clusters,
                        "backend": config.distributed.backend,
                        "workers": config.distributed.workers,
                    }
                )
                records.append(row)
                pd.DataFrame.from_records(records).to_csv(
                    output / "raw_partial.csv", index=False
                )
                print(
                    f"seed={seed} repetition={repetition} method={result.method} "
                    f"normalized={result.normalized_makespan:.6f} "
                    f"scheduler_s={result.scheduler_time_s:.4f}",
                    flush=True,
                )
        if memmap_root is not None:
            # Keep the benchmark output compact by default.  Generated arrays
            # are deterministic and can be regenerated from the saved config.
            shutil.rmtree(memmap_root, ignore_errors=True)

    raw = pd.DataFrame.from_records(records)
    metadata = {
        "config": asdict(config),
        "environment": environment_metadata(),
        "integrity": {
            "all_explicit": bool(raw["explicit_agents"].all()),
            "minimum_unique_fraction": float(raw["unique_fraction"].min()),
            "all_valid": bool(raw["valid"].all()),
        },
    }
    return write_artifacts(config, raw, output, metadata)


__all__ = ["environment_metadata", "provider_from_config", "run_benchmark"]
