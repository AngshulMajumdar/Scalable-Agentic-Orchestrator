"""Command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .benchmark import run_benchmark
from .config import BenchmarkConfig, GeneratorConfig, load_config, save_config
from .generator import distinctness_check, generate
from .model import Provider
from .reporting import summarize
from .storage import open_memmap, save_memmap


def _provider_from_config(config: BenchmarkConfig) -> Provider:
    return Provider(
        np.asarray(config.provider.capacities, dtype=np.float64),
        tuple(config.provider.resource_names),
        config.provider.name,
    )


def command_init(args: argparse.Namespace) -> int:
    config = BenchmarkConfig()
    if args.agents is not None:
        config.generator.n_agents = args.agents
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    save_config(config, args.path)
    print(args.path)
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = _provider_from_config(config)
    generator_config = replace(config.generator, seed=args.seed or config.generator.seed)
    workload = generate(generator_config, provider)
    manifest = save_memmap(workload.agents, args.output, overwrite=args.overwrite)
    integrity = distinctness_check(workload.agents)
    print(json.dumps({"manifest": manifest.as_dict(), "integrity": integrity}, indent=2))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    agents = open_memmap(args.dataset)
    integrity = distinctness_check(agents, sample=args.sample)
    data = {
        "n_agents": agents.n_agents,
        "n_resources": agents.n_resources,
        "dtype": str(agents.demands.dtype),
        "duration_dtype": str(agents.durations.dtype),
        "unit_duration": agents.unit_duration,
        "metadata": agents.metadata,
        "integrity": integrity,
        "demand_min": np.min(agents.demands, axis=0).tolist(),
        "demand_mean": np.mean(agents.demands, axis=0).tolist(),
        "demand_max": np.max(agents.demands, axis=0).tolist(),
    }
    print(json.dumps(data, indent=2, default=str))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.methods:
        config.scheduler.methods = args.methods
    if args.output:
        config.output_dir = args.output
    if args.workers is not None:
        config.distributed.workers = args.workers
        config.distributed.backend = "process" if args.workers > 1 else "local"
    artifacts = run_benchmark(config)
    print(artifacts.root)
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    import pandas as pd

    raw = pd.read_csv(args.raw)
    summary = summarize(raw)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparse-orchestrator",
        description="Sparse optimization for scheduling explicit heterogeneous agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a complete benchmark configuration")
    init.add_argument("path", type=Path)
    init.add_argument("--agents", type=int)
    init.add_argument("--output-dir")
    init.set_defaults(func=command_init)

    generate_parser = sub.add_parser("generate", help="generate a memory-mapped dataset")
    generate_parser.add_argument("config", type=Path)
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.set_defaults(func=command_generate)

    inspect_parser = sub.add_parser("inspect", help="inspect an explicit-agent dataset")
    inspect_parser.add_argument("dataset", type=Path)
    inspect_parser.add_argument("--sample", type=int, default=100_000)
    inspect_parser.set_defaults(func=command_inspect)

    run = sub.add_parser("run", help="run a benchmark configuration")
    run.add_argument("config", type=Path)
    run.add_argument("--methods", nargs="+")
    run.add_argument("--output")
    run.add_argument("--workers", type=int)
    run.set_defaults(func=command_run)

    summary_parser = sub.add_parser("summarize", help="summarize a raw CSV")
    summary_parser.add_argument("raw", type=Path)
    summary_parser.add_argument("--output", type=Path)
    summary_parser.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
