"""CSV, JSON, and Markdown reporting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import BenchmarkConfig
from .model import ExperimentArtifacts, SimulationResult


def results_frame(rows: Iterable[SimulationResult | Mapping[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, SimulationResult):
            records.append(row.as_dict())
        else:
            records.append(dict(row))
    return pd.DataFrame.from_records(records)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "normalized_makespan",
        "makespan",
        "scheduler_time_s",
        "wall_time_s",
        "mean_utilization",
        "valid",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw results missing columns: {sorted(missing)}")
    grouped = raw.groupby("method", sort=False)
    summary = grouped.agg(
        runs=("method", "size"),
        valid_runs=("valid", "sum"),
        normalized_makespan_mean=("normalized_makespan", "mean"),
        normalized_makespan_std=("normalized_makespan", "std"),
        makespan_mean=("makespan", "mean"),
        makespan_std=("makespan", "std"),
        scheduler_time_s_mean=("scheduler_time_s", "mean"),
        scheduler_time_s_std=("scheduler_time_s", "std"),
        wall_time_s_mean=("wall_time_s", "mean"),
        wall_time_s_std=("wall_time_s", "std"),
        utilization_mean=("mean_utilization", "mean"),
        utilization_std=("mean_utilization", "std"),
        dispatches_mean=("dispatches", "mean"),
        dispatches_std=("dispatches", "std"),
    ).reset_index()
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    return summary.sort_values(
        ["normalized_makespan_mean", "scheduler_time_s_mean"],
        kind="mergesort",
    ).reset_index(drop=True)


def _format_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_report(config: BenchmarkConfig, raw: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# {config.name}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Explicit agents per instance: **{config.generator.n_agents:,}**")
    lines.append(f"- Resources: **{config.generator.n_resources}**")
    lines.append(f"- Workload pattern: **{config.generator.pattern}**")
    lines.append(f"- Seeds: `{config.seeds}`")
    lines.append(f"- Candidate backend: **{config.distributed.backend}**")
    lines.append(f"- Backend workers: **{config.distributed.workers}**")
    lines.append("")
    lines.append(
        "LangChain and LangGraph entries, when requested, are scheduling-policy "
        "baselines. They are not measurements of installed framework runtime overhead."
    )
    lines.append("")
    lines.append("## Makespan")
    lines.append("")
    lines.append("| Method | Normalized makespan | Makespan | Utilization | Valid runs |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            "| {method} | {norm} | {makespan} | {util} | {valid}/{runs} |".format(
                method=row.method,
                norm=_format_pm(row.normalized_makespan_mean, row.normalized_makespan_std),
                makespan=_format_pm(row.makespan_mean, row.makespan_std, 3),
                util=_format_pm(row.utilization_mean, row.utilization_std, 4),
                valid=int(row.valid_runs),
                runs=int(row.runs),
            )
        )
    lines.append("")
    lines.append("## Scheduler cost")
    lines.append("")
    lines.append("| Method | Scheduler seconds | Wall seconds | Dispatches |")
    lines.append("|---|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            "| {method} | {sched} | {wall} | {dispatches} |".format(
                method=row.method,
                sched=_format_pm(row.scheduler_time_s_mean, row.scheduler_time_s_std, 4),
                wall=_format_pm(row.wall_time_s_mean, row.wall_time_s_std, 4),
                dispatches=_format_pm(row.dispatches_mean, row.dispatches_std, 2),
            )
        )
    lines.append("")
    lines.append("## Integrity checks")
    lines.append("")
    explicit = raw.get("explicit_agents")
    if explicit is not None:
        lines.append(f"- All runs explicit: **{bool(explicit.all())}**")
    if "unique_fraction" in raw:
        lines.append(f"- Minimum sampled unique fraction: **{raw['unique_fraction'].min():.6f}**")
    lines.append(f"- All schedules valid: **{bool(raw['valid'].all())}**")
    return "\n".join(lines) + "\n"


def write_artifacts(
    config: BenchmarkConfig,
    raw: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> ExperimentArtifacts:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = summarize(raw)
    raw_path = root / "raw.csv"
    summary_path = root / "summary.csv"
    metadata_path = root / "metadata.json"
    report_path = root / "REPORT.md"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(dict(metadata or {}), indent=2, default=str), encoding="utf-8"
    )
    report_path.write_text(markdown_report(config, raw, summary), encoding="utf-8")
    return ExperimentArtifacts(root, raw_path, summary_path, metadata_path, report_path)


__all__ = [
    "markdown_report",
    "results_frame",
    "summarize",
    "write_artifacts",
]
