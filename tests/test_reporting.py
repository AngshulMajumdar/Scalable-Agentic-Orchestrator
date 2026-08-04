from __future__ import annotations

import pandas as pd

from sparse_orchestrator.config import BenchmarkConfig
from sparse_orchestrator.reporting import markdown_report, summarize, write_artifacts


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(method="A", normalized_makespan=1.0, makespan=10.0, scheduler_time_s=0.2, wall_time_s=0.3, mean_utilization=0.9, valid=True, dispatches=10, explicit_agents=True, unique_fraction=1.0),
            dict(method="A", normalized_makespan=1.1, makespan=11.0, scheduler_time_s=0.3, wall_time_s=0.4, mean_utilization=0.8, valid=True, dispatches=11, explicit_agents=True, unique_fraction=1.0),
            dict(method="B", normalized_makespan=1.5, makespan=15.0, scheduler_time_s=0.1, wall_time_s=0.2, mean_utilization=0.7, valid=True, dispatches=15, explicit_agents=True, unique_fraction=1.0),
        ]
    )


def test_summary_statistics() -> None:
    summary = summarize(raw_frame())
    a = summary[summary.method == "A"].iloc[0]
    assert abs(a.normalized_makespan_mean - 1.05) < 1e-12
    assert a.runs == 2
    assert a.valid_runs == 2


def test_markdown_and_artifacts(tmp_path) -> None:
    config = BenchmarkConfig(name="report-test", seeds=[1])
    raw = raw_frame()
    summary = summarize(raw)
    text = markdown_report(config, raw, summary)
    assert "# report-test" in text
    assert "Scheduler cost" in text
    artifacts = write_artifacts(config, raw, tmp_path, {"x": 1})
    assert artifacts.raw_csv.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.report_md.exists()
    assert artifacts.metadata_json.exists()
