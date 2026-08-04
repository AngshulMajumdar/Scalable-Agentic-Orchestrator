#!/usr/bin/env python3
"""Fail if a benchmark result violates basic paper-integrity checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--agents", type=int, default=1_000_000)
    args = parser.parse_args()
    frame = pd.read_csv(args.raw)
    failures: list[str] = []
    if not frame["valid"].all():
        failures.append("at least one schedule is invalid")
    if not (frame["n_agents"] == args.agents).all():
        failures.append(f"not every run contains {args.agents:,} agents")
    if "explicit_agents" in frame and not frame["explicit_agents"].all():
        failures.append("at least one run is not marked explicit")
    if "unique_fraction" in frame and frame["unique_fraction"].min() < 0.999:
        failures.append("sampled distinctness is below 99.9%")
    if (frame["normalized_makespan"] < 1.0 - 1e-8).any():
        failures.append("a makespan is below its lower bound")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(frame)} rows satisfy all integrity checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
