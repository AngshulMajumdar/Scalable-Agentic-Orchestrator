#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e '.[fast,test]'
pytest
sparse-orchestrator run configs/million_agents.yaml
python scripts/validate_results.py results/million_agents/raw.csv --agents 1000000
