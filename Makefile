.PHONY: install test smoke million figures clean

install:
	python -m pip install -e '.[fast,test]'

test:
	pytest

smoke:
	sparse-orchestrator run configs/smoke.yaml

million:
	sparse-orchestrator run configs/million_agents.yaml

figures:
	python -m pip install -r requirements-figures.txt
	python scripts/generate_readme_figures.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info results
