from __future__ import annotations

from sparse_orchestrator.cli import main
from sparse_orchestrator.config import load_config


def test_init_command(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    code = main(["init", str(path), "--agents", "1234", "--output-dir", "out"])
    assert code == 0
    config = load_config(path)
    assert config.generator.n_agents == 1234
    assert config.output_dir == "out"
