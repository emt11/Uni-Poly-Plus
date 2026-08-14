"""Behavioral checks for the deliberately paused MTS configuration layer."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command, *, env=None):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )


def test_mts_config_tree_has_no_json_files():
    assert not list((ROOT / "configs" / "mts").rglob("*.json"))


def test_launchers_fail_before_python_or_outputs(tmp_path):
    env = os.environ.copy()
    env.pop("EXPERIMENT_CONFIG", None)
    for launcher in (
        "scripts/run_mips_trimer_scage.sh",
        "scripts/run_mts.sh",
        "scripts/run_train.sh",
    ):
        result = _run(["bash", launcher], env=env)
        assert result.returncode != 0, (launcher, result.stdout)
        assert "EXPERIMENT_CONFIG" in result.stdout
    assert not list(tmp_path.iterdir())


def test_resolver_rejects_explicit_json_without_side_effects(tmp_path):
    config = tmp_path / "candidate.json"
    config.write_text('{"schema_version": "retired", "foo_hash": "x"}\n')
    result = _run([sys.executable, "scripts/resolve_mips_trimer_scage.py", str(config)])
    assert result.returncode != 0
    assert "No active MTS configuration schema" in result.stdout


def test_benchmarks_require_an_explicit_new_configuration(tmp_path):
    for script in (
        "scripts/benchmark_mts_pretrain.py",
        "scripts/benchmark_mts_finetune.py",
    ):
        result = _run([sys.executable, script, "--dry-run"])
        assert result.returncode != 0, (script, result.stdout)
        assert "No active MTS configuration" in result.stdout


def test_training_parser_defaults_do_not_request_full_cache_audit():
    from src.training.pretrain.config import parse_arguments as parse_pretrain
    from src.training.finetune.config import parse_arguments as parse_finetune

    old = sys.argv
    try:
        sys.argv = ["pretrain.py"]
        pretrain = parse_pretrain()
        sys.argv = ["train.py"]
        finetune = parse_finetune()
    finally:
        sys.argv = old
    assert pretrain.cache_validate == "sample"
    assert finetune.cache_validate == "sample"

