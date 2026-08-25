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


def test_mts_config_tree_has_only_explicit_b0_configs():
    configs = list((ROOT / "configs" / "mts").rglob("*.json"))
    assert ROOT / "configs" / "mts" / "b0_v2_probe.json" in configs
    assert ROOT / "configs" / "mts" / "b0_v2_resume_probe.json" in configs
    assert ROOT / "configs" / "mts" / "b0.json" not in configs
    assert ROOT / "configs" / "mts" / "b0_probe.json" not in configs


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


def test_resolver_accepts_b0_v2_probe_and_writes_resolved_input(tmp_path):
    import json
    source = ROOT / "configs" / "mts" / "b0_v2_probe.json"
    result = _run([
        sys.executable, "scripts/resolve_mips_trimer_scage.py",
        str(source), "--print-path",
    ])
    assert result.returncode == 0, result.stdout
    resolved = Path(result.stdout.strip())
    assert resolved.is_file()
    payload = json.loads(resolved.read_text())
    assert payload["schema"] == "mts-b0-v2"
    assert payload["use_mcl"] is False
    assert payload["probe_steps"] == [5000, 10000, 20000]
    assert payload["star_rbf_upper"] == 3.75


def test_resolver_rejects_wrong_star_rbf_upper(tmp_path):
    import json
    source = ROOT / "configs" / "mts" / "b0_v2_probe.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["star_rbf_upper"] = 3.0
    candidate = tmp_path / "wrong_upper.json"
    candidate.write_text(json.dumps(payload))
    result = _run([
        sys.executable, "scripts/resolve_mips_trimer_scage.py", str(candidate)
    ])
    assert result.returncode != 0
    assert "star_rbf_upper" in result.stdout


def test_b0_downstream_finetune_identity_switches():
    from src.training.finetune.config import parse_arguments as parse_finetune

    old = sys.argv
    try:
        sys.argv = ["train.py", "--topology_attention_variant", "o8",
                    "--use_star_rbf", "--no-use_mcl", "--use_md200",
                    "--star_rbf_upper", "3.75"]
        parsed = parse_finetune()
    finally:
        sys.argv = old
    assert parsed.topology_attention_variant == "o8"
    assert parsed.use_star_rbf is True
    assert parsed.use_mcl is False
    assert parsed.use_md200 is True
    assert parsed.star_rbf_upper == 3.75


def test_resolver_rejects_historical_b0_v1_config_without_side_effects(tmp_path):
    source = ROOT / "configs" / "mts" / "b0.json"
    result = _run([
        sys.executable, "scripts/resolve_mips_trimer_scage.py", str(source)
    ])
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
