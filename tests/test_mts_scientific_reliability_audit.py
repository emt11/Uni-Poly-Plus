from types import SimpleNamespace

import pytest

from scripts.audit_mts_glt_v2_nonbonded import periodic_spd, periodic_spd_map
from scripts.run_mts_finetune_scheduler import (
    _append_forwarded,
    assert_no_duplicate_scientific_flags,
    resolved_config_from_command,
)
from src.dataset.periodic_line_glt import canonical_line_token
from src.training.finetune.mode_specs import MODE_SPECS, missing_batch_fields


def test_scheduler_preserves_single_scientific_value():
    command = ["python", "train.py", "--warmup_epochs", "9"]
    _append_forwarded(command, "--warmup_epochs", "9")
    assert command.count("--warmup_epochs") == 1
    assert command[command.index("--warmup_epochs") + 1] == "9"
    assert_no_duplicate_scientific_flags(command)


def test_scheduler_rejects_duplicate_or_conflicting_scientific_value():
    with pytest.raises(ValueError, match="duplicate scientific"):
        assert_no_duplicate_scientific_flags([
            "python", "train.py", "--head_dropout", "0.1",
            "--head_dropout", "0.2",
        ])
    with pytest.raises(ValueError, match="conflicting forwarded"):
        _append_forwarded(
            ["python", "train.py", "--regression_loss", "huber"],
            "--regression_loss", "mse",
        )


def _resolved_command(task="xc", fold=0):
    return [
        "python", "scripts/train.py", "--tasks", task, "--fold_ids", str(fold),
        "--seed", "42", "--pretrained_model_path", "/tmp/model.pth",
        "--mts_glt_mode", "o8_glt_atom", "--target_transform", "standard",
        "--regression_loss", "huber", "--huber_beta", "0.7",
        "--head_dropout", "0.31", "--weight_decay", "0.012",
        "--warmup_epochs", "7", "--mts_glt_fusion_strategy", "legacy_zero",
        "--mts_glt_fusion_warm_epochs", "5", "--batch_size", "23",
        "--eval_batch_size", "47", "--amp_dtype", "fp32",
        "--loader_workers", "3",
    ]


def test_resolved_config_preserves_scientific_values_and_effective_schedule():
    resolved = resolved_config_from_command(
        _resolved_command(), {"git_commit": "abc", "working_tree_dirty": True}, "sha"
    )
    assert resolved["loss"] == "huber"
    assert resolved["beta"] == pytest.approx(0.7)
    assert resolved["head_dropout"] == pytest.approx(0.31)
    assert resolved["weight_decay"] == pytest.approx(0.012)
    assert resolved["lr_warmup_epochs"] == 7
    assert resolved["configured_encoder_freeze_warm_epochs"] == 5
    assert resolved["encoder_freeze_warm_epochs"] == 0


def test_task_fold_changes_do_not_change_resolved_scientific_values():
    provenance = {"git_commit": "abc", "working_tree_dirty": False}
    left = resolved_config_from_command(_resolved_command("xc", 0), provenance, "sha")
    right = resolved_config_from_command(_resolved_command("ei", 3), provenance, "sha")
    ignored = {"task", "fold"}
    assert {k: v for k, v in left.items() if k not in ignored} == {
        k: v for k, v in right.items() if k not in ignored
    }


def test_mode_specs_cover_current_formal_and_recent_modes():
    required = {
        "o8_glt_atom", "o8_glt_atom_x23", "o8_glt_atom_line_x2l",
        "o8_glt_atom_attn_x2a", "o8_glt_atom_torsion",
        "o8_glt_atom_sbf_radial_angle",
    }
    assert required <= set(MODE_SPECS)
    formal = SimpleNamespace(**{
        field: object() for field in MODE_SPECS["o8_glt_atom"].required_batch_fields
    })
    assert missing_batch_fields("o8_glt_atom", formal) == ()
    assert "glt_relation_torsion_count" in missing_batch_fields(
        "o8_glt_atom_torsion", formal
    )


def test_periodic_contact_identity_and_spd_are_translation_invariant():
    assert canonical_line_token(2, 0, 5, 1) == canonical_line_token(5, 1, 2, 0)
    assert canonical_line_token(2, -1, 5, 0) == canonical_line_token(2, 0, 5, 1)
    # Internal 0-1 plus periodic 1@q -- 0@(q+1).
    adjacency = (((1, 0), (1, -1)), ((0, 0), (0, 1)))
    assert periodic_spd(adjacency, 0, 1, 0) == 1
    assert periodic_spd(adjacency, 0, 1, -1) == 1
    assert periodic_spd(adjacency, 0, 0, 1) == 2
    distances = periodic_spd_map(adjacency, 0, shift_limit=4)
    for target in (0, 1):
        for shift in range(-2, 3):
            assert distances.get((target, shift)) == periodic_spd(
                adjacency, 0, target, shift
            )
