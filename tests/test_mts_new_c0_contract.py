"""New-C0（GLT-V2 revision-2 无蒸馏 baseline）契约：MD200 可训练与实验身份隔离。"""

import json
from pathlib import Path

import pytest
import torch

from src.modules.mts_glt_distill import DistillStudent, NPlusGLTTeacher
from src.training.pretrain.glt_distill_engine import StudentContainer, run_stage

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_r2_c0_gelu138_mipshead.json"
PROTECTED_FORMAL_ROOT = ROOT / "results/mts_glt_distill_repair_control"
# Frozen branches allowed by contract: the unused line projection of a student
# without a teacher, and the disabled periodic star-distance bias of the O8.
FROZEN_ALLOWED = ("line_projection.", "student.o8.star_distance_bias.")


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_md200_is_trainable_and_o8_md_residual_has_no_parameters():
    student = DistillStudent()

    assert isinstance(student.o8.md_residual, torch.nn.Identity)
    assert list(student.o8.md_residual.parameters()) == []

    md200 = list(student.md_residual.named_parameters())
    assert len(md200) == 10
    assert all(parameter.requires_grad for _, parameter in md200)


def test_c0_container_keeps_md200_trainable_and_freezes_only_disabled_branches():
    container = StudentContainer(None)

    assert container.teacher is None
    frozen = {name for name, parameter in container.named_parameters() if not parameter.requires_grad}
    assert frozen and all(name.startswith(FROZEN_ALLOWED) for name in frozen)
    assert not any(name.startswith("student.md_residual.") for name in frozen)

    md200_ids = {id(parameter) for parameter in container.student.md_residual.parameters()}
    optimizable = [parameter for parameter in container.parameters() if parameter.requires_grad]
    assert md200_ids <= {id(parameter) for parameter in optimizable}
    optimizer = torch.optim.AdamW(optimizable, lr=2e-4, betas=(0.9, 0.98), weight_decay=0.0)
    listed = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert md200_ids <= listed


def test_distillation_container_keeps_md200_trainable_and_freezes_only_teacher():
    container = StudentContainer(NPlusGLTTeacher())

    frozen = {name for name, parameter in container.named_parameters() if not parameter.requires_grad}
    assert frozen and all(
        name.startswith(("teacher.", "student.o8.star_distance_bias.")) for name in frozen
    )
    assert not any(name.startswith("student.md_residual.") for name in frozen)
    assert all(parameter.requires_grad for parameter in container.student.md_residual.parameters())
    assert all(parameter.requires_grad for parameter in container.line_projection.parameters())


def test_new_c0_config_is_isolated_from_formal_roots():
    config = _config()

    assert config["experiment_id"] == "glt_v2_r2_c0_gelu138_mipshead"
    assert config["version"] == "none"
    assert config["geometry_revision"] is None
    assert config["schema"] == "mts-glt-distill-repair-control-v1"

    result_root = (ROOT / config["result_root"]).resolve()
    assert result_root == (ROOT / "results/glt_v2_r2_c0_gelu138_mipshead").resolve()

    protected = [(ROOT / value).resolve() for value in config["protect_result_roots"]]
    assert PROTECTED_FORMAL_ROOT.resolve() in protected
    assert all(result_root != root and root not in result_root.parents for root in protected)
    assert (PROTECTED_FORMAL_ROOT / "c0/student").is_dir()


def test_run_stage_refuses_protected_result_root():
    with pytest.raises(ValueError, match="protected formal artefact root"):
        run_stage(CONFIG, "student", result_root="results/mts_glt_distill_repair_control/c0")
