"""Focused contracts for the O8 + frozen PolyPaiNN DND student."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from scripts.finetune_glt_o8_control import ALLOWED_TASKS_8, DEFAULT_TASKS_7
from scripts.run_glt_sci_o8_control_grid import validate_gpus
from src.dataset.glt_o8_control import build_o8_pretrain_sample
from src.dataset.o8_dnd_student import build_student_pretrain_sample, student_pretrain_collate
from src.modules.glt_o8_control import (
    common_initialized_o8_pretrainer, load_o8_deployment,
)
from src.modules.o8_dnd_student import (
    O8DNDStudentPretrainer, common_initialized_o8_dnd_student,
    student_deployment_package, student_global_objective,
)


def _topology():
    edges = torch.tensor([[0, 1, 1, 0, 1, 2, 2, 1],
                          [1, 0, 2, 1, 0, 1, 1, 2]])
    relation = edges.size(1)
    path = torch.full((relation, 3), -1, dtype=torch.long)
    path[:, :2] = edges.t()
    x = torch.zeros(3, 137)
    x[:, 0] = 1
    return Data(
        mips_x=x, z=torch.tensor([6, 8, 7]),
        mips_backbone_mask=torch.zeros(3, dtype=torch.long),
        lga_edge_index=edges, lga_spd=torch.ones(relation, dtype=torch.long),
        lga_path_index=path, lga_path_mask=path >= 0,
        canonical_to_trimer_base_atom_id=torch.arange(3),
        canonical_ru_atom_index=torch.arange(3), graph_available=True,
        mts_canonical_periodic=True,
    )


def _static(topology):
    relation = topology.lga_edge_index.size(1)
    features = torch.zeros(relation, 2, 14)
    features[..., 0] = 1
    features[..., 7] = 1
    return {"bond_path_features": features.numpy(),
            "bond_path_mask": np.ones((relation, 2), dtype=bool)}


def _trimer():
    return Data(
        trimer_pos=torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 1.], [5., 0., 0.],
        ]),
        trimer_heavy_indices=torch.tensor([0, 1, 2, 3, 4]),
        trimer_atomic_number=torch.tensor([6, 8, 7, 1, 6]),
        trimer_central_ru_mask=torch.tensor([True, True, True, False, False]),
        mips_to_trimer_central_index=torch.tensor([0, 1, 2]),
        o8_heavy_mask=torch.tensor([True, True, True]),
        trimer_geometry_valid=True, trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
    )


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, data):
        count = int(data.central_index.numel())
        return {"central_scalar_states": torch.zeros(count, 256, device=data.pos.device),
                "central_batch": data.central_batch}


def _sample(position=7):
    top, trimer = _topology(), _trimer()
    target = {"brics_groups": ((0,), (1,), (2,)),
              "fingerprint_packed": np.zeros(256, dtype=np.uint8)}
    return build_student_pretrain_sample(
        top, trimer, "*C(*)O", seed=42, key="a" * 64, position=position,
        static=_static(top), target=target,
    )


def test_common_initialization_is_exactly_arm_b_common_modules():
    arm_b = common_initialized_o8_pretrainer(42)
    student = common_initialized_o8_dnd_student(42)
    assert student.common_init_digest == arm_b.common_init_digest
    for left, right in (
        (student.encoder.o8, arm_b.encoder.o8),
        (student.encoder.norm2, arm_b.encoder.norm2),
        (student.atom_head, arm_b.atom_head),
        (student.fp_head, arm_b.fp_head),
    ):
        assert all(torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict())


def test_student_o8_input_mask_and_fingerprint_match_arm_b():
    o8, labels = build_o8_pretrain_sample(
        _topology(), _static(_topology()), {"brics_groups": ((0,), (1,), (2,)),
        "fingerprint_packed": np.zeros(256, dtype=np.uint8)},
        seed=42, key="a" * 64, position=7,
    )
    dnd_o8, dnd_labels, _, _ = _sample(7)
    for name in ("mips_x", "lga_edge_index", "lga_path_index", "bond_path_features", "bond_path_mask"):
        assert torch.equal(getattr(o8, name), getattr(dnd_o8, name))
    for name in ("atom_mask", "atom_label", "fingerprint", "fallback"):
        if torch.is_tensor(labels[name]):
            assert torch.equal(labels[name], dnd_labels[name])
        else:
            assert labels[name] == dnd_labels[name]
    assert dnd_labels["position"] == 7 and dnd_labels["key"] == "a" * 64


def test_student_collate_preserves_canonical_teacher_alignment():
    first, second = _sample(3), _sample(4)
    data, labels, teacher = student_pretrain_collate([first, second])
    assert labels["student_central_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert teacher.central_batch.tolist() == [0, 0, 0, 1, 1, 1]
    assert labels["teacher_positions"].tolist() == [3, 4]
    assert data.mips_x.size(0) == 6


def test_mapping_failure_is_hard_error():
    trimer = _trimer()
    trimer.mips_to_trimer_central_index = torch.tensor([0, 0, 2])
    with pytest.raises(ValueError, match="duplicates"):
        build_student_pretrain_sample(
            _topology(), trimer, "*C(*)O", seed=42, key="a" * 64,
            position=7, static=_static(_topology()),
            target={"brics_groups": ((0,), (1,), (2,)),
                    "fingerprint_packed": np.zeros(256, dtype=np.uint8)},
        )


def test_distillation_adapter_shape_and_teacher_is_detached():
    o8, labels, teacher_data, _ = _sample()
    student = common_initialized_o8_dnd_student(42, dropout=0.0)
    teacher = _Teacher()
    result = student(o8, labels, teacher_data, teacher)
    assert result["distill_values"].shape == (3,)
    loss = (result["sums"] / result["counts"].clamp_min(1)).sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.distill_adapter.parameters())
    assert any(parameter.grad is not None for parameter in student.encoder.o8.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert student.distill_adapter.in_features == 512
    assert student.distill_adapter.out_features == 256


def test_teacher_parameters_are_not_in_student_optimizer():
    student = O8DNDStudentPretrainer(dropout=0.0)
    teacher = _Teacher()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    selected = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(next(teacher.parameters())) not in selected
    assert all(parameter.requires_grad for parameter in student.distill_adapter.parameters())


def test_teacher_forward_does_not_consume_student_rng():
    data, labels, teacher_data, _ = _sample()
    student = common_initialized_o8_dnd_student(42, dropout=0.0)
    teacher = _Teacher()
    torch.manual_seed(9182)
    reference = torch.rand(16)
    torch.manual_seed(9182)
    student(data, labels, teacher_data, teacher)
    observed = torch.rand(16)
    assert torch.equal(observed, reference)


def test_global_three_component_objective():
    sums = torch.tensor([2.0, 4.0, 6.0])
    counts = torch.tensor([2.0, 4.0, 3.0])
    expected = (2.0 / 2.0) + 0.1 * (4.0 / 4.0) + 6.0 / 3.0
    assert torch.equal(student_global_objective(sums, counts), torch.tensor(expected))
    with pytest.raises(ValueError):
        student_global_objective(torch.ones(2), counts)


def test_student_deployment_is_o8_norm2_only_and_loads_strictly():
    student = common_initialized_o8_dnd_student(42, dropout=0.0)
    package = student_deployment_package(student, 5000, teacher_sha256="t")
    assert package["arm"] == "DND"
    assert not any(any(token in name.lower() for token in ("teacher", "adapter", "head", "trimer", "coord"))
                    for name in package["state_dict"])
    restored = student.encoder.__class__(dropout=0.0)
    load_o8_deployment(restored, package, expected_step=5000)
    for key in package["state_dict"]:
        assert torch.equal(package["state_dict"][key], restored.state_dict()[key])


def test_eight_task_cli_contract_and_legacy_default():
    assert DEFAULT_TASKS_7 == ("eat", "eea", "egb", "ei", "eps", "nc", "xc")
    assert "egc" in ALLOWED_TASKS_8 and len(ALLOWED_TASKS_8) == 8
    assert validate_gpus(["0", "1", "2", "3"]) == ["0", "1", "2", "3"]
    with pytest.raises(ValueError, match="distinct"):
        validate_gpus(["0", "1", "1", "3"])
