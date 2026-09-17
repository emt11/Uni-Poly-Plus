"""O8-only student with a frozen PolyPaiNN node-distillation target.

The student keeps the matched Arm-B O8 encoder and masked-chemistry/Morgan
heads unchanged.  A small, student-side ``512 -> 256`` adapter is trained to
match the frozen teacher's central-RU scalar states.  The teacher is passed to
``forward`` rather than registered as a submodule, so it can never enter the
student optimizer or deployment package.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .glt_o8_control import (
    O8ControlPretrainer,
    common_initialized_o8_pretrainer,
    state_digest,
)
from .glt_dual_pretrain import per_graph


class O8DNDStudentPretrainer(nn.Module):
    """Matched O8 pretrainer plus a frozen-teacher node objective."""

    architecture_name = "O8-BondPath-DND-Student"

    def __init__(self, *, dropout: float = 0.1):
        super().__init__()
        common = O8ControlPretrainer(dropout=dropout)
        self.encoder = common.encoder
        self.atom_head = common.atom_head
        self.fp_head = common.fp_head
        # Only this projection is new and trainable relative to Arm B.
        self.distill_adapter = nn.Linear(512, 256)
        self.student_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.teacher_norm = nn.LayerNorm(256, elementwise_affine=False)
        self.common_init_digest = None
        del common

    def forward(self, data, labels, teacher_data, teacher_model):
        if teacher_model is None:
            raise ValueError("DND student requires a frozen teacher model")
        encoded = self.encoder.encode(data, atom_mask=labels["atom_mask"])
        graphs = int(data.graph_available.numel())
        mask = labels["atom_mask"].bool()
        atom_logits = self.atom_head(encoded["atom_states"], data.lga_edge_index)
        if bool(mask.any()):
            chem_values = F.cross_entropy(
                atom_logits[mask].float(), labels["atom_label"][mask].long(), reduction="none"
            )
            chem, chem_valid = per_graph(
                chem_values, data.canonical_graph_index[mask], graphs
            )
        else:
            chem = atom_logits.new_zeros(graphs)
            chem_valid = torch.zeros(graphs, dtype=torch.bool, device=atom_logits.device)

        fp_logits = self.fp_head(self.encoder.fuse(encoded)).float()
        fp_target = labels["fingerprint"].float()
        if fp_target.ndim == 1:
            fp_target = fp_target.unsqueeze(0)
        fp_values = F.binary_cross_entropy_with_logits(
            fp_logits, fp_target, reduction="none"
        ).mean(-1)
        fp_valid = data.graph_available.bool()

        student_index = labels["student_central_index"].long().reshape(-1)
        with torch.no_grad():
            teacher_result = teacher_model(teacher_data)
            teacher_states = teacher_result["central_scalar_states"].detach().float()
        if student_index.numel() != teacher_states.size(0):
            raise ValueError("student/teacher central-node count mismatch")
        if student_index.numel() == 0:
            raise ValueError("DND student has no central heavy nodes")
        if int(student_index.min()) < 0 or int(student_index.max()) >= encoded["atom_states"].size(0):
            raise ValueError("student central-node index is out of range")
        student_states = self.student_norm(
            self.distill_adapter(encoded["atom_states"][student_index]).float()
        )
        teacher_states = self.teacher_norm(teacher_states)
        node_values = (student_states - teacher_states).square().mean(dim=-1)
        central_batch = teacher_data.central_batch.long().reshape(-1)
        if central_batch.numel() != node_values.numel():
            raise ValueError("teacher central batch/node count mismatch")
        distill, distill_valid = per_graph(node_values, central_batch, graphs)

        sums = torch.stack((chem[chem_valid].sum(), fp_values[fp_valid].sum(),
                            distill[distill_valid].sum()))
        counts = torch.stack((chem_valid.sum(), fp_valid.sum(), distill_valid.sum()))
        targets = torch.stack((mask.sum(), fp_valid.sum() * 2048,
                               node_values.new_tensor(node_values.numel())))
        return {
            "sums": sums,
            "counts": counts,
            "targets": targets,
            "atom_logits": atom_logits,
            "fingerprint_logits": fp_logits,
            "distill_values": node_values,
            "teacher_states": teacher_states,
        }


def common_initialized_o8_dnd_student(seed: int = 42, *, dropout: float = 0.1):
    """Create a student whose Arm-B common tensors are bitwise identical."""

    common = common_initialized_o8_pretrainer(int(seed), dropout=dropout)
    model = O8DNDStudentPretrainer(dropout=dropout)
    for name in ("o8", "norm2"):
        getattr(model.encoder, name).load_state_dict(
            {key: value.detach().cpu().clone()
             for key, value in getattr(common.encoder, name).state_dict().items()},
            strict=True,
        )
    model.atom_head.load_state_dict(
        {key: value.detach().cpu().clone() for key, value in common.atom_head.state_dict().items()},
        strict=True,
    )
    model.fp_head.load_state_dict(
        {key: value.detach().cpu().clone() for key, value in common.fp_head.state_dict().items()},
        strict=True,
    )
    model.common_init_digest = state_digest(
        model.encoder.o8, model.encoder.norm2, model.atom_head, model.fp_head
    )
    if model.common_init_digest != common.common_init_digest:
        raise ValueError("DND common O8 initialization digest mismatch")
    del common
    return model


def student_global_objective(sums, global_counts, world_size=1,
                             weights=(1.0, 0.1, 1.0)):
    """Reduce chemistry/fingerprint/distillation by global graph counts."""

    if sums.numel() != 3 or global_counts.numel() != 3:
        raise ValueError("DND objective expects three components")
    if int(world_size) <= 0:
        raise ValueError("DND objective has invalid world size")
    return (sums * sums.new_tensor(weights) * int(world_size)
            / global_counts.clamp_min(1)).sum()


def student_deployment_package(pretrainer, step: int, *, teacher_sha256: str | None = None):
    """Emit an Arm-B-compatible deployment with only O8 and norm2 tensors."""

    state = {}
    for prefix, module in (("o8", pretrainer.encoder.o8),
                           ("norm2", pretrainer.encoder.norm2)):
        state.update({f"{prefix}.{key}": value.detach().cpu().clone()
                      for key, value in module.state_dict().items()})
    return {
        "architecture": pretrainer.encoder.architecture_name,
        "fusion_mode": "o8_control",
        "step": int(step),
        "use_md200": False,
        "arm": "DND",
        "student_architecture": pretrainer.architecture_name,
        "common_init_digest": pretrainer.common_init_digest,
        "teacher_deployment_sha256": teacher_sha256,
        "state_dict": state,
    }


__all__ = [
    "O8DNDStudentPretrainer",
    "common_initialized_o8_dnd_student",
    "student_global_objective",
    "student_deployment_package",
]
