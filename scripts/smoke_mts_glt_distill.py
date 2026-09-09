#!/usr/bin/env python3
"""Two-update GPU smoke for both N+ teacher/student paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mts_glt_v3_sidecars import dataset_for_build  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.md200_sidecar import DatasetWithMD200  # noqa: E402
from src.dataset.periodic_line_distill import PeriodicLineDistillSidecar  # noqa: E402
from src.training.pretrain.glt_distill_engine import StudentContainer, TeacherContainer  # noqa: E402
from src.modules.mts_glt_distill import NPlusGLTTeacher  # noqa: E402


def attach(data, row):
    data.glt3_geometry_valid = bool(row["geometry_valid"])
    for name, value in row["tokens"].items():
        dtype = torch.float32 if name == "token_distance" else torch.bool if name in {"token_valid", "token_center_internal"} else torch.long
        setattr(data, "glt3_" + name, torch.as_tensor(value.copy(), dtype=dtype))
    for name, value in row["relations"].items():
        dtype = torch.float32 if name == "relation_angle" else torch.bool if name == "relation_valid" else torch.long
        setattr(data, "glt3_" + name, torch.as_tensor(value.copy(), dtype=dtype))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--version", choices=("n_plus_2", "n_plus_1"), required=True)
    args = parser.parse_args(argv)
    namespace = argparse.Namespace(cache_root="data", dataset="PI1M_v2")
    base = DatasetWithMD200(
        dataset_for_build(namespace),
        "data/processed/mips_trimer_scage/md200_pi1m_v1/PI1M_v2",
    )
    sidecar = PeriodicLineDistillSidecar(Path(args.sidecar_root) / args.version)
    rows = []
    for index in range(len(sidecar)):
        row = sidecar.model_row(index)
        if row["geometry_valid"]:
            rows.append(attach(base[index], row))
        if len(rows) == 4:
            break
    batch = mips_trimer_collate(rows).cuda()
    teacher = TeacherContainer().cuda().train()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = teacher(batch, torch.Generator(device="cuda").manual_seed(42 + step))
        loss = output["chem_sum"] / output["graph_count"] + output["length_sum"] / output["graph_count"] + output["angle_sum"] / max(1, output["angle_graph_count"])
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    teacher_grad = sum(float(p.grad.abs().sum()) for p in teacher.teacher.parameters() if p.grad is not None)
    frozen = NPlusGLTTeacher().cuda(); frozen.load_state_dict(teacher.teacher.state_dict())
    student = StudentContainer(frozen).cuda().train()
    student.zero_grad(set_to_none=True)
    audit = student(batch, 0, torch.Generator(device="cuda").manual_seed(141))
    (audit["local_sum"] / audit["local_count"] + audit["global"]).backward()
    distill_o8_grad = sum(
        float(p.grad.abs().sum())
        for p in student.student.o8.parameters() if p.grad is not None
    )
    assert distill_o8_grad > 0
    assert all(p.grad is None for p in student.student.md_residual.parameters())
    student.zero_grad(set_to_none=True)
    audit = student(batch, 0, torch.Generator(device="cuda").manual_seed(141))
    (audit["atom_sum"] / audit["atom_count"]).backward()
    atom_o8_grad = sum(
        float(p.grad.abs().sum())
        for p in student.student.o8.parameters() if p.grad is not None
    )
    atom_md_grad = sum(
        float(p.grad.abs().sum())
        for p in student.student.md_residual.parameters() if p.grad is not None
    )
    assert atom_o8_grad > 0 and atom_md_grad > 0
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = student(batch, step, torch.Generator(device="cuda").manual_seed(142 + step))
        loss = output["atom_sum"] / output["atom_count"] + 0.1 * output["local_sum"] / output["local_count"] + 0.1 * output["global"]
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
    o8_grad = sum(float(p.grad.abs().sum()) for p in student.student.o8.parameters() if p.grad is not None)
    md_grad = sum(float(p.grad.abs().sum()) for p in student.student.md_residual.parameters() if p.grad is not None)
    assert teacher_grad > 0 and o8_grad > 0 and md_grad > 0
    assert all(not p.requires_grad and p.grad is None for p in student.teacher.parameters())
    deploy = {key for key in student.student.state_dict()}
    assert not any("teacher" in key or "line_projection" in key or "atom_head" in key for key in deploy)
    print({
        "version": args.version, "teacher_grad": teacher_grad,
        "distill_o8_grad": distill_o8_grad,
        "atom_o8_grad": atom_o8_grad, "atom_md_grad": atom_md_grad,
        "o8_grad": o8_grad, "md_grad": md_grad, "status": "pass",
    })


if __name__ == "__main__":
    main()
