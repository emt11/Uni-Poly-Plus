"""Training data for the O8 + frozen PolyPaiNN distillation route."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .glt_o8_control import build_o8_pretrain_sample, o8_pretrain_collate
from .poly_painn_teacher import build_teacher_sample, teacher_collate, validate_central_mapping


def build_student_pretrain_sample(topology, trimer, smiles, *, seed, key, position,
                                  ratio=0.30, static=None, target=None):
    """Build matched O8 input and clean Teacher input for one absolute position."""

    o8_data, labels = build_o8_pretrain_sample(
        topology, static, target, seed=seed, key=key, position=position, ratio=ratio,
    )
    identity = validate_central_mapping(topology, trimer)
    teacher_data, teacher_labels = build_teacher_sample(
        topology, trimer, seed=seed, key=key, position=position, sigma=0.0,
    )
    canonical_heavy = identity["canonical_heavy_indices"].long()
    if canonical_heavy.numel() != teacher_labels["central_index"].numel():
        raise ValueError("student/teacher central mapping count mismatch")
    labels = dict(labels)
    labels["student_central_index"] = canonical_heavy
    labels["teacher_central_index"] = teacher_labels["central_index"].clone()
    labels["key"] = str(key)
    labels["position"] = int(position)
    return o8_data, labels, teacher_data, teacher_labels


def student_pretrain_collate(records):
    """Collate O8 and Teacher views while preserving canonical identity offsets."""

    if not records:
        raise ValueError("empty DND student batch")
    o8_records = [(item[0], item[1]) for item in records]
    teacher_records = [(item[2], item[3]) for item in records]
    data, labels = o8_pretrain_collate(o8_records)
    central_indices = []
    atom_offset = 0
    for o8_data, item_labels, _, _ in records:
        local = torch.as_tensor(item_labels["student_central_index"], dtype=torch.long)
        central_indices.append(local + atom_offset)
        atom_offset += int(o8_data.mips_x.size(0))
    labels["student_central_index"] = torch.cat(central_indices)
    teacher_data, teacher_labels = teacher_collate(teacher_records)
    labels["teacher_central_index"] = teacher_data.central_index.clone()
    # The teacher's collated central order is the contract used by the module;
    # retain explicit graph/key/position fields for audit and resume checks.
    labels["teacher_keys"] = list(teacher_labels["keys"])
    labels["teacher_positions"] = teacher_labels["positions"]
    return data, labels, teacher_data


class StudentRankMicrobatchStream(Dataset):
    """Deterministic rank-local prefetch stream for Student training."""

    def __init__(self, source, *, seed, world, rank, microbatch, accumulation,
                 start_step, max_steps, ratio):
        from src.training.glt_dual_runtime import OrderedSampleStream

        self.source = source
        self.stream = OrderedSampleStream(len(source), int(seed))
        self.seed = int(seed)
        self.world = int(world)
        self.rank = int(rank)
        self.microbatch = int(microbatch)
        self.accumulation = int(accumulation)
        self.global_batch = self.microbatch * self.world * self.accumulation
        self.start_step = int(start_step)
        self.steps = max(0, int(max_steps) - self.start_step)
        self.ratio = float(ratio)

    def __len__(self):
        return self.steps * self.accumulation

    def __getitem__(self, item):
        step = self.start_step + int(item) // self.accumulation
        offset = int(item) % self.accumulation
        records = []
        for local in range(self.microbatch):
            position = (
                step * self.global_batch + offset * self.world * self.microbatch
                + self.rank * self.microbatch + local
            )
            index = self.stream.index_at(position)
            key = self.source.samples[index][0].hex()
            topology, trimer, smiles = self.source[index]
            records.append(build_student_pretrain_sample(
                topology, trimer, smiles, seed=self.seed, key=key,
                position=position, ratio=self.ratio,
                static=self.source.static_for(index), target=self.source.target_for(index),
            ))
        return student_pretrain_collate(records)


__all__ = [
    "build_student_pretrain_sample", "student_pretrain_collate",
    "StudentRankMicrobatchStream",
]
