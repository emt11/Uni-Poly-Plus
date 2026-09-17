"""Frozen-Trimer adapter for the independent PolyPaiNN teacher."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .glt_dual_pretrain import sample_generator


def _as_long(value, name):
    result = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"{name} is empty")
    return result


def validate_central_mapping(topology, trimer):
    """Return heavy-local central indices after strict frozen identity checks."""
    required_topology = ("z", "mips_x")
    required_trimer = (
        "trimer_pos", "trimer_heavy_indices", "trimer_atomic_number",
        "trimer_central_ru_mask", "mips_to_trimer_central_index",
        "trimer_geometry_valid", "trimer_geometry_is_3d", "trimer_2d_fallback",
    )
    missing = [name for name in required_topology if not hasattr(topology, name)]
    missing += [name for name in required_trimer if not hasattr(trimer, name)]
    if missing:
        raise ValueError("teacher identity fields missing: " + ",".join(missing))
    topology_z = _as_long(topology.z, "topology.z")
    canonical_count = int(topology.mips_x.size(0))
    if topology_z.numel() != canonical_count:
        raise ValueError("teacher topology z/mips_x count mismatch")
    positions = torch.as_tensor(trimer.trimer_pos, dtype=torch.float32)
    if positions.ndim != 2 or positions.size(1) != 3:
        raise ValueError("trimer_pos must be [N,3]")
    heavy_all = _as_long(trimer.trimer_heavy_indices, "trimer_heavy_indices")
    if heavy_all.unique().numel() != heavy_all.numel() or int(heavy_all.max()) >= positions.size(0):
        raise ValueError("trimer heavy-atom indices are not unique/in range")
    atomic = _as_long(trimer.trimer_atomic_number, "trimer_atomic_number")
    if atomic.numel() != positions.size(0):
        raise ValueError("trimer atomic number/position count mismatch")
    central_mask = torch.as_tensor(trimer.trimer_central_ru_mask, dtype=torch.bool).reshape(-1)
    if central_mask.numel() != positions.size(0):
        raise ValueError("trimer central mask/position count mismatch")
    mapping_all = _as_long(trimer.mips_to_trimer_central_index, "mips_to_trimer_central_index")
    if mapping_all.numel() != canonical_count:
        raise ValueError("canonical-to-Trimer mapping length mismatch")
    if mapping_all.unique().numel() != mapping_all.numel():
        raise ValueError("canonical-to-Trimer mapping contains duplicates")
    if int(mapping_all.min()) < 0 or int(mapping_all.max()) >= positions.size(0):
        raise ValueError("canonical-to-Trimer mapping is out of range")
    if not bool(central_mask[mapping_all].all()):
        raise ValueError("canonical-to-Trimer mapping leaves centre RU")
    if not bool(torch.isin(mapping_all, heavy_all).all()):
        raise ValueError("canonical-to-Trimer mapping contains non-heavy atoms")
    if not torch.equal(atomic[mapping_all], topology_z):
        raise ValueError("canonical-to-Trimer atomic-number mapping mismatch")
    heavy_local = torch.full((positions.size(0),), -1, dtype=torch.long)
    heavy_local[heavy_all] = torch.arange(heavy_all.numel(), dtype=torch.long)
    central_heavy = heavy_local[mapping_all]
    if bool((central_heavy < 0).any()) or central_heavy.unique().numel() != central_heavy.numel():
        raise ValueError("canonical-to-heavy mapping is not one-to-one")
    return {
        "heavy_all_indices": heavy_all,
        "heavy_atomic_number": atomic[heavy_all],
        "central_heavy_index": central_heavy,
        "central_all_index": mapping_all,
        "positions": positions[heavy_all],
    }


def build_teacher_sample(topology, trimer, *, seed, key, position, sigma=0.03):
    """Build one deterministic noisy atom cloud and centre-RU target."""
    if not bool(getattr(trimer, "trimer_geometry_valid", False)):
        raise ValueError("teacher requires valid frozen 3D geometry")
    if not bool(torch.as_tensor(getattr(trimer, "trimer_geometry_is_3d", False)).item()):
        raise ValueError("teacher requires 3D geometry, not a 2D fallback")
    if bool(torch.as_tensor(getattr(trimer, "trimer_2d_fallback", False)).item()):
        raise ValueError("teacher rejects trimer 2D fallback")
    if float(sigma) < 0 or not torch.isfinite(torch.tensor(float(sigma))):
        raise ValueError("invalid teacher noise sigma")
    identity = validate_central_mapping(topology, trimer)
    clean = identity["positions"].float()
    generator = sample_generator(int(seed), str(key), int(position))
    epsilon = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    noisy = clean + float(sigma) * epsilon
    node_count = clean.size(0)
    data = Data(
        z=identity["heavy_atomic_number"].clone(),
        pos=noisy,
        batch=torch.zeros(node_count, dtype=torch.long),
        central_index=identity["central_heavy_index"].clone(),
        central_batch=torch.zeros(identity["central_heavy_index"].numel(), dtype=torch.long),
    )
    data.clean_pos = clean
    data.key = str(key)
    data.position = int(position)
    target = {
        "epsilon": epsilon[identity["central_heavy_index"]].clone(),
        "clean_pos": clean,
        "noisy_pos": noisy,
        "central_index": identity["central_heavy_index"].clone(),
        "key": str(key),
        "position": int(position),
    }
    return data, target


def teacher_collate(records):
    if not records:
        raise ValueError("empty teacher batch")
    data_values, target_values = zip(*records)
    z, pos, batch, central, central_batch, clean = [], [], [], [], [], []
    eps, keys, positions = [], [], []
    node_offset = 0
    for graph, (item, target) in enumerate(records):
        count = int(item.z.numel())
        z.append(item.z)
        pos.append(item.pos)
        clean.append(item.clean_pos)
        batch.append(torch.full((count,), graph, dtype=torch.long))
        central.append(item.central_index + node_offset)
        central_batch.append(torch.full((item.central_index.numel(),), graph, dtype=torch.long))
        eps.append(target["epsilon"])
        keys.append(target["key"])
        positions.append(int(target["position"]))
        node_offset += count
    result = Data(
        z=torch.cat(z), pos=torch.cat(pos), batch=torch.cat(batch),
        central_index=torch.cat(central), central_batch=torch.cat(central_batch),
        clean_pos=torch.cat(clean),
    )
    labels = {
        "epsilon": torch.cat(eps),
        "keys": keys,
        "positions": torch.tensor(positions, dtype=torch.long),
    }
    return result, labels


class TeacherRankMicrobatchStream(Dataset):
    """Deterministic rank-local microbatches for worker prefetch/resume."""

    def __init__(self, source, *, seed, world, rank, microbatch, accumulation,
                 start_step, max_steps, sigma):
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
        self.sigma = float(sigma)

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
            topology, trimer, _ = self.source[index]
            records.append(build_teacher_sample(
                topology, trimer, seed=self.seed, key=key,
                position=position, sigma=self.sigma,
            ))
        return teacher_collate(records)


__all__ = [
    "validate_central_mapping", "build_teacher_sample", "teacher_collate",
    "TeacherRankMicrobatchStream",
]
