"""Topology-only O8 control data path for GLT-SCI-O8CTRL.

The control route deliberately opens the frozen topology artifact and the
training-ready static/target sidecars, but never opens or materialises the
Trimer artifact.  Static sidecars are used only for O8 bond-path chemistry;
all coordinate/geometry fields are rejected at the boundary rather than
silently ignored.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .cache_lifecycle import CacheLifecycleError, ReadonlyArtifact
from .glt_dual_cache import load_active_dual_store, load_dual_cohort
from .glt_dual_pretrain import _unpack_fingerprint, motif_mask, sample_generator
from .glt_dual_static import CHUNK_CACHE_CAPACITY, load_static_caches


O8_STATIC_FIELDS = ("bond_path_features", "bond_path_mask")
O8_FORBIDDEN_FIELDS = {
    "trimer_pos", "bond_distance", "line_angle", "geometry_valid",
    "geometry_invalid_reason", "line_source", "line_target", "line_path",
    "line_path_mask", "line_path_group", "line_is_self", "angle_pos_triplet",
}


def _clone_tensor(value, *, dtype=None):
    # ``DualStaticCache`` returns read-only mmap views; make the copy before
    # converting so PyTorch never exposes a non-writable NumPy-backed tensor.
    if isinstance(value, np.ndarray):
        value = np.array(value, copy=True)
    tensor = torch.as_tensor(value, dtype=dtype)
    return tensor.clone()


def _tensor_payload_bytes(value):
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def build_o8_sample(topology, static, *, key=None):
    """Build one label-free O8 sample from topology and bond-path sidecar.

    ``static`` is intentionally a mapping returned by ``DualStaticCache``;
    geometry-dependent sidecar fields are not needed and are rejected if a
    caller tries to pass them through this route.  No source SMILES parsing or
    Trimer lookup occurs here.
    """

    if static is None:
        raise CacheLifecycleError("O8 control requires dual_static_v1")
    missing = [name for name in O8_STATIC_FIELDS if name not in static]
    if missing:
        raise CacheLifecycleError("O8 static row is missing " + ", ".join(missing))
    n = int(torch.as_tensor(topology.mips_x).size(0))
    edge_index = _clone_tensor(topology.lga_edge_index, dtype=torch.long)
    spd = _clone_tensor(topology.lga_spd, dtype=torch.long).reshape(-1)
    path_index = _clone_tensor(topology.lga_path_index, dtype=torch.long)
    path_mask = _clone_tensor(topology.lga_path_mask, dtype=torch.bool)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise CacheLifecycleError("O8 topology lga_edge_index must be [2,R]")
    relation_count = int(edge_index.shape[1])
    if spd.numel() != relation_count or path_index.shape[0] != relation_count:
        raise CacheLifecycleError("O8 topology relation/path lengths disagree")
    features = _clone_tensor(static["bond_path_features"], dtype=torch.float32)
    feature_mask = _clone_tensor(static["bond_path_mask"], dtype=torch.bool)
    if features.shape != (relation_count, 2, 14) or feature_mask.shape != (relation_count, 2):
        raise CacheLifecycleError("O8 static bond-path shape does not match topology")
    if not bool(torch.isfinite(features).all()):
        raise CacheLifecycleError("O8 static bond-path features are nonfinite")
    if int(path_index.numel()) and (
        int(path_index[path_mask].min()) < 0 or int(path_index[path_mask].max()) >= n
    ):
        raise CacheLifecycleError("O8 topology path index is out of bounds")
    data = Data(
        mips_x=_clone_tensor(topology.mips_x, dtype=torch.float32),
        mips_backbone_mask=_clone_tensor(topology.mips_backbone_mask, dtype=torch.long),
        lga_edge_index=edge_index,
        lga_spd=spd,
        lga_path_index=path_index,
        lga_path_mask=path_mask,
        lga_relation_mask=torch.zeros(relation_count, dtype=torch.bool),
        bond_path_features=features,
        bond_path_mask=feature_mask,
        graph_available=torch.tensor([bool(getattr(topology, "graph_available", False))], dtype=torch.bool),
        canonical_graph_index=torch.zeros(n, dtype=torch.long),
        canonical_local_index=torch.arange(n, dtype=torch.long),
        canonical_first_node_index=torch.tensor([0], dtype=torch.long),
        canonical_ru_atom_index=_clone_tensor(
            getattr(topology, "canonical_ru_atom_index", torch.arange(n)), dtype=torch.long
        ),
        batch=torch.zeros(n, dtype=torch.long),
        num_nodes=n,
    )
    if data.canonical_ru_atom_index.numel() != n:
        raise CacheLifecycleError("O8 canonical atom identity length mismatch")
    # Keep the topology schema markers required by downstream diagnostics, but
    # do not copy arbitrary payload fields from the frozen record.
    for name in ("mts_canonical_periodic", "feature_schema",
                 "mips_local_lga_schema_version", "topology_representation"):
        if hasattr(topology, name):
            setattr(data, name, getattr(topology, name))
    if key is not None:
        data.sample_key = bytes(key)
    return data


def build_o8_pretrain_sample(topology, static, target, *, seed, key, position,
                             ratio=0.30):
    """Build an O8-only masked-chemistry/fingerprint training sample."""

    if target is None or "brics_groups" not in target or "fingerprint_packed" not in target:
        raise CacheLifecycleError("O8 pretraining target row is incomplete")
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("O8 masking ratio must lie in (0,1)")
    generator = sample_generator(int(seed), str(key), int(position))
    groups = tuple(tuple(int(atom) for atom in group) for group in target["brics_groups"])
    mask, fallback = motif_mask(topology, groups, generator, float(ratio))
    data = build_o8_sample(topology, static, key=bytes.fromhex(str(key)) if isinstance(key, str) else key)
    atom_features = torch.as_tensor(topology.mips_x, dtype=torch.float32)
    labels = {
        "atom_mask": mask,
        "atom_label": atom_features[:, :101].argmax(dim=-1).long(),
        "fingerprint": _unpack_fingerprint(target["fingerprint_packed"]),
        "fallback": bool(fallback),
        "skip_reasons": ["chem:single_atom"] if not bool(mask.any()) else [],
        "sample_key": bytes(key) if not isinstance(key, str) else bytes.fromhex(key),
        "position": int(position),
    }
    return data, labels


def o8_collate(samples):
    """Collate topology-only O8 ``Data`` records with canonical offsets."""

    if not samples:
        raise ValueError("empty O8 batch")
    output = Data()
    fields = {name: [] for name in (
        "mips_x", "mips_backbone_mask", "lga_spd", "lga_path_mask",
        "lga_relation_mask", "bond_path_features", "bond_path_mask",
    )}
    edge_values, path_values = [], []
    graph_values, local_values, first_values, batches = [], [], [], []
    keys, available = [], []
    atom_offset = 0
    for graph, item in enumerate(samples):
        n = int(item.mips_x.size(0))
        for name in fields:
            fields[name].append(getattr(item, name))
        edge_values.append(item.lga_edge_index.long() + atom_offset)
        path = item.lga_path_index.long()
        path_values.append(torch.where(path >= 0, path + atom_offset, path))
        graph_values.append(torch.full((n,), graph, dtype=torch.long))
        local_values.append(torch.arange(n, dtype=torch.long))
        first_values.append(torch.tensor(atom_offset, dtype=torch.long))
        batches.append(torch.full((n,), graph, dtype=torch.long))
        available.append(bool(item.graph_available))
        keys.append(getattr(item, "sample_key", None))
        atom_offset += n
    for name, values in fields.items():
        setattr(output, name, torch.cat(values, dim=0))
    output.lga_edge_index = torch.cat(edge_values, dim=1)
    output.lga_path_index = torch.cat(path_values, dim=0)
    output.canonical_graph_index = torch.cat(graph_values, dim=0)
    output.canonical_local_index = torch.cat(local_values, dim=0)
    output.canonical_first_node_index = torch.stack(first_values)
    output.batch = torch.cat(batches, dim=0)
    output.graph_available = torch.tensor(available, dtype=torch.bool)
    output.num_nodes = atom_offset
    output.sample_keys = keys
    if all(hasattr(item, "y") and item.y is not None for item in samples):
        output.y = torch.stack([item.y.reshape(1) for item in samples])
    return output


def o8_pretrain_collate(records):
    if not records:
        raise ValueError("empty O8 pretraining batch")
    samples, labels = zip(*records)
    data = o8_collate(samples)
    graph_offsets = []
    offset = 0
    for item in samples:
        graph_offsets.append(offset)
        offset += int(item.mips_x.size(0))
    output = {
        "atom_mask": torch.cat([item["atom_mask"] for item in labels]),
        "atom_label": torch.cat([item["atom_label"] for item in labels]),
        "fingerprint": torch.stack([item["fingerprint"] for item in labels]),
        "fallback_count": sum(bool(item["fallback"]) for item in labels),
        "skip_reasons": [reason for item in labels for reason in item["skip_reasons"]],
        "sample_keys": [item["sample_key"] for item in labels],
        "positions": [int(item["position"]) for item in labels],
    }
    del graph_offsets
    return data, output


class O8OnlySource(Dataset):
    """Read a cohort's topology and static sidecars without opening Trimer."""

    def __init__(self, cohort_root, cache_root, *, static_root, target_root=None,
                 task=None, chunk_cache_capacity=CHUNK_CACHE_CAPACITY):
        self.cache_root = Path(cache_root).resolve()
        self.cohort = load_dual_cohort(cohort_root, self.cache_root)
        self.store = load_active_dual_store(self.cache_root)
        self.bundle_hash = self.store["bundle_hash"]
        self.topology = self.static_cache = self.target_cache = None
        try:
            self.topology = ReadonlyArtifact(self.cache_root, self.store["artifacts"]["topology"])
            self.static_cache, self.target_cache = load_static_caches(
                static_root, target_root,
                parent_bundle_hash=self.cohort["manifest"]["main_bundle_hash"],
                cohort_manifest_hash=self.cohort["manifest_hash"],
                chunk_cache_capacity=chunk_cache_capacity,
            )
            records = list(self.cohort["records"])
            if task is not None:
                records = [row for row in records if str(row.get("task")) == str(task)]
                if not records:
                    raise ValueError(f"O8 cohort has no rows for task={task}")
            self.entries = records
            self.samples = [
                (bytes.fromhex(str(row["sample_key"])), str(row["source_smiles"]))
                for row in records
            ]
            expected = self.cohort["manifest"].get("ordered_sample_key_hash")
            for cache in (self.static_cache, self.target_cache):
                if cache is None:
                    continue
                bound = cache.manifest.get("cohort_ordered_sample_key_hash")
                if bound is not None and bound != expected:
                    raise CacheLifecycleError("O8 cache cohort ordered-key binding mismatch")
            for key, _ in self.samples:
                self.static_cache.index_for_key(key)
                if target_root is not None:
                    self.target_cache.index_for_key(key)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        key, _ = self.samples[int(index)]
        topology = self.topology[key]
        # Topology is the only frozen payload accepted by this source.  A
        # Trimer/geometry field cannot enter by accidental ``Data`` merging.
        if any(hasattr(topology, name) for name in O8_FORBIDDEN_FIELDS):
            raise CacheLifecycleError("O8 topology source contains forbidden geometry payload")
        return topology

    def static_for(self, index):
        key = self.samples[int(index)][0]
        return self.static_cache.get_by_key(key)

    def target_for(self, index):
        if self.target_cache is None:
            return None
        key = self.samples[int(index)][0]
        return self.target_cache.get_by_key(key)

    def close(self):
        for name in ("topology", "static_cache", "target_cache"):
            resource = getattr(self, name, None)
            setattr(self, name, None)
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass


class O8RankMicrobatchStream(Dataset):
    """Deterministic pretraining microbatches for optional worker prefetch."""

    def __init__(self, source, *, seed, world, rank, microbatch, accumulation,
                 start_step, max_steps, ratio):
        from src.training.glt_dual_runtime import OrderedSampleStream
        self.source = source
        self.seed, self.world, self.rank = int(seed), int(world), int(rank)
        self.microbatch, self.accumulation = int(microbatch), int(accumulation)
        self.batch_size = self.microbatch * self.world * self.accumulation
        self.start_step = int(start_step)
        self.steps = max(0, int(max_steps) - self.start_step)
        self.stream = OrderedSampleStream(len(source), self.seed)

    def __len__(self):
        return self.steps * self.accumulation

    def __getitem__(self, item):
        step = self.start_step + int(item) // self.accumulation
        offset = int(item) % self.accumulation
        records = []
        for local in range(self.microbatch):
            position = (step * self.batch_size + offset * self.world * self.microbatch
                        + self.rank * self.microbatch + local)
            index = self.stream.index_at(position)
            key = self.source.samples[index][0]
            records.append(build_o8_pretrain_sample(
                self.source[index], self.source.static_for(index),
                self.source.target_for(index), seed=self.seed, key=key,
                position=position,
            ))
        return o8_pretrain_collate(records)


class O8CleanLabeledDataset(Dataset):
    """Clean topology/static data with bounded label-free caching."""

    def __init__(self, source, targets, *, cache_capacity_bytes=0):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        self.targets = self.raw_targets.copy()
        self.cache_capacity_bytes = max(0, int(cache_capacity_bytes))
        if self.raw_targets.size != len(source):
            raise ValueError("O8 target count differs from frozen source")
        self._cache = OrderedDict()
        self._cache_bytes = self._cache_hits = self._cache_misses = 0
        self._cache_evictions = self._cache_skipped_bytes = 0

    def __len__(self):
        return len(self.source)

    def set_target_override(self, targets):
        values = np.asarray(targets).reshape(-1)
        if values.size != len(self.source):
            raise ValueError("O8 target override count differs from frozen source")
        self.targets = values

    def __getitem__(self, index):
        index = int(index)
        key = self.source.samples[index][0]
        entry = self._cache.pop(key, None) if self.cache_capacity_bytes else None
        if entry is not None:
            self._cache[key] = entry
            self._cache_hits += 1
            data = entry[0].clone()
        else:
            self._cache_misses += int(bool(self.cache_capacity_bytes))
            built = build_o8_sample(self.source[index], self.source.static_for(index), key=key)
            payload = _tensor_payload_bytes(built.to_dict())
            if self.cache_capacity_bytes and payload <= self.cache_capacity_bytes:
                while self._cache and self._cache_bytes + payload > self.cache_capacity_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._cache_bytes -= old[1]
                    self._cache_evictions += 1
                stored = built.clone()
                self._cache[key] = (stored, payload)
                self._cache_bytes += payload
                data = stored.clone()
            else:
                if self.cache_capacity_bytes:
                    self._cache_skipped_bytes += payload
                data = built.clone()
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data

    def cache_stats(self):
        return {
            "enabled": bool(self.cache_capacity_bytes),
            "capacity_bytes": int(self.cache_capacity_bytes),
            "payload_bytes": int(self._cache_bytes),
            "entries": len(self._cache), "hits": int(self._cache_hits),
            "misses": int(self._cache_misses), "evictions": int(self._cache_evictions),
            "skipped_bytes": int(self._cache_skipped_bytes),
        }


__all__ = [
    "O8OnlySource", "O8CleanLabeledDataset", "O8RankMicrobatchStream",
    "build_o8_sample", "build_o8_pretrain_sample", "o8_collate",
    "o8_pretrain_collate", "O8_FORBIDDEN_FIELDS",
]
