#!/usr/bin/env python3
"""Read one frozen sidecar row and prove Dataset/collate row alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.lmdb_cache import LmdbLayerStore
from src.dataset.mts_relation_geometry import RelationGeometrySidecar
from src.dataset.dataset import UniDataset
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "data/processed/mips_trimer_scage/relation_geometry/ef70d42497bc15f51fc38728f1400aa73ea1431f0f8cbf10c310b0a784e36bf1/PI1M_v2/0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049"
TOPOLOGY = ROOT / "data/processed/mips_trimer_scage/topology/1658c6a602aebb7b73a6e77a3612f5faa613bb93b1d78c57a198ee4d787fa47a"


class _RealSidecarRows(Dataset):
    def __init__(self, root, rows):
        self.root = str(root)
        self.rows = list(rows)
        self.reader = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = RelationGeometrySidecar(
                self.root, expected_cohort="PI1M_v2",
                expected_artifact_hash="134016a47280e61fdc4154047b1fb48a9c8a27f1ac9ed3b8ee9abd166860a289",
                verify_array_hashes=False,
            )
        row = self.reader.row(self.rows[index])
        return {
            "sample_key": torch.from_numpy(row["sample_key"].copy()),
            "relation_row": torch.from_numpy(row["relations"]["relation_row"].copy()),
            "path_offsets": torch.from_numpy(row["relations"]["relation_path_offsets"].copy()),
        }


def _read_with_workers(workers):
    loader = DataLoader(
        _RealSidecarRows(SIDECAR, [0, 1]), batch_size=None,
        num_workers=workers, persistent_workers=bool(workers),
    )
    return [
        {
            key: value.cpu().tolist()
            for key, value in item.items()
        }
        for item in loader
    ]


def _prepare_item(item):
    item = item.clone()
    item.topology_representation = "canonical_lifted"
    item.canonical_graph_index = torch.zeros(int(item.canonical_ru_atom_index.numel()), dtype=torch.long)
    item.batch = torch.zeros(int(item.x.size(0)), dtype=torch.long)
    item.y = torch.zeros(1)
    item.mips_md = torch.zeros(200)
    item.mips_md_valid = torch.tensor(True)
    item.smiles = str(getattr(item, "smiles", "CC"))
    item.mts_sample_hash64 = torch.tensor(0)
    return item


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/mts_multiscale_topology/g_family_readiness_v1/sidecar_alignment.json")
    args = parser.parse_args(argv)
    sidecar = RelationGeometrySidecar(SIDECAR, verify_array_hashes=True)
    store = LmdbLayerStore(TOPOLOGY, require_done=True)
    try:
        key = bytes(np.asarray(sidecar.arrays["sample_keys"][0], dtype=np.uint8).tobytes())
        topology = _prepare_item(store[key])
        dataset = UniDataset.__new__(UniDataset)
        dataset.g_family_arm = "g1"
        dataset._relation_geometry_sidecar = sidecar
        dataset._g3_permutation = None
        attached = dataset._attach_relation_geometry(topology, key, row_hint=0)
        batch = mips_trimer_collate([attached, attached])
        expected_rows = np.asarray(sidecar.row(0)["relations"]["relation_row"], dtype=np.int64)
        observed_rows = batch.mts_relation_geometry_relation_row[: expected_rows.size].cpu().numpy()
        if not np.array_equal(expected_rows, observed_rows):
            raise RuntimeError("sidecar relation rows changed during Dataset/collate")
        workers_zero = _read_with_workers(0)
        workers_one = _read_with_workers(1)
        if workers_zero != workers_one:
            raise RuntimeError("workers=0 and workers=1 read different sidecar rows")
        report = {
            "schema": "mts-g-family-sidecar-alignment-v1",
            "artifact": sidecar.artifact_hash,
            "cohort_hash": sidecar.cohort_hash,
            "sample_relation_count": int(expected_rows.size),
            "sample_path_count": int(len(sidecar.row(0)["paths"]["path_cos_angle"])),
            "topology_lga_relation_count": int(topology.lga_edge_index.size(1)),
            "collated_relation_rows_first_sample": observed_rows.tolist(),
            "collated_path_offsets": batch.mts_relation_geometry_path_offsets.cpu().tolist(),
            "workers_0_and_multiprocess_safe": workers_zero == workers_one,
            "worker_sample_keys": [row["sample_key"] for row in workers_zero],
        }
    finally:
        store.close()
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
