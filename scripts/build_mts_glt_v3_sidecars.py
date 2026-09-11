#!/usr/bin/env python3
"""Build new GLT-v3 image or full PI1M MD200 sidecars.

This command only reads the current immutable O8/Trimer cache.  It never
generates coordinates and refuses an existing destination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (
    UniDataset,
    build_complete_trimer_glt_sample,
    build_periodic_line_image_sample,
)  # noqa: E402
from src.dataset.md200_sidecar import SCHEMA as MD_SCHEMA  # noqa: E402
from src.dataset.periodic_line_glt_image import SIDECAR_SCHEMA  # noqa: E402
from src.dataset.periodic_line_glt_complete import (
    SIDECAR_SCHEMA as COMPLETE_SIDECAR_SCHEMA,
)  # noqa: E402
from src.modules.original_mips_md200 import OriginalMIPSMD200  # noqa: E402


TOKEN_DTYPES = {
    "token_atom_a": np.int32, "token_atom_b": np.int32,
    "token_shift": np.int16, "token_endpoint_z_a": np.int16,
    "token_endpoint_z_b": np.int16, "token_distance": np.float32,
    "token_bond_type": np.int8, "token_stereo": np.int8,
    "token_conjugated": np.int8, "token_anchor_q_a": np.int8,
    "token_anchor_q_b": np.int8, "token_valid": np.bool_,
}
COMPLETE_TOKEN_DTYPES = {
    "token_atom_a": np.int32, "token_atom_b": np.int32,
    "token_shift": np.int16, "token_endpoint_z_a": np.int16,
    "token_endpoint_z_b": np.int16, "token_distance": np.float32,
    "token_bond_type": np.int8, "token_stereo": np.int8,
    "token_conjugated": np.int8, "token_ring": np.int8,
    "token_bond_features": np.float32,
    "token_anchor_q_a": np.int8, "token_anchor_q_b": np.int8,
    "token_valid": np.bool_, "token_center_internal": np.bool_,
}
RELATION_DTYPES = {
    "relation_source": np.int32, "relation_target": np.int32,
    "relation_center_atom": np.int32,
    "relation_source_image_shift": np.int16,
    "relation_angle": np.float32, "relation_valid": np.bool_,
}
COMPLETE_RELATION_DTYPES = dict(RELATION_DTYPES)


def dataset_for_build(args):
    return UniDataset(
        root=args.cache_root, dataset=args.dataset, smiles_model_name="",
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        geom_input="repeat_unit", use_feature_cache=True,
        feature_source_dataset=args.dataset, fp_mode="disabled",
        cache_layers="ru_base,topology,trimer", cache_validate="sample",
        mips_core="paper_corrected", mips_max_hops=2,
        mips_use_descriptors=False, mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage", graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted", trimer_num_candidates=4,
        trimer_max_heavy_atoms=384, modalities=("graph",),
        experiment_id="mts_glt_v3_sidecar_build", feature_config_hash="manual",
    )


def row_key(dataset, index, data):
    if getattr(dataset, "_cohort_row_mode", False):
        return bytes(dataset._cohort["row_keys_array"][index])
    from src.dataset.lmdb_cache import sample_key_from_smiles
    return sample_key_from_smiles(str(data.smiles))


def build_line(args, dataset, count, *, complete=False):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    token_dtypes = COMPLETE_TOKEN_DTYPES if complete else TOKEN_DTYPES
    relation_dtypes = COMPLETE_RELATION_DTYPES if complete else RELATION_DTYPES
    builder = build_complete_trimer_glt_sample if complete else build_periodic_line_image_sample
    schema = COMPLETE_SIDECAR_SCHEMA if complete else SIDECAR_SCHEMA
    token_width = {"token_bond_features": 14} if complete else {}
    token_offsets = np.lib.format.open_memmap(
        root / "token_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,)
    )
    relation_offsets = np.lib.format.open_memmap(
        root / "relation_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,)
    )
    geometry_valid = np.lib.format.open_memmap(
        root / "geometry_valid.npy", mode="w+", dtype=np.bool_, shape=(count,)
    )
    keys = np.lib.format.open_memmap(root / "sample_keys.npy", mode="w+", dtype=np.uint8, shape=(count, 32))
    raw_paths = {
        name: root / f".{name}.raw"
        for name in (*token_dtypes, *relation_dtypes)
    }
    handles = {name: path.open("wb") for name, path in raw_paths.items()}
    token_offsets[0] = relation_offsets[0] = 0
    try:
        for index in range(count):
            data = dataset[index]
            keys[index] = np.frombuffer(row_key(dataset, index, data), dtype=np.uint8)
            row = builder(data, data, str(data.smiles))
            geometry_valid[index] = row["geometry_valid"]
            token_offsets[index + 1] = token_offsets[index] + len(row["tokens"]["token_atom_a"])
            relation_offsets[index + 1] = relation_offsets[index] + len(row["relations"]["relation_source"])
            for name, dtype in token_dtypes.items():
                np.asarray(row["tokens"][name], dtype=dtype).tofile(handles[name])
            for name, dtype in relation_dtypes.items():
                np.asarray(row["relations"][name], dtype=dtype).tofile(handles[name])
            if (index + 1) % 10000 == 0 or index + 1 == count:
                print(
                    f"{'line-complete' if complete else 'line-image'} "
                    f"{index + 1}/{count} valid={int(geometry_valid[:index + 1].sum())}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()
    for name, dtype in {**token_dtypes, **relation_dtypes}.items():
        rows = int(token_offsets[-1]) if name.startswith("token_") else int(relation_offsets[-1])
        width = int(token_width.get(name, 1))
        shape = (rows,) if width == 1 else (rows, width)
        if rows == 0:
            np.save(root / f"{name}.npy", np.empty(shape, dtype=dtype))
            raw_paths[name].unlink()
            continue
        raw = np.memmap(raw_paths[name], mode="r", dtype=dtype, shape=(rows * width,))
        target = np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        target_flat = target.reshape(-1)
        for start in range(0, rows * width, 1_000_000):
            target_flat[start:start + 1_000_000] = raw[start:start + 1_000_000]
        target.flush()
        del target, raw
        raw_paths[name].unlink()
    (root / "metadata.json").write_text(json.dumps({
        "schema": schema, "builder_version": 1,
        "sample_count": count, "valid_count": int(geometry_valid.sum()),
        "geometry_semantics": (
            "complete_trimer_physical_bonds"
            if complete else "single_center_anchored_image_no_moments"
        ),
        **({"bond_feature_dim": 14} if complete else {}),
    }, indent=2, sort_keys=True) + "\n")
    (root / ".done").write_text("complete\n")


def build_md200(args, dataset, count):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    keys = np.lib.format.open_memmap(root / "sample_keys.npy", mode="w+", dtype=np.uint8, shape=(count, 32))
    values = np.lib.format.open_memmap(root / "values.f32.npy", mode="w+", dtype=np.float32, shape=(count, 200))
    valid = np.lib.format.open_memmap(root / "valid.bool.npy", mode="w+", dtype=np.bool_, shape=(count,))
    descriptor = OriginalMIPSMD200()
    for index in range(count):
        data = dataset[index]
        keys[index] = np.frombuffer(row_key(dataset, index, data), dtype=np.uint8)
        before = descriptor.stats()["md_zero_fallback_count"]
        try:
            values[index] = descriptor.one(str(data.smiles)).astype(np.float32)
            valid[index] = descriptor.stats()["md_zero_fallback_count"] == before
        except Exception:
            values[index] = 0.0
            valid[index] = False
        if (index + 1) % 10000 == 0 or index + 1 == count:
            print(f"md200 {index + 1}/{count} valid={int(valid[:index + 1].sum())}", flush=True)
    (root / "metadata.json").write_text(json.dumps({
        "schema": MD_SCHEMA, "sample_count": count, "dimension": 200,
        "valid_count": int(valid.sum()), "statistics": descriptor.stats(),
        "protocol": descriptor.protocol,
    }, indent=2, sort_keys=True) + "\n")
    (root / ".done").write_text("complete\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["line-image", "line-complete", "md200"])
    parser.add_argument("--cache-root", default="data")
    parser.add_argument("--dataset", default="PI1M_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    dataset = dataset_for_build(args)
    count = len(dataset) if not args.limit else min(len(dataset), args.limit)
    if args.mode == "line-image":
        build_line(args, dataset, count)
    elif args.mode == "line-complete":
        build_line(args, dataset, count, complete=True)
    else:
        build_md200(args, dataset, count)


if __name__ == "__main__":
    main()
