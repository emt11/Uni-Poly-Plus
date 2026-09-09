#!/usr/bin/env python3
"""Build N+1/N+2 packed sidecars from existing immutable GLT sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.periodic_line_distill import SCHEMA_PREFIX, transform_row  # noqa: E402
from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar  # noqa: E402
from src.dataset.periodic_line_glt_image import PeriodicLineImageSidecar  # noqa: E402


TOKEN_DTYPES = {
    "token_atom_a": np.int32, "token_atom_b": np.int32,
    "token_shift": np.int16, "token_endpoint_z_a": np.int16,
    "token_endpoint_z_b": np.int16, "token_distance": np.float32,
    "token_bond_type": np.int8, "token_stereo": np.int8,
    "token_conjugated": np.int8, "token_anchor_q_a": np.int8,
    "token_anchor_q_b": np.int8, "token_valid": np.bool_,
    "token_center_internal": np.bool_,
}
RELATION_DTYPES = {
    "relation_source": np.int32, "relation_target": np.int32,
    "relation_center_atom": np.int32,
    "relation_source_image_shift": np.int16,
    "relation_angle": np.float32, "relation_valid": np.bool_,
}


class PackedWriter:
    def __init__(self, root: Path, count: int, version: str, keys, *, schema_prefix=SCHEMA_PREFIX, metadata_extra=None):
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        self.root = root
        self.version = version
        self.schema_prefix = schema_prefix
        self.metadata_extra = dict(metadata_extra or {})
        self.count = count
        np.save(root / "sample_keys.npy", np.asarray(keys))
        self.token_offsets = np.lib.format.open_memmap(root / "token_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,))
        self.relation_offsets = np.lib.format.open_memmap(root / "relation_offsets.npy", mode="w+", dtype=np.int64, shape=(count + 1,))
        self.valid = np.lib.format.open_memmap(root / "geometry_valid.npy", mode="w+", dtype=np.bool_, shape=(count,))
        self.token_offsets[0] = self.relation_offsets[0] = 0
        self.raw = {name: root / f".{name}.raw" for name in (*TOKEN_DTYPES, *RELATION_DTYPES)}
        self.handles = {name: path.open("wb") for name, path in self.raw.items()}

    def add(self, index, row):
        self.valid[index] = row["geometry_valid"]
        self.token_offsets[index + 1] = self.token_offsets[index] + len(row["tokens"]["token_atom_a"])
        self.relation_offsets[index + 1] = self.relation_offsets[index] + len(row["relations"]["relation_source"])
        for name, dtype in TOKEN_DTYPES.items():
            np.asarray(row["tokens"][name], dtype=dtype).tofile(self.handles[name])
        for name, dtype in RELATION_DTYPES.items():
            np.asarray(row["relations"][name], dtype=dtype).tofile(self.handles[name])

    def finish(self):
        for handle in self.handles.values():
            handle.close()
        token_total, relation_total = int(self.token_offsets[-1]), int(self.relation_offsets[-1])
        for name, dtype in {**TOKEN_DTYPES, **RELATION_DTYPES}.items():
            length = token_total if name.startswith("token_") else relation_total
            raw = np.memmap(self.raw[name], mode="r", dtype=dtype, shape=(length,))
            target = np.lib.format.open_memmap(self.root / f"{name}.npy", mode="w+", dtype=dtype, shape=(length,))
            for start in range(0, length, 1_000_000):
                target[start:start + 1_000_000] = raw[start:start + 1_000_000]
            target.flush()
            del target, raw
            self.raw[name].unlink()
        metadata = {
            "schema": self.schema_prefix + self.version,
            "sample_count": self.count,
            "valid_count": int(self.valid.sum()),
            "token_count": token_total,
            "relation_count": relation_total,
            "source_sidecars": [
                "periodic_line_glt_v1/PI1M_v2",
                "periodic_line_glt_image_v1/PI1M_v2",
            ],
            "geometry_semantics": self.version,
            **self.metadata_extra,
        }
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        (self.root / ".done").write_text("complete\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default="data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2")
    parser.add_argument("--image", default="data/processed/mips_trimer_scage/periodic_line_glt_image_v1/PI1M_v2")
    parser.add_argument("--output-root", default="data/processed/mips_trimer_scage/periodic_line_glt_distill_v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    old = PeriodicLineGLTSidecar(args.v1)
    image = PeriodicLineImageSidecar(args.image)
    old_keys = old.arrays["sample_keys"]
    if len(old) != len(image) or not np.array_equal(old_keys, image.sample_keys):
        raise RuntimeError("source sidecar sample identities do not align")
    count = len(image) if not args.limit else min(len(image), int(args.limit))
    root = Path(args.output_root)
    writers = {
        version: PackedWriter(root / version, count, version, image.sample_keys[:count])
        for version in ("n_plus_2", "n_plus_1")
    }
    try:
        for index in range(count):
            image_row, old_row = image.model_row(index), old.model_row(index)
            for version, writer in writers.items():
                writer.add(index, transform_row(image_row, old_row, version))
            if (index + 1) % 10000 == 0 or index + 1 == count:
                print(f"distill-sidecars {index + 1}/{count}", flush=True)
        for writer in writers.values():
            writer.finish()
    except Exception:
        for writer in writers.values():
            for handle in writer.handles.values():
                if not handle.closed:
                    handle.close()
        raise


if __name__ == "__main__":
    main()
