#!/usr/bin/env python3
"""Build geometry-revision-2 N+ sidecars directly from frozen Trimers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mts_glt_distill_sidecars import PackedWriter  # noqa: E402
from scripts.build_mts_glt_v3_sidecars import dataset_for_build, row_key  # noqa: E402
from src.dataset.periodic_line_glt_image import PeriodicLineImageSidecar  # noqa: E402
from src.dataset.periodic_line_distill_v2 import (  # noqa: E402
    GEOMETRY_REVISION, SCHEMA_PREFIX, build_periodic_line_distill_v2_pair,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="data")
    parser.add_argument("--dataset", default="PI1M_v2")
    parser.add_argument(
        "--output-root",
        default="data/processed/mips_trimer_scage/periodic_line_glt_distill_v2",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=0)
    args = parser.parse_args(argv)
    if int(args.workers) > 1:
        torch.set_num_threads(1)
    dataset = dataset_for_build(args)
    dataset_count = len(dataset) if not args.limit else min(len(dataset), int(args.limit))
    start = int(args.start)
    stop = int(args.stop) if args.stop else dataset_count
    if not 0 <= start < stop <= dataset_count:
        raise ValueError(f"invalid row range [{start}, {stop}) for {dataset_count}")
    count = stop - start
    # Obtain the immutable cohort keys without a second Dataset traversal.
    keys = dataset._cohort["row_keys_array"][start:stop]
    chemistry_sidecar = PeriodicLineImageSidecar(
        "data/processed/mips_trimer_scage/periodic_line_glt_image_v1/PI1M_v2",
        build_key_index=False,
    )
    if len(chemistry_sidecar) < stop:
        raise RuntimeError("chemistry source sidecar is shorter than the cohort")
    root = Path(args.output_root)
    writers = {
        version: PackedWriter(
            root / version, count, version, keys,
            schema_prefix=SCHEMA_PREFIX,
            metadata_extra={
                "geometry_revision": GEOMETRY_REVISION,
                "source": "frozen_trimer_coordinates_direct",
                "source_sidecars": ["periodic_line_glt_image_v1/PI1M_v2 (chemistry only)"],
                "relation_semantics": "distinct_physical_bonds_incident_on_center_ru",
            },
        )
        for version in ("n_plus_2", "n_plus_1")
    }
    try:
        def build(global_index):
            data = dataset[global_index]
            expected = bytes(keys[global_index - start])
            if row_key(dataset, global_index, data) != expected:
                raise RuntimeError(f"cohort identity mismatch at row {global_index}")
            chemistry_index = chemistry_sidecar.index_for_key(expected, row_hint=global_index)
            return build_periodic_line_distill_v2_pair(
                data, data, str(data.smiles),
                chemistry_row=chemistry_sidecar.model_row(chemistry_index),
            )
        workers = max(1, int(args.workers))
        next_report = 10000
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for local_start in range(0, count, max(1, int(args.chunk_size))):
                local_stop = min(count, local_start + max(1, int(args.chunk_size)))
                global_range = range(start + local_start, start + local_stop)
                for local_index, rows in zip(range(local_start, local_stop), executor.map(build, global_range)):
                    for version, writer in writers.items():
                        writer.add(local_index, rows[version])
                if local_stop >= next_report or local_stop == count:
                    print(f"distill-revision2 range={start}:{stop} {local_stop}/{count}", flush=True)
                    while next_report <= local_stop:
                        next_report += 10000
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
