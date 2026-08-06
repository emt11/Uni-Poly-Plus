#!/usr/bin/env python3
"""Production readiness checks for MIPS-Trimer-SCAGE (MTS).

This command is intentionally read-only.  It never opens an LMDB writer and
it reports a failed cache bundle as "not ready" instead of attempting to
repair or rebuild it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from scripts.finalize_mips_trimer_cache import _active_writer_pids  # noqa: E402
from src.dataset.mips_cache_validation import verify_frozen_cache_bundle  # noqa: E402
from src.modules.mips_local_graph import MIPSLocalGraphEncoder  # noqa: E402
from src.dataset.mips_trimer_contract import ROUTE_NAME, ROUTE_SHORT_NAME  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _check_cli(python: str) -> None:
    for script in ("scripts/pretrain.py", "scripts/train.py"):
        result = subprocess.run(
            [python, script, "--help"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{script} --help failed: {result.stderr[-500:]}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used for CLI smoke checks",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="only run environment/config/model checks",
    )
    args = parser.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0,1,2":
        raise RuntimeError(
            f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) requires CUDA_VISIBLE_DEVICES=0,1,2; "
            f"got {visible!r}"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError(
            "doctor requires exactly three visible CUDA devices (0,1,2); "
            f"available={torch.cuda.device_count()}"
        )
    _check_cli(args.python)

    # Constructing the fixed encoder catches accidental reintroduction of
    # retired PBC/SCAGE/descriptor branches without allocating a dataset.
    encoder = MIPSLocalGraphEncoder()
    assert encoder.max_hops == 2 and len(encoder.layers) == 6
    assert encoder.emb_dim == 512 and encoder.num_heads == 8
    assert encoder.descriptor_components == "md200"
    if _active_writer_pids():
        raise RuntimeError("cache writer is still active")

    split_root = PROJECT_ROOT / "data/splits/mips_shared5"
    missing_splits = [task for task in TASKS if not (split_root / f"{task}.json").is_file()]
    if missing_splits:
        raise RuntimeError("missing shared 5-fold manifests: " + ", ".join(missing_splits))

    if not args.skip_cache:
        specs = _specs(PROJECT_ROOT)
        store = Path(specs["trimer"]["root"]) / "validation" / "store.json"
        verify_frozen_cache_bundle(
            specs,
            store_path=store,
            required_layers=specs.keys(),
            pretraining_cohort_hash=None,
        )
    print(f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) doctor: ready (3 GPUs, fixed O8/Trimer/MD200 contract)")


if __name__ == "__main__":
    main()
