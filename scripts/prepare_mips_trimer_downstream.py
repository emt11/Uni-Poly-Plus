#!/usr/bin/env python3
"""Prepare downstream-union features before cache finalization.

This command intentionally does not hold the finalizer's exclusive lifecycle
lock.  The normal LMDB writer acquires the shared lifecycle lock and can
therefore populate missing downstream records safely.  A separate coordinator
lock prevents two preparation jobs from running concurrently.
"""

from __future__ import annotations

import atexit
import fcntl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finalize_mips_trimer_cache import (  # noqa: E402
    _active_writer_pids,
    _assert_no_writer_locks,
    _prepare_downstream,
    _specs,
)


def main() -> int:
    root = PROJECT_ROOT / "data/processed/mips_trimer_scage"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".downstream_prepare.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit("another downstream preparation is active")
    atexit.register(handle.close)

    active = _active_writer_pids()
    if active:
        raise SystemExit(
            "cache writer still active; downstream preparation will not race it: "
            + "; ".join(f"pid={pid}" for pid, _ in active)
        )
    specs = _specs(PROJECT_ROOT)
    _assert_no_writer_locks(specs)
    _prepare_downstream()
    _assert_no_writer_locks(_specs(PROJECT_ROOT))
    print("downstream union feature preparation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
