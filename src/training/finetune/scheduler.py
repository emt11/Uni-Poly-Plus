"""Four-slot Python scheduler for independent MTS fine-tune folds.

The scheduler owns only operational concerns: unit ordering, verified-shard
skips, subprocess slots, failure propagation, and process-group cleanup.  A
single worker receives one task/seed/fold and the training engine remains
unaware of the global 8x5 campaign.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


def dispatch_units(tasks, folds, *, task_order=None, fold_order=None):
    """Return deterministic single-fold units in the requested order."""

    tasks = list(tasks)
    folds = [int(value) for value in folds]
    if task_order is not None:
        order = {str(name): index for index, name in enumerate(task_order)}
        tasks = sorted(tasks, key=lambda name: order.get(str(name), len(order)))
    if fold_order is not None:
        order = {int(value): index for index, value in enumerate(fold_order)}
        folds = sorted(folds, key=lambda value: order.get(int(value), len(order)))
    return [(str(task), int(fold)) for task in tasks for fold in folds]


@dataclass(frozen=True)
class ScheduledUnit:
    seed: int
    task: str
    fold: int

    @property
    def label(self) -> str:
        return f"{self.seed}|{self.task}|{self.fold}"


class SchedulerError(RuntimeError):
    """A unit failed; all in-flight children have already been reaped."""

    def __init__(self, unit: ScheduledUnit, returncode: int):
        self.unit = unit
        self.returncode = int(returncode)
        super().__init__(
            f"failed unit={unit.label} status={self.returncode}; stopping all slots"
        )


def _terminate_process_group(process: subprocess.Popen, *, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_subprocess_scheduler(
    units: Iterable[ScheduledUnit],
    *,
    gpu_ids: Sequence[str | int],
    command_factory: Callable[[ScheduledUnit, str], Sequence[str]],
    should_skip: Callable[[ScheduledUnit], bool] | None = None,
    log_dir: str | os.PathLike[str],
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    max_slots: int | None = None,
) -> dict:
    """Run independent units with bounded GPU slots and fail-stop cleanup."""

    gpu_ids = [str(value) for value in gpu_ids]
    if not gpu_ids:
        raise ValueError("at least one GPU slot is required")
    slots = min(len(gpu_ids), int(max_slots or len(gpu_ids)))
    if slots <= 0:
        raise ValueError("max_slots must be positive")
    pending = []
    skipped = []
    for unit in units:
        if should_skip is not None and should_skip(unit):
            skipped.append(unit.label)
        else:
            pending.append(unit)

    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ if env is None else env)
    active: dict[subprocess.Popen, tuple[ScheduledUnit, str, object]] = {}
    free = list(gpu_ids)
    next_index = 0
    completed: list[str] = []
    started: list[str] = []

    def cleanup() -> None:
        for process in list(active):
            _terminate_process_group(process)
        active.clear()

    try:
        while next_index < len(pending) or active:
            while next_index < len(pending) and free and len(active) < slots:
                unit = pending[next_index]
                next_index += 1
                gpu = free.pop(0)
                command = list(command_factory(unit, gpu))
                log_path = log_root / (
                    f"finetune_seed{unit.seed}_{unit.task}_fold{unit.fold}.log"
                )
                handle = log_path.open("a", encoding="utf-8")
                child_env = dict(base_env)
                child_env["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=child_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[process] = (unit, gpu, handle)
                started.append(unit.label)
            if not active:
                continue
            finished = None
            for process in list(active):
                if process.poll() is not None:
                    finished = process
                    break
            if finished is None:
                time.sleep(0.05)
                continue
            unit, gpu, handle = active.pop(finished)
            handle.close()
            free.append(gpu)
            status = int(finished.returncode)
            if status != 0:
                cleanup()
                raise SchedulerError(unit, status)
            completed.append(unit.label)
            print(
                f"[finetune] completed unit={unit.label}; "
                f"queued={len(pending) - next_index} active={len(active)}",
                flush=True,
            )
            # Give peer processes from the same slot wave a brief chance to
            # publish their exit status before refilling the freed slot.  This
            # makes fail-stop deterministic: a simultaneous failure cannot be
            # hidden behind a newly launched unit.
            if active:
                time.sleep(0.02)
                for peer in list(active):
                    peer_status = peer.poll()
                    if peer_status is not None and int(peer_status) != 0:
                        peer_unit, _peer_gpu, _peer_handle = active[peer]
                        cleanup()
                        raise SchedulerError(peer_unit, int(peer_status))
    except BaseException:
        cleanup()
        raise
    return {
        "started": started,
        "completed": completed,
        "skipped": skipped,
        "pending": len(pending) - len(completed),
        "slots": slots,
        "gpu_ids": gpu_ids,
    }
