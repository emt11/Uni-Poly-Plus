#!/usr/bin/env python3
"""Wait for the KFuse 5k run, then launch two isolated fine-tuning smokes.

The two jobs are the planned ``eat/fold0`` validation-only runs for the
Concat and KFuse deployment packages.  They are launched in separate tmux
windows and never read outer-test data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = (
    REPO / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _pid_command(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _checkpoint_metadata(path: Path, expected_fusion: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing deployment checkpoint: {path}")
    # Importing torch here only reads the package; no model is instantiated.
    import torch

    package = torch.load(path, map_location="cpu", weights_only=False)
    if (package.get("fusion_mode") != expected_fusion
            or package.get("step") != 5000
            or package.get("use_md200") is not False
            or not isinstance(package.get("state_dict"), dict)):
        raise ValueError(
            f"deployment metadata mismatch for {path}: "
            f"fusion={package.get('fusion_mode')!r}, step={package.get('step')!r}"
        )
    return {
        "path": str(path),
        "fusion_mode": package["fusion_mode"],
        "step": int(package["step"]),
        "state_tensor_count": len(package["state_dict"]),
    }


def _shell_command(*, config: Path, checkpoint: Path, raw_root: Path,
                   cohort_root: Path, cache_root: Path, static_root: Path,
                   split_root: Path, output: Path, log_path: Path,
                   status_path: Path, task: str, fold: int, gpu: str) -> str:
    command = [
        sys.executable, "scripts/finetune_glt_dual.py",
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--raw-root", str(raw_root), "--cohort-root", str(cohort_root),
        "--cache-root", str(cache_root), "--dual-static-root", str(static_root),
        "--output", str(output), "--split-root", str(split_root),
        "--task", task, "--fold", str(fold), "--smoke",
    ]
    command_text = shlex.join(command)
    log_text = shlex.quote(str(log_path))
    status_text = shlex.quote(str(status_path))
    status_tmp = shlex.quote(str(status_path) + ".tmp")
    return (
        f"cd {shlex.quote(str(REPO))} && set -o pipefail && export PYTHONPATH=. && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} "
        f"{command_text} 2>&1 | tee {log_text}; "
        f"rc=${{PIPESTATUS[0]}}; "
        f"printf 'EXIT_CODE=%s\\n' \"$rc\" | tee -a {log_text}; "
        f"printf '{{\"exit_code\":%s}}\\n' \"$rc\" > {status_tmp} && "
        f"mv {status_tmp} {status_text}; exit \"$rc\""
    )


def _validate_paths(args: argparse.Namespace) -> None:
    required = [
        Path(args.concat_config), Path(args.kfuse_config), Path(args.raw_root),
        Path(args.cohort_root), Path(args.cache_root), Path(args.dual_static_root),
        Path(args.split_root) / f"{args.task}.json",
        Path(args.raw_root) / f"smi_{args.task}.csv",
        Path(args.concat_checkpoint), Path(args.kfuse_log),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing smoke prerequisite(s): " + ", ".join(missing))


def _wait_for_kfuse(args: argparse.Namespace, report: dict) -> dict:
    pid = int(args.kfuse_pid)
    expected_output = str(Path(args.kfuse_output).resolve())
    # The tmux wrapper records the command exactly as it was launched.  It
    # may contain either the repository-relative output path or its absolute
    # spelling, so identity matching must accept both representations.
    expected_output_relative = os.path.relpath(expected_output, REPO)
    command = _pid_command(pid)
    if command is not None and (
        "pretrain_glt_dual.py" not in command
        or "--output" not in command
        or (expected_output not in command and expected_output_relative not in command)
        or "kfuse" not in command.lower()
    ):
        raise RuntimeError(
            f"--kfuse-pid does not identify the expected KFuse run: pid={pid}; "
            f"command={command!r}"
        )
    report["kfuse_pid"] = pid
    report["kfuse_pid_command_at_start"] = command
    started = time.time()
    while _pid_command(pid) is not None:
        if args.timeout_seconds and time.time() - started > args.timeout_seconds:
            raise TimeoutError(f"timed out waiting for KFuse pid {pid}")
        time.sleep(args.poll_seconds)
    log_text = Path(args.kfuse_log).read_text(encoding="utf-8", errors="replace")
    marker = None
    for line in reversed(log_text.splitlines()):
        if line.startswith("EXIT_CODE="):
            marker = line.split("=", 1)[1].strip()
            break
    if marker != "0":
        raise RuntimeError(
            f"KFuse did not finish successfully: EXIT_CODE={marker!r}; "
            f"see {args.kfuse_log}"
        )
    metadata = _checkpoint_metadata(Path(args.kfuse_checkpoint), "kfuse")
    report["kfuse_exit_code"] = 0
    report["kfuse_checkpoint"] = metadata
    report["kfuse_wait_seconds"] = time.time() - started
    return metadata


def _launch_jobs(args: argparse.Namespace, report: dict) -> list[dict]:
    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    if output_root.exists() or log_root.exists():
        raise FileExistsError(
            "refusing to reuse smoke output/log root: "
            f"output={output_root}, logs={log_root}"
        )
    output_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    jobs = [
        ("concat", Path(args.concat_config).resolve(),
         Path(args.concat_checkpoint).resolve(), str(args.gpu_concat)),
        ("kfuse", Path(args.kfuse_config).resolve(),
         Path(args.kfuse_checkpoint).resolve(), str(args.gpu_kfuse)),
    ]
    names = [f"glt_ft_auto_{mode}" for mode, *_ in jobs]
    existing = subprocess.run(
        ["tmux", "list-windows", "-t", args.tmux_session,
         "-F", "#W"], capture_output=True, text=True, check=False,
    )
    if existing.returncode != 0:
        raise RuntimeError(f"tmux session is unavailable: {args.tmux_session}")
    occupied = set(existing.stdout.splitlines())
    collision = sorted(set(names) & occupied)
    if collision:
        raise FileExistsError("refusing to reuse tmux window(s): " + ", ".join(collision))

    records = []
    common = dict(
        raw_root=Path(args.raw_root).resolve(),
        cohort_root=Path(args.cohort_root).resolve(),
        cache_root=Path(args.cache_root).resolve(),
        static_root=Path(args.dual_static_root).resolve(),
        split_root=Path(args.split_root).resolve(),
        task=args.task, fold=int(args.fold),
    )
    for mode, config, checkpoint, gpu in jobs:
        output = output_root / mode
        log_path = log_root / f"{mode}.log"
        status_path = output_root / f"{mode}.status.json"
        command = _shell_command(
            config=config, checkpoint=checkpoint, output=output,
            log_path=log_path, status_path=status_path, gpu=gpu, **common,
        )
        subprocess.run(
            ["tmux", "new-window", "-d", "-t", args.tmux_session,
             "-n", f"glt_ft_auto_{mode}", command],
            check=True,
        )
        records.append({
            "mode": mode, "gpu": gpu, "window": f"glt_ft_auto_{mode}",
            "config": str(config), "checkpoint": str(checkpoint),
            "output": str(output), "log": str(log_path),
            "status_file": str(status_path), "command": command,
        })
    report["jobs"] = records
    return records


def _wait_for_jobs(args: argparse.Namespace, report: dict, records: list[dict]) -> None:
    pending = {record["mode"]: record for record in records}
    started = time.time()
    results = {}
    while pending:
        for mode, record in list(pending.items()):
            status_path = Path(record["status_file"])
            if not status_path.is_file():
                continue
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            results[mode] = status
            del pending[mode]
        if pending:
            if args.timeout_seconds and time.time() - started > args.timeout_seconds:
                raise TimeoutError("timed out waiting for fine-tuning smoke jobs")
            time.sleep(args.poll_seconds)
    report["jobs_result"] = results
    report["job_wait_seconds"] = time.time() - started
    failed = {mode: value for mode, value in results.items()
              if int(value.get("exit_code", 1)) != 0}
    if failed:
        raise RuntimeError(f"fine-tuning smoke failed: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kfuse-pid", required=True, type=int,
                        help="PID of the current KFuse torchrun wrapper")
    parser.add_argument("--kfuse-output", default="results/glt_dual_static_pretrain_5k_kfuse")
    parser.add_argument("--kfuse-log", default="logs/dual_static_pretrain_kfuse5k.log")
    parser.add_argument("--kfuse-checkpoint")
    parser.add_argument("--concat-checkpoint",
                        default="results/glt_dual_static_pretrain_5k_concat/deploy_05000.pt")
    parser.add_argument("--concat-config",
                        default="configs/mts/glt_dual_three_task_concat.json")
    parser.add_argument("--kfuse-config",
                        default="configs/mts/glt_dual_three_task_kfuse.json")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--cohort-root", default=str(DEFAULT_COHORT))
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage_downstream")
    parser.add_argument("--dual-static-root",
                        default="data/processed/glt_dual_v2/downstream/dual_static_v1")
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--output-root", default="results/glt_dual_static_finetune_smoke_auto")
    parser.add_argument("--log-root", default="logs/glt_dual_static_finetune_smoke_auto")
    parser.add_argument("--task", default="eat")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--gpu-concat", default="0")
    parser.add_argument("--gpu-kfuse", default="1")
    parser.add_argument("--tmux-session", default="Uni-Poly")
    parser.add_argument("--poll-seconds", default=30, type=int)
    parser.add_argument("--timeout-seconds", default=0, type=int,
                        help="0 means no timeout; running child windows are not killed")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.timeout_seconds < 0:
        raise ValueError("poll/timeout must be positive/non-negative")
    args.kfuse_output = str(Path(args.kfuse_output).resolve())
    args.kfuse_log = str(Path(args.kfuse_log).resolve())
    args.kfuse_checkpoint = str(Path(args.kfuse_checkpoint or
                                      (Path(args.kfuse_output) / "deploy_05000.pt")).resolve())
    args.concat_checkpoint = str(Path(args.concat_checkpoint).resolve())
    args.concat_config = str(Path(args.concat_config).resolve())
    args.kfuse_config = str(Path(args.kfuse_config).resolve())
    args.raw_root = str(Path(args.raw_root).resolve())
    args.cohort_root = str(Path(args.cohort_root).resolve())
    args.cache_root = str(Path(args.cache_root).resolve())
    args.dual_static_root = str(Path(args.dual_static_root).resolve())
    args.split_root = str(Path(args.split_root).resolve())
    report = {
        "status": "WAITING_FOR_KFUSE",
        "protocol": "outer5_inner20_smoke",
        "smoke": True,
        "task": args.task,
        "fold": int(args.fold),
        "outer_test": "NOT_RUN",
        "tmux_session": args.tmux_session,
        "started_at_unix": time.time(),
    }
    report_path = Path(args.output_root).resolve() / "report.json"
    try:
        _validate_paths(args)
        _wait_for_kfuse(args, report)
        report["status"] = "LAUNCHING"
        records = _launch_jobs(args, report)
        _wait_for_jobs(args, report, records)
        report["status"] = "PASS"
        report["finished_at_unix"] = time.time()
        _atomic_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["finished_at_unix"] = time.time()
        # The output root is intentionally created only after KFuse succeeds;
        # for an early failure put the report beside the requested root.
        _atomic_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
