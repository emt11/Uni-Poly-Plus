#!/usr/bin/env python3
"""Stage-controlled MTS-GLT-v3 pretrain and two-mode finetune pipeline.

The two deployment variants intentionally share one matched joint-pretraining
trajectory.  That trajectory emits a Galformer (O8+MD200) bundle and a
MIPS-concat (O8+GLT+concat+MD200) bundle at 5k/10k/20k.  Fine-tuning is then
run independently for the two bundles.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/mts_glt_v3_galformer_20k"
LOG_ROOT = ROOT / "logs/mts_glt_v3_galformer_20k/formal"
PI1M_LINE = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_image_v1/PI1M_v2"
PI1M_MD = ROOT / "data/processed/mips_trimer_scage/md200_pi1m_v1/PI1M_v2"
DOWNSTREAM_LINE = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_image_v1/downstream_union"
PRETRAIN_ROOT = RESULT_ROOT / "pretrain"


def run_logged(name, command, *, env=None):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    print("RUN", " ".join(str(value) for value in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(value) for value in command], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit(f"{name} failed with exit code {code}; log={log_path}")


def complete_sidecar(path):
    return (path / ".done").is_file() and (path / "metadata.json").is_file()


def build_inputs():
    jobs = (
        ("line_pi1m", "line-image", "PI1M_v2", PI1M_LINE),
        ("md200_pi1m", "md200", "PI1M_v2", PI1M_MD),
        ("line_downstream", "line-image", "smi_all", DOWNSTREAM_LINE),
    )
    for name, mode, dataset, output in jobs:
        if complete_sidecar(output):
            print(f"REUSE complete sidecar {output}", flush=True)
            continue
        if output.exists():
            raise SystemExit(f"incomplete/conflicting sidecar directory: {output}")
        run_logged(name, [
            sys.executable, "scripts/build_mts_glt_v3_sidecars.py", mode,
            "--dataset", dataset, "--output", output,
        ])


def pretrain(gpu_ids):
    for required in (PI1M_LINE, PI1M_MD):
        if not complete_sidecar(required):
            raise SystemExit(f"pretraining input is incomplete: {required}")
    if PRETRAIN_ROOT.exists():
        raise SystemExit(f"pretraining output already exists: {PRETRAIN_ROOT}")
    selected = gpu_ids[:3]
    if len(selected) != 3:
        raise SystemExit("pretraining requires three GPU ids")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    run_logged("pretrain_20k", [
        "torchrun", "--standalone", "--nproc_per_node=3", "scripts/pretrain.py",
        "--experiment_config", "configs/mts/glt_v3_galformer_20k.json",
    ], env=env)


def finetune(mode, gpu_ids):
    checkpoint = PRETRAIN_ROOT / f"mts_glt_v3_{mode}_20k.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"missing 20k {mode} deploy bundle: {checkpoint}")
    if mode == "mips_concat" and not complete_sidecar(DOWNSTREAM_LINE):
        raise SystemExit(f"missing downstream image sidecar: {DOWNSTREAM_LINE}")
    run_logged(f"finetune_{mode}", [
        sys.executable, "scripts/run_mts_glt_v3_finetune.py",
        "--checkpoint", checkpoint, "--mode", mode,
        "--run-name", "formal_20k", "--gpu-ids", ",".join(gpu_ids),
    ])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=(
            "build-inputs", "pretrain", "finetune-galformer",
            "finetune-mips", "all",
        ),
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if args.stage in {"build-inputs", "all"}:
        build_inputs()
    if args.stage in {"pretrain", "all"}:
        pretrain(gpu_ids)
    if args.stage in {"finetune-galformer", "all"}:
        finetune("galformer", gpu_ids)
    if args.stage in {"finetune-mips", "all"}:
        finetune("mips_concat", gpu_ids)


if __name__ == "__main__":
    main()
