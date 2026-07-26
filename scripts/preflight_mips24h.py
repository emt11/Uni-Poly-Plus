#!/usr/bin/env python3
"""Measure the current SCAGE cache/model path before a PI1M-50k run."""

import argparse
import json
import math
import random
import subprocess
import sys
import time
import os
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="data/raw/PI1M_v2.csv")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument("--stage2-batch-size", type=int, default=64)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse the current PI1M_preflight cache and its recorded build time.",
    )
    # Retained only so old shell history does not fail at argument parsing.
    parser.add_argument("--partial-cache", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def run(command):
    print("[preflight] running: " + " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    if "--cache_only" in command:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(command, check=True, env=environment)
    return time.monotonic() - started


def write_sample(source_path, output_path, sample_size, seed):
    frame = pd.read_csv(source_path)
    smiles_column = frame.columns[0]
    frame = frame.drop_duplicates(subset=[smiles_column], keep="first")
    if sample_size > len(frame):
        sample_size = len(frame)
    indices = random.Random(seed).sample(range(len(frame)), sample_size)
    sample = frame.iloc[indices].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"[preflight] wrote {len(sample)} unique samples to {output_path}", flush=True)
    return len(sample)


def model_args(dataset, args):
    return [
        "--dataset_name", dataset,
        "--modalities", "graph",
        "--graph_encoder_type", "scage",
        "--graph_input", "star_linking",
        "--geom_input", "polygen_periodic",
        "--fusion_type", "parallel_attention",
        "--pretrain_stage", "scage_m4p",
        "--pretrain_profile", "mips24h",
        "--graph_num_layers", "6",
        "--graph_emb_dim", "512",
        "--scage_num_heads", "8",
        "--scage_ffn_hidden_dim", "2048",
        "--scage_num_kernels", "128",
        "--scage_attention_dropout", "0.1",
        "--scage_distance_mode", "bias",
        "--scage_distance_rbf", "64",
        "--scage_distance_cutoff", "12.0",
        "--no-scage_use_descriptors",
        "--scage_mips_mask_weight", "1.0",
        "--scage_ecfp_weight", "0",
        "--scage_periodic_sp_weight", "0.5",
        "--scage_periodic_geometry_weight", "0.75",
        "--scage_geometry_max_pairs", "32",
        "--scage_periodic_contrast_weight", "0",
        "--dynamic_pretrain_loss",
        "--dynamic_loss_warmup_steps", "200",
        "--graph_mask_ratio", "0.30",
        "--batch_size", str(args.stage1_batch_size),
        "--gradient_accumulation_steps", "4",
        "--loader_workers", "0",
        "--amp_dtype", "bf16",
        "--warmup_ratio", "0.10",
        "--max_smiles_length", "141",
        "--conformer_profile", "quality",
        "--conformer_3d_count", "4",
        "--conformer_keep_count", "4",
        "--feature_cache_item_timeout", "45",
    ]


def main():
    args = parse_args()
    dataset = "PI1M_preflight"
    sample_path = Path("data/raw") / f"{dataset}.csv"
    measured_samples = write_sample(
        Path(args.source_csv), sample_path, args.sample_size, args.seed
    )
    cache_checkpoint = Path("/tmp/scage_cache_preflight.pth")
    cache_matches = sorted(Path("data/processed/scage").glob(
        "feature_cache_PI1M_preflight_*quality_hardtimeout45_fp-ecfp_tok141.pt"
    ))
    if args.reuse_cache:
        if not cache_matches:
            raise FileNotFoundError("No compatible PI1M_preflight cache to reuse")
        cache_payload = torch.load(cache_matches[-1], map_location="cpu", weights_only=False)
        cache_seconds = float(cache_payload.get("meta", {}).get("build_elapsed_seconds", 0.0))
        if cache_seconds <= 0:
            raise ValueError("Reused cache does not contain build_elapsed_seconds")
        print(
            f"[preflight] reusing {cache_matches[-1]} (build={cache_seconds:.2f}s)",
            flush=True,
        )
    else:
        cache_seconds = run([
            sys.executable, "scripts/pretrain.py", *model_args(dataset, args),
            "--cache_only",
            "--feature_cache_workers", str(args.cache_workers),
            "--feature_cache_partial_every", "50",
            "--rebuild_feature_cache",
        ])
        cache_matches = sorted(Path("data/processed/scage").glob(
            "feature_cache_PI1M_preflight_*quality_hardtimeout45_fp-ecfp_tok141.pt"
        ))
        if not cache_matches:
            raise FileNotFoundError("Preflight cache build completed but no cache was found")
        cache_payload = torch.load(
            cache_matches[-1], map_location="cpu", weights_only=False
        )

    features = list(cache_payload["features"].values())

    def distribution(values):
        if not values:
            return {"count": 0}
        tensor = torch.tensor(values, dtype=torch.float)
        return {
            "count": int(tensor.numel()),
            "min": float(tensor.min()),
            "p50": float(torch.quantile(tensor, 0.50)),
            "p90": float(torch.quantile(tensor, 0.90)),
            "p99": float(torch.quantile(tensor, 0.99)),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
        }

    def histogram(name):
        values = [int(getattr(item, name, -1)) for item in features]
        return {
            str(value): values.count(value) for value in sorted(set(values))
        }

    cache_health = {
        "repeat_factor": histogram("mips_repeat_factor"),
        "model_cell_ru": histogram("model_cell_ru"),
        "atom_count": distribution([int(item.x.size(0)) for item in features]),
        "lga_edge_count": distribution([
            int(item.lga_edge_index.size(1)) for item in features
        ]),
        "periodic_geometry_valid": sum(
            bool(getattr(item, "periodic_geometry_valid", False))
            for item in features
        ),
        "graph_unavailable": sum(
            not bool(getattr(item, "graph_available", False))
            for item in features
        ),
        "alias_failure": sum(
            bool(getattr(item, "mips_condition_valid", False))
            and not bool(getattr(item, "mips_alias_free", False))
            for item in features
        ),
        "sample_seconds": distribution([
            float(getattr(item, "feature_compute_seconds", 0.0))
            for item in features
            if float(getattr(item, "feature_compute_seconds", 0.0)) > 0
        ]),
    }

    train_checkpoint = Path("/tmp/scage_train_preflight.pth")
    measured_microbatches = 100
    train_seconds = run([
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node", str(args.nproc),
        "scripts/pretrain.py", *model_args(dataset, args),
        "--epochs", str(measured_microbatches),
        "--max_steps", str(measured_microbatches),
        "--save_path", str(train_checkpoint),
    ])

    cache_forecast = cache_seconds * float(args.target_size) / max(measured_samples, 1)
    stage1_batches = math.ceil(
        args.target_size / (args.nproc * args.stage1_batch_size)
    ) * args.stage1_epochs
    stage2_batches = math.ceil(
        args.target_size / (args.nproc * args.stage2_batch_size)
    ) * args.stage2_epochs
    seconds_per_microbatch = train_seconds / measured_microbatches
    # Stage 2 additionally executes SMILES/FP encoders and two fusion views.
    stage2_multiplier = 2.0
    forecast_seconds = (
        cache_forecast
        + stage1_batches * seconds_per_microbatch
        + stage2_batches * seconds_per_microbatch * stage2_multiplier
    )
    report = {
        "schema": "scage-mips-pbc-pyg-v1",
        "measured_samples": measured_samples,
        "target_samples": args.target_size,
        "cache_sample_seconds": cache_seconds,
        "cache_target_forecast_seconds": cache_forecast,
        "stage1_100_distributed_microbatches_seconds": train_seconds,
        "seconds_per_distributed_microbatch": seconds_per_microbatch,
        "stage1_forecast_distributed_microbatches": stage1_batches,
        "stage1_batch_size_per_rank": args.stage1_batch_size,
        "stage2_forecast_distributed_microbatches": stage2_batches,
        "stage2_batch_size_per_rank": args.stage2_batch_size,
        "stage2_multiplier": stage2_multiplier,
        "forecast_hours": forecast_seconds / 3600.0,
        "limit_hours": args.max_hours,
        "cache_health": cache_health,
        "pass": forecast_seconds <= args.max_hours * 3600.0,
    }
    output = Path("data/processed/scage/mips24h_preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("SCAGE preflight forecast exceeds the requested time budget")


if __name__ == "__main__":
    main()
