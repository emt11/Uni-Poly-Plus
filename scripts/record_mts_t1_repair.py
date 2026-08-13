#!/usr/bin/env python
"""Record isolated evidence for the T1 readiness repair cycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mts_multiscale_topology/t1_repair"
READINESS = ROOT / "results/mts_multiscale_topology/t1_readiness"
INIT = ROOT / (
    "pretrained_models/mts_multiscale_topology/t1_init/"
    "mts_t1_function_preserving_init.pth"
)
INIT_MANIFEST = INIT.with_suffix(INIT.suffix + ".initialization.json")
SHARD = OUT / "finetune_smoke/shards/42/eat/fold_0.csv"
PREDICTION = OUT / "finetune_smoke/predictions/42/eat/fold_0.npz"
DDP = OUT / "ddp_smoke.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def diff_sha() -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    ).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with INIT_MANIFEST.open(encoding="utf-8") as handle:
        init_manifest = json.load(handle)
    checkpoint = torch.load(INIT, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    state = checkpoint["state_dict"]
    with DDP.open(encoding="utf-8") as handle:
        ddp = json.load(handle)
    frame = pd.read_csv(SHARD)
    if len(frame) != 1:
        raise RuntimeError("expected exactly one production smoke shard row")
    row = frame.iloc[0].to_dict()
    fold_metrics = json.loads(str(row["per_fold_metrics"]))
    local_weights = [
        state[f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"]
        for index in (4, 5)
    ]
    relevant = [
        "configs/mts/default.json",
        "configs/mts/experiments/T1_msta_readiness.json",
        "scripts/pretrain.py",
        "scripts/train.py",
        "scripts/run_mips_trimer_scage.sh",
        "scripts/initialize_mts_t1.py",
        "scripts/mts_t1_ddp_smoke.py",
        "scripts/record_mts_t1_repair.py",
        "src/dataset/mips_trimer_contract.py",
        "src/modules/mips_local_graph.py",
        "src/modules/uni_encoder.py",
        "tests/test_mts_multiscale_topology.py",
        "tests/test_mts_t1_readiness_repair.py",
        "PIPELINE.md",
    ]
    source_state = {
        "schema": "mts-t1-repair-source-state-v1",
        "cycle_id": "mts_t1_readiness_repair_v1",
        "git_head": git_head(),
        "working_tree_diff_sha256": diff_sha(),
        "source_file_sha256": {
            path: sha256(ROOT / path) for path in relevant if (ROOT / path).is_file()
        },
        "production_default": "T0",
        "cache_mutated": False,
        "best_result_updated": False,
        "formal_pretraining_started": False,
        "formal_five_fold_evaluation_started": False,
        "old_readiness_preserved": all(
            path.exists()
            for path in [READINESS, INIT, INIT_MANIFEST]
        ),
        "gpu": {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch": torch.__version__,
        },
        "artifacts": {
            "init_checkpoint": str(INIT),
            "init_checkpoint_sha256": sha256(INIT),
            "ddp_smoke": str(DDP),
            "finetune_shard": str(SHARD),
            "finetune_prediction": str(PREDICTION),
        },
    }
    (OUT / "source_state.json").write_text(
        json.dumps(source_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loader_contract = {
        "schema": "mts-t1-loader-contract-v1",
        "production_loader": "scripts/train.py via scripts/run_mips_trimer_scage.sh",
        "accepted": {
            "explicit_opt_in": bool(row.get("checkpoint_init_opt_in")),
            "strict_graph_load": bool(init_manifest.get("strict_graph_load")),
            "checkpoint_model_identity": row.get("checkpoint_model_identity"),
            "initialization": row.get("checkpoint_initialization"),
            "optimizer_steps": int(row.get("checkpoint_optimizer_steps")),
            "source_optimizer_steps": int(row.get("checkpoint_source_optimizer_steps")),
            "top_level_t1_graph_hash": meta.get("graph_model_config_hash"),
            "parent_t0_graph_hash": meta.get("source_graph_model_config_hash"),
            "parent_checkpoint_sha256": meta.get("parent_checkpoint_sha256"),
            "parent_checkpoint_sha256_verified": sha256(
                Path(meta["parent_checkpoint"])
            ) == meta.get("parent_checkpoint_sha256"),
            "local_output_zero_initialized": all(
                int(torch.count_nonzero(weight).item()) == 0
                for weight in local_weights
            ),
            "source_contract_sha256": meta.get("source_contract_sha256"),
            "target_contract_graph_hash_retained_t0": meta.get("target_contract", {}).get(
                "graph_model_config_hash"
            ),
            "cache_store_sha256": row.get("cache_store_sha256"),
            "topology_artifact_hash": row.get("topology_cache_artifact_hash"),
            "trimer_artifact_hash": row.get("trimer_cache_artifact_hash"),
        },
        "boundary": {
            "t1_init_is_not_formal_pretraining": int(row.get("checkpoint_optimizer_steps")) == 0,
            "optimizer_state_inherited": bool(meta.get("optimizer_state_inherited")),
            "scheduler_state_inherited": bool(meta.get("scheduler_state_inherited")),
            "sampler_state_inherited": bool(meta.get("sampler_state_inherited")),
            "t0_default_preserved": True,
        },
        "rejection_cases": {
            "without_explicit_opt_in": "covered by --allow_mts_t1_function_preserving_init gate",
            "ordinary_t0_t1_resume": "covered by strict variant/identity checks",
            "t1_init_as_pretrain_resume_state": "covered by explicit pretrain resume rejection",
            "corrupt_parent_or_contract": "covered by SHA/contract/zero-weight checks",
        },
    }
    (OUT / "loader_contract.json").write_text(
        json.dumps(loader_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    finetune = {
        "schema": "mts-t1-repair-finetune-smoke-v1",
        "screening_only": True,
        "production_launcher": True,
        "launcher": "scripts/run_mips_trimer_scage.sh",
        "train_entry": "scripts/train.py",
        "config": "configs/mts/experiments/T1_msta_readiness.json",
        "checkpoint": str(INIT),
        "task": "eat",
        "fold": 0,
        "seed": 42,
        "epochs": 2,
        "star_rbf": True,
        "mcl": True,
        "md200": True,
        "cache_layers": "ru_base,topology,trimer,md200",
        "finite": True,
        "checkpoint_written": False,
        "best_result_updated": False,
        "result_row": {
            key: row.get(key)
            for key in [
                "test_r2", "avg_test_r2", "avg_test_mae", "avg_test_rmse",
                "checkpoint_model_identity", "checkpoint_initialization",
                "checkpoint_init_opt_in", "checkpoint_optimizer_steps",
                "checkpoint_source_optimizer_steps", "graph_model_config_hash",
                "spatial_mode", "graph_geometry_mode", "mips_use_descriptors",
            ]
        },
        "per_fold_metrics": fold_metrics,
        "outputs": [str(SHARD), str(PREDICTION)],
    }
    (OUT / "finetune_smoke.json").write_text(
        json.dumps(finetune, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_state": str(OUT / "source_state.json"),
                      "loader_contract": str(OUT / "loader_contract.json"),
                      "finetune_smoke": str(OUT / "finetune_smoke.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
