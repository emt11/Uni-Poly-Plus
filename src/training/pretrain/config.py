"""Configuration for the single retained MTS-GLT-v2 pretraining route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset.mips_trimer_contract import (
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
)


BASELINE_SCHEMA = "mts-glt-v2"
V3_SCHEMA = "mts-glt-v3-galformer-20k"
SUPPORTED_MODALITIES = ("graph",)


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"unsupported modality: {value}; baseline is graph-only"
        )
    return value


_GLT_REQUIRED = {
    "schema", "experiment_id", "dataset_name", "cache_root",
    "line_sidecar_root", "line_label_counts", "result_root", "output_path",
    "output_kind", "masked_atom", "masked_line", "infonce",
    "use_star_rbf", "use_mcl", "use_md200", "coordinate_denoising",
    "topology_attention_variant", "glt_layers", "glt_attention_variant",
    "atom_mask_ratio", "line_mask_ratio", "infonce_temperature",
    "atom_loss_weight", "line_loss_weight", "infonce_loss_weight",
    "projection_dim", "batch_size", "loader_workers", "prefetch_factor",
    "gradient_accumulation_steps", "global_batch_size", "max_optimizer_steps",
    "stop_after_steps", "checkpoint_interval_steps", "probe_steps",
    "amp_dtype", "seed", "lr", "warmup_steps", "end_lr", "weight_decay",
    "cache_layers",
}
_GLT_ALLOWED = _GLT_REQUIRED | {"source_config", "project_root"}

_V3_REQUIRED = {
    "schema", "experiment_id", "dataset_name", "cache_root",
    "line_sidecar_root", "md200_sidecar_root", "result_root", "output_path",
    "atom_mask_ratio", "line_mask_ratio", "infonce_temperature",
    "batch_size", "loader_workers", "prefetch_factor", "global_batch_size",
    "max_optimizer_steps", "stop_after_steps", "probe_steps", "amp_dtype",
    "seed", "lr", "warmup_steps", "end_lr", "weight_decay", "cache_layers",
    "glt_readout_mode",
}


def _apply_glt_v2_config(args, config_path):
    path = Path(config_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MTS-GLT-v2 config must be a JSON object")
    unknown = sorted(set(payload) - _GLT_ALLOWED)
    missing = sorted(_GLT_REQUIRED - set(payload))
    if unknown or missing:
        raise ValueError(
            "MTS-GLT-v2 config fields are invalid; "
            f"unknown={unknown} missing={missing}"
        )
    if payload["schema"] != BASELINE_SCHEMA:
        raise ValueError(f"unsupported pretraining schema: {payload['schema']!r}")
    for name in ("masked_atom", "masked_line", "infonce"):
        if payload[name] is not True:
            raise ValueError(f"MTS-GLT-v2 requires {name}=true")
    for name in ("use_star_rbf", "use_mcl", "use_md200", "coordinate_denoising"):
        if payload[name] is not False:
            raise ValueError(f"MTS-GLT-v2 requires {name}=false")
    if payload["topology_attention_variant"] != "o8":
        raise ValueError("MTS-GLT-v2 requires topology_attention_variant=o8")
    if int(payload["glt_layers"]) != 6:
        raise ValueError("MTS-GLT-v2 baseline fixes glt_layers=6")
    if str(payload["glt_attention_variant"]) != "mips":
        raise ValueError("MTS-GLT-v2 baseline fixes glt_attention_variant=mips")
    if abs(float(payload["atom_mask_ratio"]) - 0.30) > 1e-12:
        raise ValueError("MTS-GLT-v2 baseline fixes atom_mask_ratio=0.30")
    if abs(float(payload["line_mask_ratio"]) - 0.40) > 1e-12:
        raise ValueError("MTS-GLT-v2 baseline fixes line_mask_ratio=0.40")
    if abs(float(payload["infonce_temperature"]) - 0.10) > 1e-12:
        raise ValueError("MTS-GLT-v2 baseline fixes infonce_temperature=0.10")
    if any(
        abs(float(payload[name]) - 1.0) > 1e-12
        for name in ("atom_loss_weight", "line_loss_weight", "infonce_loss_weight")
    ):
        raise ValueError("MTS-GLT-v2 baseline fixes all objective weights at 1.0")
    if int(payload["projection_dim"]) != 256:
        raise ValueError("MTS-GLT-v2 baseline fixes projection_dim=256")
    if int(payload["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(payload["global_batch_size"]) < 1:
        raise ValueError("global_batch_size must be positive")
    if int(payload["checkpoint_interval_steps"]) < 1:
        raise ValueError("checkpoint_interval_steps must be positive")
    if str(payload["output_kind"]) != "trajectory":
        raise ValueError("MTS-GLT-v2 output_kind must be trajectory")
    stop_after = int(payload["stop_after_steps"])
    if stop_after < 1 or stop_after > int(payload["max_optimizer_steps"]):
        raise ValueError("stop_after_steps must be within max_optimizer_steps")
    probes = tuple(int(value) for value in payload["probe_steps"])
    if not probes or any(value < 1 or value > stop_after for value in probes):
        raise ValueError("probe_steps must fall within stop_after_steps")
    if str(payload["amp_dtype"]) not in {"fp32", "bf16"}:
        raise ValueError("amp_dtype must be fp32 or bf16")

    args.config_schema = BASELINE_SCHEMA
    args.config_source_schema = BASELINE_SCHEMA
    args.experiment_id = str(payload["experiment_id"])
    args.dataset_name = str(payload["dataset_name"])
    args.root = str(payload["cache_root"])
    args.periodic_line_glt_sidecar = str(payload["line_sidecar_root"])
    args.glt_line_label_counts = str(payload["line_label_counts"])
    args.save_path = str(payload["output_path"])
    args.glt_result_root = str(payload["result_root"])
    args.glt_output_kind = str(payload["output_kind"])
    args.graph_mask_ratio = float(payload["atom_mask_ratio"])
    args.glt_line_mask_ratio = float(payload["line_mask_ratio"])
    args.glt_infonce_temperature = float(payload["infonce_temperature"])
    args.glt_atom_loss_weight = float(payload["atom_loss_weight"])
    args.glt_line_loss_weight = float(payload["line_loss_weight"])
    args.glt_infonce_loss_weight = float(payload["infonce_loss_weight"])
    args.glt_projection_dim = int(payload["projection_dim"])
    args.glt_layers = int(payload["glt_layers"])
    args.glt_attention_variant = str(payload["glt_attention_variant"])
    args.batch_size = int(payload["batch_size"])
    args.loader_workers = int(payload["loader_workers"])
    args.loader_prefetch_factor = int(payload["prefetch_factor"])
    args.gradient_accumulation_steps = int(payload["gradient_accumulation_steps"])
    args.global_batch_size = int(payload["global_batch_size"])
    args.max_optimizer_steps = int(payload["max_optimizer_steps"])
    args.checkpoint_interval_steps = int(payload["checkpoint_interval_steps"])
    args.glt_probe_steps = probes
    args.glt_stop_after_steps = stop_after
    args.amp_dtype = str(payload["amp_dtype"])
    args.seed = int(payload["seed"])
    args.lr = float(payload["lr"])
    args.warmup_steps = int(payload["warmup_steps"])
    args.end_lr = float(payload["end_lr"])
    args.weight_decay = float(payload["weight_decay"])
    args.cache_layers = str(payload["cache_layers"])

    # Fixed dataset/model contract consumed by UniDataset and the baseline model.
    args.graph_encoder_type = MTS_ROUTE_INTERNAL
    args.graph_input = "star_linking"
    args.geom_input = "repeat_unit"
    args.smiles_model_name = ""
    args.feature_source_dataset = args.dataset_name
    args.disable_feature_cache = False
    args.rebuild_feature_cache = False
    args.max_smiles_length = None
    args.max_smiles_length_cap = 256
    args.fp_mode = "disabled"
    args.feature_cache_workers = 0
    args.feature_cache_chunksize = 4
    args.feature_cache_partial_every = 200
    args.feature_cache_item_timeout = 45
    args.cache_validate = "sample"
    args.cache_commit_size = 128
    args.embed_tries_multiplier = 8
    args.conformer_3d_count = 8
    args.conformer_keep_count = 4
    args.conformer_profile = "full"
    args.scage_distance_mode = "bias"
    args.scage_distance_rbf = 32
    args.scage_distance_cutoff = 12.0
    args.mips_core = "paper_corrected"
    args.mips_max_hops = 2
    args.mips_use_descriptors = True
    args.mips_descriptor_protocol = "source_star_sub"
    args.spatial_mode = "trimer_scage"
    args.graph_geometry_mode = "trimer_scage_mcl"
    args.topology_representation = "canonical_lifted"
    args.trimer_num_candidates = 4
    args.trimer_max_heavy_atoms = 384
    args.mips_variant = "O8"
    args.finite_variant = "none"
    args.conformer_mode = "none"
    args.field_layout = "none"
    args.field_channels = "none"
    args.modalities = ["graph"]
    args.max_grad_norm = 1.0
    args.resume_state = getattr(args, "resume_state", None)
    args.cache_only = False
    return args


def _apply_glt_v3_config(args, path, payload):
    unknown = sorted(set(payload) - _V3_REQUIRED)
    missing = sorted(_V3_REQUIRED - set(payload))
    if unknown or missing:
        raise ValueError(f"MTS-GLT-v3 config fields invalid; unknown={unknown} missing={missing}")
    fixed = {
        "atom_mask_ratio": 0.30, "line_mask_ratio": 0.40,
        "infonce_temperature": 0.10, "weight_decay": 0.0,
    }
    for name, expected in fixed.items():
        if abs(float(payload[name]) - expected) > 1e-12:
            raise ValueError(f"MTS-GLT-v3 fixes {name}={expected}")
    if payload["glt_readout_mode"] != "galformer":
        raise ValueError("the formal MTS-GLT-v3 pretraining config defaults to galformer")
    if int(payload["global_batch_size"]) != 3 * int(payload["batch_size"]):
        raise ValueError("MTS-GLT-v3 global batch must equal three local batches")
    if int(payload["max_optimizer_steps"]) != 20000:
        raise ValueError("MTS-GLT-v3 formal trajectory fixes 20,000 updates")
    if int(payload["warmup_steps"]) != 2000:
        raise ValueError("MTS-GLT-v3 fixes 2,000 warmup updates")
    probes = tuple(int(value) for value in payload["probe_steps"])
    if probes != (5000, 10000, 20000):
        raise ValueError("MTS-GLT-v3 probes must be exactly 5k/10k/20k")
    stop_after = int(payload["stop_after_steps"])
    if not 1 <= stop_after <= 20000:
        raise ValueError("MTS-GLT-v3 stop_after_steps out of range")
    args.config_schema = V3_SCHEMA
    args.config_source_schema = V3_SCHEMA
    args.experiment_id = str(payload["experiment_id"])
    args.dataset_name = str(payload["dataset_name"])
    args.root = str(payload["cache_root"])
    args.periodic_line_glt_sidecar = str(payload["line_sidecar_root"])
    args.md200_sidecar_root = str(payload["md200_sidecar_root"])
    args.save_path = str(payload["output_path"])
    args.glt_result_root = str(payload["result_root"])
    args.graph_mask_ratio = float(payload["atom_mask_ratio"])
    args.glt_line_mask_ratio = float(payload["line_mask_ratio"])
    args.glt_infonce_temperature = float(payload["infonce_temperature"])
    args.glt_atom_loss_weight = args.glt_line_loss_weight = args.glt_infonce_loss_weight = 1.0
    args.glt_projection_dim = 256
    args.glt_layers, args.glt_attention_variant = 6, "mips"
    args.batch_size = int(payload["batch_size"])
    args.loader_workers = int(payload["loader_workers"])
    args.loader_prefetch_factor = int(payload["prefetch_factor"])
    args.gradient_accumulation_steps = 1
    args.global_batch_size = int(payload["global_batch_size"])
    args.max_optimizer_steps = 20000
    args.glt_stop_after_steps = stop_after
    args.glt_probe_steps = probes
    args.amp_dtype = str(payload["amp_dtype"])
    args.seed = int(payload["seed"])
    args.lr = float(payload["lr"])
    args.warmup_steps = 2000
    args.end_lr = float(payload["end_lr"])
    args.weight_decay = 0.0
    args.cache_layers = str(payload["cache_layers"])
    # Shared immutable Dataset contract.
    args.graph_encoder_type = MTS_ROUTE_INTERNAL
    args.graph_input, args.geom_input = "star_linking", "repeat_unit"
    args.smiles_model_name, args.feature_source_dataset = "", args.dataset_name
    args.max_smiles_length, args.max_smiles_length_cap = None, 256
    args.fp_mode = "disabled"
    args.feature_cache_workers, args.feature_cache_chunksize = 0, 4
    args.feature_cache_partial_every, args.feature_cache_item_timeout = 200, 45
    args.cache_validate, args.cache_commit_size, args.embed_tries_multiplier = "sample", 128, 8
    args.conformer_3d_count, args.conformer_keep_count, args.conformer_profile = 8, 4, "full"
    args.scage_distance_mode, args.scage_distance_rbf, args.scage_distance_cutoff = "bias", 32, 12.0
    args.mips_core, args.mips_max_hops = "paper_corrected", 2
    args.mips_use_descriptors, args.mips_descriptor_protocol = True, "source_star_sub"
    args.spatial_mode, args.graph_geometry_mode = "trimer_scage", "trimer_scage_mcl"
    args.topology_representation = "canonical_lifted"
    args.trimer_num_candidates, args.trimer_max_heavy_atoms = 4, 384
    args.mips_variant, args.finite_variant = "O8", "none"
    args.conformer_mode, args.field_layout, args.field_channels = "none", "none", "none"
    args.modalities, args.max_grad_norm = ["graph"], 1.0
    args.cache_only = False
    return args


def dataset_kwargs_from_args(args):
    """Translate the fixed baseline config into the Dataset constructor contract."""
    return {
        "root": args.root,
        "dataset": args.dataset_name,
        "smiles_model_name": args.smiles_model_name,
        "graph_encoder_type": args.graph_encoder_type,
        "graph_input": args.graph_input,
        "geom_input": args.geom_input,
        "use_feature_cache": True,
        "feature_source_dataset": args.feature_source_dataset,
        "rebuild_feature_cache": False,
        "max_smiles_length": args.max_smiles_length,
        "max_smiles_length_cap": args.max_smiles_length_cap,
        "fp_mode": args.fp_mode,
        "feature_cache_workers": args.feature_cache_workers,
        "feature_cache_chunksize": args.feature_cache_chunksize,
        "feature_cache_partial_every": args.feature_cache_partial_every,
        "feature_cache_item_timeout": args.feature_cache_item_timeout,
        "cache_layers": str(args.cache_layers),
        "cache_validate": args.cache_validate,
        "cache_commit_size": args.cache_commit_size,
        "embed_tries_multiplier": args.embed_tries_multiplier,
        "conformer_3d_count": args.conformer_3d_count,
        "conformer_keep_count": args.conformer_keep_count,
        "conformer_profile": args.conformer_profile,
        "scage_distance_mode": args.scage_distance_mode,
        "scage_distance_rbf": args.scage_distance_rbf,
        "scage_distance_cutoff": args.scage_distance_cutoff,
        "mips_core": args.mips_core,
        "mips_max_hops": args.mips_max_hops,
        "mips_use_descriptors": args.mips_use_descriptors,
        "mips_descriptor_protocol": args.mips_descriptor_protocol,
        "spatial_mode": args.spatial_mode,
        "graph_geometry_mode": args.graph_geometry_mode,
        "topology_representation": args.topology_representation,
        "trimer_num_candidates": args.trimer_num_candidates,
        "trimer_max_heavy_atoms": args.trimer_max_heavy_atoms,
        "mips_variant": args.mips_variant,
        "finite_variant": args.finite_variant,
        "conformer_mode": args.conformer_mode,
        "field_layout": args.field_layout,
        "field_channels": args.field_channels,
        "modalities": ("graph",),
        "experiment_id": args.experiment_id,
        "feature_config_hash": "manual",
        "periodic_line_glt_sidecar": args.periodic_line_glt_sidecar,
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretrain MTS-GLT-v2 or MTS-GLT-v3-Galformer"
    )
    parser.add_argument(
        "--experiment_config",
        required=True,
        help="Path to an explicit MTS-GLT-v2 or MTS-GLT-v3 JSON config",
    )
    parser.add_argument("--resume_state", default=None)
    args = parser.parse_args(argv)
    path = Path(args.experiment_config).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == V3_SCHEMA:
        if args.resume_state:
            raise ValueError("MTS-GLT-v3 deliberately has no resume path")
        return _apply_glt_v3_config(args, path, payload)
    return _apply_glt_v2_config(args, path)


__all__ = [
    "BASELINE_SCHEMA",
    "V3_SCHEMA",
    "dataset_kwargs_from_args",
    "parse_arguments",
]
