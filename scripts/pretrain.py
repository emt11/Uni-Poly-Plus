import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import json
import hashlib
import math
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    CHECKPOINT_SCHEMA as MIPS_TRIMER_CHECKPOINT_SCHEMA,
    CACHE_LAYOUT_SCHEMA as MIPS_TRIMER_CACHE_LAYOUT_SCHEMA,
    CACHE_BUNDLE_SCHEMA as MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as MIPS_TRIMER_TOPOLOGY_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION as MIPS_CANONICAL_LGA_SCHEMA_VERSION,
    CONFIG_SCHEMA as MIPS_TRIMER_CONFIG_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA as MIPS_EXPERIMENT_CONFIG_SCHEMA,
    FEATURE_SCHEMA as MIPS_TRIMER_FEATURE_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA as MIPS_EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_TOPOLOGY_LMDB_SCHEMA as MIPS_EXPLICIT_TOPOLOGY_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION as MIPS_EXPLICIT_LGA_SCHEMA_VERSION,
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
    TRIMER_BUILDER_VERSION as MIPS_TRIMER_BUILDER_VERSION,
    TRIMER_CONTENT_SCHEMA as MIPS_TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA as MIPS_TRIMER_LMDB_SCHEMA,
    TRIMER_ACCEPTANCE as MIPS_TRIMER_ACCEPTANCE,
    TRIMER_PROTOCOL as MIPS_TRIMER_PROTOCOL,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS as MIPS_TRIMER_MMFF_RELAX_STEPS,
    TRIMER_REQUIRE_MMFF_CONVERGENCE as MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE,
    TRIMER_SELECTION as MIPS_TRIMER_SELECTION,
    ROUTE_NAME as MTS_ROUTE_NAME,
    ROUTE_SHORT_NAME as MTS_ROUTE_SHORT_NAME,
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
    STAGE1_ID as MTS_STAGE1_ID,
    STAGE2_ID as MTS_STAGE2_ID,
    BUILDER_VERSION as MIPS_BUILDER_VERSION,
    CACHE_TOPOLOGY_COST_SCHEMA as MIPS_CACHE_TOPOLOGY_COST_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA,
    PRETRAIN_TRAIN_STATE_SCHEMA,
    PRETRAIN_TARGET_CONTRACT_SCHEMA,
    PRETRAIN_PROFILE_SCHEMA,
    PRETRAIN_PROFILE_ID,
    CACHE_BOND_ANGLE_SCHEMA,
    build_pretrain_target_contract,
    _canonical_json_hash,
    stage_display_name,
    normalize_stage,
    cache_bundle_binding_hash,
    validate_runtime_args as validate_mips_trimer_runtime,
)
from src.dataset.mips_cache_validation import trimer_can_enter_mcl

SUPPORTED_MODALITIES = ('graph', 'smiles', 'fp')

def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_t1_function_preserving_init_payload(payload) -> bool:
    """Identify an architecture-init artifact, never a resumable train state."""

    meta = payload.get("meta") if isinstance(payload, dict) else None
    return bool(
        isinstance(meta, dict)
        and meta.get("init_artifact") is True
        and meta.get("initialization") == "function_preserving"
    )


def _path_tree_sha256(path):
    """Hash model/tokenizer bytes and relative filenames deterministically."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return _file_sha256(path)
    digest = hashlib.sha256()
    for root, _, files in os.walk(path):
        for name in sorted(files):
            filename = os.path.join(root, name)
            relative = os.path.relpath(filename, path)
            digest.update(relative.encode("utf-8"))
            digest.update(_file_sha256(filename).encode("ascii"))
    return digest.hexdigest()


_PRETRAIN_CODE_FILES = (
    "configs/mts/default.json",
    "configs/mts/pretraining/canonical_ru_angle20_v1.json",
    "configs/mts/experiments/T0_o8_pretrain20k_matched_v1.json",
    "configs/mts/experiments/T1_msta_pretrain20k_matched_v1.json",
    "configs/mts/experiments/T1_msta_readiness.json",
    "scripts/initialize_mts_t_pretrain0.py",
    "scripts/audit_mts_t_pretrain0.py",
    "scripts/initialize_mts_t1.py",
    "scripts/pretrain.py",
    "scripts/run_mips_trimer_scage.sh",
    "src/dataset/dataloader.py",
    "src/dataset/dataset.py",
    "src/dataset/mips_trimer_contract.py",
    "src/dataset/trimer_mcl.py",
    "src/modules/mips_local_graph.py",
    "src/modules/uni_encoder.py",
)


def _pretrain_code_identity(root=PROJECT_ROOT):
    """Return the immutable source identity used by resume and checkpoints."""
    root = Path(root)
    files = {}
    digest = hashlib.sha256()
    for relative in _PRETRAIN_CODE_FILES:
        path = root / relative
        if not path.is_file():
            continue
        value = _file_sha256(path)
        files[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {"files": files, "sha256": digest.hexdigest()}


def _load_pretrain_profile(profile_id_or_path):
    """Load and validate the one active formal pretraining profile."""
    value = str(profile_id_or_path or "").strip()
    if not value or value == "mips24h":
        return None
    path = Path(value)
    if not path.is_file():
        path = Path(PROJECT_ROOT) / "configs/mts/pretraining" / f"{value}.json"
    if not path.is_file():
        raise RuntimeError(f"pretraining profile is missing: {path}")
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "profile_id", "representation", "dataset", "cohort_hash",
        "objective", "angle_objective", "angle_cache_schema",
        "angle_cache_artifact", "masked_atom_ratio", "masked_atom_weight",
        "angle_weight", "angle_bins", "focal_gamma", "optimizer_steps",
        "seed", "world_size", "global_batch", "optimizer", "betas", "eps",
        "weight_decay", "peak_lr", "warmup_steps", "scheduler",
        "scheduler_power", "end_lr", "amp", "gradient_clipping",
    }
    missing = sorted(required - set(profile))
    if missing or profile.get("schema") != PRETRAIN_PROFILE_SCHEMA:
        raise RuntimeError(
            f"invalid pretraining profile {path}: schema={profile.get('schema')!r}, "
            f"missing={missing}"
        )
    if profile.get("profile_id") != PRETRAIN_PROFILE_ID:
        raise RuntimeError("only canonical_ru_angle20_v1 is active in this cycle")
    checks = {
        "dataset": "PI1M_v2", "angle_objective": "categorical",
        "angle_cache_schema": CACHE_BOND_ANGLE_SCHEMA,
        "angle_cache_artifact": "6fa4c12268378566099afda1e56987da03fb65efc8b3a21c93a9b28a534b0e7c",
        "representation": "canonical_lifted",
        "objective": "masked_atom_plus_trimer_angle20_focal",
        "angle_bins": 20, "optimizer_steps": 20000, "seed": 42,
        "world_size": 3, "global_batch": 1008, "optimizer": "Adam",
        "weight_decay": 0.0, "peak_lr": 2e-4, "warmup_steps": 2000,
        "scheduler": "polynomial", "scheduler_power": 1,
        "end_lr": 1e-9, "amp": "bf16", "gradient_clipping": "disabled",
    }
    for key, expected in checks.items():
        if profile.get(key) != expected:
            raise RuntimeError(
                f"pretraining profile {key} mismatch: expected {expected!r}, "
                f"got {profile.get(key)!r}"
            )
    if list(profile.get("betas", [])) != [0.9, 0.98] or float(profile.get("eps")) != 1e-8:
        raise RuntimeError("pretraining profile Adam contract mismatch")
    if (
        float(profile.get("masked_atom_ratio")) != 0.30
        or float(profile.get("masked_atom_weight")) != 1.0
        or float(profile.get("angle_weight")) != 0.25
        or float(profile.get("focal_gamma")) != 2.0
    ):
        raise RuntimeError("pretraining profile objective weights mismatch")
    return profile


def _build_final_cache_binding(profile=None):
    """Read-only snapshot of the frozen cache and derived sidecars."""
    from scripts.audit_mips_trimer_cache import _specs

    specs = _specs(Path(PROJECT_ROOT))
    topo_root = Path(specs["topology"]["root"])
    trimer_root = Path(specs["trimer"]["root"])
    store_path = topo_root.parents[1] / "validation" / "store.json"
    exact_path = store_path.parent / "exact_union_validation.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    binding = {
        "schema": "mts-final-cache-binding-v1",
        "store_schema": str(store.get("schema", "")),
        "store_sha256": _file_sha256(store_path),
        "store_transaction_id": store.get("transaction_id"),
        "exact_union_sha256": _file_sha256(exact_path),
        "exact_union_validator_version": str(
            store.get("full_validation", {}).get("validator_version", "")
        ),
        "done_artifact_id": {},
        "done_file_sha256": {},
        "frozen_file_sha256": {},
        "layers": {},
        "angle_sidecars": [],
        "mcl_sidecars": [],
    }
    for layer in ("topology", "trimer"):
        root = Path(specs[layer]["root"])
        done_id = (root / ".done").read_text(encoding="utf-8").strip()
        done_sha = _file_sha256(root / ".done")
        frozen_sha = _file_sha256(root / ".frozen")
        binding["done_artifact_id"][layer] = done_id
        binding["done_file_sha256"][layer] = done_sha
        binding["frozen_file_sha256"][layer] = frozen_sha
        binding["layers"][layer] = {
            "root": str(root),
            "done_artifact_id": done_id,
            "done_file_sha256": done_sha,
            "frozen_file_sha256": frozen_sha,
        }
    cohorts = {
        "PI1M_v2": "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049",
        "downstream_union": "ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2",
    }
    for cohort_name, cohort_hash in cohorts.items():
        angle_root = trimer_root / "derived" / "bond_angle" / cohort_hash
        metadata = json.loads((angle_root / "metadata.json").read_text(encoding="utf-8"))
        angle_item = {
            "cohort": cohort_name,
            "cohort_hash": cohort_hash,
            "schema": metadata.get("schema"),
            "artifact_hash": (angle_root / ".done").read_text(encoding="utf-8").strip(),
            "metadata_sha256": _file_sha256(angle_root / "metadata.json"),
            "done_file_sha256": _file_sha256(angle_root / ".done"),
            "frozen_file_sha256": _file_sha256(angle_root / ".frozen"),
            "trimer_artifact_hash": binding["done_artifact_id"]["trimer"],
            "trimer_done_file_sha256": binding["done_file_sha256"]["trimer"],
        }
        if profile is not None and cohort_name == "PI1M_v2":
            if (
                angle_item["schema"] != profile["angle_cache_schema"]
                or angle_item["artifact_hash"] != profile["angle_cache_artifact"]
            ):
                raise RuntimeError("frozen categorical Angle-20 sidecar binding is stale")
        binding["angle_sidecars"].append(angle_item)
        mcl_root = (
            Path(PROJECT_ROOT) / "data/processed/mips_trimer_scage/cohorts"
            / cohort_name / cohort_hash
        )
        mcl_meta_path = mcl_root / "mcl_thresholds_metadata.json"
        mcl_values = mcl_root / "mcl_thresholds.npy"
        if mcl_meta_path.is_file() and mcl_values.is_file():
            mcl_metadata = json.loads(mcl_meta_path.read_text(encoding="utf-8"))
            binding["mcl_sidecars"].append({
                "cohort": cohort_name,
                "cohort_hash": cohort_hash,
                "schema": mcl_metadata.get("schema"),
                "values_sha256": _file_sha256(mcl_values),
                "metadata_sha256": _file_sha256(mcl_meta_path),
                "trimer_artifact_hash": binding["done_artifact_id"]["trimer"],
                "trimer_done_file_sha256": binding["done_file_sha256"]["trimer"],
                "shape": mcl_metadata.get("shape"),
            })
    binding["frozen_bundle_sha256"] = _canonical_json_hash(binding)
    return binding


def _capture_rng_state():
    state = {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    __import__("random").setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _gather_rng_states(local_state, distributed, rank, world_size):
    """Collect rank-local stochastic state before a distributed checkpoint."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    # All ranks call this function at the same optimizer boundary.  Object
    # gather is intentional: Python/NumPy RNG tuples are not tensors.
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _gather_rank_states(local_state, distributed, rank, world_size):
    """Gather a rank-local DataLoader/sampler state at a checkpoint barrier."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _atomic_torch_save(payload, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain UniEncoderAttention Model")
    parser.add_argument('--experiment_id', default='manual')
    parser.add_argument('--feature_config_hash', default='manual')
    parser.add_argument('--o8_feature_config_hash', default='manual')
    parser.add_argument('--model_config_hash', default='manual')
    parser.add_argument('--graph_model_config_hash', default='manual')
    parser.add_argument('--geometry_model_config_hash', default='manual')
    parser.add_argument('--source_geometry_model_config_hash', default='manual')
    parser.add_argument('--alignment_model_config_hash', default='manual')
    parser.add_argument('--training_config_hash', default='manual')
    parser.add_argument('--config_schema', default='manual')
    parser.add_argument('--config_source_schema', default='')
    parser.add_argument(
        '--modalities',
        nargs='+',
        type=parse_modality,
        default=['graph'],
        help="MTS pretraining uses the graph modality only."
    )

    parser.add_argument(
        '--fusion_type',
        type=str,
        choices=['none'],
        default='none',
        help="MTS joint pretraining has no multimodal fusion.",
    )
    parser.add_argument(
        '--fp_mode',
        type=str,
        choices=['disabled', 'ecfp', 'mixfp', 'attachment_count'],
        default='ecfp',
        help="Fingerprint implementation. ecfp keeps the original Morgan/ECFP 1024-bit FP; mixfp uses MACCSKeys + PubChemFingerprints.",
    )
    parser.add_argument('--fusion_dropout', type=float, default=0.0, help=argparse.SUPPRESS)
    # Accepted for the shared launcher COMMON vector; joint pretraining does
    # not consume per-modality dropout, the defaults mirror train.py.
    parser.add_argument('--smiles_modality_dropout', type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument('--fp_modality_dropout', type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument('--graph_modality_dropout', type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument(
        '--mips_fusion_mode',
        choices=['none'],
        default='none',
    )
    parser.add_argument(
        '--projection_mode',
        choices=['plain', 'shared_private'],
        default='shared_private',
    )
    parser.add_argument(
        '--modality_control',
        choices=['real', 'batch_shuffled', 'constant_zero'],
        default='real',
    )
    parser.add_argument(
        '--graph_input',
        type=str,
        choices=['repeat_unit', 'star_linking'],
        default='star_linking',
        help="Graph input type. 'repeat_unit' keeps the original graph; 'star_linking' removes two attachment atoms and connects their boundary atoms for graph-only topology input."
    )
    parser.add_argument(
        '--smiles_model_name',
        type=str,
        default="./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        help="Pretrained model name or path for SMILES"
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='smi_all',
        help="Name of the dataset for pretraining (unlabeled or labeled, but labels unused here)"
    )
    parser.add_argument(
        '--pretrain_stage',
        type=str,
        choices=['mts_joint_pretraining'],
        default='mts_joint_pretraining',
        help=(
            "MTS pretraining uses one joint masked-atom + Trimer bond-angle "
            "stage; property fine-tuning is handled by train.py."
        )
    )
    parser.add_argument(
        '--pretrain_profile',
        default=PRETRAIN_PROFILE_ID,
        help='Fixed MTS joint-pretraining profile.'
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='',
        help="Optional MTS joint-pretraining checkpoint to initialize or resume the fixed topology/Trimer stage."
    )
    parser.add_argument(
        '--initialization_state',
        type=str,
        default='',
        help=(
            "Fresh-paired step-0 model state. This is distinct from a "
            "pretrained checkpoint and never carries optimizer/sampler state."
        ),
    )
    parser.add_argument(
        '--paired_init_id',
        type=str,
        default='',
        help='Expected fresh-paired initialization identity.',
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help="Root directory of the dataset."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for pretraining."
    )
    parser.add_argument('--loader_workers', type=int, default=4)
    parser.add_argument('--loader_prefetch_factor', type=int, default=2)
    parser.add_argument(
        '--batch_balance', choices=['none', 'cost'], default='none',
        help='Optional topology cost-balanced distributed batches.',
    )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--amp_dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--max_steps', type=int, default=0, help='Optional preflight step limit; 0 disables it.')
    parser.add_argument(
        '--max_optimizer_steps', type=int, default=0,
        help='Exact optimizer-update budget; 0 uses the epoch budget.',
    )
    parser.add_argument(
        '--benchmark_only', action='store_true',
        help='Run a finite forward/backward throughput benchmark and exit.',
    )
    parser.add_argument(
        '--resume_smoke', action='store_true',
        help='Test-only short run; never writes a production checkpoint.',
    )
    parser.add_argument('--benchmark_batches', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducible pretraining.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=-1.0,
        help="Maximum gradient norm; values <=0 disable clipping for MTS."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pretrained_models/saved_pretrained_model.pth',
        help="Path to save the pretrained model."
    )
    parser.add_argument(
        '--resume_state', type=str, default='',
        help='Optional atomic pretraining train-state checkpoint to resume.'
    )
    parser.add_argument(
        '--checkpoint_interval_steps', type=int, default=250,
        help='Optimizer steps between resumable train-state checkpoints.'
    )
    parser.add_argument(
        '--diagnostics_dir', type=str, default='',
        help='Optional non-intrusive MSTA milestone diagnostics directory.',
    )
    parser.add_argument(
        '--diagnostic_steps', type=str,
        default='0,500,2000,5000,10000,20000',
        help='Comma-separated optimizer milestones for optional diagnostics.',
    )
    parser.add_argument(
        '--mts_num_layers',
        type=int,
        default=6,
        dest='graph_num_layers',
        help="Number of O8 MTS topology layers (fixed at 6)."
    )
    parser.add_argument(
        '--mts_hidden_dim',
        type=int,
        default=512,
        dest='graph_emb_dim',
        help="O8 MTS hidden dimension (fixed at 512)."
    )
    parser.add_argument(
        '--mts_dropout',
        type=float,
        default=0.1,
        dest='graph_dropout',
        help="O8 MTS attention dropout."
    )
    parser.add_argument(
        '--mts_num_heads',
        type=int,
        default=8,
        dest='scage_num_heads',
        help="Number of O8 MTS attention heads (fixed at 8)."
    )
    parser.add_argument(
        '--graph_encoder_type',
        type=str,
        choices=['mts', 'mips_trimer_scage'],
        default='mips_trimer_scage',
        help="Graph encoder backend: the production non-PBC MIPS-Trimer-SCAGE encoder."
    )
    parser.add_argument(
        '--mips_core',
        choices=['paper_corrected'],
        default='paper_corrected',
    )
    parser.add_argument('--mips_max_hops', type=int, default=None)
    parser.add_argument('--mips_atom_feature_mode', choices=['mips137'], default='mips137')
    parser.add_argument('--mips_attention_scale', choices=['head_dim'], default='head_dim')
    parser.add_argument('--mips_norm_mode', choices=['post'], default='post')
    parser.add_argument('--mips_activation', choices=['relu'], default='relu')
    parser.add_argument('--mips_spd_bias_mode', choices=['per_head'], default='per_head')
    parser.add_argument(
        '--mips_path_bias_mode',
        choices=['per_head_single_path_node'],
        default='per_head_single_path_node',
    )
    parser.add_argument(
        '--mips_multi_scale_hop_gate',
        action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--mips_semantics',
        choices=['paper_semantic'],
        default='paper_semantic',
    )
    parser.add_argument(
        '--mips_descriptor_fusion_mode',
        choices=['graph_md_residual'],
        default='graph_md_residual',
    )
    parser.add_argument(
        '--mips_descriptor_components',
        choices=['md200'],
        default='md200',
    )
    parser.add_argument(
        '--mips_descriptor_protocol',
        choices=['source_star_sub'],
        default='source_star_sub',
    )
    parser.add_argument('--mips_descriptor_disturbance', type=float, default=0.0)
    parser.add_argument(
        '--mips_backbone_mode',
        choices=['independent'],
        default='independent',
    )
    parser.add_argument(
        '--mips_input_norm',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        '--mips_mask_mode', choices=['zero'], default='zero',
    )
    parser.add_argument(
        '--mips_mask_policy',
        choices=['canonical_exact'],
        default='canonical_exact',
    )
    parser.add_argument(
        '--mips_masked_loss_reduction',
        choices=['atom_mean'],
        default='atom_mean',
    )
    parser.add_argument(
        '--mips_use_descriptors',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--spatial_mode',
        choices=['trimer_scage'],
        default='trimer_scage',
    )
    parser.add_argument(
        '--graph_geometry_mode',
        choices=['trimer_scage_mcl', 'current_mcl', 'g0', 'g1', 'g2', 'g3'],
        default='trimer_scage_mcl',
    )
    parser.add_argument(
        '--topology_attention_variant',
        choices=['o8', 'msta_last2'],
        default='msta_last2',
        help='T0 O8 attention or T1 MSTA in the final two layers.',
    )
    parser.add_argument('--msta_layer_indices', nargs=2, type=int, default=[4, 5])
    parser.add_argument('--msta_local_spd', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--msta_context_spd', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument(
        '--msta_share_relation_dropout',
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--msta_local_output_bias',
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument('--msta_local_output_init', choices=['zero'], default='zero')
    parser.add_argument('--g_family_arm', choices=['g0', 'g1', 'g2', 'g3'], default=None)
    parser.add_argument('--relation_geometry_sidecar', default=None)
    parser.add_argument('--relation_geometry_artifact_hash', default=None)
    parser.add_argument('--g3_permutation_sidecar', default=None)
    parser.add_argument('--g3_permutation_artifact_hash', default=None)
    parser.add_argument('--g_family_bundle_hash', default=None)
    parser.add_argument('--pretraining_objective', choices=['joint', 'masked_atom_only'], default='joint')
    parser.add_argument('--angle_loss_weight', type=float, default=0.25)
    parser.add_argument('--shared_step0_id', default=None)
    parser.add_argument(
        '--topology_representation',
        choices=['canonical_lifted', 'explicit_k_ru'],
        default='canonical_lifted',
    )
    parser.add_argument(
        '--mcl_distance_percentiles', nargs=2, type=float,
        default=[0.20, 0.50],
    )
    parser.add_argument('--trimer_num_candidates', type=int, default=4)
    parser.add_argument('--trimer_max_heavy_atoms', type=int, default=384)
    parser.set_defaults(
        finite_variant='none', conformer_mode='none',
        field_layout='none', field_channels='none',
    )
    parser.add_argument(
        '--mips_variant',
        choices=['O8'],
        default='O8',
    )
    parser.add_argument(
        '--scage_dist_bar',
        nargs='+',
        type=float,
        default=[20.0, 50.0],
        help="SCAGE multi-scale distance percentiles, e.g. 20 50."
    )
    parser.add_argument(
        '--scage_num_heads',
        type=int,
        default=16,
        help="Number of attention heads for SCAGE graph encoder."
    )
    parser.add_argument('--scage_ffn_hidden_dim', type=int, default=256)
    parser.add_argument('--scage_num_kernels', type=int, default=128)
    parser.add_argument('--scage_attention_dropout', type=float, default=0.1)
    parser.add_argument(
        '--scage_use_descriptors',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Inject explicit 3D descriptor tokens into SCAGE. Disabled by default.',
    )
    parser.add_argument(
        '--scage_distance_mode',
        choices=['bias', 'mask', 'multiscale_bias', 'mips_dual'],
        default='mips_dual',
    )
    parser.add_argument('--scage_distance_rbf', type=int, default=32)
    parser.add_argument('--scage_distance_cutoff', type=float, default=12.0)
    parser.add_argument('--scage_distance_scales', nargs='+', type=float, default=[4.0, 8.0, 12.0])
    parser.add_argument('--scage_distance_taus', nargs='+', type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument(
        '--scage_topology_bias', action=argparse.BooleanOptionalAction, default=True,
        help="Add shortest-path and direct-bond attention bias to every SCAGE layer.",
    )
    parser.add_argument('--scage_topology_max_distance', type=int, default=20)
    parser.add_argument('--scage_topology_locality_mode', choices=['hard', 'soft', 'none'], default='soft')
    parser.add_argument('--scage_topology_locality_threshold', type=int, default=5)
    parser.add_argument('--scage_topology_locality_tau', type=float, default=1.0)
    parser.add_argument(
        '--joint_embedding_dim',
        type=int,
        default=256,
        help="Shared projection dimension for modality fusion."
    )
    parser.add_argument(
        '--feature_source_dataset',
        type=str,
        default=None,
        help="Dataset used to build the SMILES-level feature cache. Defaults to --dataset_name."
    )
    parser.add_argument(
        '--disable_feature_cache',
        action='store_true',
        help="Disable SMILES-level feature cache and use legacy per-dataset caching."
    )
    parser.add_argument(
        '--rebuild_feature_cache',
        action='store_true',
        help="Force rebuild the feature cache even if it already exists."
    )
    parser.add_argument(
        '--cache_only',
        action='store_true',
        help=(
            "Build/load and validate the CPU feature cache, then exit before "
            "CUDA/NCCL/model initialization."
        ),
    )
    parser.add_argument(
        '--max_smiles_length',
        type=int,
        default=None,
        help="Override SMILES token max length. Computed from feature_source_dataset when not set."
    )
    parser.add_argument(
        '--max_smiles_length_cap',
        type=int,
        default=256,
        help="Cap for auto-computed max SMILES token length (default: 256)."
    )
    parser.add_argument(
        '--feature_cache_workers',
        type=int,
        default=0,
        help="Number of worker processes for feature cache construction. 0 or 1 keeps serial behavior."
    )
    parser.add_argument(
        '--feature_cache_chunksize',
        type=int,
        default=4,
        help="Chunksize for multiprocessing feature cache construction."
    )
    parser.add_argument(
        '--feature_cache_partial_every',
        type=int,
        default=200,
        help="Save feature cache .partial every N successful entries. Set 0 to disable partial writes."
    )
    parser.add_argument(
        '--feature_cache_item_timeout',
        type=int,
        default=45,
        help="Hard wall-clock seconds per cache item before topology-only fallback (0 disables)."
    )
    parser.add_argument(
        '--cache_layers',
        type=str,
        default='ru_base,topology,trimer,md200',
        help=(
            "Comma-separated MIPS LMDB layers to prepare/read: "
            "ru_base,topology,trimer,md200."
        ),
    )
    parser.add_argument(
        '--cache_validate',
        choices=['sample', 'full'],
        default='sample',
        help="Validate 128 records or the complete LMDB cohort after building.",
    )
    parser.add_argument(
        '--cache_commit_size',
        type=int,
        default=128,
        help="Maximum number of generated records per LMDB write transaction.",
    )
    parser.add_argument(
        '--embed_tries_multiplier',
        type=int,
        default=8,
        help="Multiplier for RDKit 3D embedding attempts per requested conformer. Total tries = CONFORMER_3D_COUNT * multiplier."
    )
    parser.add_argument(
        '--conformer_3d_count',
        type=int,
        default=8,
        help='Number of ETKDG conformer candidates before energy ranking.'
    )
    parser.add_argument(
        '--conformer_keep_count',
        type=int,
        default=4,
        help='Number of lowest-energy optimized conformers retained in the feature cache.'
    )
    parser.add_argument(
        '--conformer_profile',
        choices=['fast', 'full', 'quality'],
        default='full',
        help='Conformer search budget. fast uses bounded 100/300 ETKDG and 50-step force-field optimization.'
    )
    parser.add_argument(
        '--geom_denoise_weight',
        type=float,
        default=1.0,
        help="Weight for geometry coordinate denoising pretraining loss."
    )
    parser.add_argument(
        '--geom_noise_std',
        type=float,
        default=0.2,
        help="Fixed Gaussian coordinate noise std for geometry denoising pretraining."
    )
    parser.add_argument(
        '--geom_noise_std_min',
        type=float,
        default=None,
        help="Minimum Gaussian coordinate noise std for random-range geometry denoising. "
             "When set with --geom_noise_std_max, overrides fixed --geom_noise_std per batch."
    )
    parser.add_argument(
        '--geom_noise_std_max',
        type=float,
        default=None,
        help="Maximum Gaussian coordinate noise std for random-range geometry denoising. "
             "When set with --geom_noise_std_min, overrides fixed --geom_noise_std per batch."
    )
    parser.add_argument(
        '--graph_pretrain_weight',
        type=float,
        default=1.0,
        help="Weight for graph-specific pretraining losses."
    )
    parser.add_argument(
        '--graph_mask_atom_weight',
        type=float,
        default=1.0,
        help="Weight for masked graph atom-feature reconstruction."
    )
    parser.add_argument(
        '--graph_periodic_aug_weight',
        type=float,
        default=0.5,
        help="Weight for PerioGT-style periodicity augmentation contrastive loss."
    )
    parser.add_argument(
        '--graph_shortest_path_weight',
        type=float,
        default=1.0,
        help="Weight for SCAGE-style graph shortest-path distance prediction."
    )
    parser.add_argument(
        '--graph_angle_weight',
        type=float,
        default=0.25,
        help="Weight for SCAGE-style 3D angle prediction."
    )
    parser.add_argument(
        '--angle_objective',
        choices=['categorical', 'cosine'],
        default='categorical',
        help=(
            'Trimer angle target. categorical preserves the completed '
            '20-bin checkpoint; cosine enables the optional Angle-v2 '
            'SmoothL1 experiment.'
        ),
    )
    parser.add_argument(
        '--angle_cache_root_override', default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--scage_sp_max_distance',
        type=int,
        default=20,
        help="Maximum shortest-path class; larger distances are clipped to this bucket."
    )
    parser.add_argument(
        '--scage_angle_bins',
        type=int,
        default=20,
        help="Number of angle bins for SCAGE-style angle prediction."
    )
    parser.add_argument('--scage_mips_mask_weight', type=float, default=1.0)
    parser.add_argument('--scage_ecfp_weight', type=float, default=0.0)
    parser.add_argument(
        '--mips_spd_weight',
        type=float, default=0.0,
    )
    parser.add_argument('--mips_path_bond_weight', type=float, default=0.0)
    parser.set_defaults(scage_screw_geometry_weight=0.0)
    parser.set_defaults(
        mips_repeat_consistency_weight=0.0,
        mips_distance_weight=0.0,
        mips_conformer_weight=0.0,
    )
    parser.add_argument('--scage_ecfp_effective_max', type=float, default=0.20)
    parser.add_argument('--scage_sp_max_pairs', type=int, default=256)
    parser.add_argument('--scage_focal_gamma', type=float, default=2.0)
    parser.add_argument('--scage_boundary_angle_weight', type=float, default=0.5)
    parser.add_argument('--scage_boundary_torsion_weight', type=float, default=0.40)
    parser.add_argument('--scage_torsion_bins', type=int, default=12)
    parser.add_argument(
        '--scage_torsion_objective',
        choices=['circular', 'categorical'],
        default='circular',
        help=(
            "Boundary torsion objective. 'circular' regresses sin/cos and avoids "
            "bin discontinuities; 'categorical' preserves the legacy binned focal loss."
        ),
    )
    parser.add_argument('--scage_geometry_max_pairs', type=int, default=32)
    parser.add_argument('--scage_geometry_mask_ratio', type=float, default=0.15)
    parser.add_argument('--scage_geometry_distance_weight', type=float, default=0.25)
    parser.add_argument('--scage_geometry_angle_weight', type=float, default=0.35)
    parser.add_argument('--scage_geometry_screw_weight', type=float, default=0.20)
    parser.add_argument(
        '--scage_shift_balance_power', type=float, default=0.5,
        help=(
            "Inverse-frequency exponent for periodic image-shift cross entropy. "
            "Zero disables class balancing; 0.5 uses square-root balancing."
        ),
    )
    parser.add_argument('--repeat_cut_views', type=int, default=2)
    parser.add_argument('--repeat_cut_projection_dim', type=int, default=256)
    parser.add_argument(
        '--repeat_cut_max_mrus',
        type=int,
        default=3,
        help="Maximum MRU multiplier used by periodicity augmentation."
    )
    parser.add_argument(
        '--repeat_cut_retry',
        type=int,
        default=5,
        help="Number of augmentation attempts per SMILES before skipping it."
    )
    parser.add_argument(
        '--repeat_cut_temperature',
        type=float,
        default=0.2,
        help="Temperature for graph periodicity augmentation InfoNCE loss."
    )
    parser.add_argument(
        '--graph_mask_ratio',
        type=float,
        default=0.30,
        help="Node masking ratio for graph atom-feature reconstruction."
    )
    parser.add_argument(
        '--dynamic_pretrain_loss',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use SCAGE-style dynamic multi-task loss weighting for enabled "
            "pretraining losses. Disabled by default; enable explicitly when needed."
        ),
    )
    parser.add_argument(
        '--dynamic_loss_warmup_steps', type=int, default=200,
        help='Steps used to collect and then freeze dynamic-loss baselines.',
    )
    parser.add_argument('--dynamic_loss_recent_window', type=int, default=20)
    parser.add_argument('--dynamic_loss_temperature', type=float, default=0.5)
    parser.add_argument('--pretrain_unique_smiles', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--alignment_projection_dim', type=int, default=256)
    parser.add_argument('--alignment_fp_drop', type=float, default=0.40)
    parser.add_argument('--alignment_smiles_drop', type=float, default=0.15)
    parser.add_argument('--alignment_graph_drop', type=float, default=0.10)
    parser.add_argument('--alignment_fused_weight', type=float, default=1.0)
    parser.add_argument('--alignment_graph_smiles_weight', type=float, default=0.5)
    parser.add_argument('--alignment_graph_fp_weight', type=float, default=0.25)
    parser.add_argument('--alignment_lomo_weight', type=float, default=0.25)
    parser.add_argument('--alignment_pooling_kl_weight', type=float, default=0.01)
    parser.add_argument('--alignment_fused_mask_weight', type=float, default=0.25)
    parser.add_argument(
        '--alignment_shared_private_weight', type=float, default=0.01
    )
    parser.add_argument('--alignment_graph_lr', type=float, default=1e-5)
    parser.add_argument('--alignment_smiles_lr', type=float, default=1e-5)
    parser.add_argument('--alignment_fp_lr', type=float, default=2e-5)
    parser.add_argument('--alignment_projection_lr', type=float, default=1e-4)
    parser.add_argument('--alignment_fusion_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.10,
        help='Fraction of optimizer updates used for linear warm-up before cosine decay.',
    )
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--scheduler_power', type=float, default=1.0)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument(
        '--mips_scheduler', choices=('polynomial', 'cosine', 'linear'), default='polynomial',
        help='Learning-rate decay after warm-up. MTS uses polynomial power 1.',
    )
    args = parser.parse_args()
    # Fixed MTS route internals.  Retired geometry and peer-fusion options are
    # deliberately absent from the public CLI.
    args.gnn_model_name = ""
    args.freeze_encoder = False
    args.geom_input = 'repeat_unit'
    args.screw_kabsch_rmsd_max = 1.5
    args.screw_rotation_consistency_deg = 30.0
    args.screw_translation_relative_max = 0.30
    args.screw_final_rmsd_max = 1.5
    args.ff_gradient_rms_max = 0.05
    args.ff_gradient_max = 0.25
    args.ff_probe_steps = 20
    args.ff_probe_energy_delta_per_atom_max = 5e-5
    args.screw_energy_per_atom_max = 5.0
    args.screw_center_gradient_rms_max = 10.0
    args.scage_dist_bar = [20.0, 50.0]
    args.scage_num_heads = 8
    args.scage_ffn_hidden_dim = 2048
    args.scage_num_kernels = 128
    args.scage_attention_dropout = 0.1
    args.scage_use_descriptors = False
    args.scage_distance_mode = 'mips_dual'
    args.scage_distance_rbf = 32
    args.scage_distance_cutoff = 12.0
    args.scage_distance_scales = [4.0, 8.0, 12.0]
    args.scage_distance_taus = [0.5, 1.0, 1.5]
    args.scage_topology_bias = True
    args.scage_topology_max_distance = 20
    args.scage_topology_locality_mode = 'soft'
    args.scage_topology_locality_threshold = 5
    args.scage_topology_locality_tau = 1.0
    args.scage_periodic_image_mode = 'none'
    args.scage_periodic_image_cap = 0
    args.scage_periodic_image_temperature = 0.5
    args.scage_force_topology_only = True
    args.scage_use_pbc_distance = False
    # ``mts`` is the public backend spelling.  Keep the historical internal
    # selector so Dataset/model code and immutable cache metadata remain
    # compatible during the naming migration.
    if args.graph_encoder_type == "mts":
        args.graph_encoder_type = MTS_ROUTE_INTERNAL
    return args

def _base_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(model, MIPSPretrainContainer):
        return model.model
    return model


def _load_fresh_paired_initialization(model, args):
    """Strictly load an optimizer-free fresh-paired step-0 model state."""

    path = Path(str(args.initialization_state)).resolve()
    if not path.is_file():
        raise RuntimeError(f"fresh-paired initialization state is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("fresh-paired initialization must contain state_dict")
    meta = dict(payload.get("meta") or {})
    expected_identity = (
        "T1" if str(args.topology_attention_variant) == "msta_last2" else "T0"
    )
    required = {
        "schema": "mts-pretrain-init-v1",
        "initialization": "fresh_paired",
        "model_identity": expected_identity,
        "optimizer_steps": 0,
        "paired_init_id": str(args.paired_init_id or "mts_t_pretrain0_matched_v1"),
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"fresh-paired initialization metadata mismatch for {key}: "
                f"expected={expected!r}, observed={meta.get(key)!r}"
            )
    # G-family step-0 artifacts are full UniEncoder states, but their
    # scientific identity is distinct from the historical T-Pretrain-0 pair.
    # Bind every requested arm to its own shared step-0 id before loading any
    # tensor, so a T-family or wrong-arm initialization cannot be reused.
    if getattr(args, "g_family_arm", None) is not None:
        g_arm = str(args.g_family_arm)
        g_required = {
            "g_family_arm": g_arm,
            "geometry_mode": g_arm,
            "pretraining_objective": "masked_atom_only",
            "angle_loss_weight": 0.0,
            "shared_step0_id": str(args.shared_step0_id or args.paired_init_id or ""),
            "g_family_bundle_hash": str(args.g_family_bundle_hash or ""),
        }
        if not g_required["shared_step0_id"]:
            raise RuntimeError("G-family initialization requires --shared_step0_id")
        for key, expected in g_required.items():
            observed = meta.get(key)
            if key == "angle_loss_weight":
                try:
                    matches = float(observed) == float(expected)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = observed == expected
            if not matches:
                raise RuntimeError(
                    "G-family step-0 initialization metadata mismatch for "
                    f"{key}: expected={expected!r}, observed={observed!r}"
                )
    state = payload["state_dict"]
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    if missing or unexpected:
        raise RuntimeError(
            "fresh-paired initialization architecture mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_mismatch = sorted(
        key for key in expected_state
        if tuple(state[key].shape) != tuple(expected_state[key].shape)
        or state[key].dtype != expected_state[key].dtype
    )
    if shape_mismatch:
        raise RuntimeError(
            "fresh-paired initialization shape/dtype mismatch: "
            + ", ".join(shape_mismatch[:8])
        )
    model.load_state_dict(state, strict=True)
    if expected_identity == "T1":
        for index in (4, 5):
            key = f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"
            value = state[key]
            if not torch.isfinite(value).all() or torch.count_nonzero(value).item() != 0:
                raise RuntimeError(f"T1 fresh local_output is not finite zero: {key}")
    return {
        "schema": str(meta["schema"]),
        "initialization": str(meta["initialization"]),
        "model_identity": expected_identity,
        "paired_init_id": str(meta["paired_init_id"]),
        "path": str(path),
        "sha256": _file_sha256(path),
        "target_config_hash": meta.get("target_config_hash"),
        "target_graph_model_config_hash": meta.get("target_graph_model_config_hash"),
        "optimizer_steps": int(meta["optimizer_steps"]),
        "g_family_arm": meta.get("g_family_arm"),
        "shared_step0_id": meta.get("shared_step0_id"),
        "pretraining_objective": meta.get("pretraining_objective"),
        "g_family_bundle_hash": meta.get("g_family_bundle_hash"),
    }


def _msta_attention_modules(model):
    """Return the declared T1 local/context attention modules."""

    base = _base_model(model)
    encoder = base.encoders["graph"].encoder
    output = []
    for index in (4, 5):
        layer = encoder.layers[index]
        attention = getattr(layer, "attention", None)
        if hasattr(attention, "local_output"):
            output.append((index, attention))
    return output


def _set_msta_diagnostic_mode(model, *, capture=False, local_off=False):
    for _, attention in _msta_attention_modules(model):
        attention.diagnostic_capture = bool(capture)
        attention.diagnostic_local_off = bool(local_off)
        if not capture:
            attention.last_diagnostic = None


def _make_fixed_probe(data):
    """Keep a deterministic CPU snapshot without touching the loader."""

    # The project collator intentionally returns a custom DataBatch that is
    # not reconstructible through PyG's ``to_data_list``.  A detached CPU
    # snapshot is still a fixed probe: it preserves the first batch's sample
    # keys and never advances the train iterator or RNG.
    try:
        return data.detach().cpu()
    except Exception:
        return data.cpu()


@torch.no_grad()
def _fixed_probe_losses(owner, probe_cpu, args, step, device):
    if probe_cpu is None:
        return None
    previous_training = owner.training
    previous_flags = [
        (attention, bool(attention.diagnostic_capture), bool(attention.diagnostic_local_off))
        for _, attention in _msta_attention_modules(owner)
    ]
    rng_state = _capture_rng_state()
    owner.eval()
    probe = probe_cpu.to(device)

    def evaluate(local_off):
        _set_msta_diagnostic_mode(owner, capture=False, local_off=local_off)
        payload = owner(MTS_STAGE1_ID, probe, args, int(step))
        counts = payload["counts"]
        atom = payload["loss_terms"]["masked_atom_sum"] / max(1, int(counts["masked_atoms"]))
        angle = payload["loss_terms"]["angle_sum"] / max(1, int(counts["angle_graphs"]))
        return float(atom.float().item()), float(angle.float().item())

    normal_mask, normal_angle = evaluate(False)
    off_mask, off_angle = evaluate(True)
    for attention, capture, local_off in previous_flags:
        attention.diagnostic_capture = capture
        attention.diagnostic_local_off = local_off
    if previous_training:
        owner.train()
    _restore_rng_state(rng_state)
    return {
        "delta_L_mask": off_mask - normal_mask,
        "delta_L_angle": off_angle - normal_angle,
        "normal_mask": normal_mask,
        "normal_angle": normal_angle,
        "probe_graph_count": int(len(probe.smiles)) if hasattr(probe, "smiles") else None,
    }


def _diagnostic_record(model, optimizer_step, probe, args, device, owner):
    """Collect scalar-only training/probe diagnostics at one declared step."""

    layers = {}
    for index, attention in _msta_attention_modules(model):
        row = dict(attention.last_diagnostic or {})
        parameter = attention.local_output.weight
        gradient = parameter.grad
        row.update({
            "weight_norm": float(parameter.detach().float().norm().item()),
            "grad_norm": (
                float(gradient.detach().float().norm().item())
                if gradient is not None else None
            ),
            "grad_present": gradient is not None,
            "grad_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "weight_finite": bool(torch.isfinite(parameter).all()),
        })
        layers[str(index)] = row
    probe_result = _fixed_probe_losses(owner, probe, args, optimizer_step, device)
    return {
        "optimizer_step": int(optimizer_step),
        "layers": layers,
        "probe": probe_result,
        "finite": bool(
            all(bool(row.get("finite", True)) for row in layers.values())
            and all(bool(row.get("weight_finite", False)) for row in layers.values())
            and all(bool(row.get("grad_finite", False)) for row in layers.values())
        ),
    }


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _distributed_sum_count_mean(local_sum, local_count, device):
    """Convert a local sum into a global sample-weighted mean.

    DDP averages gradients across ranks, so multiplying the local sum by
    ``world_size / global_count`` yields the same gradient as a single global
    mean even when ranks contain different numbers of valid targets.
    """
    count = torch.tensor(float(local_count), device=device, dtype=torch.float32)
    global_count = count.clone()
    world_size = 1
    if _distributed_enabled():
        world_size = dist.get_world_size()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0.0:
        return local_sum * 0.0, 0
    return local_sum * (float(world_size) / global_count), int(global_count.item())


def _distributed_sum_int(local_value, device):
    """All-reduce a small integer statistic without synchronizing gradients."""
    value = torch.tensor(int(local_value), device=device, dtype=torch.long)
    if _distributed_enabled():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _distributed_joint_counts(counts, device):
    """Reduce all joint-pretraining counters in one collective."""
    names = (
        "masked_atoms", "angle_graphs", "masked_correct", "angle_correct",
        "angle_targets", "mcl_valid_graphs", "graphs",
    )
    packed = torch.tensor(
        [int(counts[name]) for name in names], device=device, dtype=torch.long
    )
    if _distributed_enabled():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().tolist()
    return dict(zip(names, (int(value) for value in values)))


def _global_mean_from_count(local_sum, global_count):
    """DDP-correct global mean after counts have already been reduced."""
    if int(global_count) <= 0:
        return local_sum * 0.0
    world_size = dist.get_world_size() if _distributed_enabled() else 1
    return local_sum * (float(world_size) / float(global_count))


def _all_reduce_gradients(modules):
    """Legacy synchronization used only by non-MIPS training routes.

    The non-PBC MIPS Stage 1/2 route is wrapped in real
    ``DistributedDataParallel`` and must never call this helper.
    """
    if not _distributed_enabled():
        return
    world_size = dist.get_world_size()
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


@torch.no_grad()
def _assert_ddp_parameters_synced(module, step, samples_per_tensor=16):
    """Cheap every-update divergence detector for the three-rank campaign."""
    if not dist.is_available() or not dist.is_initialized():
        return
    checksum = torch.zeros(3, device=next(module.parameters()).device, dtype=torch.float64)
    for parameter in module.parameters():
        flat = parameter.detach().reshape(-1)
        if not flat.numel():
            continue
        if flat.numel() <= int(samples_per_tensor):
            sample = flat.double()
        else:
            sample_count = int(samples_per_tensor)
            indices = (
                torch.arange(sample_count, device=flat.device, dtype=torch.long)
                * (flat.numel() - 1)
                // (sample_count - 1)
            )
            sample = flat[indices].double()
        checksum[0] += sample.sum()
        checksum[1] += sample.square().sum()
        checksum[2] += sample.abs().max()
    minimum, maximum = checksum.clone(), checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    tolerance = 1e-10 * maximum.abs().clamp_min(1.0)
    if bool(((maximum - minimum).abs() > tolerance).any()):
        raise RuntimeError(
            f"DDP parameter divergence detected after optimizer step {step}: "
            f"min={minimum.tolist()}, max={maximum.tolist()}"
        )


def _distributed_sum_mapping(values, device):
    """Sum a scalar mapping across ranks, including keys absent on some ranks."""
    if not _distributed_enabled():
        return dict(values)
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, sorted(values))
    keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
    if not keys:
        return {}
    tensor = torch.tensor(
        [float(values.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def _gradient_vector(modules):
    values = [
        parameter.grad.detach().float().flatten()
        for module in modules for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(values) if values else torch.zeros(1)


def _bf16_parity_gate(base_model, data, atom_head, args, geometry_adapt=False):
    modules = (base_model, atom_head)
    for module in modules:
        module.zero_grad(set_to_none=True)
    fp32_loss, _ = _mips_masked_atom_loss(
        base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0,
        geometry_adapt=geometry_adapt,
    )
    fp32_loss.backward()
    fp32_grad = _gradient_vector(modules)
    for module in modules:
        module.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss, _ = _mips_masked_atom_loss(
            base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0,
            geometry_adapt=geometry_adapt,
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector(modules)
    relative_delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    for module in modules:
        module.zero_grad(set_to_none=True)
    passed = finite and relative_delta <= 0.02 and cosine >= 0.98
    return passed, {"relative_loss_delta": relative_delta, "gradient_cosine": cosine, "finite": finite}


def _unique_smiles_indices(dataset):
    seen = set()
    indices = []
    raw_items = getattr(dataset, "data_list", None)
    for index in range(len(dataset)):
        # Lazy v2 caches keep ``(smiles, target)`` rows and only deserialize a
        # PyG object in __getitem__. Reading the tuple avoids one million
        # random shard reads merely to discover that PI1M_v2 is already unique.
        raw_item = raw_items[index] if raw_items is not None else dataset[index]
        smiles = str(raw_item[0] if isinstance(raw_item, tuple) else raw_item.smiles)
        if smiles in seen:
            continue
        seen.add(smiles)
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


class _ShardAwareDistributedSampler(torch.utils.data.Sampler):
    """DDP sampler that preserves lazy 2048-item cache locality.

    Each rank owns one fixed contiguous source range, so it reads only about
    one third of the immutable cache shards instead of all ranks repeatedly
    deserializing every shard. Shard order and rows inside each owned shard are
    independently shuffled every epoch. The equal contiguous ranges preserve
    identical global coverage (apart from the normal distributed tail drop).
    """

    def __init__(
        self, dataset_size, num_replicas, rank, seed, shard_size=2048
    ):
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shard_size = int(shard_size)
        self.epoch = 0
        self.num_samples = self.dataset_size // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples


class _RankSliceSampler(torch.utils.data.Sampler):
    """Non-padding deterministic sampler for distributed evaluation."""

    def __init__(self, length, rank, world_size):
        self.indices = tuple(range(int(rank), int(length), int(world_size)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        rank_start = self.rank * self.num_samples
        rank_end = rank_start + self.num_samples
        blocks = []
        cursor = rank_start
        while cursor < rank_end:
            shard_end = ((cursor // self.shard_size) + 1) * self.shard_size
            block_end = min(rank_end, shard_end)
            blocks.append(list(range(cursor, block_end)))
            cursor = block_end
        block_order = torch.randperm(
            len(blocks), generator=generator
        ).tolist()
        permutation = []
        for block_id in block_order:
            block = blocks[block_id]
            order = torch.randperm(
                len(block), generator=generator
            ).tolist()
            permutation.extend(block[offset] for offset in order)
        if len(permutation) != self.num_samples:
            raise RuntimeError(
                "rank-contiguous shard sampler produced an invalid length"
            )
        return iter(permutation)


class _CostBalancedDistributedSampler(torch.utils.data.Sampler):
    """Deterministic fixed-size batches balanced by graph cost."""

    def __init__(
        self, costs, batch_size, num_replicas, rank, seed, drop_last=False,
    ):
        super().__init__()
        values = np.asarray(costs, dtype=np.float64)
        if values.ndim == 2:
            values = values[:, 0] + 0.25 * values[:, 1]
        if values.ndim != 1:
            raise ValueError("topology costs must be [N] or [N,2]")
        self.costs = values
        self.dataset_size = int(values.size)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        global_batch = self.batch_size * self.num_replicas
        self.num_batches = (
            self.dataset_size // global_batch
            if self.drop_last else int(math.ceil(self.dataset_size / global_batch))
        )
        self.epoch = 0
        # Number of rank-local samples already consumed in the current epoch.
        # This cursor is intentionally not included in ``__len__``: training
        # uses absolute batch indices for accumulation and end-of-epoch logic.
        # A resumed iterator can therefore start directly at the checkpointed
        # batch without deserializing every preceding LMDB record.
        self.start_sample_index = 0

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.start_sample_index = 0

    def set_start_batch(self, batch_index):
        sample_index = int(batch_index) * self.batch_size
        if sample_index < 0 or sample_index > len(self):
            raise ValueError(
                f"invalid cost-sampler resume batch {batch_index}: "
                f"sample offset {sample_index}, length {len(self)}"
            )
        self.start_sample_index = sample_index

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(self.dataset_size, dtype=np.int64)
        rng.shuffle(order)
        global_batch = self.batch_size * self.num_replicas
        required = self.num_batches * global_batch
        if required > order.size:
            if self.drop_last:
                order = order[:required]
            else:
                order = np.concatenate([order, np.resize(order, required - order.size)])
        else:
            order = order[:required]
        output = []
        for start in range(0, required, global_batch):
            window = order[start:start + global_batch]
            rank_indices = [[] for _ in range(self.num_replicas)]
            rank_costs = [0.0 for _ in range(self.num_replicas)]
            for index in window[np.argsort(-self.costs[window], kind="stable")]:
                candidates = [r for r in range(self.num_replicas)
                              if len(rank_indices[r]) < self.batch_size]
                target = min(candidates, key=lambda r: (rank_costs[r], r))
                rank_indices[target].append(int(index))
                rank_costs[target] += float(self.costs[index])
            output.extend(rank_indices[self.rank])
        return iter(output[self.start_sample_index:])


def _polymer_ecfp_pos_weight(dataset, indices, device):
    positives = torch.zeros(2048, dtype=torch.float64)
    valid_count = 0
    for index in indices:
        sample = dataset[int(index)]
        valid = getattr(sample, 'polymer_ecfp_valid', False)
        valid = bool(valid.flatten()[0].item()) if torch.is_tensor(valid) else bool(valid)
        if not valid:
            continue
        target = getattr(sample, 'polymer_ecfp_target', None)
        if target is None or target.numel() != 2048:
            continue
        positives += target.detach().cpu().double().view(-1)
        valid_count += 1
    if valid_count == 0:
        raise ValueError("No valid 2048-bit Polymer ECFP targets were found")
    negatives = float(valid_count) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp_(1.0, 20.0).float().to(device)
    return pos_weight, valid_count


def _merge_periodic_aug_stats(total, update):
    for key in (
        'attempted', 'success', 'skipped', 'same_smiles', 'valid_contrastive',
        'graph_cache_hits', 'graph_cache_misses',
    ):
        total[key] = int(total.get(key, 0)) + int(update.get(key, 0))
    return total


def _finalize_periodic_aug_stats(epoch, stats):
    attempted = int(stats.get('attempted', 0))
    success = int(stats.get('success', 0))
    return {
        'epoch': int(epoch),
        'attempted': attempted,
        'success': success,
        'skipped': int(stats.get('skipped', 0)),
        'same_smiles': int(stats.get('same_smiles', 0)),
        'valid_contrastive': int(stats.get('valid_contrastive', 0)),
        'graph_cache_hits': int(stats.get('graph_cache_hits', 0)),
        'graph_cache_misses': int(stats.get('graph_cache_misses', 0)),
        'success_rate': float(success / attempted) if attempted else None,
        'same_smiles_rate': float(stats.get('same_smiles', 0) / attempted) if attempted else None,
    }


def _stage_loss_weights(args):
    if args.pretrain_stage == 'scage_m4p':
        return {'contrastive': 0.0, 'graph': 1.0, 'geom': 0.0}
    if args.pretrain_stage == 'graph_geom':
        return {
            'contrastive': 0.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    if args.pretrain_stage == 'alignment':
        if args.graph_encoder_type == 'mips_trimer_scage':
            return {'contrastive': 1.0, 'graph': 0.0, 'geom': 0.0}
        return {
            'contrastive': 1.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    return {
        'contrastive': 1.0,
        'graph': float(args.graph_pretrain_weight),
        'geom': float(args.geom_denoise_weight),
    }


class DynamicPretrainLossWeighter:
    """SCAGE-style dynamic weighting with priors applied after normalization."""

    def __init__(
        self,
        task_names,
        init_window=200,
        recent_window=20,
        temperature=1.0,
        task_priors=None,
        effective_caps=None,
        device=None,
    ):
        self.task_names = list(task_names)
        self.init_window = int(init_window)
        self.recent_window = int(recent_window)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("dynamic loss temperature must be positive")
        self.device = device
        self.task_priors = {
            name: float((task_priors or {}).get(name, 1.0)) for name in self.task_names
        }
        self.effective_caps = {
            name: float(value) for name, value in (effective_caps or {}).items()
        }
        self.step = 0
        self.loss_init = {name: [] for name in self.task_names}
        self.loss_recent = {name: [] for name in self.task_names}
        self.loss_prev_recent = {name: [] for name in self.task_names}
        self.last_weights = {name: 1.0 / max(1, len(self.task_names)) for name in self.task_names}
        self.last_normalized = {name: 0.0 for name in self.task_names}
        self.last_effective = {name: 0.0 for name in self.task_names}
        self.last_baselines = {name: None for name in self.task_names}

    def _synchronized_statistics(self, loss_terms, reference=None):
        """Return detached per-task means shared by every distributed rank."""
        if not _distributed_enabled():
            return {
                name: loss_terms[name].detach()
                for name in self.task_names if name in loss_terms
            }

        reference = (
            next(iter(loss_terms.values()))
            if loss_terms else reference
        )
        if reference is None:
            raise ValueError("A differentiable reference is required for an empty local task set")
        values = reference.new_zeros(len(self.task_names))
        counts = reference.new_zeros(len(self.task_names))
        for idx, name in enumerate(self.task_names):
            if name in loss_terms:
                values[idx] = loss_terms[name].detach()
                counts[idx] = 1.0
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return {
            name: values[idx] / counts[idx].clamp_min(1.0)
            for idx, name in enumerate(self.task_names)
            if float(counts[idx].item()) > 0.0
        }

    def _mean_or_current(self, values, current_value):
        if not values:
            return current_value.detach().clamp_min(1e-8)
        return torch.stack(values).mean().to(current_value.device).clamp_min(1e-8)

    def _push(self, values, value, max_len):
        values.append(value.detach())
        if len(values) > max_len:
            values.pop(0)

    def _apply_effective_caps(self, effective, active_names):
        """Cap final coefficients and redistribute excess among uncapped tasks."""
        effective = effective / effective.sum().clamp_min(1e-8)
        for _ in range(len(active_names)):
            capped = []
            excess = effective.new_tensor(0.0)
            for idx, name in enumerate(active_names):
                cap = self.effective_caps.get(name)
                if cap is not None and float(effective[idx].item()) > cap:
                    excess = excess + effective[idx] - cap
                    effective[idx] = cap
                    capped.append(idx)
            if float(excess.item()) <= 0:
                break
            recipients = [idx for idx in range(len(active_names)) if idx not in capped]
            if not recipients:
                break
            recipient_values = effective[recipients]
            if float(recipient_values.sum().item()) <= 0:
                effective[recipients] += excess / len(recipients)
            else:
                effective[recipients] += excess * recipient_values / recipient_values.sum()
        return effective / effective.sum().clamp_min(1e-8)

    def __call__(self, loss_terms, reference=None):
        statistics = self._synchronized_statistics(loss_terms, reference=reference)
        active_names = [name for name in self.task_names if name in statistics]
        if not active_names:
            if reference is None:
                raise ValueError("dynamic pretrain loss received no active terms")
            self.last_weights = {}
            self.last_normalized = {}
            self.last_effective = {}
            return reference * 0.0
        reference = next(iter(loss_terms.values())) if loss_terms else reference

        def local_loss(name):
            # A task can be valid on other ranks but absent from this rank's
            # batch. Its local gradient contribution is correctly zero while
            # every rank still updates identical dynamic-weight state.
            return loss_terms.get(name, reference * 0.0)

        if len(active_names) == 1:
            only_name = active_names[0]
            self.last_weights = {only_name: 1.0}
            self.last_normalized = {
                only_name: float(statistics[only_name].detach().cpu().item())
            }
            self.last_effective = {only_name: 1.0}
            self._record(statistics)
            return local_loss(only_name)

        if self.step < self.init_window:
            priors = reference.new_tensor(
                [self.task_priors[name] for name in active_names]
            )
            prior_sum = priors.sum().clamp_min(1e-8)
            effective = self._apply_effective_caps(priors / prior_sum, active_names)
            total = torch.sum(torch.stack([local_loss(name) for name in active_names]) * effective)
            self._record(statistics)
            self.last_weights = {name: 1.0 / len(active_names) for name in active_names}
            self.last_normalized = {
                name: float(statistics[name].detach().cpu().item()) for name in active_names
            }
            self.last_effective = {
                name: float(effective[idx].detach().cpu().item())
                for idx, name in enumerate(active_names)
            }
            return total

        normalized = []
        normalized_statistics = []
        trend_ratios = []
        for name in active_names:
            statistic = statistics[name]
            init_mean = self._mean_or_current(self.loss_init[name], statistic)
            recent_mean = self._mean_or_current(self.loss_recent[name], statistic)
            prev_recent_mean = self._mean_or_current(self.loss_prev_recent[name], recent_mean)
            normalized.append(local_loss(name) / init_mean)
            normalized_statistics.append(statistic / init_mean)
            trend_ratios.append(recent_mean / prev_recent_mean)

        trend_tensor = torch.stack(trend_ratios)
        weights = F.softmax(trend_tensor / self.temperature, dim=0).detach()
        normalized_tensor = torch.stack(normalized)
        priors = normalized_tensor.new_tensor([self.task_priors[name] for name in active_names])
        effective = priors * weights
        effective = self._apply_effective_caps(effective, active_names)
        total = torch.sum(normalized_tensor * effective)
        self.last_weights = {name: float(weights[idx].detach().cpu().item()) for idx, name in enumerate(active_names)}
        self.last_normalized = {
            name: float(normalized_statistics[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_effective = {
            name: float(effective[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_baselines = {
            name: float(self._mean_or_current(self.loss_init[name], statistics[name]).cpu().item())
            for name in active_names
        }
        self._record(statistics)
        return total

    def _record(self, loss_terms):
        for name in self.task_names:
            if name not in loss_terms:
                continue
            value = loss_terms[name]
            # Baselines are collected only during warm-up and then frozen.
            if self.step < self.init_window:
                self._push(self.loss_init[name], value, self.init_window)
            if self.step % self.recent_window == 0 and self.loss_recent[name]:
                self.loss_prev_recent[name] = list(self.loss_recent[name])
                self.loss_recent[name] = []
            self._push(self.loss_recent[name], value, self.recent_window)
        self.step += 1


def _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio):
    if 'graph' not in base_model.encoders:
        return data.x.new_tensor(0.0)

    graph_module = base_model.encoders['graph']
    graph_encoder = graph_module.encoder
    from src.dataset.graph_data import allowable_features
    num_atom_symbols = len(allowable_features['possible_atom_symbols'])
    atom_symbol_targets = data.x[:, :num_atom_symbols].argmax(dim=1).long()
    mask = torch.rand(data.x.size(0), device=data.x.device) < float(mask_ratio)
    if not mask.any():
        mask[torch.randint(data.x.size(0), (1,), device=data.x.device)] = True

    masked_x = data.x.clone()
    masked_x[mask] = 0.0
    _, node_rep = _graph_encode_nodes(base_model, data, x_override=masked_x)
    return F.cross_entropy(graph_atom_head(node_rep[mask]), atom_symbol_targets[mask])


def _empty_periodic_aug_stats():
    return {
        'attempted': 0,
        'success': 0,
        'skipped': 0,
        'same_smiles': 0,
        'valid_contrastive': 0,
        'graph_cache_hits': 0,
        'graph_cache_misses': 0,
    }


def _graph_periodic_aug_loss(base_model, data, graph_input, temperature, max_mrus, retry):
    stats = _empty_periodic_aug_stats()
    if 'graph' not in base_model.encoders or not hasattr(data, 'smiles'):
        return data.x.new_tensor(0.0), stats

    from torch_geometric.data import Batch
    from src.dataset.graph_data import build_mips_graph_for_input, repeat_cut_augment_smiles

    graph_encoder = base_model.encoders['graph'].encoder
    aug_graphs = []
    valid_indices = []
    for sample_idx, smiles in enumerate(data.smiles):
        stats['attempted'] += 1
        aug_graph = None
        aug_smiles = None
        attempts = max(1, int(retry))
        for _ in range(attempts):
            try:
                aug_smiles, _ = repeat_cut_augment_smiles(
                    smiles,
                    max_mrus=max_mrus,
                    return_n=True,
                )
                aug_graph = build_mips_graph_for_input(aug_smiles, graph_input=graph_input)
                break
            except Exception:
                aug_graph = None
        if aug_graph is None:
            stats['skipped'] += 1
            continue
        if str(aug_smiles) == str(smiles):
            stats['same_smiles'] += 1
            stats['skipped'] += 1
            continue
        stats['success'] += 1
        aug_graphs.append(aug_graph)
        valid_indices.append(sample_idx)

    stats['valid_contrastive'] = len(valid_indices)
    if len(valid_indices) < 2:
        return data.x.new_tensor(0.0), stats

    # PerioGT augmentation changes the repeat-unit graph but does not provide a
    # synchronized PBC conformer.  SCAGE therefore uses topology distance for
    # both views of this auxiliary task, avoiding asymmetric geometry inputs.
    if getattr(graph_encoder, 'uses_geometry', False):
        orig_graph, _ = graph_encoder.forward_topology_only(data)
    else:
        orig_graph, _ = _graph_encode_nodes(base_model, data)
    valid_index = torch.tensor(valid_indices, dtype=torch.long, device=orig_graph.device)
    z_orig = orig_graph.index_select(0, valid_index)

    aug_batch = Batch.from_data_list(aug_graphs).to(data.x.device)
    if getattr(graph_encoder, 'uses_geometry', False):
        z_aug, _ = graph_encoder.forward_topology_only(aug_batch)
    else:
        z_aug, _ = graph_encoder(aug_batch.x, aug_batch.edge_index, aug_batch.edge_attr, aug_batch.batch)

    z_orig = F.normalize(z_orig, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z_orig, z_aug.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss, stats


def _is_scage_graph_encoder(base_model):
    return (
        'graph' in base_model.encoders
        and getattr(
            base_model.encoders['graph'].encoder, 'architecture_name', ''
        ) == 'mips_localized_non_pbc'
    )


def _graph_encode_nodes(
    base_model, data, x_override=None, geometry_adapt=False
):
    graph_encoder = base_model.encoders['graph'].encoder
    if x_override is not None and hasattr(graph_encoder, 'forward_with_x'):
        if geometry_adapt:
            return graph_encoder.forward_geometry_with_x(data, x_override)
        return graph_encoder.forward_with_x(data, x_override)
    if getattr(graph_encoder, 'expects_data', False):
        return graph_encoder(data)
    if getattr(graph_encoder, 'uses_geometry', False):
        return graph_encoder(data)
    x = data.x if x_override is None else x_override
    return graph_encoder(x, data.edge_index, data.edge_attr, data.batch)


def _graph_encode_geometry_pretext(base_model, data):
    graph_encoder = base_model.encoders['graph'].encoder
    if not hasattr(graph_encoder, 'forward_geometry_pretext'):
        raise ValueError("geometry pretext encoding requires a SCAGE graph encoder")
    return graph_encoder.forward_geometry_pretext(data)


def _mean_loss_by_graph(per_target_loss, graph_ids):
    """Average targets within graph, then average graphs with valid targets."""
    graph_losses = []
    for graph_id in torch.unique(graph_ids, sorted=True):
        selected = graph_ids == graph_id
        if bool(selected.any()):
            graph_losses.append(per_target_loss[selected].mean())
    if not graph_losses:
        return per_target_loss.sum() * 0.0
    return torch.stack(graph_losses).mean()


def _mips_masked_atom_loss(
    base_model, data, prediction_head, mask_ratio, seed, epoch,
    geometry_adapt=False,
):
    """MIPS fused masked-atom prediction with deterministic per-polymer masks."""
    graph_encoder = base_model.encoders['graph'].encoder
    mask_policy = getattr(graph_encoder, "mask_policy", "canonical_exact")
    masks = []
    geometry_valid = _mcl_valid_graph_mask(data) if geometry_adapt else None
    for graph_idx, smiles in enumerate(data.smiles):
        if hasattr(data, "graph_available") and not bool(
            data.graph_available.flatten()[graph_idx].item()
        ):
            continue
        if geometry_adapt and not bool(geometry_valid[graph_idx]):
            continue
        node_indices = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not node_indices.numel():
            continue
        digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{smiles}".encode("utf-8")).digest()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2 ** 63 - 1))
        if mask_policy == "expanded_bernoulli":
            selected = (
                torch.rand(node_indices.numel(), generator=generator)
                < float(mask_ratio)
            ).to(node_indices.device)
            selected_nodes = node_indices[selected]
        elif mask_policy == "canonical_exact":
            canonical = getattr(data, "canonical_ru_atom_index", None)
            if canonical is None:
                canonical = torch.arange(
                    node_indices.numel(), device=node_indices.device
                )
            else:
                canonical = canonical[node_indices]
            groups = torch.unique(canonical, sorted=True)
            count = max(1, int(round(float(mask_ratio) * groups.numel())))
            selected = torch.randperm(groups.numel(), generator=generator)[:count]
            selected_groups = groups[selected.to(groups.device)]
            selected_nodes = node_indices[
                torch.isin(canonical, selected_groups)
            ]
        else:
            raise ValueError(f"unsupported MIPS mask policy: {mask_policy}")
        if not selected_nodes.numel():
            continue
        masks.append(selected_nodes)
    if not masks:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(masks)
    x_override = data.x.clone()
    x_override[mask_indices] = 0.0
    _, node_rep = _graph_encode_nodes(
        base_model, data, x_override=x_override,
        geometry_adapt=geometry_adapt,
    )
    # Released MIPS derives labels from the first 101 entries of its 137-wide
    # atom feature vector (100 elements plus the unknown category).
    canonical = getattr(data, "canonical_ru_atom_index", None)
    if canonical is not None and mask_policy == "canonical_exact":
        # Count one representative per canonical atom even though every
        # equivalent copy was hidden from the encoder.
        representatives = []
        identities = torch.stack(
            [data.batch[mask_indices], canonical[mask_indices]], dim=1
        )
        for identity in torch.unique(identities, dim=0, sorted=True):
            matching = (
                (data.batch[mask_indices] == identity[0])
                & (canonical[mask_indices] == identity[1])
            )
            representatives.append(
                mask_indices[
                    torch.nonzero(matching, as_tuple=False)[0, 0]
                ]
            )
        target_indices = torch.stack(representatives)
    else:
        target_indices = mask_indices
    if getattr(graph_encoder, "masked_atom_target", "scage119") == "mips101":
        targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    elif getattr(graph_encoder, "architecture_name", "") == "mips_localized_non_pbc":
        targets = data.atomic_num[target_indices].long()
    else:
        targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[target_indices])
    per_target = F.cross_entropy(
        logits.float(), targets, reduction="none"
    )
    if getattr(graph_encoder, "masked_loss_reduction", "graph_mean") == "atom_mean":
        loss = per_target.mean()
    else:
        loss = _mean_loss_by_graph(per_target, data.batch[target_indices])
    return loss, int(target_indices.numel())


def _undirected_lga_relation_ids(data):
    """Return graph-local undirected identities for every LGA edge.

    Cached topology records created before this fix contain directed
    ``canonical_pair_index`` values.  Deriving the identity from the cached
    node mapping at runtime fixes reverse-edge leakage without invalidating
    the immutable topology LMDB.
    """
    source, target = data.lga_edge_index.long()
    canonical = data.canonical_ru_atom_index.long()
    graph = data.batch[target].long()
    left = torch.minimum(canonical[source], canonical[target])
    right = torch.maximum(canonical[source], canonical[target])
    keys = torch.stack((graph, left, right), dim=1)
    _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    return inverse


def _mcl_valid_graph_mask(data):
    """Return the exact per-graph geometry mask used by Trimer-MCL."""
    cached = getattr(data, "mcl_valid", None)
    if cached is not None:
        return torch.as_tensor(
            cached, dtype=torch.bool, device=data.graph_available.device
        ).flatten()
    graph_available = data.graph_available.bool().flatten()
    return torch.tensor(
        [trimer_can_enter_mcl(data, graph_idx)
         for graph_idx in range(int(graph_available.numel()))],
        dtype=torch.bool,
        device=graph_available.device,
    )


def _select_mips_relation_targets(data, max_pairs):
    """Select one shared, undirected relation set for SPD and path losses."""
    valid = data.graph_available[data.batch[data.lga_edge_index[1]]].bool()
    edge_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not edge_indices.numel():
        return None
    selected_edges = []
    edge_graphs = data.batch[data.lga_edge_index[1, edge_indices]]
    for graph_id in torch.unique(edge_graphs, sorted=True):
        graph_edges = edge_indices[edge_graphs == graph_id]
        selection = _stratified_pair_selection(
            data.lga_spd[graph_edges], max_pairs=max_pairs
        )
        selected_edges.append(graph_edges[selection])
    if not selected_edges:
        return None
    edge_indices = torch.cat(selected_edges)
    relation_ids = _undirected_lga_relation_ids(data)
    selected_pairs = torch.unique(relation_ids[edge_indices])
    relation_mask = torch.isin(relation_ids, selected_pairs)
    # Pick the first selected edge for each undirected relation without a
    # Python loop.  The relation id is already graph-local, so this remains
    # safe when the same canonical pair occurs in different graphs.
    selected_ids = relation_ids[edge_indices]
    sorted_order = torch.argsort(selected_ids, stable=True)
    sorted_ids = selected_ids[sorted_order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    if sorted_ids.numel() > 1:
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    representatives = edge_indices[sorted_order[first]]
    return relation_mask, representatives


def _mips_lga_relation_losses(base_model, data, spd_head, path_head, max_pairs):
    """Compute SPD and path/bond losses from one relation-corrupted forward."""
    selected = _select_mips_relation_targets(data, max_pairs)
    if selected is None:
        zero = data.x.new_tensor(0.0)
        return zero, 0, zero, 0
    relation_mask, edge_indices = selected
    data.lga_relation_mask = relation_mask
    try:
        _, node_rep = base_model.encoders["graph"].encoder.forward_relation_pretext(data)
    finally:
        delattr(data, "lga_relation_mask")
    source, target = data.lga_edge_index[:, edge_indices]
    representation = torch.cat(
        [node_rep[source], node_rep[target],
         torch.abs(node_rep[source] - node_rep[target])], dim=-1
    )
    spd_targets = data.lga_spd[edge_indices].long()
    if spd_head is not None:
        spd_logits = spd_head(representation)
        spd_per_target = F.cross_entropy(
            spd_logits.float(), spd_targets, reduction="none"
        )
        spd_loss = _mean_loss_by_graph(
            spd_per_target, data.batch[target]
        )
    else:
        spd_loss = representation.sum() * 0.0

    hist = data.lga_path_bond_hist[edge_indices]
    position_mask = hist.sum(dim=-1) > 0
    if path_head is not None and bool(position_mask.any()):
        path_logits = path_head(
            representation.unsqueeze(1).expand(
                -1, hist.size(1), -1
            ).reshape(-1, representation.size(-1))
        ).reshape(hist.size(0), hist.size(1), -1)
        path_targets = hist.argmax(dim=-1).long()
        path_per_target = F.cross_entropy(
            path_logits[position_mask].float(),
            path_targets[position_mask], reduction="none",
        )
        path_graph_ids = data.batch[target].unsqueeze(1).expand_as(position_mask)
        path_loss = _mean_loss_by_graph(
            path_per_target, path_graph_ids[position_mask]
        )
        path_count = int(position_mask.sum().item())
    else:
        path_loss = representation.sum() * 0.0
        path_count = 0
    return spd_loss, int(spd_targets.numel()), path_loss, path_count


def _mips_lga_spd_loss(base_model, data, prediction_head, max_pairs):
    """Compatibility wrapper for callers that request SPD only."""
    spd_loss, spd_count, _, _ = _mips_lga_relation_losses(
        base_model, data, prediction_head,
        None,
        max_pairs,
    )
    return spd_loss, spd_count


def _mips_path_bond_loss(base_model, data, prediction_head, max_pairs):
    """Compatibility wrapper for callers that request path/bond only."""
    # The standalone legacy entry point is retained for diagnostics.  The
    # production Stage-1 path calls _mips_lga_relation_losses directly.
    _, _, path_loss, path_count = _mips_lga_relation_losses(
        base_model, data, None,
        prediction_head, max_pairs,
    )
    return path_loss, path_count


def _mips_trimer_distance_loss(base_model, data, prediction_head, max_pairs):
    """Regress selected central-RU pair distances with those edges hidden.

    Pair identity is local to each graph.  The selected undirected pair is
    removed in both directions from every conformer's radius graph during the
    same forward pass, so the target distance cannot be read as an input edge.
    """
    required = (
        "trimer_positions", "trimer_conformer_mask", "trimer_geometry_valid",
        "trimer_central_atom_index", "trimer_central_atom_mask",
        "canonical_ru_atom_local_index",
    )
    if any(not hasattr(data, name) for name in required):
        return data.x.new_tensor(0.0), 0
    graph_count = int(data.graph_available.numel())
    atom_capacity = int(data.trimer_positions.size(2))
    hidden_pair_mask = torch.zeros(
        graph_count, atom_capacity, atom_capacity,
        dtype=torch.bool, device=data.x.device,
    )
    selected = []
    for graph_id in range(graph_count):
        if (
            not bool(data.graph_available[graph_id])
            or not bool(data.trimer_geometry_valid[graph_id])
            or not bool(data.trimer_conformer_mask[graph_id, 0])
        ):
            continue
        central = data.trimer_central_atom_index[
            graph_id, data.trimer_central_atom_mask[graph_id]
        ].long()
        if central.numel() < 2:
            continue
        positions = data.trimer_positions[graph_id, 0, central]
        distances = torch.cdist(positions, positions)
        source, target = torch.triu_indices(
            central.numel(), central.numel(), offset=1, device=data.x.device
        )
        keep = distances[source, target] < 6.0
        source, target = source[keep], target[keep]
        if not source.numel():
            continue
        order = torch.argsort(distances[source, target], stable=True)
        order = order[:int(max_pairs)]
        source, target = source[order], target[order]
        for local_source, local_target in zip(source.tolist(), target.tolist()):
            atom_source = int(central[local_source])
            atom_target = int(central[local_target])
            hidden_pair_mask[graph_id, atom_source, atom_target] = True
            hidden_pair_mask[graph_id, atom_target, atom_source] = True
            selected.append((
                graph_id, local_source, local_target,
                distances[local_source, local_target],
            ))
    if not selected:
        return data.x.new_tensor(0.0), 0
    data.trimer_distance_pair_mask = hidden_pair_mask
    try:
        _, node_rep = _graph_encode_nodes(base_model, data)
    finally:
        delattr(data, "trimer_distance_pair_mask")
    pair_representations = []
    targets = []
    target_graphs = []
    local_canonical = data.canonical_ru_atom_local_index.long()
    for graph_id, source_id, target_id, distance in selected:
        graph_nodes = torch.nonzero(
            data.batch == graph_id, as_tuple=False
        ).flatten()
        source_nodes = graph_nodes[local_canonical[graph_nodes] == source_id]
        target_nodes = graph_nodes[local_canonical[graph_nodes] == target_id]
        if not source_nodes.numel() or not target_nodes.numel():
            continue
        source_rep = node_rep[source_nodes[0]]
        target_rep = node_rep[target_nodes[0]]
        pair_representations.append(torch.cat((
            source_rep, target_rep, torch.abs(source_rep - target_rep)
        )))
        targets.append(distance)
        target_graphs.append(graph_id)
    if not pair_representations:
        return data.x.new_tensor(0.0), 0
    prediction = prediction_head(
        torch.stack(pair_representations)
    ).squeeze(-1)
    target = torch.stack(targets).float()
    per_target = F.smooth_l1_loss(
        prediction.float(), target, beta=0.5, reduction="none"
    )
    return (
        _mean_loss_by_graph(
            per_target,
            torch.tensor(target_graphs, device=data.x.device, dtype=torch.long),
        ),
        int(target.numel()),
    )


def _multiclass_focal_loss(logits, targets, gamma=2.0):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    ce = F.cross_entropy(logits, targets.long(), reduction='none')
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def _scage_ecfp_loss(graph_rep, data, ecfp_head, pos_weight=None):
    valid = getattr(data, 'polymer_ecfp_valid', None)
    target = getattr(data, 'polymer_ecfp_target', None)
    if valid is None or target is None:
        return graph_rep.new_tensor(0.0), 0
    valid = valid.flatten().bool()
    if not valid.any():
        return graph_rep.new_tensor(0.0), 0
    logits = ecfp_head(graph_rep[valid])
    return F.binary_cross_entropy_with_logits(
        logits, target[valid].float(), pos_weight=pos_weight
    ), int(valid.sum().item())


def _stratified_pair_selection(distances, max_pairs):
    if distances.numel() <= int(max_pairs):
        return torch.arange(distances.numel(), device=distances.device)
    buckets = [
        distances == 1,
        distances == 2,
        (distances >= 3) & (distances <= 5),
        (distances >= 6) & (distances <= 10),
        distances > 10,
    ]
    quota = max(1, int(max_pairs) // len(buckets))
    selected = []
    for mask in buckets:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > quota:
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[:quota]]
        selected.append(indices)
    selected = torch.cat(selected) if selected else distances.new_empty(0, dtype=torch.long)
    remaining = int(max_pairs) - int(selected.numel())
    if remaining > 0:
        all_indices = torch.arange(distances.numel(), device=distances.device)
        available = all_indices[~torch.isin(all_indices, selected)]
        if available.numel() > remaining:
            available = available[torch.randperm(available.numel(), device=available.device)[:remaining]]
        selected = torch.cat([selected, available])
    return selected


def _scage_periodic_sp_loss(data, node_rep, sp_head, max_distance, max_pairs, gamma):
    reps, targets = [], []
    metadata_valid = getattr(data, 'star_link_metadata_valid', None)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        if metadata_valid is not None and not bool(metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        n = int(node_idx.numel())
        if n < 2:
            continue
        local = torch.full((data.batch.numel(),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        distance = torch.full((n, n), float(max_distance), device=data.x.device)
        diagonal = torch.arange(n, device=data.x.device)
        distance[diagonal, diagonal] = 0
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        ls, ld = local[src[keep]], local[dst[keep]]
        distance[ls, ld] = 1
        for k in range(n):
            distance = torch.minimum(distance, distance[:, k:k + 1] + distance[k:k + 1, :])
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        pair_targets = distance[row, col].clamp(max=float(max_distance)).long()
        selection = _stratified_pair_selection(pair_targets, max_pairs=max_pairs)
        row, col, pair_targets = row[selection], col[selection], pair_targets[selection]
        left, right = node_rep[node_idx[row]], node_rep[node_idx[col]]
        reps.append(torch.cat([left, right, torch.abs(left - right)], dim=-1))
        targets.append(pair_targets)
    if not reps:
        return node_rep.new_tensor(0.0), 0
    reps = torch.cat(reps, dim=0)
    targets = torch.cat(targets, dim=0)
    return _multiclass_focal_loss(sp_head(reps), targets, gamma=gamma), int(targets.numel())


def _angle_value(p0, center, p2):
    v1, v2 = p0 - center, p2 - center
    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0))


def _unsigned_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    denom = torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v, w) / denom, -1.0, 1.0))


def _signed_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    if float(torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)) <= 1e-8:
        return None
    return torch.atan2(torch.dot(torch.cross(b1u, v, dim=0), w), torch.dot(v, w))


def _angle_bin(value, bins):
    return torch.clamp((value / torch.pi * int(bins)).long(), max=int(bins) - 1)


def _circular_regression_loss(prediction, target_angles):
    """Regress a periodic angle through its unit-circle representation."""
    prediction = prediction.float()
    target_angles = target_angles.to(device=prediction.device, dtype=prediction.dtype)
    target = torch.stack([target_angles.sin(), target_angles.cos()], dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = prediction / prediction_norm
    direction_loss = 1.0 - (direction * target).sum(dim=-1)
    magnitude_loss = (prediction_norm.squeeze(-1) - 1.0).pow(2)
    return direction_loss.mean() + 0.05 * magnitude_loss.mean()


def _balanced_shift_cross_entropy(logits, targets, balance_power):
    """Balance signed periodic-image classes without unstable full inverse weighting."""
    balance_power = float(balance_power)
    if balance_power <= 0:
        return F.cross_entropy(logits, targets)
    class_count = int(logits.size(-1))
    counts = torch.bincount(targets, minlength=class_count).to(logits)
    present = counts > 0
    if int(present.sum().item()) <= 1:
        return F.cross_entropy(logits, targets)
    weights = torch.zeros_like(counts)
    present_counts = counts[present]
    weights[present] = (present_counts.mean() / present_counts).pow(balance_power)
    weights[present] = weights[present].clamp(0.25, 4.0)
    weights[present] /= weights[present].mean().clamp_min(1e-8)
    return F.cross_entropy(logits, targets, weight=weights)


def _scage_screw_geometry_loss(
    data,
    graph_rep,
    node_rep,
    angle_head,
    torsion_head,
    distance_head,
    shift_head,
    screw_head,
    angle_bins,
    torsion_bins,
    boundary_angle_weight,
    boundary_torsion_weight,
    torsion_objective,
    distance_component_weight,
    angle_component_weight,
    screw_component_weight,
    shift_balance_power,
    gamma,
    max_pairs=128,
    image_cap=1,
):
    internal_reps, internal_targets = [], []
    boundary_reps, boundary_targets = [], []
    torsion_reps, torsion_targets = [], []
    distance_reps, distance_targets, shift_targets = [], [], []
    screw_reps, screw_targets = [], []
    valid_samples = 0
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen):
            continue
        if not bool(data.star_link_metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != node_idx.numel() or (mapping < 0).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(node_idx.numel(), device=data.x.device)
        pair_global = data.attachment_pair[graph_idx].long()
        pair_local = global_to_local[pair_global]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = int(pair_local[0]), int(pair_local[1])
        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_global = data.ordered_backbone_path[ptr0:ptr1]
        path_local = global_to_local[path_global]
        if path_local.numel() < 2 or (path_local < 0).any():
            continue

        rotation = data.screw_rotation[graph_idx]
        translation = data.screw_translation[graph_idx]
        row, col = torch.triu_indices(pos.size(0), pos.size(0), offset=1, device=pos.device)
        pair_mask = getattr(data, 'scage_geometry_pair_mask', None)
        if pair_mask is not None:
            selected_mask = pair_mask[graph_idx, :pos.size(0), :pos.size(0)][row, col]
            row, col = row[selected_mask], col[selected_mask]
        same_distance = torch.linalg.vector_norm(pos[row] - pos[col], dim=-1)
        images = []
        shifts = []
        finite_images = None
        if has_screw:
            plus, minus = pos, pos
            for shift in range(1, int(image_cap) + 1):
                plus = plus @ rotation.transpose(0, 1) + translation
                minus = (minus - translation) @ rotation
                images.extend([plus, minus])
                shifts.extend([shift, -shift])
        elif has_polygen:
            active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
            if active_axes.numel() == 1:
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                for shift in range(1, int(image_cap) + 1):
                    images.extend([pos + shift * vector_t, pos - shift * vector_t])
                    shifts.extend([shift, -shift])
        elif hasattr(data, 'smer_image_pos3d'):
            finite_images = data.smer_image_pos3d[node_idx].permute(1, 0, 2)
            if finite_images.shape == (3, pos.size(0), 3) and torch.isfinite(finite_images).all():
                images = [finite_images[0], finite_images[2]]
                shifts = [-1, 1]
        if images and row.numel():
            cross = torch.stack([
                torch.linalg.vector_norm(pos[row] - image[col], dim=-1) for image in images
            ])
            nearest_cross = cross.min(dim=0).values
            all_distances = torch.cat([same_distance.unsqueeze(0), cross], dim=0)
            all_shifts = pos.new_tensor([0, *shifts], dtype=torch.long)
            nearest_all_idx = all_distances.min(dim=0).indices
            nearest_shift = all_shifts[nearest_all_idx]
            buckets = torch.clamp((nearest_cross / 2.0).floor().long() + 1, max=20)
            selection = _stratified_pair_selection(buckets, max_pairs=max_pairs)
            left_rep, right_rep = node_rep[node_idx[row[selection]]], node_rep[node_idx[col[selection]]]
            distance_reps.append(torch.cat([
                left_rep, right_rep, torch.abs(left_rep - right_rep), left_rep * right_rep
            ], dim=-1))
            distance_targets.append(torch.stack([
                same_distance[selection], nearest_cross[selection]
            ], dim=-1))
            shift_targets.append(nearest_shift[selection] + int(image_cap))

        if has_screw:
            trace = torch.trace(rotation)
            screw_angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
            axis = torch.stack([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ])
            axis_norm = axis.norm()
            if axis_norm < 1e-6:
                axis = translation / translation.norm().clamp_min(1e-8)
            else:
                axis = axis / axis_norm
            rise = torch.dot(axis, translation).abs()
            screw_reps.append(graph_rep[graph_idx])
            screw_targets.append(torch.stack([rise, screw_angle.sin(), screw_angle.cos()]))

        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(int(node_idx.numel()))]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()):
            if tuple(sorted((source, target))) in star_pair:
                continue
            neighbors[source].add(target)
        local_geometry_mask = None
        if pair_mask is not None:
            local_geometry_mask = pair_mask[
                graph_idx, :pos.size(0), :pos.size(0)
            ]

        def target_geometry_is_masked(atom_indices):
            if local_geometry_mask is None:
                return True
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    if not bool(local_geometry_mask[
                        atom_indices[first_idx], atom_indices[second_idx]
                    ].item()):
                        return False
            return True

        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    left, right = adjacent[first_idx], adjacent[second_idx]
                    if not target_geometry_is_masked((left, center, right)):
                        continue
                    angle = _angle_value(pos[left], pos[center], pos[right])
                    if angle is None:
                        continue
                    internal_reps.append(torch.cat([
                        node_rep[node_idx[left]], node_rep[node_idx[center]], node_rep[node_idx[right]]
                    ]))
                    internal_targets.append(_angle_bin(angle, angle_bins))

        if path_local.numel() >= 3:
            left_neighbor = int(path_local[1].item())
            right_neighbor = int(path_local[-2].item())
            if has_screw:
                previous_right = (pos[right_boundary] - translation) @ rotation
                next_left = pos[left_boundary] @ rotation.transpose(0, 1) + translation
                next_left_neighbor = pos[left_neighbor] @ rotation.transpose(0, 1) + translation
            elif has_polygen:
                active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
                if active_axes.numel() != 1:
                    continue
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                previous_right = pos[right_boundary] - vector_t
                next_left = pos[left_boundary] + vector_t
                next_left_neighbor = pos[left_neighbor] + vector_t
            elif finite_images is not None:
                previous_right = finite_images[0, right_boundary]
                next_left = finite_images[2, left_boundary]
                next_left_neighbor = finite_images[2, left_neighbor]
            else:
                continue
            left_angle = _angle_value(previous_right, pos[left_boundary], pos[left_neighbor])
            right_angle = _angle_value(pos[right_neighbor], pos[right_boundary], next_left)
            if left_angle is not None and target_geometry_is_masked(
                (right_boundary, left_boundary, left_neighbor)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_boundary]], node_rep[node_idx[left_boundary]],
                    node_rep[node_idx[left_neighbor]]
                ]))
                boundary_targets.append(_angle_bin(left_angle, angle_bins))
            if right_angle is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]]
                ]))
                boundary_targets.append(_angle_bin(right_angle, angle_bins))
            torsion = _signed_dihedral(
                pos[right_neighbor], pos[right_boundary], next_left, next_left_neighbor
            )
            if torsion is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary, left_neighbor)
            ):
                torsion_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]], node_rep[node_idx[left_neighbor]]
                ]))
                torsion_targets.append(torsion)
        valid_samples += 1

    geometry_components = {}
    if internal_targets:
        logits = angle_head(torch.stack(internal_reps))
        geometry_components['internal_angle'] = _multiclass_focal_loss(
            logits, torch.stack(internal_targets), gamma
        )
    if boundary_targets:
        logits = angle_head(torch.stack(boundary_reps))
        boundary_loss = _multiclass_focal_loss(logits, torch.stack(boundary_targets), gamma)
        if 'internal_angle' in geometry_components:
            geometry_components['angle'] = (
                geometry_components.pop('internal_angle') + float(boundary_angle_weight) * boundary_loss
            ) / (1.0 + float(boundary_angle_weight))
        else:
            geometry_components['angle'] = boundary_loss
    elif 'internal_angle' in geometry_components:
        geometry_components['angle'] = geometry_components.pop('internal_angle')
    if torsion_targets:
        logits = torsion_head(torch.stack(torsion_reps))
        torsion_values = torch.stack(torsion_targets)
        if torsion_objective == 'circular':
            geometry_components['torsion'] = _circular_regression_loss(
                logits, torsion_values
            )
        else:
            categorical_targets = torch.clamp(
                (((torsion_values + torch.pi) / (2.0 * torch.pi)) * int(torsion_bins)).long(),
                min=0, max=int(torsion_bins) - 1,
            )
            geometry_components['torsion'] = _multiclass_focal_loss(
                logits, categorical_targets, gamma
            )
    if distance_targets:
        pair_rep = torch.cat(distance_reps)
        distance_target = torch.cat(distance_targets)
        distance_loss = F.smooth_l1_loss(distance_head(pair_rep), distance_target, beta=0.5)
        shift_loss = _balanced_shift_cross_entropy(
            shift_head(pair_rep), torch.cat(shift_targets), shift_balance_power
        )
        geometry_components['distance'] = distance_loss + 0.2 * shift_loss
    if screw_targets:
        geometry_components['screw'] = F.smooth_l1_loss(
            screw_head(torch.stack(screw_reps)), torch.stack(screw_targets), beta=0.25
        )
    if not geometry_components:
        return node_rep.new_tensor(0.0), {
            'samples': 0, 'internal': 0, 'boundary': 0, 'torsion': 0,
            'distance_pairs': 0, 'shift_negative': 0, 'shift_center': 0,
            'shift_positive': 0, 'screw_ops': 0,
        }, {}
    component_priors = {
        'distance': float(distance_component_weight),
        'angle': float(angle_component_weight),
        'torsion': float(boundary_torsion_weight),
        'screw': float(screw_component_weight),
    }
    active_weight = sum(component_priors[name] for name in geometry_components)
    total = sum(
        component_priors[name] * value for name, value in geometry_components.items()
    ) / max(active_weight, 1e-8)
    signed_shift_targets = (
        torch.cat(shift_targets) - int(image_cap)
        if shift_targets else node_rep.new_empty(0, dtype=torch.long)
    )
    component_logs = {
        name: value.detach() for name, value in geometry_components.items()
    }
    if distance_targets:
        component_logs['distance_regression'] = distance_loss.detach()
        component_logs['image_shift'] = shift_loss.detach()
    return total, {
        'samples': valid_samples,
        'internal': len(internal_targets),
        'boundary': len(boundary_targets),
        'torsion': len(torsion_targets),
        'distance_pairs': sum(item.size(0) for item in distance_targets),
        'shift_negative': int((signed_shift_targets < 0).sum().item()),
        'shift_center': int((signed_shift_targets == 0).sum().item()),
        'shift_positive': int((signed_shift_targets > 0).sum().item()),
        'screw_ops': len(screw_targets),
    }, component_logs


def _make_scage_geometry_pair_mask(data, ratio):
    batch_size, max_nodes, _ = data.scage_spd.shape
    mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool, device=data.x.device)
    for graph_idx in range(batch_size):
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen) or float(ratio) <= 0:
            continue
        count = int((data.batch == graph_idx).sum().item())
        row, col = torch.triu_indices(count, count, offset=1, device=data.x.device)
        target_count = min(
            int(max(1, round(float(ratio) * row.numel()))),
            int(row.numel()),
        )
        if target_count <= 0:
            continue
        choice = torch.randperm(row.numel(), device=data.x.device)[:target_count]
        selected_row, selected_col = row[choice], col[choice]
        mask[graph_idx, selected_row, selected_col] = True
        mask[graph_idx, selected_col, selected_row] = True

        def mask_clique(atom_indices):
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    first = atom_indices[first_idx]
                    second = atom_indices[second_idx]
                    mask[graph_idx, first, second] = True
                    mask[graph_idx, second, first] = True

        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(count, device=data.x.device)
        pair_local = global_to_local[data.attachment_pair[graph_idx].long()]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = pair_local.tolist()
        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(count)]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(
            global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()
        ):
            if tuple(sorted((source, target))) not in star_pair:
                neighbors[source].add(target)
        angle_candidates = []
        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    angle_candidates.append((adjacent[first_idx], center, adjacent[second_idx]))
        if angle_candidates:
            angle_count = min(
                max(1, int(round(float(ratio) * len(angle_candidates)))),
                len(angle_candidates),
            )
            selected = torch.randperm(len(angle_candidates), device=data.x.device)[:angle_count]
            for candidate_idx in selected.tolist():
                mask_clique(angle_candidates[candidate_idx])

        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_local = global_to_local[data.ordered_backbone_path[ptr0:ptr1]]
        if path_local.numel() >= 3 and not (path_local < 0).any():
            mask_clique((
                int(path_local[-2]), right_boundary,
                left_boundary, int(path_local[1]),
            ))
    return mask


def _multi_positive_supcon_loss(embeddings, identities, temperature):
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.t() / float(temperature)
    identity_mask = identities[:, None] == identities[None, :]
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = identity_mask & ~self_mask
    logits = logits.masked_fill(self_mask, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count.clamp_min(1))
    return per_anchor[valid].mean()


def _repeat_cut_identity_key(value):
    """Normalize cached and online cut identities to stable hashable keys."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_repeat_cut_identity_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _repeat_cut_identity_key(item))
                for key, item in value.items()
            )
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _scage_repeat_cut_consistency_loss(
    base_model, data, projection_head, graph_input, temperature, max_mrus, retry, views_per_polymer
):
    from src.dataset.graph_data import repeat_cut_augment_smiles
    from src.dataset.dataloader import custom_collate

    stats = {
        'identities_attempted': len(data.smiles), 'views_requested': len(data.smiles) * int(views_per_polymer),
        'views_valid': 0, 'duplicate_views': 0, 'failed_views': 0,
        'distinct_cut_views': 0, 'valid_identities': 0, 'positive_pairs': 0,
        'positive_count_sum': 0, 'graph_cache_hits': 0, 'graph_cache_misses': 0,
    }
    graph_encoder = base_model.encoders['graph'].encoder
    original_rep, _ = graph_encoder(data)
    augmented_graphs, augmented_ids, valid_original_ids = [], [], []
    for sample_idx, smiles in enumerate(data.smiles):
        unique_views = set()
        unique_cuts = set()
        cached_views = (
            list(data.repeat_cut_smiles[sample_idx])
            if hasattr(data, 'repeat_cut_smiles') else []
        )
        cached_cuts = (
            list(data.repeat_cut_identities[sample_idx])
            if hasattr(data, 'repeat_cut_identities') else []
        )
        for view_idx in range(int(views_per_polymer)):
            accepted = None
            # Feature-cache views are generated with max_mrus=3. Reusing them
            # under a different runtime cap silently violates the requested
            # augmentation policy and can reintroduce oversized 3-MRU graphs.
            if int(max_mrus) == 3 and view_idx < len(cached_views):
                try:
                    cached_cut = _repeat_cut_identity_key(
                        cached_cuts[view_idx] if view_idx < len(cached_cuts) else None
                    )
                    cached_graph, cache_hit = _cached_repeat_cut_graph(
                        cached_views[view_idx], graph_input,
                        mips_core=graph_encoder.core,
                        mips_max_hops=graph_encoder.max_hops,
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(cached_views[view_idx]), cached_graph, cached_cut)
                except Exception:
                    accepted = None
            for _attempt in range(max(1, int(retry))):
                if accepted is not None:
                    break
                try:
                    aug_smiles, _, metadata = repeat_cut_augment_smiles(
                        smiles, max_mrus=max_mrus, return_n=True, return_metadata=True
                    )
                    cut_identity = _repeat_cut_identity_key(
                        metadata.get('cut_identity')
                    )
                    if (
                        str(aug_smiles) == str(smiles)
                        or str(aug_smiles) in unique_views
                        or (cut_identity is not None and cut_identity in unique_cuts)
                    ):
                        stats['duplicate_views'] += 1
                        continue
                    augmented_graph, cache_hit = _cached_repeat_cut_graph(
                        aug_smiles, graph_input,
                        mips_core=graph_encoder.core,
                        mips_max_hops=graph_encoder.max_hops,
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(aug_smiles), augmented_graph, cut_identity)
                    break
                except Exception:
                    accepted = None
            if accepted is None:
                stats['failed_views'] += 1
                continue
            unique_views.add(accepted[0])
            if accepted[2] is not None:
                unique_cuts.add(accepted[2])
                stats['distinct_cut_views'] += 1
            augmented = accepted[1]
            augmented.smiles = accepted[0]
            augmented.input_ids_smiles = data.input_ids_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.attention_mask_smiles = data.attention_mask_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.fp = data.fp[sample_idx:sample_idx + 1].detach().cpu()
            augmented.y = data.y[sample_idx].detach().cpu().reshape(-1)
            augmented.z = torch.ones(1, dtype=torch.long)
            augmented.pos = torch.zeros(1, 3)
            augmented.pos_confs = torch.zeros(1, 1, 3)
            augmented.graph_to_geom_index = torch.full(
                (augmented.x.size(0),), -1, dtype=torch.long
            )
            augmented.geom_build_ok = False
            augmented.geom_coordinate_ok = False
            augmented.geom_context_id = 1
            augmented.screw_valid = False
            augmented.smer_valid = False
            augmented.polymer_ecfp_target = torch.zeros(2048)
            augmented.polymer_ecfp_valid = False
            augmented.polymer_ecfp_source = 'repeat_cut_view'
            descriptor_valid = bool(data.scage_descriptor_valid[sample_idx].item())
            augmented.scage_descriptor_valid_confs = torch.tensor([descriptor_valid])
            for descriptor_name in ('shape', 'usrcat', 'autocorr3d', 'rdf', 'morse', 'whim'):
                selected = getattr(data, f'scage_descriptor_{descriptor_name}')[sample_idx]
                setattr(
                    augmented,
                    f'scage_descriptor_{descriptor_name}_confs',
                    selected.detach().cpu().unsqueeze(0),
                )
            augmented_graphs.append(augmented)
            augmented_ids.append(sample_idx)
            stats['views_valid'] += 1
        if unique_views:
            valid_original_ids.append(sample_idx)
    if len(valid_original_ids) < 2 or not augmented_graphs:
        return original_rep.new_tensor(0.0), stats
    aug_batch = custom_collate(augmented_graphs, random_conformer=False).to(data.x.device)
    augmented_rep, _ = graph_encoder(aug_batch)
    representations, identities = [], []
    for sample_idx in valid_original_ids:
        representations.append(original_rep[sample_idx])
        identities.append(sample_idx)
    representations.extend(list(augmented_rep))
    identities.extend(augmented_ids)
    identities = torch.tensor(identities, dtype=torch.long, device=data.x.device)
    projected = projection_head(torch.stack(representations))
    stats['valid_identities'] = len(valid_original_ids)
    counts = torch.bincount(identities, minlength=len(data.smiles))
    stats['positive_pairs'] = int((counts * (counts - 1)).sum().item())
    stats['positive_count_sum'] = int((counts[counts > 0] - 1).sum().item())
    return _multi_positive_supcon_loss(projected, identities, temperature), stats


def _symmetric_infonce(left, right, temperature, valid_mask=None):
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=left.device, dtype=torch.bool).view(-1)
        left = left[valid_mask]
        right = right[valid_mask]
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    if _distributed_enabled():
        from torch.distributed.nn.functional import all_gather

        local_count = torch.tensor([left.size(0)], device=left.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        global_count = sum(counts)
        if global_count < 2:
            return (left.sum() + right.sum()) * 0.0
        max_count = max(counts)
        if left.size(0) < max_count:
            padding = left.new_zeros((max_count - left.size(0), left.size(1)))
            left_padded = torch.cat([left, padding], dim=0)
            right_padded = torch.cat([right, padding], dim=0)
        else:
            left_padded, right_padded = left, right
        gathered_left_parts = list(all_gather(left_padded))
        gathered_right_parts = list(all_gather(right_padded))
        gathered_left = torch.cat([
            part[:count] for part, count in zip(gathered_left_parts, counts)
        ], dim=0)
        gathered_right = torch.cat([
            part[:count] for part, count in zip(gathered_right_parts, counts)
        ], dim=0)
        if left.size(0) == 0:
            return (gathered_left.sum() + gathered_right.sum()) * 0.0
        offset = sum(counts[:dist.get_rank()])
        labels = torch.arange(left.size(0), device=left.device) + offset
        left_logits = left @ gathered_right.t() / float(temperature)
        right_logits = right @ gathered_left.t() / float(temperature)
        local_loss = 0.5 * (
            F.cross_entropy(left_logits.float(), labels)
            + F.cross_entropy(right_logits.float(), labels)
        )
        # DDP averages gradients over ranks. Rescale the local mean so the
        # resulting gradient equals a global per-sample mean for unequal counts.
        return local_loss * (dist.get_world_size() * left.size(0) / global_count)
    if left.size(0) < 2:
        return (left.sum() + right.sum()) * 0.0
    logits = left @ right.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits.float(), labels) + F.cross_entropy(logits.t().float(), labels)
    )


def _scage_semantic_alignment_losses(base_model, data, args):
    if 'graph' not in base_model.encoders or len(base_model.encoders) < 2:
        raise ValueError("Parallel semantic alignment requires Graph plus another modality")
    embeddings = base_model.encode_modalities(data)
    shared_embeddings = base_model.shared_modality_embeddings
    private_embeddings = base_model.private_modality_embeddings
    modality_index = {name: idx for idx, name in enumerate(base_model.modality_list)}
    graph = shared_embeddings[:, modality_index['graph']]
    projected_graph = base_model.project_alignment('graph', graph)
    projected_smiles = (
        base_model.project_alignment(
            'smiles', shared_embeddings[:, modality_index['smiles']]
        ) if 'smiles' in modality_index else None
    )
    projected_fp = (
        base_model.project_alignment(
            'fp', shared_embeddings[:, modality_index['fp']]
        ) if 'fp' in modality_index else None
    )

    drop_probabilities = {
        'fp': args.alignment_fp_drop,
        'smiles': args.alignment_smiles_drop,
        'graph': args.alignment_graph_drop,
    }
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    masks = [
        base_model.sample_availability_mask(
            embeddings.size(0), drop_probabilities,
            min_available=min(2, len(base_model.modality_list)),
            device=embeddings.device, intrinsic_mask=intrinsic_mask,
        )
        for _ in range(2)
    ]
    fused_views = []
    for mask in masks:
        fused, _ = base_model.fuse_embeddings(embeddings, availability_mask=mask)
        fused_views.append(base_model.project_alignment('fusion', fused))
    fused_view_loss = _symmetric_infonce(fused_views[0], fused_views[1], args.temperature)

    full_mask = intrinsic_mask
    full_fused, full_weights = base_model.fuse_embeddings(embeddings, availability_mask=full_mask)
    teacher = F.normalize(base_model.project_alignment('fusion', full_fused), dim=-1).detach()
    lomo_losses = []
    for missing_name in base_model.modality_list:
        present = full_mask[:, modality_index[missing_name]]
        if not bool(present.any()):
            continue
        lomo_mask = full_mask.clone()
        lomo_mask[:, modality_index[missing_name]] = False
        student, _ = base_model.fuse_embeddings(embeddings, availability_mask=lomo_mask)
        student = F.normalize(base_model.project_alignment('fusion', student), dim=-1)
        lomo_losses.append(1.0 - (student[present] * teacher[present]).sum(dim=-1).mean())
    lomo_loss = (
        torch.stack(lomo_losses).mean()
        if lomo_losses else embeddings.sum() * 0.0
    )

    prior_by_name = {'smiles': 0.30, 'graph': 0.40, 'fp': 0.30}
    prior = full_weights.new_tensor([prior_by_name[name] for name in base_model.modality_list])
    sample_prior = prior.unsqueeze(0) * full_mask.to(dtype=full_weights.dtype)
    sample_prior = sample_prior / sample_prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    safe_weights = full_weights.clamp_min(1e-8)
    pooling_kl = (
        safe_weights
        * (safe_weights.log() - sample_prior.clamp_min(1e-8).log())
        * full_mask
    ).sum(dim=1).mean()
    mean_weights = full_weights.mean(dim=0)
    graph_valid = full_mask[:, modality_index['graph']]
    shared_private_orthogonal = torch.stack([
        F.cosine_similarity(
            shared_embeddings[:, idx],
            private_embeddings[:, idx],
            dim=-1,
        ).square().mean()
        for idx in range(shared_embeddings.size(1))
    ]).mean()
    losses = {
        'fused_view': fused_view_loss,
        'graph_smiles': (
            _symmetric_infonce(
                projected_graph, projected_smiles, args.temperature,
                valid_mask=graph_valid,
            ) if projected_smiles is not None else embeddings.sum() * 0.0
        ),
        'graph_fp': (
            _symmetric_infonce(
                projected_graph, projected_fp, args.temperature,
                valid_mask=graph_valid,
            ) if projected_fp is not None else embeddings.sum() * 0.0
        ),
        'lomo': lomo_loss,
        'pooling_kl': pooling_kl,
        'shared_private': shared_private_orthogonal,
    }
    missing_rates = {
        name: float(torch.stack([~mask[:, idx] for mask in masks]).float().mean().item())
        for idx, name in enumerate(base_model.modality_list)
    }
    stats = {
        'missing_rates': missing_rates,
        'pooling_entropy': float(
            (-(full_weights.clamp_min(1e-8) * full_weights.clamp_min(1e-8).log()).sum(dim=1).mean()).item()
        ),
        'pooling_weights': {
            name: float(mean_weights[idx].item())
            for idx, name in enumerate(base_model.modality_list)
        },
    }
    return losses, stats


def _joint_canonical_mask(data, seed, stream_step, mask_ratio):
    """Vectorized stateless mask over canonical atoms.

    Keys depend only on sample identity, local canonical atom id, optimizer
    stream step and seed, so rank assignment and batch composition cannot
    change the selected atoms.
    """
    device = data.mips_x.device
    canonical_periodic = bool(
        getattr(data, "mts_canonical_periodic", False)
        or getattr(data, "mips_local_lga_schema_version", 0) == 2
    )
    if canonical_periodic:
        # One mask decision per canonical node.  The Trimer collator lifts
        # these states to all three RU copies downstream.
        canonical = torch.arange(
            int(data.mips_x.size(0)), device=data.mips_x.device,
            dtype=torch.long,
        )
    else:
        canonical = data.canonical_ru_atom_index.long()
    graph_available = torch.as_tensor(
        getattr(data, "graph_available", torch.ones(len(data.smiles))),
        device=device,
    ).bool().flatten()
    required = (
        "canonical_graph_index", "canonical_local_index",
        "canonical_first_node_index", "mts_sample_hash64",
    )
    if not all(hasattr(data, name) for name in required):
        raise ValueError(
            "MTS joint pretraining requires vectorized canonical metadata "
            "from mips_trimer_collate"
        )
    graph_ids = data.canonical_graph_index.long()
    local_ids = data.canonical_local_index.long()
    sample_hash = data.mts_sample_hash64.long()[graph_ids]
    values = sample_hash.clone()
    values ^= local_ids * 6364136223846793005
    values ^= (int(stream_step) * 1442695040888963407) & ((1 << 63) - 1)
    values ^= (int(seed) * 2862933555777941757) & ((1 << 63) - 1)
    values ^= values >> 30
    values *= 3935559000370003845
    values ^= values >> 27
    values *= 2691343689449507681
    values ^= values >> 31
    scores = (values & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    canonical_selected = (scores < float(mask_ratio)) & graph_available[graph_ids]

    selected_counts = torch.zeros(
        graph_available.numel(), dtype=torch.long, device=device
    )
    selected_counts.index_add_(0, graph_ids, canonical_selected.long())
    minimum = torch.full(
        (graph_available.numel(),), float("inf"), dtype=scores.dtype, device=device
    )
    minimum.scatter_reduce_(0, graph_ids, scores, reduce="amin", include_self=True)
    empty = graph_available & (selected_counts == 0)
    canonical_selected |= empty[graph_ids] & (scores == minimum[graph_ids])
    return canonical_selected[canonical]


def _joint_masked_atom_terms(data, node_rep, prediction_head, atom_mask):
    if not hasattr(data, "canonical_first_node_index"):
        raise ValueError("Missing canonical_first_node_index in MTS batch")
    representatives = data.canonical_first_node_index.long()
    # Canonical topology has one node per atom, so representatives are the
    # nodes themselves.  Explicit test-reference batches still use the first
    # copy mapping supplied by the legacy collator.
    if atom_mask.numel() == node_rep.size(0) and representatives.numel() == node_rep.size(0):
        target_indices = torch.nonzero(atom_mask, as_tuple=False).flatten()
    else:
        target_indices = representatives[atom_mask[representatives]]
    if target_indices.numel() == 0:
        zero = node_rep.sum() * 0.0
        return zero, zero.detach(), 0, 0
    targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[target_indices])
    per_target = F.cross_entropy(logits.float(), targets, reduction="none")
    correct = int((logits.detach().argmax(dim=-1) == targets).sum().item())
    return (
        per_target.sum(), per_target.detach().sum(), int(per_target.numel()), correct
    )


def _multiclass_focal_terms(logits, targets, alpha, gamma=2.0):
    log_prob = F.log_softmax(logits.float(), dim=-1)
    selected_log_prob = log_prob.gather(1, targets.long().view(-1, 1)).squeeze(1)
    probability = selected_log_prob.exp()
    weights = alpha.to(logits.device, dtype=logits.dtype)[targets.long()]
    return -weights * (1.0 - probability).pow(float(gamma)) * selected_log_prob


class TrimerAngleHead(nn.Module):
    """SCAGE-style 20-bin angle head with DDP-safe LayerNorm."""

    def __init__(self, dim=512, hidden=256, bins=20, alpha=None, dropout=0.10,
                 objective='categorical'):
        super().__init__()
        self.objective = str(objective)
        output_dim = int(bins) if self.objective == 'categorical' else 1
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), output_dim),
        )
        if alpha is None:
            alpha = torch.ones(int(bins), dtype=torch.float)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float))
        self.bins = int(bins)

    def forward(self, values):
        return self.net(values)


def _angle_alpha_from_dataset(dataset, bins=20):
    """Build the fixed SCAGE-style inverse-frequency focal weights."""
    counts = getattr(dataset, "angle_class_counts", None)
    if counts is None:
        store = getattr(dataset, "_lazy_feature_store", None)
        counts = getattr(store, "angle_class_counts", None)
    if counts is None:
        raise RuntimeError(
            "MTS joint pretraining requires the frozen bond-angle cache "
            "with angle_class_counts.npy"
        )
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.shape != (int(bins),):
        raise RuntimeError(
            f"angle class histogram must have shape [{int(bins)}], got {counts.shape}"
        )
    raw = np.log(200000.0 / (counts.astype(np.float64) + 1.0) + 1.0)
    raw = raw / max(float(raw.mean()), 1e-12)
    return torch.as_tensor(raw, dtype=torch.float32)


def _joint_angle_terms(data, trimer_states, angle_head, gamma=2.0):
    if not hasattr(data, "trimer_angle_index"):
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    indices = data.trimer_angle_index.long()
    objective = getattr(angle_head, 'objective', 'categorical')
    bins = getattr(data, 'trimer_angle_bins', None)
    cosine_targets = getattr(data, 'trimer_angle_cos', None)
    ptr = getattr(data, "trimer_angle_ptr", None)
    valid_graph = getattr(data, "mcl_valid", None)
    if ptr is None or indices.numel() == 0:
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    graph_total = int(ptr.numel()) - 1
    counts = (ptr[1:] - ptr[:-1]).long()
    graph_ids = torch.repeat_interleave(
        torch.arange(graph_total, device=indices.device), counts
    )
    selected = torch.ones_like(graph_ids, dtype=torch.bool)
    if valid_graph is not None:
        selected &= torch.as_tensor(
            valid_graph, device=indices.device, dtype=torch.bool
        )[graph_ids]
    triplets = indices[selected]
    if objective == 'categorical':
        if bins is None or bins.numel() != indices.size(0):
            raise RuntimeError('categorical angle objective requires angle bins')
        targets = bins.long()[selected]
    else:
        if cosine_targets is None or cosine_targets.numel() != indices.size(0):
            raise RuntimeError('cosine angle objective requires Angle-v2 targets')
        targets = cosine_targets.float()[selected]
    selected_graphs = graph_ids[selected]
    if triplets.numel() == 0:
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    if bool((triplets < 0).any()) or bool(
        (triplets >= trimer_states.size(0)).any()
    ):
        raise ValueError("Trimer angle index is outside the collated Trimer range")
    representation = (
        trimer_states[triplets[:, 0]]
        + trimer_states[triplets[:, 1]]
        + trimer_states[triplets[:, 2]]
    )
    logits = angle_head(representation)
    if objective == 'categorical':
        per_angle = _multiclass_focal_terms(
            logits, targets, angle_head.alpha, gamma=gamma
        )
        correct_count = int(
            (logits.detach().argmax(dim=-1) == targets).sum().item()
        )
        per_angle_mae = per_angle.detach()
    else:
        predictions = logits.float().reshape(-1).clamp(-1.0, 1.0)
        per_angle = F.smooth_l1_loss(
            predictions, targets.float(), reduction='none'
        )
        # Retain the existing packed-count/logging interface: for continuous
        # targets, "accuracy" means |cos prediction error| <= 0.1.
        correct_count = int(
            ((predictions.detach() - targets).abs() <= 0.1).sum().item()
        )
        per_angle_mae = (predictions.detach() - targets).abs()
    graph_loss_sum = per_angle.new_zeros(graph_total)
    graph_loss_count = per_angle.new_zeros(graph_total)
    graph_loss_sum.index_add_(0, selected_graphs, per_angle)
    graph_loss_count.index_add_(
        0, selected_graphs, torch.ones_like(per_angle)
    )
    active = graph_loss_count > 0
    graph_means = graph_loss_sum[active] / graph_loss_count[active]
    graph_mae_sum = per_angle_mae.new_zeros(graph_total)
    graph_mae_sum.index_add_(0, selected_graphs, per_angle_mae)
    graph_mae_means = graph_mae_sum[active] / graph_loss_count[active]
    return (
        graph_means.sum(), graph_means.detach().sum(),
        int(active.sum().item()), correct_count, int(targets.numel()),
        graph_mae_means.sum(),
    )


def _bf16_joint_parity_gate(base_model, data, atom_head, angle_head, args):
    """Compare the complete one-forward joint objective in FP32/BF16."""
    mask = _joint_canonical_mask(data, args.seed, 0, args.graph_mask_ratio)

    def evaluate(dtype=None):
        base_model.zero_grad(set_to_none=True)
        atom_head.zero_grad(set_to_none=True)
        angle_head.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if dtype is not None else nullcontext()
        )
        with context:
            _, nodes, aux = base_model.encoders["graph"].encoder.forward_joint_pretrain(
                data, mask
            )
            atom_sum, _, atom_count, _ = _joint_masked_atom_terms(
                data, nodes, atom_head, mask
            )
            if str(getattr(args, "pretraining_objective", "joint")) == "masked_atom_only":
                angle_sum = atom_sum.new_zeros(())
                angle_count = 0
            else:
                angle_sum, _, angle_count, _, _, _ = _joint_angle_terms(
                    data, aux["final_trimer_states"], angle_head,
                    gamma=float(args.scage_focal_gamma),
                )
            atom_loss = atom_sum / max(1, atom_count)
            angle_loss = angle_sum / max(1, angle_count)
            loss = atom_loss + float(args.graph_angle_weight) * angle_loss
        loss.backward()
        grad = _gradient_vector((base_model, atom_head, angle_head))
        return loss.detach(), grad.detach()

    fp32_loss, fp32_grad = evaluate(None)
    bf16_loss, bf16_grad = evaluate(torch.bfloat16)
    delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    base_model.zero_grad(set_to_none=True)
    atom_head.zero_grad(set_to_none=True)
    angle_head.zero_grad(set_to_none=True)
    return finite and delta <= 0.02 and cosine >= 0.98, {
        "relative_loss_delta": delta, "gradient_cosine": cosine, "finite": finite,
    }


@torch.no_grad()
def _evaluate_angle_v2_validation(train_module, loader, args, device):
    """Evaluate the deterministic 1% holdout on every DDP rank."""
    was_training = train_module.training
    train_module.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for data in loader:
        data = data.to(device)
        with torch.autocast(
            device_type='cuda', dtype=torch.bfloat16,
            enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
        ):
            payload = train_module(MTS_STAGE1_ID, data, args, 0)
        totals[0] += payload['detached_terms']['masked_atom_sum'].double()
        totals[1] += float(payload['counts']['masked_atoms'])
        totals[2] += payload['detached_terms']['angle_mae_sum'].double()
        totals[3] += float(payload['counts']['angle_graphs'])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        train_module.train()
    atom_ce = float((totals[0] / totals[1].clamp_min(1.0)).item())
    angle_mae = float((totals[2] / totals[3].clamp_min(1.0)).item())
    if not np.isfinite(atom_ce) or not np.isfinite(angle_mae):
        raise RuntimeError('non-finite Angle-v2 validation metric')
    return atom_ce, angle_mae


def _fusion_conditioned_masked_atom_loss(base_model, data, prediction_head, mask_ratio):
    mask_indices = []
    graph_available = getattr(data, 'graph_available', None)
    if graph_available is None:
        graph_available = torch.ones(
            len(data.smiles), dtype=torch.bool, device=data.x.device
        )
    else:
        graph_available = torch.as_tensor(
            graph_available, device=data.x.device
        ).view(-1).bool()
    for graph_idx in range(len(data.smiles)):
        if not bool(graph_available[graph_idx]):
            continue
        nodes = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not nodes.numel():
            continue
        count = max(1, int(round(float(mask_ratio) * nodes.numel())))
        mask_indices.append(nodes[torch.randperm(nodes.numel(), device=nodes.device)[:count]])
    if not mask_indices:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(mask_indices)
    masked_x = data.x.clone()
    masked_x[mask_indices] = 0.0
    graph_module = base_model.encoders['graph']
    raw_graph, raw_nodes = graph_module.encoder.forward_with_x(data, masked_x)
    masked_graph = graph_module.projection(graph_module.norm(raw_graph))
    modality_embeddings = []
    for name in base_model.modality_list:
        value = masked_graph if name == 'graph' else base_model.encoders[name](data)
        if base_model.projection_mode == "shared_private":
            value, _, _ = base_model.shared_private_projections[name](value)
        modality_embeddings.append(value)
    embeddings = torch.stack(modality_embeddings, dim=1)
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    fused, _ = base_model.fuse_embeddings(
        embeddings, availability_mask=intrinsic_mask
    )
    context = fused[data.batch[mask_indices]]
    logits = prediction_head(torch.cat([raw_nodes[mask_indices], context], dim=-1))
    targets = data.atomic_num[mask_indices].long()
    return F.cross_entropy(logits.float(), targets), int(mask_indices.numel())


class MIPSPretrainContainer(nn.Module):
    """One DDP-visible module containing the encoder and every Stage 1 head.

    Calling encoder submodules behind DDP bypasses reducer preparation.  This
    container keeps the existing loss helpers while ensuring every trainable
    tensor is reached from one real DDP forward.
    """

    def __init__(self, model, heads=None):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleDict(dict(heads or {}))

    def forward(self, stage, data, args, epoch=0):
        if stage == "mts_joint_pretraining":
            return self._mts_joint_pretraining(data, args, epoch)
        if stage in {"mips_stage1", "stage2_geometry_adapt"}:
            return self._mips_stage1(
                data, args, epoch,
                geometry_adapt=stage == "stage2_geometry_adapt"
            )
        if stage == "alignment":
            losses, stats = _scage_semantic_alignment_losses(
                self.model, data, args
            )
            fused_mask, fused_count = _fusion_conditioned_masked_atom_loss(
                self.model, data, self.model.alignment_mask_head,
                args.graph_mask_ratio,
            )
            losses["fused_mask"] = fused_mask
            return {
                "losses": losses,
                "stats": stats,
                "fused_mask_count": fused_count,
            }
        raise ValueError(f"unsupported MIPS DDP stage: {stage}")

    def _mts_joint_pretraining(self, data, args, stream_step):
        """One shared O8+Trimer-MCL forward for both MTS pretext tasks."""
        mask = _joint_canonical_mask(
            data, args.seed, stream_step, args.graph_mask_ratio
        )
        graph, node_states, aux = self.model.encoders["graph"].encoder.forward_joint_pretrain(
            data, canonical_atom_mask=mask
        )
        atom_sum, atom_detached, atom_count, atom_correct = _joint_masked_atom_terms(
            data, node_states, self.heads["mips_atom"], mask
        )
        if str(getattr(args, "pretraining_objective", "joint")) == "masked_atom_only":
            # G-family input geometry is deliberately not reused as an
            # Angle-20 target.  Keep the angle head in the strict checkpoint
            # layout and zero-anchor it for DDP, but never inspect angle
            # labels or invoke the angle loss helper in this objective.
            angle_sum = atom_sum.new_zeros(())
            angle_detached = angle_sum.detach()
            angle_graph_count = 0
            angle_correct = 0
            angle_targets = 0
            angle_mae_sum = angle_detached
        else:
            angle_sum, angle_detached, angle_graph_count, angle_correct, angle_targets, angle_mae_sum = _joint_angle_terms(
                data,
                aux["final_trimer_states"],
                self.heads["angle"],
                gamma=float(args.scage_focal_gamma),
            )
        # The zero anchors are only for the rare rank-local empty target case;
        # they do not alter any numerical loss value or gradient of active
        # parameters.
        zero_reference = atom_sum * 0.0 + angle_sum * 0.0
        for head_name in ("mips_atom", "angle"):
            for parameter in self.heads[head_name].parameters():
                zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0
        return {
            "graph": graph,
            "loss_terms": {
                "masked_atom_sum": atom_sum,
                "angle_sum": angle_sum,
            },
            "detached_terms": {
                "masked_atom_sum": atom_detached,
                "angle_sum": angle_detached,
                "angle_mae_sum": angle_mae_sum.detach(),
            },
            "counts": {
            "masked_atoms": int(atom_count),
            "angle_graphs": int(angle_graph_count),
            "masked_correct": int(atom_correct),
            "angle_correct": int(angle_correct),
            "angle_targets": int(angle_targets),
            "mcl_valid_graphs": int(aux["mcl_valid_graph_mask"].sum().item()),
            "graphs": int(data.graph_available.numel()),
            },
            "mcl_valid": aux["mcl_valid_graph_mask"],
            "angle_valid": aux["angle_valid_graph_mask"],
            "zero_reference": zero_reference,
        }

    def _mips_stage1(self, data, args, epoch, geometry_adapt=False):
        zero = data.x.new_tensor(0.0)
        mask_loss, mask_count = _mips_masked_atom_loss(
            self.model, data, self.heads["mips_atom"],
            mask_ratio=args.graph_mask_ratio, seed=args.seed, epoch=epoch,
            geometry_adapt=geometry_adapt,
        )
        loss_terms = {}
        counts = {}
        if mask_count and float(args.scage_mips_mask_weight) > 0:
            loss_terms["mips_mask"] = mask_loss
            counts["masked_atoms"] = mask_count

        sp_loss, sp_count = zero, 0
        path_loss, path_count = zero, 0
        if not geometry_adapt and (
            float(args.mips_spd_weight) > 0
            or float(args.mips_path_bond_weight) > 0
        ):
            # SPD and path/bond share one relation-corrupted O8 forward.  The
            # atom task remains a separate masked-atom view so relation labels
            # cannot alter its semantics.
            sp_loss, sp_count, path_loss, path_count = _mips_lga_relation_losses(
                self.model, data,
                self.heads["masked_spd"] if float(args.mips_spd_weight) > 0 else None,
                self.heads["path_bond"] if float(args.mips_path_bond_weight) > 0 else None,
                max_pairs=args.scage_sp_max_pairs,
            )
        if sp_count:
            loss_terms["masked_spd"] = sp_loss
            counts["sp_pairs"] = sp_count
        if path_count:
            loss_terms["path_bond"] = path_loss
            counts["path_bond_pairs"] = path_count

        # Stage 2 intentionally does not train the SPD/path heads.  Keep only
        # the small head collection in the autograd graph with an exact zero
        # so a fixed ``find_unused_parameters=False`` reducer is safe.  The
        # complete UniEncoderAttention object is frozen/configured by the
        # caller; anchoring every unrelated Trimer/MD/fusion parameter here
        # was a substantial per-microbatch overhead.
        zero_reference = mask_loss * 0.0
        for parameter in self.heads.parameters():
            if parameter.requires_grad:
                zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0
        if geometry_adapt:
            for name in ("masked_spd", "path_bond"):
                for parameter in self.heads[name].parameters():
                    zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0

        return {
            "loss_terms": loss_terms,
            "counts": counts,
            "mask_loss": mask_loss,
            "spd_loss": sp_loss,
            "path_loss": path_loss,
            # Keeps the graph representation in the DDP output tree even when
            # a batch has no valid auxiliary target.
            "zero_reference": zero_reference,
        }


def _alignment_total(losses, args):
    return (
        args.alignment_fused_weight * losses['fused_view']
        + args.alignment_graph_smiles_weight * losses['graph_smiles']
        + args.alignment_graph_fp_weight * losses['graph_fp']
        + args.alignment_lomo_weight * losses['lomo']
        + args.alignment_pooling_kl_weight * losses['pooling_kl']
        + args.alignment_fused_mask_weight * losses.get('fused_mask', losses['fused_view'].new_tensor(0.0))
        + args.alignment_shared_private_weight * losses['shared_private']
    )


def _bf16_alignment_parity_gate(base_model, data, args):
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    fp32_loss = _alignment_total(_scage_semantic_alignment_losses(base_model, data, args)[0], args)
    fp32_loss.backward()
    fp32_grad = _gradient_vector((base_model,))
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss = _alignment_total(
            _scage_semantic_alignment_losses(base_model, data, args)[0], args
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector((base_model,))
    delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    base_model.zero_grad(set_to_none=True)
    return finite and delta <= 0.02 and cosine >= 0.98, {
        'relative_loss_delta': delta, 'gradient_cosine': cosine, 'finite': finite,
    }


def _scage_alignment_optimizer(base_model, args):
    groups = []
    learning_rates = {
        "graph": args.alignment_graph_lr,
        "smiles": args.alignment_smiles_lr,
        "fp": args.alignment_fp_lr,
    }
    for name in base_model.modality_list:
        groups.append({
            'params': list(base_model.encoders[name].parameters()),
            'lr': float(learning_rates[name]),
            'name': name,
        })
    groups.extend([
        {
            'params': (
                list(base_model.alignment_projections.parameters())
                + list(base_model.shared_private_projections.parameters())
            ),
            'lr': float(args.alignment_projection_lr),
            'name': 'alignment_projection',
        },
    ])
    parallel_fusion = getattr(base_model, "parallel_attention_fusion", None)
    if parallel_fusion is not None:
        groups.append({
            'params': list(parallel_fusion.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'fusion',
        })
    if hasattr(base_model, 'alignment_mask_head'):
        groups.append({
            'params': list(base_model.alignment_mask_head.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'alignment_mask_head',
        })
    return optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _module_gradient_norm(module):
    if module is None:
        return 0.0
    squares = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu().item())


def _module_gradient_diagnostic(module):
    if module is None:
        return {"finite": None, "nonzero": None, "norm": None}
    if isinstance(module, torch.Tensor):
        parameters = (module,)
    else:
        parameters = module.parameters()
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return {"finite": False, "nonzero": False, "norm": 0.0}
    finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    norm = float(torch.sqrt(torch.stack([value.pow(2).sum() for value in gradients]).sum()).cpu().item())
    return {"finite": bool(finite), "nonzero": bool(norm > 0.0), "norm": norm}


def _batched_shortest_path_targets(data, max_distance):
    targets = []
    pair_indices = []
    max_distance = int(max_distance)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 2:
            continue
        dist = torch.full((n, n), max_distance, device=data.x.device, dtype=torch.long)
        diag = torch.arange(n, device=data.x.device)
        dist[diag, diag] = 0
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        if keep.any():
            ls = local[src[keep]]
            ld = local[dst[keep]]
            valid = (ls >= 0) & (ld >= 0)
            dist[ls[valid], ld[valid]] = 1
        dist_float = dist.float()
        inf = float(max_distance)
        for k in range(n):
            dist_float = torch.minimum(dist_float, dist_float[:, k:k + 1] + dist_float[k:k + 1, :])
        dist = dist_float.clamp(max=inf).long()
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        if row.numel() == 0:
            continue
        pair_indices.append(torch.stack([node_idx[row], node_idx[col]], dim=1))
        targets.append(dist[row, col])
    if not targets:
        empty_pairs = data.edge_index.new_empty((0, 2))
        empty_targets = data.x.new_empty((0,), dtype=torch.long)
        return empty_pairs, empty_targets
    return torch.cat(pair_indices, dim=0), torch.cat(targets, dim=0)


def _graph_shortest_path_loss(base_model, data, sp_head, max_distance):
    if not _is_scage_graph_encoder(base_model):
        return data.x.new_tensor(0.0)
    _, node_rep = _graph_encode_nodes(base_model, data)
    pairs, targets = _batched_shortest_path_targets(data, max_distance=max_distance)
    if pairs.numel() == 0:
        return data.x.new_tensor(0.0)
    pair_rep = torch.cat([node_rep[pairs[:, 0]], node_rep[pairs[:, 1]], torch.abs(node_rep[pairs[:, 0]] - node_rep[pairs[:, 1]])], dim=-1)
    return F.cross_entropy(sp_head(pair_rep), targets)


def _graph_angle_loss(base_model, data, angle_head, angle_bins):
    if not _is_scage_graph_encoder(base_model) or not hasattr(data, 'pos3d') or not hasattr(data, 'batch3d'):
        return None
    _, node_rep = _graph_encode_nodes(base_model, data)
    reps = []
    targets = []
    angle_bins = int(angle_bins)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 3:
            continue
        coordinate_ok = getattr(data, 'geom_coordinate_ok', getattr(data, 'geom_build_ok', None))
        if coordinate_ok is not None and not bool(coordinate_ok.flatten()[graph_idx].item()):
            continue
        if not hasattr(data, 'graph_to_geom_index'):
            continue
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != n or (mapping < 0).any() or (mapping >= data.pos3d.size(0)).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        neigh = [[] for _ in range(n)]
        for s_idx, d_idx in zip(src[keep].tolist(), dst[keep].tolist()):
            center = int(local[s_idx].item())
            nb = int(local[d_idx].item())
            if center >= 0 and nb >= 0 and nb not in neigh[center]:
                neigh[center].append(nb)
        for center, ns in enumerate(neigh):
            if len(ns) < 2:
                continue
            for a_i in range(len(ns)):
                for b_i in range(a_i + 1, len(ns)):
                    left, right = ns[a_i], ns[b_i]
                    v1 = pos[left] - pos[center]
                    v2 = pos[right] - pos[center]
                    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
                    if float(denom.detach().cpu().item()) <= 1e-8:
                        continue
                    cos = torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0)
                    angle = torch.acos(cos)
                    target = torch.clamp((angle / torch.pi * angle_bins).long(), max=angle_bins - 1)
                    global_center = node_idx[center]
                    global_left = node_idx[left]
                    global_right = node_idx[right]
                    reps.append(torch.cat([node_rep[global_left], node_rep[global_center], node_rep[global_right]], dim=-1))
                    targets.append(target)
    if not reps:
        return None
    return F.cross_entropy(angle_head(torch.stack(reps, dim=0)), torch.stack(targets, dim=0).long())

def _graph_pretrain_loss(
    base_model,
    data,
    graph_atom_head,
    mask_ratio,
    mask_weight,
    periodic_aug_weight,
    graph_input,
    repeat_cut_max_mrus,
    repeat_cut_retry,
    repeat_cut_temperature,
):
    mask_loss = _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio)
    periodic_aug_loss = data.x.new_tensor(0.0)
    periodic_aug_stats = _empty_periodic_aug_stats()
    if float(periodic_aug_weight) > 0:
        periodic_aug_loss, periodic_aug_stats = _graph_periodic_aug_loss(
            base_model,
            data,
            graph_input=graph_input,
            temperature=repeat_cut_temperature,
            max_mrus=repeat_cut_max_mrus,
            retry=repeat_cut_retry,
        )
    total = float(mask_weight) * mask_loss + float(periodic_aug_weight) * periodic_aug_loss
    return total, mask_loss, periodic_aug_loss, periodic_aug_stats

def _geom_denoise_loss(
    base_model,
    data,
    geom_noise_head,
    noise_std,
    noise_std_min=None,
    noise_std_max=None,
):
    if 'geom' not in base_model.encoders:
        return data.pos3d.new_tensor(0.0)

    geom_module = base_model.encoders['geom']
    clean_pos = data.pos3d
    target_pos = clean_pos.detach().clone()
    if noise_std_min is not None or noise_std_max is not None:
        if noise_std_min is None or noise_std_max is None:
            raise ValueError("--geom_noise_std_min and --geom_noise_std_max must be set together")
        min_std = float(noise_std_min)
        max_std = float(noise_std_max)
        if min_std < 0.0 or max_std <= 0.0 or min_std > max_std:
            raise ValueError("Require 0 <= --geom_noise_std_min <= --geom_noise_std_max")
        std = torch.empty((), device=target_pos.device, dtype=target_pos.dtype).uniform_(min_std, max_std)
    else:
        std = torch.as_tensor(float(noise_std), device=target_pos.device, dtype=target_pos.dtype)
    noise = torch.randn_like(target_pos) * std
    try:
        data.pos3d = target_pos + noise
        node_rep, _ = geom_module.encoder.encode_nodes(data)
        pred_noise = geom_noise_head(node_rep)
        per_node = F.smooth_l1_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        context = getattr(data, 'geom_context_id', None)
        node_batch = getattr(data, 'batch3d', None)
        if context is None or node_batch is None:
            return per_node.mean()
        sample_weights = torch.where(context.to(per_node.device).long() == 1, 0.3, 1.0)
        weights = sample_weights[node_batch.to(per_node.device)]
        return (per_node * weights).sum() / weights.sum().clamp_min(1e-8)
    finally:
        data.pos3d = clean_pos

def _dataset_kwargs_from_args(args):
    return dict(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
        graph_encoder_type=args.graph_encoder_type,
        graph_input=args.graph_input,
        geom_input=args.geom_input,
        use_feature_cache=not args.disable_feature_cache,
        feature_source_dataset=args.feature_source_dataset,
        rebuild_feature_cache=args.rebuild_feature_cache,
        max_smiles_length=args.max_smiles_length,
        max_smiles_length_cap=args.max_smiles_length_cap,
        fp_mode=args.fp_mode,
        feature_cache_workers=args.feature_cache_workers,
        feature_cache_chunksize=args.feature_cache_chunksize,
        feature_cache_partial_every=args.feature_cache_partial_every,
        feature_cache_item_timeout=args.feature_cache_item_timeout,
        cache_layers=args.cache_layers,
        cache_validate=args.cache_validate,
        cache_commit_size=args.cache_commit_size,
        embed_tries_multiplier=args.embed_tries_multiplier,
        conformer_3d_count=args.conformer_3d_count,
        conformer_keep_count=args.conformer_keep_count,
        conformer_profile=args.conformer_profile,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        mips_core=args.mips_core,
        mips_max_hops=args.mips_max_hops,
        mips_use_descriptors=args.mips_use_descriptors,
        mips_descriptor_protocol=args.mips_descriptor_protocol,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        topology_representation=args.topology_representation,
        mcl_distance_percentiles=args.mcl_distance_percentiles,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        finite_variant=args.finite_variant,
        conformer_mode=args.conformer_mode,
        field_layout=args.field_layout,
        field_channels=args.field_channels,
        experiment_id=args.experiment_id,
        feature_config_hash=args.feature_config_hash,
        g_family_arm=args.g_family_arm,
        relation_geometry_sidecar=args.relation_geometry_sidecar,
        relation_geometry_artifact_hash=args.relation_geometry_artifact_hash,
        g3_permutation_sidecar=args.g3_permutation_sidecar,
        g3_permutation_artifact_hash=args.g3_permutation_artifact_hash,
        angle_cache_root_override=getattr(args, 'angle_cache_root_override', None),
    )


def validate_mts_pretrain_execution(profile, batch_size, accumulation, world_size):
    """Validate the execution-only batch decomposition of an MTS profile."""
    expected_world_size = int(profile.get("world_size", 3))
    if int(world_size) != expected_world_size:
        raise ValueError(
            "pretrain profile/world-size mismatch: "
            f"profile={expected_world_size}, runtime={world_size}"
        )
    effective = int(batch_size) * int(world_size) * int(accumulation)
    target = int(profile["global_batch"])
    if effective != target:
        raise ValueError(
            "canonical_ru_angle20_v1 requires batch_size * world_size * "
            f"gradient_accumulation_steps = {target}; got {effective}"
        )
    return effective


def main():
    pretrain_started = time.monotonic()
    args = parse_arguments()
    # Cache materialisation has no optimizer/objective identity and must not
    # be rejected merely because an explicit topology comparison does not use
    # the canonical-only formal training profile.
    pretrain_profile = (
        None if args.cache_only
        else _load_pretrain_profile(args.pretrain_profile)
    )
    if pretrain_profile is not None:
        if str(pretrain_profile["representation"]) != str(
            args.topology_representation
        ):
            raise RuntimeError(
                "formal pretraining profile/topology representation mismatch: "
                f"profile={pretrain_profile['representation']!r}, "
                f"requested={args.topology_representation!r}"
            )
        # Scientific choices in the formal profile are immutable.  Batch and
        # accumulation are the only values selected by the external benchmark.
        args.dataset_name = "PI1M_v2"
        args.seed = int(pretrain_profile["seed"])
        if not args.resume_smoke:
            args.max_optimizer_steps = int(pretrain_profile["optimizer_steps"])
        elif int(args.max_optimizer_steps) <= 0 or int(args.max_optimizer_steps) > 300:
            raise ValueError("resume_smoke requires 1 <= max_optimizer_steps <= 300")
        args.graph_mask_ratio = float(pretrain_profile["masked_atom_ratio"])
        args.scage_mips_mask_weight = float(pretrain_profile["masked_atom_weight"])
        args.graph_angle_weight = float(pretrain_profile["angle_weight"])
        args.scage_angle_bins = int(pretrain_profile["angle_bins"])
        args.scage_focal_gamma = float(pretrain_profile["focal_gamma"])
        args.lr = float(pretrain_profile["peak_lr"])
        args.warmup_steps = int(pretrain_profile["warmup_steps"])
        args.mips_scheduler = str(pretrain_profile["scheduler"])
        args.scheduler_power = float(pretrain_profile["scheduler_power"])
        args.end_lr = float(pretrain_profile["end_lr"])
        args.amp_dtype = "bf16"
        args.max_grad_norm = -1.0
        args.angle_objective = "categorical"
        if getattr(args, "g_family_arm", None) is not None:
            if args.pretraining_objective != "masked_atom_only" or float(args.angle_loss_weight) != 0.0:
                raise RuntimeError("G-family pretraining requires masked_atom_only and angle_loss_weight=0")
            if not args.g_family_bundle_hash:
                raise RuntimeError("G-family pretraining requires --g_family_bundle_hash")
            if args.g_family_arm != "g0" and (
                not args.relation_geometry_sidecar
                or not args.relation_geometry_artifact_hash
            ):
                raise RuntimeError("G1/G2/G3 pretraining requires the active PI1M_v2 artifact binding")
            if args.g_family_arm == "g3" and (
                not args.g3_permutation_sidecar
                or not args.g3_permutation_artifact_hash
            ):
                raise RuntimeError("G3 pretraining requires the active permutation artifact binding")
            # G-family readiness deliberately does not open or optimize the
            # legacy Angle-20 target.  The formal profile remains unchanged
            # for T-family runs.
            args.graph_angle_weight = 0.0
            args.pretraining_objective = "masked_atom_only"
        # torchrun exports WORLD_SIZE before the process group is initialized.
        # Validate the execution contract early without referring to the
        # post-init local ``world_size`` defined later in this function.
        requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        validate_mts_pretrain_execution(
            pretrain_profile, args.batch_size,
            args.gradient_accumulation_steps, requested_world_size,
        )
        if args.pretrained_model_path:
            raise RuntimeError(
                "canonical_ru_angle20_v1 must start from random initialization; "
                "parent checkpoints are forbidden"
            )
        if args.initialization_state and args.resume_state:
            raise RuntimeError(
                "fresh-paired initialization cannot be combined with resume_state"
            )
        if not args.cache_only and not args.benchmark_only and not args.resume_smoke:
            output_path = Path(args.save_path).resolve()
            historical = "mts_joint_pretraining_pi1m_v2_seed42_canonical_20260808"
            if historical in output_path.name:
                raise RuntimeError(
                    "formal canonical pretraining cannot target the historical "
                    "20260808 checkpoint"
                )
            complete_path = Path(str(output_path) + ".complete.json")
            last_path = Path(str(output_path) + ".last.pt")
            if args.resume_state:
                if output_path.exists() or complete_path.exists() or not last_path.exists():
                    raise RuntimeError(
                        "resume requires an existing matching .last.pt and no final/complete output"
                    )
            elif output_path.exists() or complete_path.exists() or last_path.exists():
                raise RuntimeError(
                    "fresh canonical pretraining refuses to overwrite an existing output; "
                    "use --resume with its matching .last.pt or choose a new path"
                )
    validate_mips_trimer_runtime(args)
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and args.pretrain_stage in {
            "mts_topology", "mts_geometry_adaptation",
            "topology_pretrain", "stage2_geometry_adapt", "scage_m4p"
        }
    ):
        raise ValueError(
            f"The legacy MTS stage spelling {args.pretrain_stage!r} is retired; "
            f"use {MTS_STAGE1_ID}."
        )
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL:
        requested_stage = normalize_stage(args.pretrain_stage)
    else:
        requested_stage = args.pretrain_stage
    args.geometry_adapt = False
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and requested_stage != MTS_STAGE1_ID
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} pretraining only supports {MTS_STAGE1_ID}; "
            f"property fine-tuning belongs to train.py as {MTS_STAGE2_ID}"
        )
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL:
        args.pretrain_stage = MTS_STAGE1_ID
    args.checkpoint_stage = requested_stage
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and args.dataset_name != "PI1M_v2"
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} pretraining is fixed to the full PI1M_v2 "
            "cohort; PI1M_50k/PI1M_200k are not active pretraining datasets."
        )
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and not args.cache_only
        and not args.resume_smoke
        and int(args.max_optimizer_steps) != 20000
    ):
        raise ValueError(
            f"{MTS_STAGE1_ID} is fixed to exactly 20000 optimizer steps"
        )
    # Ordinary ``mts-experiment-v3`` descriptors remain fine-tune-only.  The
    # matched G-family cycle is the one explicitly authorized exception: its
    # resolver has already bound the arm, MSTA identity, masked-atom-only
    # objective, zero angle weight, shared step-0 and (for G1) immutable
    # relation sidecar.  Keep the exception narrow so a random experiment
    # descriptor cannot silently become a formal pretraining entry point.
    g_family_formal = (
        args.config_schema == MIPS_EXPERIMENT_CONFIG_SCHEMA
        and getattr(args, "g_family_arm", None) in {"g0", "g1"}
        and not args.cache_only
        and not args.resume_smoke
        and getattr(args, "pretraining_objective", None) == "masked_atom_only"
        and float(getattr(args, "angle_loss_weight", 1.0)) == 0.0
        and getattr(args, "shared_step0_id", None)
        == "mts_g_family_step0_v2_seed42"
    )
    allowed_config_schema = (
        args.config_schema == MIPS_TRIMER_CONFIG_SCHEMA
        or (
            args.config_schema == MIPS_EXPERIMENT_CONFIG_SCHEMA
            and (args.cache_only or args.resume_smoke or g_family_formal)
        )
    )
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and not allowed_config_schema
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} formal training accepts only "
            f"{MIPS_TRIMER_CONFIG_SCHEMA}; experiment configs are limited "
            "to cache-only, short resume-smoke, or the strict G0/G1 matched "
            "pretraining contract"
        )
    stage1_weights = (
        args.scage_mips_mask_weight,
        args.graph_angle_weight,
        args.mips_spd_weight,
        args.mips_path_bond_weight,
        args.mips_repeat_consistency_weight,
        args.mips_distance_weight,
        args.mips_conformer_weight,
        args.scage_screw_geometry_weight,
    )
    if any(weight < 0 for weight in stage1_weights):
        raise ValueError("SCAGE Stage 1 task weights must be non-negative")
    if args.config_schema == MIPS_TRIMER_CONFIG_SCHEMA:
        selected_weights = (
            float(args.scage_mips_mask_weight),
            float(args.graph_angle_weight),
            float(args.mips_spd_weight),
            float(args.mips_path_bond_weight),
        )
        allowed_angle_weights = (
            {0.10, 0.25, 0.50}
            if args.angle_objective == 'cosine' else {0.25}
        )
        if (
            selected_weights[0] != 1.0
            or selected_weights[1] not in allowed_angle_weights
            or selected_weights[2:] != (0.0, 0.0)
        ):
            raise ValueError(
                "MTS joint weights require atom=1.0 and SPD/path=0; "
                "categorical angle uses 0.25 while Angle-v2 calibration "
                "allows 0.10/0.25/0.50"
            )
        retired_weights = (
            args.scage_ecfp_weight,
            args.mips_repeat_consistency_weight,
            args.mips_distance_weight,
            args.mips_conformer_weight,
            args.scage_screw_geometry_weight,
            # The Trimer bond-angle task is the only enabled auxiliary target.
        )
        if any(float(weight) != 0.0 for weight in retired_weights):
            raise ValueError(
                "selected O8 disables ECFP, repeat/cut, distance, conformer, "
                "screw, repeat/cut, distance and other retired objectives"
            )
        if args.dynamic_pretrain_loss:
            raise ValueError("MTS joint pretraining uses fixed loss weights")
    if args.scage_geometry_max_pairs < 1:
        raise ValueError("--scage_geometry_max_pairs must be positive")
    if not 0.0 <= args.scage_shift_balance_power <= 1.0:
        raise ValueError("--scage_shift_balance_power must be in [0, 1]")
    if args.cache_only:
        # Must happen before any CUDA query and before forking RDKit/torch
        # workers. CPU Adam otherwise performs a CUDA graph health check in a
        # bad fork and rejects valid periodic candidates.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    requested_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if args.cache_only and requested_distributed:
        raise ValueError("--cache_only must run as one CPU process, not through torchrun")
    distributed = requested_distributed
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=24),
            device_id=torch.device('cuda', local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
    if args.geom_input in {"polygen_periodic", "screw_periodic", "smer_context"} and args.graph_encoder_type != "mips_trimer_scage":
        raise ValueError("polygen_periodic, screw_periodic and smer_context require graph_encoder_type=scage")
    if args.graph_encoder_type == 'mips_trimer_scage':
        if distributed and world_size != 3:
            raise ValueError(
                "The non-PBC MIPS campaign requires exactly three DDP ranks "
                f"(physical GPUs 1,2,3); received world_size={world_size}."
            )
        if args.mips_max_hops is None:
            args.mips_max_hops = (
                2 if args.mips_core == "paper_corrected" else 5
            )
        fixed = {
            'graph_num_layers': (args.graph_num_layers, 6),
            'graph_emb_dim': (args.graph_emb_dim, 512),
            'scage_num_heads': (args.scage_num_heads, 8),
            'scage_ffn_hidden_dim': (args.scage_ffn_hidden_dim, 2048),
            'scage_num_kernels': (args.scage_num_kernels, 128),
        }
        mismatched = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in fixed.items()
            if actual != expected
        ]
        if (
            args.graph_input != 'star_linking'
            or args.geom_input != 'repeat_unit'
            or args.scage_use_pbc_distance
            or mismatched
        ):
            raise ValueError(
                "The second route is fixed to non-PBC sparse MIPS with "
                "star_linking and geom_input=repeat_unit; "
                + ", ".join(mismatched)
            )
    from src.dataset import UniDataset
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and args.angle_objective == 'cosine'
        and not args.angle_cache_root_override
    ):
        from scripts.audit_mips_trimer_cache import _specs as _cache_specs
        from src.dataset.lmdb_cache import build_or_load_cohort
        from src.dataset.trimer_angle_continuous_cache import continuous_angle_root
        cache_specs = _cache_specs(Path(PROJECT_ROOT))
        cohort = build_or_load_cohort(
            Path(args.root) / 'processed' / 'mips_trimer_scage',
            'PI1M_v2', Path(args.root) / 'raw' / 'PI1M_v2.csv',
            load_text=False, verify_integrity=True,
        )
        args.angle_cache_root_override = str(continuous_angle_root(
            cache_specs['trimer']['root'],
            cohort['manifest']['cohort_hash'],
        ))
    dataset_kwargs = _dataset_kwargs_from_args(args)
    if args.graph_encoder_type == "mips_trimer_scage" and not args.cache_only:
        from scripts.audit_mips_trimer_cache import _specs as _cache_specs
        from src.dataset.mips_cache_validation import verify_frozen_cache_bundle
        cache_specs = _cache_specs(Path(PROJECT_ROOT))
        verify_frozen_cache_bundle(
            cache_specs,
            store_path=Path(cache_specs["topology"]["root"]).parents[1]
            / "validation" / "store.json",
            required_layers=cache_specs.keys(),
        )
    if args.cache_only:
        warnings.filterwarnings("ignore")
        dataset = UniDataset(**dataset_kwargs)
        print(
            f"[feature_cache] cache-only complete: path={dataset.feature_cache_path}, "
            f"usable_rows={len(dataset)}"
        )
        return

    from src.modules import UniEncoderAttention
    from src.utils import compute_contrastive_loss, get_data_loader, set_global_seed
    # All ranks must construct exactly the same parameters.  Rank-specific
    # randomness is enabled only after DDP has broadcast the model state.
    set_global_seed(args.seed)

    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if rank == 0:
            print(f"Found {n_gpus} GPUs available; distributed world_size={world_size}")
        device = torch.device("cuda", local_rank)
    else:
        if args.graph_encoder_type == "mips_trimer_scage":
            raise RuntimeError(f"{MTS_ROUTE_NAME} pretraining requires CUDA")
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    if device.type == "cuda":
        # These affect only float32 matmuls outside the BF16 autocast region
        # and are deterministic on the fixed CUDA hardware used by the route.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    if distributed:
        # Cache construction is a single-writer CPU operation.  Rank zero may
        # build a missing/rebuilt cache; all other ranks open it only after the
        # atomic final save has completed.
        if rank == 0:
            dataset = UniDataset(**dataset_kwargs)
        dist.barrier()
        if rank != 0:
            dataset_kwargs['rebuild_feature_cache'] = False
            dataset = UniDataset(**dataset_kwargs)
    else:
        dataset = UniDataset(**dataset_kwargs)
    if args.graph_encoder_type == "mips_trimer_scage":
        cohort = getattr(dataset, "_cohort", None)
        if (
            cohort is None
            or int(cohort["manifest"].get("unique_count", -1)) != 995799
        ):
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} topology/geometry stages require the complete "
                "PI1M_v2 cohort (995799 records)."
            )
        if pretrain_profile is not None:
            observed_cohort_hash = str(cohort["manifest"].get("cohort_hash", ""))
            if observed_cohort_hash != str(pretrain_profile["cohort_hash"]):
                raise RuntimeError(
                    "canonical_ru_angle20_v1 cohort hash mismatch: "
                    f"expected={pretrain_profile['cohort_hash']} "
                    f"observed={observed_cohort_hash}"
                )
        # Resolve all production layers for the lifecycle gate, even though
        # Stage 1 itself only requests ru_base/topology and Stage 2 adds
        # Trimer.  The finalized downstream union is frozen before training.
        original_layers = dataset.cache_layers
        dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
        specs = dataset._lmdb_cache_specs({})
        dataset.cache_layers = original_layers
        unfrozen = [
            name for name, spec in specs.items()
            if not os.path.isfile(os.path.join(spec["root"], ".frozen"))
        ]
        if unfrozen:
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} cache artifacts are not frozen: "
                + ", ".join(unfrozen)
            )
        if (
            getattr(args, "g_family_arm", None) is None
            and getattr(getattr(dataset, "_lazy_feature_store", None), "angle_offsets", None) is None
        ):
            raise RuntimeError(
                "MTS joint pretraining requires the frozen Trimer bond-angle "
                "cache; run scripts/prepare_mts_angle_cache.py first"
            )
        if pretrain_profile is not None and getattr(args, "g_family_arm", None) is None:
            angle_meta = getattr(dataset, "angle_cache_metadata", None) or {}
            if (
                angle_meta.get("schema") != pretrain_profile["angle_cache_schema"]
                or getattr(dataset, "angle_cache_artifact_hash", None)
                != pretrain_profile["angle_cache_artifact"]
            ):
                raise RuntimeError(
                    "canonical_ru_angle20_v1 requires the frozen categorical "
                    "Angle-20 v2 sidecar; continuous or stale sidecars are rejected"
                )
        if args.angle_objective == 'categorical' and getattr(args, "g_family_arm", None) is None:
            angle_counts = np.asarray(
                getattr(dataset, "angle_class_counts", np.zeros(20, dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            if angle_counts.shape != (20,) or int(angle_counts.sum()) <= 0:
                raise RuntimeError("MTS angle cache has no valid class histogram")
            args.angle_majority_class = int(angle_counts.argmax())
            args.angle_majority_baseline = float(
                angle_counts.max() / angle_counts.sum()
            )
        else:
            args.angle_majority_class = -1
            args.angle_majority_baseline = float('nan')
    indices = np.arange(len(dataset))
    if args.pretrain_unique_smiles and args.pretrain_stage in {'mts_joint_pretraining', 'scage_m4p', 'alignment'}:
        # PI1M_v2 is a content-addressed, already unique cohort.  Do not
        # rebuild a million-entry Python ``set`` (or deserialize graph rows)
        # just to rediscover that invariant on every DDP rank.  Other
        # datasets retain the historical canonical-SMILES deduplication.
        manifest = getattr(dataset, "_cohort", {}).get("manifest", {})
        cohort_is_unique = (
            args.graph_encoder_type == "mips_trimer_scage"
            and int(manifest.get("unique_count", -1)) == len(dataset)
            and int(manifest.get("record_count", len(dataset))) == len(dataset)
        )
        if cohort_is_unique:
            indices = np.arange(len(dataset), dtype=np.int64)
            if rank == 0:
                print(
                    "Pretraining identity deduplication skipped: "
                    "PI1M_v2 manifest guarantees unique sample keys"
                )
        else:
            indices = _unique_smiles_indices(dataset)
            print(
                f"Pretraining identity deduplication: {len(dataset)} rows -> "
                f"{len(indices)} unique SMILES"
            )
    angle_validation_indices = np.empty((0,), dtype=np.int64)
    if args.angle_objective == 'cosine':
        validation_mask = getattr(dataset, 'angle_validation_mask', None)
        if validation_mask is None:
            raise RuntimeError(
                'Angle-v2 requires its deterministic 1% validation mask'
            )
        validation_mask = np.asarray(validation_mask, dtype=np.bool_)
        if validation_mask.shape != (len(dataset),):
            raise RuntimeError('Angle-v2 validation mask shape mismatch')
        angle_validation_indices = indices[validation_mask[indices]]
        indices = indices[~validation_mask[indices]]
        if not len(angle_validation_indices) or not len(indices):
            raise RuntimeError('Angle-v2 train/validation split is empty')
        if rank == 0:
            print(
                'Angle-v2 deterministic split: '
                f'train={len(indices)}, validation={len(angle_validation_indices)}'
            )
    sampler = None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed) + 9176 * int(rank))
    if distributed:
        from torch.utils.data import Subset
        subset = Subset(dataset, [int(index) for index in indices])
        if args.graph_encoder_type == "mips_trimer_scage":
            cost_path = None
            if getattr(cohort, "get", None) is not None:
                cost_path = Path(cohort["root"]) / "topology_cost.npy"
            if args.batch_balance == "cost":
                if cost_path is None or not cost_path.is_file():
                    raise RuntimeError(
                        "cost-balanced MIPS sampling requires the frozen "
                        "topology_cost.npy derived artifact"
                    )
                costs = np.load(cost_path, mmap_mode="r")
                cost_meta_path = cost_path.with_name(
                    "topology_cost_metadata.json"
                )
                if not cost_meta_path.is_file():
                    raise RuntimeError(
                        "topology_cost.npy metadata is missing; rebuild the "
                        "bound cost artifact before training"
                    )
                with open(cost_meta_path, encoding="utf-8") as handle:
                    cost_meta = json.load(handle)
                cost_artifact = getattr(
                    dataset, "topology_cache_artifact_hash", None
                )
                topology_content_hash = None
                try:
                    from scripts.audit_mips_trimer_cache import _specs as _cache_specs
                    topo_manifest = json.loads(
                        (
                            Path(_cache_specs(Path(PROJECT_ROOT))["topology"]["root"])
                            / "manifest.json"
                        ).read_text(encoding="utf-8")
                    )
                    topology_content_hash = topo_manifest.get("metadata_hash")
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    topology_content_hash = None
                expected_cost_meta = {
                    "schema": MIPS_CACHE_TOPOLOGY_COST_SCHEMA,
                    "cohort_hash": cohort["manifest"]["cohort_hash"],
                    "ordered_key_hash": cohort["manifest"][
                        "ordered_sample_key_hash"
                    ],
                    "topology_done_artifact_hash": cost_artifact,
                    "topology_content_hash": topology_content_hash,
                    "shape": [len(dataset), 2],
                    "dtype": "uint32",
                    "file_sha256": _file_sha256(cost_path),
                    "builder_version": MIPS_BUILDER_VERSION,
                    "columns": ["node_count", "lga_edge_count"],
                }
                if any(
                    cost_meta.get(key) != value
                    for key, value in expected_cost_meta.items()
                ):
                    raise RuntimeError(
                        "topology_cost.npy metadata is not bound to the "
                        "current frozen PI1M_v2 topology artifact"
                    )
                if (
                    costs.ndim != 2
                    or tuple(costs.shape) != (len(dataset), 2)
                    or costs.dtype != np.uint32
                    or cost_meta.get("file_sha256") != _file_sha256(cost_path)
                ):
                    raise RuntimeError(
                        "topology_cost.npy is incomplete or has a stale hash"
                    )
                sampler = _CostBalancedDistributedSampler(
                    costs[np.asarray(indices, dtype=np.int64)],
                    batch_size=args.batch_size,
                    num_replicas=world_size,
                    rank=rank,
                    seed=args.seed,
                    drop_last=False,
                )
            else:
                # LMDB provides single-record random access. DistributedSampler
                # pads deterministically and covers every source row.
                sampler = torch.utils.data.DistributedSampler(
                    subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed,
                    drop_last=False,
                )
        else:
            sampler = _ShardAwareDistributedSampler(
                len(subset),
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
    dataloader = get_data_loader(
        dataset, indices=indices, batch_size=args.batch_size, shuffle=True,
        drop_last=(
            distributed and args.graph_encoder_type != "mips_trimer_scage"
        ),
        random_conformer=True,
        num_workers=args.loader_workers, pin_memory=True,
        prefetch_factor=args.loader_prefetch_factor,
        persistent_workers=args.loader_workers > 0, sampler=sampler,
        generator=loader_generator,
    )
    angle_validation_loader = None
    if args.angle_objective == 'cosine':
        validation_sampler = None
        if distributed:
            from torch.utils.data import Subset
            validation_subset = Subset(
                dataset, [int(index) for index in angle_validation_indices]
            )
            validation_sampler = _RankSliceSampler(
                len(validation_subset), rank, world_size
            )
        angle_validation_loader = get_data_loader(
            dataset, indices=angle_validation_indices,
            batch_size=args.batch_size, shuffle=False, drop_last=False,
            random_conformer=False, num_workers=args.loader_workers,
            pin_memory=True, prefetch_factor=args.loader_prefetch_factor,
            persistent_workers=args.loader_workers > 0,
            sampler=validation_sampler,
        )

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=args.joint_embedding_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name=args.gnn_model_name,
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder,
        graph_num_layers=args.graph_num_layers,
        graph_emb_dim=args.graph_emb_dim,
        graph_dropout=args.graph_dropout,
        graph_encoder_type=args.graph_encoder_type,
        scage_dist_bar=args.scage_dist_bar,
        scage_num_heads=args.scage_num_heads,
        scage_ffn_hidden_dim=args.scage_ffn_hidden_dim,
        scage_num_kernels=args.scage_num_kernels,
        scage_attention_dropout=args.scage_attention_dropout,
        scage_use_pbc_distance=args.scage_use_pbc_distance,
        scage_use_descriptors=args.scage_use_descriptors,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        scage_distance_scales=args.scage_distance_scales,
        scage_distance_taus=args.scage_distance_taus,
        scage_topology_bias=args.scage_topology_bias,
        scage_topology_max_distance=args.scage_topology_max_distance,
        scage_topology_locality_mode=args.scage_topology_locality_mode,
        scage_topology_locality_threshold=args.scage_topology_locality_threshold,
        scage_topology_locality_tau=args.scage_topology_locality_tau,
        scage_periodic_image_mode=args.scage_periodic_image_mode,
        scage_periodic_image_cap=args.scage_periodic_image_cap,
        scage_periodic_image_temperature=args.scage_periodic_image_temperature,
        scage_force_topology_only=args.scage_force_topology_only,
        mips_core=args.mips_core,
        mips_max_hops=args.mips_max_hops,
        mips_use_descriptors=args.mips_use_descriptors,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        mcl_distance_percentiles=args.mcl_distance_percentiles,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        mips_fusion_mode=args.mips_fusion_mode,
        projection_mode=args.projection_mode,
        modality_control=args.modality_control,
        mips_atom_feature_mode=args.mips_atom_feature_mode,
        mips_attention_scale=args.mips_attention_scale,
        mips_norm_mode=args.mips_norm_mode,
        mips_activation=args.mips_activation,
        mips_spd_bias_mode=args.mips_spd_bias_mode,
            mips_path_bias_mode=args.mips_path_bias_mode,
            mips_multi_scale_hop_gate=args.mips_multi_scale_hop_gate,
            mips_semantics=args.mips_semantics,
            mips_descriptor_fusion_mode=args.mips_descriptor_fusion_mode,
            mips_descriptor_components=args.mips_descriptor_components,
            mips_descriptor_disturbance=args.mips_descriptor_disturbance,
            mips_backbone_mode=args.mips_backbone_mode,
            mips_input_norm=args.mips_input_norm,
            mips_mask_mode=args.mips_mask_mode,
            mips_mask_policy=args.mips_mask_policy,
            mips_masked_loss_reduction=args.mips_masked_loss_reduction,
        topology_attention_variant=args.topology_attention_variant,
        msta_layer_indices=args.msta_layer_indices,
        msta_local_spd=args.msta_local_spd,
        msta_context_spd=args.msta_context_spd,
        msta_share_relation_dropout=args.msta_share_relation_dropout,
        msta_local_output_bias=args.msta_local_output_bias,
        msta_local_output_init=args.msta_local_output_init,
        g_family_arm=args.g_family_arm,
        relation_geometry_sidecar=args.relation_geometry_sidecar,
        g3_permutation_sidecar=args.g3_permutation_sidecar,
        fusion_type=args.fusion_type,
        fp_mode=args.fp_mode,
        fusion_dropout=args.fusion_dropout,
        alignment_projection_dim=args.alignment_projection_dim,
    )

    initialization_info = None
    if args.initialization_state:
        initialization_info = _load_fresh_paired_initialization(model, args)
        if rank == 0:
            print(
                "Loaded fresh-paired step-0 initialization: "
                f"identity={initialization_info['model_identity']} "
                f"sha256={initialization_info['sha256']}"
            )

    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        if args.graph_encoder_type == 'mips_trimer_scage':
            expected_schema = MIPS_TRIMER_CHECKPOINT_SCHEMA
            if not isinstance(checkpoint, dict) or checkpoint.get('meta', {}).get('schema') != expected_schema:
                raise RuntimeError(
                    f"{MTS_ROUTE_NAME} requires a {expected_schema} checkpoint; "
                    f"rerun {MTS_STAGE1_ID} with the matching model."
                )
            checkpoint_stage = checkpoint.get('meta', {}).get('stage')
            if args.geometry_adapt and checkpoint_stage != MTS_STAGE1_ID:
                raise RuntimeError(
                    f"{MTS_STAGE2_ID} requires an {MTS_STAGE1_ID} checkpoint; "
                    f"received stage={checkpoint_stage!r}."
                )
            meta = checkpoint.get("meta", {})
            checkpoint_representation = meta.get(
                "topology_representation", TOPOLOGY_CANONICAL
            )
            if checkpoint_representation != args.topology_representation:
                raise RuntimeError(
                    "MTS checkpoint topology representation mismatch: "
                    f"checkpoint={checkpoint_representation!r}, "
                    f"requested={args.topology_representation!r}. "
                    "canonical_lifted and explicit_k_ru checkpoints are not "
                    "cross-loadable."
                )
            if (
                meta.get("baseline") != MTS_ROUTE_NAME
                or meta.get("route_short_name") != MTS_ROUTE_SHORT_NAME
                or meta.get("config_schema") != MIPS_TRIMER_CONFIG_SCHEMA
                or meta.get("feature_schema") != MIPS_TRIMER_FEATURE_SCHEMA
                or meta.get("cache_bundle_schema")
                != MIPS_TRIMER_CACHE_BUNDLE_SCHEMA
                or meta.get("topology_lmdb_schema")
                != MIPS_TRIMER_TOPOLOGY_SCHEMA
                or int(meta.get("mips_local_lga_schema_version", -1))
                != MIPS_CANONICAL_LGA_SCHEMA_VERSION
                or not meta.get("cache_bundle_hash")
                or meta.get("cache_bundle_hash")
                != cache_bundle_binding_hash(
                    cohort_hash=meta.get("source_cohort_hash"),
                    topology_artifact_hash=getattr(
                        dataset, "topology_cache_artifact_hash", None
                    ),
                    trimer_artifact_hash=None,
                )
                or meta.get("mips_core") != args.mips_core
                or int(meta.get("mips_max_hops", -1)) != int(args.mips_max_hops)
                or bool(meta.get("mips_use_descriptors", False))
                != bool(args.mips_use_descriptors)
                or meta.get("spatial_mode", "none") != args.spatial_mode
                or meta.get("mips_variant") != args.mips_variant
                or meta.get("topology_attention_variant", "o8")
                != args.topology_attention_variant
                or list(meta.get("msta_layer_indices", [4, 5]))
                != list(args.msta_layer_indices)
                or list(meta.get("msta_local_spd", [0, 1]))
                != list(args.msta_local_spd)
                or list(meta.get("msta_context_spd", [0, 1, 2]))
                != list(args.msta_context_spd)
                or meta.get("o8_feature_config_hash")
                != args.o8_feature_config_hash
                or meta.get("graph_model_config_hash")
                != args.graph_model_config_hash
                or meta.get("geometry_model_config_hash", "manual")
                != args.geometry_model_config_hash
                or meta.get("topology_cache_hash")
                != getattr(dataset, "topology_cache_hash", None)
                or meta.get("topology_cache_artifact_hash")
                != getattr(dataset, "topology_cache_artifact_hash", None)
                or int(meta.get("optimizer_steps", -1)) != 20000
                or meta.get("pretraining_dataset") != "PI1M_v2"
                or (
                    args.geometry_adapt
                    and meta.get("source_cohort_hash")
                    != getattr(dataset, "feature_cohort_hash", None)
                )
            ):
                raise RuntimeError(
                    "MIPS checkpoint configuration does not match the requested "
                    "source/tier/feature/Graph settings."
                )
            checkpoint_state = {
                key: value for key, value in checkpoint['state_dict'].items()
                if key.startswith('encoders.graph.encoder.')
            }
            expected_graph_state = {
                key: value for key, value in model.state_dict().items()
                if key.startswith('encoders.graph.encoder.')
            }
            if set(checkpoint_state) != set(expected_graph_state):
                missing_graph = sorted(
                    set(expected_graph_state) - set(checkpoint_state)
                )
                unexpected_graph = sorted(set(checkpoint_state) - set(expected_graph_state))
                raise RuntimeError(
                    f"{MTS_STAGE1_ID} must transfer the complete fixed "
                    f"architecture into {MTS_STAGE2_ID}; "
                    f"missing={missing_graph[:5]}, "
                    f"unexpected={unexpected_graph[:5]}"
                )
            shape_mismatch = [
                key for key in checkpoint_state
                if checkpoint_state[key].shape != expected_graph_state[key].shape
            ]
            if shape_mismatch:
                raise RuntimeError(
                    "SCAGE Stage 1 Graph tensor shape mismatch: "
                    + ", ".join(shape_mismatch[:10])
                )
        else:
            checkpoint_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_state = model.state_dict()
        missing = sorted(set(model_state) - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - set(model_state))
        if args.graph_encoder_type == 'mips_trimer_scage':
            graph_mismatch = [
                key for key in list(missing) + list(unexpected)
                if 'encoders.graph.encoder' in key
                and ".trimer_mcl." not in key
            ]
            if graph_mismatch:
                raise RuntimeError(
                    "SCAGE checkpoint architecture mismatch. The original-input SCAGE backbone "
                    "cannot load an older polymer-SCAGE checkpoint. Mismatched keys: "
                    + ", ".join(graph_mismatch[:10])
                )
        model_state.update({
            key: value for key, value in checkpoint_state.items()
            if key in model_state
        })
        model.load_state_dict(model_state, strict=True)
        print(f"Loaded pretraining checkpoint from {args.pretrained_model_path}")
        print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    model = model.to(device)
    base_model = _base_model(model)
    if args.geometry_adapt:
        raise RuntimeError("MTS Geometry Adaptation is retired; use joint pretraining")
    loss_weights = _stage_loss_weights(args)
    dynamic_loss_weighter = None
    display_stage = (
        stage_display_name(args.checkpoint_stage)
        if args.graph_encoder_type == MTS_ROUTE_INTERNAL
        else args.pretrain_stage
    )
    print(
        "Pretraining stage: "
        f"{display_stage} "
        f"(contrastive={loss_weights['contrastive']}, graph={loss_weights['graph']}, geom={loss_weights['geom']}, "
        f"dynamic_loss={args.dynamic_pretrain_loss})"
    )

    aux_modules = nn.ModuleList()
    graph_atom_head = None
    geom_noise_head = None
    graph_sp_head = None
    graph_angle_head = None
    m4p_ecfp_head = None
    m4p_torsion_head = None
    m4p_distance_head = None
    m4p_shift_head = None
    m4p_screw_head = None
    m4p_projection_head = None
    mips_atom_head = None
    mips_path_bond_head = None
    m4p_ecfp_pos_weight = None
    m4p_ecfp_valid_count = 0
    if args.pretrain_stage in {'scage_m4p', MTS_STAGE1_ID}:
        graph_encoder = base_model.encoders['graph'].encoder
        graph_dim = graph_encoder.emb_dim
        atom_classes = int(
            getattr(graph_encoder, "masked_atom_classes", 119)
        )
        if getattr(graph_encoder, "semantics", "") == "public_code_diagnostic":
            mips_atom_head = nn.Sequential(
                nn.Linear(graph_dim, graph_dim),
                nn.GELU(),
                nn.Dropout(args.graph_dropout),
                nn.Linear(graph_dim, atom_classes),
            ).to(device)
        else:
            mips_atom_head = nn.Linear(graph_dim, atom_classes).to(device)
        if args.pretrain_stage == MTS_STAGE1_ID:
            graph_angle_head = TrimerAngleHead(
                dim=graph_dim,
                hidden=256,
                bins=int(args.scage_angle_bins),
                alpha=(
                    (
                        torch.ones(int(args.scage_angle_bins), device=device)
                        if getattr(args, "g_family_arm", None) is not None
                        else _angle_alpha_from_dataset(dataset, args.scage_angle_bins)
                    )
                    if args.angle_objective == 'categorical' else None
                ),
                dropout=0.10,
                objective=args.angle_objective,
            ).to(device)
        else:
            # Retained only for non-production historical diagnostics.
            graph_sp_head = nn.Linear(graph_dim * 3, 3).to(device)
            mips_path_bond_head = nn.Linear(graph_dim * 3, 6).to(device)
            aux_modules.extend([
                mips_atom_head, graph_sp_head, mips_path_bond_head,
            ])
    elif not (args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage') \
            and 'graph' in args.modalities and loss_weights['graph'] > 0:
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import allowable_features
        graph_atom_head = nn.Linear(graph_dim, len(allowable_features['possible_atom_symbols'])).to(device)
        aux_modules.append(graph_atom_head)
        if args.graph_encoder_type == 'mips_trimer_scage':
            graph_sp_head = nn.Linear(graph_dim * 3, int(args.scage_sp_max_distance) + 1).to(device)
            graph_angle_head = nn.Linear(graph_dim * 3, int(args.scage_angle_bins)).to(device)
            aux_modules.extend([graph_sp_head, graph_angle_head])
    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        atom_classes = int(
            base_model.encoders['graph'].encoder.masked_atom_classes
        )
        base_model.alignment_mask_head = nn.Linear(
            graph_dim + base_model.joint_embedding_dim,
            atom_classes,
        ).to(device)
    if args.pretrain_stage != 'scage_m4p' and 'geom' in args.modalities and loss_weights['geom'] > 0:
        geom_dim = base_model.encoders['geom'].encoder.hidden_channels
        geom_noise_head = nn.Linear(geom_dim, 3).to(device)
        aux_modules.append(geom_noise_head)

    if args.amp_dtype == 'bf16':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            args.amp_dtype = 'fp32'
            if rank == 0:
                print("BF16 disabled: CUDA BF16 support is unavailable")
        elif args.pretrain_stage == MTS_STAGE1_ID:
            parity_batch = None
            for candidate in dataloader:
                candidate = candidate.to(device)
                if not args.geometry_adapt or bool(
                    _mcl_valid_graph_mask(candidate).any()
                ):
                    parity_batch = candidate
                    break
            if parity_batch is None:
                raise RuntimeError(
                    "BF16 parity gate could not find a valid geometry batch"
                )
            passed, parity = _bf16_joint_parity_gate(
                base_model, parity_batch, mips_atom_head, graph_angle_head, args
            )
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'
        elif args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
            parity_batch = next(iter(dataloader)).to(device)
            passed, parity = _bf16_alignment_parity_gate(base_model, parity_batch, args)
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 alignment parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'

    mips_ddp_route = (
        args.graph_encoder_type == "mips_trimer_scage"
        and args.pretrain_stage == MTS_STAGE1_ID
    )
    if mips_ddp_route:
        # Restrict the trainable/DDP parameter set to the active stage.  The
        # model object still owns the complete production encoder for strict
        # checkpoint compatibility, but frozen branches must not receive
        # zero-gradient anchors or optimizer state during pretraining.
        for parameter in model.parameters():
            parameter.requires_grad = False
        graph_encoder = base_model.encoders["graph"].encoder
        active_modules = (
            graph_encoder.atom_embedding,
            graph_encoder.spd_embedding,
            graph_encoder.path_bias,
            graph_encoder.layers,
        )
        if getattr(graph_encoder, "g_family_arm", None) is None:
            active_modules = active_modules + (
                graph_encoder.star_distance_bias,
                graph_encoder.trimer_mcl,
            )
        elif str(graph_encoder.g_family_arm) in {"g1", "g2", "g3"}:
            active_modules = active_modules + (graph_encoder.relation_geometry_bias,)
        for module in active_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    mips_heads = {}
    if args.pretrain_stage == MTS_STAGE1_ID:
        mips_heads = {
            "mips_atom": mips_atom_head,
            "angle": graph_angle_head,
        }
    train_container = (
        MIPSPretrainContainer(model, mips_heads).to(device)
        if mips_ddp_route else None
    )
    train_module = train_container if train_container is not None else model
    if distributed and mips_ddp_route:
        from torch.nn.parallel import DistributedDataParallel
        train_module = DistributedDataParallel(
            train_container,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
        _assert_ddp_parameters_synced(train_module, step=0)
        # Parameter initialization is synchronized by DDP.  Only stochastic
        # data transforms and dropout now receive rank-specific streams.
        set_global_seed(args.seed + 100003 * rank)

    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
        optimizer = _scage_alignment_optimizer(base_model, args)
    elif args.pretrain_stage == MTS_STAGE1_ID and args.graph_encoder_type == 'mips_trimer_scage':
        trainable_parameters = [
            parameter for parameter in train_container.parameters()
            if parameter.requires_grad
        ]
        adam_kwargs = dict(
            lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
        )
        # Match the public MIPS implementation: ordinary torch.optim.Adam,
        # without forcing a backend solely for bitwise interruption parity.
        optimizer = optim.Adam(trainable_parameters, **adam_kwargs)
    else:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(aux_modules.parameters()),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
        )

    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    available_batches = int(args.epochs) * len(dataloader)
    scheduled_batches = (
        min(available_batches, int(args.max_steps))
        if int(args.max_steps) > 0 else available_batches
    )
    total_optimizer_steps = max(1, math.ceil(scheduled_batches / accumulation_steps))
    if int(args.max_optimizer_steps) > 0:
        total_optimizer_steps = min(
            total_optimizer_steps, int(args.max_optimizer_steps)
        )
    if args.pretrain_stage == MTS_STAGE1_ID:
        warmup_steps = min(
            max(0, total_optimizer_steps - 1),
            max(0, int(args.warmup_steps)),
        )
    else:
        warmup_steps = min(
            total_optimizer_steps - 1,
            max(0, int(round(total_optimizer_steps * float(args.warmup_ratio)))),
        )

    def lr_scale(update_step):
        if warmup_steps > 0 and update_step < warmup_steps:
            return float(update_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, (update_step - warmup_steps) / decay_steps))
        if args.pretrain_stage == MTS_STAGE1_ID or args.mips_scheduler == 'polynomial':
            base_lr = max(float(args.lr), 1e-12)
            floor = min(1.0, max(0.0, float(args.end_lr) / base_lr))
            return floor + (1.0 - floor) * (1.0 - progress) ** float(args.scheduler_power)
        if args.mips_scheduler == 'linear':
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    if rank == 0:
        print(
            f"Optimizer schedule: updates={total_optimizer_steps}, "
            f"warmup={warmup_steps}, {args.mips_scheduler} decay"
        )

    resume_contract = {
        "config_schema": args.config_schema,
        "stage": args.checkpoint_stage,
        "topology_representation": args.topology_representation,
        "feature_config_hash": args.feature_config_hash,
        "o8_feature_config_hash": args.o8_feature_config_hash,
        "graph_model_config_hash": args.graph_model_config_hash,
        "topology_attention_variant": args.topology_attention_variant,
        "msta_layer_indices": list(args.msta_layer_indices),
        "msta_local_spd": list(args.msta_local_spd),
        "msta_context_spd": list(args.msta_context_spd),
        "msta_share_relation_dropout": bool(args.msta_share_relation_dropout),
        "msta_local_output_bias": bool(args.msta_local_output_bias),
        "msta_local_output_init": args.msta_local_output_init,
        "g_family_arm": getattr(args, "g_family_arm", None),
        "g_family_bundle_hash": getattr(args, "g_family_bundle_hash", None),
        "relation_geometry_sidecar": getattr(args, "relation_geometry_sidecar", None),
        "relation_geometry_artifact_hash": getattr(args, "relation_geometry_artifact_hash", None),
        "g3_permutation_sidecar": getattr(args, "g3_permutation_sidecar", None),
        "g3_permutation_artifact_hash": getattr(args, "g3_permutation_artifact_hash", None),
        "pretraining_objective": getattr(args, "pretraining_objective", "joint"),
        "angle_loss_weight": float(getattr(args, "angle_loss_weight", 0.25)),
        "shared_step0_id": getattr(args, "shared_step0_id", None),
        "geometry_model_config_hash": args.geometry_model_config_hash,
        "source_geometry_model_config_hash": args.source_geometry_model_config_hash,
        "training_config_hash": args.training_config_hash,
        "pretraining_dataset": args.dataset_name,
        "source_cohort_hash": getattr(dataset, "feature_cohort_hash", None),
        "topology_cache_hash": getattr(dataset, "topology_cache_hash", None),
        "topology_cache_artifact_hash": getattr(
            dataset, "topology_cache_artifact_hash", None
        ),
        "trimer_cache_hash": getattr(dataset, "trimer_cache_hash", None),
        "trimer_cache_artifact_hash": getattr(
            dataset, "trimer_cache_artifact_hash", None
        ),
    }
    resume_contract["cache_bundle_hash"] = cache_bundle_binding_hash(
        cohort_hash=resume_contract["source_cohort_hash"],
        topology_artifact_hash=resume_contract["topology_cache_artifact_hash"],
        trimer_artifact_hash=resume_contract["trimer_cache_artifact_hash"],
    )
    # The train-state identity includes every numerical/sampling choice that
    # can alter a resumed trajectory.  Cache identity alone is insufficient:
    # changing accumulation, AMP, the polynomial schedule, or the angle
    # objective must force a fresh run.
    resume_contract.update({
        "pretrain_profile": (
            str(pretrain_profile.get("profile_id"))
            if pretrain_profile is not None else str(args.pretrain_profile)
        ),
        "pretrain_profile_hash": (
            _canonical_json_hash(pretrain_profile)
            if pretrain_profile is not None else None
        ),
        "pretrain_code_hash": _pretrain_code_identity()["sha256"],
        "pretrain_code_files": _pretrain_code_identity()["files"],
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "lr": float(args.lr),
        "weight_decay": (
            0.0 if args.graph_encoder_type == MTS_ROUTE_INTERNAL
            else float(args.weight_decay)
        ),
        "adam_betas": [0.9, 0.98],
        "adam_eps": 1e-8,
        "warmup_steps": int(args.warmup_steps),
        "scheduler": str(args.mips_scheduler),
        "scheduler_power": float(args.scheduler_power),
        "end_lr": float(args.end_lr),
        "max_grad_norm": float(args.max_grad_norm),
        "amp_dtype": str(args.amp_dtype),
        "angle_bins": int(args.scage_angle_bins),
        "angle_focal_gamma": float(args.scage_focal_gamma),
        "angle_objective": str(args.angle_objective),
        "angle_cache_schema": getattr(
            dataset, "angle_cache_metadata", {}
        ).get("schema") if getattr(dataset, "angle_cache_metadata", None) else None,
        "angle_cache_artifact_hash": getattr(dataset, "angle_cache_artifact_hash", None),
        "masked_atom_weight": float(args.scage_mips_mask_weight),
        "trimer_bond_angle_weight": float(args.graph_angle_weight),
        "pretraining_objective": (
            getattr(args, "pretraining_objective", "joint")
            if args.pretrain_stage == MTS_STAGE1_ID else None
        ),
        "initialization_state": (
            initialization_info["sha256"] if initialization_info is not None else None
        ),
        "paired_init_id": (
            initialization_info["paired_init_id"] if initialization_info is not None else None
        ),
    })
    resume_epoch = 0
    resume_step_idx = 0
    global_step = 0
    optimizer_steps_completed = 0
    # Iterating over already-consumed batches to reconstruct the sampler
    # position can itself consume Python/NumPy/Torch/CUDA RNG state (for
    # example through dataset-side stochastic feature handling).  Keep the
    # checkpointed per-rank state and restore it again immediately before the
    # first batch that is actually optimized after a resume.
    resume_rng_state = None
    if args.resume_state:
        resume_path = os.path.abspath(args.resume_state)
        if not os.path.isfile(resume_path):
            raise RuntimeError(f"resume state does not exist: {resume_path}")
        # This is an internally generated, identity-bound train-state file.
        # PyTorch 2.6+ defaults to ``weights_only=True``; that mode cannot
        # deserialize the NumPy/RNG state captured for exact trajectory
        # resumption.  The resume contract is validated immediately below,
        # and the file is only accepted from the explicit user-provided path.
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if _is_t1_function_preserving_init_payload(resume_payload):
            raise RuntimeError(
                "T1 function-preserving init is an architecture warm start and "
                "cannot be used as --resume_state"
            )
        if resume_payload.get("schema") != PRETRAIN_TRAIN_STATE_SCHEMA:
            raise RuntimeError(
                "resume state uses an obsolete checkpoint schema; rerun from "
                "the latest stage checkpoint"
            )
        resume_meta = resume_payload.get("meta", {})
        if (
            resume_meta.get("pretrain_profile") != resume_contract["pretrain_profile"]
            or resume_meta.get("pretrain_profile_hash")
                != resume_contract["pretrain_profile_hash"]
            or resume_meta.get("pretrain_code_hash")
                != resume_contract["pretrain_code_hash"]
            or resume_meta.get("pretrain_code_files", {})
                != resume_contract["pretrain_code_files"]
            or resume_meta.get("config_schema") != args.config_schema
            or resume_meta.get("stage") != args.checkpoint_stage
            or resume_meta.get("topology_representation")
                != args.topology_representation
            or resume_meta.get("feature_config_hash") != args.feature_config_hash
            or resume_meta.get("o8_feature_config_hash")
                != args.o8_feature_config_hash
            or resume_meta.get("graph_model_config_hash")
                != args.graph_model_config_hash
            or resume_meta.get("g_family_bundle_hash")
                != resume_contract["g_family_bundle_hash"]
            or resume_meta.get("relation_geometry_artifact_hash")
                != resume_contract["relation_geometry_artifact_hash"]
            or resume_meta.get("g3_permutation_artifact_hash")
                != resume_contract["g3_permutation_artifact_hash"]
            or resume_meta.get("topology_attention_variant", "o8")
                != args.topology_attention_variant
            or list(resume_meta.get("msta_layer_indices", [4, 5]))
                != list(args.msta_layer_indices)
            or list(resume_meta.get("msta_local_spd", [0, 1]))
                != list(args.msta_local_spd)
            or list(resume_meta.get("msta_context_spd", [0, 1, 2]))
                != list(args.msta_context_spd)
            or resume_meta.get("geometry_model_config_hash", "manual")
                != args.geometry_model_config_hash
            or resume_meta.get(
                "source_geometry_model_config_hash",
                resume_meta.get("geometry_model_config_hash", "manual"),
            ) != resume_contract["source_geometry_model_config_hash"]
            or resume_meta.get("training_config_hash")
                != args.training_config_hash
            or resume_meta.get("pretraining_dataset") != args.dataset_name
            or resume_meta.get("source_cohort_hash")
                != resume_contract["source_cohort_hash"]
            or resume_meta.get("topology_cache_hash")
                != resume_contract["topology_cache_hash"]
            or resume_meta.get("topology_cache_artifact_hash")
                != resume_contract["topology_cache_artifact_hash"]
            or resume_meta.get("trimer_cache_hash")
                != resume_contract["trimer_cache_hash"]
            or resume_meta.get("trimer_cache_artifact_hash")
                != resume_contract["trimer_cache_artifact_hash"]
            or resume_meta.get("cache_bundle_hash")
                != resume_contract["cache_bundle_hash"]
            or int(resume_meta.get("batch_size", -1))
                != int(resume_contract["batch_size"])
            or int(resume_meta.get("gradient_accumulation_steps", -1))
                != int(resume_contract["gradient_accumulation_steps"])
            or float(resume_meta.get("lr", float("nan")))
                != float(resume_contract["lr"])
            or float(resume_meta.get("weight_decay", float("nan")))
                != float(resume_contract["weight_decay"])
            or list(resume_meta.get("adam_betas", []))
                != list(resume_contract["adam_betas"])
            or float(resume_meta.get("adam_eps", float("nan")))
                != float(resume_contract["adam_eps"])
            or int(resume_meta.get("warmup_steps", -1))
                != int(resume_contract["warmup_steps"])
            or resume_meta.get("scheduler")
                != resume_contract["scheduler"]
            or float(resume_meta.get("scheduler_power", float("nan")))
                != float(resume_contract["scheduler_power"])
            or float(resume_meta.get("end_lr", float("nan")))
                != float(resume_contract["end_lr"])
            or float(resume_meta.get("max_grad_norm", float("nan")))
                != float(resume_contract["max_grad_norm"])
            or resume_meta.get("amp_dtype") != resume_contract["amp_dtype"]
            or resume_meta.get("angle_cache_schema")
                != resume_contract["angle_cache_schema"]
            or resume_meta.get("angle_cache_artifact_hash")
                != resume_contract["angle_cache_artifact_hash"]
            or int(resume_meta.get("angle_bins", -1))
                != int(resume_contract["angle_bins"])
            or float(resume_meta.get("angle_focal_gamma", float("nan")))
                != float(resume_contract["angle_focal_gamma"])
            or float(resume_meta.get("masked_atom_weight", float("nan")))
                != float(resume_contract["masked_atom_weight"])
            or float(resume_meta.get("trimer_bond_angle_weight", float("nan")))
                != float(resume_contract["trimer_bond_angle_weight"])
            or resume_meta.get("pretraining_objective")
                != resume_contract["pretraining_objective"]
            or int(resume_meta.get("target_optimizer_steps", -1))
                != int(args.max_optimizer_steps)
            or int(resume_meta.get("epoch_cap", -1)) != int(args.epochs)
            or int(resume_meta.get("loader_workers", -1))
                != int(args.loader_workers)
            or int(resume_meta.get("loader_prefetch_factor", -1))
                != int(args.loader_prefetch_factor)
            or resume_meta.get("sampler_type")
                != (type(sampler).__name__ if sampler is not None else "random")
            or resume_meta.get("batch_balance", "none")
                != str(args.batch_balance)
            or resume_meta.get("optimizer_impl") != "adam"
            or resume_meta.get("matmul_precision")
                != ("high_tf32" if device.type == "cuda" else "default")
        ):
            raise RuntimeError("resume state does not match the requested run")
        train_module.load_state_dict(resume_payload["train_module"], strict=True)
        if "aux_modules" in resume_payload:
            aux_modules.load_state_dict(resume_payload["aux_modules"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        resume_epoch = int(resume_payload.get("epoch", 0))
        resume_step_idx = int(resume_payload.get("next_step_idx", 0))
        global_step = int(resume_payload.get("global_step", 0))
        optimizer_steps_completed = int(
            resume_payload.get("optimizer_steps_completed", 0)
        )
        rng_by_rank = resume_payload.get("rng_state_by_rank")
        if rng_by_rank is None:
            # v1 checkpoints had only rank zero state.  They remain readable
            # for single-process diagnostics, but distributed resume is not
            # silently treated as exact.
            if distributed:
                raise RuntimeError(
                    "distributed resume requires rng_state_by_rank; "
                    "the checkpoint was created by the old single-RNG format"
                )
            _restore_rng_state(resume_payload.get("rng_state"))
        else:
            if int(rank) >= len(rng_by_rank):
                raise RuntimeError("checkpoint has no RNG state for this rank")
            resume_rng_state = rng_by_rank[int(rank)]
            _restore_rng_state(resume_rng_state)
        loader_states = resume_payload.get("loader_generator_state_by_rank")
        if loader_states is None:
            # A rank-zero loader state is not sufficient for a distributed
            # exact resume once workers are enabled: each rank has its own
            # worker-seeding stream.  Refuse the old format instead of
            # silently changing the masking/augmentation sequence.
            if distributed:
                raise RuntimeError(
                    "distributed resume requires loader_generator_state_by_rank"
                )
            loader_state = resume_payload.get("loader_generator_state")
        else:
            if int(rank) >= len(loader_states):
                raise RuntimeError(
                    "checkpoint has no DataLoader generator state for this rank"
                )
            loader_state = loader_states[int(rank)]
        if loader_state is not None:
            loader_generator.set_state(loader_state)
        sampler_states = resume_payload.get("sampler_state_by_rank")
        if distributed:
            if sampler_states is None or int(rank) >= len(sampler_states):
                raise RuntimeError(
                    "distributed resume requires sampler_state_by_rank"
                )
            saved_sampler = sampler_states[int(rank)] or {}
            if int(saved_sampler.get("world_size", -1)) != int(world_size):
                raise RuntimeError("resume sampler world_size mismatch")
            if int(saved_sampler.get("next_batch_index", -1)) != int(resume_step_idx):
                raise RuntimeError("resume sampler/batch position mismatch")
        if rank == 0:
            print(
                f"Resumed pretraining at epoch={resume_epoch}, "
                f"step={optimizer_steps_completed}"
            )

    train_state_path = (
        os.path.abspath(args.resume_state)
        if args.resume_state else os.path.abspath(args.save_path + ".last.pt")
    )

    if args.benchmark_only:
        if args.pretrain_stage != MTS_STAGE1_ID or not mips_ddp_route:
            raise ValueError("benchmark_only is currently supported for MIPS pretraining")
        benchmark_batches = int(args.benchmark_batches)
        if benchmark_batches <= 0:
            raise ValueError("--benchmark_batches must be positive")
        train_module.train()
        iterator = iter(dataloader)
        warmup_batches = 50
        total_batches = warmup_batches + benchmark_batches
        timings = []
        data_timings = []
        h2d_timings = []
        forward_timings = []
        backward_timings = []
        optimizer_timings = []
        sample_count = 0
        optimizer_steps = 0
        accumulation = max(1, int(args.gradient_accumulation_steps))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        finite_loss = True
        finite_gradient = True
        finite_parameters = True
        for batch_index in range(total_batches):
            data_started = time.monotonic()
            try:
                data = next(iterator)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(getattr(sampler, "epoch", 0) + 1)
                iterator = iter(dataloader)
                data = next(iterator)
            data_elapsed = time.monotonic() - data_started
            transfer_started = time.monotonic()
            data = data.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_elapsed = time.monotonic() - transfer_started
            started = time.monotonic()
            should_step = (batch_index + 1) % accumulation == 0
            sync_context = (
                train_module.no_sync()
                if distributed and not should_step else nullcontext()
            )
            with sync_context:
                forward_started = time.monotonic()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=args.amp_dtype == "bf16" and device.type == "cuda",
                ):
                    payload = train_module(MTS_STAGE1_ID, data, args, batch_index)
                    priors = {
                        "masked_atom": args.scage_mips_mask_weight,
                        "angle": args.graph_angle_weight,
                    }
                    loss = (
                        priors["masked_atom"]
                        * payload["loss_terms"]["masked_atom_sum"]
                        / max(1, payload["counts"]["masked_atoms"])
                        + priors["angle"]
                        * payload["loss_terms"]["angle_sum"]
                        / max(1, payload["counts"]["angle_graphs"])
                    )
                    loss = loss + payload["zero_reference"]
                finite_loss = finite_loss and bool(torch.isfinite(loss.detach()))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_elapsed = time.monotonic() - forward_started
                backward_started = time.monotonic()
                (loss / accumulation).backward()
                if batch_index + 1 == total_batches:
                    finite_gradient = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in train_module.parameters()
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                backward_elapsed = time.monotonic() - backward_started
            if should_step:
                optimizer_started = time.monotonic()
                if float(args.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(train_module.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                optimizer_elapsed = time.monotonic() - optimizer_started
                if batch_index + 1 == total_batches:
                    finite_parameters = all(
                        bool(torch.isfinite(parameter).all())
                        for parameter in train_module.parameters()
                    )
                optimizer_steps += int(batch_index >= warmup_batches)
            else:
                optimizer_elapsed = 0.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            if batch_index >= warmup_batches:
                timings.append(elapsed + data_elapsed + transfer_elapsed)
                data_timings.append(data_elapsed)
                h2d_timings.append(transfer_elapsed)
                forward_timings.append(forward_elapsed)
                backward_timings.append(backward_elapsed)
                optimizer_timings.append(optimizer_elapsed)
                sample_count += int(data.graph_available.numel())
        local_elapsed = float(sum(timings))
        finite_tensor = torch.tensor(
            [finite_loss, finite_gradient, finite_parameters],
            device=device, dtype=torch.int32,
        )
        if distributed:
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        local_compute = torch.tensor(
            [local_elapsed, local_elapsed], device=device, dtype=torch.float64
        )
        if distributed:
            maximum = local_compute[:1].clone()
            minimum = local_compute[1:].clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            elapsed = float(maximum.item())
            rank_wait_fraction = max(0.0, (elapsed - float(minimum.item())) / max(elapsed, 1e-9))
        else:
            elapsed = local_elapsed
            rank_wait_fraction = 0.0
        if rank == 0:
            effective_samples = sample_count * max(1, int(world_size))
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0
            )
            print(json.dumps({
                "benchmark_batches": benchmark_batches,
                "warmup_batches": warmup_batches,
                "batch_size_per_rank": int(args.batch_size),
                "world_size": int(world_size),
                "elapsed_seconds": elapsed,
                "samples_per_second": effective_samples / max(elapsed, 1e-9),
                "optimizer_steps_per_second": optimizer_steps / max(elapsed, 1e-9),
                "mean_data_seconds": float(np.mean(data_timings)),
                "mean_h2d_seconds": float(np.mean(h2d_timings)),
                "mean_forward_seconds": float(np.mean(forward_timings)),
                "mean_backward_ddp_seconds": float(np.mean(backward_timings)),
                "mean_optimizer_seconds": float(np.mean(optimizer_timings)),
                "rank_wait_fraction": rank_wait_fraction,
                "loss_finite": bool(finite_tensor[0].item()),
                "gradient_finite": bool(finite_tensor[1].item()),
                "parameters_finite": bool(finite_tensor[2].item()),
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
                "peak_memory_fraction": (
                    peak_reserved / float(torch.cuda.get_device_properties(device).total_memory)
                    if device.type == "cuda" else 0.0
                ),
                "loader_workers": int(args.loader_workers),
                "loader_prefetch_factor": int(args.loader_prefetch_factor),
                "batch_balance": str(args.batch_balance),
                "stage": args.checkpoint_stage,
            }, sort_keys=True))
        if distributed:
            dist.barrier()
            # The benchmark is an intentional short-lived DDP job.  Tear
            # down the process group explicitly so PyTorch does not emit a
            # misleading leaked-process-group warning at interpreter exit.
            dist.destroy_process_group()
        return

    def save_train_state(epoch_number, next_step_idx):
        local_rng_state = _capture_rng_state()
        local_loader_state = loader_generator.get_state()
        rng_state_by_rank = _gather_rng_states(
            local_rng_state, distributed, rank, world_size
        )
        loader_state_by_rank = _gather_rank_states(
            local_loader_state, distributed, rank, world_size
        )
        sampler_state_by_rank = _gather_rank_states(
            {
                "epoch": int(epoch_number),
                "next_batch_index": int(next_step_idx),
                "seed": int(args.seed),
                "world_size": int(world_size),
            },
            distributed,
            rank,
            world_size,
        )
        if rank == 0:
            payload = {
                "schema": PRETRAIN_TRAIN_STATE_SCHEMA,
                "meta": {
                    **resume_contract,
                    "target_optimizer_steps": int(args.max_optimizer_steps),
                    "epoch_cap": int(args.epochs),
                    "random_seed": int(args.seed),
                    "loader_workers": int(args.loader_workers),
                    "loader_prefetch_factor": int(args.loader_prefetch_factor),
                    "sampler_type": (
                        type(sampler).__name__ if sampler is not None else "random"
                    ),
                    "batch_balance": str(args.batch_balance),
                    "warmup_ratio": float(args.warmup_ratio),
                    "optimizer_impl": "adam",
                    "matmul_precision": (
                        "high_tf32" if device.type == "cuda" else "default"
                    ),
                },
                "train_module": {
                    key: value.detach().cpu()
                    for key, value in train_module.state_dict().items()
                },
                "aux_modules": {
                    key: value.detach().cpu()
                    for key, value in aux_modules.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": int(epoch_number),
                "next_step_idx": int(next_step_idx),
                "global_step": int(global_step),
                "optimizer_steps_completed": int(optimizer_steps_completed),
                "rng_state_by_rank": rng_state_by_rank,
                "loader_generator_state_by_rank": loader_state_by_rank,
                "sampler_state_by_rank": sampler_state_by_rank,
            }
            _atomic_torch_save(payload, train_state_path)
        # Checkpointing must be transparent to every rank's stochastic stream.
        # Synchronize the atomic rank-zero write, then restore the exact local
        # states captured before object collectives and serialization.
        if distributed:
            dist.barrier()
        _restore_rng_state(local_rng_state)
        loader_generator.set_state(local_loader_state)

    def save_categorical_milestone(step):
        """Persist an immutable, optimizer-free graph milestone."""
        if rank != 0 or pretrain_profile is None:
            return
        base_state = {
            key: value.detach().cpu().clone()
            for key, value in _base_model(model).state_dict().items()
        }
        head_owner = train_module.module if hasattr(train_module, "module") else train_module
        heads = {
            key: value.detach().cpu().clone()
            for key, value in head_owner.heads.state_dict().items()
        }
        milestone = Path(args.save_path).with_name(
            Path(args.save_path).stem + f".step_{int(step):05d}.pth"
        )
        _atomic_torch_save({
            "schema": "mts-pretrain-milestone-v1",
            "checkpoint_schema": PRETRAIN_CHECKPOINT_SCHEMA,
            "optimizer_step": int(step),
            "profile_id": str(pretrain_profile["profile_id"]),
            "profile_sha256": _canonical_json_hash(pretrain_profile),
            "pretrain_code_identity": _pretrain_code_identity(),
            "resume_contract": dict(resume_contract),
            "state_dict": base_state,
            "heads": heads,
        }, milestone)

    if args.dynamic_pretrain_loss:
        active_loss_names = []
        task_priors = None
        if args.pretrain_stage == 'scage_m4p':
            active_loss_names = [
                name for name, weight in (
                    ('mips_mask', args.scage_mips_mask_weight),
                    ('ecfp', args.scage_ecfp_weight),
                    ('masked_spd', args.mips_spd_weight),
                    ('path_bond', args.mips_path_bond_weight),
                    ('finite_distance', args.mips_distance_weight),
                    ('repeat_consistency', args.mips_repeat_consistency_weight),
                ) if float(weight) > 0
            ]
            task_priors = {
                'mips_mask': args.scage_mips_mask_weight,
                'ecfp': args.scage_ecfp_weight,
                'masked_spd': args.mips_spd_weight,
                'path_bond': args.mips_path_bond_weight,
                'finite_distance': args.mips_distance_weight,
                'repeat_consistency': args.mips_repeat_consistency_weight,
            }
        elif args.pretrain_stage == 'graph_geom':
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                if float(args.graph_mask_atom_weight) > 0:
                    active_loss_names.append('graph_mask')
                if float(args.graph_periodic_aug_weight) > 0:
                    active_loss_names.append('graph_periodic_aug')
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    active_loss_names.append('shortest_path')
                if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                    active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        else:
            if loss_weights['contrastive'] > 0:
                active_loss_names.append('contrastive')
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                active_loss_names.append('graph')
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                active_loss_names.append('shortest_path')
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        dynamic_loss_weighter = DynamicPretrainLossWeighter(
            active_loss_names,
            init_window=args.dynamic_loss_warmup_steps,
            recent_window=args.dynamic_loss_recent_window,
            temperature=args.dynamic_loss_temperature,
            task_priors=task_priors,
            effective_caps={'ecfp': args.scage_ecfp_effective_max},
            device=device,
        )
        print(f"Dynamic pretraining loss terms: {active_loss_names}")

    diagnostics_enabled = bool(args.diagnostics_dir)
    diagnostic_steps = set()
    diagnostic_rows = []
    diagnostic_probe = None
    diagnostic_jsonl = None
    if diagnostics_enabled:
        try:
            diagnostic_steps = {
                int(value.strip()) for value in str(args.diagnostic_steps).split(",")
                if value.strip()
            }
        except ValueError as exc:
            raise ValueError("--diagnostic_steps must be comma-separated integers") from exc
        if min(diagnostic_steps, default=0) < 0:
            raise ValueError("diagnostic steps must be non-negative")
        diagnostic_root = Path(args.diagnostics_dir).resolve()
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        diagnostic_jsonl = diagnostic_root / "milestones.jsonl"
        if rank == 0 and diagnostic_jsonl.exists():
            raise RuntimeError(
                f"diagnostics output already exists; refusing overwrite: {diagnostic_jsonl}"
            )
        _set_msta_diagnostic_mode(model, capture=False, local_off=False)
        if rank == 0:
            (diagnostic_root / "contract.json").write_text(
                json.dumps({
                    "schema": "mts-msta-branch-diagnostics-v1",
                    "model_identity": (
                        "T1" if args.topology_attention_variant == "msta_last2" else "T0"
                    ),
                    "steps": sorted(diagnostic_steps),
                    "no_extra_backward": True,
                    "probe_mode": "eval_no_grad_fixed_first_two_graphs",
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    model.train()
    angle_v2_validation_history = []
    angle_v2_validation_reference = None
    angle_v2_best_composite = float('inf')
    angle_v2_best_model_state = None
    joint_progress = None
    if args.pretrain_stage == MTS_STAGE1_ID and rank == 0:
        joint_progress = tqdm(
            total=int(args.max_optimizer_steps),
            initial=int(optimizer_steps_completed),
            desc="MTS Joint Pretraining",
            unit="step",
            disable=not (
                sys.stderr.isatty()
                and os.environ.get("MIPS_TQDM", "1") == "1"
            ),
        )
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if epoch < resume_epoch:
            continue
        first_step_idx = resume_step_idx if epoch == resume_epoch else 0
        epoch_loss = 0.0
        epoch_periodic_aug_stats = _empty_periodic_aug_stats()
        epoch_term_steps = 0
        show_progress = (
            rank == 0
            and sys.stderr.isatty()
            and os.environ.get("MIPS_TQDM", "1") == "1"
            and args.pretrain_stage != MTS_STAGE1_ID
        )
        progress_bar = tqdm(
            dataloader,
            desc=f"Pretraining Epoch {epoch + 1}/{args.epochs}",
            disable=not show_progress,
        )
        optimizer.zero_grad(set_to_none=True)
        structured_log_time = time.monotonic()
        structured_log_step = optimizer_steps_completed
        # Reconstruct the iterator position without fetching the first
        # optimized batch under a stale RNG state.  The collate function may
        # sample a 3D conformer with torch.randint(); restoring only after
        # that batch has been fetched changes the input (and can consume a
        # different number of random draws inside the loss).  Consume the
        # skipped batches first, restore the checkpointed per-rank RNG, then
        # fetch the first actual optimization batch.
        direct_sampler_resume = bool(
            first_step_idx and hasattr(sampler, "set_start_batch")
        )
        if direct_sampler_resume:
            sampler.set_start_batch(first_step_idx)
        data_iterator = iter(progress_bar)
        if first_step_idx and not direct_sampler_resume:
            for _ in range(first_step_idx):
                try:
                    next(data_iterator)
                except StopIteration:
                    break
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
            resume_rng_state = None
        for step_idx, data in enumerate(data_iterator, start=first_step_idx):
            if (
                (args.max_steps > 0 and global_step >= args.max_steps)
                or (
                    args.max_optimizer_steps > 0
                    and optimizer_steps_completed >= args.max_optimizer_steps
                )
            ):
                break
            global_step += 1
            data = data.to(device, non_blocking=True)
            trace_record = None
            if (
                args.pretrain_stage == MTS_STAGE1_ID
                and rank == 0
                and os.environ.get("MTS_RESUME_TRACE")
            ):
                mask_preview = _joint_canonical_mask(
                    data, args.seed, global_step, args.graph_mask_ratio
                )
                trace_record = {
                    "global_step": int(global_step),
                    "optimizer_step": int(
                        optimizer_steps_completed + (1 if ((step_idx + 1) % accumulation_steps == 0) else 0)
                    ),
                    "sample_hash": hashlib.sha256(
                        json.dumps([str(value) for value in data.smiles], sort_keys=False).encode()
                    ).hexdigest(),
                    "mask_hash": hashlib.sha256(
                        mask_preview.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            should_step = (
                (step_idx + 1) % accumulation_steps == 0
                or (step_idx + 1) == len(dataloader)
                or (args.max_steps > 0 and global_step >= args.max_steps)
            )
            diagnostic_step = None
            if diagnostics_enabled:
                if optimizer_steps_completed in diagnostic_steps:
                    diagnostic_step = int(optimizer_steps_completed)
                elif should_step and optimizer_steps_completed + 1 in diagnostic_steps:
                    diagnostic_step = int(optimizer_steps_completed + 1)
            _set_msta_diagnostic_mode(
                model,
                capture=bool(diagnostics_enabled and diagnostic_step is not None),
                local_off=False,
            )
            base_model = _base_model(model)
            loss_terms = {}

            if args.pretrain_stage == MTS_STAGE1_ID:
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            MTS_STAGE1_ID, data, args, global_step
                        )
                        global_counts = _distributed_joint_counts(
                            payload["counts"], device
                        )
                        global_atom_count = global_counts["masked_atoms"]
                        global_angle_count = global_counts["angle_graphs"]
                        global_masked_correct = global_counts["masked_correct"]
                        global_angle_correct = global_counts["angle_correct"]
                        global_angle_targets = global_counts["angle_targets"]
                        global_mcl_valid = global_counts["mcl_valid_graphs"]
                        global_graphs = global_counts["graphs"]
                        atom_mean = _global_mean_from_count(
                            payload["loss_terms"]["masked_atom_sum"],
                            global_atom_count,
                        )
                        angle_mean = _global_mean_from_count(
                            payload["loss_terms"]["angle_sum"],
                            global_angle_count,
                        )
                        loss = (
                            float(args.scage_mips_mask_weight) * atom_mean
                            + float(args.graph_angle_weight) * angle_mean
                            + payload["zero_reference"]
                        )
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite MTS joint loss for batch smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                if diagnostics_enabled and rank == 0 and diagnostic_step is not None:
                    if diagnostic_probe is None:
                        diagnostic_probe = _make_fixed_probe(data)
                    owner = (
                        train_module.module
                        if isinstance(train_module, torch.nn.parallel.DistributedDataParallel)
                        else train_module
                    )
                    record = _diagnostic_record(
                        base_model,
                        diagnostic_step,
                        diagnostic_probe,
                        args,
                        device,
                        owner,
                    )
                    diagnostic_rows.append(record)
                    with diagnostic_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                if trace_record is not None:
                    trace_record.update({
                        "loss": float(loss.detach().float().item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    })
                    with open(os.environ["MTS_RESUME_TRACE"], "a", encoding="utf-8") as trace_handle:
                        trace_handle.write(json.dumps(trace_record, sort_keys=True) + "\n")
                gradient_norm = loss.new_tensor(0.0)
                gradient_diag = None
                if should_step:
                    # Clipping is disabled for the canonical profile, but the
                    # diagnostic norm must still reflect the actual gradient.
                    gradient_norm = loss.new_tensor(_module_gradient_norm(train_module))
                    if (
                        optimizer_steps_completed < 3
                        or (optimizer_steps_completed + 1) % 50 == 0
                    ):
                        graph_encoder = _base_model(model).encoders["graph"].encoder
                        gradient_diag = {
                            "o8": _module_gradient_diagnostic(graph_encoder.layers),
                            "star_rbf": _module_gradient_diagnostic(graph_encoder.star_distance_bias),
                            "trimer_mcl": _module_gradient_diagnostic(graph_encoder.trimer_mcl),
                            "geometry_gate": _module_gradient_diagnostic(
                                getattr(graph_encoder.trimer_mcl, "geometry_gate", None)
                            ),
                            "masked_atom_head": _module_gradient_diagnostic(mips_atom_head),
                            "angle_head": _module_gradient_diagnostic(graph_angle_head),
                        }
                    if float(args.max_grad_norm) > 0:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            train_module.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if joint_progress is not None:
                        joint_progress.update(1)
                        joint_progress.set_postfix(
                            loss=f"{loss.item():.4f}",
                            atom=f"{atom_mean.item():.3f}",
                            angle=f"{angle_mean.item():.3f}",
                            acc=f"{global_masked_correct/max(1, global_atom_count):.2%}",
                            aacc=f"{global_angle_correct/max(1, global_angle_targets):.2%}",
                            mcl=f"{global_mcl_valid/max(1, global_graphs):.1%}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        )
                    if (
                        rank == 0 and not show_progress and
                        (optimizer_steps_completed <= 3
                         or optimizer_steps_completed % 50 == 0
                         or optimizer_steps_completed >= args.max_optimizer_steps)
                    ):
                        elapsed = max(1e-6, time.monotonic() - structured_log_time)
                        delta_steps = max(1, optimizer_steps_completed - structured_log_step)
                        effective_samples = (
                            delta_steps * int(args.batch_size)
                            * max(1, int(world_size)) * accumulation_steps
                        )
                        total_elapsed = max(1e-6, time.monotonic() - pretrain_started)
                        eta_seconds = total_elapsed * max(
                            0.0,
                            float(args.max_optimizer_steps)
                            / max(1, optimizer_steps_completed) - 1.0,
                        )
                        peak_gib = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda" else 0.0
                        )
                        print(
                            "[pretrain] "
                            f"stage={MTS_STAGE1_ID} "
                            f"step={optimizer_steps_completed}/{args.max_optimizer_steps} "
                            f"samples_per_s={effective_samples / elapsed:.2f} "
                            f"loss={float(loss.detach().float().item()):.5f} "
                            f"atom={float(atom_mean.detach().float().item()):.5f} "
                            f"angle={float(angle_mean.detach().float().item()):.5f} "
                            f"atom_acc={global_masked_correct/max(1, global_atom_count):.4f} "
                            f"angle_acc={global_angle_correct/max(1, global_angle_targets):.4f} "
                            f"angle_targets={global_angle_targets} "
                            f"angle_majority={getattr(args, 'angle_majority_baseline', 0.0):.4f} "
                            f"mcl_rate={global_mcl_valid/max(1, global_graphs):.4f} "
                            f"grad_norm={float(gradient_norm.detach().float().item()):.6g} "
                            f"peak_mem_gib={peak_gib:.2f} "
                            f"eta_h={eta_seconds/3600.0:.2f}",
                            flush=True,
                        )
                        if gradient_diag is not None:
                            print(
                                "[pretrain-gradient] "
                                + json.dumps({
                                    "step": int(optimizer_steps_completed),
                                    "modules": gradient_diag,
                                }, sort_keys=True),
                                flush=True,
                            )
                        structured_log_time = time.monotonic()
                        structured_log_step = optimizer_steps_completed
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or optimizer_steps_completed >= args.max_optimizer_steps
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        args.checkpoint_interval_steps > 0
                        and (
                            optimizer_steps_completed % args.checkpoint_interval_steps == 0
                            or optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    ):
                        next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                        next_step = 0 if next_epoch > epoch else step_idx + 1
                        save_train_state(next_epoch, next_step)
                    if (
                        pretrain_profile is not None
                        and optimizer_steps_completed > 0
                        and optimizer_steps_completed % 2000 == 0
                    ):
                        save_categorical_milestone(optimizer_steps_completed)
                    if (
                        args.angle_objective == 'cosine'
                        and optimizer_steps_completed > 0
                        and optimizer_steps_completed % 2000 == 0
                    ):
                        atom_val, angle_val = _evaluate_angle_v2_validation(
                            train_module, angle_validation_loader, args, device
                        )
                        if angle_v2_validation_reference is None:
                            angle_v2_validation_reference = (
                                max(atom_val, 1e-12), max(angle_val, 1e-12)
                            )
                        composite = (
                            atom_val / angle_v2_validation_reference[0]
                            + angle_val / angle_v2_validation_reference[1]
                        )
                        if rank == 0:
                            record = {
                                'optimizer_step': int(optimizer_steps_completed),
                                'masked_atom_ce': float(atom_val),
                                'angle_cos_mae': float(angle_val),
                                'normalized_composite': float(composite),
                            }
                            angle_v2_validation_history.append(record)
                            milestone = (
                                Path(args.save_path).with_suffix('')
                                .with_name(
                                    Path(args.save_path).stem
                                    + f'.step_{optimizer_steps_completed:05d}.pth'
                                )
                            )
                            _atomic_torch_save({
                                'state_dict': _base_model(model).state_dict(),
                                'heads': (
                                    train_module.module.heads.state_dict()
                                    if isinstance(
                                        train_module,
                                        torch.nn.parallel.DistributedDataParallel,
                                    ) else train_module.heads.state_dict()
                                ),
                                'validation': record,
                                'angle_objective': 'cosine',
                            }, milestone)
                            if composite < angle_v2_best_composite:
                                angle_v2_best_composite = float(composite)
                                angle_v2_best_model_state = {
                                    key: value.detach().cpu().clone()
                                    for key, value in _base_model(model).state_dict().items()
                                }
                epoch_loss += float(loss.detach().item())
                epoch_term_steps += 1
                if show_progress:
                    progress_bar.set_postfix(
                        step=f"{optimizer_steps_completed}/{args.max_optimizer_steps}",
                        loss=f"{loss.item():.4f}",
                        atom=f"{atom_mean.item():.3f}",
                        angle=f"{angle_mean.item():.3f}",
                    )
                continue

            if args.pretrain_stage == 'scage_m4p':
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            (
                                "stage2_geometry_adapt"
                                if args.geometry_adapt else "mips_stage1"
                            ),
                            data, args, epoch,
                        )
                    loss_terms = payload["loss_terms"]
                    zero_reference = payload["zero_reference"]
                    if dynamic_loss_weighter is not None:
                        loss = dynamic_loss_weighter(
                            loss_terms, reference=zero_reference
                        )
                        dynamic_weights = dynamic_loss_weighter.last_weights
                    else:
                        priors = {
                            'mips_mask': args.scage_mips_mask_weight,
                            'masked_spd': args.mips_spd_weight,
                            'path_bond': args.mips_path_bond_weight,
                        }
                        loss = (
                            sum(
                                priors[name] * value
                                for name, value in loss_terms.items()
                            )
                            if loss_terms else zero_reference
                        )
                        dynamic_weights = {}
                    # Include inactive-head zero anchors in the scalar loss;
                    # otherwise DDP cannot see those parameters even though
                    # they are part of the fixed container.
                    loss = loss + payload["zero_reference"]
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite MIPS Stage 1 loss for batch "
                            f"smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        train_container.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if (
                        rank == 0
                        and not show_progress
                        and (
                            optimizer_steps_completed == 1
                            or optimizer_steps_completed % 50 == 0
                            or (
                                args.max_optimizer_steps > 0
                                and optimizer_steps_completed >= args.max_optimizer_steps
                            )
                        )
                    ):
                        elapsed = max(1e-6, time.monotonic() - structured_log_time)
                        delta_steps = optimizer_steps_completed - structured_log_step
                        effective_samples = (
                            delta_steps * int(args.batch_size)
                            * max(1, int(world_size)) * accumulation_steps
                        )
                        print(
                            "[pretrain] "
                            f"stage={args.checkpoint_stage} "
                            f"step={optimizer_steps_completed}/{args.max_optimizer_steps} "
                            f"samples_per_s={effective_samples / elapsed:.2f} "
                            f"loss={float(loss.detach().float().item()):.5f}",
                            flush=True,
                        )
                        structured_log_time = time.monotonic()
                        structured_log_step = optimizer_steps_completed
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        args.checkpoint_interval_steps > 0
                        and (
                            optimizer_steps_completed
                            % args.checkpoint_interval_steps == 0
                            or (
                                args.max_optimizer_steps > 0
                                and optimizer_steps_completed
                                >= args.max_optimizer_steps
                            )
                        )
                    ):
                        next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                        next_step = 0 if next_epoch > epoch else step_idx + 1
                        save_train_state(next_epoch, next_step)
                epoch_loss += float(loss.item())
                epoch_term_steps += 1
                if show_progress and (
                    should_step or optimizer_steps_completed % 50 == 0
                ):
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        mask=f"{payload['mask_loss'].item():.3f}",
                        sp=f"{payload['spd_loss'].item():.3f}",
                        path=f"{payload['path_loss'].item():.3f}",
                    )
                continue

            if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            "alignment", data, args, epoch
                        )
                        alignment_losses = payload["losses"]
                        loss = _alignment_total(alignment_losses, args)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite parallel alignment loss for batch "
                            f"smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.item())
                epoch_term_steps += 1
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    fusion=f"{alignment_losses['fused_view'].item():.3f}",
                    gs=f"{alignment_losses['graph_smiles'].item():.3f}",
                    gf=f"{alignment_losses['graph_fp'].item():.3f}",
                    lomo=f"{alignment_losses['lomo'].item():.3f}",
                )
                continue

            geom_loss = data.y.new_tensor(0.0)
            if geom_noise_head is not None:
                geom_loss = _geom_denoise_loss(
                    base_model,
                    data,
                    geom_noise_head,
                    noise_std=args.geom_noise_std,
                    noise_std_min=args.geom_noise_std_min,
                    noise_std_max=args.geom_noise_std_max,
                )
                loss_terms['geom'] = geom_loss

            contrastive_loss = data.y.new_tensor(0.0)
            if loss_weights['contrastive'] > 0:
                _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
                contrastive_loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
                loss_terms['contrastive'] = contrastive_loss

            graph_loss = data.y.new_tensor(0.0)
            if graph_atom_head is not None:
                graph_loss, graph_mask_loss, graph_paug_loss, graph_paug_stats = _graph_pretrain_loss(
                    base_model,
                    data,
                    graph_atom_head,
                    mask_ratio=args.graph_mask_ratio,
                    mask_weight=args.graph_mask_atom_weight,
                    periodic_aug_weight=args.graph_periodic_aug_weight,
                    graph_input=args.graph_input,
                    repeat_cut_max_mrus=args.repeat_cut_max_mrus,
                    repeat_cut_retry=args.repeat_cut_retry,
                    repeat_cut_temperature=args.repeat_cut_temperature,
                )
                _merge_periodic_aug_stats(epoch_periodic_aug_stats, graph_paug_stats)
                loss_terms['graph'] = graph_loss
            else:
                graph_mask_loss = data.y.new_tensor(0.0)
                graph_paug_loss = data.y.new_tensor(0.0)
                graph_paug_stats = _empty_periodic_aug_stats()

            graph_sp_loss = data.y.new_tensor(0.0)
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                graph_sp_loss = _graph_shortest_path_loss(
                    base_model,
                    data,
                    graph_sp_head,
                    max_distance=args.scage_sp_max_distance,
                )

            graph_angle_loss = None
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                graph_angle_loss = _graph_angle_loss(
                    base_model,
                    data,
                    graph_angle_head,
                    angle_bins=args.scage_angle_bins,
                )
            graph_angle_log = graph_angle_loss if graph_angle_loss is not None else data.y.new_tensor(0.0)

            weighted_loss_terms = {}
            if args.pretrain_stage == 'graph_geom':
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    if float(args.graph_mask_atom_weight) > 0:
                        weighted_loss_terms['graph_mask'] = float(args.graph_mask_atom_weight) * graph_mask_loss
                    if float(args.graph_periodic_aug_weight) > 0:
                        weighted_loss_terms['graph_periodic_aug'] = float(args.graph_periodic_aug_weight) * graph_paug_loss
                    if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                        weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                    if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                        weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss
            else:
                if loss_weights['contrastive'] > 0 and 'contrastive' in loss_terms:
                    weighted_loss_terms['contrastive'] = loss_weights['contrastive'] * contrastive_loss
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    weighted_loss_terms['graph'] = loss_weights['graph'] * graph_loss
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                    weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss

            if dynamic_loss_weighter is not None:
                loss = dynamic_loss_weighter(weighted_loss_terms)
                dynamic_weights = dynamic_loss_weighter.last_weights
            elif args.pretrain_stage == 'graph_geom' and weighted_loss_terms:
                loss = sum(weighted_loss_terms.values()) / len(weighted_loss_terms)
                dynamic_weights = {}
            else:
                loss = sum(weighted_loss_terms.values()) if weighted_loss_terms else data.y.new_tensor(0.0)
                dynamic_weights = {}


            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite pretraining loss for batch smiles={getattr(data, 'smiles', [])}")
            (loss / accumulation_steps).backward()
            gradient_norm = loss.new_tensor(0.0)
            if should_step:
                _all_reduce_gradients((model, aux_modules))
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(aux_modules.parameters()), max_norm=args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer_steps_completed += 1
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.checkpoint_interval_steps > 0
                    and (
                        optimizer_steps_completed
                        % args.checkpoint_interval_steps == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    )
                ):
                    next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                    next_step = 0 if next_epoch > epoch else step_idx + 1
                    save_train_state(next_epoch, next_step)
            epoch_loss += loss.item()
            epoch_term_steps += 1
            paug_attempted = int(graph_paug_stats.get('attempted', 0))
            paug_success = int(graph_paug_stats.get('success', 0))
            paug_rate = (paug_success / paug_attempted) if paug_attempted else 0.0
            postfix = {
                'loss': f"{loss.item():.4f}",
                'con': f"{contrastive_loss.item():.4f}",
                'g_mask': f"{graph_mask_loss.item():.4f}",
                'g_paug': f"{graph_paug_loss.item():.4f}",
                'paug_rate': f"{paug_rate:.2f}",
                'paug_valid': int(graph_paug_stats.get('valid_contrastive', 0)),
                'paug_same': int(graph_paug_stats.get('same_smiles', 0)),
                'geom': f"{geom_loss.item():.4f}",
                'sp': f"{graph_sp_loss.item():.4f}",
                'angle': f"{graph_angle_log.item():.4f}",
            }
            for name, value in dynamic_weights.items():
                postfix[f'w_{name}'] = f"{value:.2f}"
            progress_bar.set_postfix(**postfix)
        epoch_summary = torch.tensor(
            [float(epoch_loss), float(epoch_term_steps)], dtype=torch.float64, device=device
        )
        if distributed:
            dist.all_reduce(epoch_summary, op=dist.ReduceOp.SUM)
        global_epoch_steps = max(1, int(round(epoch_summary[1].item())))
        avg_loss = float(epoch_summary[0].item()) / global_epoch_steps
        epoch_periodic_aug_stats = _distributed_sum_mapping(
            epoch_periodic_aug_stats, device
        )
        paug_summary = _finalize_periodic_aug_stats(epoch + 1, epoch_periodic_aug_stats)
        if rank == 0 and paug_summary['attempted'] > 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f} | "
                f"PerioGT aug success: {paug_summary['success']}/{paug_summary['attempted']} "
                f"= {paug_summary['success_rate']:.2%}, "
                f"same_smiles={paug_summary['same_smiles']}, skipped={paug_summary['skipped']}"
            )
        elif rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f}")
        if (
            (args.max_steps > 0 and global_step >= args.max_steps)
            or (
                args.max_optimizer_steps > 0
                and optimizer_steps_completed >= args.max_optimizer_steps
            )
        ):
            break

    if joint_progress is not None:
        joint_progress.close()

    if args.resume_smoke:
        if diagnostics_enabled and rank == 0:
            diagnostic_root = Path(args.diagnostics_dir).resolve()
            (diagnostic_root / "report.json").write_text(
                json.dumps({
                    "schema": "mts-msta-branch-diagnostics-v1",
                    "model_identity": (
                        "T1" if args.topology_attention_variant == "msta_last2" else "T0"
                    ),
                    "requested_steps": sorted(diagnostic_steps),
                    "observed_steps": [int(row["optimizer_step"]) for row in diagnostic_rows],
                    "rows": diagnostic_rows,
                    "all_rows_finite": all(bool(row.get("finite")) for row in diagnostic_rows),
                    "probe_no_extra_backward": True,
                    "probe_rng_restored": True,
                    "smoke_only": True,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if getattr(args, "g_family_arm", None) is None:
            if distributed:
                dist.barrier()
                dist.destroy_process_group()
            return

    if distributed and rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.angle_objective == 'cosine':
        if angle_v2_best_model_state is None:
            raise RuntimeError('Angle-v2 produced no 2,000-step validation checkpoint')
        _base_model(model).load_state_dict(angle_v2_best_model_state, strict=True)
        print(
            'Angle-v2 export selected the minimum validation composite: '
            f'{angle_v2_best_composite:.6f}'
        )

    if diagnostics_enabled and rank == 0:
        diagnostic_root = Path(args.diagnostics_dir).resolve()
        diagnostic_report = {
            "schema": "mts-msta-branch-diagnostics-v1",
            "model_identity": (
                "T1" if args.topology_attention_variant == "msta_last2" else "T0"
            ),
            "requested_steps": sorted(diagnostic_steps),
            "observed_steps": [int(row["optimizer_step"]) for row in diagnostic_rows],
            "rows": diagnostic_rows,
            "all_rows_finite": all(bool(row.get("finite")) for row in diagnostic_rows),
            "probe_no_extra_backward": True,
            "probe_rng_restored": True,
        }
        (diagnostic_root / "report.json").write_text(
            json.dumps(diagnostic_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    state_dict = model.state_dict()
    if args.graph_encoder_type == 'mips_trimer_scage':
        source_csv = os.path.join(args.root, "raw", f"{args.dataset_name}.csv")
        source_data_hash = (
            _file_sha256(source_csv) if os.path.isfile(source_csv) else None
        )
        model_config = {
            "core": args.mips_core,
            "variant": args.mips_variant,
            "max_hops": int(args.mips_max_hops),
            "spatial_mode": args.spatial_mode,
            "graph_geometry_mode": args.graph_geometry_mode,
            "descriptors": bool(args.mips_use_descriptors),
            "modalities": list(args.modalities),
            "fusion_type": args.fusion_type,
            "topology_representation": args.topology_representation,
        }
        calculated_model_hash = hashlib.sha256(
            json.dumps(model_config, sort_keys=True).encode()
        ).hexdigest()
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
        except Exception:
            git_sha = "unknown"
        try:
            dirty_diff = subprocess.check_output(
                ["git", "diff", "--binary", "--", "."],
                cwd=PROJECT_ROOT,
            )
            dirty_digest = hashlib.sha256(dirty_diff)
            untracked = subprocess.check_output(
                [
                    "git", "ls-files", "--others", "--exclude-standard",
                    "--", "scripts", "src", "configs", "tests",
                    "AGENT.MD", "PIPELINE.MD",
                ],
                cwd=PROJECT_ROOT, text=True,
            ).splitlines()
            for relative in sorted(untracked):
                dirty_digest.update(relative.encode("utf-8"))
                with open(
                    os.path.join(PROJECT_ROOT, relative), "rb"
                ) as handle:
                    for block in iter(
                        lambda: handle.read(1024 * 1024), b""
                    ):
                        dirty_digest.update(block)
            dirty_diff_sha = dirty_digest.hexdigest()
        except Exception:
            dirty_diff_sha = "unknown"
        tokenizer_hash = (
            _path_tree_sha256(args.smiles_model_name)
            if "smiles" in base_model.encoders else None
        )
        trimer_geometry_active = args.graph_geometry_mode in {
            "trimer_scage_mcl", "current_mcl"
        }
        # G-family does not read the full Trimer payload at forward time, but
        # its frozen relation sidecar is derived from and strictly bound to
        # the same Trimer artifact. Keep that source identity in the
        # checkpoint/cache bundle without re-enabling legacy MCL.
        trimer_identity_bound = (
            trimer_geometry_active
            or getattr(args, "g_family_arm", None) is not None
        )
        cache_bundle_hash = cache_bundle_binding_hash(
            cohort_hash=getattr(dataset, "feature_cohort_hash", None),
            topology_artifact_hash=getattr(
                dataset, "topology_cache_artifact_hash", None
            ),
            trimer_artifact_hash=(
                getattr(dataset, "trimer_cache_artifact_hash", None)
                if trimer_identity_bound
                else None
            ),
        )
        final_cache_binding = _build_final_cache_binding(pretrain_profile)
        pretrain_code_identity = _pretrain_code_identity()
        source_contract = {
            "schema": PRETRAIN_CHECKPOINT_SCHEMA,
            "profile_id": (
                str(pretrain_profile["profile_id"])
                if pretrain_profile is not None else str(args.pretrain_profile)
            ),
            "stage": str(args.checkpoint_stage),
            "baseline": MTS_ROUTE_NAME,
            "topology_representation": args.topology_representation,
            "pretraining_dataset": str(args.dataset_name),
            "source_cohort_hash": getattr(dataset, "feature_cohort_hash", None),
            "topology_artifact_hash": getattr(dataset, "topology_cache_artifact_hash", None),
            "trimer_artifact_hash": getattr(dataset, "trimer_cache_artifact_hash", None),
            "angle_cache_schema": (
                (getattr(dataset, "angle_cache_metadata", None) or {}).get("schema")
                or (
                    pretrain_profile.get("angle_cache_schema")
                    if pretrain_profile is not None
                    and getattr(args, "g_family_arm", None) is not None
                    else None
                )
            ),
            "angle_cache_artifact_hash": (
                getattr(dataset, "angle_cache_artifact_hash", None)
                or (
                    pretrain_profile.get("angle_cache_artifact")
                    if pretrain_profile is not None
                    and getattr(args, "g_family_arm", None) is not None
                    else None
                )
            ),
            "g_family_arm": getattr(args, "g_family_arm", None),
            "g_family_bundle_hash": getattr(args, "g_family_bundle_hash", None),
            "relation_geometry_sidecar": getattr(args, "relation_geometry_sidecar", None),
            "relation_geometry_artifact_hash": getattr(args, "relation_geometry_artifact_hash", None),
            "g3_permutation_sidecar": getattr(args, "g3_permutation_sidecar", None),
            "g3_permutation_artifact_hash": getattr(args, "g3_permutation_artifact_hash", None),
            "pretraining_objective": getattr(args, "pretraining_objective", "joint"),
            "angle_loss_weight": float(getattr(args, "angle_loss_weight", 0.25)),
            "shared_step0_id": getattr(args, "shared_step0_id", None),
            "cache_store_sha256": final_cache_binding["store_sha256"],
            "topology_frozen_sha256": final_cache_binding["frozen_file_sha256"]["topology"],
            "trimer_frozen_sha256": final_cache_binding["frozen_file_sha256"]["trimer"],
            "feature_config_hash": str(args.feature_config_hash),
            "graph_model_config_hash": str(args.graph_model_config_hash),
            "geometry_model_config_hash": str(args.geometry_model_config_hash),
            "source_geometry_model_config_hash": str(args.source_geometry_model_config_hash),
            "pretrain_code_identity": pretrain_code_identity,
            "training_hash": str(args.training_config_hash),
            "seed": int(args.seed),
            "world_size": int(world_size),
            "batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(accumulation_steps),
            "optimizer": "Adam",
            "optimizer_steps": int(optimizer_steps_completed),
            "random_initialization": True,
            "parent_checkpoint": None,
            "initialization": (
                "fresh_paired" if initialization_info is not None else "random"
            ),
            "initialization_state": (
                dict(initialization_info) if initialization_info is not None else None
            ),
            "paired_init_id": (
                initialization_info["paired_init_id"]
                if initialization_info is not None else None
            ),
        }
        source_contract_sha256 = _canonical_json_hash(source_contract)
        target_contract = None
        if pretrain_profile is not None:
            target_contract = build_pretrain_target_contract(
                profile_id=pretrain_profile["profile_id"],
                source_cohort_hash=getattr(dataset, "feature_cohort_hash", None),
                feature_config_hash=args.feature_config_hash,
                graph_model_config_hash=args.graph_model_config_hash,
                geometry_model_config_hash=args.source_geometry_model_config_hash,
                topology_cache_artifact_hash=getattr(
                    dataset, "topology_cache_artifact_hash", None
                ),
                trimer_cache_artifact_hash=getattr(
                    dataset, "trimer_cache_artifact_hash", None
                ),
                angle_cache_schema=pretrain_profile["angle_cache_schema"],
                angle_cache_artifact_hash=pretrain_profile["angle_cache_artifact"],
                cache_bundle_hash=cache_bundle_hash,
                store_json_sha256=final_cache_binding["store_sha256"],
                topology_frozen_payload_sha256=final_cache_binding[
                    "frozen_file_sha256"
                ]["topology"],
                trimer_frozen_payload_sha256=final_cache_binding[
                    "frozen_file_sha256"
                ]["trimer"],
                optimizer_steps=optimizer_steps_completed,
                pretraining_objective=(
                    "masked_atom_only"
                    if getattr(args, "g_family_arm", None) is not None
                    else "masked_atom_plus_trimer_angle20_focal"
                ),
                topology_representation=args.topology_representation,
            )
        _atomic_torch_save({
            'state_dict': state_dict,
            'meta': {
                'schema': (
                    PRETRAIN_CHECKPOINT_SCHEMA
                    if pretrain_profile is not None else MIPS_TRIMER_CHECKPOINT_SCHEMA
                ),
                'baseline': MTS_ROUTE_NAME,
                'route_short_name': MTS_ROUTE_SHORT_NAME,
                'config_schema': args.config_schema,
                'stage': args.checkpoint_stage,
                'modalities': list(args.modalities),
                'graph_input': args.graph_input,
                'geom_input': args.geom_input,
                'fusion_type': args.fusion_type,
                'pretraining_unique_smiles': bool(args.pretrain_unique_smiles),
                'feature_schema': (
                    MIPS_EXPLICIT_FEATURE_SCHEMA
                    if args.topology_representation == TOPOLOGY_EXPLICIT
                    else MIPS_TRIMER_FEATURE_SCHEMA
                ),
                'topology_representation': args.topology_representation,
                'cache_layout_schema': MIPS_TRIMER_CACHE_LAYOUT_SCHEMA,
                'cache_bundle_schema': MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
                'topology_lmdb_schema': (
                    MIPS_EXPLICIT_TOPOLOGY_SCHEMA
                    if args.topology_representation == TOPOLOGY_EXPLICIT
                    else MIPS_TRIMER_TOPOLOGY_SCHEMA
                ),
                'cache_bundle_hash': cache_bundle_hash,
                'pretrain_profile': (
                    dict(pretrain_profile) if pretrain_profile is not None else None
                ),
                'pretrain_profile_sha256': (
                    _canonical_json_hash(pretrain_profile)
                    if pretrain_profile is not None else None
                ),
                'pretrain_code_identity': pretrain_code_identity,
                'source_contract': source_contract if pretrain_profile is not None else None,
                'source_contract_sha256': (
                    source_contract_sha256 if pretrain_profile is not None else None
                ),
                'target_contract': target_contract,
                'final_cache_binding': final_cache_binding if pretrain_profile is not None else None,
                'mips_local_lga_schema_version': (
                    MIPS_EXPLICIT_LGA_SCHEMA_VERSION
                    if args.topology_representation == TOPOLOGY_EXPLICIT
                    else MIPS_CANONICAL_LGA_SCHEMA_VERSION
                ),
                'mips_core': args.mips_core,
                'mips_max_hops': int(args.mips_max_hops),
                'mips_use_descriptors': bool(args.mips_use_descriptors),
                'mips_semantics': args.mips_semantics,
                'mips_descriptor_components': args.mips_descriptor_components,
                'mips_descriptor_protocol': args.mips_descriptor_protocol,
                'mips_descriptor_fusion_mode': (
                    args.mips_descriptor_fusion_mode
                ),
                'mips_descriptor_disturbance': float(
                    args.mips_descriptor_disturbance
                ),
                'mips_backbone_mode': args.mips_backbone_mode,
                'mips_input_norm': args.mips_input_norm,
                'mips_mask_mode': args.mips_mask_mode,
                'mips_mask_policy': args.mips_mask_policy,
                'mips_masked_loss_reduction': (
                    args.mips_masked_loss_reduction
                ),
                'spatial_mode': args.spatial_mode,
                'graph_geometry_mode': args.graph_geometry_mode,
                'trimer_cache_hash': (
                    getattr(dataset, 'trimer_cache_hash', None)
                    if trimer_identity_bound
                    else None
                ),
                'trimer_cache_artifact_hash': (
                    getattr(dataset, 'trimer_cache_artifact_hash', None)
                    if trimer_identity_bound
                    else None
                ),
                'topology_cache_hash': getattr(
                    dataset, 'topology_cache_hash', None
                ),
                'topology_cache_artifact_hash': getattr(
                    dataset, 'topology_cache_artifact_hash', None
                ),
                'feature_cohort_hash': getattr(
                    dataset, 'feature_cohort_hash', None
                ),
                'source_cohort_hash': getattr(
                    dataset, 'feature_cohort_hash', None
                ),
                'feature_cache_item_timeout': int(
                    args.feature_cache_item_timeout
                ),
                'trimer_conformer_protocol': (
                    MIPS_TRIMER_PROTOCOL
                    if trimer_identity_bound
                    else 'none'
                ),
                'trimer_content_schema': (
                    MIPS_TRIMER_CONTENT_SCHEMA
                    if trimer_identity_bound
                    else None
                ),
                'trimer_lmdb_schema': (
                    MIPS_TRIMER_LMDB_SCHEMA
                    if trimer_identity_bound
                    else None
                ),
                'trimer_builder_version': (
                    MIPS_TRIMER_BUILDER_VERSION
                    if trimer_identity_bound
                    else None
                ),
                'trimer_mmff_relax_steps': (
                    MIPS_TRIMER_MMFF_RELAX_STEPS
                    if trimer_identity_bound
                    else None
                ),
                'trimer_require_mmff_convergence': (
                    MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE
                    if trimer_identity_bound
                    else None
                ),
                'trimer_acceptance': (
                    MIPS_TRIMER_ACCEPTANCE
                    if trimer_identity_bound
                    else None
                ),
                'trimer_selection': (
                    MIPS_TRIMER_SELECTION
                    if trimer_identity_bound
                    else None
                ),
                'trimer_num_candidates': int(args.trimer_num_candidates),
                'trimer_max_heavy_atoms': int(args.trimer_max_heavy_atoms),
                'mcl_distance_percentiles': list(
                    args.mcl_distance_percentiles
                ),
                'angle_cache_schema': (
                    (
                        pretrain_profile.get('angle_cache_schema')
                        if pretrain_profile is not None
                        and getattr(args, 'g_family_arm', None) is not None
                        else None
                    )
                    or (
                        'mts-angle-continuous-cache-v1'
                        if args.angle_objective == 'cosine'
                        else 'mts-trimer-bond-angle-cache-v2'
                    )
                ),
                'angle_objective': str(args.angle_objective),
                'angle_cache_artifact_hash': (
                    (
                        pretrain_profile.get('angle_cache_artifact')
                        if pretrain_profile is not None
                        and getattr(args, 'g_family_arm', None) is not None
                        else None
                    )
                    or getattr(
                        getattr(dataset, '_lazy_feature_store', None),
                        'angle_cache_artifact_hash', None
                    )
                ),
                'angle_bins': int(args.scage_angle_bins),
                'angle_focal_gamma': float(args.scage_focal_gamma),
                'pretraining_objective': (
                    getattr(args, 'pretraining_objective', 'joint')
                    if getattr(args, 'g_family_arm', None) is not None
                    else (
                        'masked_atom_plus_trimer_bond_angle'
                        if args.pretrain_stage == MTS_STAGE1_ID else None
                    )
                ),
                'angle_loss_weight': float(getattr(args, 'angle_loss_weight', 0.25)),
                'g_family_arm': getattr(args, 'g_family_arm', None),
                'g_family_bundle_hash': getattr(args, 'g_family_bundle_hash', None),
                'relation_geometry_sidecar': getattr(args, 'relation_geometry_sidecar', None),
                'relation_geometry_artifact_hash': getattr(args, 'relation_geometry_artifact_hash', None),
                'g3_permutation_sidecar': getattr(args, 'g3_permutation_sidecar', None),
                'g3_permutation_artifact_hash': getattr(args, 'g3_permutation_artifact_hash', None),
                'shared_step0_id': getattr(args, 'shared_step0_id', None),
                'initialization': (
                    'fresh_paired' if initialization_info is not None else 'random'
                ),
                'initialization_state': (
                    dict(initialization_info) if initialization_info is not None else None
                ),
                'paired_init_id': (
                    initialization_info['paired_init_id']
                    if initialization_info is not None else None
                ),
                'reference_commits': {
                    'mips': '26aafe52926a3f33bf2d3d382ae263360319812d',
                    'scage': '82bcbb4647e31bf0d413a317e69a2526df75ce01',
                },
                'mips_variant': args.mips_variant,
                'model_identity': (
                    'T1' if args.topology_attention_variant == 'msta_last2'
                    else 'T0'
                ),
                'topology_attention_variant': args.topology_attention_variant,
                'msta_layer_indices': list(args.msta_layer_indices),
                'msta_local_spd': list(args.msta_local_spd),
                'msta_context_spd': list(args.msta_context_spd),
                'msta_share_relation_dropout': bool(
                    args.msta_share_relation_dropout
                ),
                'msta_local_output_bias': bool(args.msta_local_output_bias),
                'msta_local_output_init': args.msta_local_output_init,
                'experiment_id': args.experiment_id,
                'feature_config_hash': args.feature_config_hash,
                'o8_feature_config_hash': args.o8_feature_config_hash,
                'model_config_hash': args.model_config_hash,
                'graph_model_config_hash': args.graph_model_config_hash,
                'geometry_model_config_hash': args.geometry_model_config_hash,
                'source_geometry_model_config_hash': args.geometry_model_config_hash,
                'alignment_model_config_hash': (
                    args.alignment_model_config_hash
                    if args.pretrain_stage == 'alignment' else None
                ),
                'calculated_legacy_model_hash': calculated_model_hash,
                'source_data_hash': source_data_hash,
                'pretraining_dataset': args.dataset_name,
                'tier': (
                    '1m' if args.graph_encoder_type == 'mips_trimer_scage'
                    else args.dataset_name
                ),
                'random_seed': int(args.seed),
                'optimizer_steps': int(optimizer_steps_completed),
                'smoke_only': bool(args.resume_smoke),
                'training_wall_seconds': float(
                    time.monotonic() - pretrain_started
                ),
                'git_sha': git_sha,
                'dirty_diff_sha': dirty_diff_sha,
                'training_config_hash': args.training_config_hash,
                'tokenizer_mlm_hash': tokenizer_hash,
                'model': {
                    'architecture': MTS_ROUTE_NAME,
                    'layers': 6,
                    'embedding_dim': 512,
                    'heads': 8,
                    'ffn_hidden_dim': 2048,
                    'max_hops': int(args.mips_max_hops),
                    'path_nodes': int(args.mips_max_hops) + 1,
                    'readout': 'atom_mean',
                    'descriptor': 'graph_level_MD200',
                    'graph_token': False,
                },
                'dynamic_loss_config': ({
                    'distributed_statistics': 'all_reduce_mean_over_valid_ranks',
                    'warmup_steps': int(args.dynamic_loss_warmup_steps),
                    'recent_window': int(args.dynamic_loss_recent_window),
                    'temperature': float(args.dynamic_loss_temperature),
                    'task_priors': dict(dynamic_loss_weighter.task_priors),
                } if dynamic_loss_weighter is not None else None),
                'optimizer_schedule': {
                    'optimizer': 'Adam',
                    'betas': [0.9, 0.98],
                    'eps': 1e-8,
                    'weight_decay': 0.0,
                    'type': f'linear_warmup_{args.mips_scheduler}_decay',
                    'batch_size_per_rank': int(args.batch_size),
                    'world_size': int(world_size),
                    'effective_batch_size': int(args.batch_size) * int(world_size) * accumulation_steps,
                    'warmup_ratio': float(args.warmup_ratio),
                    'warmup_steps': int(warmup_steps),
                    'scheduler_power': float(args.scheduler_power),
                    'end_lr': float(args.end_lr),
                    'total_optimizer_steps': total_optimizer_steps,
                },
                'amp_dtype': args.amp_dtype,
                'm4p_priors': {
                    'masked_atom': args.scage_mips_mask_weight,
                    'trimer_bond_angle': args.graph_angle_weight,
                    'masked_spd': 0.0,
                    'path_bond': 0.0,
                },
                'm4p_geometry_objective': None,
                'alignment_objective': (
                    'fusion-view-lomo-fused-mask-v2'
                    if args.pretrain_stage == 'alignment' else None
                ),
                'fusion_dropout': args.fusion_dropout,
            },
        }, args.save_path)
        if pretrain_profile is not None and not args.resume_smoke:
            if optimizer_steps_completed != int(pretrain_profile["optimizer_steps"]):
                raise RuntimeError(
                    "formal pretraining ended before the fixed optimizer-step budget"
                )
            complete_path = Path(args.save_path).with_name(
                Path(args.save_path).name + ".complete.json"
            )
            complete_tmp = complete_path.with_name(
                complete_path.name + f".tmp.{os.getpid()}"
            )
            complete_tmp.write_text(json.dumps({
                "schema": "mts-pretrain-complete-v1",
                "checkpoint_schema": PRETRAIN_CHECKPOINT_SCHEMA,
                "checkpoint": str(Path(args.save_path).resolve()),
                "checkpoint_sha256": _file_sha256(args.save_path),
                "optimizer_steps": int(optimizer_steps_completed),
                "profile_id": str(pretrain_profile["profile_id"]),
                "pretrain_code_sha256": pretrain_code_identity["sha256"],
            }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(complete_tmp, complete_path)
    else:
        torch.save(state_dict, args.save_path)
    print(f"Pretrained model saved at {args.save_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
