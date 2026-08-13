import os
import sys
import argparse
import json
import warnings
import hashlib
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    CACHE_LAYOUT_SCHEMA as MIPS_TRIMER_CACHE_LAYOUT_SCHEMA,
    CACHE_BUNDLE_SCHEMA as MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as MIPS_TRIMER_TOPOLOGY_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION as MIPS_CANONICAL_LGA_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA as MIPS_TRIMER_CHECKPOINT_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA as MTS_PRETRAIN_CHECKPOINT_SCHEMA,
    PRETRAIN_TARGET_CONTRACT_SCHEMA as MTS_PRETRAIN_TARGET_CONTRACT_SCHEMA,
    PRETRAIN_PROFILE_ID as MTS_PRETRAIN_PROFILE_ID,
    CACHE_BOND_ANGLE_SCHEMA as MTS_CATEGORICAL_ANGLE_SCHEMA,
    CONFIG_SCHEMA as MIPS_TRIMER_CONFIG_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA as MTS_EXPERIMENT_CONFIG_SCHEMA,
    FEATURE_SCHEMA as MIPS_TRIMER_FEATURE_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA as MIPS_EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_TOPOLOGY_LMDB_SCHEMA as MIPS_EXPLICIT_TOPOLOGY_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION as MIPS_EXPLICIT_LGA_SCHEMA_VERSION,
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
    TRIMER_ACCEPTANCE as MIPS_TRIMER_ACCEPTANCE,
    TRIMER_BUILDER_VERSION as MIPS_TRIMER_BUILDER_VERSION,
    TRIMER_CONTENT_SCHEMA as MIPS_TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA as MIPS_TRIMER_LMDB_SCHEMA,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS as MIPS_TRIMER_MMFF_RELAX_STEPS,
    TRIMER_PROTOCOL as MIPS_TRIMER_PROTOCOL,
    TRIMER_REQUIRE_MMFF_CONVERGENCE as MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE,
    TRIMER_SELECTION as MIPS_TRIMER_SELECTION,
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
    ROUTE_NAME as MTS_ROUTE_NAME,
    ROUTE_SHORT_NAME as MTS_ROUTE_SHORT_NAME,
    STAGE1_ID as MTS_STAGE1_ID,
    STAGE2_ID as MTS_STAGE2_ID,
    cache_bundle_binding_hash,
    validate_runtime_args as validate_mips_trimer_runtime,
)
from src.dataset.lmdb_cache import sample_key_from_smiles


def _cohort_hash_from_current_manifest(root, dataset_name):
    """Read an existing immutable cohort pointer without rebuilding it."""

    pointer = os.path.join(
        root, "processed", "mips_trimer_scage", "cohorts",
        str(dataset_name), "current.json",
    )
    try:
        with open(pointer, encoding="utf-8") as handle:
            value = json.load(handle)
        cohort_hash = str(value.get("cohort_hash", ""))
        return cohort_hash or None
    except (OSError, ValueError, TypeError):
        return None


def _angle_artifact_hash_for_cohort(root, cohort_hash):
    """Read the immutable PI1M_v2 angle artifact without loading angle data."""
    if not cohort_hash:
        return None
    try:
        from scripts.audit_mips_trimer_cache import _specs
        trimer_root = Path(_specs(Path(PROJECT_ROOT))["trimer"]["root"])
        marker = (
            trimer_root / "derived" / "bond_angle" / str(cohort_hash) / ".done"
        )
        value = marker.read_text(encoding="utf-8").strip()
        return value if len(value) == 64 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _continuous_angle_artifact_hash_for_cohort(root, cohort_hash):
    if not cohort_hash:
        return None
    try:
        from scripts.audit_mips_trimer_cache import _specs
        trimer_root = Path(_specs(Path(PROJECT_ROOT))["trimer"]["root"])
        marker = (
            trimer_root / "derived" / "bond_angle_continuous"
            / str(cohort_hash) / ".done"
        )
        value = marker.read_text(encoding="utf-8").strip()
        return value if len(value) == 64 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None

SUPPORTED_MODALITIES = ('graph', 'smiles', 'fp')
SUPPORTED_FUSION_TYPES = ('none', 'zero_gated_residual')
CROSS_TASK_AUXILIARY_MAP = {
    task: tuple(other for other in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc') if other != task)
    for task in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
}


def _target_contract_mismatch(
    contract,
    *,
    args,
    dataset,
    pretraining_cohort_hash,
    expected_angle_artifact,
    store_json_sha256,
    topology_frozen_payload_sha256,
    trimer_frozen_payload_sha256,
    expected_graph_model_config_hash=None,
):
    """Return True when the checkpoint target_contract does not bind the
    current frozen production identity (Plan contract-finalization §4 / G0
    baseline §3.2).

    The target contract is the single identity a production loader may rely
    on.  Every schema constant is compared against the unique contract module,
    every hash/artifact binding against the frozen store, .frozen payloads and
    the current resolved config, and the self-excluding digest is recomputed.
    The angle cache schema is fixed to the continuous sidecar contract.
    A mismatch rejects the checkpoint before any model instantiation.
    """
    from src.dataset.mips_trimer_contract import (
        CACHE_BUNDLE_SCHEMA,
        CACHE_CONTINUOUS_ANGLE_SCHEMA,
        PRETRAIN_CHECKPOINT_SCHEMA,
        PRETRAIN_TARGET_CONTRACT_SCHEMA,
        PRETRAIN_PROFILE_ID,
        CACHE_BOND_ANGLE_SCHEMA,
        CACHE_LAYOUT_SCHEMA,
        CANONICAL_LGA_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA,
        CONFIG_SCHEMA,
        EXPERIMENT_CONFIG_SCHEMA,
        FEATURE_SCHEMA,
        TARGET_CONTRACT_SCHEMA,
        TOPOLOGY_LMDB_SCHEMA,
        TRIMER_BUILDER_VERSION,
        TRIMER_CONTENT_SCHEMA,
        TRIMER_LMDB_SCHEMA,
        TRIMER_PROTOCOL,
        TOPOLOGY_CANONICAL,
        _canonical_json_hash,
        cache_bundle_binding_hash,
    )
    if not isinstance(contract, dict):
        return True
    contract_schema = contract.get("schema")
    if contract_schema not in {TARGET_CONTRACT_SCHEMA, PRETRAIN_TARGET_CONTRACT_SCHEMA}:
        return True
    pretrain_v2 = contract_schema == PRETRAIN_TARGET_CONTRACT_SCHEMA
    constant_checks = {
        "config_schema": CONFIG_SCHEMA,
        "experiment_config_schema": EXPERIMENT_CONFIG_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "cache_bundle_schema": CACHE_BUNDLE_SCHEMA,
        "topology_lmdb_schema": TOPOLOGY_LMDB_SCHEMA,
        "trimer_content_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
        "trimer_builder_version": TRIMER_BUILDER_VERSION,
        "canonical_lga_schema_version": CANONICAL_LGA_SCHEMA_VERSION,
        "trimer_protocol": TRIMER_PROTOCOL,
        "checkpoint_schema": (
            PRETRAIN_CHECKPOINT_SCHEMA if pretrain_v2 else CHECKPOINT_SCHEMA
        ),
        "feature_config_hash": args.feature_config_hash,
        "graph_model_config_hash": (
            expected_graph_model_config_hash
            if expected_graph_model_config_hash is not None
            else args.graph_model_config_hash
        ),
        "geometry_model_config_hash": args.source_geometry_model_config_hash,
        "source_cohort_hash": pretraining_cohort_hash,
        "topology_cache_artifact_hash": getattr(
            dataset, "topology_cache_artifact_hash", None
        ),
        "trimer_cache_artifact_hash": getattr(
            dataset, "trimer_cache_artifact_hash", None
        ),
        "angle_cache_schema": (
            CACHE_BOND_ANGLE_SCHEMA if pretrain_v2 else CACHE_CONTINUOUS_ANGLE_SCHEMA
        ),
        "angle_cache_artifact_hash": expected_angle_artifact,
        "store_json_sha256": store_json_sha256,
        "topology_frozen_payload_sha256": topology_frozen_payload_sha256,
        "trimer_frozen_payload_sha256": trimer_frozen_payload_sha256,
        "optimizer_steps": 20000,
        "pretraining_objective": (
            "masked_atom_only"
            if getattr(args, "g_family_arm", None) is not None
            else "masked_atom_plus_trimer_angle20_focal"
            if pretrain_v2 else "masked_atom_plus_trimer_bond_angle"
        ),
    }
    if pretrain_v2:
        constant_checks["profile_id"] = PRETRAIN_PROFILE_ID
        constant_checks["topology_representation"] = getattr(
            args, "topology_representation", TOPOLOGY_CANONICAL
        )
    for key, expected in constant_checks.items():
        if contract.get(key) != expected:
            return True
    if contract.get("cache_bundle_hash") != cache_bundle_binding_hash(
        cohort_hash=contract.get("source_cohort_hash"),
        topology_artifact_hash=contract.get("topology_cache_artifact_hash"),
        trimer_artifact_hash=contract.get("trimer_cache_artifact_hash"),
    ):
        return True
    # Self-excluding digest: recomputed over every field except the digest
    # itself, so it cannot be made to match by editing the digest alone.
    base = {
        key: value for key, value in contract.items()
        if key != "target_contract_sha256"
    }
    if contract.get("target_contract_sha256") != _canonical_json_hash(base):
        return True
    return False


def _source_contract_digest_mismatch(source_contract, declared_sha256) -> bool:
    """Return True when the source_contract digest does not match its declared
    sha256 (Plan G0-baseline §3.1).  The digest is computed over the immutable
    source dict itself; the digest field lives at checkpoint-meta level."""
    from src.dataset.mips_trimer_contract import _canonical_json_hash

    if not isinstance(source_contract, dict):
        return True
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        return True
    return _canonical_json_hash(source_contract) != declared_sha256


def validate_g_family_checkpoint_binding(checkpoint_meta, args, *, allow_smoke=False):
    """Validate stable G identity while allowing cohort-specific active roots."""
    arm = getattr(args, "g_family_arm", None)
    if arm is None:
        if checkpoint_meta.get("g_family_arm") is not None:
            raise RuntimeError("G-family checkpoint cannot enter a T-family run")
        return
    expected = {
        "g_family_arm": arm,
        "topology_attention_variant": "msta_last2",
        "shared_step0_id": getattr(args, "shared_step0_id", None),
        "pretraining_objective": "masked_atom_only",
        "g_family_bundle_hash": getattr(args, "g_family_bundle_hash", None),
    }
    mismatches = {
        key: (checkpoint_meta.get(key), value)
        for key, value in expected.items()
        if checkpoint_meta.get(key) != value
    }
    steps = int(checkpoint_meta.get("optimizer_steps", -1))
    smoke_ok = allow_smoke and bool(checkpoint_meta.get("smoke_only")) and steps in {1, 2}
    if steps != 20000 and not smoke_ok:
        mismatches["optimizer_steps"] = (steps, "20000 or explicit 1-2 step smoke")
    relation_hash = checkpoint_meta.get("relation_geometry_artifact_hash")
    if arm != "g0" and (not isinstance(relation_hash, str) or len(relation_hash) != 64):
        mismatches["relation_geometry_artifact_hash"] = (relation_hash, "bound SHA256")
    permutation_hash = checkpoint_meta.get("g3_permutation_artifact_hash")
    if arm == "g3" and (not isinstance(permutation_hash, str) or len(permutation_hash) != 64):
        mismatches["g3_permutation_artifact_hash"] = (permutation_hash, "bound SHA256")
    if mismatches:
        raise RuntimeError(f"G-family checkpoint identity mismatch: {mismatches}")


def _is_mts_t1_function_preserving_init(meta) -> bool:
    """Identify the explicit T0 -> T1 architecture-init artifact family."""

    return bool(
        isinstance(meta, dict)
        and meta.get("init_artifact") is True
        and meta.get("initialization") == "function_preserving"
    )


def _validate_mts_t1_function_preserving_init(
    checkpoint,
    checkpoint_path,
    *,
    args,
    dataset,
    pretraining_cohort_hash,
    expected_angle_artifact,
    store_json_sha256,
    topology_frozen_payload_sha256,
    trimer_frozen_payload_sha256,
    model,
):
    """Validate the opt-in T1 init contract without weakening normal loading.

    The artifact carries the immutable T0 target contract because that is the
    parent pretraining fact.  The top-level metadata separately carries the
    T1 graph hash and zero-step init identity.  This helper checks both sides
    before the common strict state-dict transfer path is entered.
    """

    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(meta, dict) or not isinstance(state, dict):
        raise RuntimeError("T1 function-preserving init must contain meta/state_dict")
    if not bool(getattr(args, "allow_mts_t1_function_preserving_init", False)):
        raise RuntimeError(
            "T1 function-preserving init requires explicit "
            "--allow_mts_t1_function_preserving_init"
        )
    if args.topology_attention_variant != "msta_last2":
        raise RuntimeError("T1 function-preserving init requires topology_attention_variant=msta_last2")
    required_identity = {
        "init_artifact": True,
        "initialization": "function_preserving",
        "model_identity": "T1",
        "source_model_identity": "T0",
        "topology_attention_variant": "msta_last2",
        "source_optimizer_steps": 20000,
        "optimizer_steps": 0,
    }
    for key, expected in required_identity.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"T1 init metadata mismatch: {key}={meta.get(key)!r}, "
                f"expected {expected!r}"
            )
    if meta.get("schema") != MTS_PRETRAIN_CHECKPOINT_SCHEMA:
        raise RuntimeError("T1 init must use the immutable mts-model-v4 checkpoint schema")
    if meta.get("stage") != MTS_STAGE1_ID:
        raise RuntimeError("T1 init stage must be mts_joint_pretraining")
    if meta.get("graph_model_config_hash") != args.graph_model_config_hash:
        raise RuntimeError("T1 init top-level graph hash does not match the resolved T1 config")
    if meta.get("target_config_hash") != args.resolved_config_hash:
        raise RuntimeError("T1 init target config hash does not match the resolved experiment")
    if list(meta.get("msta_layer_indices", [])) != list(args.msta_layer_indices):
        raise RuntimeError("T1 init MSTA layer indices do not match the resolved config")
    if list(meta.get("msta_local_spd", [])) != list(args.msta_local_spd):
        raise RuntimeError("T1 init MSTA local SPD support does not match the resolved config")
    if list(meta.get("msta_context_spd", [])) != list(args.msta_context_spd):
        raise RuntimeError("T1 init MSTA context SPD support does not match the resolved config")
    if bool(meta.get("msta_share_relation_dropout")) is not bool(
        args.msta_share_relation_dropout
    ) or bool(meta.get("msta_local_output_bias")) is not bool(
        args.msta_local_output_bias
    ) or meta.get("msta_local_output_init") != args.msta_local_output_init:
        raise RuntimeError("T1 init MSTA identity fields do not match the resolved config")

    parent = Path(str(meta.get("parent_checkpoint", ""))).expanduser()
    if not parent.is_file():
        raise RuntimeError(f"T1 init parent checkpoint is missing: {parent}")
    parent_sha = meta.get("parent_checkpoint_sha256")
    if not isinstance(parent_sha, str) or len(parent_sha) != 64:
        raise RuntimeError("T1 init parent checkpoint SHA256 is missing")
    if _sha256_file(parent) != parent_sha:
        raise RuntimeError("T1 init parent checkpoint SHA256 mismatch")

    source_graph_hash = meta.get("source_graph_model_config_hash")
    source_contract = meta.get("source_contract")
    target_contract = meta.get("target_contract")
    if not isinstance(source_graph_hash, str) or len(source_graph_hash) != 64:
        raise RuntimeError("T1 init source graph hash is missing")
    if not isinstance(source_contract, dict) or not isinstance(target_contract, dict):
        raise RuntimeError("T1 init must retain both parent source and target contracts")
    if source_contract.get("graph_model_config_hash") != source_graph_hash:
        raise RuntimeError("T1 init source contract graph hash mismatch")
    if target_contract.get("graph_model_config_hash") != source_graph_hash:
        raise RuntimeError("T1 init target contract must retain the T0 graph hash")
    if _source_contract_digest_mismatch(
        source_contract, meta.get("source_contract_sha256")
    ):
        raise RuntimeError("T1 init source contract digest mismatch")
    if _target_contract_mismatch(
        target_contract,
        args=args,
        dataset=dataset,
        pretraining_cohort_hash=pretraining_cohort_hash,
        expected_angle_artifact=expected_angle_artifact,
        store_json_sha256=store_json_sha256,
        topology_frozen_payload_sha256=topology_frozen_payload_sha256,
        trimer_frozen_payload_sha256=trimer_frozen_payload_sha256,
        expected_graph_model_config_hash=source_graph_hash,
    ):
        raise RuntimeError("T1 init parent target contract does not bind the current frozen cache")
    pretrain_profile = meta.get("pretrain_profile")
    if not isinstance(pretrain_profile, dict) or int(
        pretrain_profile.get("optimizer_steps", -1)
    ) != 20000:
        raise RuntimeError("T1 init parent pretrain profile is not the completed 20k T0 source")
    if pretrain_profile.get("dataset") != "PI1M_v2":
        raise RuntimeError("T1 init parent pretrain dataset is not PI1M_v2")

    graph_prefix = "encoders.graph.encoder."
    graph_state = {
        key[len(graph_prefix):]: value
        for key, value in state.items()
        if key.startswith(graph_prefix)
    }
    expected_graph_state = {
        key[len(graph_prefix):]: value
        for key, value in model.state_dict().items()
        if key.startswith(graph_prefix)
    }
    if set(graph_state) != set(expected_graph_state):
        missing = sorted(set(expected_graph_state) - set(graph_state))
        unexpected = sorted(set(graph_state) - set(expected_graph_state))
        raise RuntimeError(
            "T1 init graph state is not transferable: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_mismatch = sorted(
        key for key in graph_state
        if tuple(graph_state[key].shape) != tuple(expected_graph_state[key].shape)
    )
    if shape_mismatch:
        raise RuntimeError(
            "T1 init graph state shape mismatch: " + ", ".join(shape_mismatch[:8])
        )
    for index in args.msta_layer_indices:
        key = f"encoders.graph.encoder.layers.{int(index)}.attention.local_output.weight"
        value = state.get(key)
        if value is None or tuple(value.shape) != tuple(
            expected_graph_state[f"layers.{int(index)}.attention.local_output.weight"].shape
        ):
            raise RuntimeError(f"T1 init local_output weight missing or malformed: {key}")
        if torch.count_nonzero(value).item() != 0:
            raise RuntimeError(f"T1 init local_output weight is not zero initialized: {key}")
    return meta


def _sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ablation_flags_from_env():
    """Read the causal geometry ablation switches (Plan mts_geometry_injection
    A0-A4) from the launcher environment; absent vars default to the full 3D
    production recipe (A3 semantics)."""
    import os

    use_star = os.environ.get("MTS_USE_STAR_RBF", "true").lower()
    use_mcl = os.environ.get("MTS_USE_MCL", "true").lower()
    random_mask = os.environ.get("MTS_MCL_RANDOM_MASK", "false").lower()
    return {
        "use_star_rbf": use_star not in ("0", "false", "no", "off"),
        "use_mcl": use_mcl not in ("0", "false", "no", "off"),
        "mcl_mask_mode": (
            "count_matched_random" if random_mask not in ("0", "false", "no", "off")
            else "real"
        ),
        "ablation_id": os.environ.get("MTS_ABLATION_ID"),
        "random_mask_sidecar": os.environ.get("MTS_RANDOM_MASK_SIDECAR"),
    }


class _MTSMultiTaskFoldDataset(TorchDataset):
    """Leakage-filtered, task-balanced view over the eight downstream sets."""

    is_mts_route = True

    def __init__(self, entries):
        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dataset, row, target, task_index = self.entries[int(index)]
        data = dataset[int(row)]
        data.y = torch.tensor([float(target)], dtype=torch.float)
        data.mts_task_index = torch.tensor(int(task_index), dtype=torch.long)
        return data


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value


def collect_attention_pooling_weights(model, data_loader, device):
    """Collect final attention/gate weights over the whole test set."""
    model.eval()
    attention_batches = []
    attention_labels = None
    named_batches = {}
    named_labels = {}

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            model(batch)
            attention_weights = model.attention_visual_weights.detach().cpu().numpy()
            if attention_weights.ndim != 2:
                raise ValueError(
                    "Expected attention/gate weights with shape "
                    f"[batch_size, num_inputs], got {attention_weights.shape}"
                )
            batch_labels = list(getattr(model, 'attention_visual_labels', model.modality_list))
            if attention_labels is None:
                attention_labels = batch_labels
            elif attention_labels != batch_labels:
                raise ValueError(f"Attention labels changed across batches: {attention_labels} vs {batch_labels}")
            attention_batches.append(attention_weights)

            for name, value in getattr(model, 'fusion_visual_weights', {}).items():
                weights, labels = value
                if weights is None or weights.numel() == 0:
                    continue
                labels = list(labels)
                if name in named_labels and named_labels[name] != labels:
                    raise ValueError(f"Fusion labels changed for {name}: {named_labels[name]} vs {labels}")
                named_labels[name] = labels
                named_batches.setdefault(name, []).append(weights.detach().cpu().numpy())

    if not attention_batches:
        raise ValueError("Cannot compute attention statistics from an empty data loader.")

    all_attention = np.concatenate(attention_batches, axis=0)
    fold_attention = all_attention.mean(axis=0)
    named_means = {}
    for name, batches in named_batches.items():
        values = np.concatenate(batches, axis=0).mean(axis=0)
        named_means[name] = (values, named_labels[name])
    return fold_attention, all_attention.shape, attention_labels, named_means


def format_attention_weights(modalities, attention_weights):
    if len(modalities) != len(attention_weights):
        raise ValueError(
            f"Modalities length ({len(modalities)}) does not match attention length "
            f"({len(attention_weights)})."
        )
    return ";".join(
        f"{modality}:{float(weight):.6f}"
        for modality, weight in zip(modalities, attention_weights)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train UniEncoderAttention Model")
    parser.add_argument('--experiment_id', default='manual')
    parser.add_argument('--feature_config_hash', default='manual')
    parser.add_argument('--o8_feature_config_hash', default='manual')
    parser.add_argument('--model_config_hash', default='manual')
    parser.add_argument('--graph_model_config_hash', default='manual')
    parser.add_argument('--geometry_model_config_hash', default='manual')
    parser.add_argument('--source_geometry_model_config_hash', default='manual')
    parser.add_argument('--alignment_model_config_hash', default='manual')
    parser.add_argument('--training_config_hash', default='manual')
    parser.add_argument('--resolved_config_hash', default='manual')
    parser.add_argument('--checkpoint_sha256', default='')
    parser.add_argument('--cache_store_sha256', default='')
    parser.add_argument('--finetune_config_hash', default='manual')
    # Retired F phase schedule is intentionally not a public argument.
    parser.add_argument(
        '--finetune_profile',
        choices=['legacy_mts_huber_v1'],
        default='legacy_mts_huber_v1',
        help='Explicit downstream optimization profile for MTS experiments.',
    )
    parser.add_argument('--finetune_profile_hash', default='manual')
    parser.add_argument('--predictions_dir', default='')
    parser.add_argument(
        '--checkpoint_seed', type=int, default=None,
        help='Pretraining seed recorded by the checkpoint; independent of the fine-tuning seed.',
    )
    parser.add_argument('--checkpoint_pretraining_dataset', default='')
    parser.add_argument('--checkpoint_tier', default='')
    parser.add_argument(
        '--config_schema', default='mips-experiment-config-v2'
    )
    # The shared MTS launcher passes the source schema to both pretraining
    # and downstream entrypoints.  Downstream validation uses this metadata
    # to distinguish an experiment input from the resolved production
    # contract; it does not alter the fine-tuning objective.
    parser.add_argument('--config_source_schema', default='')
    parser.add_argument(
        '--split_manifest_dir', default='data/splits/mips_shared5'
    )
    parser.add_argument(
        '--cache_only', action='store_true',
        help='Build/validate downstream feature cache and exit before training.',
    )
    parser.add_argument(
        '--smiles_model_name',
        default="./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
    )
    parser.add_argument('--root', default='./data', help='Data root containing raw/ and processed/.')
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc'],
        help="List of tasks to train on. Default excludes tg: eat eea egb egc ei eps nc xc"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='UniEncoderAttention',
        help="Name of the model."
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        type=parse_modality,
        default=['graph'],
        help="MTS modalities: graph, optionally smiles and/or fp."
    )
    parser.add_argument(
        '--fusion_type',
        type=str,
        choices=SUPPORTED_FUSION_TYPES,
        default='none',
        help="MTS fusion: none for graph-only or zero_gated_residual for optional views."
    )

    parser.add_argument(
        '--fp_mode',
        type=str,
        choices=['disabled', 'ecfp', 'mixfp', 'attachment_count'],
        default='ecfp',
        help="Fingerprint implementation. ecfp keeps the original Morgan/ECFP 1024-bit FP; mixfp uses MACCSKeys + PubChemFingerprints.",
    )
    parser.add_argument('--fusion_dropout', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--head_dropout', type=float, default=0.25)
    parser.add_argument('--fp_bit_dropout', type=float, default=0.15)
    parser.add_argument('--fp_modality_dropout', type=float, default=0.25)
    parser.add_argument('--smiles_modality_dropout', type=float, default=0.10)
    parser.add_argument('--graph_modality_dropout', type=float, default=0.05)
    parser.add_argument(
        '--graph_input',
        type=str,
        choices=['repeat_unit', 'star_linking'],
        default='star_linking',
        help="Graph input type. 'repeat_unit' keeps the original graph; 'star_linking' removes two attachment atoms and connects their boundary atoms for graph-only topology input."
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default="./pretrained_models/saved_pretrained_model.pth",
        help="Path to the pretrained model."
    )
    parser.add_argument(
        '--allow_mts_t1_function_preserving_init',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Explicitly allow the isolated T0->T1 function-preserving init '
            'artifact for downstream warm-start; never enables ordinary resume.'
        ),
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Early stopping patience."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/results.csv',
        help="Directory to save results CSV."
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./saved_models',
        help="Directory to save trained models."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument('--loader_workers', type=int, default=4)
    parser.add_argument(
        '--eval_batch_size', type=int, default=64,
        help='Validation/test batch size; training batch size is unchanged.',
    )
    parser.add_argument(
        '--amp_dtype', choices=['fp32', 'bf16'], default='fp32',
        help='Fine-tuning precision. BF16 is enabled only after its parity gate.',
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument('--smiles_lr', type=float, default=5e-6)
    parser.add_argument('--graph_lr', type=float, default=1e-5)
    parser.add_argument('--fp_lr', type=float, default=1e-4)
    parser.add_argument('--fusion_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument(
        '--regression_loss', choices=['huber'], default='huber',
        help='MTS production regression objective: SmoothL1/Huber(beta=0.5).'
    )
    parser.add_argument('--huber_beta', type=float, default=0.5)
    parser.add_argument(
        '--unimodal_aux_weight', type=float, default=0.0,
        help=(
            'Stage-3 deep-supervision weight for independent SMILES/SCAGE/FP '
            'regression heads. Zero preserves fused-head-only training.'
        ),
    )
    parser.add_argument(
        '--cross_task_aux_weight', type=float, default=0.0,
        help=(
            'Weight for paired-property auxiliary supervision. Only labels for '
            'the target fold-training SMILES are used; held-out SMILES '
            'are explicitly excluded.'
        ),
    )
    parser.add_argument(
        '--cross_task_aux_tasks', nargs='*',
        choices=sorted(CROSS_TASK_AUXILIARY_MAP), default=[],
        help=(
            'Target tasks that receive paired-property auxiliary supervision. '
            'An empty list preserves the existing behavior and enables every '
            'mapped target when cross_task_aux_weight is positive.'
        ),
    )
    parser.add_argument('--fusion_prior_kl_weight', type=float, default=0.0)
    parser.add_argument(
        '--fusion_prior', nargs=3, type=float, default=[0.30, 0.40, 0.30],
        metavar=('SMILES', 'GRAPH', 'FP'),
        help='Target mean pooling distribution for SMILES/Graph/FP.',
    )
    # Validation-selected SWA was part of the retired F profile and is fixed
    # off by the active legacy MTS profile.
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for model, dropout, and data order.')
    parser.add_argument(
        '--fold_ids', nargs='+', type=int, default=[0, 1, 2, 3, 4],
        help='Zero-based cross-validation folds to execute. Default runs all five folds.',
    )
    parser.add_argument(
        '--refit_full_train',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'After selecting the epoch on the shared held-out validation fold, restart from the '
            'same pretrained state and fit that many epochs on the complete '
            'fold-training partition before the final held-out evaluation.'
        ),
    )
    parser.add_argument(
        '--target_transform',
        choices=['recommended', 'auto', 'standard', 'log'],
        default='recommended',
        help='Target preprocessing: recommended logs eps/nc and standardizes other current tasks; auto keeps legacy transforms.',
    )
    parser.add_argument(
        '--evaluation_protocol',
        choices=['historical_shared5', 'nested5'],
        default='historical_shared5',
    )
    parser.add_argument(
        '--finetune_mode', choices=['single_task', 'multitask_pcgrad'],
        default='single_task',
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
    parser.add_argument(
        '--mips_descriptor_disturbance', type=float, default=0.0,
    )
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
        '--mips_downstream_head',
        choices=['unipoly'],
        default='unipoly',
    )
    parser.add_argument(
        '--spatial_mode',
        choices=['trimer_scage'],
        default='trimer_scage',
    )
    parser.add_argument(
        '--graph_geometry_mode',
        choices=[
            'trimer_scage_mcl', 'current_mcl', 'mcl_rbf', 'disabled',
            'coordinate_shuffled', 'mcl_rbf_coordinate_shuffled',
            'g0', 'g1', 'g2', 'g3',
        ],
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
        '--mips_fusion_mode',
        choices=['none'],
        default='none',
    )
    parser.add_argument(
        '--projection_mode', choices=['plain', 'shared_private'],
        default='plain',
    )
    parser.add_argument(
        '--modality_control',
        choices=['real', 'batch_shuffled', 'constant_zero'], default='real',
    )
    parser.add_argument('--controlled_modality', choices=['smiles', 'fp'], default=None)
    parser.add_argument(
        '--mips_variant',
        choices=['O8'],
        default='O8',
    )
    parser.add_argument(
        '--joint_embedding_dim',
        type=int,
        default=256,
        help="Shared projection dimension for modality fusion."
    )
    parser.add_argument(
        '--feature_source_dataset',
        type=str,
        default='smi_all',
        help="Dataset used to build the SMILES-level feature cache (default: smi_all)."
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
        help='Conformer search budget. Must match the profile used to build the feature cache.'
    )
    args = parser.parse_args()
    if int(args.eval_batch_size) < int(args.batch_size):
        raise ValueError("--eval_batch_size must be >= --batch_size")
    # These are fixed MTS topology values, not user-selectable route
    # parameters.  Retired geometry and legacy staged-finetuning controls are
    # intentionally absent from the runtime namespace.
    args.geom_input = 'repeat_unit'
    # MTS downstream always constructs the encoder explicitly and applies the
    # legacy trainability profile in ``utils._configure_legacy_mts_trainability``.
    # Keep this internal compatibility value rather than exposing the retired
    # blanket-freeze CLI switch.
    args.freeze_encoder = False
    # These names are internal call-compatibility slots only.  They are not
    # user-selectable geometry learning-rate controls: the complete MTS
    # graph wrapper always uses graph_lr=1e-5.
    args.geom_lr = 1e-5
    args.mts_o8_lr = 1e-5
    args.mts_geometry_lr = 1e-5
    args.mts_adapter_lr = 1e-5
    args.freeze_smiles_epochs = 0
    args.deep_unfreeze_epoch = 0
    args.fp_unfreeze_epoch = -1
    args.swa_start_epoch = -1
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
    if args.finetune_mode == 'multitask_pcgrad' and args.refit_full_train:
        parser.error(
            '--refit_full_train is not supported with multitask_pcgrad; '
            'nested validation already supplies leakage-safe model selection'
        )
    if args.graph_encoder_type == "mts":
        args.graph_encoder_type = MTS_ROUTE_INTERNAL
    return args


def _scage_checkpoint_key_compatibility(
    model_keys,
    checkpoint_keys,
    expected_stage,
    unimodal_aux_weight=0.0,
    cross_task_aux_weight=0.0,
):
    """Classify checkpoint keys under the Stage 1/Stage 2 transfer contract."""
    model_keys = set(model_keys)
    checkpoint_keys = set(checkpoint_keys)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if expected_stage == 'mips_pretrain':
        # Graph-only downstream deliberately skips Stage 2. Only the graph
        # encoder is transferred; projection/fusion/head parameters start from
        # their deterministic Stage 3 initialization.
        allowed_missing = {
            key for key in missing
            if not key.startswith('encoders.graph.')
        }
        allowed_unexpected = set()
    else:
        # Multimodal downstream must strictly inherit Stage 2
        # encoder/projection/fusion parameters. Only downstream auxiliary heads
        # may be newly initialized.
        allowed_unexpected = {
            'alignment_mask_head.weight',
            'alignment_mask_head.bias',
        }
        allowed_missing = {
            key for key in missing
            if key.startswith('mlp.') or (
                float(unimodal_aux_weight) > 0.0
                and key.startswith('modality_heads.')
            ) or (
                float(cross_task_aux_weight) > 0.0
                and key.startswith('cross_task_aux_heads.')
            )
        }
    incompatible = [
        key for key in missing if key not in allowed_missing
    ] + [
        key for key in unexpected if key not in allowed_unexpected
    ]
    retained_unexpected = [
        key for key in unexpected if key not in allowed_unexpected
    ]
    return missing, retained_unexpected, incompatible


def select_mts_checkpoint_transfer_keys(
    model_keys, checkpoint_keys, *, allowed_missing=()
):
    """Return the exact learned MTS topology keys transferred to a fold.

    Stage-3 fine-tuning deliberately reinitializes the MD200 residual,
    graph projection/norm and regression head.  Every other common
    ``encoders.graph.encoder`` tensor is part of the learned topology state,
    including T1 MSTA and G-family relation-geometry parameters.  Keeping the
    selection in one small, testable function prevents a descriptive log line
    from becoming a weaker migration contract.
    """

    model_keys = set(model_keys)
    checkpoint_keys = set(checkpoint_keys)
    prefix = "encoders.graph.encoder."
    candidates = tuple(sorted(
        key for key in model_keys
        if key.startswith(prefix) and ".md_residual." not in key
    ))
    allowed_missing = set(allowed_missing)
    invalid_allowed = allowed_missing - set(candidates)
    if invalid_allowed:
        raise RuntimeError(
            "MTS checkpoint allowed-missing set is not a topology tensor: "
            + ", ".join(sorted(invalid_allowed)[:10])
        )
    missing = tuple(
        key for key in candidates
        if key not in checkpoint_keys and key not in allowed_missing
    )
    if missing:
        raise RuntimeError(
            "MTS checkpoint missing transferable topology tensors: "
            + ", ".join(missing[:10])
        )
    return tuple(key for key in candidates if key in checkpoint_keys)


def main():
    args = parse_arguments()
    validate_mips_trimer_runtime(args)
    if args.graph_encoder_type == "mips_trimer_scage":
        ablation_smoke = os.environ.get("MTS_ABLATION_SMOKE", "0") == "1"
        g_family_smoke = os.environ.get("MTS_G_FAMILY_SMOKE", "0") == "1"
        short_smoke = ablation_smoke or g_family_smoke
        if getattr(args, "g_family_arm", None) is not None:
            if not args.g_family_bundle_hash or not args.shared_step0_id:
                raise ValueError("G-family fine-tuning requires bundle and shared step-0 identities")
            if args.g_family_arm != "g0" and (
                not args.relation_geometry_sidecar
                or not args.relation_geometry_artifact_hash
            ):
                raise ValueError("G1/G2/G3 fine-tuning requires the active downstream artifact binding")
            if args.g_family_arm == "g3" and (
                not args.g3_permutation_sidecar
                or not args.g3_permutation_artifact_hash
            ):
                raise ValueError("G3 fine-tuning requires the active permutation artifact binding")
        if args.finetune_profile != "legacy_mts_huber_v1":
            raise ValueError(
                "MTS production fine-tuning uses legacy_mts_huber_v1; "
                "the retired F/Phase-A-B-C profile is not supported"
            )
        if args.regression_loss != "huber" or float(args.huber_beta) != 0.5:
            raise ValueError(
                "MTS production fine-tuning is fixed to Huber(beta=0.5)"
            )
        fixed_finetune = {
            "epochs": (int(args.epochs), 2 if short_smoke else 100),
            "patience": (int(args.patience), 2 if short_smoke else 10),
            "batch_size": (int(args.batch_size), 32),
            "graph_lr": (float(args.graph_lr), 1e-5),
            "fusion_lr": (float(args.fusion_lr), 1e-4),
            "head_lr": (float(args.head_lr), 1e-4),
            "weight_decay": (float(args.weight_decay), 0.02),
            "warmup_epochs": (int(args.warmup_epochs), 5),
            "max_grad_norm": (float(args.max_grad_norm), 1.0),
            "head_dropout": (float(args.head_dropout), 0.25),
            "swa_start_epoch": (int(args.swa_start_epoch), -1),
            "fp_unfreeze_epoch": (int(args.fp_unfreeze_epoch), -1),
            "target_transform": (str(args.target_transform), "recommended"),
            "freeze_smiles_epochs": (int(args.freeze_smiles_epochs), 0),
            "deep_unfreeze_epoch": (int(args.deep_unfreeze_epoch), 0),
        }
        mismatched_finetune = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in fixed_finetune.items()
            if actual != expected
        ]
        if mismatched_finetune:
            raise ValueError(
                "legacy_mts_huber_v1 has fixed fine-tuning parameters; "
                "the retired F/Phase-A-B-C overrides are not supported: "
                + ", ".join(mismatched_finetune)
            )
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and args.config_schema not in {
            MIPS_TRIMER_CONFIG_SCHEMA, MTS_EXPERIMENT_CONFIG_SCHEMA
        }
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} only accepts "
            f"{MIPS_TRIMER_CONFIG_SCHEMA} or {MTS_EXPERIMENT_CONFIG_SCHEMA}"
        )
    if args.graph_encoder_type == 'mips_trimer_scage':
        production_contract = args.config_schema == MIPS_TRIMER_CONFIG_SCHEMA
        allowed_experiment_modalities = (
            ["graph"], ["graph", "smiles"], ["graph", "fp"],
            ["graph", "smiles", "fp"],
        )
        expected_fusion = (
            "none" if args.modalities == ["graph"] else "zero_gated_residual"
        )
        invalid_route = (
            args.mips_fusion_mode != "none"
            or args.projection_mode != "plain"
            or args.spatial_mode != "trimer_scage"
            or args.fusion_type != expected_fusion
            or args.modalities not in allowed_experiment_modalities
            or (
                "fp" in args.modalities and args.fp_mode != "attachment_count"
            )
            or (
                "fp" not in args.modalities and args.fp_mode != "disabled"
            )
        )
        if production_contract:
            invalid_route = invalid_route or (
                args.modalities != ["graph"]
                or args.graph_geometry_mode != "trimer_scage_mcl"
            )
        if invalid_route:
            raise ValueError(
                f"invalid {MTS_ROUTE_NAME} production/experiment combination"
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
    from src.modules import UniEncoderAttention
    from src.utils import (
        TargetScaler, fit_fixed_epochs, get_data_loader, scale_targets,
        set_global_seed, test_model, train_and_evaluate,
    )
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
    # Ignore warnings
    warnings.filterwarnings("ignore")

    pre_trained_model_dict = {
        'smiles_model_name': args.smiles_model_name,
        'gnn_model_name': "",
    }

    result_output_dir = args.results_dir
    model_output_dir = args.models_dir
    model_modality_list = args.modalities

    task_list = args.tasks
    dataset_task_list = list(task_list)
    selected_cross_task_aux = set(args.cross_task_aux_tasks)
    if float(args.cross_task_aux_weight) > 0:
        for task in task_list:
            if selected_cross_task_aux and task not in selected_cross_task_aux:
                continue
            for auxiliary_task in CROSS_TASK_AUXILIARY_MAP.get(task, ()):
                if auxiliary_task not in dataset_task_list:
                    dataset_task_list.append(auxiliary_task)
    dataset_name_list = ['smi_' + task for task in dataset_task_list]
    _ablation = _ablation_flags_from_env()
    dataset_list = [
        UniDataset(
            root=args.root,
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            graph_encoder_type=args.graph_encoder_type,
            graph_input=args.graph_input,
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
            modalities=args.modalities,
            g_family_arm=args.g_family_arm,
        relation_geometry_sidecar=args.relation_geometry_sidecar,
        relation_geometry_artifact_hash=args.relation_geometry_artifact_hash,
        g3_permutation_sidecar=args.g3_permutation_sidecar,
        g3_permutation_artifact_hash=args.g3_permutation_artifact_hash,
            ablation_config=(
                {
                    "id": _ablation["ablation_id"],
                    "use_star_rbf": _ablation["use_star_rbf"],
                    "use_mcl": _ablation["use_mcl"],
                    "mcl_mask_mode": _ablation["mcl_mask_mode"],
                    "random_mask_sidecar": _ablation["random_mask_sidecar"],
                }
                if _ablation["ablation_id"]
                else None
            ),
        )
        for dataset_name in dataset_name_list
    ]
    dataset_by_task = dict(zip(dataset_task_list, dataset_list))
    raw_targets_by_task = {
        task: np.asarray(dataset.raw_targets, dtype=np.float64)
        for task, dataset in dataset_by_task.items()
    }
    if args.graph_encoder_type == "mips_trimer_scage" and not args.cache_only:
        cache_dataset = dataset_list[0]
        original_layers = cache_dataset.cache_layers
        cache_dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
        specs = cache_dataset._lmdb_cache_specs({})
        cache_dataset.cache_layers = original_layers
        unfrozen = [
            name for name, spec in specs.items()
            if not os.path.isfile(os.path.join(spec["root"], ".frozen"))
        ]
        if unfrozen:
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} training requires frozen cache artifacts: "
                + ", ".join(unfrozen)
            )
    pretraining_cohort_hash = (
        _cohort_hash_from_current_manifest(args.root, args.checkpoint_pretraining_dataset)
        if args.graph_encoder_type == "mips_trimer_scage"
        else None
    )
    expected_angle_cache_artifact_hash = _angle_artifact_hash_for_cohort(
        args.root, pretraining_cohort_hash
    )
    if args.graph_encoder_type == "mips_trimer_scage" and not args.cache_only:
        if expected_angle_cache_artifact_hash is None:
            raise RuntimeError(
                "MTS property fine-tuning requires the frozen PI1M_v2 "
                "bond-angle artifact"
            )
    if args.cache_only:
        print(
            "Downstream feature-cache prebuild complete: "
            + ", ".join(
                f"{task}={len(dataset_by_task[task])}"
                for task in dataset_task_list
            )
        )
        return
    raw_target_maps = {}
    for task, dataset in dataset_by_task.items():
        target_map = {}
        for data, value in zip(dataset, raw_targets_by_task[task]):
            smiles = str(data.smiles)
            if smiles in target_map and not np.isclose(target_map[smiles], value):
                raise ValueError(f'Conflicting duplicate labels for {task}: {smiles}')
            target_map[smiles] = float(value)
        raw_target_maps[task] = target_map

    def configure_cross_task_targets(task, dataset, training_indices, scope):
        auxiliary_tasks = (
            CROSS_TASK_AUXILIARY_MAP.get(task, ())
            if (
                float(args.cross_task_aux_weight) > 0
                and (
                    not selected_cross_task_aux
                    or task in selected_cross_task_aux
                )
            ) else ()
        )
        auxiliary_tasks = tuple(
            name for name in auxiliary_tasks if name in raw_target_maps
        )
        aux_values = np.zeros(
            (len(dataset), len(auxiliary_tasks)), dtype=np.float32
        )
        aux_masks = np.zeros_like(aux_values, dtype=bool)
        for aux_idx, auxiliary_task in enumerate(auxiliary_tasks):
            auxiliary_map = raw_target_maps[auxiliary_task]
            matched_train = [
                int(index) for index in training_indices
                if str(dataset[int(index)].smiles) in auxiliary_map
            ]
            if not matched_train:
                continue
            auxiliary_values = np.array([
                auxiliary_map[str(dataset[index].smiles)]
                for index in matched_train
            ], dtype=np.float64)
            auxiliary_scaler = TargetScaler(
                auxiliary_task,
                StandardScaler(),
                transform_mode=args.target_transform,
            )
            auxiliary_scaler.scaler.fit(
                auxiliary_scaler._pre_transform(
                    auxiliary_values.reshape(-1, 1)
                )
            )
            scaled_values = auxiliary_scaler.transform(
                auxiliary_values.reshape(-1, 1)
            ).reshape(-1)
            for index, value in zip(matched_train, scaled_values):
                aux_values[index, aux_idx] = float(value)
                aux_masks[index, aux_idx] = True
            print(
                f'Cross-task auxiliary {task} <- {auxiliary_task}: '
                f'{len(matched_train)}/{len(training_indices)} {scope} labels; '
                'all samples outside that training partition excluded'
            )
        dataset.set_auxiliary_target_overrides(aux_values, aux_masks)
        return auxiliary_tasks

    def build_pcgrad_fold(task, fold_train_indices, test_indices, ordered_smiles, fold_seed):
        """Build an eight-task balanced training view without target-fold leakage."""
        task_order = (task,) + tuple(
            name for name in dataset_task_list if name != task
        )
        held_out_keys = {
            sample_key_from_smiles(ordered_smiles[int(index)])
            for index in test_indices
        }
        entries = []
        counts = []
        for task_index, name in enumerate(task_order):
            source = dataset_by_task[name]
            source_smiles = list(getattr(source, "_row_smiles", ()))
            if len(source_smiles) != len(source):
                raise RuntimeError(f"missing immutable source-row SMILES for {name}")
            if name == task:
                allowed = [int(index) for index in fold_train_indices]
            else:
                allowed = [
                    index for index, smiles in enumerate(source_smiles)
                    if sample_key_from_smiles(smiles) not in held_out_keys
                ]
            if not allowed:
                raise RuntimeError(f"no leakage-safe multitask rows remain for {name}")
            values = raw_targets_by_task[name][allowed]
            task_scaler = TargetScaler(
                name, StandardScaler(), transform_mode=args.target_transform
            )
            task_scaler.scaler.fit(task_scaler._pre_transform(values.reshape(-1, 1)))
            scaled = task_scaler.transform(values.reshape(-1, 1)).reshape(-1)
            entries.extend(
                (source, row, value, task_index)
                for row, value in zip(allowed, scaled)
            )
            counts.append(len(allowed))
        multitask_dataset = _MTSMultiTaskFoldDataset(entries)
        weights = []
        cursor = 0
        for count in counts:
            weights.extend([1.0 / float(count)] * count)
            cursor += count
        generator = torch.Generator().manual_seed(int(fold_seed))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(counts) * len(counts), replacement=True,
            generator=generator,
        )
        return multitask_dataset, sampler, task_order[1:]

    # Feature-cache workers must be created before this process initializes
    # CUDA. Cache workers are CPU-only and PolyGen's CPU optimizer must not
    # inherit the downstream training CUDA context.
    set_global_seed(args.seed)
    if args.graph_encoder_type == "mips_trimer_scage" and not torch.cuda.is_available():
        raise RuntimeError(f"{MTS_ROUTE_NAME} Stage 3 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience
    t1_init_artifact = False
    checkpoint_meta = {}

    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_by_task[task]
        raw_targets = raw_targets_by_task[task]
        # ``--results_dir`` historically had two meanings.  Treat an
        # existing directory (or a path without a CSV suffix) as a result
        # root and give every task its own file; this prevents concurrent
        # Stage-3 workers from writing the same CSV.
        result_root = Path(result_output_dir)
        if result_root.exists() and result_root.is_dir():
            task_result_output = result_root / f"{task}.csv"
        elif len(task_list) > 1 and result_root.suffix.lower() != ".csv":
            task_result_output = result_root / f"{task}.csv"
        else:
            task_result_output = result_root
        task_result_output = str(task_result_output)
        task_result_file_initialized = False

        print("Start 5-fold Cross Validation")
        if args.graph_encoder_type == 'mips_trimer_scage':
            manifest_path = Path(args.split_manifest_dir) / f"{task}.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Missing fixed split manifest: {manifest_path}. Run "
                    "scripts/create_mips_split_manifests.py first."
                )
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            # The manifest indexes are defined over the task CSV rows.  Do
            # not hash ``data.smiles`` from the feature store here: content
            # keyed caches may return a representative P-SMILES for a
            # chemically equivalent row (the source task row can therefore
            # differ while resolving to the same feature key).  Such a
            # representative string is not the split order and made valid
            # manifests fail spuriously.  The Dataset preserves the CSV row
            # order, so bind the fixed folds to that immutable source order.
            task_csv = Path(args.root) / "raw" / f"smi_{task}.csv"
            if not task_csv.is_file():
                raise RuntimeError(
                    f"Task CSV required for fixed split validation is missing: "
                    f"{task_csv}"
                )
            ordered_smiles = (
                pd.read_csv(task_csv, usecols=[0])
                .iloc[:, 0]
                .astype(str)
                .str.strip()
                .tolist()
            )
            order_hash = hashlib.sha256(
                "\n".join(ordered_smiles).encode("utf-8")
            ).hexdigest()
            if (
                manifest.get("schema")
                != "mips-shared-validation-test-fold-v1"
                or int(manifest.get("sample_count", -1)) != len(dataset)
                or manifest.get("sample_order_hash") != order_hash
                or not bool(manifest.get("validation_is_test", False))
            ):
                raise RuntimeError(
                    f"Fixed split manifest does not match task cohort: {manifest_path}"
                )
            splits = [
                (
                    np.asarray(item["train_indices"], dtype=np.int64),
                    np.asarray(item["test_indices"], dtype=np.int64),
                )
                for item in manifest["folds"]
            ]
            split_hash = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest()
        else:
            splits = list(
                KFold(n_splits=5, shuffle=True, random_state=1).split(
                    np.arange(len(dataset))
                )
            )
            split_hash = "legacy-inline-kfold-random-state-1"
        fold_metrics = []
        fold_attention_weights = []
        fold_named_attention_weights = {}
        best_fold_val_r2 = -float('inf')
        best_model_state = None

        selected_folds = set(args.fold_ids)
        if not selected_folds or any(fold < 0 or fold >= 5 for fold in selected_folds):
            raise ValueError("--fold_ids must contain one or more values from 0 to 4")
        for fold, (train_indices, test_indices) in enumerate(splits):
            if fold not in selected_folds:
                continue
            fold_started = time.monotonic()
            print(f"\nFold {fold + 1}")
            task_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(task))
            fold_seed = int(args.seed) + 1009 * task_offset + fold
            set_global_seed(fold_seed)
            print(f"Fold seed: {fold_seed}")
            if args.evaluation_protocol == 'nested5':
                ranked = sorted(
                    (hashlib.sha256(sample_key_from_smiles(ordered_smiles[int(index)])).digest(), int(index))
                    for index in train_indices
                )
                inner_count = max(1, int(round(0.10 * len(ranked))))
                val_set = {index for _, index in ranked[:inner_count]}
                val_indices = np.asarray(sorted(val_set), dtype=np.int64)
                fold_train_indices = np.asarray(
                    [int(index) for index in train_indices if int(index) not in val_set],
                    dtype=np.int64,
                )
                print("Nested 5-fold protocol: outer test is excluded from model selection")
            else:
                fold_train_indices = train_indices
                val_indices = test_indices
                print("Shared 5-fold protocol: validation and test use the same held-out fold")
            print(
                f"Fold partitions: train={len(fold_train_indices)}, "
                f"validation={len(val_indices)}, test={len(test_indices)}"
            )
            scaler = scale_targets(
                dataset,
                task,
                train_indices=fold_train_indices,
                raw_targets=raw_targets,
                transform_mode=args.target_transform,
            )
            if args.finetune_mode == 'multitask_pcgrad':
                dataset.clear_auxiliary_target_overrides()
                multitask_dataset, multitask_sampler, auxiliary_tasks = build_pcgrad_fold(
                    task, fold_train_indices, test_indices, ordered_smiles, fold_seed
                )
                train_loader = get_data_loader(
                    multitask_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=multitask_sampler,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
            else:
                auxiliary_tasks = configure_cross_task_targets(
                    task, dataset, fold_train_indices, 'fold-train'
                )
                train_loader = get_data_loader(
                    dataset,
                    indices=fold_train_indices,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
            test_loader = get_data_loader(
                dataset,
                indices=test_indices,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            val_loader = get_data_loader(
                dataset,
                indices=val_indices,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )

            model = UniEncoderAttention(
                joint_embedding_dim=args.joint_embedding_dim,
                smiles_model_name=pre_trained_model_dict['smiles_model_name'],
                gnn_model_name=pre_trained_model_dict['gnn_model_name'],
                modality_list=model_modality_list,
                freeze_encoder=freeze_encoder,
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
                controlled_modality=args.controlled_modality,
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
                use_star_rbf=_ablation["use_star_rbf"],
                use_mcl=_ablation["use_mcl"],
                mcl_mask_mode=_ablation["mcl_mask_mode"],
                fusion_type=args.fusion_type,
                fp_mode=args.fp_mode,
                fusion_dropout=args.fusion_dropout,
                head_dropout=args.head_dropout,
                unimodal_auxiliary=args.unimodal_aux_weight > 0,
                cross_task_auxiliary_tasks=auxiliary_tasks,
                fp_bit_dropout=args.fp_bit_dropout,
                modality_dropout={
                    'fp': args.fp_modality_dropout,
                    'smiles': args.smiles_modality_dropout,
                    'graph': args.graph_modality_dropout,
                },
            )
            if pretrained_model_path:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                if args.graph_encoder_type == 'mips_trimer_scage':
                    expected_schema = checkpoint.get('meta', {}).get('schema') if isinstance(checkpoint, dict) else None
                    if expected_schema not in {
                        MIPS_TRIMER_CHECKPOINT_SCHEMA,
                        MTS_PRETRAIN_CHECKPOINT_SCHEMA,
                    }:
                        raise RuntimeError(
                            "MTS requires an mts-model-v3 historical or mts-model-v4 "
                            "joint-pretraining checkpoint. "
                            "Rerun MTS Joint Pretraining."
                        )
                    checkpoint_stage = checkpoint.get('meta', {}).get('stage')
                    expected_stage = MTS_STAGE1_ID
                    if checkpoint_stage != expected_stage:
                        raise RuntimeError(
                            "MTS property fine-tuning checkpoint stage mismatch; "
                            f"expected={expected_stage!r}, "
                            f"received stage={checkpoint_stage!r}."
                        )
                    checkpoint_meta = checkpoint.get("meta", {})
                    original_checkpoint_meta = checkpoint_meta
                    validate_g_family_checkpoint_binding(
                        checkpoint_meta, args, allow_smoke=g_family_smoke
                    )
                    t1_init_artifact = _is_mts_t1_function_preserving_init(
                        checkpoint_meta
                    )
                    checkpoint_representation = checkpoint_meta.get(
                        "topology_representation", TOPOLOGY_CANONICAL
                    )
                    if (
                        expected_schema == MTS_PRETRAIN_CHECKPOINT_SCHEMA
                        and checkpoint_meta.get("pretrain_profile") is None
                    ):
                        raise RuntimeError(
                            "mts-model-v4 checkpoint is missing its immutable pretrain profile"
                        )
                    checkpoint_angle_schema = checkpoint_meta.get(
                        "angle_cache_schema"
                    )
                    expected_checkpoint_angle_artifact = (
                        _continuous_angle_artifact_hash_for_cohort(
                            args.root, pretraining_cohort_hash
                        )
                        if checkpoint_angle_schema == "mts-angle-continuous-cache-v1"
                        else expected_angle_cache_artifact_hash
                    )
                    # Plan §4: the target contract binds the frozen store and
                    # the .frozen layer payloads byte-for-byte, and its digest
                    # must recompute over every field except itself.
                    _tc_store_sha = _sha256_file(
                        Path(cache_specs["topology"]["root"]).parents[1]
                        / "validation" / "store.json"
                    )
                    _tc_topo_frozen = _sha256_file(
                        Path(cache_specs["topology"]["root"]) / ".frozen"
                    )
                    _tc_trimer_frozen = _sha256_file(
                        Path(cache_specs["trimer"]["root"]) / ".frozen"
                    )
                    if t1_init_artifact:
                        _validate_mts_t1_function_preserving_init(
                            checkpoint,
                            pretrained_model_path,
                            args=args,
                            dataset=dataset,
                            pretraining_cohort_hash=pretraining_cohort_hash,
                            expected_angle_artifact=expected_checkpoint_angle_artifact,
                            store_json_sha256=_tc_store_sha,
                            topology_frozen_payload_sha256=_tc_topo_frozen,
                            trimer_frozen_payload_sha256=_tc_trimer_frozen,
                            model=model,
                        )
                        # The common gate below describes a completed 20k
                        # pretraining checkpoint.  Validate a local view only;
                        # restore the original T1 init metadata before writing
                        # any downstream result shard.
                        from src.dataset.mips_trimer_contract import _canonical_json_hash

                        validation_target = dict(
                            original_checkpoint_meta["target_contract"]
                        )
                        validation_target["graph_model_config_hash"] = (
                            args.graph_model_config_hash
                        )
                        validation_target["target_contract_sha256"] = _canonical_json_hash(
                            {
                                key: value
                                for key, value in validation_target.items()
                                if key != "target_contract_sha256"
                            }
                        )
                        checkpoint_meta = dict(original_checkpoint_meta)
                        checkpoint_meta["optimizer_steps"] = 20000
                        checkpoint_meta["target_contract"] = validation_target
                    # Fresh paired pretraining checkpoints carry the
                    # authoritative source-geometry identity inside
                    # source_contract.  Older checkpoint writers also emitted
                    # the resolved geometry hash in the top-level field; use
                    # the contract value when it is present and valid rather
                    # than rejecting an otherwise strictly bound checkpoint.
                    checkpoint_source_geometry_hash = checkpoint_meta.get(
                        "source_geometry_model_config_hash",
                        checkpoint_meta.get("geometry_model_config_hash"),
                    )
                    source_contract = checkpoint_meta.get("source_contract")
                    if isinstance(source_contract, dict):
                        contract_source_geometry_hash = source_contract.get(
                            "source_geometry_model_config_hash"
                        )
                        if contract_source_geometry_hash:
                            checkpoint_source_geometry_hash = contract_source_geometry_hash
                    if (
                        checkpoint_meta.get("baseline") != MTS_ROUTE_NAME
                        or checkpoint_representation
                        != args.topology_representation
                        or checkpoint_meta.get("route_short_name") != MTS_ROUTE_SHORT_NAME
                        # The migrated checkpoint records the pre-migration
                        # schema snapshot names (config v2, feature v4, bundle
                        # v1, no topology_lmdb_schema, lga version 1) while the
                        # frozen canonical cache uses the post-migration names.
                        # These snapshots are not part of the frozen contract;
                        # the authoritative bindings are the cache_bundle_hash
                        # (computed from source cohort + layer artifact ids)
                        # and the strict state-dict load, both validated below.
                        or checkpoint_meta.get("cache_layout_schema")
                        != MIPS_TRIMER_CACHE_LAYOUT_SCHEMA
                        or not checkpoint_meta.get("cache_bundle_hash")
                        or checkpoint_meta.get("cache_bundle_hash")
                        != cache_bundle_binding_hash(
                            cohort_hash=checkpoint_meta.get("source_cohort_hash"),
                            topology_artifact_hash=getattr(
                                dataset, "topology_cache_artifact_hash", None
                            ),
                            trimer_artifact_hash=getattr(
                                dataset, "trimer_cache_artifact_hash", None
                            ),
                        )
                        or checkpoint_meta.get("mips_core") != args.mips_core
                        or int(checkpoint_meta.get("mips_max_hops", -1))
                        != int(args.mips_max_hops)
                        or bool(checkpoint_meta.get("mips_use_descriptors", False))
                        != bool(args.mips_use_descriptors)
                        or checkpoint_meta.get("spatial_mode", "none")
                        != args.spatial_mode
                        # The joint checkpoint is the immutable topology/MCL
                        # source artifact.  A G/V/MT experiment may resolve a
                        # different downstream geometry mode, so its resolved
                        # geometry hash belongs to the shard identity rather
                        # than being compared to the source checkpoint hash.
                        # What must match here is the explicit source hash.
                        or checkpoint_source_geometry_hash
                        != args.source_geometry_model_config_hash
                        or (
                            args.graph_encoder_type == "mips_trimer_scage"
                            and not checkpoint_meta.get("trimer_cache_hash")
                        )
                        # The migrated checkpoint intentionally carries the
                        # source model's feature/o8/graph config hashes (its
                        # pre-migration identity), while the frozen canonical
                        # cache and the resolved experiment use the post-migration
                        # schema names, so the hashes no longer compare equal.
                        # Model compatibility is enforced structurally by the
                        # strict state-dict load and by the artifact/source
                        # bindings below; the snapshot hashes are not part of
                        # the frozen contract (mirrors the doctor's binding).
                        or checkpoint_meta.get("mips_variant")
                        != args.mips_variant
                        or checkpoint_meta.get("topology_attention_variant", "o8")
                        != args.topology_attention_variant
                        or checkpoint_meta.get("g_family_arm")
                        != getattr(args, "g_family_arm", None)
                        or checkpoint_meta.get("shared_step0_id")
                        != getattr(args, "shared_step0_id", None)
                        or (
                            getattr(args, "g_family_arm", None) is not None
                            and checkpoint_meta.get("g_family_bundle_hash")
                            != getattr(args, "g_family_bundle_hash", None)
                        )
                        or list(checkpoint_meta.get(
                            "msta_layer_indices", [4, 5]
                        )) != list(args.msta_layer_indices)
                        or list(checkpoint_meta.get(
                            "msta_local_spd", [0, 1]
                        )) != list(args.msta_local_spd)
                        or list(checkpoint_meta.get(
                            "msta_context_spd", [0, 1, 2]
                        )) != list(args.msta_context_spd)
                        or (
                            expected_stage == "alignment"
                            and checkpoint_meta.get(
                                "alignment_model_config_hash"
                            ) != args.alignment_model_config_hash
                        )
                        or int(checkpoint_meta.get("random_seed", -1))
                        != int(
                            args.checkpoint_seed
                            if args.checkpoint_seed is not None else args.seed
                        )
                        or checkpoint_meta.get("pretraining_dataset")
                        != "PI1M_v2"
                        or (
                            args.checkpoint_pretraining_dataset
                            and checkpoint_meta.get("pretraining_dataset")
                            != args.checkpoint_pretraining_dataset
                        )
                        or (
                            args.checkpoint_tier
                            and checkpoint_meta.get("tier")
                            != args.checkpoint_tier
                        )
                        or checkpoint_meta.get("trimer_conformer_protocol")
                        != MIPS_TRIMER_PROTOCOL
                        # The content/lmdb schema names and builder version in
                        # the migrated checkpoint are pre-migration snapshots
                        # (e.g. trimer v5/builder 10) while the frozen cache
                        # was rebuilt under v8/builder 12.  The authoritative
                        # binding is the Trimer .done artifact id, which is
                        # validated below and matches; the snapshot names are
                        # not part of the frozen contract, mirroring the
                        # doctor's artifact-based binding check.
                        or int(checkpoint_meta.get("trimer_mmff_relax_steps", -1))
                        != MIPS_TRIMER_MMFF_RELAX_STEPS
                        or bool(checkpoint_meta.get("trimer_require_mmff_convergence", True))
                        != MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE
                        or checkpoint_meta.get("trimer_acceptance")
                        != MIPS_TRIMER_ACCEPTANCE
                        or checkpoint_meta.get("trimer_selection")
                        != MIPS_TRIMER_SELECTION
                        # The layer "cache hash" is the legacy directory-name
                        # identity of the migration-time layer root; after the
                        # canonical single-RU migration the roots were rebuilt
                        # under new names while the authoritative .done artifact
                        # id is unchanged.  Binding is validated via the artifact
                        # hashes below, matching the doctor's contract; the old
                        # directory-name equality is not part of the frozen
                        # contract and would reject an otherwise identical cache.
                        or checkpoint_meta.get("topology_cache_artifact_hash")
                            != getattr(dataset, "topology_cache_artifact_hash", None)
                        or not checkpoint_meta.get("trimer_cache_artifact_hash")
                        or checkpoint_meta.get("trimer_cache_artifact_hash")
                            != getattr(dataset, "trimer_cache_artifact_hash", None)
                        or not checkpoint_meta.get("source_cohort_hash")
                        or (
                            args.checkpoint_pretraining_dataset
                            and checkpoint_meta.get("source_cohort_hash")
                            != _cohort_hash_from_current_manifest(
                                args.root,
                                args.checkpoint_pretraining_dataset,
                            )
                        )
                        or (
                            int(checkpoint_meta.get("optimizer_steps", -1)) != 20000
                            and not (
                                g_family_smoke
                                and getattr(args, "g_family_arm", None) is not None
                                and int(checkpoint_meta.get("optimizer_steps", -1)) in {1, 2}
                                and bool(checkpoint_meta.get("smoke_only", False))
                            )
                        )
                        or checkpoint_angle_schema not in {
                            "mts-trimer-bond-angle-cache-v1",
                            "mts-trimer-bond-angle-cache-v2",
                            "mts-angle-continuous-cache-v1",
                        }
                        or (
                            expected_schema == MTS_PRETRAIN_CHECKPOINT_SCHEMA
                            and (
                                checkpoint_angle_schema != MTS_CATEGORICAL_ANGLE_SCHEMA
                                or checkpoint_meta.get("pretrain_profile", {}).get("profile_id")
                                != MTS_PRETRAIN_PROFILE_ID
                                or checkpoint_meta.get("target_contract", {}).get("schema")
                                != MTS_PRETRAIN_TARGET_CONTRACT_SCHEMA
                            )
                        )
                        or not checkpoint_meta.get("angle_cache_artifact_hash")
                        or checkpoint_meta.get("angle_cache_artifact_hash")
                        != expected_checkpoint_angle_artifact
                        or checkpoint_meta.get("pretraining_objective")
                        != (
                            "masked_atom_only"
                            if getattr(args, "g_family_arm", None) is not None
                            else "masked_atom_plus_trimer_bond_angle"
                        )
                        or checkpoint_meta.get("reference_commits", {}).get("mips")
                        != "26aafe52926a3f33bf2d3d382ae263360319812d"
                        or checkpoint_meta.get("reference_commits", {}).get("scage")
                        != "82bcbb4647e31bf0d413a317e69a2526df75ce01"
                        # The production loader only accepts a full dual-identity
                        # checkpoint: both source_contract and target_contract
                        # must be present, the source digest must match, and
                        # the target contract must bind the current frozen
                        # production identity exactly (Plan §3 / G0-baseline
                        # §3.1).  Legacy pre-dual-contract checkpoints are
                        # rejected.
                        or not isinstance(
                            checkpoint_meta.get("source_contract"), dict
                        )
                        or not isinstance(
                            checkpoint_meta.get("target_contract"), dict
                        )
                        or _source_contract_digest_mismatch(
                            checkpoint_meta.get("source_contract"),
                            checkpoint_meta.get("source_contract_sha256"),
                        )
                        or (
                            not (
                                g_family_smoke
                                and getattr(args, "g_family_arm", None) is not None
                                and bool(checkpoint_meta.get("smoke_only", False))
                            )
                            and _target_contract_mismatch(
                                checkpoint_meta.get("target_contract") or {},
                                args=args,
                                dataset=dataset,
                                pretraining_cohort_hash=pretraining_cohort_hash,
                                expected_angle_artifact=expected_checkpoint_angle_artifact,
                                store_json_sha256=_tc_store_sha,
                                topology_frozen_payload_sha256=_tc_topo_frozen,
                                trimer_frozen_payload_sha256=_tc_trimer_frozen,
                            )
                        )
                    ):
                        raise RuntimeError(
                            "Alignment checkpoint MIPS configuration does not "
                            "match the requested core/hops/descriptor settings."
                        )
                    if t1_init_artifact:
                        checkpoint_meta = original_checkpoint_meta
                    checkpoint_fusion = checkpoint.get('meta', {}).get('fusion_type')
                    checkpoint_state = checkpoint['state_dict']
                else:
                    checkpoint_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model_keys = set(model.state_dict())
                checkpoint_keys = set(checkpoint_state)
                missing = sorted(model_keys - checkpoint_keys)
                unexpected = sorted(checkpoint_keys - model_keys)
                allowed_transfer_missing = set()
                if args.graph_encoder_type == 'mips_trimer_scage':
                    missing, unexpected, incompatible = (
                        _scage_checkpoint_key_compatibility(
                            model_keys,
                            checkpoint_keys,
                            expected_stage,
                            unimodal_aux_weight=args.unimodal_aux_weight,
                            cross_task_aux_weight=args.cross_task_aux_weight,
                        )
                    )
                    if incompatible:
                        if "mcl_rbf" in str(args.graph_geometry_mode):
                            # The completed immutable joint checkpoint predates
                            # the optional MCL-v2 continuous-distance channel.
                            # Permit only newly introduced, deterministically
                            # initialized MCL-v2 tensors to be absent.  An
                            # unexpected checkpoint tensor remains an error.
                            allowed_mcl_v2_missing = {
                                key for key in missing
                                if key.startswith(
                                    "encoders.graph.encoder.trimer_mcl.layers."
                                ) and (
                                    ".distance_projection." in key
                                    or key.endswith(".distance_centers")
                                )
                            }
                            allowed_transfer_missing = set(allowed_mcl_v2_missing)
                            incompatible = [
                                key for key in incompatible
                                if key not in allowed_mcl_v2_missing
                            ]
                    if incompatible:
                        raise RuntimeError(
                            f"{args.graph_encoder_type.upper()} {expected_stage} checkpoint mismatch. "
                            "Re-run MTS Joint Pretraining "
                            "with the same run.sh model configuration. Mismatched keys: "
                            + ", ".join(incompatible[:10])
                        )
                merged_state = model.state_dict()
                if args.graph_encoder_type == 'mips_trimer_scage':
                    # Joint pretraining intentionally exports a complete model
                    # container, but Stage 3 migrates only learned structural
                    # modules.  MD200, graph norm/projection and regression
                    # head remain at their fold-seeded initialization.
                    transfer_keys = select_mts_checkpoint_transfer_keys(
                        merged_state,
                        checkpoint_state,
                        allowed_missing=allowed_transfer_missing,
                    )
                    transferred = {
                        key: checkpoint_state[key] for key in transfer_keys
                    }
                    merged_state.update(transferred)
                    print(
                        'MTS checkpoint migration: loaded topology encoder '
                        '(including MSTA/G-family geometry parameters when present) '
                        f'({len(transferred)} tensors); '
                        'MD200/projection/head reinitialized by fold seed.'
                    )
                else:
                    merged_state.update({
                        key: value for key, value in checkpoint_state.items()
                        if key in merged_state
                    })
                model.load_state_dict(merged_state, strict=True)
                print(f"Loaded pretrained model from {pretrained_model_path}")
                print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
            initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            model.to(device)
            print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

            metrics = train_and_evaluate(
                model, scaler, train_loader, val_loader, test_loader,
                device, num_epochs=epochs, patience=patience, max_grad_norm=args.max_grad_norm,
                smiles_lr=args.smiles_lr, graph_lr=args.graph_lr,
                geom_lr=args.geom_lr, fp_lr=args.fp_lr, fusion_lr=args.fusion_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                freeze_smiles_epochs=args.freeze_smiles_epochs,
                deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                regression_loss=args.regression_loss,
                huber_beta=args.huber_beta,
                unimodal_aux_weight=args.unimodal_aux_weight,
                fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                fusion_prior=args.fusion_prior,
                cross_task_aux_weight=args.cross_task_aux_weight,
                swa_start_epoch=args.swa_start_epoch,
                evaluate_test=not args.refit_full_train,
                return_predictions=not args.refit_full_train,
                mts_o8_lr=args.mts_o8_lr,
                mts_geometry_lr=args.mts_geometry_lr,
                mts_adapter_lr=args.mts_adapter_lr,
                mts_finetune_profile=args.finetune_profile,
                pcgrad=args.finetune_mode == 'multitask_pcgrad',
                amp_dtype=args.amp_dtype,
            )
            # The speed benchmark can request several evaluation batch sizes
            # after training has selected the fold's best model.  All of these
            # evaluations therefore use the exact same in-memory model state,
            # validation split, scaler, and sample order; they are not
            # independent training runs.  The feature is opt-in and has no
            # effect on production launches.
            benchmark_eval_batches = os.environ.get(
                'MTS_BENCHMARK_EVAL_BATCHES', ''
            ).strip()
            if benchmark_eval_batches and args.predictions_dir:
                try:
                    requested_eval_batches = sorted({
                        int(value.strip())
                        for value in benchmark_eval_batches.split(',')
                        if value.strip()
                    })
                except ValueError as exc:
                    raise ValueError(
                        'MTS_BENCHMARK_EVAL_BATCHES must be comma-separated '
                        'positive integers'
                    ) from exc
                if not requested_eval_batches or any(
                    value <= 0 for value in requested_eval_batches
                ):
                    raise ValueError(
                        'MTS_BENCHMARK_EVAL_BATCHES must contain positive '
                        'integers'
                    )
                benchmark_eval_root = Path(os.environ.get(
                    'MTS_BENCHMARK_EVAL_OUTPUT_DIR',
                    str(Path(args.predictions_dir).parent / 'benchmark_eval_predictions'),
                ))
                benchmark_eval_records = []
                for eval_batch_size in requested_eval_batches:
                    eval_started = time.perf_counter()
                    benchmark_loader = get_data_loader(
                        dataset,
                        indices=test_indices,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.loader_workers,
                        pin_memory=True,
                        persistent_workers=args.loader_workers > 0,
                    )
                    benchmark_metrics = test_model(
                        model,
                        benchmark_loader,
                        scaler,
                        device,
                        return_predictions=True,
                        amp_dtype=args.amp_dtype,
                    )
                    benchmark_path = (
                        benchmark_eval_root / f'batch_{eval_batch_size}'
                        / task / f'fold_{fold}.npz'
                    )
                    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
                    benchmark_metadata = {
                        'task': task,
                        'fold': int(fold),
                        'seed': int(args.seed),
                        'fold_seed': int(fold_seed),
                        'eval_batch_size': int(eval_batch_size),
                        'amp_dtype': args.amp_dtype,
                        'loader_workers': int(args.loader_workers),
                        'sample_order': 'test_indices_in_source_order',
                        'model_state_scope': (
                            'same_train_and_evaluate_fold_best_model_state'
                        ),
                        'source_primary_eval_batch_size': int(args.eval_batch_size),
                        'eval_seconds': float(time.perf_counter() - eval_started),
                    }
                    benchmark_tmp = benchmark_path.with_name(
                        benchmark_path.name + f'.tmp.{os.getpid()}'
                    )
                    try:
                        with benchmark_tmp.open('wb') as handle:
                            np.savez(
                                handle,
                                y_true=np.asarray(
                                    benchmark_metrics['_y_true'],
                                    dtype=np.float64,
                                ),
                                y_pred=np.asarray(
                                    benchmark_metrics['_y_pred'],
                                    dtype=np.float64,
                                ),
                                sample_indices=np.asarray(
                                    test_indices, dtype=np.int64
                                ),
                                metadata=np.asarray(json.dumps(
                                    benchmark_metadata, sort_keys=True
                                )),
                            )
                        os.replace(benchmark_tmp, benchmark_path)
                    finally:
                        if benchmark_tmp.exists():
                            benchmark_tmp.unlink()
                    benchmark_eval_records.append({
                        **benchmark_metadata,
                        'path': str(benchmark_path),
                        'test_r2': float(benchmark_metrics['test_r2']),
                        'test_mae': float(benchmark_metrics['test_mae']),
                        'test_rmse': float(benchmark_metrics['test_rmse']),
                    })
                metrics['benchmark_eval_batch_records'] = benchmark_eval_records
            if args.refit_full_train:
                refit_epochs = int(metrics.get('best_epoch', -1))
                if refit_epochs <= 0:
                    raise RuntimeError(
                        'Full-train refit requires a raw validation-selected '
                        'best_epoch; disable SWA or refit_full_train.'
                    )
                print(
                    f'Refitting fold {fold + 1} from the initial checkpoint on '
                    f'all {len(train_indices)} outer-train samples for '
                    f'{refit_epochs} selected epoch(s).'
                )
                refit_scaler = scale_targets(
                    dataset,
                    task,
                    train_indices=train_indices,
                    raw_targets=raw_targets,
                    transform_mode=args.target_transform,
                )
                auxiliary_tasks = configure_cross_task_targets(
                    task, dataset, train_indices, 'outer-train refit'
                )
                refit_train_loader = get_data_loader(
                    dataset,
                    indices=train_indices,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                test_loader = get_data_loader(
                    dataset,
                    indices=test_indices,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                set_global_seed(fold_seed)
                model.load_state_dict(initial_model_state)
                model.to(device)
                refit_details = fit_fixed_epochs(
                    model,
                    refit_train_loader,
                    device,
                    num_epochs=refit_epochs,
                    max_grad_norm=args.max_grad_norm,
                    smiles_lr=args.smiles_lr,
                    graph_lr=args.graph_lr,
                    geom_lr=args.geom_lr,
                    fp_lr=args.fp_lr,
                    fusion_lr=args.fusion_lr,
                    head_lr=args.head_lr,
                    weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs,
                    freeze_smiles_epochs=args.freeze_smiles_epochs,
                    deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                    fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                    regression_loss=args.regression_loss,
                    huber_beta=args.huber_beta,
                    unimodal_aux_weight=args.unimodal_aux_weight,
                    fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                    fusion_prior=args.fusion_prior,
                    cross_task_aux_weight=args.cross_task_aux_weight,
                    mts_o8_lr=args.mts_o8_lr,
                    mts_geometry_lr=args.mts_geometry_lr,
                    mts_adapter_lr=args.mts_adapter_lr,
                    mts_finetune_profile=args.finetune_profile,
                    amp_dtype=args.amp_dtype,
                )
                refit_test_metrics = test_model(
                    model, test_loader, refit_scaler, device,
                    return_predictions=True,
                    amp_dtype=args.amp_dtype,
                )
                metrics.update(refit_test_metrics)
                metrics.update(refit_details)
                metrics['refit_full_train'] = True
            else:
                metrics['refit_full_train'] = False
                metrics['refit_epochs'] = 0
            if (
                args.graph_encoder_type == 'mips_trimer_scage'
                and len(args.modalities) > 1
                and hasattr(model, 'modality_control')
            ):
                original_control = model.modality_control
                original_modality = model.controlled_modality
                for modality in model.modality_list:
                    model.controlled_modality = modality
                    for control in ('batch_shuffled', 'constant_zero'):
                        model.modality_control = control
                        control_scaler = (
                            refit_scaler if args.refit_full_train else scaler
                        )
                        controlled = test_model(
                            model, test_loader, control_scaler, device,
                            amp_dtype=args.amp_dtype,
                        )
                        for key, value in controlled.items():
                            metrics[
                                f'{modality}_{control}_{key}'
                            ] = float(value)
                model.modality_control = original_control
                model.controlled_modality = original_modality
            prediction_true = metrics.pop('_y_true', None)
            prediction_values = metrics.pop('_y_pred', None)
            prediction_path = None
            prediction_sha256 = None
            if args.graph_encoder_type == 'mips_trimer_scage':
                if prediction_true is None or prediction_values is None:
                    raise RuntimeError('MTS fold did not produce raw-space predictions')
                if not args.predictions_dir:
                    raise RuntimeError('MTS Stage 3 requires --predictions_dir')
                prediction_path = (
                    Path(args.predictions_dir) / task / f'fold_{fold}.npz'
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_tmp = prediction_path.with_name(
                    prediction_path.name + f'.tmp.{os.getpid()}'
                )
                metadata = {
                    'task': task,
                    'fold': int(fold),
                    'seed': int(args.seed),
                    'fold_seed': int(fold_seed),
                    'finetune_config_hash': args.finetune_config_hash,
                    'finetune_profile': args.finetune_profile,
                    'finetune_profile_hash': args.finetune_profile_hash,
                    'checkpoint_sha256': args.checkpoint_sha256,
                    'cache_store_sha256': args.cache_store_sha256,
                    'split_manifest_hash': split_hash,
                    'fold_validation_protocol': (
                        'nested_outer5_inner_hash10'
                        if args.evaluation_protocol == 'nested5'
                        else 'shared_validation_test_fold'
                    ),
                    'independent_blind_test': args.evaluation_protocol == 'nested5',
                    'amp_dtype': args.amp_dtype,
                    'train_batch_size': int(args.batch_size),
                    'eval_batch_size': int(args.eval_batch_size),
                    'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                }
                try:
                    with prediction_tmp.open('wb') as handle:
                        np.savez(
                            handle,
                            y_true=np.asarray(prediction_true, dtype=np.float64),
                            y_pred=np.asarray(prediction_values, dtype=np.float64),
                            sample_indices=np.asarray(test_indices, dtype=np.int64),
                            metadata=np.asarray(
                                json.dumps(metadata, sort_keys=True)
                            ),
                        )
                    os.replace(prediction_tmp, prediction_path)
                    prediction_sha256 = hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest()
                finally:
                    if prediction_tmp.exists():
                        prediction_tmp.unlink()
            metrics["fold_wall_seconds"] = float(
                time.monotonic() - fold_started
            )
            metrics["fold"] = int(fold)
            fold_attention, attention_shape, attention_labels, fold_named_attention = collect_attention_pooling_weights(model, test_loader, device)
            fold_attention_weights.append(fold_attention)
            for name, (weights, labels) in fold_named_attention.items():
                fold_named_attention_weights.setdefault(name, {'labels': labels, 'weights': []})
                fold_named_attention_weights[name]['weights'].append(weights)

            fold_metrics.append(metrics)
            print(
                f"Fold {fold + 1} Test R2: {metrics['test_r2']:.3f}, "
                f"MAE: {metrics['test_mae']:.3f}, RMSE: {metrics['test_rmse']:.3f}"
            )
            print(f"Fold {fold + 1} attention shape: {attention_shape}")
            print(f"Fold {fold + 1} mean attention: {format_attention_weights(attention_labels, fold_attention)}")

            if best_model_state is None or metrics['best_val_r2'] > best_fold_val_r2:
                best_fold_val_r2 = metrics['best_val_r2']
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            torch.cuda.empty_cache()

        cv_attention = np.mean(np.stack(fold_attention_weights, axis=0), axis=0)

        avg_test_r2 = np.mean([metric['test_r2'] for metric in fold_metrics])
        std_test_r2 = np.std([metric['test_r2'] for metric in fold_metrics])
        avg_test_mae = np.mean([metric['test_mae'] for metric in fold_metrics])
        std_test_mae = np.std([metric['test_mae'] for metric in fold_metrics])
        avg_test_rmse = np.mean([metric['test_rmse'] for metric in fold_metrics])
        std_test_rmse = np.std([metric['test_rmse'] for metric in fold_metrics])
        avg_val_r2 = np.mean([metric['best_val_r2'] for metric in fold_metrics])
        std_val_r2 = np.std([metric['best_val_r2'] for metric in fold_metrics])

        print("\nAverage of metrics over all folds")
        print(f"Test R2 = {avg_test_r2:.3f}")
        print(f"Test MAE = {avg_test_mae:.3f}")
        print(f"Test RMSE = {avg_test_rmse:.3f}")
        print(f"Standard Deviation of Test R2 = {std_test_r2:.3f}")
        print(f"Standard Deviation of Test MAE = {std_test_mae:.3f}")
        print(f"Standard Deviation of Test RMSE = {std_test_rmse:.3f}")
        print(f"Best Validation R2 = {avg_val_r2:.3f} +/- {std_val_r2:.3f}")
        print(f"5-fold mean attention: {format_attention_weights(attention_labels, cv_attention)}")

        # Downstream model persistence is intentionally disabled. The best
        # fold state remains in memory for evaluation, but saved_models is not
        # populated after training.
        # os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)
        # torch.save(best_model_state, os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth'))
        # print(f"Best fold model saved by validation R2: {best_fold_val_r2:.3f}")

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            # Keep every result shard self-describing.  This is intentionally
            # redundant with the checkpoint metadata: a shard must be safe to
            # resume/merge without consulting a mutable command line or the
            # current cache directory.
            'experiment_id': args.experiment_id,
            'config_hash': args.resolved_config_hash,
            'resolved_config_hash': args.resolved_config_hash,
            'feature_config_hash': args.feature_config_hash,
            'graph_model_config_hash': args.graph_model_config_hash,
            'geometry_model_config_hash': args.geometry_model_config_hash,
            'source_geometry_model_config_hash': args.source_geometry_model_config_hash,
            'training_config_hash': args.training_config_hash,
            'finetune_config_hash': args.finetune_config_hash,
            'finetune_profile': args.finetune_profile,
            'finetune_profile_hash': args.finetune_profile_hash,
            'amp_dtype': args.amp_dtype,
            'train_batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'checkpoint_schema': (
                checkpoint_meta.get('checkpoint_schema')
                or checkpoint_meta.get('schema')
                if pretrained_model_path else None
            ),
            'checkpoint_path': str(pretrained_model_path) if pretrained_model_path else None,
            'checkpoint_sha256': args.checkpoint_sha256 or None,
            'checkpoint_model_identity': (
                checkpoint_meta.get('model_identity')
                if pretrained_model_path else None
            ),
            'checkpoint_initialization': (
                checkpoint_meta.get('initialization')
                if pretrained_model_path else None
            ),
            'checkpoint_init_opt_in': bool(
                t1_init_artifact and args.allow_mts_t1_function_preserving_init
            ) if pretrained_model_path else False,
            'checkpoint_optimizer_steps': (
                checkpoint_meta.get('optimizer_steps')
                if pretrained_model_path else None
            ),
            'checkpoint_source_optimizer_steps': (
                checkpoint_meta.get('source_optimizer_steps')
                if pretrained_model_path else None
            ),
            'cache_store_sha256': args.cache_store_sha256 or None,
            'checkpoint_cache_bundle_hash': (
                checkpoint_meta.get('cache_bundle_hash')
                if pretrained_model_path else None
            ),
            'cache_bundle_hash': (
                checkpoint_meta.get('cache_bundle_hash')
                if pretrained_model_path else None
            ),
            'model_modality_list': model_modality_list,
            'fusion_type': args.fusion_type,
            'fp_mode': args.fp_mode,
            'fp_dim': {
                'ecfp': 1024,
                'mixfp': 1048,
                'attachment_count': 2570,
                'disabled': 0,
            }[args.fp_mode],
            'mips_core': args.mips_core if args.graph_encoder_type == 'mips_trimer_scage' else None,
            'mips_max_hops': args.mips_max_hops if args.graph_encoder_type == 'mips_trimer_scage' else None,
            'mips_use_descriptors': (
                bool(args.mips_use_descriptors)
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'spatial_mode': (
                args.spatial_mode if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'graph_geometry_mode': (
                args.graph_geometry_mode
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'topology_attention_variant': args.topology_attention_variant,
            'g_family_arm': getattr(args, 'g_family_arm', None),
            'g_family_bundle_hash': getattr(args, 'g_family_bundle_hash', None),
            'checkpoint_g_family_bundle_hash': (
                checkpoint_meta.get('g_family_bundle_hash')
                if pretrained_model_path else None
            ),
            'relation_geometry_sidecar': getattr(args, 'relation_geometry_sidecar', None),
            'relation_geometry_artifact_hash': getattr(args, 'relation_geometry_artifact_hash', None),
            'g3_permutation_sidecar': getattr(args, 'g3_permutation_sidecar', None),
            'g3_permutation_artifact_hash': getattr(args, 'g3_permutation_artifact_hash', None),
            'smoke_only': bool(g_family_smoke),
            'shared_step0_id': getattr(args, 'shared_step0_id', None),
            'trimer_cache_hash': (
                getattr(dataset, 'trimer_cache_hash', None)
                if args.graph_encoder_type == 'mips_trimer_scage'
                else None
            ),
            'topology_cache_artifact_hash': (
                getattr(dataset, 'topology_cache_artifact_hash', None)
                if args.graph_encoder_type == 'mips_trimer_scage'
                else None
            ),
            'trimer_cache_artifact_hash': (
                getattr(dataset, 'trimer_cache_artifact_hash', None)
                if args.graph_encoder_type == 'mips_trimer_scage'
                else None
            ),
            'source_cohort_hash': (
                checkpoint_meta.get('source_cohort_hash')
                if (
                    args.graph_encoder_type == 'mips_trimer_scage'
                    and pretrained_model_path
                )
                else None
            ),
            'feature_cohort_hash': getattr(
                dataset, 'feature_cohort_hash', None
            ),
            'feature_cache_item_timeout': int(
                args.feature_cache_item_timeout
            ),
            'alignment_trimer_cache_hash': (
                checkpoint_meta.get('trimer_cache_hash')
                if (
                    args.graph_encoder_type == 'mips_trimer_scage'
                    and pretrained_model_path
                )
                else None
            ),
            'mcl_distance_percentiles': (
                list(args.mcl_distance_percentiles)
                if args.graph_encoder_type == 'mips_trimer_scage' else []
            ),
            'mts_sidecar_hash': getattr(dataset, 'mts_sidecar_hash', None),
            'evaluation_protocol': args.evaluation_protocol,
            'mips_fusion_mode': args.mips_fusion_mode,
            'projection_mode': args.projection_mode,
            'modality_control': args.modality_control,
            'mips_variant': (
                args.mips_variant if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'fusion_dropout': args.fusion_dropout,
            'head_dropout': args.head_dropout,
            'fp_bit_dropout': args.fp_bit_dropout,
            'modality_dropout': (
                f"smiles={args.smiles_modality_dropout};graph={args.graph_modality_dropout};"
                f"fp={args.fp_modality_dropout}"
            ),
            'regression_loss': args.regression_loss,
            'finetune_mode': args.finetune_mode,
            'huber_beta': args.huber_beta,
            'unimodal_aux_weight': args.unimodal_aux_weight,
            'cross_task_aux_weight': args.cross_task_aux_weight,
            'cross_task_auxiliary_tasks': (
                ';'.join(CROSS_TASK_AUXILIARY_MAP.get(task, ()))
                if (
                    args.cross_task_aux_weight > 0
                    and (
                        not selected_cross_task_aux
                        or task in selected_cross_task_aux
                    )
                ) else ''
            ),
            'cross_task_aux_target_tasks': ';'.join(
                args.cross_task_aux_tasks
            ),
            'fusion_prior_kl_weight': args.fusion_prior_kl_weight,
            'fusion_prior': args.fusion_prior,
            'swa_start_epoch': args.swa_start_epoch,
            'swa_selected_folds': sum(
                int(metric.get('swa_selected', False)) for metric in fold_metrics
            ),
            'swa_mean_snapshots': np.mean([
                metric.get('swa_snapshots', 0) for metric in fold_metrics
            ]),
            'seed': args.seed,
            'prediction_path': str(prediction_path) if prediction_path else None,
            'prediction_sha256': prediction_sha256,
            'target_transform': args.target_transform,
            'fold_validation_protocol': (
                'nested_outer5_inner_hash10'
                if args.evaluation_protocol == 'nested5'
                else 'shared_validation_test_fold'
            ),
            'independent_blind_test': args.evaluation_protocol == 'nested5',
            'split_manifest_hash': split_hash,
            'refit_full_train': bool(args.refit_full_train),
            'avg_refit_epochs': np.mean([
                metric.get('refit_epochs', 0) for metric in fold_metrics
            ]),
            'avg_best_val_r2': float(avg_val_r2),
            'std_best_val_r2': float(std_val_r2),
            'optimizer_lrs': (
                f"mts_o8={args.mts_o8_lr};mts_geometry={args.mts_geometry_lr};"
                f"mts_adapter={args.mts_adapter_lr};head={args.head_lr}"
                if args.graph_encoder_type == 'mips_trimer_scage' else
                f"smiles={args.smiles_lr};graph={args.graph_lr};"
                f"fp={'frozen' if args.fp_unfreeze_epoch < 0 else args.fp_lr};"
                f"fusion={args.fusion_lr};head={args.head_lr}"
            ),
            'fp_unfreeze_epoch': args.fp_unfreeze_epoch,
            'batch_size': args.batch_size,
            'total_fold_wall_seconds': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            )),
            'estimated_gpu_hours': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            ) / 3600.0),
            'fusion_inputs': attention_labels,
            'graph_input': args.graph_input,
            'geom_input': args.geom_input,
            'graph_encoder_type': args.graph_encoder_type,
            'topology_representation': args.topology_representation,
            'baseline': (
                MTS_ROUTE_NAME
                if args.graph_encoder_type == 'mips_trimer_scage' else 'retired_route'
            ),
            'route_short_name': (
                MTS_ROUTE_SHORT_NAME
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_backbone': (
                'sparse_non_pbc_mips_starlink_spd_single_path_node'
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_input': (
                'mips137_independent_backbone_embedding'
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_checkpoint_schema': (
                MIPS_TRIMER_CHECKPOINT_SCHEMA
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'cache_bundle_schema': (
                MIPS_TRIMER_CACHE_BUNDLE_SCHEMA
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'topology_lmdb_schema': (
                (
                    MIPS_EXPLICIT_TOPOLOGY_SCHEMA
                    if args.topology_representation == TOPOLOGY_EXPLICIT
                    else MIPS_TRIMER_TOPOLOGY_SCHEMA
                )
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'pretraining_dataset': (
                checkpoint_meta.get('pretraining_dataset')
                if args.graph_encoder_type == 'mips_trimer_scage' and pretrained_model_path
                else None
            ),
            'avg_test_r2': float(avg_test_r2),
            'std_test_r2': float(std_test_r2),
            'avg_test_mae': float(avg_test_mae),
            'std_test_mae': float(std_test_mae),
            'avg_test_rmse': float(avg_test_rmse),
            'std_test_rmse': float(std_test_rmse),
            'per_fold_metrics': json.dumps(fold_metrics, sort_keys=True),
            'attention': format_attention_weights(attention_labels, cv_attention),
        }

        # Save to CSV.  Stage-3 campaign units write exactly one fold per
        # shard.  Use an atomic replacement for those paths so an interrupted
        # process can never leave a partially written CSV that looks complete
        # to the resume logic.  Legacy aggregate outputs retain append mode.
        os.makedirs(os.path.dirname(task_result_output) or ".", exist_ok=True)
        results_df = pd.DataFrame([result])
        write_header = not task_result_file_initialized
        shard_path = str(task_result_output).replace("\\", "/")
        atomic_shard = "/shards/" in shard_path and write_header
        if atomic_shard:
            tmp_path = f"{task_result_output}.tmp.{os.getpid()}"
            try:
                results_df.to_csv(
                    tmp_path,
                    mode='w',
                    header=True,
                    index=False,
                    float_format="%.17g",
                )
                os.replace(tmp_path, task_result_output)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            results_df.to_csv(
                task_result_output,
                mode='w' if write_header else 'a',
                header=write_header,
                index=False,
                float_format="%.17g",
            )
        print(f"Results have been appended to '{task_result_output}'.")


if __name__ == "__main__":
    main()
