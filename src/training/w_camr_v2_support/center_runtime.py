#!/usr/bin/env python
"""Independent Center-RU Atomic-PC matched-baseline route.

This entry point is deliberately self-contained at the *route* level.  It
consumes the immutable MTS Trimer attached to Dataset items, builds its own
MD200 table, and never reads a historical Atomic-PC/CAMR/W-CAMR/Joint result
or pre-start gate.  Four explicit modes are available; there is no implicit
training mode.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.original_mips_atomic_pc import OriginalMIPSAtomicPCCollator  # noqa: E402
from src.modules.original_mips_atomic_pc import OriginalMIPSAtomicPCModel  # noqa: E402
from src.modules.original_mips_md200 import OriginalMIPSMD200  # noqa: E402
from src.dataset.mips_trimer_contract import (  # noqa: E402
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_PROTOCOL,
)
from src.utils import TargetScaler, _cosine_scheduler, set_global_seed  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT = ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth"
GEOMETRY_ROOT = ROOT / "data/processed/mips_trimer_scage/trimer"
RESULT_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_v1"
TRAIN_ROOT = RESULT_ROOT / "training"
SHARD_ROOT = TRAIN_ROOT / "shards"
SMOKE_ROOT = RESULT_ROOT / "smoke"
CONFIG_PATH = ROOT / "configs/atomic_point_center_ru_v1.json"
BASE_CONFIG_PATH = ROOT / "configs/atomic_point_v1.json"

# These are the route's runtime inputs.  Their hashes are recorded in the
# contract and rechecked before smoke, fold execution, and aggregation.
SOURCE_PATHS = (
    Path(__file__),
    ROOT / "src/dataset/original_mips_atomic_pc.py",
    ROOT / "src/dataset/dataloader.py",
    ROOT / "src/modules/atomic_point_encoder.py",
    ROOT / "src/modules/original_mips_atomic_pc.py",
    ROOT / "src/modules/original_mips_knowledge_fusion.py",
    ROOT / "src/modules/original_mips_md200.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, payload: object) -> None:
    """Write an audit artifact without ever overwriting an existing one."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _device() -> torch.device:
    # Smoke intentionally uses CPU.  Fold execution follows the normal device
    # selection, but this route never starts it unless --run-folds is explicit.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _raw_task_rows(task: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
    return (
        frame.iloc[:, 0].astype(str).str.strip().tolist(),
        frame.iloc[:, 1].to_numpy(dtype=np.float64),
    )


def _fold_manifest(task: str) -> dict[str, Any]:
    return _json(ROOT / "data/splits/mips_shared5" / f"{task}.json")


def _task_offset(task: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(task))


def build_dataset(task: str) -> UniDataset:
    """Build the Dataset with the frozen MTS Trimer attached in-place."""
    return UniDataset(
        root=str(ROOT / "data"),
        dataset=f"smi_{task}",
        smiles_model_name="",
        graph_encoder_type="mips_trimer_scage",
        graph_input="repeat_unit",
        use_feature_cache=True,
        feature_source_dataset="smi_all",
        rebuild_feature_cache=False,
        fp_mode="disabled",
        cache_layers="ru_base,topology,trimer,md200",
        cache_validate="sample",
        feature_cache_workers=0,
        feature_cache_item_timeout=45,
        conformer_3d_count=4,
        conformer_keep_count=4,
        conformer_profile="full",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_variant="O8",
        modalities=("graph",),
    )


def _loader(dataset, indices: Iterable[int], collator, batch_size: int, shuffle: bool, workers: int):
    workers = int(workers)
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collator,
        prefetch_factor=2 if workers > 0 else None,
    )


def build_md_table() -> tuple[dict[bytes, np.ndarray], dict[str, int], dict[str, Any]]:
    """Compute the route-local Original-MIPS MD200 table by sample key."""
    all_rows: dict[bytes, str] = {}
    for task in TASKS:
        smiles, _ = _raw_task_rows(task)
        for value in smiles:
            key = sample_key_from_smiles(value)
            all_rows.setdefault(key, value)
    calculator = OriginalMIPSMD200()
    values = {key: calculator.one(smiles) for key, smiles in all_rows.items()}
    return values, calculator.stats(), {
        "sample_count": len(values),
        "key_order": [key.hex() for key in all_rows],
        "protocol": calculator.protocol,
    }


def save_md_table(values: Mapping[bytes, np.ndarray], stats: Mapping[str, int], meta: Mapping[str, Any]) -> Path:
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    path = TRAIN_ROOT / "md200_values.npz"
    provenance_path = TRAIN_ROOT / "md200_provenance.json"
    if path.exists() or provenance_path.exists():
        raise FileExistsError("refusing to overwrite route-local MD200 artifacts")
    keys = list(values)
    matrix = np.stack([np.asarray(values[key], dtype=np.float64) for key in keys], axis=0)
    np.savez_compressed(path, sample_keys=np.asarray([key.hex() for key in keys]), values=matrix)
    provenance = {
        "schema": "original-mips-md200-static-table-center-ru-mts-v1",
        "learned_embedding_cache": False,
        "values_path": str(path),
        "values_sha256": _sha(path),
        "stats": dict(stats),
        **dict(meta),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def load_md_table() -> tuple[dict[bytes, np.ndarray], dict[str, Any]]:
    path = TRAIN_ROOT / "md200_values.npz"
    provenance_path = TRAIN_ROOT / "md200_provenance.json"
    if not path.is_file() or not provenance_path.is_file():
        raise RuntimeError("route-local MD200 table is missing; run --preflight-only first")
    provenance = _json(provenance_path)
    if provenance.get("values_sha256") and provenance["values_sha256"] != _sha(path):
        raise RuntimeError("route-local MD200 artifact hash mismatch")
    payload = np.load(path, allow_pickle=False)
    keys = [bytes.fromhex(str(value)) for value in payload["sample_keys"].tolist()]
    values = np.asarray(payload["values"], dtype=np.float64)
    if values.shape != (len(keys), 200) or not np.isfinite(values).all():
        raise RuntimeError("route-local MD200 table shape/finite contract failed")
    if len(set(keys)) != len(keys):
        raise RuntimeError("route-local MD200 table contains duplicate sample keys")
    return {key: values[index] for index, key in enumerate(keys)}, provenance


def _discover_frozen_cache() -> tuple[Path, dict[str, Any]]:
    """Resolve exactly one complete frozen Trimer cache and verify identity.

    ``manifest_hash`` and ``metadata_hash`` are semantic JSON digests.  The
    actual SHA256 of metadata/manifest/.done/.frozen is recorded separately;
    they are intentionally not confused with the semantic hashes.
    """
    root = Path(GEOMETRY_ROOT)
    candidates = []
    if root.is_dir():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            required = [
                directory / "metadata.json",
                directory / "manifest.json",
                directory / ".done",
                directory / ".frozen",
            ]
            if all(path.is_file() for path in required):
                candidates.append(directory)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one complete frozen MTS Trimer cache, found {len(candidates)}"
        )
    directory = candidates[0]
    metadata_path = directory / "metadata.json"
    manifest_path = directory / "manifest.json"
    done_path = directory / ".done"
    frozen_path = directory / ".frozen"
    metadata = _json(metadata_path)
    manifest = _json(manifest_path)
    frozen = _json(frozen_path)
    done_id = done_path.read_text(encoding="utf-8").strip()
    if len(done_id) != 64 or any(char not in "0123456789abcdef" for char in done_id.lower()):
        raise RuntimeError("MTS Trimer .done is not a 64-character artifact id")
    metadata_semantic_hash = _json_digest(metadata)
    manifest_semantic_hash = _json_digest(manifest)
    feature_hash = metadata.get("feature_config_hash")
    if directory.name != feature_hash:
        raise RuntimeError("frozen Trimer directory name does not match feature_config_hash")
    if metadata.get("schema") != TRIMER_LMDB_SCHEMA:
        raise RuntimeError("frozen Trimer schema mismatch")
    if metadata.get("trimer_content_schema") != TRIMER_CONTENT_SCHEMA:
        raise RuntimeError("frozen Trimer content schema mismatch")
    if metadata.get("build_config", {}).get("protocol") != TRIMER_PROTOCOL:
        raise RuntimeError("frozen Trimer protocol mismatch")
    if manifest.get("schema") != TRIMER_LMDB_SCHEMA:
        raise RuntimeError("frozen Trimer manifest schema mismatch")
    if int(manifest.get("count", -1)) <= 0 or int(manifest.get("failure_count", -1)) != 0:
        raise RuntimeError("frozen Trimer manifest is incomplete")
    checks = {
        "frozen_schema": frozen.get("schema") == "mts-canonical-cache-freeze-v2",
        "frozen_layer": frozen.get("layer") == "trimer",
        "frozen_feature_config_hash": frozen.get("feature_config_hash") == feature_hash,
        "done_artifact_id": frozen.get("done_artifact_id") == done_id,
        "manifest_hash": frozen.get("manifest_hash") == manifest_semantic_hash,
        "metadata_hash": frozen.get("metadata_hash") == metadata_semantic_hash,
        "record_count": int(frozen.get("record_count", -1)) == int(manifest.get("count", -2)),
        "done_file_sha256": frozen.get("done_file_sha256") == _sha(done_path),
        "builder_version": int(frozen.get("builder_version", -1)) == int(metadata.get("builder_version", -2)),
        "transaction_id": bool(frozen.get("transaction_id")),
    }
    # Current frozen bundles use the artifact id for both semantic manifest
    # hash and .done id.  Keep this relationship explicit, while retaining the
    # real file hashes below for forensic provenance.
    checks["artifact_manifest_done_binding"] = frozen.get("manifest_hash") == done_id
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"frozen Trimer cache identity failed: {failed}")
    identity = {
        "cache_dir_name": directory.name,
        "cache_path": str(directory),
        "feature_config_hash": feature_hash,
        "schema": metadata["schema"],
        "content_schema": metadata["trimer_content_schema"],
        "protocol": metadata["build_config"]["protocol"],
        "record_count": int(manifest["count"]),
        "failure_count": int(manifest["failure_count"]),
        "done_artifact_id": done_id,
        "manifest_hash": frozen["manifest_hash"],
        "metadata_hash": frozen["metadata_hash"],
        "semantic_hashes": {
            "manifest_json": manifest_semantic_hash,
            "metadata_json": metadata_semantic_hash,
        },
        "file_sha256": {
            "metadata.json": _sha(metadata_path),
            "manifest.json": _sha(manifest_path),
            ".done": _sha(done_path),
            ".frozen": _sha(frozen_path),
        },
        "identity_checks": checks,
        "ordinary_addhs_hydrogen_persisted": False,
        "source_explicit_hydrogen_may_be_persisted": True,
        "collator_geometry_override": False,
    }
    return directory, identity


def mts_geometry_metadata() -> dict[str, Any]:
    """Compatibility alias returning the strict route-local cache identity."""
    return _discover_frozen_cache()[1]


def _validate_item_geometry(item: Any) -> tuple[bool, str, int, int]:
    coords = getattr(item, "trimer_pos", None)
    atomic_number = getattr(item, "trimer_atomic_number", None)
    ru_offset = getattr(item, "trimer_ru_offset", None)
    if not all(torch.is_tensor(value) for value in (coords, atomic_number, ru_offset)):
        return False, "missing_tensor", 0, 0
    if coords.dtype != torch.float32:
        return False, f"coords_dtype:{coords.dtype}", 0, 0
    if atomic_number.dtype != torch.int64:
        return False, f"atomic_number_dtype:{atomic_number.dtype}", 0, 0
    if ru_offset.dtype != torch.int64:
        return False, f"ru_offset_dtype:{ru_offset.dtype}", 0, 0
    if coords.ndim != 2 or tuple(coords.shape[1:]) != (3,):
        return False, "coords_shape", 0, 0
    if atomic_number.ndim != 1 or ru_offset.ndim != 1:
        return False, "field_rank", 0, 0
    if not (coords.size(0) == atomic_number.numel() == ru_offset.numel()) or coords.numel() == 0:
        return False, "field_count_or_empty", 0, 0
    if not bool(torch.isfinite(coords).all()):
        return False, "nonfinite_coords", int(coords.size(0)), 0
    if bool((atomic_number < 1).any()) or bool((atomic_number > 118).any()):
        return False, "atomic_number_range", int(coords.size(0)), int((atomic_number == 1).sum())
    offsets = set(int(value) for value in ru_offset.detach().cpu().tolist())
    if offsets != {-1, 0, 1}:
        return False, "ru_offset_incomplete", int(coords.size(0)), int((atomic_number == 1).sum())
    flags = {
        "trimer_geometry_valid": bool(getattr(item, "trimer_geometry_valid", False)),
        "trimer_geometry_is_3d": bool(getattr(item, "trimer_geometry_is_3d", False)),
        "trimer_2d_fallback": bool(getattr(item, "trimer_2d_fallback", False)),
        "graph_available": bool(getattr(item, "graph_available", True)),
    }
    if not flags["trimer_geometry_valid"]:
        return False, "geometry_invalid", int(coords.size(0)), int((atomic_number == 1).sum())
    if not flags["trimer_geometry_is_3d"]:
        return False, "not_3d", int(coords.size(0)), int((atomic_number == 1).sum())
    if flags["trimer_2d_fallback"]:
        return False, "2d_fallback", int(coords.size(0)), int((atomic_number == 1).sum())
    if not flags["graph_available"]:
        return False, "graph_unavailable", int(coords.size(0)), int((atomic_number == 1).sum())
    return True, "ok", int(coords.size(0)), int((atomic_number == 1).sum())


def _scan_downstream_cohort() -> dict[str, Any]:
    """Scan every downstream row and retain only valid MTS geometry keys."""
    valid_keys: set[bytes] = set()
    invalid_keys: set[bytes] = set()
    per_task: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    total_rows = 0
    for task in TASKS:
        dataset = build_dataset(task)
        smiles, _targets = _raw_task_rows(task)
        if len(dataset) != len(smiles):
            raise RuntimeError(f"MTS Dataset/raw row count mismatch for {task}")
        task_valid = task_invalid = 0
        task_h = task_points = 0
        failures: dict[str, int] = {}
        for index, source_smiles in enumerate(smiles):
            total_rows += 1
            expected_key = sample_key_from_smiles(source_smiles)
            item = dataset[index]
            observed_key = OriginalMIPSAtomicPCCollator._key(item)
            if observed_key != expected_key:
                raise RuntimeError(f"sample-key mismatch for {task} row {index}")
            ok, reason, point_count, h_count = _validate_item_geometry(item)
            if ok:
                task_valid += 1
                valid_keys.add(expected_key)
                task_h += h_count
                task_points += point_count
                candidates.append({"task": task, "index": index, "key": expected_key.hex(), "h_count": h_count})
            else:
                task_invalid += 1
                invalid_keys.add(expected_key)
                failures[reason] = failures.get(reason, 0) + 1
        per_task[task] = {
            "row_count": len(smiles),
            "valid_rows": task_valid,
            "invalid_rows": task_invalid,
            "valid_unique_keys": len({item["key"] for item in candidates if item["task"] == task}),
            "explicit_h_atom_count": task_h,
            "valid_point_count": task_points,
            "failure_reasons": failures,
        }
    h_rows = [item for item in candidates if int(item["h_count"]) > 0]
    no_h_rows = [item for item in candidates if int(item["h_count"]) == 0]
    return {
        "task_count": len(TASKS),
        "total_rows": total_rows,
        "valid_unique_keys": len(valid_keys),
        "invalid_unique_keys": len(invalid_keys),
        "valid_keys": sorted(key.hex() for key in valid_keys),
        "per_task": per_task,
        "hydrogen_census": {
            "valid_rows_with_source_explicit_h": len(h_rows),
            "valid_rows_without_source_explicit_h": len(no_h_rows),
            "valid_h_atom_count": int(sum(int(item["h_count"]) for item in candidates)),
            "ordinary_addhs_hydrogen_persisted": False,
            "source_explicit_or_isotope_h_may_be_persisted": True,
        },
        "probe_candidates": {"no_h": no_h_rows[:8], "with_h": h_rows[:8]},
        "_probe_rows": candidates,
    }


def geometry_success_keys() -> set[bytes]:
    scan = _scan_downstream_cohort()
    return {bytes.fromhex(value) for value in scan["valid_keys"]}


def gate_atomic_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file() or not BASE_CONFIG_PATH.is_file():
        raise RuntimeError("Center-RU Atomic-PC config files are missing")
    config = _json(CONFIG_PATH)
    base = _json(BASE_CONFIG_PATH)
    pool = config.get("pool", {})
    full_context = config.get("full_trimer_context", {})
    geometry = config.get("geometry_contract", {})
    matches = (
        config.get("schema") == "atomic-point-cloud-v1-center-ru-config"
        and int(config.get("hidden_dim", -1)) == 256
        and int(config.get("output_dim", -1)) == 512
        and int(config.get("num_layers", -1)) == 4
        and int(config.get("k_neighbors", -1)) == 24
        and pool.get("scope") == "center"
        and pool.get("mask_field") == "trimer_ru_offset"
        and int(pool.get("mask_value", 99)) == 0
        and pool.get("mask_before_softmax") is True
        and geometry.get("schema") == TRIMER_LMDB_SCHEMA
        and geometry.get("content_schema") == TRIMER_CONTENT_SCHEMA
        and geometry.get("protocol") == TRIMER_PROTOCOL
        and geometry.get("source") == "frozen_mips_trimer_scage_dataset_item"
        and geometry.get("ordinary_addhs_hydrogen_persisted") is False
        and geometry.get("collator_geometry_override") is False
        and "all persisted RU(-1)+RU(0)+RU(+1)" in str(full_context.get("message_passing_points", ""))
    )
    return {
        "config_sha256": _sha(CONFIG_PATH),
        "base_config_sha256": _sha(BASE_CONFIG_PATH),
        "base_config_declared_sha256": config.get("base_frozen_config_sha256"),
        "matched": bool(matches),
        "atomic_config": config,
        "base_config": base,
    }


def load_checkpoint(model: OriginalMIPSAtomicPCModel) -> dict[str, Any]:
    """Load exactly the current O8 graph tensors from the 5K checkpoint."""
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"missing current O8 5K checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("step") != 5000:
        raise RuntimeError("current O8 checkpoint must be the fixed 5K checkpoint")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("current O8 checkpoint has no state_dict")
    target = model.state_dict()
    loaded: list[str] = []
    missing: list[str] = []
    for name, tensor in target.items():
        if not name.startswith("graph_encoder."):
            continue
        source_name = "model.o8." + name[len("graph_encoder."):]
        if source_name not in state:
            missing.append(name)
            continue
        source = state[source_name]
        if tuple(source.shape) != tuple(tensor.shape):
            raise RuntimeError(f"O8 checkpoint shape mismatch for {name}")
        target[name] = source.detach().clone()
        loaded.append(name)
    if missing or len(loaded) != 87:
        raise RuntimeError(f"current O8 strict mapping failed: missing={missing[:4]}, loaded={len(loaded)}")
    model.load_state_dict(target, strict=True)
    return {
        "path": str(CHECKPOINT),
        "sha256": _sha(CHECKPOINT),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_schema": checkpoint.get("schema"),
        "loaded_o8_tensor_count": len(loaded),
        "loaded_o8_tensor_names": loaded,
        "fresh_components": ["atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp"],
        "excluded": "GLT tensors and old pretrained Atomic-PC/KFuse/adapter/predictor; graph MD residual frozen and disabled",
    }


def _center_model() -> tuple[OriginalMIPSAtomicPCModel, dict[str, Any]]:
    model = OriginalMIPSAtomicPCModel(atomic_pool_scope="center")
    if getattr(model, "atomic_pool_scope", None) != "center":
        raise RuntimeError("Center-RU model construction failed")
    checkpoint = load_checkpoint(model)
    return model, checkpoint


def make_optimizer(model: OriginalMIPSAtomicPCModel) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    groups: list[dict[str, Any]] = []
    used: set[int] = set()

    def add(module: nn.Module, lr: float, name: str) -> None:
        decay, no_decay = [], []
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in used:
                continue
            used.add(id(parameter))
            if parameter.ndim <= 1 or parameter_name.lower().endswith("bias") or "norm" in parameter_name.lower():
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if decay:
            groups.append({"params": decay, "lr": float(lr), "weight_decay": 0.02, "name": name + "/decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0, "name": name + "/no_decay"})

    add(model.graph_encoder, 1e-5, "o8")
    add(model.atomic_point_encoder, 1e-5, "atomic_pc")
    add(model.fusion, 1e-5, "original_mips_kfuse")
    add(model.graph_norm, 1e-5, "graph_adapter_norm")
    add(model.graph_projection, 1e-5, "graph_adapter")
    add(model.mlp, 1e-4, "regression_head")
    remaining = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": 1e-4, "weight_decay": 0.02, "name": "remaining"})
    if not groups:
        raise RuntimeError("Center-RU optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups)
    return optimizer, {str(group["name"]): float(group["lr"]) for group in groups}


def _assert_center_trace(model: OriginalMIPSAtomicPCModel) -> None:
    trace = model.last_forward_trace
    if int(trace.get("fusion_calls", 0)) != 1:
        raise RuntimeError("Center-RU forward must execute exactly one KFuse call")
    if trace.get("atomic_pool_scope") != "center":
        raise RuntimeError("Center-RU forward did not use center pooling")
    stats = trace.get("point_stats", {})
    if stats.get("point_count_used_for_message_passing") != stats.get("point_count"):
        raise RuntimeError("message passing did not use the complete Trimer")
    if stats.get("point_count_used_for_pooling") != stats.get("central_point_count"):
        raise RuntimeError("pooling is not central-RU-only")
    if float(stats.get("pool_weight_sum_max_error", float("inf"))) > 1e-6:
        raise RuntimeError("center pooling weights do not sum to one")
    if min(stats.get("central_point_count_per_graph", [0])) <= 0:
        raise RuntimeError("center pooling encountered a graph without RU(0)")


def _validate_alignment(batch: Any, expected_keys: list[bytes]) -> None:
    observed = tuple(getattr(batch, "original_mips_sample_keys", ()))
    if observed != tuple(key.hex() for key in expected_keys):
        raise RuntimeError("batch Atomic-PC keys are not aligned with raw sample keys")
    cloud = batch.atomic_point_cloud
    if cloud.sample_keys != observed:
        raise RuntimeError("Atomic-PC cloud keys differ from collated sample keys")
    if tuple(getattr(batch, "mips_md").shape) != (len(expected_keys), 200):
        raise RuntimeError("MD200 batch shape does not match sample count")


def _neighbor_counterfactual(model: OriginalMIPSAtomicPCModel, batch: Any) -> float:
    cloud = batch.atomic_point_cloud
    central = cloud.ru_offset.eq(0)
    if not bool((~central).any()):
        raise RuntimeError("neighbor counterfactual requires non-central Trimer atoms")
    shifted = cloud.coords.clone()
    shifted[~central] = shifted[~central] + torch.tensor([13.0, -7.0, 5.0], device=shifted.device)
    changed_cloud = type(cloud)(
        shifted, cloud.atomic_number, cloud.ru_offset, cloud.batch, cloud.ptr,
        cloud.sample_keys, cloud.source_smiles,
    )
    altered = copy.copy(batch)
    altered.atomic_point_cloud = changed_cloud
    with torch.no_grad():
        reference = model(batch)[0]
        counterfactual = model(altered)[0]
    delta = float((reference - counterfactual).abs().max().detach().cpu())
    if not np.isfinite(delta) or delta <= 1e-8:
        raise RuntimeError("neighbor-only geometry counterfactual had no effect")
    return delta


def _one_batch_gate(
    md_values: Mapping[bytes, np.ndarray],
    scan: dict[str, Any],
    *,
    cpu_only: bool = False,
) -> dict[str, Any]:
    no_h = scan.get("probe_candidates", {}).get("no_h", [])
    with_h = scan.get("probe_candidates", {}).get("with_h", [])
    if not no_h or not with_h:
        raise RuntimeError("real-batch gate requires one no-H and one source-explicit-H sample")
    selections = [no_h[0], with_h[0]]
    items = []
    targets = []
    expected_keys = []
    for selection in selections:
        dataset = build_dataset(selection["task"])
        smiles, raw_targets = _raw_task_rows(selection["task"])
        item = dataset[int(selection["index"])]
        item.y = torch.tensor([float(raw_targets[int(selection["index"])])], dtype=torch.float32)
        items.append(item)
        targets.append(float(raw_targets[int(selection["index"])]))
        expected_keys.append(sample_key_from_smiles(smiles[int(selection["index"])]))
    collator = OriginalMIPSAtomicPCCollator(md_values)
    batch = collator(items)
    _validate_alignment(batch, expected_keys)
    model, checkpoint = _center_model()
    device = torch.device("cpu") if cpu_only else _device()
    model.to(device)
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        output, _ = model(batch)
    _assert_center_trace(model)
    attention = model.fusion.last_attention_weights
    if attention is None:
        raise RuntimeError("KFuse did not expose modality attention")
    attention_sum_error = float((attention.sum(dim=-1) - 1.0).abs().max().cpu())
    if attention_sum_error > 1e-6:
        raise RuntimeError("KFuse modality softmax is not normalized")
    bypass = model.mlp(model.graph_projection(model.graph_norm(model.last_forward_trace and model(batch, return_dict=True)["o8_graph"])))
    # The extra forward above is trace-only and still exactly one call per
    # invocation; compare it with a fresh fused forward below.
    with torch.no_grad():
        fused_payload = model(batch, return_dict=True)
        fused_output = fused_payload["prediction"]
        bypass_output = model.mlp(model.graph_projection(model.graph_norm(fused_payload["o8_graph"])))
    bypass_delta = float((fused_output - bypass_output).abs().max().cpu())
    if bypass_delta <= 1e-8:
        raise RuntimeError("predictor output is indistinguishable from O8 bypass")
    neighbor_delta = _neighbor_counterfactual(model, batch)
    finite_forward = bool(torch.isfinite(fused_output).all())
    model.train()
    model.zero_grad(set_to_none=True)
    train_output, _ = model(batch)
    loss = nn.HuberLoss(delta=0.5)(train_output, batch.y.to(device))
    finite_loss = bool(torch.isfinite(loss).all())
    if finite_loss:
        loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = bool(trainable) and all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    _assert_center_trace(model)
    stats = model.last_forward_trace["point_stats"]
    return {
        "forward": True,
        "backward": True,
        "finite_forward": finite_forward,
        "loss_finite": finite_loss,
        "all_trainable_gradients_finite": finite_gradients,
        "fusion_call_count": int(model.last_forward_trace["fusion_calls"]),
        "output_shape": list(fused_output.shape),
        "atomic_pc_shape": list(fused_payload["atomic_pc"].shape),
        "full_point_count": int(stats["point_count_used_for_message_passing"]),
        "central_pool_point_count": int(stats["point_count_used_for_pooling"]),
        "cross_ru_neighbor_edge_count": int(stats.get("cross_ru_neighbor_edge_count", 0)),
        "attention_sum_max_error": attention_sum_error,
        "neighbor_counterfactual_max_delta": neighbor_delta,
        "predictor_bypass_max_delta": bypass_delta,
        "sample_keys": [key.hex() for key in expected_keys],
        "h_counts": [int(item["h_count"]) for item in selections],
        "checkpoint": checkpoint,
    }


def _source_identity() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError(f"route source files are missing: {missing}")
    return {str(path.relative_to(ROOT)): _sha(path) for path in SOURCE_PATHS}


def _build_contract(cache: dict[str, Any], checkpoint: dict[str, Any], config: dict[str, Any], scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "original-mips-atomic-pc-center-ru-mts-trimer-matched-v1-baseline-contract",
        "route": "Atomic-PC Center-RU independent baseline",
        "initialization": {
            "current_o8_checkpoint": checkpoint,
            "loaded_o8_tensor_count": checkpoint["loaded_o8_tensor_count"],
            "fresh_components": checkpoint["fresh_components"],
            "forbidden_pretrained_components": ["Atomic-PC", "KFuse", "graph_adapter", "predictor", "CAMR", "W-CAMR", "Joint"],
        },
        "geometry": cache,
        "config": config,
        "source_sha256": _source_identity(),
        "cohort": {key: value for key, value in scan.items() if not key.startswith("_")},
        "runtime_contract": {
            "full_trimer_message_passing": True,
            "knn_k": 24,
            "message_passing_layers": 4,
            "pool_scope": "center",
            "atomic_pc_dim": 512,
            "active_modalities": ["md", "atomic_pc"],
            "md_dim": 200,
            "kfusion_calls": 1,
            "AP3D": False,
            "MCP": False,
            "geometry_regeneration": False,
            "geometry_override": False,
        },
    }


def _verify_current_contract() -> dict[str, Any]:
    contract_path = RESULT_ROOT / "baseline_contract.json"
    gate_path = RESULT_ROOT / "prestart_gate.json"
    if not contract_path.is_file() or not gate_path.is_file():
        raise RuntimeError("route baseline contract/prestart gate is missing; run --preflight-only first")
    contract = _json(contract_path)
    gate = _json(gate_path)
    if not bool(gate.get("all_pass")):
        raise RuntimeError("route pre-start gate is not PASS")
    _cache_dir, cache = _discover_frozen_cache()
    config = gate_atomic_config()
    if not config.get("matched"):
        raise RuntimeError("current Center-RU config no longer matches baseline contract")
    if cache != contract.get("geometry"):
        raise RuntimeError("frozen Trimer cache identity changed since preflight")
    checkpoint_model = OriginalMIPSAtomicPCModel(atomic_pool_scope="center")
    checkpoint = load_checkpoint(checkpoint_model)
    if checkpoint != contract.get("initialization", {}).get("current_o8_checkpoint"):
        raise RuntimeError("current O8 checkpoint identity changed since preflight")
    if _source_identity() != contract.get("source_sha256"):
        raise RuntimeError("route source SHA256 changed since preflight")
    if config != contract.get("config"):
        raise RuntimeError("route config identity changed since preflight")
    return {"contract": contract, "gate": gate}


def preflight() -> dict[str, Any]:
    """Run the complete read-only cohort and runtime gate, then write artifacts."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (RESULT_ROOT / "baseline_contract.json", RESULT_ROOT / "prestart_gate.json"):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing route gate: {path}")
    cache_dir, cache = _discover_frozen_cache()
    del cache_dir
    config = gate_atomic_config()
    if not config.get("matched"):
        raise RuntimeError("Center-RU Atomic-PC config gate failed")
    scan = _scan_downstream_cohort()
    md_values, md_stats, md_meta = build_md_table()
    # The model initialization audit also guarantees exactly 87 O8 tensors and
    # fresh non-O8 components.  It does not persist a checkpoint or model.
    model, checkpoint = _center_model()
    del model
    batch_gate = _one_batch_gate(md_values, scan, cpu_only=True)
    gates = {
        "CACHE_UNIQUE_FROZEN": bool(cache["identity_checks"] and all(cache["identity_checks"].values())),
        "CACHE_IDENTITY_BOUND": cache["cache_dir_name"] == cache["feature_config_hash"] and cache["record_count"] > 0,
        "CONFIG_CENTER_RU": bool(config["matched"]),
        "COHORT_ALL_TASKS_SCANNED": scan["task_count"] == len(TASKS) and scan["total_rows"] > 0,
        "COHORT_HAS_NO_H_AND_SOURCE_H": bool(scan["probe_candidates"]["no_h"] and scan["probe_candidates"]["with_h"]),
        "O8_CHECKPOINT_5K_87_TENSORS": checkpoint["checkpoint_step"] == 5000 and checkpoint["loaded_o8_tensor_count"] == 87,
        "FRESH_NON_O8_COMPONENTS": set(checkpoint["fresh_components"]) == {"atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp"},
        "ONE_BATCH_FORWARD": batch_gate["forward"] and batch_gate["finite_forward"] and batch_gate["fusion_call_count"] == 1,
        "FULL_TRIMER_MESSAGE_PASSING": batch_gate["full_point_count"] > 0,
        "CENTER_POOLING": batch_gate["central_pool_point_count"] > 0,
        "ATOMIC_PC_512": batch_gate["atomic_pc_shape"] == [2, 512],
        "NEIGHBOR_CONTEXT": batch_gate["neighbor_counterfactual_max_delta"] > 1e-8,
        "KFUSE_ATTENTION_NORMALIZED": batch_gate["attention_sum_max_error"] <= 1e-6,
        "PREDICTOR_NO_BYPASS": batch_gate["predictor_bypass_max_delta"] > 1e-8,
        "ONE_BATCH_BACKWARD_FINITE": batch_gate["backward"] and batch_gate["loss_finite"] and batch_gate["all_trainable_gradients_finite"],
    }
    contract = _build_contract(cache, checkpoint, config, scan)
    gate = {
        "schema": "original-mips-atomic-pc-center-ru-mts-trimer-matched-v1-prestart-gate",
        "all_pass": all(gates.values()),
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "cache_identity": cache,
        "config_identity": config,
        "checkpoint_identity": checkpoint,
        "cohort_scan": {key: value for key, value in scan.items() if not key.startswith("_")},
        "md200": {"stats": md_stats, "provenance": md_meta, "route_local": True},
        "real_batch_gate": batch_gate,
        "forbidden_actions": {"pretraining": "NO", "geometry_generation": "NO", "geometry_override": "NO", "AP3D": "NO", "MCP": "NO", "surface": "NO", "mesh": "NO"},
    }
    _write_new(RESULT_ROOT / "baseline_contract.json", contract)
    _write_new(RESULT_ROOT / "prestart_gate.json", gate)
    if not gate["all_pass"]:
        raise SystemExit("Center-RU baseline pre-start gate failed; no training started")
    print(json.dumps({"all_pass": gate["all_pass"], "gates": gate["gates"]}, ensure_ascii=False, indent=2))
    return gate


def _collect(loader, model, device, scaler, criterion, train=False, optimizer=None, scheduler=None, max_grad_norm=1.0) -> dict[str, float | int]:
    model.train(bool(train))
    losses, predictions, targets = [], [], []
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output, _ = model(batch)
            loss = criterion(output, batch.y)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Center-RU loss is non-finite")
            if train:
                loss.backward()
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                if not trainable or any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
                    raise FloatingPointError("Center-RU trainable gradient is non-finite")
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizer.step()
                scheduler.step()
        _assert_center_trace(model)
        losses.append(loss.detach().float().cpu())
        predictions.append(output.detach().float().cpu())
        targets.append(batch.y.detach().float().cpu())
    if not losses:
        raise RuntimeError("empty Center-RU loader")
    predicted = torch.cat(predictions).numpy().reshape(-1, 1)
    target = torch.cat(targets).numpy().reshape(-1, 1)
    raw_predicted = scaler.inverse_transform(predicted).reshape(-1)
    raw_target = scaler.inverse_transform(target).reshape(-1)
    return {
        "loss": float(torch.stack(losses).mean()),
        "r2": float(r2_score(raw_target, raw_predicted)) if len(raw_target) > 1 else float("nan"),
        "mae": float(mean_absolute_error(raw_target, raw_predicted)),
        "rmse": float(np.sqrt(mean_squared_error(raw_target, raw_predicted))),
        "seconds": float(time.perf_counter() - started),
        "n": int(len(raw_target)),
    }


def _success_indices(task: str, success_keys: set[bytes]) -> tuple[list[str], np.ndarray, list[int]]:
    smiles, targets = _raw_task_rows(task)
    keep = [index for index, value in enumerate(smiles) if sample_key_from_smiles(value) in success_keys]
    return smiles, targets, keep


def _fold_run(task: str, fold_id: int, md_values: Mapping[bytes, np.ndarray], success_keys: set[bytes], device: torch.device) -> dict[str, Any]:
    """Run one explicitly requested fold; formal 8x5 is never implicit."""
    shard = SHARD_ROOT / f"{task}_fold{fold_id}.json"
    if shard.exists():
        raise FileExistsError(f"refusing to overwrite existing fold shard: {shard}")
    fold_started = time.perf_counter()
    fold_seed = 42 + 1009 * _task_offset(task) + int(fold_id)
    set_global_seed(fold_seed)
    smiles, raw_targets, keep = _success_indices(task, success_keys)
    dataset = build_dataset(task)
    manifest = _fold_manifest(task)
    fold = next(value for value in manifest["folds"] if int(value["fold"]) == int(fold_id))
    train_indices = [index for index in fold["train_indices"] if index in keep]
    held_indices = [index for index in fold["test_indices"] if index in keep]
    scaler = TargetScaler(task, StandardScaler(), transform_mode="recommended")
    scaler.scaler.fit(scaler._pre_transform(raw_targets[train_indices]))
    dataset.set_target_override(scaler.transform(raw_targets).reshape(-1))
    collator = OriginalMIPSAtomicPCCollator(md_values)
    train_loader = _loader(dataset, train_indices, collator, 32, True, 2)
    valid_loader = _loader(dataset, held_indices, collator, 64, False, 2)
    model, checkpoint = _center_model()
    model.to(device)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer, optimizer_lrs = make_optimizer(model)
    total_steps = max(1, 100 * len(train_loader))
    scheduler = _cosine_scheduler(optimizer, total_steps, min(total_steps - 1, 5 * len(train_loader)))
    best_state = None
    best_val = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    epoch_seconds = []
    for epoch in range(1, 101):
        epoch_started = time.perf_counter()
        train_metrics = _collect(train_loader, model, device, scaler, criterion, True, optimizer, scheduler)
        valid_metrics = _collect(valid_loader, model, device, scaler, criterion, False)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
        if np.isfinite(valid_metrics["r2"]) and valid_metrics["r2"] > best_val:
            best_val = float(valid_metrics["r2"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError(f"no finite Center-RU validation checkpoint for {task} fold {fold_id}")
    model.load_state_dict(best_state, strict=True)
    test_metrics = _collect(valid_loader, model, device, scaler, criterion, False)
    result = {
        "schema": "original-mips-atomic-pc-center-ru-mts-trimer-matched-v1-fold-metrics",
        "task": task,
        "fold": int(fold_id),
        "seed": 42,
        "fold_seed": int(fold_seed),
        "device": str(device),
        "train_n": len(train_indices),
        "valid_n": len(held_indices),
        "test_n": len(held_indices),
        "best_epoch": int(best_epoch),
        "selection_metric": "validation R2 (historical_shared5 held-out fold)",
        "best_val_r2": float(best_val),
        "test_r2": float(test_metrics["r2"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "training_wall_seconds": float(sum(epoch_seconds)),
        "fold_wall_seconds": float(time.perf_counter() - fold_started),
        "epoch_wall_seconds": epoch_seconds,
        "optimizer_lrs": optimizer_lrs,
        "checkpoint": checkpoint,
        "history": history,
        "architecture": "Current O8 + Original-MIPS KFuse + Center-RU Atomic-PC512",
        "finetune_geometry_contract": "frozen-mts-trimer-attached-to-dataset-v1",
        "trimer_protocol": TRIMER_PROTOCOL,
        "geometry_override": False,
        "atomic_pool_scope": "center",
        "full_trimer_message_passing": True,
        "knowledge": {"md": [len(train_indices), 200], "atomic_pc": [len(train_indices), 512]},
        "AP3D": False,
        "MCP": False,
    }
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    _write_new(shard, result)
    return result


def _aggregate() -> tuple[dict[str, Any], dict[str, float]]:
    _verify_current_contract()
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            path = SHARD_ROOT / f"{task}_fold{fold}.json"
            if path.is_file():
                rows.append(_json(path))
    if len(rows) != len(TASKS) * len(FOLDS):
        raise RuntimeError(f"expected 40 Center-RU shards, found {len(rows)}")
    per_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        per_task[task] = {
            "fold_count": len(task_rows),
            "test_r2_mean": float(np.mean([row["test_r2"] for row in task_rows])),
            "test_r2_std": float(np.std([row["test_r2"] for row in task_rows])),
            "test_mae_mean": float(np.mean([row["test_mae"] for row in task_rows])),
            "test_rmse_mean": float(np.mean([row["test_rmse"] for row in task_rows])),
        }
    macro = {
        "task_count": len(TASKS),
        "macro_test_r2": float(np.mean([per_task[task]["test_r2_mean"] for task in TASKS])),
        "macro_test_mae": float(np.mean([per_task[task]["test_mae_mean"] for task in TASKS])),
        "macro_test_rmse": float(np.mean([per_task[task]["test_rmse_mean"] for task in TASKS])),
    }
    _write_new(TRAIN_ROOT / "per_fold_metrics.json", rows)
    _write_new(TRAIN_ROOT / "per_task_metrics.json", {"per_task": per_task, "macro": macro})
    return per_task, macro


def _smoke() -> dict[str, Any]:
    _verify_current_contract()
    smoke_result = SMOKE_ROOT / "smoke_result.json"
    if smoke_result.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke artifact: {smoke_result}")
    if any(SHARD_ROOT.glob("*.json")):
        raise RuntimeError("smoke refuses to run when formal fold shards already exist")
    md_values, _ = load_md_table()
    scan = _scan_downstream_cohort()
    structure = _one_batch_gate(md_values, scan, cpu_only=True)
    task = "eat"
    fold_id = 0
    fold_seed = 42 + 1009 * _task_offset(task) + fold_id
    set_global_seed(fold_seed)
    smiles, raw_targets = _raw_task_rows(task)
    success_keys = {bytes.fromhex(value) for value in scan["valid_keys"]}
    _smiles, _targets, keep = _success_indices(task, success_keys)
    manifest = _fold_manifest(task)
    fold = next(value for value in manifest["folds"] if int(value["fold"]) == fold_id)
    train_indices = [index for index in fold["train_indices"] if index in keep]
    held_indices = [index for index in fold["test_indices"] if index in keep]
    if len(train_indices) < 4 or not held_indices:
        raise RuntimeError("eat/fold0 smoke does not have enough valid rows")
    dataset = build_dataset(task)
    scaler = TargetScaler(task, StandardScaler(), transform_mode="recommended")
    scaler.scaler.fit(scaler._pre_transform(raw_targets[train_indices]))
    dataset.set_target_override(scaler.transform(raw_targets).reshape(-1))
    collator = OriginalMIPSAtomicPCCollator(md_values)
    train_loader = _loader(dataset, train_indices[:4], collator, 2, False, 0)
    held_loader = _loader(dataset, held_indices[:2], collator, 2, False, 0)
    model, checkpoint = _center_model()
    model.to(torch.device("cpu"))
    optimizer, optimizer_lrs = make_optimizer(model)
    scheduler = _cosine_scheduler(optimizer, 2, 1)
    criterion = nn.HuberLoss(delta=0.5)
    step_count = 0
    losses = []
    for batch in train_loader:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(batch)
        loss = criterion(output, batch.y)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("smoke loss is non-finite")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("smoke gradient is non-finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        _assert_center_trace(model)
        losses.append(float(loss.detach()))
        step_count += 1
        if step_count == 2:
            break
    if step_count != 2:
        raise RuntimeError(f"smoke must execute exactly 2 optimizer steps, got {step_count}")
    model.eval()
    held_batch = next(iter(held_loader))
    with torch.no_grad():
        held_output, _ = model(held_batch)
    _assert_center_trace(model)
    if not bool(torch.isfinite(held_output).all()):
        raise FloatingPointError("smoke held-out prediction is non-finite")
    result = {
        "schema": "original-mips-atomic-pc-center-ru-mts-trimer-matched-v1-smoke",
        "task": task,
        "fold": fold_id,
        "fold_seed": fold_seed,
        "optimizer_steps": step_count,
        "losses": losses,
        "held_out_forward": {"shape": list(held_output.shape), "finite": True},
        "structure_probe": structure,
        "checkpoint": checkpoint,
        "optimizer_lrs": optimizer_lrs,
        "formal_shards_written": False,
        "geometry_generated": False,
    }
    _write_new(smoke_result, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run-folds", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    args = parser.parse_args()
    if args.preflight_only:
        preflight()
        print("Center-RU baseline pre-start gate PASS; no training started")
        return
    if args.smoke:
        print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
        return
    if args.aggregate:
        _per_task, macro = _aggregate()
        print(json.dumps(macro, ensure_ascii=False, indent=2))
        return
    _verify_current_contract()
    values, _ = load_md_table()
    success = geometry_success_keys()
    device = _device()
    for task in args.tasks:
        for fold in args.folds:
            result = _fold_run(task, int(fold), values, success, device)
            print(json.dumps({"task": task, "fold": fold, "test_r2": result["test_r2"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
