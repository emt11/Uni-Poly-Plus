#!/usr/bin/env python
"""Run the fixed 50K O8 + Original-MIPS + Center-RU joint pilot.

The command has two explicit phases:

``--preflight-only``
    Build/read the 50K MD200 table, audit frozen Trimer coverage and execute
    the required one-batch gradient contract.  No optimizer step is taken.

``--train``
    Require the persisted all-PASS gate, then make exactly one pass over the
    eligible rows of the fixed PI1M_50k subset and save a joint checkpoint.

No AP3D/MCP branch, fixed-vector fingerprint disturbance, geometry retry or
full-1M cache generation is reachable from this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from rdkit import RDLogger
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.original_mips_atomic_pc_joint import (  # noqa: E402
    OriginalMIPSAtomicPCJointCollator,
)
from src.modules.original_mips_md200 import OriginalMIPSMD200  # noqa: E402
from src.training.pretrain.original_mips_atomic_pc_joint import (  # noqa: E402
    ATOM_MASK_RATE,
    COORD_NOISE_SIGMA,
    JOINT_CHECKPOINT_SCHEMA,
    JOINT_PRETRAIN_SCHEMA,
    MD_KVEC_MASK_RATE,
    OriginalMIPSAtomicPCJointPretrainer,
    gradient_summary,
    trainable_group_parameters,
)
from src.utils import set_global_seed  # noqa: E402


RESULT_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_joint_pretrain_v1"
PRETRAIN_ROOT = RESULT_ROOT / "pretraining"
SUBSET_CSV = ROOT / "data/raw/PI1M_50k.csv"
SUBSET_META = ROOT / "data/raw/PI1M_50k.subset.json"
O8_CHECKPOINT = ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth"
REFERENCE_MACRO_R2 = 0.8403314216055405


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_key_hash(keys: list[bytes]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(bytes(key))
        digest.update(b"\n")
    return digest.hexdigest()


def _sorted_key_hash(keys: list[bytes]) -> str:
    return _ordered_key_hash(sorted(bytes(key) for key in keys))


def _build_dataset() -> UniDataset:
    """Open only immutable O8/Trimer layers; MD200 is restored below."""

    RDLogger.DisableLog("rdApp.*")
    return UniDataset(
        root=str(ROOT / "data"),
        dataset="PI1M_50k",
        smiles_model_name="",
        graph_encoder_type="mips_trimer_scage",
        graph_input="repeat_unit",
        geom_input="repeat_unit",
        use_feature_cache=True,
        feature_source_dataset="PI1M_v2",
        rebuild_feature_cache=False,
        fp_mode="disabled",
        cache_layers="ru_base,topology,trimer",
        cache_validate="sample",
        feature_cache_workers=0,
        feature_cache_item_timeout=240,
        conformer_3d_count=4,
        conformer_keep_count=4,
        conformer_profile="full",
        mips_core="paper_corrected",
        mips_max_hops=2,
        # The graph-only O8 constructor forces its descriptor flag on, but the
        # actual descriptor tensor is supplied by the restored route collator.
        mips_use_descriptors=False,
        mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_variant="O8",
        modalities=("graph",),
        experiment_id="original_mips_atomic_pc_center_ru_joint_pretrain_v1",
        feature_config_hash="original_mips_atomic_pc_center_ru_joint_pretrain_v1",
    )


def _geometry_ok(item: Any) -> bool:
    return bool(getattr(item, "trimer_geometry_valid", False)) and bool(
        getattr(item, "trimer_geometry_is_3d", False)
    ) and bool(getattr(item, "graph_available", True))


def _load_saved_md() -> tuple[dict[bytes, np.ndarray], dict[str, Any]] | None:
    values_path = PRETRAIN_ROOT / "md200_values.npz"
    provenance_path = PRETRAIN_ROOT / "md200_provenance.json"
    if not values_path.is_file() or not provenance_path.is_file():
        return None
    payload = np.load(values_path, allow_pickle=False)
    key_values = [bytes.fromhex(str(value)) for value in payload["sample_keys"].tolist()]
    matrix = np.asarray(payload["values"], dtype=np.float32)
    if matrix.shape != (len(key_values), 200) or not np.isfinite(matrix).all():
        raise RuntimeError("saved 50K MD200 table has invalid shape or non-finite values")
    return {
        key: matrix[index].copy() for index, key in enumerate(key_values)
    }, _json(provenance_path)


def _compute_md_table(smiles_by_key: dict[bytes, str]) -> tuple[dict[bytes, np.ndarray], dict[str, int]]:
    calculator = OriginalMIPSMD200()
    values: dict[bytes, np.ndarray] = {}
    failures: dict[str, int] = {}
    for key, smiles in smiles_by_key.items():
        try:
            value = np.asarray(calculator.one(smiles), dtype=np.float32)
            if value.shape != (200,) or not np.isfinite(value).all():
                raise ValueError("non-finite or wrong-shaped MD200")
            values[key] = value
        except Exception:
            # Keep the official zero fallback explicit at the route boundary;
            # the key is excluded from training and recorded as an MD failure.
            values[key] = np.zeros((200,), dtype=np.float32)
            failures[key.hex()] = failures.get(key.hex(), 0) + 1
    stats = calculator.stats()
    stats["route_exception_count"] = int(sum(failures.values()))
    stats["exception_sample_keys"] = sorted(failures)
    return values, stats


def _prepare_subset(dataset: UniDataset) -> dict[str, Any]:
    """Audit all 50K rows and build/reuse an exact-key MD200 table."""

    if not SUBSET_CSV.is_file() or not SUBSET_META.is_file():
        raise RuntimeError("PI1M_50k fixed-subset CSV/metadata is missing")
    subset_meta = _json(SUBSET_META)
    if int(subset_meta.get("sample_size", -1)) != 50000:
        raise RuntimeError("fixed subset metadata is not the required 50K cohort")
    if len(dataset) != 50000:
        raise RuntimeError(f"PI1M_50k dataset length is {len(dataset)}, expected 50000")

    keys: list[bytes] = []
    smiles_by_key: dict[bytes, str] = {}
    eligible_geometry: list[int] = []
    geometry_failures: list[dict[str, Any]] = []
    graph_failures = 0
    for index in range(len(dataset)):
        item = dataset[index]
        smiles = str(getattr(item, "smiles", "")).strip()
        key = sample_key_from_smiles(smiles)
        keys.append(key)
        if key in smiles_by_key:
            raise RuntimeError(f"duplicate sample key in fixed 50K subset: {key.hex()}")
        smiles_by_key[key] = smiles
        if _geometry_ok(item):
            eligible_geometry.append(index)
        else:
            graph_failures += int(not bool(getattr(item, "graph_available", True)))
            geometry_failures.append({
                "index": int(index),
                "sample_key": key.hex(),
                "smiles": smiles,
                "failure_code": str(getattr(item, "trimer_failure_code", "")),
                "geometry_valid": bool(getattr(item, "trimer_geometry_valid", False)),
                "geometry_is_3d": bool(getattr(item, "trimer_geometry_is_3d", False)),
                "graph_available": bool(getattr(item, "graph_available", True)),
            })

    saved = _load_saved_md()
    if saved is None or set(saved[0]) != set(smiles_by_key):
        md_values, md_stats = _compute_md_table(smiles_by_key)
        md_source = "computed_route_local_exact_original_mips"
    else:
        md_values, saved_meta = saved
        md_stats = dict(saved_meta.get("stats", {}))
        md_source = "reused_route_local_exact_original_mips"

    md_failures = [key.hex() for key, value in md_values.items() if not bool(np.isfinite(value).all())]
    if md_failures:
        raise RuntimeError("MD200 table contains non-finite values after fallback")
    # The route-local MD calculator records parse exceptions separately.  The
    # current table stores zeros for them; exclude only keys marked by the
    # persisted stats/error manifest when present.
    md_exception_keys = set(_json(PRETRAIN_ROOT / "md200_failures.json").get("sample_keys", [])) if (PRETRAIN_ROOT / "md200_failures.json").is_file() else set()
    eligible = [index for index in eligible_geometry if keys[index].hex() not in md_exception_keys]

    PRETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRETRAIN_ROOT / "md200_values.npz",
        sample_keys=np.asarray([key.hex() for key in keys]),
        values=np.stack([md_values[key] for key in keys], axis=0).astype(np.float32),
    )
    _write(PRETRAIN_ROOT / "md200_provenance.json", {
        "schema": "original-mips-md200-50k-joint-static-table-v1",
        "protocol": OriginalMIPSMD200.protocol,
        "source_module": "src/modules/original_mips_md200.py",
        "source_sha256": _sha(ROOT / "src/modules/original_mips_md200.py"),
        "sample_count": len(keys),
        "ordered_sample_key_hash": _ordered_key_hash(keys),
        "sorted_sample_key_hash": _sorted_key_hash(keys),
        "values_path": str(PRETRAIN_ROOT / "md200_values.npz"),
        "stats": md_stats,
        "source": md_source,
        "active_modality": "md",
        "failure_semantics": "Original-MIPS None/NaN zero semantics; exceptional rows recorded and excluded",
    })
    # Persist exceptional rows separately so the table remains a shape-stable
    # exact-key artifact while the training eligibility decision is auditable.
    if isinstance(md_stats.get("exception_sample_keys"), list):
        md_exception_keys.update(str(value) for value in md_stats["exception_sample_keys"])
    _write(PRETRAIN_ROOT / "md200_failures.json", {
        "schema": "original-mips-md200-50k-failures-v1",
        "sample_keys": sorted(md_exception_keys),
        "count": len(md_exception_keys),
    })
    geometry_failure_path = PRETRAIN_ROOT / "geometry_failures.jsonl"
    with geometry_failure_path.open("w", encoding="utf-8") as handle:
        for row in geometry_failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write(PRETRAIN_ROOT / "50k_geometry_coverage.json", {
        "schema": "original-mips-atomic-pc-center-ru-joint-50k-geometry-coverage-v1",
        "subset_csv": str(SUBSET_CSV),
        "subset_metadata": subset_meta,
        "subset_count": len(keys),
        "geometry_success_count": len(eligible_geometry),
        "geometry_failure_count": len(geometry_failures),
        "graph_failure_count": int(graph_failures),
        "md_exception_count": len(md_exception_keys),
        "training_eligible_count": len(eligible),
        "ordered_sample_key_hash": _ordered_key_hash(keys),
        "sorted_sample_key_hash": _sorted_key_hash(keys),
        "geometry_protocol": "etkdgv3x4-mmff94-relax200-lowest-finite-v1",
        "geometry_source": "immutable data/processed/mips_trimer_scage/trimer content-addressed LMDB",
        "reuse_existing_geometry": True,
        "generation_attempted": False,
        "retry": False,
        "fallback": False,
        "full_1m_generation": False,
        "failure_artifact": str(geometry_failure_path),
    })
    _write(PRETRAIN_ROOT / "50k_subset_manifest.json", {
        "schema": "original-mips-atomic-pc-center-ru-joint-50k-subset-v1",
        "sample_count": len(keys),
        "ordered_sample_keys": [key.hex() for key in keys],
        "eligible_indices": [int(index) for index in eligible],
        "geometry_success_indices": [int(index) for index in eligible_geometry],
        "md_exception_keys": sorted(md_exception_keys),
        "sample_key_hash": _ordered_key_hash(keys),
    })
    return {
        "keys": keys,
        "smiles_by_key": smiles_by_key,
        "md_values": md_values,
        "eligible_indices": eligible,
        "geometry_success_indices": eligible_geometry,
        "geometry_failures": geometry_failures,
        "md_exception_keys": md_exception_keys,
        "md_stats": md_stats,
    }


def _load_subset_info() -> dict[str, Any]:
    manifest = _json(PRETRAIN_ROOT / "50k_subset_manifest.json")
    coverage = _json(PRETRAIN_ROOT / "50k_geometry_coverage.json")
    payload = np.load(PRETRAIN_ROOT / "md200_values.npz", allow_pickle=False)
    keys = [bytes.fromhex(str(value)) for value in payload["sample_keys"].tolist()]
    matrix = np.asarray(payload["values"], dtype=np.float32)
    if len(keys) != 50000 or matrix.shape != (50000, 200):
        raise RuntimeError("persisted 50K subset/MD200 artifact shape mismatch")
    if manifest.get("sample_key_hash") != _ordered_key_hash(keys):
        raise RuntimeError("persisted subset manifest key hash mismatch")
    if int(coverage.get("training_eligible_count", -1)) != len(manifest.get("eligible_indices", [])):
        raise RuntimeError("persisted geometry/eligible-count mismatch")
    return {
        "keys": keys,
        "md_values": {key: matrix[index].copy() for index, key in enumerate(keys)},
        "eligible_indices": [int(value) for value in manifest["eligible_indices"]],
        "geometry_success_indices": [int(value) for value in manifest.get("geometry_success_indices", [])],
        "coverage": coverage,
        "manifest": manifest,
    }


def _load_o8_checkpoint(graph_encoder: nn.Module) -> dict[str, Any]:
    if not O8_CHECKPOINT.is_file():
        raise RuntimeError(f"missing current O8 checkpoint: {O8_CHECKPOINT}")
    checkpoint = torch.load(O8_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("current O8 checkpoint has no state_dict")
    target_state = graph_encoder.state_dict()
    loaded = []
    for name, value in target_state.items():
        source_name = "model.o8." + name
        if source_name not in state:
            raise RuntimeError(f"O8 checkpoint missing {source_name}")
        source = state[source_name]
        if tuple(source.shape) != tuple(value.shape):
            raise RuntimeError(f"O8 checkpoint shape mismatch for {name}")
        target_state[name] = source.detach().clone()
        loaded.append(name)
    graph_encoder.load_state_dict(target_state, strict=True)
    return {
        "path": str(O8_CHECKPOINT),
        "sha256": _sha(O8_CHECKPOINT),
        "checkpoint_schema": checkpoint.get("schema"),
        "checkpoint_step": int(checkpoint.get("step", 5000)),
        "loaded_tensor_count": len(loaded),
    }


def _grad_check(grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> dict[str, Any]:
    finite = bool(grads) and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    nonzero = bool(grads) and any(g is not None and float(g.detach().abs().sum().cpu()) > 0.0 for g in grads)
    norm = 0.0
    if grads:
        norm = float(torch.sqrt(sum((g.detach().float().pow(2).sum() for g in grads if g is not None), torch.zeros((), device=next(g for g in grads if g is not None).device))).cpu()) if any(g is not None for g in grads) else 0.0
    return {"all_finite": bool(finite), "any_nonzero": bool(nonzero), "l2_norm": norm, "parameter_tensors": len(grads)}


def _run_prestart_gate(dataset: UniDataset, info: dict[str, Any]) -> dict[str, Any]:
    eligible = list(info["eligible_indices"])
    if len(eligible) < 2:
        raise RuntimeError("fewer than two eligible 50K rows remain for the one-batch gate")
    collator = OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader = DataLoader(
        Subset(dataset, eligible[:2]),
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)
    model = OriginalMIPSAtomicPCJointPretrainer().to(device)
    o8_source = _load_o8_checkpoint(model.graph_encoder)
    model.train()
    batch = batch.to(device)
    groups = trainable_group_parameters(model)

    model.zero_grad(set_to_none=True)
    torch.manual_seed(4201)
    first = model(batch, seed=42, stream_step=0)
    mae_loss = first["mae_loss"]
    ccdd_loss = first["ccdd_loss"]
    mae_grads = {}
    for name in ("o8", "atomic_pc", "kfuse"):
        mae_grads[name] = _grad_check(torch.autograd.grad(
            mae_loss, groups[name], retain_graph=True, allow_unused=True
        ))
    ccdd_grads = _grad_check(torch.autograd.grad(
        ccdd_loss, groups["atomic_pc"], retain_graph=True, allow_unused=True
    ))
    selected = {
        "message_layers": [p for n, p in model.atomic_point_encoder.named_parameters() if n.startswith("layers.")],
        "center_pool": [p for n, p in model.atomic_point_encoder.named_parameters() if n.startswith("pool_projection") or n.startswith("pool_score")],
        "atomic_pc512_projection": [p for n, p in model.atomic_point_encoder.named_parameters() if n.startswith("output_projection")],
    }
    selected_ccdd = {
        name: _grad_check(torch.autograd.grad(ccdd_loss, params, retain_graph=True, allow_unused=True))
        for name, params in selected.items()
    }

    model.zero_grad(set_to_none=True)
    torch.manual_seed(4202)
    second = model(batch, seed=42, stream_step=1)
    second["total_loss"].backward()
    total_grads = {
        name: gradient_summary(groups[name]) for name in ("o8", "atomic_pc", "kfuse")
    }
    noisy_edge_contract_passed = bool(torch.equal(second["edge_index"], second["atomic_aux"]["edge_index"]))
    clean_edge_reference, _ = model.atomic_point_encoder._edges(
        second["clean_coords"], batch.atomic_point_cloud.batch.long().to(device)
    )
    same_as_clean_reference = bool(torch.equal(second["edge_index"], clean_edge_reference))
    point_stats = second["atomic_aux"]
    attention = second.get("fusion_attention")
    attention_mean = None
    if attention is not None and attention.numel():
        attention_mean = [float(value) for value in attention.float().mean(dim=(0, 1)).detach().cpu().tolist()]
    source_audit = _json(ROOT / "results/original_mips_atomic_pc_center_ru_v1/source_audit.json")
    old_kfuse = _json(ROOT / "results/original_mips_atomic_pc_center_ru_v1/kfuse_parity.json")
    pooling = _json(ROOT / "results/original_mips_atomic_pc_center_ru_v1/pooling_contract.json")
    md_provenance = _json(PRETRAIN_ROOT / "md200_provenance.json")
    gates = {
        "JOINT_FORWARD": bool(first["fusion_call_count"] == 1 and first["atom_logits"].ndim == 2),
        "JOINT_BACKWARD": all(total_grads[name]["all_finite"] for name in ("o8", "atomic_pc", "kfuse")),
        "O8_GRADIENT": bool(mae_grads["o8"]["all_finite"] and mae_grads["o8"]["any_nonzero"]),
        "ATOMIC_PC_GRADIENT": bool(mae_grads["atomic_pc"]["all_finite"] and mae_grads["atomic_pc"]["any_nonzero"]),
        "KFUSE_GRADIENT": bool(mae_grads["kfuse"]["all_finite"] and mae_grads["kfuse"]["any_nonzero"]),
        "CCDD_ATOMIC_PC_GRADIENT": bool(ccdd_grads["all_finite"] and ccdd_grads["any_nonzero"]),
        "CCDD_MESSAGE_LAYERS": bool(selected_ccdd["message_layers"]["all_finite"] and selected_ccdd["message_layers"]["any_nonzero"]),
        "CCDD_CENTER_RU_POOLING": bool(selected_ccdd["center_pool"]["all_finite"] and selected_ccdd["center_pool"]["any_nonzero"]),
        "CCDD_ATOMIC_PC512_PROJECTION": bool(selected_ccdd["atomic_pc512_projection"]["all_finite"] and selected_ccdd["atomic_pc512_projection"]["any_nonzero"]),
        "MAE_LOSS_FINITE": bool(torch.isfinite(first["mae_loss"]).all()),
        "CCDD_LOSS_FINITE": bool(torch.isfinite(first["ccdd_loss"]).all()),
        "NOISY_KNN": bool(second["edge_index"].ndim == 2 and second["edge_index"].size(0) == 2 and second["edge_index"].size(1) > 0 and int(point_stats.get("point_count_used_for_message_passing", 0)) == int(point_stats.get("point_count", -1))),
        # ``same_as_clean_reference`` is an audit-only comparison.  The model
        # itself never constructs/passes a clean kNN graph; its auxiliary edge
        # index is exactly the noisy pass consumed by CCDD.
        "CLEAN_KNN_LEAKAGE": bool(not same_as_clean_reference and noisy_edge_contract_passed),
        "MD200_PATH": md_provenance.get("protocol") == OriginalMIPSMD200.protocol and md_provenance.get("active_modality") == "md",
        "KFUSE_PARITY": old_kfuse.get("gate", {}).get("ORIGINAL_KFUSE_PARITY") == "PASS" and source_audit.get("official_head_verified") is True,
        "CENTER_RU_CONTRACT": str(model.atomic_point_encoder.pool_scope) == "center" and model.atomic_point_encoder.num_layers == 4 and model.atomic_point_encoder.k_neighbors == 24 and int(point_stats.get("point_count_used_for_pooling", -1)) == int(point_stats.get("central_point_count", -2)) and pooling.get("gate", {}).get("CENTER_ONLY_POOLING") == "PASS",
        "SAMPLE_ALIGNMENT": len(info["keys"]) == 50000 and len(set(info["keys"])) == 50000 and len(info["eligible_indices"]) == int(info.get("coverage", {}).get("training_eligible_count", len(info["eligible_indices"]))) and all(key in info["md_values"] for key in info["keys"]),
    }
    payload = {
        "schema": "original-mips-atomic-pc-center-ru-joint-prestart-gate-v1",
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "all_pass": bool(all(gates.values())),
        "objective": {"L": "L_MAE + L_CCDD", "mae_weight": 1.0, "ccdd_weight": 1.0},
        "fixed_contract": {
            "atom_mask_rate": ATOM_MASK_RATE,
            "md_kvec_mask_rate": MD_KVEC_MASK_RATE,
            "coordinate_noise_sigma": COORD_NOISE_SIGMA,
            "knn": 24,
            "atomic_layers": 4,
            "pool_scope": "center",
            "active_modalities": ["md", "atomic_pc"],
            "AP3D": False,
            "MCP": False,
            "clean_kNN_for_message_passing": False,
            "CLEAN_KNN_LEAKAGE": "NO",
        },
        "one_batch": {
            "batch_graph_count": int(batch.graph_available.numel()),
            "mae_loss": float(first["mae_loss"].detach().cpu()),
            "ccdd_loss": float(first["ccdd_loss"].detach().cpu()),
            "atom_count": int(first["atom_count"]),
            "ccdd_edge_count": int(first["ccdd_edge_count"]),
            "md_mask_fraction": float(first["md_mask_fraction"]),
            "mae_gradients": mae_grads,
            "ccdd_atomic_gradients": ccdd_grads,
            "ccdd_selected_gradients": selected_ccdd,
            "total_gradients": total_grads,
            "same_as_clean_reference": same_as_clean_reference,
            "same_as_atomic_aux_edge": noisy_edge_contract_passed,
            "kFuse_attention_mean_md_atomic": attention_mean,
            "point_stats": {key: value for key, value in point_stats.items() if not torch.is_tensor(value)},
        },
        "o8_initialization": o8_source,
        "coverage": info["coverage"],
        "forbidden_actions": {
            "atomic_pc_only_pretraining": "NO",
            "fixed_vector_disturb_fp": "NO",
            "AP3D": "NO",
            "MCP": "NO",
            "geometry_retry": "NO",
            "geometry_fallback": "NO",
            "full_1m_generation": "NO",
            "loss_weight_sweep": "NO",
        },
    }
    _write(RESULT_ROOT / "prestart_gate.json", payload)
    return payload


def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def _make_optimizer(model: nn.Module, total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=2e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.0,
    )
    warmup_steps = 2000
    end_lr = 1e-9

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, max(0.0, (float(step) - warmup_steps) / max(1.0, float(total_steps - warmup_steps))))
        return end_lr / 2e-4 + (1.0 - end_lr / 2e-4) * (1.0 - progress)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _train(dataset: UniDataset, info: dict[str, Any], gate: dict[str, Any], *, batch_size: int, workers: int, max_steps: int | None) -> dict[str, Any]:
    if not bool(gate.get("all_pass")):
        raise RuntimeError("joint pretraining gate is not all PASS; training was not started")
    eligible = list(info["eligible_indices"])
    collator = OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader_kwargs: dict[str, Any] = {
        "dataset": Subset(dataset, eligible),
        "batch_size": int(batch_size),
        "shuffle": True,
        "drop_last": False,
        "num_workers": int(workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    total_steps = len(loader) if max_steps is None else min(int(max_steps), len(loader))
    if total_steps <= 0:
        raise RuntimeError("joint pretraining loader is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)
    model = OriginalMIPSAtomicPCJointPretrainer().to(device)
    o8_source = _load_o8_checkpoint(model.graph_encoder)
    optimizer, scheduler = _make_optimizer(model, total_steps)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups = trainable_group_parameters(model)
    metrics_path = PRETRAIN_ROOT / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    iterator = iter(loader)
    model.train()
    for step in range(1, total_steps + 1):
        batch = next(iterator)
        step_started = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            output = model(
                batch,
                seed=42,
                stream_step=step,
                atom_mask_rate=ATOM_MASK_RATE,
                md_kvec_mask_rate=MD_KVEC_MASK_RATE,
                coord_noise_sigma=COORD_NOISE_SIGMA,
            )
        loss = output["total_loss"]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"non-finite joint loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite joint gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        step_seconds = time.perf_counter() - step_started
        attention = output.get("fusion_attention")
        attention_mean = [None, None]
        if attention is not None and attention.numel():
            attention_mean = [float(value) for value in attention.float().mean(dim=(0, 1)).detach().cpu().tolist()]
        geometry_seconds = float(getattr(batch, "geometry_loading_seconds", 0.0))
        row = {
            "step": int(step),
            "total_loss": float(output["total_loss"].detach().float().cpu()),
            "mae_loss": float(output["mae_loss"].detach().float().cpu()),
            "ccdd_loss": float(output["ccdd_loss"].detach().float().cpu()),
            "masked_atom_accuracy": float(output["atom_correct"] / max(1, output["atom_count"])),
            "masked_atom_count": int(output["atom_count"]),
            "ccdd_edge_count": int(output["ccdd_edge_count"]),
            "distance_mae": float((output["ccdd_pred"] - output["ccdd_target"]).abs().mean().detach().float().cpu()) if output["ccdd_pred"].numel() else 0.0,
            "o8_grad_norm": gradient_summary(groups["o8"])["l2_norm"],
            "atomic_pc_grad_norm": gradient_summary(groups["atomic_pc"])["l2_norm"],
            "kfuse_grad_norm": gradient_summary(groups["kfuse"])["l2_norm"],
            "clipped_grad_norm": grad_norm,
            "kFuse_attention_md": attention_mean[0],
            "kFuse_attention_atomic_pc": attention_mean[1],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "step_seconds": float(step_seconds),
            "throughput_graphs_per_second": float(batch.graph_available.numel() / max(step_seconds, 1e-9)),
            "geometry_loading_seconds": geometry_seconds,
            "gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        records.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step == 1 or step % 50 == 0 or step == total_steps:
            print(
                f"[joint-pretrain] step={step}/{total_steps} total={row['total_loss']:.6f} "
                f"mae={row['mae_loss']:.6f} ccdd={row['ccdd_loss']:.6f} "
                f"acc={row['masked_atom_accuracy']:.4f} throughput={row['throughput_graphs_per_second']:.2f}",
                flush=True,
            )
    final = records[-1]
    peak_allocated = int(max(row["gpu_memory_allocated_bytes"] for row in records))
    peak_reserved = int(max(row["gpu_memory_reserved_bytes"] for row in records))
    checkpoint = {
        "schema": JOINT_CHECKPOINT_SCHEMA,
        "architecture": OriginalMIPSAtomicPCJointPretrainer.architecture_name,
        "pretrain_schema": JOINT_PRETRAIN_SCHEMA,
        "step": int(total_steps),
        "subset_count": 50000,
        "training_eligible_count": len(eligible),
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "metadata": {
            "o8_initialization": o8_source,
            "active_modalities": ["md", "atomic_pc"],
            "AP3D": False,
            "MCP": False,
            "atom_mask_rate": ATOM_MASK_RATE,
            "md_kvec_mask_rate": MD_KVEC_MASK_RATE,
            "coordinate_noise_sigma": COORD_NOISE_SIGMA,
            "atomic_knn": 24,
            "atomic_layers": 4,
            "atomic_pool_scope": "center",
            "objective": {"L": "L_MAE + L_CCDD", "mae_weight": 1.0, "ccdd_weight": 1.0},
            "optimizer": {"name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.98], "weight_decay": 0.0, "warmup_steps": 2000},
            "subset_sample_key_hash": info["coverage"].get("ordered_sample_key_hash"),
            "geometry_protocol": info["coverage"].get("geometry_protocol"),
            "heads": {"mae": "atom_head", "ccdd": "ccdd_head", "downstream": "removed"},
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    checkpoint_path = PRETRAIN_ROOT / "joint_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-joint-pretrain-summary-v1",
        "status": "COMPLETE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "subset_count": 50000,
        "training_eligible_count": len(eligible),
        "optimizer_steps": int(total_steps),
        "final_total_loss": final["total_loss"],
        "final_mae_loss": final["mae_loss"],
        "final_ccdd_loss": final["ccdd_loss"],
        "final_masked_atom_accuracy": final["masked_atom_accuracy"],
        "final_distance_mae": final["distance_mae"],
        "pretraining_wall_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "o8_pretrained": True,
        "atomic_pc_pretrained": True,
        "kfuse_pretrained": True,
        "training_heads_saved": ["atom_head", "ccdd_head"],
        "training_heads_removed_downstream": True,
        "reference_no_pretrain_macro_r2": REFERENCE_MACRO_R2,
        "metrics_path": str(metrics_path),
    }
    _write(PRETRAIN_ROOT / "pretraining_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    if bool(args.preflight_only) == bool(args.train):
        raise SystemExit("choose exactly one of --preflight-only or --train")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        dataset = _build_dataset()
        info = _prepare_subset(dataset)
        gate = _run_prestart_gate(dataset, {**info, "coverage": _json(PRETRAIN_ROOT / "50k_geometry_coverage.json")})
        print(json.dumps({"all_pass": gate["all_pass"], "gates": gate["gates"]}, ensure_ascii=False, indent=2))
        if not gate["all_pass"]:
            raise SystemExit("joint pretraining pre-start gate failed; training was not started")
        print("joint pretraining pre-start gate PASS; no training started")
        return
    gate = _json(RESULT_ROOT / "prestart_gate.json")
    if not bool(gate.get("all_pass")):
        raise SystemExit("joint pretraining pre-start gate is not PASS; training was not started")
    dataset = _build_dataset()
    info = _load_subset_info()
    info["coverage"] = _json(PRETRAIN_ROOT / "50k_geometry_coverage.json")
    summary = _train(
        dataset,
        info,
        gate,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
