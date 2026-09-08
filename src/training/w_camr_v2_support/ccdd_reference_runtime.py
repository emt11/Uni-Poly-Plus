#!/usr/bin/env python
"""Atomic-PC CCDD-only gradient attribution diagnostic.

This route is intentionally separate from the completed joint-pretraining
route.  It reuses the exact joint ``OriginalMIPSAtomicPCJointPretrainer``
forward and the frozen 50K/geometry artifacts, but the optimizer objective is
``L_PC = L_CCDD`` and only ``atomic_point_encoder`` plus the temporary CCDD
head are trainable.  The downstream phase strictly transplants only the
resulting Atomic-PC component into the audited Base Center-RU route.

The script has three explicit modes:

``--preflight-only``
    Read-only artifact/protocol checks plus one-batch gradient attribution.

``--train``
    Require the persisted all-PASS gate, then run exactly the matched 1504
    optimizer steps over the persisted 48,101 eligible rows.

``--downstream`` / ``--aggregate``
    Run or aggregate the one matched Center-RU downstream transplant.

No geometry generation, model-source modification, loss sweep, Base rerun,
Full-Joint rerun, 100K/1M route, or Joint-v2 route is reachable here.
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
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.w_camr_v2_support import center_runtime as downstream_route  # noqa: E402
from src.training.w_camr_v2_support import cohort_runtime as joint_route  # noqa: E402
from src.dataset.original_mips_atomic_pc_joint import (  # noqa: E402
    OriginalMIPSAtomicPCJointCollator,
)
from src.training.pretrain.original_mips_atomic_pc_joint import (  # noqa: E402
    ATOM_MASK_RATE,
    COORD_NOISE_SIGMA,
    JOINT_CHECKPOINT_SCHEMA,
    MD_KVEC_MASK_RATE,
    OriginalMIPSAtomicPCJointPretrainer,
    gradient_summary,
)
from src.modules.original_mips_md200 import OriginalMIPSMD200  # noqa: E402
from src.utils import set_global_seed  # noqa: E402


RESULT_ROOT = ROOT / "results/original_mips_atomic_pc_ccdd_only_gradient_attribution_v1"
PRETRAIN_ROOT = RESULT_ROOT / "pretraining"
DOWNSTREAM_ROOT = RESULT_ROOT / "downstream"
SHARD_ROOT = DOWNSTREAM_ROOT / "shards"

JOINT_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_joint_pretrain_v1"
JOINT_PRETRAIN_ROOT = JOINT_ROOT / "pretraining"
JOINT_CHECKPOINT = JOINT_PRETRAIN_ROOT / "joint_checkpoint.pt"
BASE_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_v1"
OLD_T2_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_transplant_diagnosis_v1/T2_ATOMIC_PC_ONLY/downstream"
BASE_O8_CHECKPOINT = downstream_route.CHECKPOINT

TASKS = tuple(downstream_route.TASKS)
FOLDS = tuple(downstream_route.FOLDS)
REFERENCE_BASE_MACRO_R2 = 0.8403314216055405
REFERENCE_OLD_T2_MACRO_R2 = 0.8387253670489404
REFERENCE_FULL_JOINT_MACRO_R2 = 0.8325428908000778
EXPECTED_SUBSET_COUNT = 50000
EXPECTED_ELIGIBLE_COUNT = 48101
EXPECTED_PRETRAIN_STEPS = 1504
EXPECTED_BATCH_SIZE = 32

CCDD_CHECKPOINT_SCHEMA = (
    "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-checkpoint-v1"
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_items(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module: nn.Module) -> str:
    return _digest_items(module.state_dict().items())


def _state_prefix_digest(state: dict[str, torch.Tensor], prefix: str) -> str:
    values = [
        (name[len(prefix) :], tensor)
        for name, tensor in state.items()
        if name.startswith(prefix)
    ]
    if not values:
        raise RuntimeError(f"state has no tensors for prefix {prefix!r}")
    return _digest_items(values)


def _module_summary(module: nn.Module) -> dict[str, Any]:
    state = module.state_dict()
    parameter_names = {name for name, _ in module.named_parameters()}
    return {
        "tensor_count": len(state),
        "parameter_count": int(sum(parameter.numel() for parameter in module.parameters())),
        "buffer_count": int(sum(name not in parameter_names for name in state)),
        "hash": _module_digest(module),
    }


def _grad_report(
    parameters: list[nn.Parameter],
    grads: Iterable[torch.Tensor | None],
    *,
    none_is_zero: bool = False,
) -> dict[str, Any]:
    grads = list(grads)
    finite_values = [gradient for gradient in grads if gradient is not None]
    finite = bool(grads) and all(bool(torch.isfinite(gradient).all()) for gradient in finite_values)
    if not none_is_zero and len(finite_values) != len(grads):
        finite = False
    nonzero = any(
        gradient is not None and bool(torch.count_nonzero(gradient.detach()).item())
        for gradient in grads
    )
    exact_zero = all(
        gradient is None or bool(torch.count_nonzero(gradient.detach()).item()) == 0
        for gradient in grads
    )
    finite_gradients = [gradient for gradient in finite_values if bool(torch.isfinite(gradient).all())]
    norm = 0.0
    max_abs = 0.0
    if finite_gradients:
        norm = float(
            torch.sqrt(
                sum(
                    (gradient.detach().float().pow(2).sum() for gradient in finite_gradients),
                    torch.zeros((), device=finite_gradients[0].device),
                )
            ).cpu()
        )
        max_abs = float(max(float(gradient.detach().abs().max().cpu()) for gradient in finite_gradients))
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_tensors": len(parameters),
        "gradient_tensors_present": len(finite_values),
        "none_count": int(sum(gradient is None for gradient in grads)),
        "all_finite": bool(finite),
        "any_nonzero": bool(nonzero),
        "exact_zero": bool(exact_zero),
        "l2_norm": norm,
        "max_abs": max_abs,
    }


def _grad_report_from_autograd(
    parameters: list[nn.Parameter],
    loss: torch.Tensor,
    *,
    retain_graph: bool = False,
    none_is_zero: bool = False,
) -> dict[str, Any]:
    if not bool(loss.requires_grad):
        grads: tuple[torch.Tensor | None, ...] = tuple(None for _ in parameters)
    else:
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    return _grad_report(parameters, grads, none_is_zero=none_is_zero)


def _load_prior_contract() -> dict[str, Any]:
    """Read-only verification of all immutable inputs from the joint route."""

    required_paths = {
        "joint_gate": JOINT_ROOT / "prestart_gate.json",
        "joint_summary": JOINT_PRETRAIN_ROOT / "pretraining_summary.json",
        "coverage": JOINT_PRETRAIN_ROOT / "50k_geometry_coverage.json",
        "manifest": JOINT_PRETRAIN_ROOT / "50k_subset_manifest.json",
        "md_provenance": JOINT_PRETRAIN_ROOT / "md200_provenance.json",
        "base_metrics": BASE_ROOT / "training/per_task_metrics.json",
        "old_t2_metrics": OLD_T2_ROOT / "per_task_metrics.json",
        "full_joint_metrics": JOINT_ROOT / "downstream/per_task_metrics.json",
        "base_gate": BASE_ROOT / "prestart_gate.json",
        "source_audit": BASE_ROOT / "source_audit.json",
        "pooling": BASE_ROOT / "pooling_contract.json",
        "kfuse": BASE_ROOT / "kfuse_parity.json",
        "cohort": BASE_ROOT / "cohort_parity.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required immutable artifact is missing: {missing[:4]}")
    artifacts = {name: _json(path) for name, path in required_paths.items()}
    joint_gate = artifacts["joint_gate"]
    summary = artifacts["joint_summary"]
    coverage = artifacts["coverage"]
    manifest = artifacts["manifest"]
    md_provenance = artifacts["md_provenance"]
    base_macro = float(artifacts["base_metrics"]["macro"]["macro_test_r2"])
    old_t2_macro = float(artifacts["old_t2_metrics"]["macro"]["macro_test_r2"])
    full_macro = float(artifacts["full_joint_metrics"]["macro"]["macro_test_r2"])
    if not math.isclose(base_macro, REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"Base macro mismatch: {base_macro}")
    if not math.isclose(old_t2_macro, REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"old T2 macro mismatch: {old_t2_macro}")
    if not math.isclose(full_macro, REFERENCE_FULL_JOINT_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"Full-Joint macro mismatch: {full_macro}")
    if not bool(joint_gate.get("all_pass")):
        raise RuntimeError("the completed joint route prestart gate is not PASS")
    if summary.get("status") != "COMPLETE" or int(summary.get("optimizer_steps", -1)) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("completed joint pretraining summary is not the required 1504-step artifact")
    if int(summary.get("training_eligible_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("joint pretraining eligible count is not 48,101")
    if int(coverage.get("subset_count", -1)) != EXPECTED_SUBSET_COUNT or int(coverage.get("training_eligible_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("persisted 50K coverage has an unexpected count")
    if coverage.get("reuse_existing_geometry") is not True or coverage.get("generation_attempted") is not False or coverage.get("geometry_source") != "immutable data/processed/mips_trimer_scage/trimer content-addressed LMDB":
        raise RuntimeError("persisted geometry contract is not the frozen no-generation route")
    if coverage.get("geometry_protocol") != "etkdgv3x4-mmff94-relax200-lowest-finite-v1":
        raise RuntimeError("unexpected frozen geometry protocol")
    if manifest.get("schema") != "original-mips-atomic-pc-center-ru-joint-50k-subset-v1":
        raise RuntimeError("unexpected persisted 50K manifest schema")
    if int(manifest.get("sample_count", -1)) != EXPECTED_SUBSET_COUNT or len(manifest.get("ordered_sample_keys", [])) != EXPECTED_SUBSET_COUNT:
        raise RuntimeError("persisted 50K manifest is not exactly 50,000 rows")
    values_path = JOINT_PRETRAIN_ROOT / "md200_values.npz"
    if not values_path.is_file():
        raise RuntimeError("frozen joint-route MD200 values are missing")
    md_payload = np.load(values_path, allow_pickle=False)
    md_keys = [str(value) for value in md_payload["sample_keys"].tolist()]
    md_values = np.asarray(md_payload["values"], dtype=np.float32)
    if len(md_keys) != EXPECTED_SUBSET_COUNT or md_values.shape != (EXPECTED_SUBSET_COUNT, 200) or not bool(np.isfinite(md_values).all()):
        raise RuntimeError("frozen MD200 table shape/finite gate failed")
    if list(manifest.get("ordered_sample_keys", [])) != md_keys:
        raise RuntimeError("frozen subset manifest and MD200 key order differ")
    if manifest.get("sample_key_hash") != coverage.get("ordered_sample_key_hash"):
        raise RuntimeError("frozen subset/coverage key hash differs")
    fixed_contract = joint_gate.get("fixed_contract", {})
    expected_contract = {
        "atom_mask_rate": ATOM_MASK_RATE,
        "md_kvec_mask_rate": MD_KVEC_MASK_RATE,
        "coordinate_noise_sigma": COORD_NOISE_SIGMA,
        "knn": 24,
        "atomic_layers": 4,
        "pool_scope": "center",
        "clean_kNN_for_message_passing": False,
        "CLEAN_KNN_LEAKAGE": "NO",
    }
    contract_match = all(fixed_contract.get(key) == value for key, value in expected_contract.items())
    if not contract_match:
        raise RuntimeError("joint fixed Atomic-PC/CCDD contract does not match the requested route")
    return {
        "paths": {name: str(path) for name, path in required_paths.items()},
        "artifacts": artifacts,
        "base_macro_r2": base_macro,
        "old_t2_macro_r2": old_t2_macro,
        "full_joint_macro_r2": full_macro,
        "subset_key_hash": manifest.get("sample_key_hash"),
        "ordered_sample_key_hash": coverage.get("ordered_sample_key_hash"),
        "sorted_sample_key_hash": coverage.get("sorted_sample_key_hash"),
        "md_key_count": len(md_keys),
        "md_shape": list(md_values.shape),
        "md_sha256": _sha(values_path),
        "md_protocol": md_provenance.get("protocol"),
        "fixed_contract": fixed_contract,
    }


def _module_initialization_audit() -> dict[str, Any]:
    """Prove identical construction semantics without using trained weights."""

    # The historical downstream audit records this exact fold seed as the
    # Base fresh-initialization semantic anchor.  The pretraining itself uses
    # seed 42; both checks are recorded below.
    fold_seed = 42 + 1009 * downstream_route._task_offset("eat")
    set_global_seed(fold_seed)
    base_model, base_meta = downstream_route._center_model()
    base_atomic_hash = _module_digest(base_model.atomic_point_encoder)
    base_graph_hash = _module_digest(base_model.graph_encoder)
    base_fusion_hash = _module_digest(base_model.fusion)
    set_global_seed(fold_seed)
    pretrainer = OriginalMIPSAtomicPCJointPretrainer()
    joint_route._load_o8_checkpoint(pretrainer.graph_encoder)
    pre_atomic_hash = _module_digest(pretrainer.atomic_point_encoder)
    pre_graph_hash = _module_digest(pretrainer.graph_encoder)
    pre_fusion_hash = _module_digest(pretrainer.fusion)
    set_global_seed(42)
    seed42_a = OriginalMIPSAtomicPCJointPretrainer()
    seed42_hash_a = _module_digest(seed42_a.atomic_point_encoder)
    set_global_seed(42)
    seed42_b = OriginalMIPSAtomicPCJointPretrainer()
    seed42_hash_b = _module_digest(seed42_b.atomic_point_encoder)
    checks = {
        "ATOMIC_PC_CONSTRUCTION_MATCH": pre_atomic_hash == base_atomic_hash,
        "GRAPH_CONSTRUCTION_MATCH": pre_graph_hash == base_graph_hash,
        "KFUSE_CONSTRUCTION_MATCH": pre_fusion_hash == base_fusion_hash,
        "SEED42_DETERMINISTIC_RECONSTRUCTION": seed42_hash_a == seed42_hash_b,
        "ATOMIC_PC_SHAPE_MATCH": list(pretrainer.atomic_point_encoder.state_dict()["output_projection.weight"].shape) == [512, 256],
    }
    return {
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "all_pass": all(checks.values()),
        "fold_seed_anchor": int(fold_seed),
        "base_checkpoint": base_meta,
        "base_hashes": {"graph_encoder": base_graph_hash, "atomic_point_encoder": base_atomic_hash, "fusion": base_fusion_hash},
        "pretrainer_fresh_hashes": {"graph_encoder": pre_graph_hash, "atomic_point_encoder": pre_atomic_hash, "fusion": pre_fusion_hash},
        "seed42_atomic_pc_hash": seed42_hash_a,
        "constructor": "OriginalMIPSAtomicPCJointPretrainer() and audited Base _center_model() with same seed/construction order",
    }


def _new_ccdd_only_model(device: torch.device) -> tuple[OriginalMIPSAtomicPCJointPretrainer, dict[str, Any]]:
    """Construct the prior model and freeze every non-PC parameter."""

    set_global_seed(42)
    model = OriginalMIPSAtomicPCJointPretrainer().to(device)
    o8_source = joint_route._load_o8_checkpoint(model.graph_encoder)
    for module in (model.graph_encoder, model.fusion, model.atom_head):
        for parameter in module.parameters():
            parameter.requires_grad = False
    for parameter in model.atomic_point_encoder.parameters():
        parameter.requires_grad = True
    for parameter in model.ccdd_head.parameters():
        parameter.requires_grad = True
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    forbidden_trainable = [
        name for name in trainable_names
        if not (name.startswith("atomic_point_encoder.") or name.startswith("ccdd_head."))
    ]
    if forbidden_trainable:
        raise RuntimeError(f"unexpected trainable non-PC parameters: {forbidden_trainable[:5]}")
    return model, {
        "o8_initialization": o8_source,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "frozen_modules": ["graph_encoder(O8)", "fusion(KFuse)", "atom_head(MAE)", "MD200(no parameters)"],
        "objective": "L_PC = L_CCDD",
    }


def _probe_batch(info: dict[str, Any]):
    dataset = joint_route._build_dataset()
    eligible = list(info["eligible_indices"])
    if len(eligible) < 2:
        raise RuntimeError("fewer than two persisted eligible rows")
    collator = OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader = DataLoader(
        Subset(dataset, eligible[:2]),
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collator,
    )
    return dataset, next(iter(loader))


def _blocked_mae_probe(
    model: OriginalMIPSAtomicPCJointPretrainer,
    output: dict[str, Any],
    batch: Any,
    atomic_parameters: list[nn.Parameter],
) -> dict[str, Any]:
    """Run the MAE path with the Atomic-PC representation explicitly detached.

    This is an attribution gate, not a new objective: the real training loss
    below is CCDD only.  The detached branch demonstrates that any MAE
    diagnostic computation has exactly zero derivative with respect to the PC
    parameters, while the separate full-joint probe records the non-zero
    historical contamination path for context.
    """

    device = output["atomic_pc"].device
    batch_index = torch.as_tensor(batch.batch, device=device, dtype=torch.long).reshape(-1)
    model.fusion.reset_trace()
    fused_detached = model.fusion(
        output["o8_node_states"].detach(),
        {
            "md": output["disturbed_md"].detach(),
            "atomic_pc": output["atomic_pc"].detach(),
        },
        batch_index,
    )
    blocked_mae, atom_count, atom_correct, _ = model._masked_atom_loss(
        batch,
        fused_detached,
        model.atom_head,
        output["canonical_mask"],
    )
    # Keep a differentiable probe node so torch.autograd can return explicit
    # zero tensors rather than an absent graph.  The coefficient is exactly 0.
    zero_link = output["atomic_pc"].sum() * 0.0
    probe_loss = blocked_mae.detach() + zero_link
    report = _grad_report_from_autograd(
        atomic_parameters,
        probe_loss,
        retain_graph=True,
        none_is_zero=True,
    )
    report.update({
        "blocked_mae_loss": float(blocked_mae.detach().float().cpu()),
        "blocked_atom_count": int(atom_count),
        "blocked_atom_correct": int(atom_correct),
        "probe_loss_requires_grad": bool(probe_loss.requires_grad),
    })
    return report


def _gradient_audit(info: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _dataset, batch = _probe_batch(info)
    # Historical full-joint attribution reference: MAE really reaches Atomic-PC.
    set_global_seed(42)
    full_model = OriginalMIPSAtomicPCJointPretrainer().to(device)
    full_o8_source = joint_route._load_o8_checkpoint(full_model.graph_encoder)
    full_model.train()
    full_batch = batch.to(device)
    torch.manual_seed(4201)
    full_output = full_model(full_batch, seed=42, stream_step=0)
    full_atomic_parameters = list(full_model.atomic_point_encoder.parameters())
    full_mae_report = _grad_report_from_autograd(
        full_atomic_parameters,
        full_output["mae_loss"],
        retain_graph=False,
        none_is_zero=False,
    )
    del full_model, full_output, full_batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, freeze_meta = _new_ccdd_only_model(device)
    model.train()
    batch = batch.to(device)
    atomic_parameters = list(model.atomic_point_encoder.parameters())
    ccdd_head_parameters = list(model.ccdd_head.parameters())
    torch.manual_seed(4201)
    output = model(batch, seed=42, stream_step=0)
    ccdd_report = _grad_report_from_autograd(
        atomic_parameters,
        output["ccdd_loss"],
        retain_graph=True,
        none_is_zero=False,
    )
    ccdd_head_report = _grad_report_from_autograd(
        ccdd_head_parameters,
        output["ccdd_loss"],
        retain_graph=True,
        none_is_zero=False,
    )
    blocked_report = _blocked_mae_probe(model, output, batch, atomic_parameters)
    model.zero_grad(set_to_none=True)
    output["ccdd_loss"].backward()
    post_backward_atomic = gradient_summary(atomic_parameters)
    post_backward_ccdd_head = gradient_summary(ccdd_head_parameters)
    frozen_grad_nonempty = {
        "o8": any(parameter.grad is not None for parameter in model.graph_encoder.parameters()),
        "kfuse": any(parameter.grad is not None for parameter in model.fusion.parameters()),
        "atom_head": any(parameter.grad is not None for parameter in model.atom_head.parameters()),
    }
    edge_index = output["edge_index"]
    aux_edge_index = output["atomic_aux"]["edge_index"]
    noisy_edge_contract = bool(torch.equal(edge_index, aux_edge_index))
    clean_edge_reference, _ = model.atomic_point_encoder._edges(
        output["clean_coords"], batch.atomic_point_cloud.batch.long().to(device)
    )
    clean_leakage_absent = bool(not torch.equal(edge_index, clean_edge_reference))
    point_stats = output["atomic_aux"]
    model_state_after_probe = {
        "graph_encoder": _module_digest(model.graph_encoder),
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "fusion": _module_digest(model.fusion),
    }
    gates = {
        "JOINT_REFERENCE_MAE_PC_GRADIENT_NONZERO": bool(full_mae_report["all_finite"] and full_mae_report["any_nonzero"]),
        "CCDD_TO_PC_GRADIENT_NONZERO": bool(ccdd_report["all_finite"] and ccdd_report["any_nonzero"]),
        "CCDD_TO_HEAD_GRADIENT_NONZERO": bool(ccdd_head_report["all_finite"] and ccdd_head_report["any_nonzero"]),
        "MAE_TO_PC_GRADIENT_ZERO": bool(blocked_report["all_finite"] and blocked_report["exact_zero"] and blocked_report["max_abs"] == 0.0),
        "FROZEN_O8_NO_GRADIENT": frozen_grad_nonempty["o8"] is False,
        "FROZEN_KFUSE_NO_GRADIENT": frozen_grad_nonempty["kfuse"] is False,
        "FROZEN_MAE_HEAD_NO_GRADIENT": frozen_grad_nonempty["atom_head"] is False,
        "NOISY_KNN": bool(edge_index.ndim == 2 and edge_index.size(0) == 2 and edge_index.size(1) > 0 and int(point_stats.get("point_count_used_for_message_passing", -1)) == int(point_stats.get("point_count", -2)) and noisy_edge_contract),
        "CLEAN_KNN_LEAKAGE_NO": clean_leakage_absent,
        "CENTER_RU_CONTRACT": bool(
            model.atomic_point_encoder.pool_scope == "center"
            and model.atomic_point_encoder.num_layers == 4
            and model.atomic_point_encoder.k_neighbors == 24
            and int(point_stats.get("point_count_used_for_pooling", -1)) == int(point_stats.get("central_point_count", -2))
        ),
        "FINITE_LOSSES": bool(torch.isfinite(output["ccdd_loss"]).all() and torch.isfinite(output["mae_loss"]).all()),
    }
    payload = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-audit-v1",
        "device": str(device),
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "all_pass": bool(all(gates.values())),
        "objective": "L_PC = L_CCDD",
        "full_joint_reference": {
            "o8_initialization": full_o8_source,
            "mae_to_atomic_pc": full_mae_report,
            "interpretation": "non-zero is expected in the old joint graph and is the contamination path removed by this route",
        },
        "ccdd_only": {
            "freeze_contract": freeze_meta,
            "ccdd_to_atomic_pc": ccdd_report,
            "ccdd_to_head": ccdd_head_report,
            "mae_to_atomic_pc_after_explicit_detach": blocked_report,
            "post_backward_atomic_pc": post_backward_atomic,
            "post_backward_ccdd_head": post_backward_ccdd_head,
            "frozen_grad_nonempty": frozen_grad_nonempty,
            "model_state_after_probe": model_state_after_probe,
        },
        "forward_contract": {
            "mae_loss": float(output["mae_loss"].detach().float().cpu()),
            "ccdd_loss": float(output["ccdd_loss"].detach().float().cpu()),
            "ccdd_edge_count": int(output["ccdd_edge_count"]),
            "same_noisy_edge_index": noisy_edge_contract,
            "same_as_clean_edge_reference": not clean_leakage_absent,
            "point_stats": {key: value for key, value in point_stats.items() if not torch.is_tensor(value)},
        },
    }
    _write(RESULT_ROOT / "gradient_audit.json", payload)
    return payload


def _preflight() -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    contract = _load_prior_contract()
    init = _module_initialization_audit()
    gradient = _gradient_audit({
        **joint_route._load_subset_info(),
        "coverage": contract["artifacts"]["coverage"],
    })
    base_source = contract["artifacts"]["source_audit"]
    base_pooling = contract["artifacts"]["pooling"]
    base_kfuse = contract["artifacts"]["kfuse"]
    base_cohort = contract["artifacts"]["cohort"]
    source_payload = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-source-audit-v1",
        "current_source_sha256": {
            "joint_pretrainer": _sha(ROOT / "src/training/pretrain/original_mips_atomic_pc_joint.py"),
            "joint_runner": _sha(ROOT / "src/training/w_camr_v2_support/cohort_runtime.py"),
            "atomic_point_encoder": _sha(ROOT / "src/modules/atomic_point_encoder.py"),
            "joint_collator": _sha(ROOT / "src/dataset/original_mips_atomic_pc_joint.py"),
        },
        "ccdd_implementation": {
            "class": "src.training.pretrain.original_mips_atomic_pc_joint.CCDDDistanceHead",
            "forward_reused": "OriginalMIPSAtomicPCJointPretrainer.forward",
            "edge_feature_width": 1024,
            "temporary_head_only": True,
        },
    }
    _write(RESULT_ROOT / "source_audit.json", source_payload)
    checks = {
        "SAME_50K_SUBSET": contract["subset_key_hash"] == contract["ordered_sample_key_hash"] and contract["md_key_count"] == EXPECTED_SUBSET_COUNT,
        "SAME_GEOMETRY": contract["artifacts"]["coverage"].get("reuse_existing_geometry") is True and contract["artifacts"]["coverage"].get("generation_attempted") is False,
        "SAME_ATOMIC_PC_INIT": init["checks"]["ATOMIC_PC_CONSTRUCTION_MATCH"] == "PASS" and init["checks"]["SEED42_DETERMINISTIC_RECONSTRUCTION"] == "PASS",
        "SAME_CCDD_IMPLEMENTATION": source_payload["ccdd_implementation"]["forward_reused"] == "OriginalMIPSAtomicPCJointPretrainer.forward" and source_payload["ccdd_implementation"]["edge_feature_width"] == 1024,
        "MAE_TO_PC_GRADIENT_ZERO": gradient["gates"]["MAE_TO_PC_GRADIENT_ZERO"] == "PASS",
        "CCDD_TO_PC_GRADIENT_NONZERO": gradient["gates"]["CCDD_TO_PC_GRADIENT_NONZERO"] == "PASS",
        "CENTER_RU_CONTRACT": gradient["gates"]["CENTER_RU_CONTRACT"] == "PASS" and base_pooling.get("gate", {}).get("CENTER_ONLY_POOLING") == "PASS",
        "NOISY_KNN": gradient["gates"]["NOISY_KNN"] == "PASS",
        "CLEAN_KNN_LEAKAGE": gradient["gates"]["CLEAN_KNN_LEAKAGE_NO"] == "PASS",
        "MD200_PATH_UNCHANGED": base_source.get("original_mips_md200_unchanged") is True and contract["md_protocol"] == OriginalMIPSMD200.protocol,
        "KFUSE_MATH_UNCHANGED": base_source.get("original_mips_kfuse_unchanged") is True and base_kfuse.get("gate", {}).get("ORIGINAL_KFUSE_PARITY") == "PASS",
        "SAMPLE_COHORT_UNCHANGED": base_cohort.get("status") == "PASS" and base_cohort.get("old_A0_sample_set_equals_new_center_A0_sample_set") is True,
        "DOWNSTREAM_PROTOCOL_UNCHANGED": TASKS == ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc") and FOLDS == (0, 1, 2, 3, 4),
        "BASE_REFERENCE_UNCHANGED": math.isclose(contract["base_macro_r2"], REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "OLD_T2_REFERENCE_UNCHANGED": math.isclose(contract["old_t2_macro_r2"], REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "FULL_JOINT_REFERENCE_UNCHANGED": math.isclose(contract["full_joint_macro_r2"], REFERENCE_FULL_JOINT_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "NO_GEOMETRY_GENERATION": True,
        "NO_BASE_RERUN": True,
        "NO_FULL_JOINT_RERUN": True,
        "NO_100K_OR_1M": True,
        "NO_JOINT_V2": True,
    }
    gate = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-prestart-gate-v1",
        "gates": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "all_pass": bool(all(checks.values())),
        "objective": "L_PC = L_CCDD",
        "references": {
            "base_macro_r2": REFERENCE_BASE_MACRO_R2,
            "old_t2_joint_pretrained_pc_macro_r2": REFERENCE_OLD_T2_MACRO_R2,
            "full_joint_macro_r2": REFERENCE_FULL_JOINT_MACRO_R2,
        },
        "matched_contract": {
            "subset_count": EXPECTED_SUBSET_COUNT,
            "eligible_count": EXPECTED_ELIGIBLE_COUNT,
            "subset_key_hash": contract["subset_key_hash"],
            "geometry_protocol": contract["artifacts"]["coverage"].get("geometry_protocol"),
            "geometry_source": contract["artifacts"]["coverage"].get("geometry_source"),
            "same_frozen_geometry": True,
            "atomic_initialization": init,
            "sigma": COORD_NOISE_SIGMA,
            "knn": 24,
            "layers": 4,
            "pool_scope": "center",
            "pretrain_steps": EXPECTED_PRETRAIN_STEPS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "optimizer": {"name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, "warmup_steps": 2000},
            "scheduler": "joint_route._make_optimizer LambdaLR (same 1504-step warmup schedule)",
        },
        "gradient_audit": str(RESULT_ROOT / "gradient_audit.json"),
        "source_audit": str(RESULT_ROOT / "source_audit.json"),
        "forbidden_actions": {
            "model_source_modification": "NO",
            "loss_sweep": "NO",
            "geometry_generation": "NO",
            "geometry_retry": "NO",
            "base_rerun": "NO",
            "full_joint_rerun": "NO",
            "100K_or_1M": "NO",
            "joint_v2": "NO",
        },
    }
    _write(RESULT_ROOT / "prestart_gate.json", gate)
    _write(RESULT_ROOT / "initialization_audit.json", init)
    print(json.dumps({"all_pass": gate["all_pass"], "gates": gate["gates"]}, ensure_ascii=False, indent=2))
    if not gate["all_pass"]:
        raise SystemExit("CCDD-only attribution prestart gate failed; pretraining was not started")
    print("CCDD-only attribution prestart gate PASS; no training started")
    return gate


def _train() -> dict[str, Any]:
    gate_path = RESULT_ROOT / "prestart_gate.json"
    if not gate_path.is_file() or not bool(_json(gate_path).get("all_pass")):
        raise SystemExit("CCDD-only attribution prestart gate is not PASS")
    checkpoint_path = PRETRAIN_ROOT / "ccdd_only_checkpoint.pt"
    if checkpoint_path.is_file():
        raise RuntimeError(f"refusing to overwrite existing CCDD-only checkpoint: {checkpoint_path}")
    info = joint_route._load_subset_info()
    coverage = _json(JOINT_PRETRAIN_ROOT / "50k_geometry_coverage.json")
    if len(info["eligible_indices"]) != EXPECTED_ELIGIBLE_COUNT or info["manifest"].get("sample_key_hash") != coverage.get("ordered_sample_key_hash"):
        raise RuntimeError("persisted CCDD-only subset contract mismatch")
    dataset = joint_route._build_dataset()
    eligible = list(info["eligible_indices"])
    collator = OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader_kwargs: dict[str, Any] = {
        "dataset": Subset(dataset, eligible),
        "batch_size": EXPECTED_BATCH_SIZE,
        "shuffle": True,
        "drop_last": False,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    if len(loader) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError(f"eligible loader has {len(loader)} batches, expected {EXPECTED_PRETRAIN_STEPS}")
    total_steps = len(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the prior route's order: loader construction precedes seed/model.
    set_global_seed(42)
    model, freeze_meta = _new_ccdd_only_model(device)
    optimizer, scheduler = joint_route._make_optimizer(model, total_steps)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    atomic_parameters = list(model.atomic_point_encoder.parameters())
    ccdd_head_parameters = list(model.ccdd_head.parameters())
    initial_hashes = {
        "graph_encoder": _module_digest(model.graph_encoder),
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "fusion": _module_digest(model.fusion),
    }
    metrics_path = PRETRAIN_ROOT / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    PRETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    iterator = iter(loader)
    model.train()
    for step in range(1, total_steps + 1):
        batch = next(iterator)
        step_started = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with joint_route._autocast(device):
            output = model(
                batch,
                seed=42,
                stream_step=step,
                atom_mask_rate=ATOM_MASK_RATE,
                md_kvec_mask_rate=MD_KVEC_MASK_RATE,
                coord_noise_sigma=COORD_NOISE_SIGMA,
            )
        loss = output["ccdd_loss"]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"non-finite CCDD-only loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite CCDD-only gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        step_seconds = time.perf_counter() - step_started
        atomic_grad = gradient_summary(atomic_parameters)
        ccdd_head_grad = gradient_summary(ccdd_head_parameters)
        forbidden_grad = {
            "o8": any(parameter.grad is not None for parameter in model.graph_encoder.parameters()),
            "kfuse": any(parameter.grad is not None for parameter in model.fusion.parameters()),
            "atom_head": any(parameter.grad is not None for parameter in model.atom_head.parameters()),
        }
        row = {
            "step": int(step),
            "objective": "L_PC = L_CCDD",
            "ccdd_loss": float(loss.detach().float().cpu()),
            "mae_loss_observed_not_optimized": float(output["mae_loss"].detach().float().cpu()),
            "distance_mae": float((output["ccdd_pred"] - output["ccdd_target"]).abs().mean().detach().float().cpu()) if output["ccdd_pred"].numel() else 0.0,
            "ccdd_edge_count": int(output["ccdd_edge_count"]),
            "atomic_pc_grad_norm": atomic_grad["l2_norm"],
            "ccdd_head_grad_norm": ccdd_head_grad["l2_norm"],
            "forbidden_grad_nonempty": forbidden_grad,
            "clipped_grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "step_seconds": float(step_seconds),
            "throughput_graphs_per_second": float(batch.graph_available.numel() / max(step_seconds, 1e-9)),
            "geometry_loading_seconds": float(getattr(batch, "geometry_loading_seconds", 0.0)),
            "gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        if any(forbidden_grad.values()):
            raise RuntimeError(f"forbidden module received gradient at step {step}: {forbidden_grad}")
        records.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step == 1 or step % 50 == 0 or step == total_steps:
            print(
                f"[ccdd-only-pretrain] step={step}/{total_steps} ccdd={row['ccdd_loss']:.6f} "
                f"mae_observed={row['mae_loss_observed_not_optimized']:.6f} "
                f"throughput={row['throughput_graphs_per_second']:.2f}",
                flush=True,
            )
    final = records[-1]
    final_hashes = {
        "graph_encoder": _module_digest(model.graph_encoder),
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "fusion": _module_digest(model.fusion),
    }
    if final_hashes["graph_encoder"] != initial_hashes["graph_encoder"] or final_hashes["fusion"] != initial_hashes["fusion"]:
        raise RuntimeError("frozen O8/KFuse state changed during CCDD-only pretraining")
    model_state: dict[str, torch.Tensor] = {}
    for name, value in model.atomic_point_encoder.state_dict().items():
        model_state[f"atomic_point_encoder.{name}"] = value.detach().cpu().clone()
    for name, value in model.ccdd_head.state_dict().items():
        model_state[f"ccdd_head.{name}"] = value.detach().cpu().clone()
    checkpoint = {
        "schema": CCDD_CHECKPOINT_SCHEMA,
        "architecture": OriginalMIPSAtomicPCJointPretrainer.architecture_name,
        "pretrain_schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-v1",
        "step": int(total_steps),
        "subset_count": EXPECTED_SUBSET_COUNT,
        "training_eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "model_state": model_state,
        "metadata": {
            "pretrained_components": ["atomic_point_encoder"],
            "temporary_training_head": "ccdd_head",
            "excluded_components": ["graph_encoder", "fusion", "atom_head"],
            "objective": "L_PC = L_CCDD",
            "mae_weight": 0.0,
            "ccdd_weight": 1.0,
            "o8_training": False,
            "kfuse_training": False,
            "md200_training": False,
            "atom_mask_rate": ATOM_MASK_RATE,
            "md_kvec_mask_rate": MD_KVEC_MASK_RATE,
            "coordinate_noise_sigma": COORD_NOISE_SIGMA,
            "atomic_knn": 24,
            "atomic_layers": 4,
            "atomic_pool_scope": "center",
            "subset_sample_key_hash": coverage.get("ordered_sample_key_hash"),
            "geometry_protocol": coverage.get("geometry_protocol"),
            "geometry_source": coverage.get("geometry_source"),
            "optimizer": {"name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, "warmup_steps": 2000},
            "scheduler": "joint_route._make_optimizer LambdaLR",
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
            "pretraining_heads_saved": ["ccdd_head"],
            "downstream_heads_loaded": [],
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-summary-v1",
        "status": "COMPLETE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "subset_count": EXPECTED_SUBSET_COUNT,
        "training_eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "optimizer_steps": int(total_steps),
        "objective": "L_PC = L_CCDD",
        "final_ccdd_loss": final["ccdd_loss"],
        "final_mae_loss_observed_not_optimized": final["mae_loss_observed_not_optimized"],
        "final_distance_mae": final["distance_mae"],
        "pretraining_wall_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_allocated_bytes": int(max(row["gpu_memory_allocated_bytes"] for row in records)),
        "peak_gpu_memory_reserved_bytes": int(max(row["gpu_memory_reserved_bytes"] for row in records)),
        "o8_pretrained": False,
        "atomic_pc_pretrained": True,
        "kfuse_pretrained": False,
        "md200_trained": False,
        "temporary_training_head": "ccdd_head",
        "training_heads_removed_downstream": True,
        "reference_base_macro_r2": REFERENCE_BASE_MACRO_R2,
        "reference_old_t2_macro_r2": REFERENCE_OLD_T2_MACRO_R2,
        "metrics_path": str(metrics_path),
        "initial_hashes": initial_hashes,
        "final_hashes": final_hashes,
        "freeze_contract": freeze_meta,
    }
    _write(PRETRAIN_ROOT / "pretraining_summary.json", summary)
    _write(PRETRAIN_ROOT / "parameter_source_table.json", {
        "schema": "original-mips-atomic-pc-ccdd-only-parameter-source-table-v1",
        "rows": [
            {"module": "graph_encoder", "source": "base O8 checkpoint; frozen", **_module_summary(model.graph_encoder)},
            {"module": "atomic_point_encoder", "source": "same fresh Base/joint construction; CCDD-only trained", **_module_summary(model.atomic_point_encoder)},
            {"module": "fusion", "source": "fresh Base/KFuse construction; frozen", **_module_summary(model.fusion)},
            {"module": "atom_head", "source": "fresh temporary MAE head; frozen and excluded", **_module_summary(model.atom_head)},
            {"module": "ccdd_head", "source": "fresh temporary CCDD head; trained", **_module_summary(model.ccdd_head)},
        ],
        "joint_checkpoint_loaded": False,
        "unexpected_joint_weight_count": 0,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_ccdd_checkpoint() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = PRETRAIN_ROOT / "ccdd_only_checkpoint.pt"
    if not path.is_file():
        raise RuntimeError(f"missing CCDD-only checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CCDD_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected CCDD-only checkpoint schema")
    if int(checkpoint.get("step", -1)) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("CCDD-only checkpoint is not the required 1504-step artifact")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("CCDD-only checkpoint has no model_state")
    allowed = ("atomic_point_encoder.", "ccdd_head.")
    unknown = sorted(name for name in state if not name.startswith(allowed))
    if unknown:
        raise RuntimeError(f"CCDD-only checkpoint contains unexpected component weights: {unknown[:5]}")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("pretrained_components") != ["atomic_point_encoder"] or metadata.get("temporary_training_head") != "ccdd_head":
        raise RuntimeError("CCDD-only checkpoint component metadata is not exact")
    if metadata.get("objective") != "L_PC = L_CCDD" or metadata.get("atomic_pool_scope") != "center":
        raise RuntimeError("CCDD-only checkpoint objective/pooling metadata mismatch")
    if metadata.get("subset_sample_key_hash") != _json(JOINT_PRETRAIN_ROOT / "50k_geometry_coverage.json").get("ordered_sample_key_hash"):
        raise RuntimeError("CCDD-only checkpoint subset hash mismatch")
    return checkpoint, state


_BASE_CENTER_MODEL = downstream_route._center_model


def _ccdd_downstream_model() -> tuple[nn.Module, dict[str, Any]]:
    checkpoint, state = _load_ccdd_checkpoint()
    model, base_meta = _BASE_CENTER_MODEL()
    target = model.state_dict()
    prefix = "atomic_point_encoder."
    target_names = {name for name in target if name.startswith(prefix)}
    source_names = {name for name in state if name.startswith(prefix)}
    if target_names != source_names:
        raise RuntimeError("CCDD-only Atomic-PC tensor set does not match Base target")
    base_hashes = {name: _module_digest(getattr(model, name)) for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")}
    loaded_names = []
    for name in sorted(target_names):
        source = state[name]
        if tuple(source.shape) != tuple(target[name].shape):
            raise RuntimeError(f"CCDD-only tensor shape mismatch for {name}")
        target[name] = source.detach().clone()
        loaded_names.append(name)
    model.load_state_dict(target, strict=True)
    observed_hashes = {name: _module_digest(getattr(model, name)) for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")}
    if any(observed_hashes[name] != base_hashes[name] for name in observed_hashes if name != "atomic_point_encoder"):
        raise RuntimeError("a non-Atomic-PC Base module changed during CCDD-only transplant")
    atomic_hash = _state_prefix_digest(state, prefix)
    if observed_hashes["atomic_point_encoder"] != atomic_hash:
        raise RuntimeError("post-load CCDD-only Atomic-PC hash mismatch")
    source_table = []
    for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp"):
        module = getattr(model, name)
        summary = _module_summary(module)
        source_table.append({
            "module": name,
            "source": "ccdd_only_checkpoint" if name == "atomic_point_encoder" else "base_no_pretrain_initialization",
            "joint_weights_loaded": False,
            **summary,
        })
    return model, {
        "path": str(PRETRAIN_ROOT / "ccdd_only_checkpoint.pt"),
        "sha256": _sha(PRETRAIN_ROOT / "ccdd_only_checkpoint.pt"),
        "checkpoint_schema": checkpoint.get("schema"),
        "pretrain_step": int(checkpoint.get("step", -1)),
        "loaded_components": ["atomic_point_encoder"],
        "loaded_tensor_count": len(loaded_names),
        "loaded_tensor_names": sorted(loaded_names),
        "removed_pretraining_heads": ["ccdd_head"],
        "base_loader": base_meta,
        "source_table": source_table,
        "nonselected_modules_equal_base": True,
        "unexpected_joint_weight_count": 0,
        "atomic_pool_scope": "center",
        "active_modalities": ["md", "atomic_pc"],
        "AP3D": False,
        "MCP": False,
    }


def _configure_downstream() -> None:
    downstream_route.TRAIN_ROOT = DOWNSTREAM_ROOT
    downstream_route.SHARD_ROOT = SHARD_ROOT
    downstream_route._center_model = _ccdd_downstream_model


def _run_downstream(tasks: list[str], folds: list[int]) -> None:
    gate = _json(RESULT_ROOT / "prestart_gate.json")
    if not bool(gate.get("all_pass")):
        raise SystemExit("CCDD-only prestart gate is not PASS")
    _load_ccdd_checkpoint()
    _configure_downstream()
    # These functions resolve their immutable original-module globals, so the
    # call occurs before any output-root rebinding side effects matter.
    success_keys = downstream_route.geometry_success_keys()
    md_values, _ = downstream_route.load_md_table()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in tasks:
        for fold in folds:
            shard = SHARD_ROOT / f"{task}_fold{fold}.json"
            if shard.is_file():
                continue
            result = downstream_route._fold_run(task, int(fold), md_values, success_keys, device)
            print(json.dumps({"task": task, "fold": fold, "test_r2": result["test_r2"], "best_epoch": result["best_epoch"]}, ensure_ascii=False), flush=True)


def _aggregate() -> dict[str, Any]:
    _configure_downstream()
    per_task, macro = downstream_route._aggregate()
    rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    source_rows = []
    for row in rows:
        checkpoint = row.get("checkpoint", {})
        source_rows.extend(checkpoint.get("source_table", []))
    _write(DOWNSTREAM_ROOT / "aggregate_summary.json", {
        "schema": "original-mips-atomic-pc-ccdd-only-gradient-attribution-downstream-aggregate-v1",
        "row_count": len(rows),
        "task_count": len(per_task),
        "macro": macro,
        "source_table_row_count": len(source_rows),
        "output_root": str(DOWNSTREAM_ROOT),
    })
    return {"per_task": per_task, "macro": macro, "row_count": len(rows)}


def _finalize() -> dict[str, Any]:
    per_task_payload = _json(DOWNSTREAM_ROOT / "per_task_metrics.json")
    ccdd_fold_rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    base_task_payload = _json(BASE_ROOT / "training/per_task_metrics.json")
    old_task_payload = _json(OLD_T2_ROOT / "per_task_metrics.json")
    old_fold_rows = _json(OLD_T2_ROOT / "per_fold_metrics.json")
    ccdd_macro = float(per_task_payload["macro"]["macro_test_r2"])
    base_macro = float(base_task_payload["macro"]["macro_test_r2"])
    old_macro = float(old_task_payload["macro"]["macro_test_r2"])
    if not math.isclose(base_macro, REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(old_macro, REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("reference macro changed while finalizing")
    base_task = base_task_payload["per_task"]
    old_task = old_task_payload["per_task"]
    ccdd_task = per_task_payload["per_task"]
    per_task = {}
    for task in TASKS:
        ccdd_r2 = float(ccdd_task[task]["test_r2_mean"])
        base_r2 = float(base_task[task]["test_r2_mean"])
        old_r2 = float(old_task[task]["test_r2_mean"])
        per_task[task] = {
            "ccdd_only_test_r2_mean": ccdd_r2,
            "base_test_r2_mean": base_r2,
            "old_t2_test_r2_mean": old_r2,
            "delta_vs_base": ccdd_r2 - base_r2,
            "delta_vs_old_t2": ccdd_r2 - old_r2,
            "positive_fold_count_vs_base": 0,
            "positive_fold_count_vs_old_t2": 0,
            "fold_count": 5,
        }
    old_by_key = {(row["task"], int(row["fold"])): row for row in old_fold_rows}
    base_by_key = {(row["task"], int(row["fold"])): row for row in _json(BASE_ROOT / "training/per_fold_metrics.json")}
    fold_rows = []
    for row in ccdd_fold_rows:
        key = (row["task"], int(row["fold"]))
        ccdd_r2 = float(row["test_r2"])
        base_r2 = float(base_by_key[key]["test_r2"])
        old_r2 = float(old_by_key[key]["test_r2"])
        delta_base = ccdd_r2 - base_r2
        delta_old = ccdd_r2 - old_r2
        fold_rows.append({"task": key[0], "fold": key[1], "ccdd_only_test_r2": ccdd_r2, "base_test_r2": base_r2, "old_t2_test_r2": old_r2, "delta_vs_base": delta_base, "delta_vs_old_t2": delta_old})
        if delta_base > 0:
            per_task[key[0]]["positive_fold_count_vs_base"] += 1
        if delta_old > 0:
            per_task[key[0]]["positive_fold_count_vs_old_t2"] += 1
    positive_tasks = [task for task in TASKS if per_task[task]["delta_vs_base"] > 0]
    positive_folds = sum(1 for row in fold_rows if row["delta_vs_base"] > 0)
    if ccdd_macro >= base_macro:
        transfer = "POSITIVE"
        interference = "SUPPORTED"
        objective_problem = "NOT_SUPPORTED"
    elif old_macro < ccdd_macro < base_macro:
        transfer = "WEAK_NEGATIVE"
        interference = "SUPPORTED"
        objective_problem = "NOT_SUPPORTED"
    else:
        transfer = "NEGATIVE"
        interference = "NOT_SUPPORTED"
        objective_problem = "SUPPORTED"
    matrix = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-diagnosis-matrix-v1",
        "references": {"base_macro_r2": base_macro, "old_t2_joint_pretrained_pc_macro_r2": old_macro},
        "ccdd_only_pc_macro_r2": ccdd_macro,
        "delta_vs_base": ccdd_macro - base_macro,
        "delta_vs_old_t2": ccdd_macro - old_macro,
        "positive_tasks": len(positive_tasks),
        "positive_tasks_list": positive_tasks,
        "positive_folds": positive_folds,
        "fold_count": len(fold_rows),
        "per_task": per_task,
        "per_fold": fold_rows,
        "xc": per_task["xc"],
        "judgment": {
            "CCDD_TRANSFER": transfer,
            "MAE_TO_PC_INTERFERENCE": interference,
            "CCDD_OBJECTIVE_PROBLEM": objective_problem,
        },
    }
    _write(RESULT_ROOT / "diagnosis_matrix.json", matrix)
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-summary-v1",
        "status": "COMPLETE",
        "pretraining": _json(PRETRAIN_ROOT / "pretraining_summary.json"),
        "downstream": {"macro": per_task_payload["macro"], "row_count": len(ccdd_fold_rows), "protocol": "historical_shared5", "seed": 42, "tasks": list(TASKS), "folds": list(FOLDS)},
        "diagnosis": matrix,
        "ready_for_next_review": "YES",
        "stop": "YES",
        "automatic_joint_v2": "NO",
    }
    _write(RESULT_ROOT / "summary.json", summary)
    packet_json = {
        "schema": "original-mips-atomic-pc-center-ru-ccdd-only-gradient-attribution-return-packet-v1",
        "status": "COMPLETE",
        "CCDD_ONLY_PC_MACRO_R2": ccdd_macro,
        "DELTA_VS_BASE": ccdd_macro - base_macro,
        "DELTA_VS_OLD_T2": ccdd_macro - old_macro,
        "positive_tasks_over_8": len(positive_tasks),
        "positive_folds_over_40": positive_folds,
        "xc_delta_vs_base": per_task["xc"]["delta_vs_base"],
        "judgment": matrix["judgment"],
        "READY_FOR_NEXT_REVIEW": "YES",
        "STOP": "YES",
    }
    _write(RESULT_ROOT / "RETURN_PACKET.json", packet_json)
    md_lines = [
        "# Atomic-PC CCDD-only Gradient Attribution",
        "",
        "## Status",
        "",
        "- `PRESTART_GATE = PASS`; gradient and matched-input gates passed.",
        "- `READY_FOR_NEXT_REVIEW = YES`.",
        "- `STOP = YES`; no Joint-v2, larger pretraining, geometry generation, Base rerun, or Full-Joint rerun.",
        "",
        "## Results",
        "",
        f"- BASE = `{base_macro:.15f}`",
        f"- OLD_T2_JOINT_PRETRAINED_PC = `{old_macro:.15f}`",
        f"- CCDD_ONLY_PC_MACRO_R2 = `{ccdd_macro:.15f}`",
        f"- DELTA_VS_BASE = `{ccdd_macro - base_macro:.15f}`",
        f"- DELTA_VS_OLD_T2 = `{ccdd_macro - old_macro:.15f}`",
        f"- positive tasks / 8 = `{len(positive_tasks)}/8`; positive folds / 40 = `{positive_folds}/40`.",
        f"- xc delta vs Base = `{per_task['xc']['delta_vs_base']:.15f}`.",
        "",
        "## Judgment",
        "",
        f"- `CCDD_TRANSFER = {transfer}`.",
        f"- `MAE_TO_PC_INTERFERENCE = {interference}`.",
        f"- `CCDD_OBJECTIVE_PROBLEM = {objective_problem}`.",
        "",
        "Per-task and paired per-fold values are in `diagnosis_matrix.json`. The downstream protocol is `historical_shared5`; its held-out folds are shared validation/test folds, not an independent blind test.",
        "",
        "## Artifacts",
        "",
        "- `prestart_gate.json`, `initialization_audit.json`, `gradient_audit.json`, `source_audit.json`",
        "- `pretraining/ccdd_only_checkpoint.pt`, `pretraining/pretraining_summary.json`, `pretraining/parameter_source_table.json`",
        "- `downstream/per_task_metrics.json`, `downstream/per_fold_metrics.json`, `diagnosis_matrix.json`, `summary.json`, `RETURN_PACKET.json`",
        "",
        "Final action: STOP.",
    ]
    (RESULT_ROOT / "RETURN_PACKET.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(packet_json, ensure_ascii=False, indent=2))
    return packet_json


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--downstream", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        _preflight()
    elif args.train:
        _train()
    elif args.downstream:
        _run_downstream(list(args.tasks), list(args.folds))
    elif args.aggregate:
        print(json.dumps(_aggregate(), ensure_ascii=False, indent=2))
    elif args.finalize:
        _finalize()


if __name__ == "__main__":
    main()
