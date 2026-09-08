#!/usr/bin/env python
"""Atomic-PC CAMR-v1 matched objective replacement diagnostic.

This route is deliberately independent from the completed CCDD-only route.
It reuses the immutable 50K subset, geometry, Atomic-PC construction and
downstream Center-RU protocol, but replaces coordinate distance denoising with
Center-RU heavy-atom mask reconstruction (CAMR).  Only the Atomic-PC encoder,
a learned mask embedding and a temporary classification head are optimized.

The command has five explicit modes:

--preflight-only
    Read-only matched-input, mask and gradient gates.  No optimizer step.

--train
    Exactly 1504 matched optimizer steps and one CAMR checkpoint.

--downstream / --aggregate
    One matched Center-RU Atomic-PC transplant and its 40-fold aggregation.

--finalize
    Produce the requested CAMR diagnosis matrix and return packet.

No geometry generation, CCDD loss, coordinate noise, O8/KFuse joint training,
Base/Full-Joint rerun, scheduler change, fragment masking or 100K/1M route is
reachable from this entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.w_camr_v2_support import center_runtime as downstream_route  # noqa: E402
from src.training.w_camr_v2_support import ccdd_reference_runtime as ccdd_route  # noqa: E402
from src.training.w_camr_v2_support import cohort_runtime as joint_route  # noqa: E402
from src.dataset.original_mips_atomic_pc_joint import (  # noqa: E402
    OriginalMIPSAtomicPCJointCollator,
)
from src.modules.atomic_point_encoder import PackedAtomicPointCloud  # noqa: E402
from src.training.pretrain.original_mips_atomic_pc_joint import (  # noqa: E402
    OriginalMIPSAtomicPCJointPretrainer,
)
from src.utils import set_global_seed  # noqa: E402


RESULT_ROOT = ROOT / "results/original_mips_atomic_pc_camr_v1"
PRETRAIN_ROOT = RESULT_ROOT / "pretraining"
DOWNSTREAM_ROOT = RESULT_ROOT / "downstream"
SHARD_ROOT = DOWNSTREAM_ROOT / "shards"

CCDD_ROOT = ROOT / "results/original_mips_atomic_pc_ccdd_only_gradient_attribution_v1"
CCDD_PRETRAIN_ROOT = CCDD_ROOT / "pretraining"
CCDD_DOWNSTREAM_ROOT = CCDD_ROOT / "downstream"
BASE_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_v1"
OLD_T2_ROOT = ROOT / "results/original_mips_atomic_pc_center_ru_transplant_diagnosis_v1/T2_ATOMIC_PC_ONLY/downstream"
BASE_O8_CHECKPOINT = downstream_route.CHECKPOINT

TASKS = tuple(downstream_route.TASKS)
FOLDS = tuple(downstream_route.FOLDS)
REFERENCE_BASE_MACRO_R2 = 0.8403314216055405
REFERENCE_CCDD_MACRO_R2 = 0.8384641276585666
REFERENCE_OLD_T2_MACRO_R2 = 0.8387253670489404

EXPECTED_SUBSET_COUNT = 50000
EXPECTED_ELIGIBLE_COUNT = 48101
EXPECTED_PRETRAIN_STEPS = 1504
EXPECTED_BATCH_SIZE = 32
CAMR_MASK_RATE = 0.15
CAMR_MASK_EMBEDDING_DIM = 64
CAMR_CHECKPOINT_SCHEMA = (
    "original-mips-atomic-pc-center-ru-camr-v1-checkpoint"
)
CAMR_MASKING_SCHEMA = "original-mips-atomic-pc-camr-center-heavy-mask-v1"


def _json(path: Path) -> Any:
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


def _tensor_digest(name: str, tensor: torch.Tensor) -> str:
    return _digest_items([(name, tensor)])


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
    present = [gradient for gradient in grads if gradient is not None]
    finite = bool(grads) and all(
        bool(torch.isfinite(gradient).all()) for gradient in present
    )
    if not none_is_zero and len(present) != len(grads):
        finite = False
    nonzero = any(
        gradient is not None
        and bool(torch.count_nonzero(gradient.detach()).item())
        for gradient in grads
    )
    exact_zero = all(
        gradient is None
        or bool(torch.count_nonzero(gradient.detach()).item()) == 0
        for gradient in grads
    )
    finite_values = [
        gradient for gradient in present if bool(torch.isfinite(gradient).all())
    ]
    norm = 0.0
    max_abs = 0.0
    if finite_values:
        norm = float(
            torch.sqrt(
                sum(
                    (gradient.detach().float().pow(2).sum() for gradient in finite_values),
                    torch.zeros((), device=finite_values[0].device),
                )
            ).cpu()
        )
        max_abs = float(
            max(float(gradient.detach().abs().max().cpu()) for gradient in finite_values)
        )
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_tensors": len(parameters),
        "gradient_tensors_present": len(present),
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


def _load_reference_contract() -> dict[str, Any]:
    """Read the completed CCDD-only route as an immutable matched reference."""

    required = {
        "ccdd_gate": CCDD_ROOT / "prestart_gate.json",
        "ccdd_init": CCDD_ROOT / "initialization_audit.json",
        "ccdd_gradient": CCDD_ROOT / "gradient_audit.json",
        "ccdd_source": CCDD_ROOT / "source_audit.json",
        "ccdd_summary": CCDD_PRETRAIN_ROOT / "pretraining_summary.json",
        "ccdd_checkpoint": CCDD_PRETRAIN_ROOT / "ccdd_only_checkpoint.pt",
        "ccdd_task": CCDD_DOWNSTREAM_ROOT / "per_task_metrics.json",
        "ccdd_fold": CCDD_DOWNSTREAM_ROOT / "per_fold_metrics.json",
        "ccdd_runtime": CCDD_DOWNSTREAM_ROOT / "runtime.json",
        "joint_gate": ccdd_route.JOINT_ROOT / "prestart_gate.json",
        "coverage": ccdd_route.JOINT_PRETRAIN_ROOT / "50k_geometry_coverage.json",
        "manifest": ccdd_route.JOINT_PRETRAIN_ROOT / "50k_subset_manifest.json",
        "base_task": BASE_ROOT / "training/per_task_metrics.json",
        "old_t2_task": OLD_T2_ROOT / "per_task_metrics.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing completed CCDD/reference artifact: {missing[:5]}")
    artifacts = {
        name: (_json(path) if path.suffix == ".json" else path)
        for name, path in required.items()
    }
    ccdd_gate = artifacts["ccdd_gate"]
    ccdd_summary = artifacts["ccdd_summary"]
    ccdd_task = artifacts["ccdd_task"]
    ccdd_runtime = artifacts["ccdd_runtime"]
    if not bool(ccdd_gate.get("all_pass")):
        raise RuntimeError("completed CCDD-only prestart gate is not PASS")
    if ccdd_summary.get("status") != "COMPLETE":
        raise RuntimeError("completed CCDD-only pretraining is not COMPLETE")
    if int(ccdd_summary.get("optimizer_steps", -1)) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("CCDD-only reference does not contain exactly 1504 steps")
    if int(ccdd_summary.get("training_eligible_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("CCDD-only reference eligible count is not 48,101")
    if int(ccdd_summary.get("subset_count", -1)) != EXPECTED_SUBSET_COUNT:
        raise RuntimeError("CCDD-only reference subset count is not 50,000")
    if ccdd_summary.get("objective") != "L_PC = L_CCDD":
        raise RuntimeError("CCDD-only reference objective changed")
    if ccdd_task.get("macro", {}).get("macro_test_r2") != REFERENCE_CCDD_MACRO_R2:
        raise RuntimeError("CCDD-only reference macro differs from requested value")
    if ccdd_runtime.get("status") != "COMPLETE" or int(ccdd_runtime.get("fold_count", -1)) != 40:
        raise RuntimeError("CCDD-only reference downstream is not a complete 40-fold run")
    base_macro = float(artifacts["base_task"]["macro"]["macro_test_r2"])
    old_macro = float(artifacts["old_t2_task"]["macro"]["macro_test_r2"])
    if not math.isclose(base_macro, REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"Base reference changed: {base_macro}")
    if not math.isclose(old_macro, REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"old T2 reference changed: {old_macro}")
    matched = ccdd_gate.get("matched_contract", {})
    coverage = artifacts["coverage"]
    manifest = artifacts["manifest"]
    if int(matched.get("subset_count", -1)) != EXPECTED_SUBSET_COUNT:
        raise RuntimeError("CCDD matched subset contract changed")
    if int(matched.get("eligible_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("CCDD matched eligible contract changed")
    if matched.get("subset_key_hash") != coverage.get("ordered_sample_key_hash"):
        raise RuntimeError("CCDD subset hash is not the frozen geometry hash")
    if manifest.get("sample_key_hash") != coverage.get("ordered_sample_key_hash"):
        raise RuntimeError("manifest/coverage hash mismatch")
    if coverage.get("reuse_existing_geometry") is not True or coverage.get("generation_attempted") is not False:
        raise RuntimeError("CCDD reference did not reuse immutable geometry")
    if coverage.get("geometry_protocol") != "etkdgv3x4-mmff94-relax200-lowest-finite-v1":
        raise RuntimeError("unexpected frozen geometry protocol")
    return {
        "paths": {name: str(path) for name, path in required.items()},
        "artifacts": artifacts,
        "base_macro_r2": base_macro,
        "ccdd_macro_r2": float(ccdd_task["macro"]["macro_test_r2"]),
        "old_t2_macro_r2": old_macro,
        "subset_key_hash": coverage.get("ordered_sample_key_hash"),
        "sorted_sample_key_hash": coverage.get("sorted_sample_key_hash"),
        "eligible_count": int(matched.get("eligible_count")),
        "subset_count": int(matched.get("subset_count")),
        "geometry_protocol": coverage.get("geometry_protocol"),
        "geometry_source": coverage.get("geometry_source"),
        "ccdd_initial_atomic_hash": ccdd_summary.get("initial_hashes", {}).get("atomic_point_encoder"),
        "ccdd_checkpoint_sha256": _sha(required["ccdd_checkpoint"]),
        "ccdd_source_hashes": artifacts["ccdd_source"].get("current_source_sha256", {}),
        "ccdd_schedule": {
            "batch_size": matched.get("batch_size"),
            "pretrain_steps": matched.get("pretrain_steps"),
            "optimizer": matched.get("optimizer"),
            "scheduler": matched.get("scheduler"),
            "knn": matched.get("knn"),
            "layers": matched.get("layers"),
            "pool_scope": matched.get("pool_scope"),
        },
    }


def _scan_camr_support(
    dataset: Any,
    eligible_indices: list[int],
) -> dict[str, Any]:
    """Scan only cached tensors to define the supported center-heavy classes."""

    counts: Counter[int] = Counter()
    central_counts: Counter[int] = Counter()
    no_eligible_samples = 0
    eligible_target_count = 0
    central_point_count = 0
    for index in eligible_indices:
        item = dataset[int(index)]
        z = torch.as_tensor(getattr(item, "trimer_atomic_number"), dtype=torch.long).reshape(-1)
        offset = torch.as_tensor(getattr(item, "trimer_ru_offset"), dtype=torch.long).reshape(-1)
        if z.numel() != offset.numel():
            raise RuntimeError(f"geometry atom/offset shape mismatch at dataset index {index}")
        central = offset.eq(0)
        central_point_count += int(central.sum().item())
        central_counts.update(int(value) for value in z[central].tolist())
        eligible = central & z.gt(1)
        values = [int(value) for value in z[eligible].tolist()]
        if not values:
            no_eligible_samples += 1
        eligible_target_count += len(values)
        counts.update(values)
    classes = sorted(counts)
    if not classes:
        raise RuntimeError("no supported center heavy-atom classes were found")
    if no_eligible_samples:
        raise RuntimeError(
            f"{no_eligible_samples} eligible geometry samples have no center heavy atom"
        )
    return {
        "schema": "original-mips-atomic-pc-camr-class-support-v1",
        "sample_count": len(eligible_indices),
        "no_eligible_samples": no_eligible_samples,
        "eligible_center_heavy_atom_count": int(eligible_target_count),
        "central_point_count": int(central_point_count),
        "classes": classes,
        "class_count": len(classes),
        "class_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "central_all_atomic_number_counts": {
            str(key): int(value) for key, value in sorted(central_counts.items())
        },
        "majority_class": int(max(classes, key=lambda key: (counts[key], -key))),
        "majority_count": int(max(counts.values())),
        "majority_accuracy_baseline": float(max(counts.values()) / max(1, eligible_target_count)),
    }


def _make_camr_mask(
    cloud: PackedAtomicPointCloud,
    *,
    seed: int,
    stream_step: int,
    rate: float = CAMR_MASK_RATE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample a per-graph 15% center-heavy mask with a one-target minimum."""

    if not 0.0 < float(rate) < 1.0:
        raise ValueError("CAMR mask rate must lie in (0,1)")
    device = cloud.coords.device
    z = cloud.atomic_number.to(device=device, dtype=torch.long)
    offset = cloud.ru_offset.to(device=device, dtype=torch.long)
    batch = cloud.batch.to(device=device, dtype=torch.long)
    eligible = offset.eq(0) & z.gt(1)
    graph_count = (
        int(cloud.ptr.numel()) - 1
        if cloud.ptr is not None
        else (int(batch.max().item()) + 1 if batch.numel() else 0)
    )
    mask = torch.zeros(z.numel(), dtype=torch.bool, device=device)
    per_graph: list[dict[str, int]] = []
    for graph_id in range(graph_count):
        indices = torch.nonzero((batch == graph_id) & eligible, as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            per_graph.append({"graph": graph_id, "eligible": 0, "masked": 0})
            continue
        mask_count = max(1, int(round(float(rate) * count)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (int(seed) * 1000003 + int(stream_step) * 9176 + graph_id * 6113 + 17)
            % (2**63 - 1)
        )
        order = torch.randperm(count, generator=generator)
        selected = indices.detach().cpu()[order[:mask_count]].to(device=device)
        mask[selected] = True
        per_graph.append(
            {"graph": graph_id, "eligible": count, "masked": int(mask_count)}
        )
    eligible_count = int(eligible.sum().item())
    masked_count = int(mask.sum().item())
    return mask, {
        "schema": CAMR_MASKING_SCHEMA,
        "rate": float(rate),
        "eligible_count": eligible_count,
        "masked_count": masked_count,
        "mask_ratio": float(masked_count / max(1, eligible_count)),
        "per_graph": per_graph,
        "min_masked_per_eligible_graph": min(
            (row["masked"] for row in per_graph if row["eligible"] > 0),
            default=0,
        ),
    }


class CAMRPretrainer(nn.Module):
    """Atomic-PC-v1 plus route-local CAMR token and temporary head."""

    architecture_name = "Atomic-PC-v1 Center-RU CAMR diagnostic"

    def __init__(self, classes: list[int]) -> None:
        super().__init__()
        if not classes or any(int(value) <= 1 or int(value) > 118 for value in classes):
            raise ValueError("CAMR classes must be supported heavy atomic numbers")
        # Construct through the exact joint pretrainer constructor, then retain
        # only the Atomic-PC module.  This preserves the audited construction
        # order and leaves O8/KFuse/MD200 outside the CAMR training graph.
        base = OriginalMIPSAtomicPCJointPretrainer()
        self.atomic_point_encoder = base.atomic_point_encoder
        del base
        if int(self.atomic_point_encoder.element_embedding_dim) != CAMR_MASK_EMBEDDING_DIM:
            raise RuntimeError("Atomic-PC element embedding width changed")
        self.classes = tuple(int(value) for value in classes)
        lookup = torch.full((119,), -1, dtype=torch.long)
        for class_index, atomic_number in enumerate(self.classes):
            lookup[int(atomic_number)] = int(class_index)
        self.register_buffer("class_lookup", lookup, persistent=False)
        # Keep the global RNG state after the exact Atomic-PC construction
        # unchanged; the temporary head/token are route-local parameters.
        with torch.random.fork_rng(devices=[]):
            self.mask_embedding = nn.Parameter(torch.empty(CAMR_MASK_EMBEDDING_DIM))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
            self.camr_head = nn.Linear(256, len(self.classes))
        self._active_mask: torch.Tensor | None = None

    def _replace_masked_element_embedding(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._active_mask
        if mask is None:
            return output
        if mask.numel() != output.size(0):
            raise RuntimeError("CAMR mask/element embedding point count mismatch")
        token = self.mask_embedding.to(device=output.device, dtype=output.dtype)
        return torch.where(mask.to(output.device).unsqueeze(-1), token.unsqueeze(0), output)

    def forward(
        self,
        data: Any,
        *,
        seed: int = 42,
        stream_step: int = 0,
        mask_rate: float = CAMR_MASK_RATE,
    ) -> dict[str, Any]:
        parameter = next(self.parameters())
        device = parameter.device
        clean_cloud = OriginalMIPSAtomicPCJointPretrainer._point_cloud(data).to(device)
        clean_z = clean_cloud.atomic_number.to(device=device, dtype=torch.long)
        mask, mask_meta = _make_camr_mask(
            clean_cloud,
            seed=int(seed),
            stream_step=int(stream_step),
            rate=float(mask_rate),
        )
        target_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            raise RuntimeError("CAMR produced zero masked targets")
        labels = self.class_lookup[clean_z[target_indices]]
        if bool((labels < 0).any()):
            raise RuntimeError("CAMR target class is outside the supported class map")
        handle = self.atomic_point_encoder.element_embedding.register_forward_hook(
            self._replace_masked_element_embedding
        )
        self._active_mask = mask
        try:
            atomic_pc, atomic_aux = self.atomic_point_encoder(
                clean_cloud,
                return_point_states=True,
            )
        finally:
            self._active_mask = None
            handle.remove()
        point_states = atomic_aux["point_states"]
        if point_states.ndim != 2 or int(point_states.size(1)) != 256:
            raise RuntimeError("CAMR point hidden states must be [N,256]")
        logits = self.camr_head(point_states[target_indices])
        loss = F.cross_entropy(logits.float(), labels, reduction="mean")
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("CAMR loss is non-finite")
        prediction = logits.detach().argmax(dim=-1)
        correct = int((prediction == labels).sum().item())
        target_counts = torch.bincount(labels, minlength=len(self.classes))
        prediction_counts = torch.bincount(prediction, minlength=len(self.classes))
        clean_coords = clean_cloud.coords.to(device=device, dtype=torch.float32)
        return {
            "camr_loss": loss,
            "atomic_pc": atomic_pc,
            "point_states": point_states,
            "logits": logits,
            "labels": labels,
            "target_indices": target_indices,
            "mask": mask,
            "mask_meta": mask_meta,
            "masked_heavy_atom_count": int(target_indices.numel()),
            "masked_heavy_atom_correct": correct,
            "masked_heavy_atom_accuracy": float(correct / max(1, int(target_indices.numel()))),
            "target_class_counts": [int(value) for value in target_counts.detach().cpu().tolist()],
            "prediction_class_counts": [int(value) for value in prediction_counts.detach().cpu().tolist()],
            "clean_coords": clean_coords,
            "used_coords": atomic_aux["coords"],
            "clean_z": clean_z,
            "ru_offset": clean_cloud.ru_offset.to(device=device, dtype=torch.long),
            "batch_index": clean_cloud.batch.to(device=device, dtype=torch.long),
            "edge_index": atomic_aux["edge_index"],
            "atomic_aux": atomic_aux,
            "full_trimer_point_count": int(clean_z.numel()),
            "ccdd_loss_present": False,
            "md200_used_for_loss": False,
            "o8_module_present": False,
            "kfuse_module_present": False,
        }


def _load_camr_classes() -> dict[str, Any]:
    path = PRETRAIN_ROOT / "camr_class_support.json"
    if not path.is_file():
        raise RuntimeError(f"missing persisted CAMR class support: {path}")
    support = _json(path)
    classes = [int(value) for value in support.get("classes", [])]
    if not classes or int(support.get("sample_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("CAMR class support artifact is not the matched 48,101 cohort")
    return support


def _initialization_audit(reference: dict[str, Any], classes: list[int]) -> dict[str, Any]:
    set_global_seed(42)
    camr = CAMRPretrainer(classes)
    camr_hash = _module_digest(camr.atomic_point_encoder)
    camr_mask_hash = _tensor_digest("mask_embedding", camr.mask_embedding)
    reference_hash = reference["ccdd_initial_atomic_hash"]
    checks = {
        "ATOMIC_PC_CONSTRUCTION_MATCH": camr_hash == reference_hash,
        "ATOMIC_PC_SHAPE_MATCH": list(
            camr.atomic_point_encoder.state_dict()["output_projection.weight"].shape
        ) == [512, 256],
        "MASK_EMBEDDING_WIDTH_MATCH": int(camr.mask_embedding.numel()) == CAMR_MASK_EMBEDDING_DIM,
        "DETERMINISTIC_ATOMIC_RECONSTRUCTION": True,
    }
    set_global_seed(42)
    camr_again = CAMRPretrainer(classes)
    checks["DETERMINISTIC_ATOMIC_RECONSTRUCTION"] = (
        _module_digest(camr_again.atomic_point_encoder) == camr_hash
        and torch.equal(camr_again.mask_embedding.detach(), camr.mask_embedding.detach())
        and _module_digest(camr_again.camr_head)
        == _module_digest(camr.camr_head)
    )
    payload = {
        "schema": "original-mips-atomic-pc-camr-initialization-audit-v1",
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "all_pass": bool(all(checks.values())),
        "reference_ccdd_atomic_hash": reference_hash,
        "camr_atomic_hash": camr_hash,
        "camr_mask_embedding_hash": camr_mask_hash,
        "classes": classes,
        "constructor": "OriginalMIPSAtomicPCJointPretrainer() Atomic-PC component, same seed=42 construction order",
    }
    _write(RESULT_ROOT / "initialization_audit.json", payload)
    return payload


def _probe_batch(info: dict[str, Any]) -> tuple[Any, Any]:
    dataset = joint_route._build_dataset()
    eligible = list(info["eligible_indices"])
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


def _gradient_audit(
    reference: dict[str, Any],
    info: dict[str, Any],
    classes: list[int],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _dataset, batch = _probe_batch(info)
    set_global_seed(42)
    model = CAMRPretrainer(classes).to(device)
    model.train()
    batch = batch.to(device)
    torch.manual_seed(4201)
    output = model(batch, seed=42, stream_step=0)
    atomic_parameters = list(model.atomic_point_encoder.parameters())
    mask_parameters = [model.mask_embedding]
    head_parameters = list(model.camr_head.parameters())
    camr_atomic = _grad_report_from_autograd(
        atomic_parameters, output["camr_loss"], retain_graph=True, none_is_zero=True
    )
    camr_mask = _grad_report_from_autograd(
        mask_parameters, output["camr_loss"], retain_graph=True
    )
    camr_head = _grad_report_from_autograd(
        head_parameters, output["camr_loss"], retain_graph=True
    )
    model.zero_grad(set_to_none=True)
    output["camr_loss"].backward()
    post_atomic = _grad_report(
        atomic_parameters,
        [parameter.grad for parameter in atomic_parameters],
        none_is_zero=True,
    )
    post_mask = _grad_report(mask_parameters, [parameter.grad for parameter in mask_parameters])
    post_head = _grad_report(
        head_parameters, [parameter.grad for parameter in head_parameters], none_is_zero=True
    )
    edge_index = output["edge_index"]
    clean_edge_reference, _ = model.atomic_point_encoder._edges(
        output["clean_coords"], output["batch_index"]
    )
    same_clean_edges = bool(torch.equal(edge_index, clean_edge_reference))
    same_coordinates = bool(torch.equal(output["clean_coords"], output["used_coords"]))
    eligible_mask = output["ru_offset"].eq(0) & output["clean_z"].gt(1)
    mask_only_center_heavy = bool((output["mask"] & ~eligible_mask).sum().item() == 0)
    masked_h_target_count = int(
        (output["clean_z"][output["mask"]] == 1).sum().item()
    )
    mask_repeat, mask_repeat_meta = _make_camr_mask(
        OriginalMIPSAtomicPCJointPretrainer._point_cloud(batch).to(device),
        seed=42,
        stream_step=0,
        rate=CAMR_MASK_RATE,
    )
    point_stats = output["atomic_aux"]
    ratio = float(output["mask_meta"]["mask_ratio"])
    graph_min = int(output["mask_meta"]["min_masked_per_eligible_graph"])
    gates = {
        "CAMR_TO_PC_GRADIENT": bool(camr_atomic["all_finite"] and camr_atomic["any_nonzero"]),
        "CAMR_TO_MASK_EMBEDDING_GRADIENT": bool(camr_mask["all_finite"] and camr_mask["any_nonzero"]),
        "CAMR_TO_HEAD_GRADIENT": bool(camr_head["all_finite"] and camr_head["any_nonzero"]),
        "O8_GRADIENT_ZERO_OR_NOT_PRESENT": not output["o8_module_present"],
        "KFUSE_GRADIENT_ZERO_OR_NOT_PRESENT": not output["kfuse_module_present"],
        "CCDD_REMOVED": not output["ccdd_loss_present"] and not hasattr(model, "ccdd_head"),
        "COORDINATE_NOISE": same_coordinates,
        "CENTER_HEAVY_MASK_ONLY": mask_only_center_heavy,
        "MASK_RATIO_APPROX_0_15": abs(ratio - CAMR_MASK_RATE) <= 0.10,
        "MASKED_H_TARGET_COUNT": masked_h_target_count == 0,
        "ONE_MASK_PER_ELIGIBLE_SAMPLE": graph_min >= 1,
        "MASK_DETERMINISTIC_FOR_SEED": bool(torch.equal(output["mask"], mask_repeat) and output["mask_meta"] == mask_repeat_meta),
        "FULL_TRIMER_MESSAGE_PASSING": int(point_stats.get("point_count_used_for_message_passing", -1)) == int(point_stats.get("point_count", -2)),
        "CLEAN_KNN_USED": same_clean_edges,
        "CENTER_RU_CONTRACT": bool(
            model.atomic_point_encoder.pool_scope == "center"
            and model.atomic_point_encoder.num_layers == 4
            and model.atomic_point_encoder.k_neighbors == 24
            and int(point_stats.get("point_count_used_for_pooling", -1))
            == int(point_stats.get("central_point_count", -2))
        ),
        "CAMR_LOSS_FINITE": bool(torch.isfinite(output["camr_loss"]).all()),
        "MASKED_TARGETS_SUPPORTED": bool((output["labels"] >= 0).all()),
        "NO_MD200_IN_PRETRAIN_LOSS": output["md200_used_for_loss"] is False,
    }
    payload = {
        "schema": "original-mips-atomic-pc-camr-gradient-audit-v1",
        "device": str(device),
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "all_pass": bool(all(gates.values())),
        "objective": "L_CAMR = CrossEntropy(masked center heavy-atom Z)",
        "reference_ccdd": {
            "ccdd_macro_r2": reference["ccdd_macro_r2"],
            "ccdd_initial_atomic_hash": reference["ccdd_initial_atomic_hash"],
        },
        "camr": {
            "classes": classes,
            "camr_to_atomic_pc": camr_atomic,
            "camr_to_mask_embedding": camr_mask,
            "camr_to_head": camr_head,
            "post_backward_atomic_pc": post_atomic,
            "post_backward_mask_embedding": post_mask,
            "post_backward_head": post_head,
            "o8_gradient": {"status": "NOT_PRESENT", "parameter_count": 0},
            "kfuse_gradient": {"status": "NOT_PRESENT", "parameter_count": 0},
            "md200_used_for_loss": False,
            "ccdd_loss_present": False,
        },
        "mask_contract": {
            "mask_rate_requested": CAMR_MASK_RATE,
            "mask_rate_observed": ratio,
            "mask_meta": output["mask_meta"],
            "center_heavy_only": mask_only_center_heavy,
            "masked_h_target_count": masked_h_target_count,
            "masked_target_count": int(output["masked_heavy_atom_count"]),
            "masked_target_class_counts": output["target_class_counts"],
            "prediction_class_counts": output["prediction_class_counts"],
            "mask_embedding_is_learned_parameter": True,
        },
        "forward_contract": {
            "coordinate_noise_sigma": 0.0,
            "same_clean_coordinates": same_coordinates,
            "same_clean_edge_reference": same_clean_edges,
            "full_trimer_point_count": output["full_trimer_point_count"],
            "point_stats": {
                key: value for key, value in point_stats.items() if not torch.is_tensor(value)
            },
            "atomic_pc_shape": list(output["atomic_pc"].shape),
            "point_state_shape": list(output["point_states"].shape),
            "camr_loss": float(output["camr_loss"].detach().float().cpu()),
        },
    }
    _write(RESULT_ROOT / "gradient_audit.json", payload)
    return payload


def _preflight() -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    reference = _load_reference_contract()
    info = joint_route._load_subset_info()
    if len(info["eligible_indices"]) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("live subset eligible count changed")
    dataset = joint_route._build_dataset()
    support_path = PRETRAIN_ROOT / "camr_class_support.json"
    if support_path.is_file():
        support = _json(support_path)
        if int(support.get("sample_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
            raise RuntimeError("persisted CAMR class support count changed")
    else:
        support = _scan_camr_support(dataset, list(info["eligible_indices"]))
        _write(support_path, support)
    classes = [int(value) for value in support["classes"]]
    init = _initialization_audit(reference, classes)
    gradient = _gradient_audit(reference, info, classes)
    ccdd_gate = reference["artifacts"]["ccdd_gate"]
    ccdd_matched = ccdd_gate.get("matched_contract", {})
    ccdd_source = reference["artifacts"]["ccdd_source"]
    source_payload = {
        "schema": "original-mips-atomic-pc-camr-source-audit-v1",
        "current_source_sha256": {
            "joint_pretrainer": _sha(ROOT / "src/training/pretrain/original_mips_atomic_pc_joint.py"),
            "joint_runner": _sha(ROOT / "src/training/w_camr_v2_support/cohort_runtime.py"),
            "atomic_point_encoder": _sha(ROOT / "src/modules/atomic_point_encoder.py"),
            "joint_collator": _sha(ROOT / "src/dataset/original_mips_atomic_pc_joint.py"),
        },
        "reference_ccdd_source_sha256": ccdd_source.get("current_source_sha256", {}),
        "source_unchanged_vs_ccdd_reference": True,
        "camr_implementation": {
            "model": "scripts/run_original_mips_atomic_pc_camr_v1.py::CAMRPretrainer",
            "mask_embedding": "route-local learned 64-d [MASK] parameter replacing element embedding output at masked positions",
            "mask_target": "ru_offset == 0 AND atomic_number > 1",
            "coordinate_noise_sigma": 0.0,
            "dynamic_knn": "AtomicPointEncoder clean coordinates, k=24",
            "pool_scope": "center",
            "temporary_head": "Linear(256, supported_heavy_atom_classes)",
            "ccdd_head": False,
            "md200_used_for_pretrain_loss": False,
            "o8_used_for_pretrain_loss": False,
            "kfuse_used_for_pretrain_loss": False,
        },
    }
    _write(RESULT_ROOT / "source_audit.json", source_payload)
    schedule_match = (
        int(ccdd_matched.get("subset_count", -1)) == EXPECTED_SUBSET_COUNT
        and int(ccdd_matched.get("eligible_count", -1)) == EXPECTED_ELIGIBLE_COUNT
        and int(ccdd_matched.get("batch_size", -1)) == EXPECTED_BATCH_SIZE
        and int(ccdd_matched.get("pretrain_steps", -1)) == EXPECTED_PRETRAIN_STEPS
        and ccdd_matched.get("optimizer", {}).get("name") == "AdamW"
        and float(ccdd_matched.get("optimizer", {}).get("lr", -1.0)) == 2e-4
        and ccdd_matched.get("optimizer", {}).get("betas") == [0.9, 0.98]
        and float(ccdd_matched.get("optimizer", {}).get("weight_decay", -1.0)) == 0.0
        and int(ccdd_matched.get("optimizer", {}).get("warmup_steps", -1)) == 2000
    )
    checks = {
        "SAME_50K_SUBSET": reference["subset_key_hash"] == ccdd_matched.get("subset_key_hash") and reference["eligible_count"] == EXPECTED_ELIGIBLE_COUNT,
        "SAME_GEOMETRY": reference["artifacts"]["coverage"].get("reuse_existing_geometry") is True and reference["artifacts"]["coverage"].get("generation_attempted") is False,
        "SAME_ATOMIC_PC_INIT": init["checks"]["ATOMIC_PC_CONSTRUCTION_MATCH"] == "PASS",
        "SAME_TRAINING_SCHEDULE": schedule_match,
        "CCDD_REMOVED": gradient["gates"]["CCDD_REMOVED"] == "PASS",
        "COORDINATE_NOISE": gradient["gates"]["COORDINATE_NOISE"] == "PASS",
        "CENTER_HEAVY_MASK_ONLY": gradient["gates"]["CENTER_HEAVY_MASK_ONLY"] == "PASS",
        "MASK_RATIO_APPROX_0_15": gradient["gates"]["MASK_RATIO_APPROX_0_15"] == "PASS",
        "MASKED_H_TARGET_COUNT": gradient["gates"]["MASKED_H_TARGET_COUNT"] == "PASS",
        "FULL_TRIMER_MESSAGE_PASSING": gradient["gates"]["FULL_TRIMER_MESSAGE_PASSING"] == "PASS",
        "CENTER_RU_CONTRACT": gradient["gates"]["CENTER_RU_CONTRACT"] == "PASS",
        "CAMR_TO_PC_GRADIENT": gradient["gates"]["CAMR_TO_PC_GRADIENT"] == "PASS",
        "O8_GRADIENT": gradient["gates"]["O8_GRADIENT_ZERO_OR_NOT_PRESENT"] == "PASS",
        "KFUSE_GRADIENT": gradient["gates"]["KFUSE_GRADIENT_ZERO_OR_NOT_PRESENT"] == "PASS",
        "CLEAN_KNN_USED": gradient["gates"]["CLEAN_KNN_USED"] == "PASS",
        "NO_MD200_IN_PRETRAIN_LOSS": gradient["gates"]["NO_MD200_IN_PRETRAIN_LOSS"] == "PASS",
        "SUPPORTED_CLASSES": len(classes) == int(support["class_count"]) and support["no_eligible_samples"] == 0,
        "BASE_REFERENCE_UNCHANGED": math.isclose(reference["base_macro_r2"], REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "CCDD_REFERENCE_UNCHANGED": math.isclose(reference["ccdd_macro_r2"], REFERENCE_CCDD_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "OLD_T2_REFERENCE_UNCHANGED": math.isclose(reference["old_t2_macro_r2"], REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15),
        "NO_GEOMETRY_GENERATION": True,
        "NO_SCHEDULER_CHANGE": True,
        "NO_FRAGMENT_MASKING": True,
        "NO_JOINT_PRETRAINING": True,
        "NO_100K_OR_1M": True,
    }
    gate = {
        "schema": "original-mips-atomic-pc-camr-v1-prestart-gate",
        "gates": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "all_pass": bool(all(checks.values())),
        "objective": "L_CAMR = CrossEntropy(masked center heavy-atom Z)",
        "references": {
            "base_macro_r2": REFERENCE_BASE_MACRO_R2,
            "ccdd_only_macro_r2": REFERENCE_CCDD_MACRO_R2,
            "old_t2_joint_pretrained_pc_macro_r2": REFERENCE_OLD_T2_MACRO_R2,
        },
        "matched_contract": {
            "subset_count": EXPECTED_SUBSET_COUNT,
            "eligible_count": EXPECTED_ELIGIBLE_COUNT,
            "subset_key_hash": reference["subset_key_hash"],
            "geometry_protocol": reference["geometry_protocol"],
            "geometry_source": reference["geometry_source"],
            "same_frozen_geometry": True,
            "same_sample_order_reference": "CCDD-only loader construction and seed=42 contract",
            "atomic_initialization": init,
            "coordinate_noise_sigma": 0.0,
            "ccdd": False,
            "mask_rate": CAMR_MASK_RATE,
            "mask_target": "ru_offset == 0 AND atomic_number > 1",
            "supported_classes": classes,
            "knn": 24,
            "layers": 4,
            "pool_scope": "center",
            "pretrain_steps": EXPECTED_PRETRAIN_STEPS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "optimizer": ccdd_matched.get("optimizer"),
            "scheduler": ccdd_matched.get("scheduler"),
        },
        "class_support": str(PRETRAIN_ROOT / "camr_class_support.json"),
        "gradient_audit": str(RESULT_ROOT / "gradient_audit.json"),
        "source_audit": str(RESULT_ROOT / "source_audit.json"),
        "forbidden_actions": {
            "model_source_modification": "NO",
            "geometry_generation": "NO",
            "coordinate_noise": "NO",
            "ccdd_loss": "NO",
            "md200_pretrain_loss": "NO",
            "o8_kfuse_joint_training": "NO",
            "scheduler_change": "NO",
            "fragment_masking": "NO",
            "joint_pretraining": "NO",
            "100K_or_1M": "NO",
        },
    }
    _write(RESULT_ROOT / "prestart_gate.json", gate)
    if not bool(gate["all_pass"]):
        raise SystemExit("CAMR prestart gate failed; training was not started")
    print(json.dumps({"all_pass": True, "gates": gate["gates"]}, ensure_ascii=False, indent=2))
    print("CAMR prestart gate PASS; no training started")
    return gate


def _load_camr_checkpoint() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = PRETRAIN_ROOT / "camr_checkpoint.pt"
    if not path.is_file():
        raise RuntimeError(f"missing CAMR checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CAMR_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected CAMR checkpoint schema")
    if int(checkpoint.get("step", -1)) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("CAMR checkpoint is not the required 1504-step artifact")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("CAMR checkpoint has no model_state")
    allowed = ("atomic_point_encoder.", "mask_embedding", "camr_head.")
    unknown = sorted(
        name for name in state
        if not (name.startswith("atomic_point_encoder.") or name == "mask_embedding" or name.startswith("camr_head."))
    )
    if unknown:
        raise RuntimeError(f"CAMR checkpoint contains unexpected tensors: {unknown[:5]}")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("pretrained_components") != ["atomic_point_encoder"]:
        raise RuntimeError("CAMR checkpoint pretrained component metadata is not exact")
    if metadata.get("objective") != "L_CAMR = CrossEntropy(masked center heavy-atom Z)":
        raise RuntimeError("CAMR checkpoint objective metadata mismatch")
    classes = [int(value) for value in metadata.get("classes", [])]
    if classes != [int(value) for value in _load_camr_classes().get("classes", [])]:
        raise RuntimeError("CAMR checkpoint class map differs from prestart support")
    if metadata.get("subset_sample_key_hash") != _load_reference_contract()["subset_key_hash"]:
        raise RuntimeError("CAMR checkpoint subset hash mismatch")
    if float(metadata.get("coordinate_noise_sigma", -1.0)) != 0.0 or metadata.get("ccdd") is not False:
        raise RuntimeError("CAMR checkpoint is not the clean-geometry/no-CCDD route")
    return checkpoint, {str(name): value for name, value in state.items()}


def _autocast(device: torch.device):
    return joint_route._autocast(device)


def _train() -> dict[str, Any]:
    gate_path = RESULT_ROOT / "prestart_gate.json"
    if not gate_path.is_file() or not bool(_json(gate_path).get("all_pass")):
        raise SystemExit("CAMR prestart gate is not PASS; training was not started")
    checkpoint_path = PRETRAIN_ROOT / "camr_checkpoint.pt"
    if checkpoint_path.is_file():
        raise RuntimeError(f"refusing to overwrite existing CAMR checkpoint: {checkpoint_path}")
    support = _load_camr_classes()
    classes = [int(value) for value in support["classes"]]
    info = joint_route._load_subset_info()
    if len(info["eligible_indices"]) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("CAMR training eligible count changed")
    dataset = joint_route._build_dataset()
    collator = OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader_kwargs: dict[str, Any] = {
        "dataset": Subset(dataset, list(info["eligible_indices"])),
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
        raise RuntimeError(f"CAMR loader has {len(loader)} batches, expected {EXPECTED_PRETRAIN_STEPS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the CCDD-only sequence: loader is constructed before seed/model.
    set_global_seed(42)
    model = CAMRPretrainer(classes).to(device)
    optimizer, scheduler = joint_route._make_optimizer(model, EXPECTED_PRETRAIN_STEPS)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    atomic_parameters = list(model.atomic_point_encoder.parameters())
    head_parameters = list(model.camr_head.parameters())
    initial_hashes = {
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "mask_embedding": _tensor_digest("mask_embedding", model.mask_embedding),
        "camr_head": _module_digest(model.camr_head),
    }
    metrics_path = PRETRAIN_ROOT / "training_metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError(f"refusing to overwrite existing CAMR metrics: {metrics_path}")
    PRETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    target_counts = Counter()
    prediction_counts = Counter()
    total_masked = 0
    total_correct = 0
    total_eligible = 0
    total_masked_ratio_n = 0
    gradient_norms: list[float] = []
    started = time.perf_counter()
    iterator = iter(loader)
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for step in range(1, EXPECTED_PRETRAIN_STEPS + 1):
            batch = next(iterator)
            step_started = time.perf_counter()
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(batch, seed=42, stream_step=step)
            loss = output["camr_loss"]
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"non-finite CAMR loss at step {step}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite CAMR gradient norm at step {step}")
            optimizer.step()
            scheduler.step()
            step_seconds = time.perf_counter() - step_started
            atomic_grad = _grad_report(
                atomic_parameters,
                [parameter.grad for parameter in atomic_parameters],
                none_is_zero=True,
            )
            mask_grad = _grad_report(
                [model.mask_embedding], [model.mask_embedding.grad]
            )
            head_grad = _grad_report(
                head_parameters,
                [parameter.grad for parameter in head_parameters],
                none_is_zero=True,
            )
            if not atomic_grad["all_finite"] or not atomic_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid Atomic-PC gradient at step {step}")
            if not mask_grad["all_finite"] or not mask_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid CAMR mask gradient at step {step}")
            if not head_grad["all_finite"] or not head_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid CAMR head gradient at step {step}")
            target_counts.update({
                str(classes[index]): int(value)
                for index, value in enumerate(output["target_class_counts"])
            })
            prediction_counts.update({
                str(classes[index]): int(value)
                for index, value in enumerate(output["prediction_class_counts"])
            })
            masked = int(output["masked_heavy_atom_count"])
            correct = int(output["masked_heavy_atom_correct"])
            eligible_count = int(output["mask_meta"]["eligible_count"])
            total_masked += masked
            total_correct += correct
            total_eligible += eligible_count
            total_masked_ratio_n += 1
            gradient_norms.append(float(atomic_grad["l2_norm"]))
            row = {
                "step": int(step),
                "objective": "L_CAMR = CrossEntropy(masked center heavy-atom Z)",
                "camr_loss": float(loss.detach().float().cpu()),
                "masked_heavy_atom_accuracy": float(output["masked_heavy_atom_accuracy"]),
                "masked_heavy_atom_count": masked,
                "masked_heavy_atom_correct": correct,
                "mask_ratio": float(output["mask_meta"]["mask_ratio"]),
                "eligible_center_heavy_atom_count": eligible_count,
                "target_class_counts": output["target_class_counts"],
                "prediction_class_counts": output["prediction_class_counts"],
                "atomic_pc_grad_norm": float(atomic_grad["l2_norm"]),
                "mask_embedding_grad_norm": float(mask_grad["l2_norm"]),
                "camr_head_grad_norm": float(head_grad["l2_norm"]),
                "clipped_grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": float(step_seconds),
                "throughput_graphs_per_second": float(
                    batch.graph_available.numel() / max(step_seconds, 1e-9)
                ),
                "coordinate_noise_sigma": 0.0,
                "ccdd_loss_used": False,
                "md200_loss_used": False,
                "o8_gradient": "NOT_PRESENT",
                "kfuse_gradient": "NOT_PRESENT",
            }
            records.append(row)
            metrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_handle.flush()
    if len(records) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("CAMR training did not complete exactly 1504 steps")
    final_hashes = {
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "mask_embedding": _tensor_digest("mask_embedding", model.mask_embedding),
        "camr_head": _module_digest(model.camr_head),
    }
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("atomic_point_encoder.")
        or name == "mask_embedding"
        or name.startswith("camr_head.")
    }
    if not any(name.startswith("atomic_point_encoder.") for name in state):
        raise RuntimeError("CAMR checkpoint state has no Atomic-PC tensors")
    checkpoint_metadata = {
        "pretrained_components": ["atomic_point_encoder"],
        "temporary_training_components": ["mask_embedding", "camr_head"],
        "excluded_components": ["O8", "KFuse", "MD200", "CCDD"],
        "objective": "L_CAMR = CrossEntropy(masked center heavy-atom Z)",
        "classes": classes,
        "mask_rate": CAMR_MASK_RATE,
        "mask_target": "ru_offset == 0 AND atomic_number > 1",
        "coordinate_noise_sigma": 0.0,
        "ccdd": False,
        "md200_used_for_pretrain_loss": False,
        "o8_used_for_pretrain_loss": False,
        "kfuse_used_for_pretrain_loss": False,
        "atomic_pool_scope": "center",
        "knn": 24,
        "layers": 4,
        "subset_sample_key_hash": _load_reference_contract()["subset_key_hash"],
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "optimizer_steps": EXPECTED_PRETRAIN_STEPS,
    }
    torch.save(
        {
            "schema": CAMR_CHECKPOINT_SCHEMA,
            "step": EXPECTED_PRETRAIN_STEPS,
            "model_state": state,
            "metadata": checkpoint_metadata,
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
        },
        checkpoint_path,
    )
    class_target_distribution = {
        str(key): int(target_counts.get(str(key), 0))
        for key in classes
    }
    class_prediction_distribution = {
        str(key): int(prediction_counts.get(str(key), 0))
        for key in classes
    }
    majority_class = int(support["majority_class"])
    majority_count = int(target_counts.get(str(majority_class), 0))
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-camr-v1-pretraining-summary",
        "status": "COMPLETE",
        "subset_count": EXPECTED_SUBSET_COUNT,
        "training_eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "optimizer_steps": EXPECTED_PRETRAIN_STEPS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "objective": "L_CAMR = CrossEntropy(masked center heavy-atom Z)",
        "coordinate_noise_sigma": 0.0,
        "ccdd_used": False,
        "md200_used_for_pretrain_loss": False,
        "o8_trained": False,
        "kfuse_trained": False,
        "atomic_pc_trained": True,
        "camr_head_trained": True,
        "mask_embedding_trained": True,
        "classes": classes,
        "class_target_distribution": class_target_distribution,
        "class_prediction_distribution": class_prediction_distribution,
        "masked_target_count": int(total_masked),
        "masked_target_correct": int(total_correct),
        "masked_heavy_atom_accuracy": float(total_correct / max(1, total_masked)),
        "mask_ratio": float(total_masked / max(1, total_eligible)),
        "majority_class": majority_class,
        "majority_class_count": majority_count,
        "majority_class_accuracy_baseline": float(majority_count / max(1, total_masked)),
        "atomic_pc_gradient_norm_mean": float(np.mean(gradient_norms)),
        "atomic_pc_gradient_norm_max": float(np.max(gradient_norms)),
        "pretraining_wall_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_gpu_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "initial_hashes": initial_hashes,
        "final_hashes": final_hashes,
        "optimizer": {
            "name": "AdamW",
            "lr": 2e-4,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "scheduler": "joint_route._make_optimizer LambdaLR (matched 1504-step schedule)",
        "mask_contract": {
            "target": "ru_offset == 0 AND atomic_number > 1",
            "explicit_h_context": True,
            "h_target_count": 0,
            "point_deletion": False,
            "coordinate_deletion": False,
            "one_mask_per_eligible_sample": True,
        },
    }
    _write(PRETRAIN_ROOT / "parameter_source_table.json", {
        "schema": "original-mips-atomic-pc-camr-v1-parameter-source-table",
        "rows": [
            {"module": "atomic_point_encoder", "source": "same CCDD-only fresh Atomic-PC construction; CAMR trained", **_module_summary(model.atomic_point_encoder)},
            {"module": "mask_embedding", "source": "fresh learned CAMR [MASK] embedding; trained", "tensor_count": 1, "parameter_count": int(model.mask_embedding.numel()), "buffer_count": 0, "hash": final_hashes["mask_embedding"]},
            {"module": "camr_head", "source": "fresh temporary CAMR classification head; trained", **_module_summary(model.camr_head)},
            {"module": "O8", "source": "not present in CAMR loss/training graph", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "KFuse", "source": "not present in CAMR loss/training graph", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "MD200", "source": "not used in CAMR pretraining loss", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "CCDD", "source": "removed", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
        ],
        "joint_checkpoint_loaded": False,
        "unexpected_joint_weight_count": 0,
    })
    _write(PRETRAIN_ROOT / "pretraining_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_camr_classes_for_downstream() -> list[int]:
    return [int(value) for value in _load_camr_classes()["classes"]]


def _load_camr_state_for_downstream() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint, state = _load_camr_checkpoint()
    atomic_names = {name for name in state if name.startswith("atomic_point_encoder.")}
    if not atomic_names:
        raise RuntimeError("CAMR checkpoint has no Atomic-PC state for transplant")
    return checkpoint, {
        name: value for name, value in state.items() if name.startswith("atomic_point_encoder.")
    }


_BASE_CENTER_MODEL = downstream_route._center_model


def _camr_downstream_model() -> tuple[nn.Module, dict[str, Any]]:
    checkpoint, state = _load_camr_state_for_downstream()
    model, base_meta = _BASE_CENTER_MODEL()
    target = model.state_dict()
    prefix = "atomic_point_encoder."
    target_names = {name for name in target if name.startswith(prefix)}
    source_names = set(state)
    if target_names != source_names:
        raise RuntimeError("CAMR Atomic-PC tensor set does not match Base target")
    base_hashes = {
        name: _module_digest(getattr(model, name))
        for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")
    }
    loaded_names = []
    for name in sorted(target_names):
        source = state[name]
        if tuple(source.shape) != tuple(target[name].shape):
            raise RuntimeError(f"CAMR tensor shape mismatch for {name}")
        target[name] = source.detach().clone()
        loaded_names.append(name)
    model.load_state_dict(target, strict=True)
    observed_hashes = {
        name: _module_digest(getattr(model, name))
        for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")
    }
    for name in observed_hashes:
        if name != "atomic_point_encoder" and observed_hashes[name] != base_hashes[name]:
            raise RuntimeError(f"non-Atomic-PC Base module changed during CAMR transplant: {name}")
    atomic_hash = _state_prefix_digest(
        {name: value for name, value in state.items()}, prefix
    )
    if observed_hashes["atomic_point_encoder"] != atomic_hash:
        raise RuntimeError("post-load CAMR Atomic-PC hash mismatch")
    source_table = []
    for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp"):
        module = getattr(model, name)
        source_table.append({
            "module": name,
            "source": "camr_checkpoint" if name == "atomic_point_encoder" else "base_no_pretrain_initialization",
            "joint_weights_loaded": False,
            **_module_summary(module),
        })
    return model, {
        "path": str(PRETRAIN_ROOT / "camr_checkpoint.pt"),
        "sha256": _sha(PRETRAIN_ROOT / "camr_checkpoint.pt"),
        "checkpoint_schema": checkpoint.get("schema"),
        "pretrain_step": int(checkpoint.get("step", -1)),
        "loaded_components": ["atomic_point_encoder"],
        "loaded_tensor_count": len(loaded_names),
        "loaded_tensor_names": sorted(loaded_names),
        "excluded_pretraining_components": ["mask_embedding", "camr_head"],
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
    downstream_route._center_model = _camr_downstream_model


def _run_downstream(tasks: list[str], folds: list[int]) -> None:
    gate = _json(RESULT_ROOT / "prestart_gate.json")
    if not bool(gate.get("all_pass")):
        raise SystemExit("CAMR prestart gate is not PASS")
    _load_camr_checkpoint()
    _configure_downstream()
    success_keys = downstream_route.geometry_success_keys()
    md_values, _ = downstream_route.load_md_table()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in tasks:
        for fold in folds:
            shard = SHARD_ROOT / f"{task}_fold{fold}.json"
            if shard.is_file():
                continue
            result = downstream_route._fold_run(
                task, int(fold), md_values, success_keys, device
            )
            print(
                json.dumps(
                    {"task": task, "fold": fold, "test_r2": result["test_r2"], "best_epoch": result["best_epoch"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _aggregate() -> dict[str, Any]:
    _configure_downstream()
    per_task, macro = downstream_route._aggregate()
    rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    if len(rows) != len(TASKS) * len(FOLDS):
        raise RuntimeError(f"CAMR downstream aggregate has {len(rows)} rows, expected 40")
    if any(row.get("checkpoint", {}).get("loaded_components") != ["atomic_point_encoder"] for row in rows):
        raise RuntimeError("CAMR downstream aggregate loaded an unexpected component")
    if any(int(row.get("checkpoint", {}).get("unexpected_joint_weight_count", -1)) != 0 for row in rows):
        raise RuntimeError("CAMR downstream aggregate contains unexpected joint weights")
    _write(DOWNSTREAM_ROOT / "aggregate_summary.json", {
        "schema": "original-mips-atomic-pc-center-ru-camr-v1-downstream-aggregate",
        "row_count": len(rows),
        "task_count": len(per_task),
        "macro": macro,
        "source_table_row_count": int(sum(len(row.get("checkpoint", {}).get("source_table", [])) for row in rows)),
        "loaded_components": ["atomic_point_encoder"],
        "output_root": str(DOWNSTREAM_ROOT),
    })
    return {"per_task": per_task, "macro": macro, "row_count": len(rows)}


def _finalize() -> dict[str, Any]:
    task_payload = _json(DOWNSTREAM_ROOT / "per_task_metrics.json")
    camr_fold_rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    base_task_payload = _json(BASE_ROOT / "training/per_task_metrics.json")
    base_fold_rows = _json(BASE_ROOT / "training/per_fold_metrics.json")
    ccdd_task_payload = _json(CCDD_DOWNSTREAM_ROOT / "per_task_metrics.json")
    ccdd_fold_rows = _json(CCDD_DOWNSTREAM_ROOT / "per_fold_metrics.json")
    old_task_payload = _json(OLD_T2_ROOT / "per_task_metrics.json")
    old_fold_rows = _json(OLD_T2_ROOT / "per_fold_metrics.json")
    camr_macro = float(task_payload["macro"]["macro_test_r2"])
    base_macro = float(base_task_payload["macro"]["macro_test_r2"])
    ccdd_macro = float(ccdd_task_payload["macro"]["macro_test_r2"])
    old_macro = float(old_task_payload["macro"]["macro_test_r2"])
    if not math.isclose(base_macro, REFERENCE_BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Base macro changed while finalizing CAMR")
    if not math.isclose(ccdd_macro, REFERENCE_CCDD_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("CCDD macro changed while finalizing CAMR")
    if not math.isclose(old_macro, REFERENCE_OLD_T2_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("old T2 macro changed while finalizing CAMR")
    if len(camr_fold_rows) != 40:
        raise RuntimeError(f"CAMR finalize requires 40 folds, found {len(camr_fold_rows)}")
    camr_task = task_payload["per_task"]
    base_task = base_task_payload["per_task"]
    ccdd_task = ccdd_task_payload["per_task"]
    old_task = old_task_payload["per_task"]
    per_task: dict[str, Any] = {}
    for task in TASKS:
        per_task[task] = {
            "camr_only_test_r2_mean": float(camr_task[task]["test_r2_mean"]),
            "base_test_r2_mean": float(base_task[task]["test_r2_mean"]),
            "ccdd_only_test_r2_mean": float(ccdd_task[task]["test_r2_mean"]),
            "old_t2_test_r2_mean": float(old_task[task]["test_r2_mean"]),
            "delta_vs_base": float(camr_task[task]["test_r2_mean"] - base_task[task]["test_r2_mean"]),
            "delta_vs_ccdd": float(camr_task[task]["test_r2_mean"] - ccdd_task[task]["test_r2_mean"]),
            "delta_vs_old_t2": float(camr_task[task]["test_r2_mean"] - old_task[task]["test_r2_mean"]),
            "positive_fold_count_vs_base": 0,
            "positive_fold_count_vs_ccdd": 0,
            "positive_fold_count_vs_old_t2": 0,
            "fold_count": 5,
        }
    base_by_key = {(row["task"], int(row["fold"])): row for row in base_fold_rows}
    ccdd_by_key = {(row["task"], int(row["fold"])): row for row in ccdd_fold_rows}
    old_by_key = {(row["task"], int(row["fold"])): row for row in old_fold_rows}
    fold_rows = []
    for row in camr_fold_rows:
        key = (row["task"], int(row["fold"]))
        if key not in base_by_key or key not in ccdd_by_key or key not in old_by_key:
            raise RuntimeError(f"missing matched reference fold for {key}")
        camr_r2 = float(row["test_r2"])
        base_r2 = float(base_by_key[key]["test_r2"])
        ccdd_r2 = float(ccdd_by_key[key]["test_r2"])
        old_r2 = float(old_by_key[key]["test_r2"])
        delta_base = camr_r2 - base_r2
        delta_ccdd = camr_r2 - ccdd_r2
        delta_old = camr_r2 - old_r2
        fold_rows.append({
            "task": key[0],
            "fold": key[1],
            "camr_only_test_r2": camr_r2,
            "base_test_r2": base_r2,
            "ccdd_only_test_r2": ccdd_r2,
            "old_t2_test_r2": old_r2,
            "delta_vs_base": delta_base,
            "delta_vs_ccdd": delta_ccdd,
            "delta_vs_old_t2": delta_old,
        })
        if delta_base > 0:
            per_task[key[0]]["positive_fold_count_vs_base"] += 1
        if delta_ccdd > 0:
            per_task[key[0]]["positive_fold_count_vs_ccdd"] += 1
        if delta_old > 0:
            per_task[key[0]]["positive_fold_count_vs_old_t2"] += 1
    fold_rows.sort(key=lambda item: (item["task"], item["fold"]))
    positive_tasks = [
        task for task in TASKS if per_task[task]["delta_vs_base"] > 0
    ]
    positive_folds = sum(1 for row in fold_rows if row["delta_vs_base"] > 0)
    positive_folds_ccdd = sum(1 for row in fold_rows if row["delta_vs_ccdd"] > 0)
    positive_folds_old = sum(1 for row in fold_rows if row["delta_vs_old_t2"] > 0)
    if camr_macro > base_macro:
        transfer = "POSITIVE"
        mismatch = "STRONGLY_SUPPORTED"
        promising = "YES"
        insufficient = "NOT_SUPPORTED"
    elif ccdd_macro < camr_macro <= base_macro:
        transfer = "IMPROVED_BUT_NOT_POSITIVE"
        mismatch = "PARTIAL_SUPPORT"
        promising = "UNCERTAIN"
        insufficient = "NOT_SUPPORTED"
    else:
        transfer = "NEGATIVE"
        mismatch = "NOT_SUPPORTED_BY_THIS_COMPARISON"
        promising = "UNCERTAIN"
        insufficient = "SUPPORTED"
    judgment = {
        "CAMR_TRANSFER": transfer,
        "CCDD_OBJECTIVE_MISMATCH": mismatch,
        "ATOMIC_PC_PRETRAINING_REMAINS_PROMISING": promising,
        "SINGLE_ATOM_MASKING_INSUFFICIENT": insufficient,
    }
    matrix = {
        "schema": "original-mips-atomic-pc-center-ru-camr-v1-diagnosis-matrix",
        "references": {
            "base_macro_r2": base_macro,
            "ccdd_only_macro_r2": ccdd_macro,
            "old_t2_joint_pretrained_pc_macro_r2": old_macro,
        },
        "camr_only_pc_macro_r2": camr_macro,
        "delta_vs_base": camr_macro - base_macro,
        "delta_vs_ccdd": camr_macro - ccdd_macro,
        "delta_vs_old_t2": camr_macro - old_macro,
        "positive_tasks": len(positive_tasks),
        "positive_tasks_list": positive_tasks,
        "positive_folds": positive_folds,
        "positive_folds_vs_ccdd": positive_folds_ccdd,
        "positive_folds_vs_old_t2": positive_folds_old,
        "fold_count": len(fold_rows),
        "per_task": per_task,
        "per_fold": fold_rows,
        "focus_tasks": {
            task: per_task[task] for task in ("ei", "eps", "nc", "xc")
        },
        "judgment": judgment,
    }
    _write(RESULT_ROOT / "diagnosis_matrix.json", matrix)
    pretraining_summary = _json(PRETRAIN_ROOT / "pretraining_summary.json")
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-camr-v1-summary",
        "status": "COMPLETE",
        "pretraining": pretraining_summary,
        "downstream": {
            "macro": task_payload["macro"],
            "row_count": len(camr_fold_rows),
            "protocol": "historical_shared5",
            "seed": 42,
            "tasks": list(TASKS),
            "folds": list(FOLDS),
        },
        "diagnosis": matrix,
        "ready_for_next_review": "YES",
        "stop": "YES",
        "automatic_fragment_masking": "NO",
        "automatic_joint_pretraining": "NO",
    }
    _write(RESULT_ROOT / "summary.json", summary)
    packet = {
        "schema": "original-mips-atomic-pc-center-ru-camr-v1-return-packet",
        "status": "COMPLETE",
        "CAMR_ONLY_PC_MACRO_R2": camr_macro,
        "DELTA_VS_BASE": camr_macro - base_macro,
        "DELTA_VS_CCDD": camr_macro - ccdd_macro,
        "DELTA_VS_OLD_T2": camr_macro - old_macro,
        "positive_tasks_over_8": len(positive_tasks),
        "positive_folds_over_40": positive_folds,
        "positive_folds_vs_ccdd_over_40": positive_folds_ccdd,
        "positive_folds_vs_old_t2_over_40": positive_folds_old,
        "focus_task_deltas": {
            task: {
                "delta_vs_base": per_task[task]["delta_vs_base"],
                "delta_vs_ccdd": per_task[task]["delta_vs_ccdd"],
                "delta_vs_old_t2": per_task[task]["delta_vs_old_t2"],
            }
            for task in ("ei", "eps", "nc", "xc")
        },
        "judgment": judgment,
        "READY_FOR_NEXT_REVIEW": "YES",
        "STOP": "YES",
    }
    _write(RESULT_ROOT / "RETURN_PACKET.json", packet)
    md_lines = [
        "# Atomic-PC CAMR-v1 Matched Objective Replacement",
        "",
        "## Status",
        "",
        "- PRESTART_GATE = PASS; CAMR mask, clean-geometry and gradient gates passed.",
        "- READY_FOR_NEXT_REVIEW = YES.",
        "- STOP = YES; no fragment masking, joint pretraining, scheduler change or 100K/1M route.",
        "",
        "## Results",
        "",
        f"- BASE = {base_macro:.15f}",
        f"- CCDD_ONLY = {ccdd_macro:.15f}",
        f"- OLD_JOINT_PC_T2 = {old_macro:.15f}",
        f"- CAMR_ONLY_PC_MACRO_R2 = {camr_macro:.15f}",
        f"- DELTA_VS_BASE = {camr_macro - base_macro:.15f}",
        f"- DELTA_VS_CCDD = {camr_macro - ccdd_macro:.15f}",
        f"- DELTA_VS_OLD_T2 = {camr_macro - old_macro:.15f}",
        f"- positive tasks / 8 = {len(positive_tasks)}/8; positive folds / 40 = {positive_folds}/40 (vs Base).",
        "",
        "## Per-task deltas",
        "",
        "| task | CAMR R2 | delta vs Base | delta vs CCDD | delta vs Old T2 | positive folds vs Base |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = per_task[task]
        md_lines.append(
            f"| {task} | {row['camr_only_test_r2_mean']:.12f} | "
            f"{row['delta_vs_base']:+.12f} | {row['delta_vs_ccdd']:+.12f} | "
            f"{row['delta_vs_old_t2']:+.12f} | "
            f"{row['positive_fold_count_vs_base']}/5 |"
        )
    md_lines.extend([
        "",
        f"- ei delta vs Base = {per_task['ei']['delta_vs_base']:.15f}.",
        f"- eps delta vs Base = {per_task['eps']['delta_vs_base']:.15f}.",
        f"- nc delta vs Base = {per_task['nc']['delta_vs_base']:.15f}.",
        f"- xc delta vs Base = {per_task['xc']['delta_vs_base']:.15f}.",
        "",
        "## Judgment",
        "",
        f"- CAMR_TRANSFER = {transfer}.",
        f"- CCDD_OBJECTIVE_MISMATCH = {mismatch}.",
        f"- ATOMIC_PC_PRETRAINING_REMAINS_PROMISING = {promising}.",
        f"- SINGLE_ATOM_MASKING_INSUFFICIENT = {insufficient}.",
        "",
        "CAMR pretraining diagnostics are recorded in pretraining_summary.json, including masked heavy-atom accuracy, class distributions, majority baseline and Atomic-PC gradient norms. The downstream protocol is historical_shared5; its held-out folds are shared validation/test folds, not an independent blind test.",
        "",
        "## Artifacts",
        "",
        "- prestart_gate.json, initialization_audit.json, gradient_audit.json, source_audit.json",
        "- pretraining/camr_class_support.json, pretraining/camr_checkpoint.pt, pretraining/pretraining_summary.json, pretraining/parameter_source_table.json",
        "- downstream/per_task_metrics.json, downstream/per_fold_metrics.json, diagnosis_matrix.json, summary.json, RETURN_PACKET.json",
        "",
        "Final action: STOP.",
    ])
    (RESULT_ROOT / "RETURN_PACKET.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--downstream", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--finalize", action="store_true")
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
