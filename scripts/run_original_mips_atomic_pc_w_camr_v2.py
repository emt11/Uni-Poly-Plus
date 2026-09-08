#!/usr/bin/env python
"""Atomic-PC W-CAMR-v2 matched masking-strategy experiment.

This route reuses the completed CAMR-v1 route in-process and changes only the
per-sample mask selector.  The selector is the WMM implementation from Lin &
Fung (ACS Omega 2024): ``w_alpha = ln(k * (n_alpha + 1)) / n_alpha`` with the
repository's fixed ``k=0.9`` (rounded to four decimals), followed by weighted
random sampling without replacement using A-Res keys ``u ** (1 / w)``.

The route is deliberately bounded: the frozen 50K/48,101 cohort, geometry,
Atomic-PC construction, Center-RU downstream and 1504-step optimizer schedule
are inherited from CAMR-v1.  No geometry generation, model-source edit,
scheduler change, sweep, joint pretraining or 100K/1M route is reachable.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.w_camr_v2_support import camr_reference_runtime as camr_v1  # noqa: E402


RESULT_ROOT = ROOT / "results/original_mips_atomic_pc_w_camr_v2"
PRETRAIN_ROOT = RESULT_ROOT / "pretraining"
DOWNSTREAM_ROOT = RESULT_ROOT / "downstream"
SHARD_ROOT = DOWNSTREAM_ROOT / "shards"
REFERENCE_ROOT = RESULT_ROOT / "references"
CAMR_V1_ROOT = REFERENCE_ROOT / "camr_v1"
CAMR_V1_PRETRAIN_ROOT = CAMR_V1_ROOT / "pretraining"
CAMR_V1_DOWNSTREAM_ROOT = CAMR_V1_ROOT / "downstream"

TASKS = tuple(camr_v1.TASKS)
FOLDS = tuple(camr_v1.FOLDS)
BASE_MACRO_R2 = 0.8403314216055405
CCDD_MACRO_R2 = 0.8384641276585666
CAMR_V1_MACRO_R2 = 0.840736320482714
OLD_T2_MACRO_R2 = 0.8387253670489404
EXPECTED_SUBSET_COUNT = 50000
EXPECTED_ELIGIBLE_COUNT = 48101
EXPECTED_PRETRAIN_STEPS = 1504
EXPECTED_BATCH_SIZE = 32
MASK_RATE = 0.15
WMM_K = 0.9
WMM_ROUNDING_DECIMALS = 4
WMM_CHECKPOINT_SCHEMA = (
    "original-mips-atomic-pc-center-ru-w-camr-v2-checkpoint"
)
WMM_MASKING_SCHEMA = (
    "original-mips-atomic-pc-w-camr-center-heavy-mask-v2"
)
CAMR_OBJECTIVE = "L_CAMR = CrossEntropy(masked center heavy-atom Z)"

_ORIGINAL_CAMR_MASK = camr_v1._make_camr_mask
_ORIGINAL_CAMR_FORWARD = camr_v1.CAMRPretrainer.forward


def _patch_camr_route() -> None:
    """Redirect CAMR helpers to this route without editing CAMR-v1 on disk."""

    center_reference = REFERENCE_ROOT / "center_ru_base"
    joint_reference = REFERENCE_ROOT / "joint_pretrain"
    ccdd_reference = REFERENCE_ROOT / "ccdd_only"
    t2_reference = REFERENCE_ROOT / "t2_atomic_pc_only" / "downstream"
    center_runtime = camr_v1.downstream_route
    cohort_runtime = camr_v1.joint_route
    ccdd_runtime = camr_v1.ccdd_route

    # These modules were relocated unchanged from top-level route scripts into
    # W-CAMR's internal support package.  Restore their repository-root and
    # input constants before any file access; keeping cohort_runtime.py byte
    # identical preserves its audited source hash.
    for module in (camr_v1, center_runtime, cohort_runtime, ccdd_runtime):
        module.ROOT = ROOT
    center_runtime.CHECKPOINT = ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth"
    center_runtime.GEOMETRY_ROOT = ROOT / "data/processed/mips_trimer_scage/trimer"
    center_runtime.CONFIG_PATH = ROOT / "configs/atomic_point_center_ru_v1.json"
    center_runtime.BASE_CONFIG_PATH = ROOT / "configs/atomic_point_v1.json"
    center_runtime.SOURCE_PATHS = (
        ROOT / "src/training/w_camr_v2_support/center_runtime.py",
        ROOT / "src/dataset/original_mips_atomic_pc.py",
        ROOT / "src/dataset/dataloader.py",
        ROOT / "src/modules/atomic_point_encoder.py",
        ROOT / "src/modules/original_mips_atomic_pc.py",
        ROOT / "src/modules/original_mips_knowledge_fusion.py",
        ROOT / "src/modules/original_mips_md200.py",
    )
    cohort_runtime.SUBSET_CSV = ROOT / "data/raw/PI1M_50k.csv"
    cohort_runtime.SUBSET_META = ROOT / "data/raw/PI1M_50k.subset.json"
    cohort_runtime.O8_CHECKPOINT = center_runtime.CHECKPOINT
    camr_v1.BASE_O8_CHECKPOINT = center_runtime.CHECKPOINT
    ccdd_runtime.BASE_O8_CHECKPOINT = center_runtime.CHECKPOINT

    # The historical matched controls are retained under this route as
    # immutable provenance dependencies, not as independently advertised
    # top-level routes.
    camr_v1.BASE_ROOT = center_reference
    camr_v1.OLD_T2_ROOT = t2_reference
    camr_v1.CCDD_ROOT = ccdd_reference
    camr_v1.CCDD_PRETRAIN_ROOT = ccdd_reference / "pretraining"
    camr_v1.CCDD_DOWNSTREAM_ROOT = ccdd_reference / "downstream"
    cohort_runtime.RESULT_ROOT = joint_reference
    cohort_runtime.PRETRAIN_ROOT = joint_reference / "pretraining"
    ccdd_runtime.RESULT_ROOT = ccdd_reference
    ccdd_runtime.PRETRAIN_ROOT = ccdd_reference / "pretraining"
    ccdd_runtime.DOWNSTREAM_ROOT = ccdd_reference / "downstream"
    ccdd_runtime.SHARD_ROOT = ccdd_reference / "downstream" / "shards"
    ccdd_runtime.JOINT_ROOT = joint_reference
    ccdd_runtime.JOINT_PRETRAIN_ROOT = joint_reference / "pretraining"
    ccdd_runtime.BASE_ROOT = center_reference
    ccdd_runtime.OLD_T2_ROOT = t2_reference

    camr_v1.RESULT_ROOT = RESULT_ROOT
    camr_v1.PRETRAIN_ROOT = PRETRAIN_ROOT
    camr_v1.DOWNSTREAM_ROOT = DOWNSTREAM_ROOT
    camr_v1.SHARD_ROOT = SHARD_ROOT
    camr_v1.CAMR_CHECKPOINT_SCHEMA = WMM_CHECKPOINT_SCHEMA
    camr_v1.CAMR_MASKING_SCHEMA = WMM_MASKING_SCHEMA
    camr_v1._make_camr_mask = _make_weighted_mask
    camr_v1.CAMRPretrainer.forward = _w_camr_forward
    camr_v1._load_camr_checkpoint = _load_w_checkpoint
    camr_v1._camr_downstream_model = _w_downstream_model


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


def _module_digest(module: torch.nn.Module) -> str:
    return _digest_items(module.state_dict().items())


def _tensor_digest(name: str, tensor: torch.Tensor) -> str:
    return _digest_items([(name, tensor)])


def _module_summary(module: torch.nn.Module) -> dict[str, Any]:
    state = module.state_dict()
    parameter_names = {name for name, _ in module.named_parameters()}
    return {
        "tensor_count": len(state),
        "parameter_count": int(sum(parameter.numel() for parameter in module.parameters())),
        "buffer_count": int(sum(name not in parameter_names for name, name_value in module.state_dict().items())),
        "hash": _module_digest(module),
    }


def _wmm_weights(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return the fixed paper/repository WMM weight for every atom."""

    values = [int(value) for value in atomic_numbers.detach().cpu().reshape(-1).tolist()]
    counts = Counter(values)
    weights = []
    for value in values:
        n_alpha = int(counts[value])
        raw = math.log(WMM_K * (n_alpha + 1.0)) / float(n_alpha)
        weight = round(raw, WMM_ROUNDING_DECIMALS)
        if not math.isfinite(weight) or weight <= 0.0:
            raise FloatingPointError(
                f"WMM produced non-positive/non-finite weight for Z={value}, n={n_alpha}"
            )
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float64)


def _make_weighted_mask(
    cloud: Any,
    *,
    seed: int,
    stream_step: int,
    rate: float = MASK_RATE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """WMM weighted random sampling without replacement, count-matched to CAMR-v1."""

    if not 0.0 < float(rate) < 1.0:
        raise ValueError("W-CAMR mask rate must lie in (0,1)")
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
    per_graph: list[dict[str, Any]] = []
    for graph_id in range(graph_count):
        indices = torch.nonzero((batch == graph_id) & eligible, as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            per_graph.append({"graph": graph_id, "eligible": 0, "masked": 0})
            continue
        # This is intentionally byte-for-byte the CAMR-v1 target-count rule;
        # WMM changes only which eligible atoms are selected.
        mask_count = max(1, int(round(float(rate) * count)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (int(seed) * 1000003 + int(stream_step) * 9176 + graph_id * 6113 + 17)
            % (2**63 - 1)
        )
        local_z = z[indices].detach().cpu()
        weights = _wmm_weights(local_z)
        # Efraimidis--Spirakis / A-Res: retain the m largest u^(1/w) keys.
        uniforms = torch.rand(count, generator=generator, dtype=torch.float64)
        uniforms.clamp_(min=torch.finfo(torch.float64).tiny)
        keys = uniforms.pow(1.0 / weights)
        order = torch.argsort(keys, descending=True, stable=True)
        selected = indices.detach().cpu()[order[:mask_count]].to(device=device)
        mask[selected] = True
        per_graph.append(
            {
                "graph": graph_id,
                "eligible": count,
                "masked": int(mask_count),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
            }
        )
    eligible_count = int(eligible.sum().item())
    masked_count = int(mask.sum().item())
    return mask, {
        "schema": WMM_MASKING_SCHEMA,
        "strategy": "WMM weighted random sampling without replacement",
        "weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
        "k": WMM_K,
        "rounding_decimals": WMM_ROUNDING_DECIMALS,
        "sampling_key": "u ** (1 / w_alpha), retain largest keys (A-Res)",
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


def _w_camr_forward(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """CAMR-v1 forward plus per-element correctness diagnostics."""

    output = _ORIGINAL_CAMR_FORWARD(self, *args, **kwargs)
    labels = output["labels"].detach()
    predictions = output["logits"].detach().argmax(dim=-1)
    correct = labels[predictions == labels]
    output["target_class_correct_counts"] = [
        int(value)
        for value in torch.bincount(correct, minlength=len(self.classes)).cpu().tolist()
    ]
    output["masking_strategy"] = "W-CAMR-v2/WMM"
    return output


def _load_w_checkpoint() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load only the W-CAMR checkpoint for the matched Atomic-PC transplant."""

    path = PRETRAIN_ROOT / "w_camr_checkpoint.pt"
    if not path.is_file():
        raise RuntimeError(f"missing W-CAMR checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != WMM_CHECKPOINT_SCHEMA:
        raise RuntimeError("unexpected W-CAMR checkpoint schema")
    if int(checkpoint.get("step", -1)) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError("W-CAMR checkpoint is not the required 1504-step artifact")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("W-CAMR checkpoint has no model_state")
    unknown = sorted(
        name for name in state
        if not (name.startswith("atomic_point_encoder.") or name == "mask_embedding" or name.startswith("camr_head."))
    )
    if unknown:
        raise RuntimeError(f"W-CAMR checkpoint contains unexpected tensors: {unknown[:5]}")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("pretrained_components") != ["atomic_point_encoder"]:
        raise RuntimeError("W-CAMR pretrained component metadata is not exact")
    if metadata.get("objective") != CAMR_OBJECTIVE:
        raise RuntimeError("W-CAMR checkpoint objective metadata mismatch")
    if metadata.get("masking_strategy") != "W-CAMR-v2/WMM" or float(metadata.get("wmm_k", -1.0)) != WMM_K:
        raise RuntimeError("W-CAMR checkpoint WMM metadata mismatch")
    classes = [int(value) for value in metadata.get("classes", [])]
    if classes != [int(value) for value in camr_v1._load_camr_classes().get("classes", [])]:
        raise RuntimeError("W-CAMR checkpoint class map differs from prestart support")
    if metadata.get("subset_sample_key_hash") != camr_v1._load_reference_contract()["subset_key_hash"]:
        raise RuntimeError("W-CAMR checkpoint subset hash mismatch")
    if float(metadata.get("coordinate_noise_sigma", -1.0)) != 0.0 or metadata.get("ccdd") is not False:
        raise RuntimeError("W-CAMR checkpoint is not the clean-geometry/no-CCDD route")
    return checkpoint, {str(name): value for name, value in state.items()}


def _w_downstream_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    """Center-RU model with exactly one transplanted Atomic-PC component."""

    checkpoint, state = _load_w_checkpoint()
    model, base_meta = camr_v1._BASE_CENTER_MODEL()
    target = model.state_dict()
    prefix = "atomic_point_encoder."
    target_names = {name for name in target if name.startswith(prefix)}
    source_names = {name for name in state if name.startswith(prefix)}
    if target_names != source_names:
        raise RuntimeError("W-CAMR Atomic-PC tensor set does not match Base target")
    base_hashes = {
        name: camr_v1._module_digest(getattr(model, name))
        for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")
    }
    loaded_names = []
    for name in sorted(target_names):
        source = state[name]
        if tuple(source.shape) != tuple(target[name].shape):
            raise RuntimeError(f"W-CAMR tensor shape mismatch for {name}")
        target[name] = source.detach().clone()
        loaded_names.append(name)
    model.load_state_dict(target, strict=True)
    observed_hashes = {
        name: camr_v1._module_digest(getattr(model, name))
        for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp")
    }
    for name in observed_hashes:
        if name != "atomic_point_encoder" and observed_hashes[name] != base_hashes[name]:
            raise RuntimeError(f"non-Atomic-PC Base module changed during W-CAMR transplant: {name}")
    atomic_hash = camr_v1._state_prefix_digest(state, prefix)
    if observed_hashes["atomic_point_encoder"] != atomic_hash:
        raise RuntimeError("post-load W-CAMR Atomic-PC hash mismatch")
    source_table = []
    for name in ("graph_encoder", "atomic_point_encoder", "fusion", "graph_norm", "graph_projection", "mlp"):
        source_table.append({
            "module": name,
            "source": "w_camr_checkpoint" if name == "atomic_point_encoder" else "base_no_pretrain_initialization",
            "joint_weights_loaded": False,
            **camr_v1._module_summary(getattr(model, name)),
        })
    return model, {
        "path": str(PRETRAIN_ROOT / "w_camr_checkpoint.pt"),
        "sha256": _sha(PRETRAIN_ROOT / "w_camr_checkpoint.pt"),
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


def _ensure_support() -> dict[str, Any]:
    source_path = CAMR_V1_PRETRAIN_ROOT / "camr_class_support.json"
    if not source_path.is_file():
        raise RuntimeError(f"missing immutable CAMR-v1 class support: {source_path}")
    source = _json(source_path)
    expected = {
        "sample_count": EXPECTED_ELIGIBLE_COUNT,
        "classes": source.get("classes"),
        "class_counts": source.get("class_counts"),
        "eligible_center_heavy_atom_count": source.get("eligible_center_heavy_atom_count"),
    }
    if int(source.get("sample_count", -1)) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("CAMR-v1 class support is not the matched 48,101 cohort")
    if not source.get("classes") or int(source.get("eligible_center_heavy_atom_count", -1)) <= 0:
        raise RuntimeError("CAMR-v1 class support is invalid")
    destination = PRETRAIN_ROOT / "camr_class_support.json"
    if destination.is_file() and _json(destination) != source:
        raise RuntimeError("existing W-CAMR support differs from immutable CAMR-v1 support")
    _write(destination, source)
    return source


def _preflight() -> dict[str, Any]:
    _patch_camr_route()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    reference = camr_v1._load_reference_contract()
    support = _ensure_support()

    # CAMR-v1's matched initialization/gradient/geometry audit is reused, but
    # its mask function resolves to the WMM selector patched above.
    camr_v1._preflight()
    inherited = _json(RESULT_ROOT / "prestart_gate.json")
    info = camr_v1.joint_route._load_subset_info()
    _dataset, batch = camr_v1._probe_batch(info)
    clean_cloud = camr_v1.OriginalMIPSAtomicPCJointPretrainer._point_cloud(batch)
    weighted_mask, weighted_meta = _make_weighted_mask(
        clean_cloud, seed=42, stream_step=0, rate=MASK_RATE
    )
    uniform_mask, uniform_meta = _ORIGINAL_CAMR_MASK(
        clean_cloud, seed=42, stream_step=0, rate=MASK_RATE
    )
    # _ORIGINAL_CAMR_MASK resolves CAMR-v1's schema through its module globals;
    # the route patch above intentionally changes that global for W-CAMR.  Tag
    # this audit-only reference explicitly as the immutable uniform selector.
    uniform_meta = dict(uniform_meta)
    uniform_meta.update({
        "schema": "original-mips-atomic-pc-camr-center-heavy-mask-v1",
        "strategy": "uniform random sampling without replacement",
    })
    same_counts = [
        int(weighted_row["masked"]) == int(uniform_row["masked"])
        for weighted_row, uniform_row in zip(
            weighted_meta["per_graph"], uniform_meta["per_graph"]
        )
    ]
    changed_selection = bool(torch.any(weighted_mask != uniform_mask).item())
    all_weights = []
    z = clean_cloud.atomic_number
    eligible = clean_cloud.ru_offset.eq(0) & z.gt(1)
    for graph_id in range(int(clean_cloud.ptr.numel()) - 1):
        indices = torch.nonzero(
            (clean_cloud.batch == graph_id) & eligible, as_tuple=False
        ).flatten()
        if indices.numel():
            all_weights.extend(_wmm_weights(z[indices]).tolist())
    model_source = {
        "joint_pretrainer": _sha(ROOT / "src/training/pretrain/original_mips_atomic_pc_joint.py"),
        "joint_runner": _sha(ROOT / "src/training/w_camr_v2_support/cohort_runtime.py"),
        "atomic_point_encoder": _sha(ROOT / "src/modules/atomic_point_encoder.py"),
        "joint_collator": _sha(ROOT / "src/dataset/original_mips_atomic_pc_joint.py"),
    }
    camr_source = _json(CAMR_V1_ROOT / "source_audit.json")
    source_payload = {
        "schema": "original-mips-atomic-pc-w-camr-v2-source-audit",
        "current_source_sha256": {"w_camr_runner": _sha(Path(__file__)), **model_source},
        "camr_v1_reference_source_sha256": camr_source.get("current_source_sha256", {}),
        "source_unchanged_vs_camr_v1": all(
            model_source.get(name) == camr_source.get("current_source_sha256", {}).get(name)
            for name in model_source
        ),
        "wmm_provenance": {
            "paper": "Lin & Fung, Rethinking the Masking Strategy for Pretraining Molecular Graphs from a Data-Centric View, ACS Omega 2024",
            "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11097184/",
            "repository": "Austin13579/weighted-masking-molecules",
            "repository_url": "https://github.com/Austin13579/weighted-masking-molecules",
            "implementation_url": "https://github.com/Austin13579/weighted-masking-molecules/blob/main/util.py",
            "weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
            "k": WMM_K,
            "rounding_decimals": WMM_ROUNDING_DECIMALS,
            "sampling": "weighted random sampling without replacement; A-Res key u ** (1 / w), retain largest keys",
            "k_selection": "fixed repository/GraphMAE setting k=0.9; no sweep",
        },
        "camr_v2_contract": {
            "mask_target": "ru_offset == 0 AND atomic_number > 1",
            "uniform_masking": False,
            "coordinate_noise_sigma": 0.0,
            "clean_geometry_for_knn": True,
            "same_target_count_rule": "max(1, round(0.15 * eligible_count)) per graph",
            "full_trimer_message_passing": True,
            "center_ru_pooling": True,
            "explicit_h_context_only": True,
            "masked_h_target_count": 0,
        },
    }
    _write(RESULT_ROOT / "source_audit.json", source_payload)
    inherited_gates = inherited.get("gates", {})
    checks = {
        "SAME_SUBSET": inherited_gates.get("SAME_50K_SUBSET") == "PASS",
        "SAME_GEOMETRY": inherited_gates.get("SAME_GEOMETRY") == "PASS",
        "SAME_PC_INITIALIZATION": inherited_gates.get("SAME_ATOMIC_PC_INIT") == "PASS",
        "SAME_TRAINING_SCHEDULE": inherited_gates.get("SAME_TRAINING_SCHEDULE") == "PASS",
        "SAME_MASK_COUNT_CONTRACT": bool(same_counts) and all(same_counts),
        "WEIGHTED_MASKING": weighted_meta.get("schema") == WMM_MASKING_SCHEMA
        and weighted_meta.get("strategy", "").startswith("WMM")
        and bool(all_weights)
        and all(math.isfinite(value) and value > 0.0 for value in all_weights),
        "WEIGHTED_SELECTOR_CHANGES_SELECTION": changed_selection,
        "CENTER_HEAVY_ONLY": inherited_gates.get("CENTER_HEAVY_MASK_ONLY") == "PASS",
        "MASKED_H_COUNT": inherited_gates.get("MASKED_H_TARGET_COUNT") == "PASS",
        "FULL_TRIMER_MP": inherited_gates.get("FULL_TRIMER_MESSAGE_PASSING") == "PASS",
        "CENTER_RU_POOLING": inherited_gates.get("CENTER_RU_CONTRACT") == "PASS",
        "CAMR_GRADIENT_TO_PC": inherited_gates.get("CAMR_TO_PC_GRADIENT") == "PASS",
        "O8_KFUSE_NOT_IN_TRAINING_GRAPH": inherited_gates.get("O8_GRADIENT") == "PASS"
        and inherited_gates.get("KFUSE_GRADIENT") == "PASS",
        "CLEAN_KNN": inherited_gates.get("CLEAN_KNN_USED") == "PASS",
        "NO_MD200_PRETRAIN_LOSS": inherited_gates.get("NO_MD200_IN_PRETRAIN_LOSS") == "PASS",
        "SOURCE_UNCHANGED": bool(source_payload["source_unchanged_vs_camr_v1"]),
        "NO_GEOMETRY_GENERATION": True,
        "NO_SCHEDULER_CHANGE": True,
        "NO_JOINT_PRETRAINING": True,
        "NO_100K_OR_1M": True,
    }
    payload = {
        "schema": "original-mips-atomic-pc-w-camr-v2-prestart-gate",
        "gates": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "all_pass": bool(all(checks.values())),
        "references": {
            "base_macro_r2": BASE_MACRO_R2,
            "ccdd_only_macro_r2": CCDD_MACRO_R2,
            "camr_v1_macro_r2": CAMR_V1_MACRO_R2,
            "old_t2_macro_r2": OLD_T2_MACRO_R2,
        },
        "masking_contract": {
            "UNIFORM_MASKING": "NO",
            "WEIGHTED_MASKING": "PASS" if checks["WEIGHTED_MASKING"] else "FAIL",
            "rate": MASK_RATE,
            "wmm_k": WMM_K,
            "weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
            "sampling": "A-Res weighted sampling without replacement",
            "probe_weighted_meta": weighted_meta,
            "probe_uniform_meta": uniform_meta,
            "probe_selection_changed": changed_selection,
            "per_graph_mask_count_equal": same_counts,
        },
        "matched_contract": {
            "subset_count": EXPECTED_SUBSET_COUNT,
            "eligible_count": EXPECTED_ELIGIBLE_COUNT,
            "subset_key_hash": reference["subset_key_hash"],
            "geometry_protocol": reference["geometry_protocol"],
            "geometry_source": reference["geometry_source"],
            "same_frozen_geometry": True,
            "same_sample_order_reference": "CAMR-v1 DataLoader construction and seed=42",
            "same_atomic_pc_initialization_reference": str(CAMR_V1_ROOT / "initialization_audit.json"),
            "knn": 24,
            "layers": 4,
            "pool_scope": "center",
            "pretrain_steps": EXPECTED_PRETRAIN_STEPS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "mask_target": "ru_offset == 0 AND atomic_number > 1",
            "explicit_h_context": True,
            "masked_h_target_count": 0,
            "optimizer": inherited.get("matched_contract", {}).get("optimizer"),
            "scheduler": inherited.get("matched_contract", {}).get("scheduler"),
        },
        "support": {
            "path": str(PRETRAIN_ROOT / "camr_class_support.json"),
            "class_count": support.get("class_count"),
            "eligible_center_heavy_atom_count": support.get("eligible_center_heavy_atom_count"),
        },
        "source_audit": str(RESULT_ROOT / "source_audit.json"),
        "inherited_camr_gate": inherited,
        "forbidden_actions": {
            "geometry_generation": "NO",
            "scheduler_change": "NO",
            "fragment_masking": "NO",
            "joint_pretraining": "NO",
            "100K_or_1M": "NO",
            "k_sweep": "NO",
        },
    }
    _write(RESULT_ROOT / "prestart_gate.json", payload)
    if not payload["all_pass"]:
        raise SystemExit("W-CAMR-v2 prestart gate failed; training was not started")
    print(json.dumps({"all_pass": True, "gates": payload["gates"]}, ensure_ascii=False, indent=2))
    print("W-CAMR-v2 prestart gate PASS; no training started")
    return payload


def _train() -> dict[str, Any]:
    _patch_camr_route()
    gate_path = RESULT_ROOT / "prestart_gate.json"
    if not gate_path.is_file() or not bool(_json(gate_path).get("all_pass")):
        raise SystemExit("W-CAMR-v2 prestart gate is not PASS; training was not started")
    checkpoint_path = PRETRAIN_ROOT / "w_camr_checkpoint.pt"
    if checkpoint_path.is_file():
        raise RuntimeError(f"refusing to overwrite existing W-CAMR checkpoint: {checkpoint_path}")
    support = _ensure_support()
    classes = [int(value) for value in support["classes"]]
    info = camr_v1.joint_route._load_subset_info()
    if len(info["eligible_indices"]) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("W-CAMR training eligible count changed")
    dataset = camr_v1.joint_route._build_dataset()
    collator = camr_v1.OriginalMIPSAtomicPCJointCollator(info["md_values"])
    loader_kwargs: dict[str, Any] = {
        "dataset": camr_v1.Subset(dataset, list(info["eligible_indices"])),
        "batch_size": EXPECTED_BATCH_SIZE,
        "shuffle": True,
        "drop_last": False,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = camr_v1.DataLoader(**loader_kwargs)
    if len(loader) != EXPECTED_PRETRAIN_STEPS:
        raise RuntimeError(f"W-CAMR loader has {len(loader)} batches, expected {EXPECTED_PRETRAIN_STEPS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match CAMR-v1 exactly: construct the loader before seed/model.
    camr_v1.set_global_seed(42)
    model = camr_v1.CAMRPretrainer(classes).to(device)
    optimizer, scheduler = camr_v1.joint_route._make_optimizer(model, EXPECTED_PRETRAIN_STEPS)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    atomic_parameters = list(model.atomic_point_encoder.parameters())
    head_parameters = list(model.camr_head.parameters())
    initial_hashes = {
        "atomic_point_encoder": _module_digest(model.atomic_point_encoder),
        "mask_embedding": _tensor_digest("mask_embedding", model.mask_embedding),
        "camr_head": _module_digest(model.camr_head),
    }
    metrics_path = PRETRAIN_ROOT / "training_metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError(f"refusing to overwrite existing W-CAMR metrics: {metrics_path}")
    PRETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    total_masked = 0
    total_correct = 0
    total_eligible = 0
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
            with camr_v1._autocast(device):
                output = model(batch, seed=42, stream_step=step)
            loss = output["camr_loss"]
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"non-finite W-CAMR loss at step {step}")
            loss.backward()
            clipped_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(clipped_norm):
                raise FloatingPointError(f"non-finite W-CAMR gradient norm at step {step}")
            optimizer.step()
            scheduler.step()
            step_seconds = time.perf_counter() - step_started
            atomic_grad = camr_v1._grad_report(
                atomic_parameters,
                [parameter.grad for parameter in atomic_parameters],
                none_is_zero=True,
            )
            mask_grad = camr_v1._grad_report([model.mask_embedding], [model.mask_embedding.grad])
            head_grad = camr_v1._grad_report(
                head_parameters,
                [parameter.grad for parameter in head_parameters],
                none_is_zero=True,
            )
            if not atomic_grad["all_finite"] or not atomic_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid Atomic-PC gradient at step {step}")
            if not mask_grad["all_finite"] or not mask_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid W-CAMR mask gradient at step {step}")
            if not head_grad["all_finite"] or not head_grad["any_nonzero"]:
                raise FloatingPointError(f"invalid W-CAMR head gradient at step {step}")
            for index, value in enumerate(output["target_class_counts"]):
                target_counts[str(classes[index])] += int(value)
            for index, value in enumerate(output["target_class_correct_counts"]):
                correct_counts[str(classes[index])] += int(value)
            for index, value in enumerate(output["prediction_class_counts"]):
                prediction_counts[str(classes[index])] += int(value)
            masked = int(output["masked_heavy_atom_count"])
            correct = int(output["masked_heavy_atom_correct"])
            eligible_count = int(output["mask_meta"]["eligible_count"])
            total_masked += masked
            total_correct += correct
            total_eligible += eligible_count
            gradient_norms.append(float(atomic_grad["l2_norm"]))
            row = {
                "step": int(step),
                "objective": CAMR_OBJECTIVE,
                "masking_strategy": "W-CAMR-v2/WMM",
                "wmm_k": WMM_K,
                "camr_loss": float(loss.detach().float().cpu()),
                "masked_heavy_atom_accuracy": float(output["masked_heavy_atom_accuracy"]),
                "masked_heavy_atom_count": masked,
                "masked_heavy_atom_correct": correct,
                "mask_ratio": float(output["mask_meta"]["mask_ratio"]),
                "eligible_center_heavy_atom_count": eligible_count,
                "target_class_counts": output["target_class_counts"],
                "target_class_correct_counts": output["target_class_correct_counts"],
                "prediction_class_counts": output["prediction_class_counts"],
                "atomic_pc_grad_norm": float(atomic_grad["l2_norm"]),
                "mask_embedding_grad_norm": float(mask_grad["l2_norm"]),
                "camr_head_grad_norm": float(head_grad["l2_norm"]),
                "clipped_grad_norm": clipped_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": float(step_seconds),
                "throughput_graphs_per_second": float(batch.graph_available.numel() / max(step_seconds, 1e-9)),
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
        raise RuntimeError("W-CAMR training did not complete exactly 1504 steps")
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
        raise RuntimeError("W-CAMR checkpoint state has no Atomic-PC tensors")
    checkpoint_metadata = {
        "pretrained_components": ["atomic_point_encoder"],
        "temporary_training_components": ["mask_embedding", "camr_head"],
        "excluded_components": ["O8", "KFuse", "MD200", "CCDD"],
        "objective": CAMR_OBJECTIVE,
        "masking_strategy": "W-CAMR-v2/WMM",
        "wmm_provenance": "Lin & Fung ACS Omega 2024 / Austin13579 weighted-masking-molecules",
        "wmm_k": WMM_K,
        "wmm_weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
        "wmm_sampling": "A-Res weighted random sampling without replacement",
        "classes": classes,
        "mask_rate": MASK_RATE,
        "mask_target": "ru_offset == 0 AND atomic_number > 1",
        "coordinate_noise_sigma": 0.0,
        "ccdd": False,
        "md200_used_for_pretrain_loss": False,
        "o8_used_for_pretrain_loss": False,
        "kfuse_used_for_pretrain_loss": False,
        "atomic_pool_scope": "center",
        "knn": 24,
        "layers": 4,
        "subset_sample_key_hash": camr_v1._load_reference_contract()["subset_key_hash"],
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "optimizer_steps": EXPECTED_PRETRAIN_STEPS,
    }
    torch.save(
        {
            "schema": WMM_CHECKPOINT_SCHEMA,
            "step": EXPECTED_PRETRAIN_STEPS,
            "model_state": state,
            "metadata": checkpoint_metadata,
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
        },
        checkpoint_path,
    )
    class_target_distribution = {str(key): int(target_counts.get(str(key), 0)) for key in classes}
    class_correct_distribution = {str(key): int(correct_counts.get(str(key), 0)) for key in classes}
    class_prediction_distribution = {str(key): int(prediction_counts.get(str(key), 0)) for key in classes}
    per_element_training = {
        str(key): {
            "target_count": class_target_distribution[str(key)],
            "correct_count": class_correct_distribution[str(key)],
            "accuracy": float(class_correct_distribution[str(key)] / max(1, class_target_distribution[str(key)])),
            "predicted_class_count": class_prediction_distribution[str(key)],
        }
        for key in classes
    }
    majority_class = int(support["majority_class"])
    majority_count = int(target_counts.get(str(majority_class), 0))
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-w-camr-v2-pretraining-summary",
        "status": "COMPLETE",
        "subset_count": EXPECTED_SUBSET_COUNT,
        "training_eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "optimizer_steps": EXPECTED_PRETRAIN_STEPS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "objective": CAMR_OBJECTIVE,
        "masking_strategy": "W-CAMR-v2/WMM",
        "wmm_k": WMM_K,
        "wmm_weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
        "wmm_sampling": "A-Res weighted random sampling without replacement",
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
        "class_correct_distribution": class_correct_distribution,
        "class_prediction_distribution": class_prediction_distribution,
        "per_element_training_diagnostics": per_element_training,
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
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "initial_hashes": initial_hashes,
        "final_hashes": final_hashes,
        "optimizer": {"name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0},
        "scheduler": "joint_route._make_optimizer LambdaLR (same 1504-step warmup schedule)",
        "mask_contract": {
            "target": "ru_offset == 0 AND atomic_number > 1",
            "explicit_h_context": True,
            "h_target_count": 0,
            "point_deletion": False,
            "coordinate_deletion": False,
            "one_mask_per_eligible_sample": True,
            "same_target_count_rule_as_camr_v1": True,
        },
    }
    _write(PRETRAIN_ROOT / "parameter_source_table.json", {
        "schema": "original-mips-atomic-pc-w-camr-v2-parameter-source-table",
        "rows": [
            {"module": "atomic_point_encoder", "source": "same CAMR-v1 fresh Atomic-PC construction; W-CAMR trained", **_module_summary(model.atomic_point_encoder)},
            {"module": "mask_embedding", "source": "fresh learned [MASK] embedding; W-CAMR trained", "tensor_count": 1, "parameter_count": int(model.mask_embedding.numel()), "buffer_count": 0, "hash": final_hashes["mask_embedding"]},
            {"module": "camr_head", "source": "fresh temporary CAMR classification head; W-CAMR trained", **_module_summary(model.camr_head)},
            {"module": "O8", "source": "not present in W-CAMR loss/training graph", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "KFuse", "source": "not present in W-CAMR loss/training graph", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "MD200", "source": "not used in W-CAMR pretraining loss", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
            {"module": "CCDD", "source": "removed", "tensor_count": 0, "parameter_count": 0, "buffer_count": 0},
        ],
        "joint_checkpoint_loaded": False,
        "unexpected_joint_weight_count": 0,
        "wmm_provenance": {
            "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11097184/",
            "repository_url": "https://github.com/Austin13579/weighted-masking-molecules",
            "k": WMM_K,
            "weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
            "sampling": "A-Res weighted random sampling without replacement",
        },
    })
    _write(PRETRAIN_ROOT / "pretraining_summary.json", summary)
    _write_masking_diagnostics(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _write_masking_diagnostics(w_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if w_summary is None:
        w_summary = _json(PRETRAIN_ROOT / "pretraining_summary.json")
    support = _json(PRETRAIN_ROOT / "camr_class_support.json")
    u_summary = _json(CAMR_V1_PRETRAIN_ROOT / "pretraining_summary.json")
    classes = [int(value) for value in support["classes"]]
    eligible = {str(key): int(value) for key, value in support["class_counts"].items()}
    uniform_masked = {str(key): int(u_summary["class_target_distribution"].get(str(key), 0)) for key in classes}
    weighted_masked = {str(key): int(w_summary["class_target_distribution"].get(str(key), 0)) for key in classes}
    total_eligible = sum(eligible.values())
    total_uniform = sum(uniform_masked.values())
    total_weighted = sum(weighted_masked.values())
    rows: dict[str, Any] = {}
    for key in classes:
        text_key = str(key)
        rows[text_key] = {
            "atomic_number": key,
            "eligible_count": eligible[text_key],
            "eligible_fraction": float(eligible[text_key] / max(1, total_eligible)),
            "camr_v1_uniform_masked_count": uniform_masked[text_key],
            "camr_v1_uniform_masked_fraction": float(uniform_masked[text_key] / max(1, eligible[text_key])),
            "camr_v1_uniform_masked_share": float(uniform_masked[text_key] / max(1, total_uniform)),
            "w_camr_v2_weighted_masked_count": weighted_masked[text_key],
            "w_camr_v2_weighted_masked_fraction": float(weighted_masked[text_key] / max(1, eligible[text_key])),
            "w_camr_v2_weighted_masked_share": float(weighted_masked[text_key] / max(1, total_weighted)),
            "weighted_to_uniform_masked_share_ratio": float(
                (weighted_masked[text_key] / max(1, total_weighted))
                / max(1e-12, uniform_masked[text_key] / max(1, total_uniform))
            ),
        }
    uniform_share = np.asarray([uniform_masked[str(key)] / max(1, total_uniform) for key in classes])
    weighted_share = np.asarray([weighted_masked[str(key)] / max(1, total_weighted) for key in classes])
    payload = {
        "schema": "original-mips-atomic-pc-w-camr-v2-masking-diagnostics",
        "camr_v1_source": str(CAMR_V1_PRETRAIN_ROOT / "pretraining_summary.json"),
        "w_camr_v2_source": str(PRETRAIN_ROOT / "pretraining_summary.json"),
        "denominator_definition": "masked_count is one training-epoch mask event count over the 48,101-sample cohort; masked_fraction = masked_count / eligible atom count",
        "wmm_provenance": {
            "weight_formula": "w_alpha = ln(k * (n_alpha + 1)) / n_alpha",
            "k": WMM_K,
            "sampling": "weighted random sampling without replacement (A-Res)",
        },
        "weighted_masking_changes_target_distribution": bool(np.max(np.abs(weighted_share - uniform_share)) > 1e-12),
        "masked_share_l1_shift": float(np.abs(weighted_share - uniform_share).sum()),
        "per_element": rows,
        "totals": {
            "eligible_count": total_eligible,
            "camr_v1_uniform_masked_count": total_uniform,
            "w_camr_v2_weighted_masked_count": total_weighted,
            "camr_v1_uniform_mask_ratio": float(total_uniform / max(1, total_eligible)),
            "w_camr_v2_weighted_mask_ratio": float(total_weighted / max(1, total_eligible)),
        },
    }
    _write(RESULT_ROOT / "masking_diagnostics.json", payload)
    return payload


def _run_downstream(tasks: list[str], folds: list[int]) -> None:
    _patch_camr_route()
    camr_v1._run_downstream(tasks, folds)


def _aggregate() -> dict[str, Any]:
    _patch_camr_route()
    aggregate = camr_v1._aggregate()
    rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    if len(rows) != len(TASKS) * len(FOLDS):
        raise RuntimeError(f"W-CAMR aggregate has {len(rows)} rows, expected 40")
    if any(row.get("checkpoint", {}).get("loaded_components") != ["atomic_point_encoder"] for row in rows):
        raise RuntimeError("W-CAMR downstream loaded an unexpected component")
    if any(int(row.get("checkpoint", {}).get("unexpected_joint_weight_count", -1)) != 0 for row in rows):
        raise RuntimeError("W-CAMR downstream contains unexpected joint weights")
    _write(DOWNSTREAM_ROOT / "aggregate_summary.json", {
        "schema": "original-mips-atomic-pc-center-ru-w-camr-v2-downstream-aggregate",
        "row_count": len(rows),
        "task_count": len(TASKS),
        "macro": aggregate["macro"],
        "loaded_components": ["atomic_point_encoder"],
        "unexpected_joint_weight_count": 0,
        "output_root": str(DOWNSTREAM_ROOT),
    })
    return {"per_task": aggregate["per_task"], "macro": aggregate["macro"], "row_count": len(rows)}


def _finalize() -> dict[str, Any]:
    _patch_camr_route()
    # Reuse the audited aggregation/fold comparison, then add the W-CAMR-v2
    # comparison against the immutable CAMR-v1 result.
    camr_v1._finalize()
    task_payload = _json(DOWNSTREAM_ROOT / "per_task_metrics.json")
    w_fold_rows = _json(DOWNSTREAM_ROOT / "per_fold_metrics.json")
    camr_task_payload = _json(CAMR_V1_DOWNSTREAM_ROOT / "per_task_metrics.json")
    camr_fold_rows = _json(CAMR_V1_DOWNSTREAM_ROOT / "per_fold_metrics.json")
    base_task_payload = _json(camr_v1.BASE_ROOT / "training/per_task_metrics.json")
    matrix = _json(RESULT_ROOT / "diagnosis_matrix.json")
    w_macro = float(task_payload["macro"]["macro_test_r2"])
    camr_macro = float(camr_task_payload["macro"]["macro_test_r2"])
    base_macro = float(base_task_payload["macro"]["macro_test_r2"])
    if not math.isclose(base_macro, BASE_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Base reference changed while finalizing W-CAMR")
    if not math.isclose(camr_macro, CAMR_V1_MACRO_R2, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"CAMR-v1 reference changed: {camr_macro}")
    if len(w_fold_rows) != 40 or len(camr_fold_rows) != 40:
        raise RuntimeError("W-CAMR and CAMR-v1 both require complete 40-fold outputs")
    w_tasks = task_payload["per_task"]
    camr_tasks = camr_task_payload["per_task"]
    camr_by_key = {(row["task"], int(row["fold"])): row for row in camr_fold_rows}
    per_task = matrix["per_task"]
    for task in TASKS:
        w_mean = float(w_tasks[task]["test_r2_mean"])
        c_mean = float(camr_tasks[task]["test_r2_mean"])
        per_task[task].update({
            "w_camr_only_test_r2_mean": w_mean,
            "camr_v1_test_r2_mean": c_mean,
            "delta_vs_camr_v1": w_mean - c_mean,
            "positive_fold_count_vs_camr_v1": 0,
        })
    fold_rows = matrix["per_fold"]
    for row in fold_rows:
        key = (row["task"], int(row["fold"]))
        camr_row = camr_by_key[key]
        w_row = next(item for item in w_fold_rows if (item["task"], int(item["fold"])) == key)
        delta = float(w_row["test_r2"] - camr_row["test_r2"])
        row["w_camr_only_test_r2"] = float(w_row["test_r2"])
        row["camr_v1_test_r2"] = float(camr_row["test_r2"])
        row["delta_vs_camr_v1"] = delta
        if delta > 0.0:
            per_task[key[0]]["positive_fold_count_vs_camr_v1"] += 1
    positive_tasks_camr = [task for task in TASKS if per_task[task]["delta_vs_camr_v1"] > 0.0]
    positive_folds_camr = sum(1 for row in fold_rows if row["delta_vs_camr_v1"] > 0.0)
    not_single = len(positive_tasks_camr) >= 2 and positive_folds_camr >= 2
    if w_macro > camr_macro and not_single:
        transfer, promote, concentration = "POSITIVE", "YES", "NOT_SINGLE_TASK_OR_FOLD_DOMINATED"
    elif abs(w_macro - camr_macro) <= 1e-6:
        transfer, promote, concentration = "NEUTRAL", "NO", "APPROXIMATELY_EQUAL"
    elif w_macro > camr_macro:
        transfer, promote, concentration = "POSITIVE_BUT_CONCENTRATED", "NO", "SINGLE_TASK_OR_FOLD_DOMINATED"
    else:
        transfer, promote, concentration = "NEGATIVE", "NO", "NOT_APPLICABLE"
    judgment = {
        "WEIGHTED_MASKING_TRANSFER": transfer,
        "CAMR_V2_PROMOTE": promote,
        "CAMR_V1_REMAINS_REFERENCE": "YES" if promote != "YES" else "NO",
        "IMPROVEMENT_CONCENTRATION": concentration,
        "positive_tasks_vs_camr_v1": len(positive_tasks_camr),
        "positive_folds_vs_camr_v1": positive_folds_camr,
    }
    masking = _write_masking_diagnostics()
    matrix.update({
        "schema": "original-mips-atomic-pc-center-ru-w-camr-v2-diagnosis-matrix",
        "w_camr_only_pc_macro_r2": w_macro,
        "camr_v1_macro_r2": camr_macro,
        "delta_vs_camr_v1": w_macro - camr_macro,
        "delta_vs_base": w_macro - base_macro,
        "delta_vs_ccdd": w_macro - CCDD_MACRO_R2,
        "positive_tasks_vs_base": sum(1 for task in TASKS if per_task[task]["delta_vs_base"] > 0.0),
        "positive_folds_vs_base": sum(1 for row in fold_rows if row["delta_vs_base"] > 0.0),
        "positive_tasks_vs_camr_v1": len(positive_tasks_camr),
        "positive_folds_vs_camr_v1": positive_folds_camr,
        "focus_tasks": {task: per_task[task] for task in ("ei", "eps", "nc", "xc")},
        "masking_diagnostics": masking,
        "judgment": judgment,
    })
    _write(RESULT_ROOT / "diagnosis_matrix.json", matrix)
    pretraining_summary = _json(PRETRAIN_ROOT / "pretraining_summary.json")
    summary = {
        "schema": "original-mips-atomic-pc-center-ru-w-camr-v2-summary",
        "status": "COMPLETE",
        "pretraining": pretraining_summary,
        "downstream": {
            "macro": task_payload["macro"],
            "row_count": len(w_fold_rows),
            "protocol": "historical_shared5",
            "seed": 42,
            "tasks": list(TASKS),
            "folds": list(FOLDS),
            "loaded_components": ["atomic_point_encoder"],
        },
        "diagnosis": matrix,
        "ready_for_next_review": "YES",
        "stop": "YES",
        "automatic_fragment_masking": "NO",
        "automatic_joint_pretraining": "NO",
        "automatic_100k_or_1m": "NO",
    }
    _write(RESULT_ROOT / "summary.json", summary)
    packet = {
        "schema": "original-mips-atomic-pc-center-ru-w-camr-v2-return-packet",
        "status": "COMPLETE",
        "W_CAMR_MACRO_R2": w_macro,
        "DELTA_VS_BASE": w_macro - base_macro,
        "DELTA_VS_CAMR_V1": w_macro - camr_macro,
        "DELTA_VS_CCDD": w_macro - CCDD_MACRO_R2,
        "positive_tasks_over_8_vs_base": int(matrix["positive_tasks_vs_base"]),
        "positive_folds_over_40_vs_base": int(matrix["positive_folds_vs_base"]),
        "positive_tasks_over_8_vs_camr_v1": len(positive_tasks_camr),
        "positive_folds_over_40_vs_camr_v1": positive_folds_camr,
        "focus_task_deltas": {
            task: {
                "delta_vs_base": per_task[task]["delta_vs_base"],
                "delta_vs_camr_v1": per_task[task]["delta_vs_camr_v1"],
                "delta_vs_ccdd": per_task[task]["delta_vs_ccdd"],
            }
            for task in ("ei", "eps", "nc", "xc")
        },
        "judgment": judgment,
        "masking_diagnostics": str(RESULT_ROOT / "masking_diagnostics.json"),
        "READY_FOR_NEXT_REVIEW": "YES",
        "STOP": "YES",
    }
    _write(RESULT_ROOT / "RETURN_PACKET.json", packet)
    md_lines = [
        "# Atomic-PC W-CAMR-v2 Matched Masking-Strategy Experiment", "", "## Status", "",
        "- PRESTART_GATE = PASS; WMM masking, matched initialization, clean-geometry, Center-RU and gradient gates passed.",
        "- READY_FOR_NEXT_REVIEW = YES.", "- STOP = YES; no scheduler change, fragment masking, joint pretraining, k sweep or 100K/1M route.", "",
        "## Results", "",
        f"- BASE = {base_macro:.15f}", f"- CCDD_ONLY = {CCDD_MACRO_R2:.15f}", f"- CAMR_V1 = {camr_macro:.15f}",
        f"- W_CAMR_MACRO_R2 = {w_macro:.15f}", f"- DELTA_VS_BASE = {w_macro - base_macro:+.15f}",
        f"- DELTA_VS_CAMR_V1 = {w_macro - camr_macro:+.15f}", f"- DELTA_VS_CCDD = {w_macro - CCDD_MACRO_R2:+.15f}",
        f"- positive tasks / 8 vs Base = {matrix['positive_tasks_vs_base']}/8; positive folds / 40 = {matrix['positive_folds_vs_base']}/40.",
        f"- positive tasks / 8 vs CAMR-v1 = {len(positive_tasks_camr)}/8; positive folds / 40 = {positive_folds_camr}/40.", "",
        "## Per-task deltas", "", "| task | W-CAMR R2 | Δ vs Base | Δ vs CAMR-v1 | Δ vs CCDD | positive folds vs CAMR-v1 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = per_task[task]
        md_lines.append(f"| {task} | {row['w_camr_only_test_r2_mean']:.12f} | {row['delta_vs_base']:+.12f} | {row['delta_vs_camr_v1']:+.12f} | {row['delta_vs_ccdd']:+.12f} | {row['positive_fold_count_vs_camr_v1']}/5 |")
    md_lines += ["", "## Focus tasks", "", *[f"- {task}: Δ vs Base = {per_task[task]['delta_vs_base']:+.15f}; Δ vs CAMR-v1 = {per_task[task]['delta_vs_camr_v1']:+.15f}." for task in ("ei", "eps", "nc", "xc")], "", "## WMM masking diagnostics", "", "| Z | eligible | CAMR-v1 masked | CAMR-v1 frac | W-CAMR masked | W-CAMR frac | weighted/uniform share |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in masking["per_element"].items():
        md_lines.append(f"| {key} | {row['eligible_count']} | {row['camr_v1_uniform_masked_count']} | {row['camr_v1_uniform_masked_fraction']:.6f} | {row['w_camr_v2_weighted_masked_count']} | {row['w_camr_v2_weighted_masked_fraction']:.6f} | {row['weighted_to_uniform_masked_share_ratio']:.3f} |")
    md_lines += ["", f"- WMM formula: `{WMM_K}` fixed `w_alpha = ln(k * (n_alpha + 1)) / n_alpha`, rounded to four decimals; A-Res keys `u ** (1 / w)` and largest-key retention without replacement.", "- The exact source/provenance and per-element denominator definition are in `source_audit.json` and `masking_diagnostics.json`.", "", "## Judgment", "", f"- WEIGHTED_MASKING_TRANSFER = {transfer}.", f"- CAMR_V2_PROMOTE = {promote}.", f"- CAMR_V1_REMAINS_REFERENCE = {judgment['CAMR_V1_REMAINS_REFERENCE']}.", f"- IMPROVEMENT_CONCENTRATION = {concentration}.", "", "The downstream protocol is historical_shared5; its held-out folds are shared validation/test folds, not an independent blind test.", "", "## Artifacts", "", "- prestart_gate.json, initialization_audit.json, gradient_audit.json, source_audit.json", "- pretraining/w_camr_checkpoint.pt, pretraining/pretraining_summary.json, pretraining/parameter_source_table.json, masking_diagnostics.json", "- downstream/per_task_metrics.json, downstream/per_fold_metrics.json, diagnosis_matrix.json, summary.json, RETURN_PACKET.json", "", "Final action: STOP."]
    (RESULT_ROOT / "RETURN_PACKET.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
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
    _patch_camr_route()
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
