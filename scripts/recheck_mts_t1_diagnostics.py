#!/usr/bin/env python3
"""Read-only final MSTA diagnostic recheck for the existing T1 checkpoint.

This is deliberately separate from ``pretrain.py``.  It loads the immutable
20k checkpoint, reads a small deterministic set of frozen cache records, and
performs only ``eval()``/``no_grad()`` forwards.  No optimizer, backward pass,
checkpoint, cache writer, or historical diagnostic report is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.pretrain import (
    TrimerAngleHead,
    _joint_angle_terms,
    _joint_canonical_mask,
    _joint_masked_atom_terms,
)
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.lmdb_cache import LmdbFeatureStore, load_cohort
from src.modules.mips_local_graph import MIPSLocalGraphEncoder, MSTA_LAYER_INDICES


ROOT = Path(__file__).resolve().parents[1]
COHORT_HASH = "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049"
TOPOLOGY_ROOT = ROOT / (
    "data/processed/mips_trimer_scage/topology/"
    "1658c6a602aebb7b73a6e77a3612f5faa613bb93b1d78c57a198ee4d787fa47a"
)
TRIMER_ROOT = ROOT / (
    "data/processed/mips_trimer_scage/trimer/"
    "131f7c6105c86b23d1cee8050cfd35b9c4f3fc45856a5bead1728cb8ea2865c1"
)
ANGLE_ROOT = TRIMER_ROOT / "derived/bond_angle" / COHORT_HASH
EXPECTED_T1_SHA256 = (
    "d6c229daa7f01216c6f57292ec50e1ff4357a28e1a566caa74e73bd0fd780191"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (torch.Tensor,)):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    state = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _same_rng(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    return "cuda" not in left or all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


def _load_probe_batch(sample_count: int):
    cohort = load_cohort(
        ROOT / "data/processed/mips_trimer_scage/cohorts/PI1M_v2" / COHORT_HASH,
        load_text=False,
        verify_integrity=False,
    )
    store = LmdbFeatureStore(
        topology_root=TOPOLOGY_ROOT,
        trimer_root=TRIMER_ROOT,
        cohort=cohort,
        angle_cache_root=ANGLE_ROOT,
    )
    selected_keys = []
    selected_data = []
    try:
        # The first rows are deterministic but some records have no valid
        # Angle-20 triplets.  Select the first fixed valid rows so the local-off
        # recheck has nonzero masked-atom and angle counts.
        for index, key_array in enumerate(cohort["keys_array"][:512]):
            key = bytes(key_array)
            data = store[key]
            if int(getattr(data, "trimer_angle_index", torch.empty(0)).size(0)) <= 0:
                continue
            if not bool(torch.as_tensor(getattr(data, "trimer_angle_valid", False)).item()):
                continue
            selected_keys.append(key)
            selected_data.append(data)
            if len(selected_data) >= int(sample_count):
                break
    finally:
        store.close()
    if len(selected_data) < int(sample_count):
        raise RuntimeError(
            f"could not find {sample_count} deterministic angle-valid probe records"
        )
    return mips_trimer_collate(selected_data), selected_keys


def _load_encoder(checkpoint_path: Path, device: torch.device):
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != EXPECTED_T1_SHA256:
        raise RuntimeError(
            "T1 checkpoint SHA mismatch: "
            f"observed={checkpoint_sha} expected={EXPECTED_T1_SHA256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    required_meta = {
        "model_identity": "T1",
        "initialization": "fresh_paired",
        "paired_init_id": "mts_t_pretrain0_matched_v1",
        "optimizer_steps": 20000,
    }
    for field, expected in required_meta.items():
        if meta.get(field) != expected:
            raise RuntimeError(
                f"T1 checkpoint metadata mismatch for {field}: "
                f"{meta.get(field)!r} != {expected!r}"
            )
    if meta.get("parent_checkpoint") not in (None, ""):
        raise RuntimeError("fresh-paired T1 checkpoint unexpectedly has a parent")
    encoder = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2")
    prefix = "encoders.graph.encoder."
    graph_state = {
        key[len(prefix):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(graph_state, strict=True)
    encoder.to(device).eval()
    for index in MSTA_LAYER_INDICES:
        attention = encoder.layers[index].attention
        attention.diagnostic_capture = True
        attention.diagnostic_local_off = False
    return encoder, checkpoint_sha, meta


def _make_probe_heads(device: torch.device):
    # The final checkpoint intentionally stores only the graph encoder.  These
    # fixed, deterministic read-only heads let us express local-off deltas in
    # the same masked-atom/Angle-20 loss units without claiming a training run.
    generator_state = torch.get_rng_state().clone()
    torch.manual_seed(20260812)
    atom_head = nn.Linear(512, 101).to(device).eval()
    angle_head = TrimerAngleHead(
        dim=512, hidden=256, bins=20,
        alpha=torch.ones(20), dropout=0.0, objective="categorical",
    ).to(device).eval()
    torch.set_rng_state(generator_state)
    return atom_head, angle_head


def _evaluate(encoder, atom_head, angle_head, batch, *, local_off: bool):
    for index in MSTA_LAYER_INDICES:
        encoder.layers[index].attention.diagnostic_local_off = bool(local_off)
        encoder.layers[index].attention.last_diagnostic = None
    mask = _joint_canonical_mask(batch, seed=42, stream_step=20000, mask_ratio=0.30)
    graph, node_states, aux = encoder.forward_joint_pretrain(batch, mask)
    atom_sum, _, atom_count, _ = _joint_masked_atom_terms(
        batch, node_states, atom_head, mask
    )
    angle_sum, _, angle_count, _, _, _ = _joint_angle_terms(
        batch, aux["final_trimer_states"], angle_head, gamma=2.0
    )
    layers = {
        str(index): dict(encoder.layers[index].attention.last_diagnostic or {})
        for index in MSTA_LAYER_INDICES
    }
    return {
        "masked_atom_loss": float(
            (atom_sum / max(1, atom_count)).float().item()
        ),
        "angle_loss": float(
            (angle_sum / max(1, angle_count)).float().item()
        ),
        "masked_atoms": int(atom_count),
        "angle_graphs": int(angle_count),
        "layers": layers,
        "graph_finite": bool(torch.isfinite(graph).all()),
        "node_finite": bool(torch.isfinite(node_states).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / (
            "pretrained_models/mts_multiscale_topology/t_pretrain0_v1/"
            "T1_msta/mts_t_pretrain0_v1_T1_step20000.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / (
            "results/mts_multiscale_topology/t_pretrain0_v1/diagnostics_review/"
            "final_step20k_recheck.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA final diagnostic but CUDA is unavailable")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    encoder, checkpoint_sha, checkpoint_meta = _load_encoder(
        args.checkpoint.resolve(), device
    )
    atom_head, angle_head = _make_probe_heads(device)
    batch_cpu, sample_keys = _load_probe_batch(args.sample_count)
    batch = batch_cpu.to(device)
    sample_key_hash = hashlib.sha256(b"".join(sample_keys)).hexdigest()

    rng_before = _capture_rng(device)
    with torch.no_grad():
        normal = _evaluate(
            encoder, atom_head, angle_head, batch, local_off=False
        )
        local_off = _evaluate(
            encoder, atom_head, angle_head, batch, local_off=True
        )
    rng_after_forward = _capture_rng(device)
    _restore_rng(rng_before)
    rng_after_restore = _capture_rng(device)
    for index in MSTA_LAYER_INDICES:
        encoder.layers[index].attention.diagnostic_local_off = False

    deltas = {
        "masked_atom": local_off["masked_atom_loss"] - normal["masked_atom_loss"],
        "angle": local_off["angle_loss"] - normal["angle_loss"],
    }
    deltas["weighted_total"] = deltas["masked_atom"] + 0.25 * deltas["angle"]
    required_entropy = []
    for layer in normal["layers"].values():
        for name in ("local_attention_entropy", "context_attention_entropy"):
            summary = layer.get(name) or {}
            required_entropy.append(
                bool(
                    summary.get("count", 0) > 0
                    and summary.get("finite") is True
                    and summary.get("range_valid") is True
                )
            )
    finite = bool(
        all(bool(row.get("finite")) for row in normal["layers"].values())
        and normal["graph_finite"]
        and normal["node_finite"]
        and local_off["graph_finite"]
        and local_off["node_finite"]
        and all(torch.isfinite(torch.tensor(value)) for value in deltas.values())
        and _same_rng(rng_before, rng_after_restore)
    )
    complete = bool(
        finite
        and len(required_entropy) == 2 * len(MSTA_LAYER_INDICES)
        and all(required_entropy)
        and normal["masked_atoms"] > 0
        and normal["angle_graphs"] > 0
        and local_off["masked_atoms"] == normal["masked_atoms"]
        and local_off["angle_graphs"] == normal["angle_graphs"]
    )
    payload = {
        "schema": "mts-msta-final-diagnostic-recheck-v2",
        "cycle_id": "mts_t_pretrain0_review_repair_v1",
        "phase": "T-Pretrain-0 review repair",
        "mode": {
            "eval": True,
            "no_grad": True,
            "backward": False,
            "optimizer": False,
            "training_started": False,
            "device": str(device),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "model_identity": checkpoint_meta.get("model_identity"),
            "initialization": checkpoint_meta.get("initialization"),
            "paired_init_id": checkpoint_meta.get("paired_init_id"),
            "optimizer_steps": checkpoint_meta.get("optimizer_steps"),
            "pretrain_profile_id": (checkpoint_meta.get("pretrain_profile") or {}).get("profile_id"),
            "pretrain_code_identity_sha256": (
                checkpoint_meta.get("pretrain_code_identity") or {}
            ).get("sha256"),
        },
        "probe": {
            "sample_count": len(sample_keys),
            "sample_keys": [key.hex() for key in sample_keys],
            "sample_key_hash": sample_key_hash,
            "seed": 42,
            "stream_step": 20000,
            "mask_ratio": 0.30,
        },
        "diagnostics": normal["layers"],
        "local_off_loss_delta": {
            **deltas,
            "normal": {
                "masked_atom_loss": normal["masked_atom_loss"],
                "angle_loss": normal["angle_loss"],
                "masked_atoms": normal["masked_atoms"],
                "angle_graphs": normal["angle_graphs"],
            },
            "local_off": {
                "masked_atom_loss": local_off["masked_atom_loss"],
                "angle_loss": local_off["angle_loss"],
                "masked_atoms": local_off["masked_atoms"],
                "angle_graphs": local_off["angle_graphs"],
            },
            "probe_head_initialization": "deterministic_fixed_read_only",
        },
        "historical_trajectory_entropy": {
            "status": "unavailable_due_to_diagnostic_bug",
            "source_reports_unchanged": True,
            "note": "Historical milestone entropy was invalidated by masked 0*log(0) and head-summed normalization; this recheck does not backfill old rows.",
        },
        "rng": {
            "restored_after_forward": _same_rng(rng_before, rng_after_restore),
            "forward_left_rng_unchanged": _same_rng(rng_before, rng_after_forward),
        },
        "finite": finite,
        "complete": complete,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "finite": finite, "complete": complete}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
