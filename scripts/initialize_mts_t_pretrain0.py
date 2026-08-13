#!/usr/bin/env python3
"""Create the paired fresh step-0 T0/T1 model states for T-Pretrain-0.

The two models are constructed independently under seed 42.  T1 then receives
the T0 tensors by parameter name/shape, while its two new local projections
remain strict finite zeros.  No optimizer, scheduler, sampler, data cursor or
trained checkpoint is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.uni_encoder import UniEncoderAttention
from src.utils import set_global_seed


ROOT = Path(__file__).resolve().parents[1]
PAIR_ID = "mts_t_pretrain0_matched_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def resolve(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        ["/opt/conda/envs/MTS/bin/python", "scripts/resolve_mips_trimer_scage.py", str(path)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def build_model(
    variant: str,
    *,
    graph_geometry_mode: str = "trimer_scage_mcl",
    g_family_arm: str | None = None,
    relation_geometry_sidecar: str | None = None,
    g3_permutation_sidecar: str | None = None,
    use_star_rbf: bool = True,
    use_mcl: bool = True,
) -> UniEncoderAttention:
    """Mirror the fixed graph-only MTS constructor used by pretrain.py."""

    return UniEncoderAttention(
        # pretrain.py keeps the graph backbone at 512 but projects the graph
        # representation to the parser's fixed joint_embedding_dim=256.
        joint_embedding_dim=256,
        smiles_model_name="",
        gnn_model_name="",
        modality_list=["graph"],
        freeze_encoder=False,
        graph_num_layers=6,
        graph_emb_dim=512,
        graph_dropout=0.1,
        graph_encoder_type="mips_trimer_scage",
        scage_dist_bar=[20.0, 50.0],
        scage_num_heads=8,
        scage_ffn_hidden_dim=2048,
        scage_num_kernels=128,
        scage_attention_dropout=0.1,
        scage_use_pbc_distance=False,
        scage_use_descriptors=False,
        scage_distance_mode="mips_dual",
        scage_distance_rbf=32,
        scage_distance_cutoff=12.0,
        scage_distance_scales=[4.0, 8.0, 12.0],
        scage_distance_taus=[0.5, 1.0, 1.5],
        scage_topology_bias=True,
        scage_topology_max_distance=20,
        scage_topology_locality_mode="soft",
        scage_topology_locality_threshold=5,
        scage_topology_locality_tau=1.0,
        scage_periodic_image_mode="none",
        scage_periodic_image_cap=0,
        scage_periodic_image_temperature=0.5,
        scage_force_topology_only=True,
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode=graph_geometry_mode,
        mcl_distance_percentiles=[0.20, 0.50],
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_variant="O8",
        mips_fusion_mode="none",
        projection_mode="plain",
        modality_control="real",
        mips_atom_feature_mode="mips137",
        mips_attention_scale="head_dim",
        mips_norm_mode="post",
        mips_activation="relu",
        mips_spd_bias_mode="per_head",
        mips_path_bias_mode="per_head_single_path_node",
        mips_multi_scale_hop_gate=False,
        mips_semantics="paper_semantic",
        mips_descriptor_fusion_mode="graph_md_residual",
        mips_descriptor_components="md200",
        mips_descriptor_disturbance=0.0,
        mips_backbone_mode="independent",
        mips_input_norm=False,
        mips_mask_mode="zero",
        mips_mask_policy="canonical_exact",
        mips_masked_loss_reduction="atom_mean",
        use_star_rbf=use_star_rbf,
        use_mcl=use_mcl,
        mcl_mask_mode="real",
        topology_attention_variant=variant,
        msta_layer_indices=(4, 5),
        msta_local_spd=(0, 1),
        msta_context_spd=(0, 1, 2),
        msta_share_relation_dropout=True,
        msta_local_output_bias=False,
        msta_local_output_init="zero",
        g_family_arm=g_family_arm,
        relation_geometry_sidecar=relation_geometry_sidecar,
        g3_permutation_sidecar=g3_permutation_sidecar,
        fusion_type="none",
        fp_mode="disabled",
        alignment_projection_dim=256,
    )


def save_payload(path: Path, state: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite fresh init artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)[1])
    try:
        torch.save({"schema": "mts-pretrain-init-v1", "state_dict": state, "meta": meta}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-config", type=Path, default=ROOT / "configs/mts/experiments/T0_o8_pretrain20k_matched_v1.json")
    parser.add_argument("--t1-config", type=Path, default=ROOT / "configs/mts/experiments/T1_msta_pretrain20k_matched_v1.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/t_pretrain0_v1")
    parser.add_argument("--parity-output", type=Path, default=ROOT / "results/mts_multiscale_topology/t_pretrain0_v1/initialization_parity.json")
    args = parser.parse_args()
    t0_config = args.t0_config.resolve()
    t1_config = args.t1_config.resolve()
    t0_resolved = resolve(t0_config)
    t1_resolved = resolve(t1_config)
    if t0_resolved["topology_attention_variant"] != "o8":
        raise RuntimeError("T0 initialization requires topology_attention_variant=o8")
    if t1_resolved["topology_attention_variant"] != "msta_last2":
        raise RuntimeError("T1 initialization requires topology_attention_variant=msta_last2")

    set_global_seed(42)
    t0_model = build_model("o8")
    t0_state = {key: value.detach().cpu().clone() for key, value in t0_model.state_dict().items()}
    set_global_seed(42)
    t1_model = build_model("msta_last2")
    t1_state = {key: value.detach().cpu().clone() for key, value in t1_model.state_dict().items()}

    shared = sorted(set(t0_state) & set(t1_state))
    if set(t0_state) - set(t1_state):
        raise RuntimeError(f"T1 is missing shared T0 parameters: {sorted(set(t0_state)-set(t1_state))[:8]}")
    parity_rows = []
    for key in shared:
        left = t0_state[key]
        right = t1_state[key]
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            raise RuntimeError(f"shared parameter shape/dtype mismatch: {key}")
        t1_state[key] = left.clone()
        parity_rows.append({
            "parameter_name": key,
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "t0_init_hash": tensor_hash(left),
            "t1_shared_init_hash": tensor_hash(t1_state[key]),
            "equal": bool(torch.equal(left, t1_state[key])),
        })
    local_rows = []
    for index in (4, 5):
        key = f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"
        value = t1_state.get(key)
        if value is None or not torch.isfinite(value).all() or torch.count_nonzero(value).item() != 0:
            raise RuntimeError(f"T1 local_output is not strict finite zero: {key}")
        local_rows.append({
            "parameter_name": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "t1_init_hash": tensor_hash(value),
            "finite": True,
            "zero": True,
        })

    output_root = args.output_root.resolve()
    t0_path = output_root / "T0_o8" / "mts_t_pretrain0_v1_step0.pth"
    t1_path = output_root / "T1_msta" / "mts_t_pretrain0_v1_step0.pth"
    common_meta = {
        "schema": "mts-pretrain-init-v1",
        "initialization": "fresh_paired",
        "paired_init_id": PAIR_ID,
        "optimizer_steps": 0,
        "random_seed": 42,
        "parent_checkpoint": None,
        "optimizer_state_inherited": False,
        "scheduler_state_inherited": False,
        "sampler_state_inherited": False,
        "t0_config_path": str(t0_config),
        "t1_config_path": str(t1_config),
        "t0_config_hash": t0_resolved["config_hash"],
        "t1_config_hash": t1_resolved["config_hash"],
        "cache_identity_source": "Gate A audit; no payload loaded",
    }
    save_payload(t0_path, t0_state, {
        **common_meta,
        "model_identity": "T0",
        "topology_attention_variant": "o8",
        "target_config_hash": t0_resolved["config_hash"],
        "target_graph_model_config_hash": t0_resolved["graph_model_config_hash"],
    })
    save_payload(t1_path, t1_state, {
        **common_meta,
        "model_identity": "T1",
        "topology_attention_variant": "msta_last2",
        "target_config_hash": t1_resolved["config_hash"],
        "target_graph_model_config_hash": t1_resolved["graph_model_config_hash"],
        "local_output_zero": True,
    })
    for path in (t0_path, t1_path):
        manifest = {
            "schema": "mts-pretrain-init-manifest-v1",
            "paired_init_id": PAIR_ID,
            "model_identity": "T0" if path == t0_path else "T1",
            "output_checkpoint": str(path),
            "output_checkpoint_sha256": sha256(path),
            "optimizer_steps": 0,
            "optimizer_state_inherited": False,
            "scheduler_state_inherited": False,
            "sampler_state_inherited": False,
            "strict_state_dict": True,
        }
        manifest_path = path.with_suffix(path.suffix + ".json")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite init manifest: {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    parity = {
        "schema": "mts-pretrain0-initialization-parity-v1",
        "cycle_id": PAIR_ID,
        "t0": {"path": str(t0_path), "sha256": sha256(t0_path), "config": t0_resolved},
        "t1": {"path": str(t1_path), "sha256": sha256(t1_path), "config": t1_resolved},
        "shared_parameter_count": len(parity_rows),
        "shared_parameters_equal": all(row["equal"] for row in parity_rows),
        "parameters": parity_rows,
        "t1_local_output": local_rows,
        "optimizer_state_inherited": False,
        "scheduler_state_inherited": False,
        "sampler_state_inherited": False,
        "source_checkpoint_used": None,
    }
    parity_path = args.parity_output.resolve()
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    if parity_path.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {parity_path}")
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "paired_init_id": PAIR_ID,
        "t0": str(t0_path),
        "t1": str(t1_path),
        "shared_parameter_count": len(parity_rows),
        "shared_parameters_equal": parity["shared_parameters_equal"],
        "t1_local_output_zero": all(row["zero"] for row in local_rows),
        "parity_report": str(parity_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
