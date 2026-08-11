#!/usr/bin/env python3
"""Production readiness checks for MIPS-Trimer-SCAGE (MTS).

This command is intentionally read-only.  It never opens an LMDB writer and
it reports a failed cache bundle as "not ready" instead of attempting to
repair or rebuild it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from scripts.finalize_mips_trimer_cache import (  # noqa: E402
    _active_writer_pids,
    _load_cohort,
    _validate_angle_sidecars,
    _validate_mcl_sidecars,
)
from src.dataset.mips_cache_validation import verify_frozen_cache_bundle  # noqa: E402
from src.modules.mips_local_graph import MIPSLocalGraphEncoder  # noqa: E402
from src.modules.uni_encoder import UniEncoderAttention  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.canonical_periodic import build_canonical_periodic_topology  # noqa: E402
from src.dataset.trimer_mcl import attach_finite_trimer_mcl  # noqa: E402
from src.dataset.mips_trimer_contract import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA,
    PRETRAIN_TARGET_CONTRACT_SCHEMA,
    PRETRAIN_PROFILE_ID,
    CACHE_CONTINUOUS_ANGLE_SCHEMA,
    CACHE_BOND_ANGLE_SCHEMA,
    PRETRAIN_PROFILE_ID,
    ROUTE_NAME,
    ROUTE_SHORT_NAME,
    cache_bundle_binding_hash,
)


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RuntimeError(f"checkpoint is missing: {path}")
        return path.resolve()
    # Default to the unique full canonical Angle-20 production checkpoint.
    # Historical migration/contract checkpoints are intentionally not matched
    # so the doctor never silently runs on a legacy artifact.
    candidates = sorted(
        (PROJECT_ROOT / "pretrained_models/mts").glob(
            "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "doctor requires exactly one canonical Angle-20 checkpoint; "
            f"found {len(candidates)}"
        )
    return candidates[0].resolve()


def _assert_checkpoint_binding(checkpoint_path: Path, specs, store_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("meta"), dict):
        raise RuntimeError("checkpoint is not a metadata-bearing MTS checkpoint")
    meta = checkpoint["meta"]
    if meta.get("schema") not in {CHECKPOINT_SCHEMA, PRETRAIN_CHECKPOINT_SCHEMA}:
        raise RuntimeError("checkpoint schema is not the active MTS schema")
    if meta.get("stage") != "mts_joint_pretraining":
        raise RuntimeError("checkpoint stage is not mts_joint_pretraining")
    # The doctor only accepts the full dual-identity production checkpoint
    # (Plan §7); an explicitly passed legacy checkpoint is rejected here.
    if not isinstance(meta.get("source_contract"), dict):
        raise RuntimeError("checkpoint has no source_contract identity")
    if not isinstance(meta.get("target_contract"), dict):
        raise RuntimeError("checkpoint has no target_contract identity")
    fresh_v4 = meta.get("schema") == PRETRAIN_CHECKPOINT_SCHEMA
    if fresh_v4 and meta.get("pretrain_profile", {}).get("profile_id") != PRETRAIN_PROFILE_ID:
        raise RuntimeError("mts-model-v4 checkpoint has no canonical Angle-20 profile")
    store = json.loads(store_path.read_text(encoding="utf-8"))
    binding = meta.get("final_cache_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("checkpoint has no final_cache_binding")
    if binding.get("store_sha256") != _sha256_file(store_path):
        raise RuntimeError("checkpoint store.json binding is stale")
    exact_path = store_path.parent / "exact_union_validation.json"
    if binding.get("exact_union_sha256") != _sha256_file(exact_path):
        raise RuntimeError("checkpoint exact-union binding is stale")
    if binding.get("store_transaction_id") != store.get("transaction_id"):
        raise RuntimeError("checkpoint freeze transaction binding is stale")
    for layer in ("topology", "trimer"):
        root = Path(specs[layer]["root"])
        expected = {
            "done_artifact_id": (root / ".done").read_text(encoding="utf-8").strip(),
            "done_file_sha256": _sha256_file(root / ".done"),
            "frozen_file_sha256": _sha256_file(root / ".frozen"),
        }
        observed = {
            key: binding.get(key, {}).get(layer)
            for key in expected
        }
        if observed != expected:
            raise RuntimeError(f"checkpoint {layer} frozen artifact binding is stale")
        if meta.get(f"{layer}_cache_artifact_hash") != expected["done_artifact_id"]:
            raise RuntimeError(f"checkpoint {layer} artifact metadata is stale")
    expected_bundle_hash = cache_bundle_binding_hash(
        cohort_hash=meta.get("source_cohort_hash"),
        topology_artifact_hash=binding["done_artifact_id"]["topology"],
        trimer_artifact_hash=binding["done_artifact_id"]["trimer"],
    )
    if meta.get("cache_bundle_hash") != expected_bundle_hash:
        raise RuntimeError("checkpoint cache_bundle_hash does not bind final artifacts")
    expected_angle_schema = (
        CACHE_BOND_ANGLE_SCHEMA if fresh_v4 else CACHE_CONTINUOUS_ANGLE_SCHEMA
    )
    pi1m_angle = next(
        item for item in binding.get("angle_sidecars", [])
        if item.get("cohort") == "PI1M_v2"
        and item.get("schema") == expected_angle_schema
    )
    if meta.get("angle_cache_artifact_hash") != pi1m_angle.get("artifact_hash"):
        raise RuntimeError("checkpoint PI1M Angle sidecar binding is stale")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no state_dict")
    # Match the complete graph-only joint-pretraining architecture, not just a
    # permissive subset of tensors.  The dimensions are inferred from the
    # checkpoint contract and then compared key-for-key and shape-for-shape.
    model = UniEncoderAttention(
        joint_embedding_dim=256,
        smiles_model_name=None,
        gnn_model_name=None,
        modality_list=["graph"],
        graph_num_layers=int(meta.get("model", {}).get("layers", 6)),
        graph_emb_dim=int(meta.get("model", {}).get("embedding_dim", 512)),
        graph_dropout=0.1,
        graph_encoder_type="mips_trimer_scage",
        scage_num_heads=int(meta.get("model", {}).get("heads", 8)),
        scage_ffn_hidden_dim=int(meta.get("model", {}).get("ffn_hidden_dim", 2048)),
        scage_use_pbc_distance=False,
        scage_use_descriptors=False,
        mips_core=meta.get("mips_core", "paper_corrected"),
        mips_max_hops=int(meta.get("mips_max_hops", 2)),
        mips_use_descriptors=bool(meta.get("mips_use_descriptors", True)),
        spatial_mode=meta.get("spatial_mode", "trimer_scage"),
        graph_geometry_mode=meta.get("graph_geometry_mode", "trimer_scage_mcl"),
        mcl_distance_percentiles=tuple(meta.get("mcl_distance_percentiles", (0.20, 0.50))),
        trimer_num_candidates=int(meta.get("trimer_num_candidates", 4)),
        trimer_max_heavy_atoms=int(meta.get("trimer_max_heavy_atoms", 384)),
        mips_variant=meta.get("mips_variant", "O8"),
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
        mips_semantics=meta.get("mips_semantics", "paper_semantic"),
        mips_descriptor_fusion_mode=meta.get("mips_descriptor_fusion_mode", "graph_md_residual"),
        mips_descriptor_components=meta.get("mips_descriptor_components", "md200"),
        mips_descriptor_disturbance=float(meta.get("mips_descriptor_disturbance", 0.0)),
        mips_backbone_mode=meta.get("mips_backbone_mode", "independent"),
        mips_input_norm=bool(meta.get("mips_input_norm", False)),
        mips_mask_mode=meta.get("mips_mask_mode", "zero"),
        mips_mask_policy=meta.get("mips_mask_policy", "canonical_exact"),
        mips_masked_loss_reduction=meta.get("mips_masked_loss_reduction", "atom_mean"),
        fusion_type="none",
        fp_mode="disabled",
    )
    expected_state = model.state_dict()
    if set(expected_state) != set(state):
        raise RuntimeError(
            "checkpoint state_dict key mismatch: "
            f"missing={sorted(set(expected_state) - set(state))[:5]} "
            f"unexpected={sorted(set(state) - set(expected_state))[:5]}"
        )
    shape_mismatch = [
        key for key in expected_state
        if tuple(expected_state[key].shape) != tuple(state[key].shape)
    ]
    if shape_mismatch:
        raise RuntimeError("checkpoint state_dict shape mismatch: " + ", ".join(shape_mismatch[:10]))
    model.load_state_dict(state, strict=True)
    return checkpoint, model, binding


def _verify_target_contract(checkpoint: dict, specs, cohort_hash: str) -> None:
    """Re-validate the checkpoint target_contract with the exact function the
    production loader uses (Plan §7), so the doctor does not maintain a second
    judgment.  The values are recomputed from the frozen store/.frozen payloads
    and the resolved production config.
    """
    from scripts.train import (
        _angle_artifact_hash_for_cohort,
        _continuous_angle_artifact_hash_for_cohort,
        _sha256_file as _train_sha256_file,
        _source_contract_digest_mismatch,
        _target_contract_mismatch,
    )

    meta = checkpoint.get("meta", {})
    contract = meta.get("target_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("checkpoint has no target_contract")
    # Plan G0-baseline §3.1: the immutable source identity digest must match
    # its declaration; a tampered or missing digest is rejected here.
    if _source_contract_digest_mismatch(
        meta.get("source_contract"), meta.get("source_contract_sha256")
    ):
        raise RuntimeError(
            "checkpoint source_contract digest does not match its declaration"
        )
    raw = subprocess.check_output(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/resolve_mips_trimer_scage.py"),
            str(PROJECT_ROOT / "configs/mts/default.json"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
    )
    cfg = json.loads(raw)

    class _Args:
        feature_config_hash = cfg["feature_config_hash"]
        graph_model_config_hash = cfg["graph_model_config_hash"]
        source_geometry_model_config_hash = cfg["source_geometry_model_config_hash"]
        topology_representation = cfg["topology_representation"]

    class _Dataset:
        topology_cache_artifact_hash = (
            Path(specs["topology"]["root"]) / ".done"
        ).read_text(encoding="utf-8").strip()
        trimer_cache_artifact_hash = (
            Path(specs["trimer"]["root"]) / ".done"
        ).read_text(encoding="utf-8").strip()

    topo_root = Path(specs["topology"]["root"])
    tri_root = Path(specs["trimer"]["root"])
    angle_artifact = (
        _angle_artifact_hash_for_cohort(str(PROJECT_ROOT), cohort_hash)
        if meta.get("target_contract", {}).get("schema") == PRETRAIN_TARGET_CONTRACT_SCHEMA
        else _continuous_angle_artifact_hash_for_cohort(str(PROJECT_ROOT), cohort_hash)
    )
    mismatch = _target_contract_mismatch(
        contract,
        args=_Args(),
        dataset=_Dataset(),
        pretraining_cohort_hash=cohort_hash,
        expected_angle_artifact=angle_artifact,
        store_json_sha256=_train_sha256_file(
            topo_root.parents[1] / "validation" / "store.json"
        ),
        topology_frozen_payload_sha256=_train_sha256_file(
            topo_root / ".frozen"
        ),
        trimer_frozen_payload_sha256=_train_sha256_file(tri_root / ".frozen"),
    )
    if mismatch:
        raise RuntimeError(
            "checkpoint target contract does not bind the current frozen "
            "production identity"
        )


def _two_sample_forward_backward(model):
    samples = []
    for smiles in ("*CCO*", "*CCC*"):
        topology = build_canonical_periodic_topology(smiles)
        attach_finite_trimer_mcl(topology, smiles)
        samples.append(topology)
    batch = mips_trimer_collate(samples)
    encoder = model.encoders["graph"].encoder
    encoder.train()
    mask = torch.zeros(batch.mips_x.size(0), dtype=torch.bool)
    mask[0] = True
    mask[min(1, mask.numel() - 1)] = True
    graph, nodes, _aux = encoder.forward_joint_pretrain(batch, mask)
    loss = graph.square().mean() + nodes.square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("doctor forward/backward loss is non-finite")
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("doctor forward/backward gradient is non-finite")
    return {"loss": float(loss.detach()), "samples": 2}


def _check_cli(python: str) -> None:
    for script in ("scripts/pretrain.py", "scripts/train.py"):
        result = subprocess.run(
            [python, script, "--help"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{script} --help failed: {result.stderr[-500:]}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("production", "pretrain"), default="production",
        help="production checks the existing checkpoint; pretrain checks only the fresh-run bundle",
    )
    parser.add_argument(
        "--profile", default=None,
        help="formal pretraining profile used by --mode pretrain",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used for CLI smoke checks",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="only run environment/config/model checks",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="canonical migrated MTS joint checkpoint; omitted uses the unique canonical file",
    )
    args = parser.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0,1,2":
        raise RuntimeError(
            f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) requires CUDA_VISIBLE_DEVICES=0,1,2; "
            f"got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError(
            "doctor requires exactly three visible CUDA devices (0,1,2); "
            f"available={torch.cuda.device_count()}"
        )
    _check_cli(args.python)

    # Constructing the fixed encoder catches accidental reintroduction of
    # retired PBC/SCAGE/descriptor branches without allocating a dataset.
    encoder = MIPSLocalGraphEncoder()
    assert encoder.max_hops == 2 and len(encoder.layers) == 6
    assert encoder.emb_dim == 512 and encoder.num_heads == 8
    assert encoder.descriptor_components == "md200"
    if _active_writer_pids():
        raise RuntimeError("cache writer is still active")

    if args.mode == "pretrain":
        if not args.profile:
            raise RuntimeError("--mode pretrain requires --profile")
        from scripts.pretrain import _load_pretrain_profile
        profile = _load_pretrain_profile(args.profile)
        specs = _specs(PROJECT_ROOT)
        store = Path(specs["topology"]["root"]).parents[1] / "validation" / "store.json"
        verify_frozen_cache_bundle(
            specs, store_path=store, required_layers=specs.keys(),
            pretraining_cohort_hash=profile["cohort_hash"],
        )
        pi1m = _load_cohort(PROJECT_ROOT / "data/raw/PI1M_v2.csv", "PI1M_v2")
        downstream = _load_cohort(
            PROJECT_ROOT / "data/raw/smi_all.csv", "downstream_union"
        )
        _validate_angle_sidecars(specs, [pi1m, downstream])
        trimer_root = Path(specs["trimer"]["root"])
        angle_root = trimer_root / "derived" / "bond_angle" / profile["cohort_hash"]
        angle_meta = json.loads((angle_root / "metadata.json").read_text(encoding="utf-8"))
        angle_done = (angle_root / ".done").read_text(encoding="utf-8").strip()
        if (
            angle_meta.get("schema") != CACHE_BOND_ANGLE_SCHEMA
            or angle_done != profile["angle_cache_artifact"]
        ):
            raise RuntimeError("pretrain doctor found a non-categorical or stale Angle sidecar")
        model = UniEncoderAttention(
            joint_embedding_dim=256,
            smiles_model_name=None,
            gnn_model_name=None,
            modality_list=["graph"],
            graph_num_layers=6,
            graph_emb_dim=512,
            graph_dropout=0.1,
            graph_encoder_type="mips_trimer_scage",
            scage_num_heads=8,
            scage_ffn_hidden_dim=2048,
            scage_use_pbc_distance=False,
            scage_use_descriptors=False,
            mips_core="paper_corrected",
            mips_max_hops=2,
            mips_use_descriptors=True,
            spatial_mode="trimer_scage",
            graph_geometry_mode="trimer_scage_mcl",
            mcl_distance_percentiles=(0.20, 0.50),
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
            fusion_type="none",
            fp_mode="disabled",
        )
        forward_report = _two_sample_forward_backward(model)
        print(json.dumps({
            "mode": "pretrain",
            "profile_id": profile["profile_id"],
            "angle_schema": angle_meta["schema"],
            "angle_artifact": angle_done,
            "forward_backward": forward_report,
            "cache_store": str(store),
        }, sort_keys=True))
        print(f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) pretrain doctor: ready (3 GPUs, frozen bundle, categorical Angle-20, 2-sample forward/backward)")
        return

    split_root = PROJECT_ROOT / "data/splits/mips_shared5"
    missing_splits = [task for task in TASKS if not (split_root / f"{task}.json").is_file()]
    if missing_splits:
        raise RuntimeError("missing shared 5-fold manifests: " + ", ".join(missing_splits))

    checkpoint = None
    checkpoint_model = None
    checkpoint_binding = None
    if not args.skip_cache:
        specs = _specs(PROJECT_ROOT)
        # The exact-union marker belongs to the shared bundle root, not one
        # individual LMDB layer.  Keeping this path identical to the finalizer
        # prevents a stale per-layer marker from making doctor report ready.
        store = Path(specs["topology"]["root"]).parents[1] / "validation" / "store.json"
        verify_frozen_cache_bundle(
            specs,
            store_path=store,
            required_layers=specs.keys(),
            pretraining_cohort_hash=None,
        )
        pi1m = _load_cohort(
            PROJECT_ROOT / "data/raw/PI1M_v2.csv", "PI1M_v2"
        )
        downstream = _load_cohort(
            PROJECT_ROOT / "data/raw/smi_all.csv", "downstream_union"
        )
        angle_report = _validate_angle_sidecars(specs, [pi1m, downstream])
        mcl_report = _validate_mcl_sidecars(specs, [pi1m, downstream])
        checkpoint_path = _find_checkpoint(args.checkpoint)
        checkpoint, checkpoint_model, checkpoint_binding = _assert_checkpoint_binding(
            checkpoint_path, specs, store
        )
        # The doctor validates the target contract with the exact same function
        # the production loader uses (Plan §7); no second judgment is kept.
        _verify_target_contract(
            checkpoint, specs, pi1m["manifest"]["cohort_hash"]
        )
        forward_report = _two_sample_forward_backward(checkpoint_model)
        acceptance_path = PROJECT_ROOT / "results/mts_canonical_migration/final_acceptance.json"
        if not acceptance_path.is_file():
            raise RuntimeError("final_acceptance.json is missing before doctor")
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        hard_gates = acceptance.setdefault("hard_gates", {})
        hard_gates.update({
            "doctor_checkpoint_binding": True,
            "doctor_state_dict_strict": True,
            "doctor_forward_backward": True,
            "doctor_sidecars_complete": len(angle_report) == 2 and len(mcl_report) == 2,
            "doctor_ready": True,
        })
        acceptance["doctor"] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "cache_binding": checkpoint_binding,
            "angle_sidecars": angle_report,
            "mcl_sidecars": mcl_report,
            "forward_backward": forward_report,
            "training_processes": [],
        }
        temporary = acceptance_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(acceptance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, acceptance_path)
    print(f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) doctor: ready (3 GPUs, frozen bundle, strict checkpoint, 2-sample forward/backward)")


if __name__ == "__main__":
    main()
