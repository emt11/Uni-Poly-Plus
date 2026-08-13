"""Dual-cohort G-family identity tests using frozen read-only artifacts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.resolve_mips_trimer_scage import (
    _resolve_g_family_artifact_bundle,
    g_family_bundle_identity_hash,
)
from scripts.train import (
    select_mts_checkpoint_transfer_keys,
    validate_g_family_checkpoint_binding,
)
from src.dataset.mts_relation_geometry import (
    RelationGeometryPermutation,
    RelationGeometrySidecar,
)


ROOT = Path(__file__).resolve().parents[1]
PI_REL = "data/processed/mips_trimer_scage/relation_geometry/ef70d42497bc15f51fc38728f1400aa73ea1431f0f8cbf10c310b0a784e36bf1/PI1M_v2/0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049"
DOWN_REL = "data/processed/mips_trimer_scage/relation_geometry/07b9f19f85f50b878e91041f4ace8dbc16a9769b1aa668947fec37a9e9efb123/downstream_union/ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2"
PI_PERM = "data/processed/mips_trimer_scage/relation_geometry_permutation/g3_seed42/PI1M_v2/0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049"
DOWN_PERM = "data/processed/mips_trimer_scage/relation_geometry_permutation/g3_seed42/downstream_union/ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2"
PI_REL_HASH = "134016a47280e61fdc4154047b1fb48a9c8a27f1ac9ed3b8ee9abd166860a289"
DOWN_REL_HASH = "25c9524a07a1bd098c291c51fab2ee7a5d488871259335dd61a26fd46e71e0cb"
PI_PERM_HASH = "9defaec6593ba2ed15dfe17b5998602edbae2e5143ebdc2b12f09045623ff9d4"
DOWN_PERM_HASH = "812735b743fadad140fd606ed1b07b5f0013d1f29a4c76ee815d803ae5a3fde7"
G1_CHECKPOINT = ROOT / "results/mts_multiscale_topology/g_family_dual_cohort_repair_v1/g1_pretrain_smoke.pth"
STEP0_ROOT = ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2"


def _bundle(pi_root, pi_hash, down_root, down_hash):
    return {
        "PI1M_v2": {"root": pi_root, "artifact_hash": pi_hash},
        "downstream_union": {"root": down_root, "artifact_hash": down_hash},
    }


def test_real_dual_cohort_bundle_and_g3_permutation_are_strict():
    relation = _resolve_g_family_artifact_bundle(
        _bundle(PI_REL, PI_REL_HASH, DOWN_REL, DOWN_REL_HASH),
        kind="relation_geometry",
    )
    permutation = _resolve_g_family_artifact_bundle(
        _bundle(PI_PERM, PI_PERM_HASH, DOWN_PERM, DOWN_PERM_HASH),
        kind="g3_permutation",
        relation_bundle=relation,
    )
    assert relation["PI1M_v2"]["artifact_hash"] == PI_REL_HASH
    assert permutation["downstream_union"]["artifact_hash"] == DOWN_PERM_HASH
    assert len(g_family_bundle_identity_hash("g3", "g3", relation, permutation)) == 64


def test_swapped_cohort_or_wrong_hash_is_rejected():
    with pytest.raises(ValueError, match="identity mismatch"):
        _resolve_g_family_artifact_bundle(
            _bundle(DOWN_REL, DOWN_REL_HASH, PI_REL, PI_REL_HASH),
            kind="relation_geometry",
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        _resolve_g_family_artifact_bundle(
            _bundle(PI_REL, "0" * 64, DOWN_REL, DOWN_REL_HASH),
            kind="relation_geometry",
        )


def test_active_sidecar_readers_reject_cross_cohort_and_hash():
    RelationGeometrySidecar(
        ROOT / PI_REL, expected_cohort="PI1M_v2",
        expected_artifact_hash=PI_REL_HASH, verify_array_hashes=False,
    )
    with pytest.raises(RuntimeError, match="cohort name mismatch"):
        RelationGeometrySidecar(
            ROOT / PI_REL, expected_cohort="downstream_union",
            expected_artifact_hash=PI_REL_HASH, verify_array_hashes=False,
        )
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        RelationGeometrySidecar(
            ROOT / PI_REL, expected_cohort="PI1M_v2",
            expected_artifact_hash="0" * 64, verify_array_hashes=False,
        )
    RelationGeometryPermutation(
        ROOT / PI_PERM, source_sidecar=PI_REL_HASH,
        expected_artifact_hash=PI_PERM_HASH,
    )


def test_checkpoint_binding_uses_bundle_not_active_root():
    args = SimpleNamespace(
        g_family_arm="g1",
        shared_step0_id="mts_g_family_step0_v2_seed42",
        g_family_bundle_hash="a" * 64,
        relation_geometry_sidecar=DOWN_REL,
    )
    meta = {
        "g_family_arm": "g1",
        "topology_attention_variant": "msta_last2",
        "shared_step0_id": args.shared_step0_id,
        "pretraining_objective": "masked_atom_only",
        "g_family_bundle_hash": args.g_family_bundle_hash,
        "relation_geometry_sidecar": PI_REL,
        "relation_geometry_artifact_hash": PI_REL_HASH,
        "optimizer_steps": 2,
        "smoke_only": True,
    }
    validate_g_family_checkpoint_binding(meta, args, allow_smoke=True)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_g_family_checkpoint_binding(
            {**meta, "g_family_bundle_hash": "b" * 64}, args, allow_smoke=True
        )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_g_family_checkpoint_binding(
            {**meta, "g_family_arm": "g2"}, args, allow_smoke=True
        )


def test_g1_checkpoint_transfer_includes_msta_geometry_and_resets_heads():
    payload = torch.load(G1_CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_state = payload["state_dict"]
    model_keys = set(checkpoint_state)
    transfer_keys = select_mts_checkpoint_transfer_keys(model_keys, checkpoint_state)
    assert len(transfer_keys) == 116
    assert any("layers.4.attention.local_output" in key for key in transfer_keys)
    assert any("layers.5.attention.local_output" in key for key in transfer_keys)
    geometry_keys = {
        key for key in transfer_keys if "relation_geometry_bias." in key
    }
    assert len(geometry_keys) == 6
    assert not any(".md_residual." in key for key in transfer_keys)
    assert not any(
        key.startswith("encoders.graph.norm.")
        or key.startswith("encoders.graph.projection.")
        or key.startswith("mlp.")
        for key in transfer_keys
    )

    # Simulate a freshly initialized fold: only the declared topology keys are
    # merged, and the MD200/projection/head tensors retain their fold values.
    fold_state = {key: torch.zeros_like(value) for key, value in checkpoint_state.items()}
    merged_state = dict(fold_state)
    merged_state.update({key: checkpoint_state[key] for key in transfer_keys})
    for key in transfer_keys:
        assert torch.equal(merged_state[key], checkpoint_state[key])
    reset_keys = set(model_keys) - set(transfer_keys)
    assert reset_keys
    assert all(torch.equal(merged_state[key], fold_state[key]) for key in reset_keys)

    missing_geometry = set(checkpoint_state) - {
        next(key for key in geometry_keys if "relation_projection" in key)
    }
    with pytest.raises(RuntimeError, match="missing transferable topology tensors"):
        select_mts_checkpoint_transfer_keys(model_keys, missing_geometry)

    optional_mcl_v2 = "encoders.graph.encoder.trimer_mcl.layers.0.distance_centers"
    optional_model_keys = set(model_keys) | {optional_mcl_v2}
    selected_with_optional_gap = select_mts_checkpoint_transfer_keys(
        optional_model_keys,
        model_keys,
        allowed_missing={optional_mcl_v2},
    )
    assert optional_mcl_v2 not in selected_with_optional_gap


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def test_v2_step0_is_shared_and_binds_each_g_bundle():
    manifest = json.loads((STEP0_ROOT / "metadata.json").read_text())
    assert manifest["schema"] == "mts-g-family-full-step0-v2"
    assert manifest["shared_step0_id"] == "mts_g_family_step0_v2_seed42"
    assert manifest["common_state_hash"] == "28f61aa90d547d2df8cf36cbe06dc39f28ca3ab4cef06393e10c5f6e08bb620d"
    for key in ("optimizer_inherited", "scheduler_inherited", "sampler_inherited", "rng_inherited"):
        assert manifest[key] is False

    reference_state = None
    for arm in ("g0", "g1", "g2", "g3"):
        path = STEP0_ROOT / f"{arm}_step0.pth"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        assert payload["schema"] == "mts-pretrain-init-v1"
        assert meta["paired_init_id"] == manifest["shared_step0_id"]
        assert meta["shared_step0_id"] == manifest["shared_step0_id"]
        assert meta["g_family_arm"] == arm
        assert meta["geometry_mode"] == arm
        assert meta["topology_attention_variant"] == "msta_last2"
        assert meta["pretraining_objective"] == "masked_atom_only"
        assert meta["angle_loss_weight"] == 0.0
        assert meta["optimizer_steps"] == 0
        for key in ("optimizer_state_inherited", "scheduler_state_inherited", "sampler_state_inherited", "rng_inherited"):
            assert meta[key] is False
        assert meta["g_family_bundle_hash"] == manifest["g_family_bundle_hashes"][arm]
        assert "<" not in json.dumps(meta, sort_keys=True)
        assert ">" not in json.dumps(meta, sort_keys=True)
        assert _state_hash(payload["state_dict"]) == manifest["common_state_hash"]
        if reference_state is None:
            reference_state = payload["state_dict"]
        else:
            assert set(payload["state_dict"]) == set(reference_state)
            assert all(
                torch.equal(payload["state_dict"][key], reference_state[key])
                for key in reference_state
            )
