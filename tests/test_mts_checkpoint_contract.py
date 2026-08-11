"""Checkpoint dual-contract and target-contract validation tests.

Plan contract-hardening §4.3: the production loader must reject a checkpoint
whose ``target_contract`` does not bind the current frozen production identity,
and must reject any attempt to mutate a fixed model configuration.  These tests
drive the pure validation functions directly (no GPU, no Dataset construction).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.dataset.lmdb_cache import (
    _tensor_values_equal,
    _validate_trimer_placeholder_override,
)

from src.dataset.mips_trimer_contract import (
    CACHE_BUNDLE_SCHEMA,
    CACHE_LAYOUT_SCHEMA,
    CACHE_TOPOLOGY_COST_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    FEATURE_SCHEMA,
    TARGET_CONTRACT_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA,
    TRIMER_BUILDER_VERSION,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_PROTOCOL,
    build_target_contract,
    cache_bundle_binding_hash,
    validate_runtime_args,
)
import scripts.train as T


def _args(**overrides):
    base = dict(
        graph_encoder_type="mips_trimer_scage",
        config_schema="mts-experiment-v3",
        mips_core="paper_corrected",
        mips_variant="O8",
        mips_max_hops=2,
        mips_atom_feature_mode="mips137",
        mips_attention_scale="head_dim",
        mips_norm_mode="post",
        mips_activation="relu",
        mips_spd_bias_mode="per_head",
        mips_path_bias_mode="per_head_single_path_node",
        mips_descriptor_components="md200",
        mips_descriptor_protocol="source_star_sub",
        mips_descriptor_fusion_mode="graph_md_residual",
        spatial_mode="trimer_scage",
        mips_fusion_mode="none",
        projection_mode="plain",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_use_descriptors=True,
        modalities=["graph"],
        fusion_type="none",
        graph_geometry_mode="trimer_scage_mcl",
        mcl_distance_percentiles=(0.20, 0.50),
        scage_use_pbc_distance=False,
        scage_use_descriptors=False,
        feature_config_hash="f" * 64,
        graph_model_config_hash="g" * 64,
        source_geometry_model_config_hash="s" * 64,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _dataset(**overrides):
    class _D:
        topology_cache_artifact_hash = "t" * 64
        trimer_cache_artifact_hash = "r" * 64

    ds = _D()
    for key, value in overrides.items():
        setattr(ds, key, value)
    return ds


def _valid_contract(args, dataset, cohort_hash="c" * 64, angle_artifact="a" * 64):
    return build_target_contract(
        source_cohort_hash=cohort_hash,
        feature_config_hash=args.feature_config_hash,
        graph_model_config_hash=args.graph_model_config_hash,
        geometry_model_config_hash=args.source_geometry_model_config_hash,
        topology_cache_artifact_hash=dataset.topology_cache_artifact_hash,
        trimer_cache_artifact_hash=dataset.trimer_cache_artifact_hash,
        angle_cache_artifact_hash=angle_artifact,
        cache_bundle_hash=cache_bundle_binding_hash(
            cohort_hash=cohort_hash,
            topology_artifact_hash=dataset.topology_cache_artifact_hash,
            trimer_artifact_hash=dataset.trimer_cache_artifact_hash,
        ),
        store_json_sha256="s" * 64,
        topology_frozen_payload_sha256="t" * 64,
        trimer_frozen_payload_sha256="r" * 64,
        optimizer_steps=20000,
        pretraining_objective="masked_atom_plus_trimer_bond_angle",
    )


def _mismatch(contract, args=None, dataset=None, cohort_hash="c" * 64,
              angle_artifact="a" * 64):
    args = args or _args()
    dataset = dataset or _dataset()
    return T._target_contract_mismatch(
        contract,
        args=args,
        dataset=dataset,
        pretraining_cohort_hash=cohort_hash,
        expected_angle_artifact=angle_artifact,
        store_json_sha256="s" * 64,
        topology_frozen_payload_sha256="t" * 64,
        trimer_frozen_payload_sha256="r" * 64,
    )


def test_target_contract_schema_and_constants_are_active():
    # build_target_contract emits exactly the active contract constants.
    args = _args()
    ds = _dataset()
    contract = _valid_contract(args, ds)
    assert contract["schema"] == TARGET_CONTRACT_SCHEMA
    assert contract["config_schema"] == CONFIG_SCHEMA
    assert contract["feature_schema"] == FEATURE_SCHEMA
    assert contract["cache_layout_schema"] == CACHE_LAYOUT_SCHEMA
    assert contract["cache_bundle_schema"] == CACHE_BUNDLE_SCHEMA
    assert contract["topology_lmdb_schema"] == TOPOLOGY_LMDB_SCHEMA
    assert contract["trimer_content_schema"] == TRIMER_CONTENT_SCHEMA
    assert contract["trimer_lmdb_schema"] == TRIMER_LMDB_SCHEMA
    assert contract["trimer_builder_version"] == TRIMER_BUILDER_VERSION
    assert contract["canonical_lga_schema_version"] == CANONICAL_LGA_SCHEMA_VERSION
    assert contract["trimer_protocol"] == TRIMER_PROTOCOL
    assert contract["checkpoint_schema"] == CHECKPOINT_SCHEMA


def test_target_contract_valid_binds():
    args = _args()
    ds = _dataset()
    contract = _valid_contract(args, ds)
    assert _mismatch(contract) is False


@pytest.mark.parametrize("key,value", [
    ("canonical_lga_schema_version", 1),          # canonical LGA schema change
    ("trimer_builder_version", 11),               # Trimer builder version change
    ("topology_cache_artifact_hash", "0" * 64),   # Topology artifact change
    ("trimer_cache_artifact_hash", "0" * 64),     # Trimer artifact change
    ("cache_bundle_hash", "0" * 64),              # cache store hash change
    ("feature_config_hash", "0" * 64),            # feature config hash change
    ("graph_model_config_hash", "0" * 64),        # graph model hash change
    ("source_cohort_hash", "0" * 64),             # cohort hash change
    ("optimizer_steps", 19999),                   # optimizer steps change
    ("angle_cache_artifact_hash", "0" * 64),      # angle cache artifact change
    ("trimer_content_schema", "wrong"),           # Trimer content schema change
    ("experiment_config_schema", "mts-experiment-v2"),  # experiment config schema
    ("angle_cache_schema", "mts-trimer-bond-angle-cache-v2"),  # angle cache schema
    ("store_json_sha256", "0" * 64),              # frozen store SHA change
    ("topology_frozen_payload_sha256", "0" * 64), # topology .frozen change
    ("trimer_frozen_payload_sha256", "0" * 64),   # trimer .frozen change
    ("target_contract_sha256", "0" * 64),         # digest self-exclusion
])
def test_target_contract_mutated_rejected(key, value):
    args = _args()
    ds = _dataset()
    contract = _valid_contract(args, ds)
    contract[key] = value
    assert _mismatch(contract) is True


@pytest.mark.parametrize("key,value", [
    ("mips_attention_scale", "none"),      # attention scaling
    ("mips_norm_mode", "pre"),             # norm mode
    ("mips_activation", "gelu"),           # activation
    ("mips_spd_bias_mode", "none"),        # SPD bias mode
    ("mips_path_bias_mode", "none"),       # path bias mode
])
def test_model_config_mutation_rejected(key, value):
    args = _args(**{key: value})
    with pytest.raises(ValueError):
        validate_runtime_args(args)


def test_valid_model_config_passes_runtime_validation():
    validate_runtime_args(_args())  # must not raise


def test_source_contract_preserves_legacy_identity():
    # The dual-contract metadata keeps the pre-migration identity verbatim.
    source = {
        "schema": "mts-model-v2",
        "feature_config_hash": "old-feature",
        "graph_model_config_hash": "old-graph",
        "source_geometry_model_config_hash": "old-geometry",
        "optimizer_steps": 20000,
        "pretraining_objective": "masked_atom_plus_trimer_bond_angle",
    }
    migrated = dict(source)
    migrated["source_contract"] = dict(source)
    migrated["target_contract"] = _valid_contract(_args(), _dataset())
    # The source identity is preserved, never rewritten to the current schema.
    assert migrated["source_contract"]["schema"] == "mts-model-v2"
    assert migrated["source_contract"]["feature_config_hash"] == "old-feature"
    assert migrated["target_contract"]["schema"] == TARGET_CONTRACT_SCHEMA


# ---------------------------------------------------------------------------
# §5 LMDB duplicate-field merge allowlist.
# ---------------------------------------------------------------------------

def _override(name, topo, tri, n_nodes=2, mask=None):
    return _validate_trimer_placeholder_override(
        name, topo, tri, n_nodes=n_nodes,
        central_ru_mask=mask, sample_key=b"k" * 32,
    )


def _central_mask(size, indices):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[list(indices)] = True
    return mask


def test_allowlist_valid_placeholder_override_passes():
    topo = torch.full((2,), -1, dtype=torch.long)
    tri = torch.tensor([2, 3], dtype=torch.long)
    mask = _central_mask(6, (2, 3))
    assert _override("mips_to_trimer_central_index", topo, tri, mask=mask) is True


def test_unknown_allminusone_field_rejected():
    topo = torch.full((2,), -1, dtype=torch.long)
    tri = torch.tensor([2, 3], dtype=torch.long)
    with pytest.raises(RuntimeError, match="duplicate LMDB cache field"):
        _override("not_allowlisted", topo, tri)


def test_wrong_shape_or_dtype_rejected():
    topo = torch.full((2,), -1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="dtype"):
        _override("mips_to_trimer_central_index", topo,
                  torch.tensor([2.0, 3.0]))
    with pytest.raises(RuntimeError, match="shape"):
        _override("mips_to_trimer_central_index", topo,
                  torch.tensor([2], dtype=torch.long))
    with pytest.raises(RuntimeError, match="shape"):
        _override("mips_to_trimer_central_index", topo,
                  torch.tensor([2, 3, 4], dtype=torch.long))


def test_negative_or_out_of_range_mapping_rejected():
    topo = torch.full((2,), -1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="negative"):
        _override("mips_to_trimer_central_index", topo,
                  torch.tensor([-1, 3], dtype=torch.long))
    mask = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="index range"):
        _override("mips_to_trimer_central_index", topo,
                  torch.tensor([2, 3], dtype=torch.long), mask=mask)


def test_outer_ru_mapping_rejected():
    topo = torch.full((2,), -1, dtype=torch.long)
    tri = torch.tensor([2, 3], dtype=torch.long)
    mask = _central_mask(6, (2,))  # index 3 is outside the central RU
    with pytest.raises(RuntimeError, match="central RU"):
        _override("mips_to_trimer_central_index", topo, tri, mask=mask)


def test_identical_duplicate_field_passes():
    assert _tensor_values_equal(
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
    ) is True


def test_differing_duplicate_field_fails():
    assert _tensor_values_equal(
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1, 3], dtype=torch.long),
    ) is False


# ---------------------------------------------------------------------------
# §6 topology_cost.npy derived-artifact metadata contract.
# ---------------------------------------------------------------------------

def test_topology_cost_metadata_binds_current_artifact():
    import hashlib
    import json
    from pathlib import Path

    from scripts.audit_mips_trimer_cache import _specs

    root = Path(__file__).resolve().parents[1]
    specs = _specs(root)
    cohort_hash = "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049"
    meta_path = (
        root / "data/processed/mips_trimer_scage/cohorts/PI1M_v2"
        / cohort_hash / "topology_cost_metadata.json"
    )
    cost_path = meta_path.with_name("topology_cost.npy")
    assert meta_path.is_file(), "topology_cost metadata missing"
    assert cost_path.is_file(), "topology_cost array missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    topo_done = (Path(specs["topology"]["root"]) / ".done").read_text(
        encoding="utf-8"
    ).strip()
    file_sha = hashlib.sha256(cost_path.read_bytes()).hexdigest()
    assert meta["schema"] == CACHE_TOPOLOGY_COST_SCHEMA
    assert meta["shape"] == [995799, 2]
    assert meta["dtype"] == "uint32"
    assert meta["file_sha256"] == file_sha
    assert meta["topology_done_artifact_hash"] == topo_done
    assert meta["builder_version"] == TRIMER_BUILDER_VERSION
    assert meta["columns"] == ["node_count", "lga_edge_count"]


# ---------------------------------------------------------------------------
# §7 Real shell dispatcher failure-cleanup test.
# ---------------------------------------------------------------------------

_FAKE_TRAIN_SRC = '''\
#!/usr/bin/env python
"""Fake MTS finetune command driving the real shell dispatcher (Plan §8).

Every unit appends a monotonic ``START|task|fold`` / ``OK|task|fold`` /
``FAIL|task|fold`` line to a shared event file (O_APPEND), so the test can
assert the dispatcher stops starting new units after it observes a FAIL.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--fold")
    p.add_argument("--shard")
    p.add_argument("--prediction")
    a = p.parse_args()
    event_path = os.environ.get("MTS_FAKE_EVENT_FILE")

    def emit(kind):
        if event_path:
            with open(event_path, "a", encoding="utf-8") as handle:
                handle.write(f"{kind}|{a.task}|{a.fold}\\n")

    emit("START")
    time.sleep(float(os.environ.get("MTS_FAKE_DURATION_SEC", "0.2")))
    fail = os.environ.get("MTS_FAKE_FAIL_UNIT", "")
    if fail == f"{a.task}|{a.fold}":
        emit("FAIL")
        print(f"[fake-train] FAIL {a.task}|{a.fold}", flush=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(a.shard), exist_ok=True)
    os.makedirs(os.path.dirname(a.prediction), exist_ok=True)
    np.savez(
        a.prediction,
        y_true=np.array([0.0, 1.0]),
        y_pred=np.array([0.0, 1.0]),
        metadata=np.array(json.dumps(
            {"task": a.task, "fold": int(a.fold), "seed": 42}
        )),
    )
    tmp = a.shard + ".tmp"
    pd.DataFrame([{"task": a.task, "seed": 42, "fold": int(a.fold)}]).to_csv(
        tmp, index=False
    )
    os.replace(tmp, a.shard)  # atomic publish: only a renamed shard is complete
    emit("OK")
    print(f"[fake-train] OK {a.task}|{a.fold}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
'''


def test_shell_dispatcher_failure_stops_and_cleans_up(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "fake_train.py"
    fake.write_text(_FAKE_TRAIN_SRC, encoding="utf-8")
    results = tmp_path / "results"
    logs = tmp_path / "logs"
    event_file = tmp_path / "events.log"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root),
        "EXPERIMENT_CONFIG": "configs/mts/experiments/G0_canonical_angle20_v1.json",
        "FINETUNE_ONLY": "1",
        "RESUME": "0",
        "RANDOM_SEED": "42",
        "FINETUNE_SEEDS": "42",
        "TASKS": "eat egb xc",
        "FOLD_IDS": "0 1 2 3 4",
        "DATALOADER_WORKERS": "0",
        "MTS_FAKE_TRAIN": "1",
        "MTS_FAKE_TRAIN_CMD": str(fake),
        "MTS_FAKE_FAIL_UNIT": "egb|2",
        "MTS_FAKE_DURATION_SEC": "0.2",
        "MTS_FAKE_EVENT_FILE": str(event_file),
        "JOINT_CKPT": str(root / "pretrained_models/mts"
                          / "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"),
        "RESULTS_DIR": str(results),
        "LOG_DIR": str(logs),
    })
    proc = subprocess.run(
        ["bash", "scripts/run_mts.sh"], cwd=root, env=env,
        capture_output=True, text=True, timeout=180,
    )
    out = proc.stdout + "\n" + proc.stderr

    # 1. A child failure makes the whole dispatcher exit non-zero.
    assert proc.returncode != 0, out
    # 2. The failing unit is reported and dispatch stops.
    assert "failed unit=42|egb|2" in out, out
    assert "stopping all slots" in out, out
    # 3. At least one unit completed before the failure.
    assert "completed unit=" in out, out
    # 4. An atomically completed shard is preserved.
    assert (results / "shards/42/eat/fold_0.csv").is_file(), out
    # 5. The failed unit has no complete shard.
    assert not (results / "shards/42/egb/fold_2.csv").exists(), out
    # 6. Monotonic event order: three concurrent slots were filled, and after
    # the dispatcher observed FAIL no further unit is STARTed (queued units
    # stay unstarted) and the failing unit itself reached FAIL.
    events = []
    if event_file.is_file():
        for line in event_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) == 3:
                events.append(tuple(parts))
    assert len(events) >= 4, out
    assert sum(1 for e in events if e[0] == "START") >= 3, out
    assert ("FAIL", "egb", "2") in events, out
    fail_index = next(
        i for i, e in enumerate(events) if e[0] == "FAIL"
    )
    assert not any(
        e[0] == "START" for e in events[fail_index + 1:]
    ), "a unit was started after the dispatcher observed FAIL"
    # 7. No residual fake-train process is left running after cleanup.
    check = subprocess.run(
        ["pgrep", "-f", "fake_train"], capture_output=True, text=True
    )
    assert check.returncode != 0, "residual fake-train processes remain"


# ---------------------------------------------------------------------------
# §3 (contract-g0-baseline) source-contract digest and fixed angle schema.
# ---------------------------------------------------------------------------

def _source_digest(value):
    from src.dataset.mips_trimer_contract import _canonical_json_hash
    return _canonical_json_hash(value)


def test_source_contract_digest_valid_matches():
    source = {"schema": "mts-model-v2", "feature_config_hash": "f" * 64}
    declared = _source_digest(source)
    assert T._source_contract_digest_mismatch(source, declared) is False


def test_source_contract_digest_tampered_rejected():
    source = {"schema": "mts-model-v2", "feature_config_hash": "f" * 64}
    declared = _source_digest(source)
    # content tampered
    tampered = dict(source)
    tampered["feature_config_hash"] = "e" * 64
    assert T._source_contract_digest_mismatch(tampered, declared) is True
    # digest tampered
    assert T._source_contract_digest_mismatch(source, "0" * 64) is True


def test_source_contract_digest_missing_rejected():
    source = {"schema": "mts-model-v2"}
    assert T._source_contract_digest_mismatch(source, None) is True
    assert T._source_contract_digest_mismatch(None, None) is True


def test_angle_schema_switched_to_categorical_rejected():
    args = _args()
    ds = _dataset()
    contract = _valid_contract(args, ds)
    for bad_schema in ("mts-trimer-bond-angle-cache-v1",
                       "mts-trimer-bond-angle-cache-v2"):
        variant = dict(contract)
        variant["angle_cache_schema"] = bad_schema
        assert _mismatch(variant) is True


def test_angle_schema_artifact_digest_coordinated_tamper_rejected():
    args = _args()
    ds = _dataset()
    contract = _valid_contract(args, ds)
    contract["angle_cache_schema"] = "mts-trimer-bond-angle-cache-v2"
    contract["angle_cache_artifact_hash"] = "0" * 64
    contract["target_contract_sha256"] = "0" * 64
    assert _mismatch(contract) is True


# ---------------------------------------------------------------------------
# Angle-20 G0 campaign identity and g0-only entry.
# ---------------------------------------------------------------------------

def test_g0_canonical_config_is_the_only_active_identity():
    root = Path(__file__).resolve().parents[1]
    canonical = json.loads(
        (root / "configs/mts/experiments/G0_canonical_angle20_v1.json").read_text()
    )
    assert canonical["experiment_id"] == "G0_canonical_angle20_v1"
    assert not (root / "configs/mts/experiments/G0_current_mcl.json").exists()
    assert not (root / "configs/mts/experiments/G0_canonical_contract.json").exists()
    import scripts.run_mts_sota_campaign as campaign
    assert campaign.PHASES["G"] == (
        "G0_canonical_angle20_v1",
        "G1_mcl_disabled",
        "G2_coordinate_shuffled",
        "G3_mcl_rbf",
        "G4_mcl_rbf_shuffled",
    )


def _campaign(args):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "scripts/run_mts_sota_campaign.py"] + args,
        cwd=root, capture_output=True, text=True, timeout=180,
    )


def test_campaign_g0_only_dry_run_shows_only_canonical():
    proc = _campaign([
        "screen", "--phase", "G", "--g0-only", "--dry-run",
    ])
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "G0_canonical_angle20_v1" in out, out
    assert "dry-run checkpoint" in out, out
    assert (
        "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
        in out
    ), out
    assert "G1_mcl_disabled" not in out, out
    assert "G0_canonical_contract" not in out, out


def test_campaign_g0_only_requires_phase_G():
    proc = _campaign([
        "screen", "--phase", "V", "--g0-only", "--dry-run",
    ])
    assert proc.returncode != 0
    assert "only valid with" in proc.stderr


def test_campaign_partial_folds_rejected():
    proc = _campaign([
        "screen", "--phase", "G", "--folds", "0", "1", "--g0-only", "--dry-run",
    ])
    assert proc.returncode != 0
    assert "exactly folds" in proc.stderr


def test_campaign_reuse_g0_requires_canonical_identity():
    # Reuse is allowed only after the immutable checkpoint and all 40 current
    # G0 shards have passed the canonical identity checks.
    proc = _campaign([
        "screen", "--phase", "G", "--g0-only", "--reuse-g0",
    ])
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "G0_canonical_angle20_v1 reuse identity verified" in out, out


def test_campaign_phase_g_excludes_historical_g0_identities():
    import scripts.run_mts_sota_campaign as campaign

    assert campaign.G0_CANONICAL == "G0_canonical_angle20_v1"
    assert campaign.G0_CANONICAL_CHECKPOINT_SHA256 == (
        "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
    )
    assert campaign.G0_CANONICAL not in {
        campaign.G0_LEGACY,
        campaign.G0_HISTORICAL_CANONICAL,
    }
    assert campaign.G0_LEGACY not in campaign.PHASES["G"]
    assert campaign.G0_HISTORICAL_CANONICAL not in campaign.PHASES["G"]


def test_campaign_removed_g0_drift_override():
    proc = _campaign([
        "screen", "--phase", "G", "--g0-only", "--accept-g0-drift", "--dry-run",
    ])
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr
