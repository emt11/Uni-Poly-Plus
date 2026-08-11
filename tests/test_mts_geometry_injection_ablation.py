import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.mips_trimer_contract import (
    ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
    ABLATION_RANDOM_MASK_SCHEMA,
    ABLATION_RANDOM_MASK_SEED,
)
from src.dataset.mts_ablation_random_mask import AblationRandomMaskSidecar
from src.dataset.trimer_mcl import attach_finite_trimer_mcl, attach_unavailable_trimer_mcl
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from src.utils import _build_downstream_optimizer, _configure_legacy_mts_trainability


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/mts/geometry_injection_ablation"
RESOLVER = ROOT / "scripts/resolve_mips_trimer_scage.py"
EXPERIMENTS = (
    "A0_no3d_forward", "A1_star_only", "A2_mcl_real",
    "A3_star_mcl_real", "A4_star_mcl_random_mask",
)


def _resolved(path):
    return json.loads(subprocess.check_output([
        "/root/anaconda3/envs/Uni-Poly/bin/python", str(RESOLVER), str(path)
    ], cwd=ROOT, text=True))


def test_active_configs_have_unique_ablation_identity():
    resolved = [_resolved(CONFIG_ROOT / f"{name}.json") for name in EXPERIMENTS]
    assert len({item["geometry_model_config_hash"] for item in resolved}) == 5
    assert len({item["source_geometry_model_config_hash"] for item in resolved}) == 1
    assert all(item["shared_checkpoint_sha256"] ==
               "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
               for item in resolved)
    assert not list((ROOT / "configs/mts/geometry_causal_ablation").glob("*.json"))


def test_resolver_rejects_unknown_ablation_field(tmp_path):
    config = json.loads((CONFIG_ROOT / "A0_no3d_forward.json").read_text())
    config["ablation"]["angle"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([
        "/root/anaconda3/envs/Uni-Poly/bin/python", str(RESOLVER), str(path)
    ], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "ablation" in (result.stderr + result.stdout)


@pytest.mark.parametrize("flags", [
    (False, False, "real"), (True, False, "real"),
    (False, True, "real"), (True, True, "real"),
    (True, True, "count_matched_random"),
])
def test_encoder_branch_flags_and_constructor(flags):
    use_star, use_mcl, mask_mode = flags
    encoder = MIPSLocalGraphEncoder(
        use_star_rbf=use_star, use_mcl=use_mcl, mcl_mask_mode=mask_mode,
    )
    assert encoder.use_star_rbf is use_star
    assert encoder.use_mcl is use_mcl
    assert encoder.mcl_mask_mode == mask_mode
    assert all(parameter.requires_grad is use_star
               for parameter in encoder.star_distance_bias.parameters())
    assert all(parameter.requires_grad is use_mcl
               for parameter in encoder.trimer_mcl.parameters())


class _DummyEncoder(nn.Module):
    architecture_name = "MIPS-Trimer-SCAGE"

    def __init__(self, *, use_star=True, use_mcl=True):
        super().__init__()
        self.use_star_rbf = use_star
        self.use_mcl = use_mcl
        self.star_distance_bias = nn.Linear(2, 2)
        self.trimer_mcl = nn.Linear(2, 2)
        self.topology = nn.Linear(2, 2)


class _DummyGraphModule(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


class _DummyModel(nn.Module):
    def __init__(self, *, use_star, use_mcl):
        super().__init__()
        self.modality_list = ("graph",)
        self.encoders = nn.ModuleDict({
            "graph": _DummyGraphModule(_DummyEncoder(
                use_star=use_star, use_mcl=use_mcl
            ))
        })
        self.mlp = nn.Linear(2, 1)


@pytest.mark.parametrize("use_star,use_mcl", [(False, False), (True, False), (False, True), (True, True)])
def test_legacy_trainability_preserves_ablation_freeze(use_star, use_mcl):
    model = _DummyModel(use_star=use_star, use_mcl=use_mcl)
    _configure_legacy_mts_trainability(model)
    encoder = model.encoders["graph"].encoder
    assert all(p.requires_grad is use_star for p in encoder.star_distance_bias.parameters())
    assert all(p.requires_grad is use_mcl for p in encoder.trimer_mcl.parameters())
    optimizer = _build_downstream_optimizer(
        model, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 0.02,
        mts_finetune_profile="legacy_mts_huber_v1",
    )
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in optimized for p in encoder.star_distance_bias.parameters()
               if not use_star)
    assert all(id(p) not in optimized for p in encoder.trimer_mcl.parameters()
               if not use_mcl)


def _write_fixture_sidecar(root):
    keys = np.arange(96, dtype=np.uint8).reshape(3, 32)
    payloads = {
        "sample_ptr.npy": np.asarray([0, 1, 1, 2], dtype=np.int64),
        "query_ptr20.npy": np.asarray([0, 1, 2], dtype=np.int64),
        "query_ptr50.npy": np.asarray([0, 1, 2], dtype=np.int64),
        "query_keys20.npy": np.asarray([0, 1], dtype=np.int32),
        "query_keys50.npy": np.asarray([0, 2], dtype=np.int32),
        "sample_keys.npy": keys,
    }
    root.mkdir()
    for name, value in payloads.items():
        np.save(root / name, value)
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in payloads
    }
    done = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {
        "schema": ABLATION_RANDOM_MASK_SCHEMA,
        "payload_version": ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
        "seed": ABLATION_RANDOM_MASK_SEED,
        "cohort_hash": "c" * 64,
        "ordered_key_hash": hashlib.sha256(keys.tobytes()).hexdigest(),
        "record_count": 3,
        "payload_files": hashes,
        "done_payload_sha256": done,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / ".done").write_text(done + "\n")
    return keys


def test_a4_sidecar_is_sample_key_indexed(tmp_path):
    keys = _write_fixture_sidecar(tmp_path / "sidecar")
    sidecar = AblationRandomMaskSidecar(tmp_path / "sidecar")
    assert sidecar.record_for(bytes(keys[1])) is None
    record = sidecar.record_for(bytes(keys[2]))
    assert record["keys20"].tolist() == [1]
    with pytest.raises(TypeError):
        sidecar.record_for(0)


def _a4_item(smiles, invalid=False):
    item = build_canonical_periodic_topology(smiles)
    if invalid:
        attach_unavailable_trimer_mcl(item, "geometry-invalid")
    else:
        attach_finite_trimer_mcl(item, smiles)
    item.mts_ablation_id = "A4_star_mcl_random_mask"
    item.mts_use_star_rbf = True
    item.mts_use_mcl = True
    item.mts_mcl_mask_mode = "count_matched_random"
    item.y = torch.zeros(1)
    n_query = int(item.trimer_central_ru_mask.sum())
    if invalid:
        item.mcl_random_ptr20 = np.zeros(1, dtype=np.int64)
        item.mcl_random_ptr50 = np.zeros(1, dtype=np.int64)
        item.mcl_random_keys20 = np.zeros(0, dtype=np.int32)
        item.mcl_random_keys50 = np.zeros(0, dtype=np.int32)
        item.mcl_random_mask_valid = False
    else:
        item.mcl_random_ptr20 = np.arange(n_query + 1, dtype=np.int64)
        item.mcl_random_ptr50 = np.arange(n_query + 1, dtype=np.int64)
        item.mcl_random_keys20 = np.asarray([q for q in range(n_query)], dtype=np.int32)
        item.mcl_random_keys50 = np.asarray([q for q in range(n_query)], dtype=np.int32)
        item.mcl_random_mask_valid = True
    return item


def test_a4_invalid_geometry_keeps_batch_random_rows_aligned():
    batch = mips_trimer_collate([
        _a4_item("*CCO*"), _a4_item("*CCO*", invalid=True), _a4_item("*CCO*"),
    ])
    assert batch.mcl_query_start.shape == (3,)
    assert batch.mcl_trimer_start.shape == (3,)
    assert batch.mcl_random_mask_valid.tolist() == [True, False, True]
    assert batch.mcl_random_visible20.shape[0] == int(
        sum(int(item.trimer_central_ru_mask.sum()) for item in [
            _a4_item("*CCO*"), _a4_item("*CCO*", invalid=True), _a4_item("*CCO*")
        ])
    )


def test_a0_collate_has_no_geometry_fields():
    item = build_canonical_periodic_topology("*CCO*")
    item.mts_ablation_id = "A0_no3d_forward"
    item.mts_use_star_rbf = False
    item.mts_use_mcl = False
    item.mts_mcl_mask_mode = "real"
    item.mips_md = torch.zeros(200)
    item.mips_md_valid = True
    item.y = torch.zeros(1)
    batch = mips_trimer_collate([item])
    assert not hasattr(batch, "trimer_pos")
    assert not hasattr(batch, "trimer_mcl_thresholds")
    assert not hasattr(batch, "star_3d_distance")
