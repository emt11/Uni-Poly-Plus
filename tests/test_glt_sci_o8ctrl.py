"""Focused contract tests for the GLT-SCI-O8CTRL attribution route."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.dataset import glt_o8_control as data_mod
from src.dataset.glt_o8_control import (
    O8CleanLabeledDataset, O8OnlySource, build_o8_pretrain_sample,
    build_o8_sample, o8_collate,
)
from src.modules.glt_o8_control import (
    O8ControlModel, O8ControlPretrainer, apply_matched_predictor,
    common_initialized_o8_pretrainer, load_o8_deployment,
)
from scripts.run_glt_sci_o8_control_grid import build_command, validate_gpus


def _topology(n=3):
    if n == 3:
        edges = torch.tensor([[0, 1, 1, 0, 1, 2, 2, 1], [1, 0, 2, 1, 0, 1, 1, 2]])
    else:
        edges = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])
    relation = edges.size(1)
    path = torch.full((relation, 3), -1, dtype=torch.long)
    path[:, 0:2] = edges.t()
    x = torch.zeros(n, 137)
    x[:, 0] = 1
    return Data(
        mips_x=x, mips_backbone_mask=torch.zeros(n, dtype=torch.long),
        lga_edge_index=edges, lga_spd=torch.ones(relation, dtype=torch.long),
        lga_path_index=path, lga_path_mask=path >= 0,
        canonical_to_trimer_base_atom_id=torch.arange(n),
        canonical_ru_atom_index=torch.arange(n), graph_available=True,
        mts_canonical_periodic=True,
    )


def _static(top):
    relation = top.lga_edge_index.size(1)
    features = torch.zeros(relation, 2, 14)
    features[..., 0] = 1
    features[..., 7] = 1
    return {"bond_path_features": features.numpy(),
            "bond_path_mask": np.ones((relation, 2), dtype=bool)}


def test_o8_zero_pad_shape_and_no_glt():
    top = _topology()
    item = build_o8_sample(top, _static(top), key=b"k" * 32)
    model = O8ControlModel(dropout=0).eval()
    prediction, auxiliary = model(item)
    fused = model.fuse(model.encode(item))
    assert prediction.shape == (1, 1)
    assert auxiliary is None
    assert fused.shape == (1, 1024)
    assert torch.count_nonzero(fused[:, 512:]) == 0
    assert not hasattr(model, "glt")


def test_o8_pretrain_is_stateless_and_has_finite_gradients():
    top = _topology()
    target = {"brics_groups": ((0,), (1,), (2,)), "fingerprint_packed": np.zeros(256, dtype=np.uint8)}
    a = build_o8_pretrain_sample(top, _static(top), target, seed=42,
                                 key=b"a" * 32, position=17)
    b = build_o8_pretrain_sample(top, _static(top), target, seed=42,
                                 key=b"a" * 32, position=17)
    assert torch.equal(a[1]["atom_mask"], b[1]["atom_mask"])
    assert torch.equal(a[1]["fingerprint"], b[1]["fingerprint"])
    assert not ({"trimer_pos", "line_angle"} & set(a[0].keys()))
    model = O8ControlPretrainer(dropout=0)
    result = model(a[0], a[1])
    loss = (result["sums"] / result["counts"].clamp_min(1)).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert not hasattr(model, "glt") and not hasattr(model.encoder, "glt")


def test_o8_collate_offsets_mixed_sizes():
    first = build_o8_sample(_topology(3), _static(_topology(3)), key=b"a" * 32)
    second = build_o8_sample(_topology(2), _static(_topology(2)), key=b"b" * 32)
    batch = o8_collate([first, second])
    assert batch.mips_x.shape[0] == 5
    assert int(batch.lga_edge_index[:, -1].max()) < 5
    assert torch.equal(batch.canonical_graph_index, torch.tensor([0, 0, 0, 1, 1]))
    assert batch.graph_available.tolist() == [True, True]


def test_predictor_init_is_exact_and_common_init_digest_stable():
    first = common_initialized_o8_pretrainer(42)
    second = common_initialized_o8_pretrainer(42)
    assert first.common_init_digest == second.common_init_digest
    model = O8ControlModel(dropout=0)
    digest = apply_matched_predictor(model, 42, 0, dropout=0)
    state = {key: value.detach().cpu() for key, value in model.predictor.state_dict().items()}
    assert digest
    assert all(torch.equal(value, state[key]) for key, value in state.items())


def test_o8_deployment_loader_accepts_prefixed_o8_state():
    source = O8ControlModel(dropout=0)
    package = {
        "architecture": source.architecture_name,
        "fusion_mode": "o8_control", "step": 5000, "use_md200": False,
        "state_dict": {
            **{f"o8.{key}": value.detach().cpu().clone()
               for key, value in source.o8.state_dict().items()},
            **{f"norm2.{key}": value.detach().cpu().clone()
               for key, value in source.norm2.state_dict().items()},
        },
    }
    restored = O8ControlModel(dropout=0)
    load_o8_deployment(restored, package)
    assert all(torch.equal(source.state_dict()[key], restored.state_dict()[key])
               for key in package["state_dict"])


def test_o8_source_opens_topology_only(monkeypatch):
    top = _topology()
    key = b"a" * 32
    cohort = {"manifest": {"ordered_sample_key_hash": "order", "main_bundle_hash": "bundle"},
              "manifest_hash": "cohort", "records": [{
                  "sample_key": key.hex(), "source_smiles": "*C(*)C",
                  "normalized_smiles": "*C(*)C", "source_row": 0,
              }]}
    store = {"bundle_hash": "bundle", "artifacts": {"topology": {"path": "topology"}}}

    class FakeArtifact:
        opened = []
        def __init__(self, root, binding):
            FakeArtifact.opened.append(binding["path"])
        def __getitem__(self, requested):
            assert requested == key
            return top
        def close(self):
            pass

    class FakeCache:
        manifest = {"cohort_ordered_sample_key_hash": "order"}
        manifest_hash = "static"
        def index_for_key(self, requested):
            assert requested == key
            return 0
        def get_by_key(self, requested):
            assert requested == key
            return _static(top)
        def close(self):
            pass

    monkeypatch.setattr(data_mod, "load_dual_cohort", lambda *args, **kwargs: cohort)
    monkeypatch.setattr(data_mod, "load_active_dual_store", lambda *args, **kwargs: store)
    monkeypatch.setattr(data_mod, "ReadonlyArtifact", FakeArtifact)
    monkeypatch.setattr(data_mod, "load_static_caches", lambda *args, **kwargs: (FakeCache(), None))
    source = O8OnlySource("cohort", "cache", static_root="static")
    try:
        assert not hasattr(source, "trimer")
        assert FakeArtifact.opened == ["topology"]
        assert "trimer" not in FakeArtifact.opened
        assert not ({"trimer_pos", "line_angle"} & set(source[0].keys()))
    finally:
        source.close()


class _FakeSource:
    def __init__(self, keys):
        self.samples = [(key, "*") for key in keys]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return _topology(2)
    def static_for(self, index):
        return _static(_topology(2))


def test_o8_duplicate_key_keeps_row_labels_and_bounded_eviction(monkeypatch):
    calls = []
    def fake_build(topology, static, key=None):
        calls.append(bytes(key))
        return Data(x=torch.tensor([float(key[0])]))
    monkeypatch.setattr(data_mod, "build_o8_sample", fake_build)
    duplicate = _FakeSource([b"a" * 32, b"a" * 32])
    dataset = O8CleanLabeledDataset(duplicate, np.asarray([1., 2.]), cache_capacity_bytes=1024)
    assert dataset[0].y.tolist() == [1.]
    assert dataset[1].y.tolist() == [2.]
    assert calls == [b"a" * 32]
    tiny = _FakeSource([b"a" * 32, b"b" * 32])
    bounded = O8CleanLabeledDataset(tiny, np.asarray([1., 2.]), cache_capacity_bytes=4)
    first = bounded[0].x.clone(); bounded[1]; rebuilt = bounded[0].x.clone()
    assert first.tolist() == rebuilt.tolist() == [97.]
    assert bounded.cache_stats()["evictions"] == 2


def test_o8_grid_requires_three_distinct_slots_and_forwards_capacity():
    with pytest.raises(ValueError, match="distinct"):
        validate_gpus(["1", "1", "2"])
    args = SimpleNamespace(config="c", checkpoint="p", raw_root="r", cohort_root="co",
                           cache_root="ca", dual_static_root="s", split_root="sp",
                           arm="A", clean_cache_gib=4)
    command = build_command(args, "eat", 0, Path("out"))
    marker = command.index("--clean-cache-gib")
    assert command[marker + 1] == "4"
