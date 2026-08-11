import json

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.dataset import dataset as dataset_module
from src.dataset.dataset import UniDataset
from src.dataset.lmdb_cache import (
    CACHE_LAYOUT_SCHEMA,
    LmdbLayerStore,
    LmdbLayerWriter,
    sample_key_from_smiles,
)
from src.dataset.mips_trimer_contract import (
    CACHE_LAYOUT_SCHEMA as CONTRACT_LAYOUT_SCHEMA,
    CONFIG_SCHEMA,
    FEATURE_SCHEMA,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_PROTOCOL,
)
from src.dataset.graph_data import (
    build_mips_local_structure,
    build_periodic_multimer_mol,
)


def _dataset_kwargs(root, *, layers, rebuild=False):
    return dict(
        root=str(root),
        dataset="tiny",
        smiles_model_name="/path/that/must/not/be-loaded",
        geometry_encoder="painn",
        graph_encoder_type="scage",
        graph_input="star_linking",
        geom_input="repeat_unit",
        use_feature_cache=True,
        feature_source_dataset="tiny",
        rebuild_feature_cache=rebuild,
        fp_mode="disabled",
        feature_cache_workers=1,
        feature_cache_item_timeout=60,
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mcl_distance_percentiles=(0.20, 0.50),
        experiment_id="lmdb_test",
        feature_config_hash="lmdb-test-v1",
        cache_layers=layers,
        cache_validate="full",
        cache_commit_size=2,
    )


def _tiny_root(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "smiles": ["*CC*", "*CCO*"],
        "target": [0.0, 1.0],
    }).to_csv(raw / "tiny.csv", index=False)
    return tmp_path


def test_lmdb_writer_is_per_record_and_resumable(tmp_path):
    root = tmp_path / "layer"
    meta = {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "schema": "unit-layer-v1",
        "feature_config_hash": "abc",
    }
    first = sample_key_from_smiles("*CC*")
    second = sample_key_from_smiles("*CCO*")
    writer = LmdbLayerWriter(root, meta, commit_size=1)
    writer.add(first, Data(x=torch.ones(1, 2)))
    writer.finalize(cohort_hashes=["first"])

    writer = LmdbLayerWriter(root, meta, commit_size=2)
    assert first in writer
    writer.add(second, Data(x=torch.zeros(1, 2)))
    writer.finalize(cohort_hashes=["second"])

    store = LmdbLayerStore(root, expected_meta=meta)
    assert len(store) == 2
    assert torch.equal(store[first].x, torch.ones(1, 2))
    assert torch.equal(store[second].x, torch.zeros(1, 2))
    assert store.manifest["cohort_hashes"] == ["first", "second"]


def test_lmdb_writer_lock_rejects_second_writer(tmp_path):
    root = tmp_path / "locked-layer"
    meta = {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "schema": "unit-layer-v1",
        "feature_config_hash": "locked",
    }
    writer = LmdbLayerWriter(root, meta)
    with pytest.raises(RuntimeError, match="writer lock"):
        LmdbLayerWriter(root, meta)
    writer.close()
    recovered = LmdbLayerWriter(root, meta)
    recovered.close()


def test_lmdb_writer_can_replace_one_obsolete_record(tmp_path):
    root = tmp_path / "replace-layer"
    meta = {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "schema": "unit-layer-v1",
        "feature_config_hash": "replace",
    }
    key = sample_key_from_smiles("*CC*")
    writer = LmdbLayerWriter(root, meta, commit_size=1)
    writer.add(key, Data(x=torch.ones(1, 1)))
    writer.finalize()

    writer = LmdbLayerWriter(root, meta, commit_size=1)
    assert writer.replace(key, Data(x=torch.zeros(1, 1)))
    writer.finalize()
    store = LmdbLayerStore(root, expected_meta=meta)
    assert torch.equal(store[key].x, torch.zeros(1, 1))
    store.close()


def test_graph_only_bypasses_tokenizer_and_token_fields(tmp_path):
    root = _tiny_root(tmp_path)
    dataset = UniDataset(**_dataset_kwargs(
        root, layers="topology", rebuild=True
    ))
    assert dataset.smiles_tokenizer is None
    assert dataset.max_smiles_length == 0
    sample = dataset[0]
    assert not hasattr(sample, "input_ids_smiles")
    assert not hasattr(sample, "attention_mask_smiles")
    assert not hasattr(sample, "fp")
    assert sample.lga_edge_index.size(0) == 2

    diagnostics = list(
        (root / "processed" / "scage").glob("diagnostics_*.json")
    )
    assert diagnostics
    with open(diagnostics[0], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["schema"] == CACHE_LAYOUT_SCHEMA
    validation_files = list(
        (root / "processed" / "mips_trimer_scage").rglob("validation.json")
    )
    assert validation_files


def test_missing_trimer_reuses_existing_topology(tmp_path, monkeypatch):
    root = _tiny_root(tmp_path)
    UniDataset(**_dataset_kwargs(root, layers="topology", rebuild=True))

    def forbidden_topology(*_args, **_kwargs):
        raise AssertionError("topology was recomputed while only Trimer was missing")

    monkeypatch.setattr(
        dataset_module, "_compute_topology_layer", forbidden_topology
    )
    dataset = UniDataset(**_dataset_kwargs(
        root, layers="trimer", rebuild=True
    ))
    assert hasattr(dataset[0], "trimer_geometry_valid")


def test_missing_md200_does_not_compute_topology_or_trimer(tmp_path, monkeypatch):
    root = _tiny_root(tmp_path)
    UniDataset(**_dataset_kwargs(root, layers="topology", rebuild=True))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unrelated graph/geometry layer was recomputed")

    monkeypatch.setattr(dataset_module, "_compute_topology_layer", forbidden)
    monkeypatch.setattr(dataset_module, "_compute_trimer_layer", forbidden)
    dataset = UniDataset(**_dataset_kwargs(
        root, layers="md200", rebuild=True
    ))
    assert dataset[0].mips_md.shape == (200,)


def test_shared_attachment_boundary_builds_valid_repeated_graph(tmp_path):
    root = tmp_path
    raw = root / "raw"
    raw.mkdir()
    pd.DataFrame({
        "smiles": ["*C(*)C(=O)OCC(C)(C)C"],
        "target": [0.0],
    }).to_csv(raw / "tiny.csv", index=False)

    topology = UniDataset(**_dataset_kwargs(
        root, layers="topology", rebuild=True
    ))
    sample = topology[0]
    assert bool(sample.graph_available)
    assert int(sample.mips_repeat_units) == 7
    assert int(sample.mips_boundary_distance) == 6
    assert bool(sample.ru_shared_boundary)
    star_edges = sample.lga_edge_index[:, sample.lga_star_edge_mask]
    assert star_edges.size(1) > 0
    # Canonical lifted relations intentionally retain the shared-boundary
    # self rows at relative shifts -1/+1; they are not finite-copy Star edges.
    assert torch.any(star_edges[0] == star_edges[1])
    topology._lazy_feature_store.close()

    trimer = UniDataset(**_dataset_kwargs(
        root, layers="trimer", rebuild=True
    ))
    sample = trimer[0]
    # Geometry may independently fail its ETKDG/MMFF quality gate, but shared
    # attachment boundaries must no longer be rejected as a graph condition.
    assert sample.trimer_failure_code != "graph_unavailable"


def test_shared_boundary_and_mismatched_bond_rules():
    structure = build_mips_local_structure("*C(*)C(=O)OCC(C)(C)C")
    metadata = structure["repeat_metadata"]
    assert structure["mips_repeat_units"] == 7
    assert structure["mips_boundary_distance"] == 6
    assert metadata["shared_boundary"]
    assert structure["backbone_info"]["star_link_edge"][0] != (
        structure["backbone_info"]["star_link_edge"][1]
    )

    short = build_mips_local_structure("*CCO*")
    assert short["mips_repeat_units"] == 3

    mismatch = build_mips_local_structure("*C(C#*)CC")
    mismatch_meta = mismatch["repeat_metadata"]
    assert mismatch_meta["attachment_bond_mismatch"]
    assert mismatch_meta["connection_bond_policy"] == "mismatch_single"
    assert str(mismatch_meta["attachment_bond_type"]) == "SINGLE"

    trimer, trimer_meta = build_periodic_multimer_mol(
        "*C(*)C(=O)OCC(C)(C)C", 3, close_periodic=False
    )
    assert len(trimer_meta["inter_unit_edges"]) == 2
    assert trimer_meta["inter_unit_edges"][0] != trimer_meta["inter_unit_edges"][1]
    assert all(
        trimer.GetBondBetweenAtoms(int(left), int(right)) is not None
        for left, right in trimer_meta["inter_unit_edges"]
    )


def test_layer_hashes_encode_upstream_dependencies(tmp_path):
    root = _tiny_root(tmp_path)
    dataset = UniDataset(**_dataset_kwargs(
        root, layers="trimer", rebuild=True
    ))
    specs = dataset._lmdb_cache_specs(dataset._feature_cache_meta())
    ru_hash = specs["ru_base"]["meta"]["feature_config_hash"]
    topology_hash = specs["topology"]["meta"]["feature_config_hash"]
    topology_config = specs["topology"]["meta"]["build_config"]
    trimer_config = specs["trimer"]["meta"]["build_config"]
    assert topology_config["ru_base_feature_config_hash"] == ru_hash
    assert trimer_config["ru_base_feature_config_hash"] == ru_hash
    assert trimer_config["topology_feature_config_hash"] == topology_hash


def test_compact_missing_mask_and_runtime_contract(tmp_path):
    root = tmp_path / "layer"
    meta = {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "schema": "unit-layer-v1",
        "feature_config_hash": "mask",
    }
    keys = [sample_key_from_smiles(value) for value in ("*CC*", "*CCO*")]
    writer = LmdbLayerWriter(root, meta, commit_size=1)
    writer.add(keys[0], Data(x=torch.ones(1, 1)))
    mask = writer.missing_mask(keys)
    assert mask.dtype.name == "uint8"
    assert mask.tolist() == [0, 1]
    writer.finalize(cohort_hashes=["mask"])
    store = LmdbLayerStore(root, expected_meta=meta)
    assert store.missing_mask(keys).tolist() == [0, 1]
    store.close()

    config = json.load(open(
        "configs/mts/default.json", encoding="utf-8"
    ))
    assert config["schema_version"] == CONFIG_SCHEMA
    assert config["feature_schema"] == FEATURE_SCHEMA
    assert config["trimer_cache_schema"] == TRIMER_CONTENT_SCHEMA
    assert config["trimer"]["protocol"] == TRIMER_PROTOCOL
    assert config["trimer"]["require_mmff_convergence"] is False


def test_writer_flush_preserves_parent_received_results(tmp_path):
    root = tmp_path / "interrupted"
    meta = {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "schema": "unit-layer-v1",
        "feature_config_hash": "interrupt",
    }
    key = sample_key_from_smiles("*CC*")
    writer = LmdbLayerWriter(root, meta, commit_size=128)
    writer.add(key, Data(x=torch.ones(1, 1)))
    writer.flush()
    writer.close()

    resumed = LmdbLayerWriter(root, meta, commit_size=128)
    assert key in resumed
    assert resumed.missing([key]) == []
    resumed.finalize(cohort_hashes=["resume"])


def test_old_layout_is_rejected_before_record_loading(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    with open(root / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({
            "cache_layout_schema": "mips-trimer-scage-lmdb-layout-v1",
            "schema": "mips-trimer-scage-topology-lmdb-v1",
        }, handle)
    with pytest.raises(RuntimeError, match="incompatible LMDB cache layout"):
        LmdbLayerStore(root, require_done=False)


def test_expected_bad_samples_become_unavailable_records():
    for smiles in ("not-a-smiles", "CCO", "*CC"):
        ru_base = dataset_module._compute_ru_base_layer(smiles)
        topology = dataset_module._compute_topology_layer(
            smiles, ru_base, max_hops=2
        )
        assert not bool(topology.graph_available)
        assert str(topology.topology_failure_code)


def test_attachment_reversal_preserves_mips_boundary_solution():
    forward = build_mips_local_structure("*CCO*")
    reverse = build_mips_local_structure("*OCC*")
    assert forward["mips_repeat_units"] == reverse["mips_repeat_units"]
    assert forward["mips_boundary_distance"] == reverse["mips_boundary_distance"]
    assert (
        forward["structure_mol"].GetNumAtoms()
        == reverse["structure_mol"].GetNumAtoms()
    )
    assert (
        forward["structure_mol"].GetNumBonds()
        == reverse["structure_mol"].GetNumBonds()
    )
