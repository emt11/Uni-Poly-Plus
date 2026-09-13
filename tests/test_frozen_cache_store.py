"""Frozen cache store tests (MIPS-style whole-dataset cache, no versions).

Covers the thirteen contract behaviours of the build_spec/frozen-store
refactor: canonical keys, build_spec hashing, fold independence, whole-dataset
reuse across folds, frozen gating, cache-miss-as-error, geometry fallback,
identity corruption, store.json selection and derived-parent binding.
"""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_spec import (
    ROUTE_BUILD_SPECS,
    SpecError,
    build_spec_hash,
    canonical_json,
    validate_build_spec,
)
from src.dataset.frozen_store import (
    ArtifactIdentityError,
    ArtifactNotFrozen,
    CacheMissingDerivedArtifact,
    CacheMissError,
    FrozenArtifact,
    StoreError,
    open_frozen_feature_store,
    resolve_md200_paths,
)
from src.dataset.lmdb_cache import sample_key_from_smiles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dump(payload) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def make_topology_record(key: bytes, *, n_nodes=4, marker="A"):
    data = Data()
    data.x = torch.arange(n_nodes * 3, dtype=torch.float).reshape(n_nodes, 3)
    data.z = torch.arange(n_nodes, dtype=torch.long)
    data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.canonical_ru_atom_index = torch.arange(n_nodes, dtype=torch.long)
    data.graph_available = True
    data.topology_failure_code = ""
    data.mips_to_trimer_central_index = torch.full((n_nodes,), -1, dtype=torch.long)
    data.marker = marker
    return data


def make_trimer_record(key: bytes, *, n_nodes=4, n_atoms=6, geometry_valid=True,
                       mapping=None, central=None):
    data = Data()
    data.trimer_geometry_valid = geometry_valid
    data.trimer_geometry_is_3d = geometry_valid
    data.trimer_2d_fallback = False
    data.trimer_pos = (
        torch.arange(n_atoms * 3, dtype=torch.float).reshape(n_atoms, 3)
        if geometry_valid else torch.empty((0, 3), dtype=torch.float)
    )
    data.trimer_atomic_number = torch.full((n_atoms,), 6, dtype=torch.long)
    data.trimer_atomic_numbers = data.trimer_atomic_number
    data.trimer_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.trimer_bond_type = torch.tensor([1, 1], dtype=torch.long)
    if central is None:
        central = [True] + [False] * (n_atoms - 1)
    data.trimer_central_ru_mask = torch.tensor(central, dtype=torch.bool)
    if mapping is None:
        mapping = [0] * n_nodes
    data.mips_to_trimer_central_index = torch.tensor(mapping, dtype=torch.long)
    data.o8_to_trimer_atom = data.mips_to_trimer_central_index
    data.o8_heavy_mask = torch.ones(n_nodes, dtype=torch.bool)
    data.o8_heavy_indices = torch.arange(n_nodes, dtype=torch.long)
    data.trimer_heavy_mask = data.o8_heavy_mask
    data.trimer_heavy_indices = torch.arange(n_atoms, dtype=torch.long)
    data.star_3d_distance = torch.tensor(0.0, dtype=torch.float)
    data.star_3d_valid = torch.tensor(geometry_valid, dtype=torch.bool)
    return data


def make_ru_base_record(key: bytes):
    data = Data()
    data.normalized_polymer_smiles = "*CC*"
    data.ru_chemistry_valid = True
    data.ru_base_valid = True
    data.ru_base_failure_code = ""
    data.ru_atomic_number = torch.tensor([6, 6], dtype=torch.long)
    data.ru_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.ru_bond_type = torch.tensor([1, 1], dtype=torch.long)
    return data


def make_md200_record():
    data = Data()
    data.mips_md = torch.arange(200, dtype=torch.float) / 200.0
    data.mips_md_valid = True
    return data


def write_artifact(root: Path, artifact_type: str, build_spec: dict, records: dict,
                   *, frozen: bool = True) -> str:
    """Write one artifact directory.  Envelope schema strings are deliberately
    bogus version strings: the formal reader must never gate on them."""
    root.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(root / "data.lmdb"), subdir=True, map_size=1 << 26)
    with env.begin(write=True) as txn:
        for key, data in records.items():
            payload = {
                "layout_schema": "irrelevant-legacy-layout-v99",
                "content_schema": "irrelevant-legacy-content-v42",
                "sample_key": key,
                "data": data,
            }
            txn.put(key, _dump(payload))
    env.close()
    spec_hash = build_spec_hash(build_spec)
    (root / "metadata.json").write_text(json.dumps({
        "artifact_type": artifact_type,
        "build_spec": build_spec,
        "build_spec_hash": spec_hash,
        "rdkit_version": "2026.03.2",
    }, sort_keys=True), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "count": len(records),
    }, sort_keys=True), encoding="utf-8")
    (root / ".done").write_text("a" * 64 + "\n", encoding="utf-8")
    if frozen:
        (root / ".frozen").write_text(
            json.dumps({"build_spec_hash": spec_hash}), encoding="utf-8"
        )
    return spec_hash


def _binding(relpath: str, build_spec: dict, count: int, **extra) -> dict:
    binding = {
        "path": relpath,
        "build_spec": build_spec,
        "build_spec_hash": build_spec_hash(build_spec),
        "artifact_hash": "a" * 64,
        "record_count": count,
    }
    binding.update(extra)
    return binding


def make_store(tmp_path: Path, bindings: dict) -> dict:
    store = {"artifacts": bindings, "created_at": 0.0}
    (tmp_path / "store.json").write_text(
        json.dumps(store, sort_keys=True), encoding="utf-8"
    )
    return store


def cohort_from_keys(keys, *, root="cohort_test"):
    array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    return {
        "root": str(root),
        "manifest": {
            "cohort_hash": "0" * 64,
            "ordered_sample_key_hash": hashlib.sha256(array.tobytes()).hexdigest(),
            "unique_count": len(array),
        },
        "keys_array": array,
    }


def _write_md200_array(cohort, cache_root: Path, build_spec_hash: str):
    """Offline materialization fixture equivalent (what
    scripts/materialize_cache_derived.py produces)."""

    out = Path(cohort["root"]) / f"md200_{build_spec_hash}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "md200.npy",
            np.zeros((len(cohort["keys_array"]), 200), dtype=np.float32))
    np.save(out / "md200_valid.npy",
            np.ones(len(cohort["keys_array"]), dtype=bool))
    (out / "md200_metadata.json").write_text(json.dumps({
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "build_spec_hash": build_spec_hash,
        "shape": [len(cohort["keys_array"]), 200],
        "dtype": "float32",
    }, sort_keys=True), encoding="utf-8")
    return out


@pytest.fixture()
def frozen_world(tmp_path):
    """One whole-dataset frozen store binding topology+trimer+md200+ru_base."""

    keys = [
        sample_key_from_smiles(smiles)
        for smiles in ("*CC*", "*CCC*", "*CC(C)C*", "*C(C)C*")
    ]
    layers = {
        "ru_base": ("ru_base", ROUTE_BUILD_SPECS["ru_base"],
                    {key: make_ru_base_record(key) for key in keys}),
        "topology": ("topology", ROUTE_BUILD_SPECS["topology"],
                     {key: make_topology_record(key, marker="active") for key in keys}),
        "trimer": ("trimer", ROUTE_BUILD_SPECS["trimer"],
                   {key: make_trimer_record(key) for key in keys}),
        "md200": ("md200", ROUTE_BUILD_SPECS["md200"],
                  {key: make_md200_record() for key in keys}),
    }
    bindings = {}
    for name, (dirname, spec, records) in layers.items():
        artifact_dir = tmp_path / dirname / "artifact-a"
        write_artifact(artifact_dir, name, spec, records)
        bindings[name] = _binding(f"{dirname}/artifact-a", spec, len(keys))
    make_store(tmp_path, bindings)
    cohort = cohort_from_keys(keys, root=tmp_path / "cohorts" / "test")
    _write_md200_array(cohort, tmp_path, bindings["md200"]["build_spec_hash"])
    return {"tmp": tmp_path, "keys": keys, "bindings": bindings, "cohort": cohort}


# ---------------------------------------------------------------------------
# 1-5: identity and build_spec hashing
# ---------------------------------------------------------------------------

def test_1_canonical_sample_key_stable():
    # whitespace is not identity
    assert sample_key_from_smiles("*CC*") == sample_key_from_smiles(" *CC* ")
    # canonical normalization: equivalent spellings share one identity
    assert sample_key_from_smiles("*CC(C)*") == sample_key_from_smiles("*C(C)C*")
    # different molecules are different samples
    assert sample_key_from_smiles("*CC*") != sample_key_from_smiles("*CCC*")
    assert len(sample_key_from_smiles("*CC*")) == 32


def test_2_build_spec_canonical_serialization_stable():
    left = {"artifact_type": "trimer", "parameters": {"a": 1, "b": [1, 2]}}
    right = {"parameters": {"b": [1, 2], "a": 1}, "artifact_type": "trimer"}
    assert canonical_json(left) == canonical_json(right)
    assert build_spec_hash(left) == build_spec_hash(right)


def test_3_same_build_spec_same_hash():
    assert build_spec_hash(ROUTE_BUILD_SPECS["trimer"]) == build_spec_hash(
        json.loads(json.dumps(ROUTE_BUILD_SPECS["trimer"]))
    )


def test_4_key_parameter_change_changes_hash():
    changed = json.loads(json.dumps(ROUTE_BUILD_SPECS["trimer"]))
    changed["parameters"]["num_candidates_per_round"] = 2
    assert build_spec_hash(ROUTE_BUILD_SPECS["trimer"]) != build_spec_hash(changed)


def test_5_fold_identity_never_enters_build_spec():
    poisoned = json.loads(json.dumps(ROUTE_BUILD_SPECS["topology"]))
    poisoned["parameters"]["fold"] = 3
    with pytest.raises(SpecError):
        validate_build_spec(poisoned)
    poisoned2 = json.loads(json.dumps(ROUTE_BUILD_SPECS["topology"]))
    poisoned2["split_id"] = 1
    with pytest.raises(SpecError):
        validate_build_spec(poisoned2)
    for name, spec in ROUTE_BUILD_SPECS.items():
        blob = canonical_json(spec)
        for token in ("fold", "split", "scaffold", "use_idxs"):
            assert f'"{token}"' not in blob, (name, token)


# ---------------------------------------------------------------------------
# 6-8: whole-dataset reuse, frozen gating
# ---------------------------------------------------------------------------

def test_6_whole_dataset_cache_shared_by_folds(frozen_world):
    keys = frozen_world["keys"]
    cache_root = frozen_world["tmp"]
    fold0 = cohort_from_keys(keys[:3], root="fold0")
    fold1 = cohort_from_keys(keys[1:], root="fold1")
    store0, _ = open_frozen_feature_store(
        cache_root, ("ru_base", "topology", "trimer"), cohort=fold0
    )
    store1, _ = open_frozen_feature_store(
        cache_root, ("ru_base", "topology", "trimer"), cohort=fold1
    )
    # both folds read the SAME artifact directories; no per-fold cache exists
    assert store0.roots == store1.roots
    assert sorted(p.name for p in (cache_root / "topology").iterdir()) == ["artifact-a"]
    assert fold0["keys_array"].shape[0] == 3 and fold1["keys_array"].shape[0] == 3
    for store, cohort in ((store0, fold0), (store1, fold1)):
        for key in cohort["keys_array"]:
            assert bytes(key) in store
    store0.close()
    store1.close()


def test_7_frozen_artifact_readable_and_schema_agnostic(frozen_world):
    cache_root = frozen_world["tmp"]
    store, loaded = open_frozen_feature_store(
        cache_root, ("ru_base", "topology", "trimer", "md200"),
        cohort=frozen_world["cohort"],
    )
    assert loaded["artifacts"]["topology"]["build_spec_hash"] == build_spec_hash(
        ROUTE_BUILD_SPECS["topology"]
    )
    key = frozen_world["keys"][0]
    data = store[key]
    assert data.marker == "active"           # topology record (bogus envelope ignored)
    assert data.trimer_pos.shape == (6, 3)   # merged trimer fields
    assert data.mips_md.shape == (200,)      # merged md200 mmap row
    assert store.roots["md200"]
    store.close()


def test_8_unfrozen_artifact_rejected_for_training(tmp_path):
    keys = [sample_key_from_smiles("*CC*")]
    spec = ROUTE_BUILD_SPECS["topology"]
    topo_dir = tmp_path / "topology" / "unfrozen"
    write_artifact(topo_dir, "topology", spec,
                   {keys[0]: make_topology_record(keys[0])}, frozen=False)
    make_store(tmp_path, {"topology": _binding("topology/unfrozen", spec, 1)})
    with pytest.raises(ArtifactNotFrozen):
        open_frozen_feature_store(
            tmp_path, ("topology",), cohort=cohort_from_keys(keys)
        )


# ---------------------------------------------------------------------------
# 9-11: miss, geometry fallback, corruption
# ---------------------------------------------------------------------------

def test_9_cache_miss_is_error_never_rebuilt(frozen_world):
    cache_root = frozen_world["tmp"]
    store, _ = open_frozen_feature_store(
        cache_root, ("topology",), cohort=frozen_world["cohort"]
    )
    unknown = sample_key_from_smiles("*CCCCCCCCCC*")
    artifact_dir = cache_root / "topology" / "artifact-a"
    before = sorted(os.listdir(artifact_dir))
    with pytest.raises(CacheMissError):
        store[unknown]
    assert sorted(os.listdir(artifact_dir)) == before  # nothing was generated
    store.close()


def test_10_geometry_failure_placeholder_passes(frozen_world):
    tmp = frozen_world["tmp"]
    keys = frozen_world["keys"]
    # one ordinary geometry failure inside the same whole-dataset artifact
    records = {key: make_trimer_record(key) for key in keys}
    records[keys[1]] = make_trimer_record(keys[1], geometry_valid=False)
    spec = ROUTE_BUILD_SPECS["trimer"]
    trimer_dir = tmp / "trimer" / "with-placeholder"
    write_artifact(trimer_dir, "trimer", spec, records)
    bindings = dict(frozen_world["bindings"])
    bindings["trimer"] = _binding("trimer/with-placeholder", spec, len(keys))
    make_store(tmp, bindings)
    store, _ = open_frozen_feature_store(
        tmp, ("topology", "trimer"), cohort=frozen_world["cohort"]
    )
    data = store[keys[1]]  # ordinary failure: delivered, not raised
    assert not bool(data.trimer_geometry_valid)
    assert data.trimer_pos.numel() == 0
    store.close()


def _write_envelope_mismatch(root: Path, spec: dict, lmdb_key: bytes,
                             envelope_key: bytes):
    root.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(root / "data.lmdb"), subdir=True, map_size=1 << 26)
    payload = {"sample_key": envelope_key, "data": make_trimer_record(envelope_key)}
    with env.begin(write=True) as txn:
        txn.put(lmdb_key, _dump(payload))
    env.close()
    spec_hash = build_spec_hash(spec)
    (root / "metadata.json").write_text(json.dumps(
        {"artifact_type": "trimer", "build_spec": spec,
         "build_spec_hash": spec_hash}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"record_count": 1}), encoding="utf-8")
    (root / ".done").write_text("a" * 64 + "\n", encoding="utf-8")
    (root / ".frozen").write_text(json.dumps({"build_spec_hash": spec_hash}), encoding="utf-8")


def _open_single_join(root: Path, key, topo_record, trimer_record, trimer_spec):
    root.mkdir(parents=True, exist_ok=True)
    topo_dir = root / "topology" / "a"
    trimer_dir = root / "trimer" / "a"
    write_artifact(topo_dir, "topology", ROUTE_BUILD_SPECS["topology"], {key: topo_record})
    write_artifact(trimer_dir, "trimer", trimer_spec, {key: trimer_record})
    make_store(root, {
        "topology": _binding("topology/a", ROUTE_BUILD_SPECS["topology"], 1),
        "trimer": _binding("trimer/a", trimer_spec, 1),
    })
    store, _ = open_frozen_feature_store(
        root, ("topology", "trimer"), cohort=cohort_from_keys([key])
    )
    return store


def test_11_identity_corruption_hard_fails(tmp_path):
    key = sample_key_from_smiles("*CC*")
    spec = ROUTE_BUILD_SPECS["trimer"]

    # (a) envelope sample_key mismatch
    root = tmp_path / "casea"
    topo_dir = root / "topology" / "a"
    trimer_dir = root / "trimer" / "a"
    write_artifact(topo_dir, "topology", ROUTE_BUILD_SPECS["topology"],
                   {key: make_topology_record(key)})
    other_key = sample_key_from_smiles("*CCC*")
    _write_envelope_mismatch(trimer_dir, spec, lmdb_key=key, envelope_key=other_key)
    make_store(root, {
        "topology": _binding("topology/a", ROUTE_BUILD_SPECS["topology"], 1),
        "trimer": _binding("trimer/a", spec, 1),
    })
    store, _ = open_frozen_feature_store(
        root, ("topology", "trimer"), cohort=cohort_from_keys([key])
    )
    with pytest.raises(ArtifactIdentityError):
        store[key]
    store.close()

    # (b) mapping length mismatch
    store2 = _open_single_join(
        tmp_path / "caseb", key, make_topology_record(key),
        make_trimer_record(key, mapping=[0, 0]), spec,
    )
    with pytest.raises(ArtifactIdentityError):
        store2[key]
    store2.close()

    # (c) mapping out of trimer atom range
    store3 = _open_single_join(
        tmp_path / "casec", key, make_topology_record(key),
        make_trimer_record(key, mapping=[99, 0, 0, 0]), spec,
    )
    with pytest.raises(ArtifactIdentityError):
        store3[key]
    store3.close()

    # (d) mapping not on the central RU
    store4 = _open_single_join(
        tmp_path / "cased", key, make_topology_record(key),
        make_trimer_record(key, mapping=[1, 0, 0, 0],
                           central=[True, False, False, False, False, False]),
        spec,
    )
    with pytest.raises(ArtifactIdentityError):
        store4[key]
    store4.close()


# ---------------------------------------------------------------------------
# 12-13: store selection and derived parent binding
# ---------------------------------------------------------------------------

def test_12_store_json_selects_active_artifact(tmp_path):
    key = sample_key_from_smiles("*CC*")
    spec = ROUTE_BUILD_SPECS["topology"]
    dir_a = tmp_path / "topology" / "artifact-a"
    dir_b = tmp_path / "topology" / "artifact-b"
    write_artifact(dir_a, "topology", spec, {key: make_topology_record(key, marker="A")})
    write_artifact(dir_b, "topology", spec, {key: make_topology_record(key, marker="B")})
    make_store(tmp_path, {"topology": _binding("topology/artifact-b", spec, 1)})
    store, _ = open_frozen_feature_store(
        tmp_path, ("topology",), cohort=cohort_from_keys([key])
    )
    assert store[key].marker == "B"  # store.json decided, not directory scanning
    store.close()

    broken = _binding("topology/does-not-exist", spec, 1)
    make_store(tmp_path, {"topology": broken})
    with pytest.raises(StoreError):
        open_frozen_feature_store(
            tmp_path, ("topology",), cohort=cohort_from_keys([key])
        )


def test_13_derived_parent_binding(frozen_world):
    # topology route spec chains to the ru_base route spec
    assert (
        ROUTE_BUILD_SPECS["topology"]["parents"]["ru_base"]
        == build_spec_hash(ROUTE_BUILD_SPECS["ru_base"])
    )
    changed_ru = json.loads(json.dumps(ROUTE_BUILD_SPECS["ru_base"]))
    changed_ru["parameters"]["mismatched_bond_policy"] = "keep"
    assert build_spec_hash(changed_ru) != build_spec_hash(ROUTE_BUILD_SPECS["ru_base"])

    # md200: a registered materialized array with matching cohort metadata is
    # reused; a stale or missing one is a hard error and nothing is written.
    cache_root = frozen_world["tmp"]
    md_binding = dict(frozen_world["bindings"]["md200"])
    cohort = frozen_world["cohort"]
    artifact = FrozenArtifact(cache_root, "md200", md_binding, validate_route=False)
    values, _ = resolve_md200_paths(cohort, artifact)
    assert Path(values).name == "md200.npy"
    artifact.close()

    stale_dir = cache_root / "cohorts" / "test" / "md200_stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    bad_metadata = {
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": "f" * 64,
        "shape": [len(cohort["keys_array"]), 200],
        "dtype": "float32",
    }
    (stale_dir / "md200_metadata.json").write_text(json.dumps(bad_metadata), encoding="utf-8")
    md_binding2 = dict(md_binding)
    md_binding2["materialized"] = {"test": str(stale_dir.relative_to(cache_root))}
    artifact2 = FrozenArtifact(cache_root, "md200", md_binding2, validate_route=False)
    with pytest.raises(CacheMissingDerivedArtifact):
        resolve_md200_paths(cohort, artifact2)
    artifact2.close()


# ---------------------------------------------------------------------------
# Finish-up tests (round 2): semantic build specs, geometry seeds,
# zero-write reader, active-only store
# ---------------------------------------------------------------------------

import re

import src.dataset.cache_spec as cache_spec
from src.dataset.cache_spec import (
    MD200_BUILD_SPEC,
    ROUTE_BUILD_SPEC_HASHES,
    RU_BASE_BUILD_SPEC,
    TRIMER_SEED_POLICY,
    canonical_json as spec_canonical_json,
)
from src.dataset.trimer_mcl import GEOMETRY_SEED_SPEC, _geometry_seed

_VERSION_KEY_PATTERN = re.compile(
    r"(_version$|^schema$|schema$|^builder$|^generation$|^revision$|^epoch$"
    r"|^format_level$|descriptor_schema|lga_schema)"
)
_VERSION_VALUE_PATTERN = re.compile(r"_v\d+$")


def _walk_spec(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key), value
            yield from _walk_spec(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_spec(value)


def test_f1_no_version_tokens_in_active_build_specs():
    for name, spec in ROUTE_BUILD_SPECS.items():
        for key, value in _walk_spec(spec):
            assert not _VERSION_KEY_PATTERN.search(key), (name, key)
            if isinstance(value, str):
                assert not _VERSION_VALUE_PATTERN.search(value), (name, key, value)
        # serialization must not contain legacy version tokens at all
        blob = spec_canonical_json(spec)
        for token in ("descriptor_schema", "lga_schema", "builder_version",
                      "rdkit_canonical_psmiles_v1", "schema_version"):
            assert token not in blob, (name, token)
    # module seed spec and build_spec seed policy agree exactly
    assert GEOMETRY_SEED_SPEC == TRIMER_SEED_POLICY["geometry_seed_spec"]


def test_f2_md200_descriptor_is_semantic():
    assert "descriptor" in MD200_BUILD_SPEC["parameters"]
    descriptor = MD200_BUILD_SPEC["parameters"]["descriptor"]
    for key in ("family", "dimension", "source_transform",
                "nonfinite_policy", "drop_first_return_value"):
        assert key in descriptor, key
    assert descriptor["dimension"] == 200
    assert descriptor["source_transform"] == "star_sub"
    for banned in ("descriptor_schema", "components", "protocol"):
        assert banned not in MD200_BUILD_SPEC["parameters"]


def test_f3_canonicalization_is_semantic():
    params = RU_BASE_BUILD_SPEC["parameters"]
    assert "normalization" not in params
    canonicalization = params["canonicalization"]
    for key in ("toolkit", "canonical_output_smiles", "isomeric",
                "invalid_identity_policy"):
        assert key in canonicalization, key


def test_f4_seed_stable_for_same_geometry_semantics():
    identity = "canonical-trimer-identity"
    expected_material = ":".join((
        spec_canonical_json(TRIMER_SEED_POLICY["geometry_seed_spec"]),
        identity, "0",
    ))
    expected = int.from_bytes(
        hashlib.sha256(expected_material.encode()).digest()[:4], "little"
    ) & 0x7FFFFFFF
    assert _geometry_seed(identity, 0) == expected
    assert _geometry_seed(identity, 0) == _geometry_seed(identity, 0)


def test_f5_seed_changes_with_sample_identity():
    assert _geometry_seed("identity-a", 0) != _geometry_seed("identity-b", 0)


def test_f6_seed_rounds_differ():
    seeds = {_geometry_seed("identity-a", round_id) for round_id in (-1, 0, 1)}
    assert len(seeds) == 3


def test_f7_seed_changes_with_geometry_parameters():
    identity, round_id = "identity-a", 0
    base_material = ":".join((
        spec_canonical_json(GEOMETRY_SEED_SPEC), identity, str(round_id),
    ))
    changed_spec = json.loads(json.dumps(GEOMETRY_SEED_SPEC))
    changed_spec["candidates_per_round"] = 4
    changed_material = ":".join((
        spec_canonical_json(changed_spec), identity, str(round_id),
    ))
    derive = lambda material: int.from_bytes(
        hashlib.sha256(material.encode()).digest()[:4], "little"
    ) & 0x7FFFFFFF
    assert derive(base_material) != derive(changed_material)
    # and the real trimer route hash changes with the seed policy
    changed_policy = json.loads(json.dumps(TRIMER_SEED_POLICY))
    changed_policy["geometry_seed_spec"]["candidates_per_round"] = 4
    changed_spec_full = json.loads(json.dumps(ROUTE_BUILD_SPECS["trimer"]))
    changed_spec_full["parameters"]["seed_policy"] = changed_policy
    assert build_spec_hash(ROUTE_BUILD_SPECS["trimer"]) != build_spec_hash(
        changed_spec_full
    )


def test_f8_fold_split_never_affects_seed_or_spec():
    blob = spec_canonical_json(GEOMETRY_SEED_SPEC)
    for token in ("fold", "split", "scaffold", "use_idxs", "worker", "time"):
        assert f'"{token}"' not in blob
    assert not any(
        "fold" in item or "split" in item or "scaffold" in item
        for item in TRIMER_SEED_POLICY["material"]
    )
    with pytest.raises(SpecError):
        validate_build_spec({"artifact_type": "trimer", "fold": 1})


def test_f9_reader_zero_write_on_frozen_world(frozen_world):
    cache_root = frozen_world["tmp"]

    def snapshot():
        state = {}
        for base, _, files in os.walk(cache_root):
            for name in files:
                path = Path(base) / name
                stat = path.stat()
                state[str(path.relative_to(cache_root))] = (
                    stat.st_size, stat.st_mtime_ns,
                )
        return state

    before = snapshot()
    store, _ = open_frozen_feature_store(
        cache_root, ("ru_base", "topology", "trimer", "md200"),
        cohort=frozen_world["cohort"], validate_coverage="full",
    )
    for key in frozen_world["keys"]:
        _ = store[key]
    store.close()
    assert snapshot() == before  # zero creation/modification


def test_f10_missing_derived_array_fails_without_materialization(tmp_path):
    keys = [sample_key_from_smiles("*CC*")]
    spec = ROUTE_BUILD_SPECS["md200"]
    md_dir = tmp_path / "md200" / "artifact-a"
    write_artifact(md_dir, "md200", spec,
                   {keys[0]: make_md200_record()})
    topo_dir = tmp_path / "topology" / "artifact-a"
    write_artifact(topo_dir, "topology", ROUTE_BUILD_SPECS["topology"],
                   {keys[0]: make_topology_record(keys[0])})
    make_store(tmp_path, {
        "topology": _binding("topology/artifact-a",
                             ROUTE_BUILD_SPECS["topology"], 1),
        "md200": _binding("md200/artifact-a", spec, 1),
    })
    cohort = cohort_from_keys(keys, root=tmp_path / "cohorts" / "test")
    with pytest.raises(CacheMissingDerivedArtifact):
        open_frozen_feature_store(tmp_path, ("topology", "md200"), cohort=cohort)
    # nothing was materialized on the read path
    assert not (tmp_path / "cohorts" / "test" / f"md200_{build_spec_hash(spec)}").exists()


def _register(tmp_path, layers_spec_records, extra_metadata=None):
    """Registration fixture using HISTORICAL-format build_config metadata
    (same shape as the real artifacts on disk), so the semantic alias path
    in build_spec_from_metadata is exercised."""

    legacy_build_configs = {
        "ru_base": {
            "attachment_site_policy": "two_sites_shared_boundary_allowed",
            "boundary_distance_algorithm": "expanded_graph_shortest_path",
            "mismatched_bond_policy": "single",
            "molecule_binary": "rdkit_mol_binary",
            "multimer_builder_version": 2,
            "normalization": "rdkit_canonical_psmiles_v1",
        },
        "topology": {
            "atom_features": "mips137_topology_only_trimer_central_ru",
            "boundary_distance_algorithm": "diagnostic_only_canonical_ru",
            "boundary_threshold": 5,
            "builder_version": 12,
            "feature_content_schema": "mts-canonical-periodic-feature-v3",
            "lga_schema": 2,
            "max_hops": 2,
            "max_model_atoms": 384,
            "max_repeat_units": 1,
            "mismatched_bond_policy": "single",
            "ru_base_feature_config_hash": "0" * 64,
            "topology_representation": "single_canonical_ru_lifted_relations",
        },
        "trimer": {
            "allow_2d_for_mcl": False,
            "attachment_site_policy": "two_sites_shared_boundary_allowed",
            "builder_version": 12,
            "conformer_selection": "lowest-finite-mmff-energy",
            "etkdg_max_iterations": 42,
            "etkdg_retry_candidates": 2,
            "etkdg_retry_max_iterations": 200,
            "max_heavy_atoms": 384,
            "mismatched_bond_policy": "single",
            "mmff_relax_max_iterations": 200,
            "mmff_variant": "MMFF94",
            "multimer_builder_version": 2,
            "num_candidates": 4,
            "protocol": "etkdgv3x4-mmff94-relax200-lowest-finite-v1",
            "ru_base_feature_config_hash": "0" * 64,
            "topology_feature_config_hash": "0" * 64,
            "trimer_content_schema": "mips-trimer-scage-trimer-v8",
            "trimer_schema_version": 8,
            "worker_hard_timeout_seconds": 240,
        },
        "md200": {
            "components": ["RDKit2DNormalized200"],
            "descriptor_schema": 5,
            "embedding": "none",
            "optimizer": "none",
            "protocol": "source_star_sub",
        },
    }
    bindings = {}
    for name, (dirname, spec, records, metadata_extra) in layers_spec_records.items():
        artifact_dir = tmp_path / dirname / "artifact-a"
        write_artifact(artifact_dir, name, spec, records)
        # replace builder-style metadata with the historical metadata shape
        metadata = {
            "artifact_type": name,
            "build_config": dict(legacy_build_configs[name]),
            "rdkit_version": "2026.03.2",
        }
        metadata.update(metadata_extra or {})
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata))
        (artifact_dir / ".done").unlink(missing_ok=True)  # .done is optional
        bindings[name] = (dirname, spec)
    from scripts.create_cache_store import collect_active_bindings
    return collect_active_bindings(tmp_path, {})


def test_f11_legacy_inferred_build_config_is_not_registrable(tmp_path):
    keys = [sample_key_from_smiles("*CC*")]
    layers = {
        "ru_base": ("ru_base", ROUTE_BUILD_SPECS["ru_base"],
                    {keys[0]: make_ru_base_record(keys[0])}, None),
        "topology": ("topology", ROUTE_BUILD_SPECS["topology"],
                     {keys[0]: make_topology_record(keys[0])}, None),
    }
    with pytest.raises(StoreError, match="explicit build_spec"):
        _register(tmp_path, layers)


def test_f12_legacy_scanner_does_not_infer_incompatible_artifact(tmp_path):
    keys = [sample_key_from_smiles("*CC*")]
    retired_spec = json.loads(json.dumps(ROUTE_BUILD_SPECS["trimer"]))
    retired_spec["parameters"]["num_candidates_per_round"] = 4
    layers = {
        "ru_base": ("ru_base", ROUTE_BUILD_SPECS["ru_base"],
                    {keys[0]: make_ru_base_record(keys[0])}, None),
        "topology": ("topology", ROUTE_BUILD_SPECS["topology"],
                     {keys[0]: make_topology_record(keys[0])}, None),
        "trimer": ("trimer", retired_spec,
                   {keys[0]: make_trimer_record(keys[0])}, None),
    }
    with pytest.raises(StoreError, match="explicit build_spec"):
        _register(tmp_path, layers)


def test_f13_legacy_version_strings_do_not_restore_compatibility(tmp_path):
    keys = [sample_key_from_smiles("*CC*")]
    legacy_metadata = {
        "schema": "mts-canonical-periodic-topology-lmdb-v3",
        "feature_schema": "mts-canonical-periodic-feature-v3",
        "builder_version": 12,
        "cache_layout_schema": "mips-trimer-scage-lmdb-layout-v2",
        "migration_schema": "mts-canonical-cache-migration-v3",
    }
    layers = {
        "ru_base": ("ru_base", ROUTE_BUILD_SPECS["ru_base"],
                    {keys[0]: make_ru_base_record(keys[0])}, None),
        "topology": ("topology", ROUTE_BUILD_SPECS["topology"],
                     {keys[0]: make_topology_record(keys[0])}, legacy_metadata),
    }
    with pytest.raises(StoreError, match="explicit build_spec"):
        _register(tmp_path, layers)
