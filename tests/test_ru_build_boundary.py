"""RU builder capability-boundary tests (RU_BUILD_UNSUPPORTED).

A canonical, parseable P-SMILES whose RU mapping search validates zero graph
isomorphisms is an ordinary RU rejection: the build continues, no RU record,
no Topology record and no Trimer participation is produced for it, and the
whole-build hard-stop scope is unchanged.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The real production sample that first exposed the capability boundary.
BOUNDARY_SMILES = "*C(=O)c1ccc(C(=O)N2c3ccc(C(*)=O)c(=O)c2c3)c(C(=O)O)c1"
BOUNDARY_KEY = "71dcb8645e69576a3db7831ba50694ff569af7f0893eda1f289a559ac88bae51"


def test_boundary_sample_is_canonical_and_deterministically_unsupported():
    from src.dataset.dataset import _compute_ru_base_layer
    from src.dataset.lmdb_cache import (
        normalize_polymer_smiles,
        sample_key_from_normalized,
    )

    normalized, valid = normalize_polymer_smiles(BOUNDARY_SMILES)
    assert valid is True
    assert sample_key_from_normalized(normalized).hex() == BOUNDARY_KEY

    first = _compute_ru_base_layer(BOUNDARY_SMILES)
    second = _compute_ru_base_layer(BOUNDARY_SMILES)
    for ru in (first, second):
        assert bool(ru.ru_base_valid) is False
        assert str(ru.ru_base_failure_code).startswith("RU_BUILD_UNSUPPORTED")
    assert first.ru_base_failure_code == second.ru_base_failure_code


def test_struct_worker_classifies_boundary_as_rejection_not_contract_error():
    from scripts.build_mts_cache import _build_struct_one

    result = _build_struct_one({
        "phase": "struct",
        "sample_key": BOUNDARY_KEY,
        "source_smiles": BOUNDARY_SMILES,
    })
    assert result["status"] == "rejected"
    assert result["failure_code"] == "RU_BUILD_UNSUPPORTED"
    assert "ru_payload" not in result and "topology_payload" not in result


def test_identity_corruption_still_hard_stops(monkeypatch):
    """The hard-stop scope is unchanged: a non-boundary invalid RU stays a
    contract error (program/data integrity), never a silent rejection."""
    from src.dataset import trimer_mcl
    from src.dataset.trimer_mcl import TrimerContractError
    from scripts.build_mts_cache import _build_struct_one

    import scripts.build_mts_cache as builder_module

    real = builder_module._compute_ru_base_layer

    def corrupted(smiles):
        ru = real(smiles)
        ru.ru_base_valid = False
        ru.ru_base_failure_code = "ValueError:synthetic_identity_contradiction"
        return ru

    monkeypatch.setattr(builder_module, "_compute_ru_base_layer", corrupted)
    result = _build_struct_one({
        "phase": "struct",
        "sample_key": "b" * 64,
        "source_smiles": "*CC*",
    })
    assert result["status"] == "contract_error"
    assert "source_identity_corruption" in result["error"]
    del trimer_mcl, TrimerContractError


def test_full_build_continues_past_boundary_sample_and_publishes(tmp_path):
    """End-to-end multi-worker run: the real boundary sample inside a normal
    cohort becomes one RU rejection; everything else proceeds and publishes
    with layered accounting."""

    from src.dataset.cache_lifecycle import load_source_rows

    source_csv = tmp_path / "with_boundary.csv"
    smiles_rows = [
        f"*C{'C' * a}(C){'O' * b}C*" for a in range(4) for b in range(10)
    ]
    smiles_rows.append(BOUNDARY_SMILES)  # the real capability boundary
    source_csv.write_text(
        "smiles\n" + "\n".join(smiles_rows) + "\n", encoding="utf-8"
    )
    rows, _ = load_source_rows(source_csv, 0)
    assert any(row["sample_key"] == BOUNDARY_KEY for row in rows)

    cache_root = tmp_path / "cache"
    import os
    import subprocess
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    result = subprocess.run(
        [sys.executable, "scripts/build_mts_cache.py",
         "--cache-root", str(cache_root), "--source-csv", str(source_csv),
         "--limit", "0", "--workers", "4"],
        capture_output=True, text=True, timeout=600, env=environment,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    store = json.loads((cache_root / "store.json").read_text(encoding="utf-8"))
    bundle = cache_root / "builds" / store["bundle_hash"]
    ru_manifest = json.loads((bundle / "ru_base" / "manifest.json").read_text())
    topo_manifest = json.loads((bundle / "topology" / "manifest.json").read_text())
    tri_manifest = json.loads((bundle / "trimer" / "manifest.json").read_text())

    assert ru_manifest["source_count"] == 41
    assert ru_manifest["rejected_count"] == 1
    assert ru_manifest["accepted_count"] == 40
    assert ru_manifest["source_count"] == (
        ru_manifest["accepted_count"] + ru_manifest["rejected_count"]
    )
    # topology passes the RU cohort through exactly
    assert topo_manifest["record_count"] == 40
    assert topo_manifest["source_count"] == 40
    # trimer accounting sits on the RU cohort
    assert tri_manifest["source_count"] == 40
    assert tri_manifest["source_count"] == (
        tri_manifest["accepted_count"] + tri_manifest["rejected_count"]
    )

    ru_rejections = (bundle / "ru_base" / "rejections.jsonl").read_text()
    assert BOUNDARY_KEY in ru_rejections
    assert "RU_BUILD_UNSUPPORTED" in ru_rejections

    # the boundary sample has no RU, Topology or Trimer record at all
    import numpy as np
    from src.dataset.cache_lifecycle import ReadonlyArtifact
    for layer in ("ru_base", "topology", "trimer"):
        artifact = ReadonlyArtifact(cache_root, store["artifacts"][layer])
        try:
            assert bytes.fromhex(BOUNDARY_KEY) not in artifact
        finally:
            artifact.close()
    ru_rejected = np.load(bundle / "ru_base" / "rejected_keys.npy")
    assert bytes(np.asarray(ru_rejected[0])).hex() == BOUNDARY_KEY
