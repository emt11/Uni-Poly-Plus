"""Validate the single deterministic paper5_outer split contract."""

import json
from pathlib import Path

import pytest

from scripts.create_mips_split_manifests import build_manifest
from scripts.finetune_mcl_ph import PAPER_SPLIT_PROTOCOL, TASKS, validate_stage_scope
from src.training.split_protocol import resolve_fold


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/mips_paper5_outer"
OFFICIAL = ROOT / "configs/mts/periodic_tdl_official5"


def test_generated_manifests_match_released_outer_folds_for_five_tasks():
    for task in ("eea", "egb", "ei", "eps", "nc"):
        generated = json.loads((SPLITS / f"{task}.json").read_text())
        released = json.loads((OFFICIAL / f"{task}.json").read_text())
        assert generated["protocol"] == PAPER_SPLIT_PROTOCOL
        assert generated["splitter"]["random_state"] == 100
        assert generated["inner_splitter"]["random_state"] == 42
        for generated_fold, released_fold in zip(generated["folds"], released["folds"]):
            for field in ("fold", "train_indices", "validation_indices", "test_indices",
                          "inner_split_random_state"):
                assert generated_fold[field] == released_fold[field]
            assert generated_fold["sets_disjoint"] is True
            assert generated_fold["union_equals_full_cohort"] is True


def test_generator_is_deterministic_for_all_eight_tasks(tmp_path):
    for task in TASKS:
        source = ROOT / "data/raw" / f"smi_{task}.csv"
        first = build_manifest(task, source)
        second = build_manifest(task, source)
        assert first == second
        (tmp_path / f"{task}.json").write_text(json.dumps(first))
        for fold in first["folds"]:
            train, validation, evidence = resolve_fold(
                first, task, fold["fold"], cohort_rows=first["sample_count"],
                expected_protocol=PAPER_SPLIT_PROTOCOL)
            assert set(train).isdisjoint(validation)
            assert evidence["outer_test"] == "NOT_RUN"


def test_paper_scope_accepts_all_active_tasks():
    for task in TASKS:
        epochs = 60 if task == "egc" else 70
        validate_stage_scope("paper5_outer", task, 0, epochs, "trusted-index")
    with pytest.raises(ValueError):
        validate_stage_scope("full8x5_outer", "eea", 0, 70, "trusted-index")


def test_paper_scope_requires_trusted_index():
    with pytest.raises(ValueError, match="trusted cohort index"):
        validate_stage_scope("paper5_outer", "eea", 0, 70, None)
