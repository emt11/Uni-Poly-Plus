import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
import math

import scripts.report_mts_glt_distill_repair as repair_report
from scripts.build_mts_glt_v3_sidecars import dataset_for_build, row_key
from scripts.create_mips_split_manifests import build_manifest
from scripts.run_mts_glt_distill_repair_pipeline import _latest_resume
from scripts.run_mts_finetune_scheduler import _valid_shard
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_distill_v2 import build_periodic_line_distill_v2_sample
from src.modules.mts_glt_distill import AtomicConditionedMD200, NPlusGLTTeacher
from src.training.pretrain.glt_distill_engine import (
    StudentContainer, TeacherContainer, _deploy_student, exact_center_mask,
)
from src.training.finetune.engine import select_mts_glt_distill_state
from src.training.finetune.scheduler import ScheduledUnit


N_ZERO_SAMPLE_INDEX = 10
N_ZERO_SAMPLE_KEY_HEX = "2320ad8ec663f0ca98eb48191b15e8b28b0f1c401a5142be68a286e990fbcfd1"


def _attach_revision2(data, row, sample_key):
    data = copy.copy(data)
    data.glt3_geometry_valid = bool(row["geometry_valid"])
    for name, value in row["tokens"].items():
        dtype = (
            torch.float32 if name == "token_distance" else
            torch.bool if name in {"token_valid", "token_center_internal"} else
            torch.long
        )
        setattr(data, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
    for name, value in row["relations"].items():
        dtype = (
            torch.float32 if name == "relation_angle" else
            torch.bool if name == "relation_valid" else torch.long
        )
        setattr(data, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
    data.mips_md = torch.zeros(200, dtype=torch.float32)
    data.mips_md_valid = torch.tensor(True)
    data.mts_sample_hash64 = torch.tensor(
        int.from_bytes(sample_key[:8], "little") & ((1 << 63) - 1),
        dtype=torch.long,
    )
    return data


def _assert_partition(manifest):
    all_tests = []
    for fold in manifest["folds"]:
        train, val, test = map(
            set, (fold["train_indices"], fold["validation_indices"], fold["test_indices"])
        )
        assert not (train & val or train & test or val & test)
        assert len(train | val | test) == manifest["sample_count"]
        all_tests.extend(test)
    assert len(all_tests) == len(set(all_tests)) == manifest["sample_count"]


@pytest.mark.parametrize("count,expected", [(300, (192, 48, 60)), (302, None)])
def test_outer5_inner20_partitions_are_separated_and_deterministic(tmp_path, count, expected):
    csv = tmp_path / "smi_x.csv"
    pd.DataFrame({"smiles": [f"*C{'C' * i}*" for i in range(count)]}).to_csv(csv, index=False)
    first = build_manifest("x", csv, "outer5_inner20")
    second = build_manifest("x", csv, "outer5_inner20")
    assert first == second
    _assert_partition(first)
    for fold in first["folds"]:
        outer_train = count - len(fold["test_indices"])
        assert len(fold["validation_indices"]) == math.ceil(0.20 * outer_train)
        assert len(fold["train_indices"]) == outer_train - math.ceil(0.20 * outer_train)
    if expected is not None:
        assert all(
            (len(f["train_indices"]), len(f["validation_indices"]), len(f["test_indices"])) == expected
            for f in first["folds"]
        )


def test_outer5_inner20_completed_unit_requires_exact_test_order(tmp_path):
    results = tmp_path / "results"
    manifests = tmp_path / "splits"
    shard = results / "shards/42/x/fold_0.csv"
    prediction = results / "predictions/42/x/fold_0.npz"
    checkpoint = results / "shards/42/x/fold_0_best.pt"
    for path in (shard, prediction, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests / "x.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"folds": [{"test_indices": [3, 1]}]}))
    split_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pd.DataFrame([{
        "task": "x", "seed": 42, "evaluation_protocol": "outer5_inner20",
        "split_manifest_sha256": split_hash,
        "per_fold_metrics": json.dumps([{"fold": 0}]),
    }]).to_csv(shard, index=False)
    metadata = np.asarray(json.dumps({
        "task": "x", "seed": 42, "fold": 0,
        "fold_validation_protocol": "outer5_inner20",
        "split_manifest_sha256": split_hash,
    }))
    np.savez(
        prediction, y_true=np.asarray([1.0, 2.0]), y_pred=np.asarray([1.1, 1.9]),
        sample_indices=np.asarray([3, 1]), metadata=metadata,
    )
    torch.save({
        "evaluation_protocol": "outer5_inner20",
        "split_manifest_sha256": split_hash,
    }, checkpoint)
    unit = ScheduledUnit(seed=42, task="x", fold=0)
    assert _valid_shard(results, unit, protocol="outer5_inner20", split_manifest_dir=manifests)
    np.savez(
        prediction, y_true=np.asarray([1.0, 2.0]), y_pred=np.asarray([1.1, 1.9]),
        sample_indices=np.asarray([1, 3]), metadata=metadata,
    )
    assert not _valid_shard(results, unit, protocol="outer5_inner20", split_manifest_dir=manifests)


@pytest.fixture(scope="module")
def frozen_examples():
    args = type("Args", (), {"cache_root": "data", "dataset": "PI1M_v2"})()
    dataset = dataset_for_build(args)
    ordinary, n_zero = dataset[1], dataset[N_ZERO_SAMPLE_INDEX]
    keys = (row_key(dataset, 1, ordinary), row_key(dataset, N_ZERO_SAMPLE_INDEX, n_zero))
    return ordinary, n_zero, keys


def test_revision2_uses_distinct_cross_lengths_and_center_relations(frozen_examples):
    ordinary, _, _ = frozen_examples
    one = build_periodic_line_distill_v2_sample(ordinary, ordinary, ordinary.smiles, "n_plus_1")
    two = build_periodic_line_distill_v2_sample(ordinary, ordinary, ordinary.smiles, "n_plus_2")
    n = int(two["tokens"]["token_center_internal"].sum())
    assert len(one["tokens"]["token_atom_a"]) == n + 1
    assert len(two["tokens"]["token_atom_a"]) == n + 2
    left, right = two["tokens"]["token_distance"][-2:]
    assert one["tokens"]["token_distance"][-1] == pytest.approx((left + right) / 2)
    assert np.isfinite(two["relations"]["relation_angle"]).all()


def test_revision2_same_atom_double_link_relations_are_retained(frozen_examples):
    _, same_atom, (_, sample_key) = frozen_examples
    assert sample_key.hex() == N_ZERO_SAMPLE_KEY_HEX
    assert int(same_atom.ru_left_boundary) == int(same_atom.ru_right_boundary)
    one = build_periodic_line_distill_v2_sample(same_atom, same_atom, same_atom.smiles, "n_plus_1")
    two = build_periodic_line_distill_v2_sample(same_atom, same_atom, same_atom.smiles, "n_plus_2")
    # This frozen example has no center-RU internal bond (N=0).  It still has
    # canonical atoms for the student's masked-atom objective, while the line
    # teacher exposes only the periodic cross-bond state(s), never a fabricated
    # center-bond distillation target.
    assert len(torch.as_tensor(same_atom.atomic_numbers)) > 0
    assert int(one["tokens"]["token_center_internal"].sum()) == 0
    assert int(two["tokens"]["token_center_internal"].sum()) == 0
    assert len(one["tokens"]["token_atom_a"]) == 1
    assert len(two["tokens"]["token_atom_a"]) == 2
    assert list(zip(two["relations"]["relation_source"], two["relations"]["relation_target"])) == [(0, 1), (1, 0)]
    assert len(one["relations"]["relation_source"]) == 2
    assert np.all(one["relations"]["relation_source"] == one["relations"]["relation_target"])


def test_real_n_zero_collate_and_teacher_student_loss_exclude_distillation(frozen_examples):
    ordinary, n_zero, (ordinary_key, n_zero_key) = frozen_examples
    zero_row = build_periodic_line_distill_v2_sample(
        n_zero, n_zero, n_zero.smiles, "n_plus_2",
    )
    ordinary_row = build_periodic_line_distill_v2_sample(
        ordinary, ordinary, ordinary.smiles, "n_plus_2",
    )
    zero_item = _attach_revision2(n_zero, zero_row, n_zero_key)
    ordinary_item = _attach_revision2(ordinary, ordinary_row, ordinary_key)

    zero_batch = mips_trimer_collate([zero_item])
    selected = exact_center_mask(
        zero_batch, 1.0, torch.Generator().manual_seed(7),
    )
    assert not bool(selected.any())
    teacher_container = TeacherContainer().eval()
    teacher_readout = teacher_container.teacher(zero_batch)
    assert teacher_readout["center_projected"].shape == (0, 256)
    assert teacher_readout["center_batch"].numel() == 0
    teacher_loss = teacher_container(
        zero_batch, selected=selected,
    )
    assert teacher_loss["graph_count"] == 0
    assert teacher_loss["angle_graph_count"] == 0
    assert teacher_loss["masked_lines"] == 0
    assert all(
        torch.isfinite(teacher_loss[name])
        and float(teacher_loss[name].detach().item()) == 0.0
        for name in ("chem_sum", "length_sum", "angle_sum")
    )

    student = StudentContainer(NPlusGLTTeacher(dropout=0.0)).eval()
    atom_mask = torch.zeros(zero_batch.mips_x.size(0), dtype=torch.bool)
    atom_mask[0] = True
    zero_output = student(
        zero_batch, 0, generator=torch.Generator().manual_seed(11),
        atom_mask=atom_mask,
    )
    assert zero_output["atom_count"] == 1
    assert torch.isfinite(zero_output["atom_sum"])
    assert zero_output["local_count"] == zero_output["valid_graphs"] == 0
    assert zero_output["global_pool"] == 0
    assert torch.isfinite(zero_output["local_sum"])
    assert torch.isfinite(zero_output["global"])
    zero_output["atom_sum"].backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in student.atom_head.parameters()
    )

    mixed_batch = mips_trimer_collate([zero_item, ordinary_item])
    mixed_mask = torch.zeros(mixed_batch.mips_x.size(0), dtype=torch.bool)
    mixed_mask[0] = True
    mixed_mask[int((mixed_batch.canonical_graph_index == 0).sum())] = True
    mixed_output = student(
        mixed_batch, 1, generator=torch.Generator().manual_seed(13),
        atom_mask=mixed_mask,
    )
    assert mixed_output["atom_count"] == 2
    assert mixed_output["local_count"] == 1
    assert mixed_output["valid_graphs"] == 1
    assert mixed_output["global_pool"] == 1
    assert all(
        torch.isfinite(mixed_output[name])
        for name in ("atom_sum", "local_sum", "global")
    )


def test_invalid_md_never_enters_encoder_and_valid_nonfinite_fails():
    module = AtomicConditionedMD200(dropout=0.0).eval()
    data = type("Batch", (), {})()
    data.mips_md = torch.stack([torch.ones(200), torch.full((200,), torch.nan)])
    data.mips_md_valid = torch.tensor([True, False])
    data.canonical_graph_index = torch.tensor([0, 1])
    states = torch.randn(2, 512)
    output = module(states, data)
    torch.testing.assert_close(output[1], states[1])
    data.mips_md_valid[:] = True
    with pytest.raises(ValueError, match="NaN/Inf"):
        module(states, data)


def test_student_train_keeps_frozen_teacher_eval():
    container = StudentContainer(NPlusGLTTeacher())
    container.train()
    assert container.training
    assert not container.teacher.training
    assert not any(p.requires_grad for p in container.teacher.parameters())


def test_c0_repair_deploy_uses_repair_schema_without_geometry():
    payload = _deploy_student(
        StudentContainer(None), 10000, "none", None, repair_experiment=True
    )
    assert payload["schema"] == "mts-glt-distill-repair-student-deploy-v1"
    assert payload["geometry_revision"] is None


def test_latest_resume_prefers_rolling_repair_checkpoint(tmp_path):
    common = {
        "schema": "mts-glt-distill-repair-student-state-v1",
        "version": "n_plus_2",
        "geometry_revision": 2,
    }
    torch.save({**common, "step": 7000}, tmp_path / "student_007k.pt")
    torch.save({**common, "step": 7250}, tmp_path / "student_resume_latest.pt")
    assert _latest_resume(tmp_path, "student", "n_plus_2", 20000).name == "student_resume_latest.pt"


def test_teacher_construction_can_be_rng_isolated_for_matched_student_init():
    torch.manual_seed(42)
    control = StudentContainer(None)
    torch.manual_seed(42)
    with torch.random.fork_rng(devices=[]):
        teacher = NPlusGLTTeacher()
    treated = StudentContainer(teacher)
    control_state = control.state_dict()
    treated_state = treated.state_dict()
    common = [key for key in control_state if key.startswith(("student.", "atom_head."))]
    assert common
    for key in common:
        torch.testing.assert_close(control_state[key], treated_state[key], rtol=0, atol=0)


def test_repair_deployment_rejects_wrong_version_and_revision():
    model = StudentContainer(None).student
    model_state = {"encoders.graph.encoder." + key: value for key, value in model.state_dict().items()}
    checkpoint = {
        "schema": "mts-glt-distill-repair-student-deploy-v1",
        "step": 20000, "version": "n_plus_1", "geometry_revision": 2,
        "state_dict": model.state_dict(),
    }
    assert len(select_mts_glt_distill_state(model_state, checkpoint, "n_plus_1")) == len(model_state)
    with pytest.raises(RuntimeError, match="version mismatch"):
        select_mts_glt_distill_state(model_state, checkpoint, "n_plus_2")
    checkpoint["geometry_revision"] = 1
    with pytest.raises(RuntimeError, match="geometry revision"):
        select_mts_glt_distill_state(model_state, checkpoint, "n_plus_1")


def test_final_report_requires_n_zero_resume_and_unit_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_report, "ROOT", tmp_path)
    log_root = tmp_path / "logs/mts_glt_distill_repair_control"
    log_root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="required validation evidence"):
        repair_report.validation_evidence()

    (log_root / "n0_rank_smoke_20260909.log").write_text(
        "N_ZERO_RANK_DDP_PASS "
        f"sample_key={N_ZERO_SAMPLE_KEY_HEX}\n"
    )
    (log_root / "resume_current_v5_20260909.log").write_text(
        "RESUME_CURRENT_NUMERICAL_MATCH step=4 model_atol=2e-6 "
        "optimizer_atol=2e-5 rng_exact=true abandoned_tail_preserved=true\n"
    )
    (log_root / "tests_final_20260909.log").write_text(
        "22 passed, 1 warning\n"
    )
    evidence = repair_report.validation_evidence()
    assert evidence["n_zero"]["sample_key"] == N_ZERO_SAMPLE_KEY_HEX
    assert evidence["n_zero"]["empty_ddp_rank"] is True
    assert evidence["resume"]["completed_steps"] == 4
    assert evidence["resume"]["rng_exact"] is True
    assert evidence["resume"]["abandoned_log_tail_preserved"] is True
    assert evidence["unit_tests"]["passed"] == 22
