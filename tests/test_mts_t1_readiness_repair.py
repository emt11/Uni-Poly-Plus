"""Contract-bound tests for the T1 readiness repair entry points."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts.train import (
    _is_mts_t1_function_preserving_init,
    _validate_mts_t1_function_preserving_init,
)
from scripts.pretrain import _is_t1_function_preserving_init_payload


ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / (
    "pretrained_models/mts_multiscale_topology/t1_init/"
    "mts_t1_function_preserving_init.pth"
)
SHARD = ROOT / "results/mts_multiscale_topology/t1_repair/finetune_smoke/shards/42/eat/fold_0.csv"


@pytest.fixture(scope="module")
def init_payload():
    if not INIT.is_file():
        pytest.skip("T1 init artifact is not present in this checkout")
    return torch.load(INIT, map_location="cpu", weights_only=False)


def _identity_args(meta, *, allow=True, parent_sha=None):
    return SimpleNamespace(
        allow_mts_t1_function_preserving_init=allow,
        topology_attention_variant="msta_last2",
        graph_model_config_hash=meta["graph_model_config_hash"],
        resolved_config_hash=meta["target_config_hash"],
        msta_layer_indices=[4, 5],
        msta_local_spd=[0, 1],
        msta_context_spd=[0, 1, 2],
        msta_share_relation_dropout=True,
        msta_local_output_bias=False,
        msta_local_output_init="zero",
    )


def test_t1_init_marker_and_step_semantics(init_payload):
    meta = init_payload["meta"]
    assert _is_mts_t1_function_preserving_init(meta)
    assert meta["model_identity"] == "T1"
    assert meta["source_model_identity"] == "T0"
    assert meta["initialization"] == "function_preserving"
    assert meta["source_optimizer_steps"] == 20000
    assert meta["optimizer_steps"] == 0


def test_t1_init_payload_is_not_a_pretrain_resume_state(init_payload):
    assert _is_t1_function_preserving_init_payload(init_payload)
    assert not _is_t1_function_preserving_init_payload({"schema": "train-state-v1"})


def test_t1_init_without_opt_in_is_rejected(init_payload):
    meta = init_payload["meta"]
    with pytest.raises(RuntimeError, match="explicit.*allow_mts_t1"):
        _validate_mts_t1_function_preserving_init(
            init_payload,
            INIT,
            args=_identity_args(meta, allow=False),
            dataset=None,
            pretraining_cohort_hash=None,
            expected_angle_artifact=None,
            store_json_sha256=None,
            topology_frozen_payload_sha256=None,
            trimer_frozen_payload_sha256=None,
            model=None,
        )


def test_corrupt_parent_sha_is_rejected_before_cache_validation(init_payload):
    meta = dict(init_payload["meta"])
    meta["parent_checkpoint_sha256"] = "0" * 64
    corrupted = {"state_dict": init_payload["state_dict"], "meta": meta}
    with pytest.raises(RuntimeError, match="parent checkpoint SHA256 mismatch"):
        _validate_mts_t1_function_preserving_init(
            corrupted,
            INIT,
            args=_identity_args(meta),
            dataset=None,
            pretraining_cohort_hash=None,
            expected_angle_artifact=None,
            store_json_sha256=None,
            topology_frozen_payload_sha256=None,
            trimer_frozen_payload_sha256=None,
            model=None,
        )


def test_init_local_outputs_are_zero_and_bias_free(init_payload):
    state = init_payload["state_dict"]
    for index in (4, 5):
        weight = state[
            f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"
        ]
        assert weight.shape == (512, 512)
        assert torch.count_nonzero(weight) == 0
        assert (
            f"encoders.graph.encoder.layers.{index}.attention.local_output.bias"
            not in state
        )


def test_production_smoke_row_records_opt_in_and_strict_identity():
    if not SHARD.is_file():
        pytest.skip("production T1 repair smoke has not been run")
    row = pd.read_csv(SHARD).iloc[0]
    assert bool(row["checkpoint_init_opt_in"])
    assert row["checkpoint_model_identity"] == "T1"
    assert row["checkpoint_initialization"] == "function_preserving"
    assert int(row["checkpoint_optimizer_steps"]) == 0
    assert int(row["checkpoint_source_optimizer_steps"]) == 20000
    assert row["graph_model_config_hash"] != "cfd8730b4f85301c128aca25dde4b39069d13beb783a251cbd4f78662bf124db"
