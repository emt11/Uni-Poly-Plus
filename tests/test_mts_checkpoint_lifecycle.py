"""Small, runtime-only checkpoint lifecycle tests.

These tests intentionally cover atomic publication and strict final loading;
they do not inspect experiment, cache, or source identity metadata.
"""

from pathlib import Path

import pytest
import torch

from src.training.common.checkpoint import (
    TRAIN_STATE_SCHEMA,
    atomic_torch_save,
    load_resume_state,
    read_completion_marker,
    save_final_state,
)


def _resume_payload(step: int = 2) -> dict:
    return {
        "schema": TRAIN_STATE_SCHEMA,
        "meta": {"layout_version": 1, "world_size": 1},
        "train_module": {"weight": torch.ones(2)},
        "optimizer": {},
        "scheduler": {},
        "epoch": 0,
        "next_step_idx": step,
        "global_step": step,
        "optimizer_steps_completed": step,
        "rng_state_by_rank": [{}],
        "loader_generator_state_by_rank": [torch.zeros(1, dtype=torch.uint8)],
        "sampler_state_by_rank": [{"epoch": 0, "next_batch_index": step}],
    }


def test_last_state_is_atomically_replaced_and_readable(tmp_path: Path):
    target = tmp_path / "model.pth.last.pt"
    atomic_torch_save(_resume_payload(2), target)
    atomic_torch_save(_resume_payload(4), target)

    assert target.is_file()
    assert load_resume_state(target)["optimizer_steps_completed"] == 4
    assert sorted(tmp_path.glob(f"{target.name}.tmp.*")) == []


def test_failed_atomic_write_preserves_previous_state(tmp_path: Path, monkeypatch):
    target = tmp_path / "model.pth.last.pt"
    atomic_torch_save(_resume_payload(2), target)
    previous = target.read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected save failure"):
        atomic_torch_save(_resume_payload(3), target)

    assert target.read_bytes() == previous
    assert sorted(tmp_path.glob(f"{target.name}.tmp.*")) == []


def test_final_strict_load_publishes_only_minimal_marker(tmp_path: Path):
    target = tmp_path / "model.pth"
    seen = {}

    def strict_load(state_dict):
        seen.update(state_dict)
        assert set(state_dict) == {"weight"}
        assert state_dict["weight"].shape == (2,)

    save_final_state({"weight": torch.ones(2)}, target, strict_load=strict_load)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert set(loaded) == {"state_dict"}
    assert torch.equal(loaded["state_dict"]["weight"], torch.ones(2))
    assert target.with_name(target.name + ".complete.json").read_text() == (
        '{"status": "complete"}\n'
    )
    assert read_completion_marker(target) == {"status": "complete"}
    assert set(seen) == {"weight"}


def test_final_strict_failure_does_not_leave_marker(tmp_path: Path):
    target = tmp_path / "model.pth"
    marker = target.with_name(target.name + ".complete.json")
    marker.write_text('{"status": "complete"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        save_final_state(
            {"weight": torch.ones(2)},
            target,
            strict_load=lambda state: (_ for _ in ()).throw(
                RuntimeError("shape mismatch")
            ),
        )

    assert target.is_file()
    assert not marker.exists()
    assert read_completion_marker(target) is None
