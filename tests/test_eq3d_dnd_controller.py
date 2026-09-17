"""Focused completion-gate contracts for the post-teacher controller."""

import json
from types import SimpleNamespace

import scripts.run_eq3d_dnd_post_teacher as controller


def _publish_teacher_checkpoints(root):
    root.mkdir()
    for step in range(1000, 5001, 1000):
        (root / f"resume_{step:05d}.pt").touch()
        (root / f"teacher_deploy_{step:05d}.pt").touch()


def test_logged_steps_accept_concatenated_rank_records(tmp_path):
    log = tmp_path / "teacher.log"
    log.write_text(
        "torch warning\n"
        + json.dumps({"step": 1, "rank": 0})
        + json.dumps({"step": 2, "rank": 0})
        + "\n"
        + json.dumps({"step": 3, "rank": 1})
        + json.dumps({"step": 4, "rank": 0})
        + "\n",
        encoding="utf-8",
    )
    assert controller._logged_steps(log) == [1, 2, 4]


def test_teacher_complete_accepts_interleaved_json_records(tmp_path, monkeypatch):
    root = tmp_path / "teacher"
    _publish_teacher_checkpoints(root)
    log = tmp_path / "teacher.log"
    log.write_text(
        "".join(json.dumps({"step": step, "rank": 0}) for step in range(1, 5001)),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller, "_teacher_processes", lambda _: [])
    assert controller._teacher_complete(root, log)


def test_wait_teacher_resume_reuses_only_matching_running_stage(tmp_path, monkeypatch):
    teacher_root = tmp_path / "teacher"
    _publish_teacher_checkpoints(teacher_root)
    teacher_log = tmp_path / "teacher.log"
    teacher_log.write_text(
        "".join(json.dumps({"step": step, "rank": 0}) for step in range(1, 5001)),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        result_root=str(tmp_path / "result"),
        teacher_root=str(teacher_root),
        teacher_log=str(teacher_log),
        teacher_runtime_source_commit="teacher-commit",
        automation_implementation_commit="automation-commit",
    )
    instance = controller.Controller(args)
    command = ["wait", str(teacher_root.resolve())]
    inputs = {"teacher_root": str(teacher_root.resolve()),
              "teacher_log": str(teacher_log.resolve())}
    instance.state = {"stages": {"WAIT_TEACHER": {
        "stage": "WAIT_TEACHER", "status": "RUNNING",
        "command": command, "input_artifacts": inputs,
    }}}
    monkeypatch.setattr(controller, "_teacher_processes", lambda _: [])
    assert instance.wait_teacher() is True
    assert instance.state["stages"]["WAIT_TEACHER"]["status"] == "PASS"
