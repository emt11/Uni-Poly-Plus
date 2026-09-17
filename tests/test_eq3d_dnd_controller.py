"""Focused completion-gate contracts for the post-teacher controller."""

import json
from types import SimpleNamespace

import scripts.run_eq3d_dnd_post_teacher as controller
from scripts.report_poly_painn_teacher_health import _records


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


def test_health_records_accept_concatenated_rank_records(tmp_path):
    log = tmp_path / "teacher.log"
    log.write_text(
        json.dumps({"step": 1, "rank": 0, "noise_mse": 1.0})
        + json.dumps({"step": 2, "rank": 0, "noise_mse": 0.9})
        + json.dumps({"step": 3, "rank": 1, "noise_mse": 0.8}),
        encoding="utf-8",
    )
    assert [row["step"] for row in _records(log)] == [1, 2]


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


def test_stop_flushes_when_stage_already_failed(tmp_path):
    args = SimpleNamespace(
        result_root=str(tmp_path / "result"),
        teacher_root=str(tmp_path / "teacher"),
        teacher_log=str(tmp_path / "teacher.log"),
        teacher_runtime_source_commit="teacher-commit",
        automation_implementation_commit="automation-commit",
    )
    instance = controller.Controller(args)
    instance.state = {"status": "RUNNING", "final_status": None,
                      "stages": {"TEACHER_HEALTH": {"status": "FAIL"}}}
    instance.stop("STOPPED_TEACHER_HEALTH_FAIL", "TEACHER_HEALTH")
    persisted = json.loads(instance.state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "STOPPED"
    assert persisted["final_status"] == "STOPPED_TEACHER_HEALTH_FAIL"
