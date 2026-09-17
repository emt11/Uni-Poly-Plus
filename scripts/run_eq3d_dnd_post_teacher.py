#!/usr/bin/env python3
"""Fail-closed post-teacher controller for EQ3D-DND-20260917-01.

The controller is intentionally small and experiment-specific.  It waits for
the already-running teacher, records every stage atomically, and only starts a
later stage after the previous stage's artifacts and gates have been checked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLAN_ID = "EQ3D-DND-20260917-01"
DEFAULT_RESULT_ROOT = ROOT / "results/eq3d_dnd_20260917"
DEFAULT_TEACHER_ROOT = DEFAULT_RESULT_ROOT / "teacher_formal_r2"
DEFAULT_TEACHER_LOG = ROOT / "logs/eq3d_teacher_formal_r2.log"
STAGES = (
    "WAIT_TEACHER", "TEACHER_HEALTH", "TEACHER_EQUIVARIANCE", "TEACHER_PROBE_7TASK",
    "TEACHER_GATE", "STUDENT_TESTS", "STUDENT_INPUT_PARITY", "STUDENT_SMOKE", "STUDENT_FORMAL_5K",
    "STUDENT_DEPLOY_VALIDATE", "ARM_B_EGC_5FOLD", "STUDENT_FINETUNE_SMOKE",
    "STUDENT_GRID_8X5", "AGGREGATE_8TASK",
)


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(path)).encode())
                with child.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        return digest.hexdigest()
    return None


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _teacher_processes(output_root):
    needle = str(Path(output_root).resolve())
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    result = []
    for row in rows:
        pieces = row.strip().split(None, 1)
        if not pieces:
            continue
        try:
            pid = int(pieces[0])
        except ValueError:
            continue
        command = pieces[1] if len(pieces) > 1 else ""
        if pid == os.getpid():
            continue
        if "pretrain_poly_painn_teacher.py" in command and needle in command:
            result.append({"pid": pid, "command": command})
    return result


def _logged_steps(log_path):
    steps = []
    path = Path(log_path)
    if not path.is_file():
        return steps
    decoder = json.JSONDecoder()
    for line in path.open(encoding="utf-8", errors="replace"):
        # torchrun can interleave rank stdout without inserting a newline.
        # Decode every complete JSON object on the line instead of treating
        # the line as one object; warnings and non-JSON fragments are skipped.
        offset = 0
        while offset < len(line):
            start = line.find("{", offset)
            if start < 0:
                break
            try:
                row, end = decoder.raw_decode(line, start)
            except json.JSONDecodeError:
                offset = start + 1
                continue
            offset = end
            if not isinstance(row, dict) or row.get("rank") != 0 or "step" not in row:
                continue
            try:
                steps.append(int(row["step"]))
            except (TypeError, ValueError):
                pass
    return steps


def _teacher_complete(root, log):
    expected = []
    for step in range(1000, 5001, 1000):
        expected += [root / f"resume_{step:05d}.pt", root / f"teacher_deploy_{step:05d}.pt"]
    if not all(path.is_file() for path in expected):
        return False
    steps = _logged_steps(log)
    return sorted(steps) == list(range(1, 5001)) and not _teacher_processes(root)


def _existing_arm_b_deploy(root):
    """Resolve the matched Arm-B deploy from its recorded pretrain run.json."""

    candidates = set()
    for path in (ROOT / "results/glt_sci_o8ctrl_20260917").rglob("run.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        command = row.get("command", [])
        if not isinstance(command, list) or row.get("schema") != "glt-sci-o8ctrl-arm-b-run-v1":
            continue
        for index, value in enumerate(command[:-1]):
            if value == "--output":
                output_arg = Path(str(command[index + 1]))
                output = (ROOT / output_arg).resolve() if not output_arg.is_absolute() else output_arg.resolve()
                deploy = output / "deploy_05000.pt"
                if deploy.is_file():
                    candidates.add(deploy)
    if len(candidates) != 1:
        raise RuntimeError("could not uniquely resolve existing Arm-B O8 deploy from run.json")
    return next(iter(candidates))


def _compare_values(left, right):
    if type(left) is not type(right):
        return False
    import numpy as np
    import torch
    if torch.is_tensor(left):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_compare_values(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_compare_values(a, b) for a, b in zip(left, right))
    return left == right


class Controller:
    def __init__(self, args):
        self.args = args
        self.result_root = Path(args.result_root).resolve()
        self.teacher_root = Path(args.teacher_root).resolve()
        self.teacher_log = Path(args.teacher_log).resolve()
        self.logs_root = ROOT / "logs/eq3d_dnd_20260917"
        self.state_path = self.result_root / "post_teacher_state.json"
        self.lock_path = self.result_root / ".post_teacher_controller.lock"
        self.state = None

    def _new_state(self):
        return {
            "schema": "eq3d-dnd-post-teacher-state-v1", "plan_id": PLAN_ID,
            "created_at_utc": _utc(), "updated_at_utc": _utc(),
            "teacher_runtime_source_commit": self.args.teacher_runtime_source_commit,
            "automation_implementation_commit": self.args.automation_implementation_commit,
            "status": "RUNNING", "final_status": None, "stages": {},
        }

    def load_state(self):
        if self.state_path.exists():
            if not self.args.resume:
                raise FileExistsError(f"controller state exists; use --resume: {self.state_path}")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("schema") != "eq3d-dnd-post-teacher-state-v1" or self.state.get("plan_id") != PLAN_ID:
                raise ValueError("controller state identity mismatch")
            for name, row in self.state.get("stages", {}).items():
                if row.get("status") == "PASS":
                    for path, digest in row.get("artifact_hashes", {}).items():
                        if _sha256(path) != digest:
                            raise RuntimeError(f"completed stage artifact changed: {name}: {path}")
        else:
            if self.args.resume:
                raise FileNotFoundError(f"--resume state is missing: {self.state_path}")
            self.state = self._new_state()
            self.flush()

    def flush(self):
        self.state["updated_at_utc"] = _utc()
        _atomic_json(self.state_path, self.state)

    def stage_start(self, name, command=None, inputs=None):
        old = self.state["stages"].get(name)
        if old and old.get("status") == "PASS":
            return False
        if old and old.get("status") not in (None, "WAITING"):
            raise RuntimeError(f"partial or contradictory stage requires inspection: {name}")
        self.state["stages"][name] = {
            "stage": name, "status": "RUNNING", "started_at": _utc(),
            "command": command, "input_artifacts": inputs or {},
            "git_commit": self.args.automation_implementation_commit,
        }
        self.flush()
        return True

    def stage_finish(self, name, *, status, exit_code=0, outputs=(), error=None, extra=None):
        row = self.state["stages"].setdefault(name, {"stage": name})
        hashes = {str(path): _sha256(path) for path in outputs}
        missing = [path for path, digest in hashes.items() if digest is None]
        if status == "PASS" and missing:
            status, exit_code = "FAIL", 1
            error = error or ("stage completed without required artifacts: " + ", ".join(missing))
        row.update({"status": status, "finished_at": _utc(), "exit_code": int(exit_code)})
        row["artifact_hashes"] = hashes
        if error:
            row["error"] = str(error)
        if extra:
            row.update(extra)
        self.flush()

    def run_command(self, name, command, *, env=None, outputs=(), inputs=None):
        if not self.stage_start(name, command=command, inputs=inputs):
            return True
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / f"{name.lower()}.log"
        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND=" + " ".join(map(str, command)) + "\n")
            handle.write("ENV=" + json.dumps({k: run_env[k] for k in (env or {})}, sort_keys=True) + "\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=run_env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            handle.write(f"EXIT_CODE={code}\n")
        self.state["stages"][name]["log"] = str(log_path)
        if code != 0:
            self.stage_finish(name, status="FAIL", exit_code=code, outputs=outputs,
                              error=f"command exited with {code}")
            return False
        self.stage_finish(name, status="PASS", exit_code=0, outputs=outputs)
        return self.state["stages"][name].get("status") == "PASS"

    def stop(self, final_status, name=None, error=None):
        self.state["status"] = "STOPPED"
        self.state["final_status"] = final_status
        if name:
            row = self.state["stages"].get(name)
            if not row or row.get("status") != "FAIL":
                self.stage_finish(name, status="FAIL", exit_code=1, error=error)
            elif error:
                row["error"] = str(error)
                self.flush()
        else:
            self.flush()

    def wait_teacher(self):
        name = "WAIT_TEACHER"
        command = ["wait", str(self.teacher_root)]
        inputs = {"teacher_root": str(self.teacher_root),
                  "teacher_log": str(self.teacher_log)}
        existing = self.state["stages"].get(name)
        if existing and existing.get("status") == "RUNNING":
            # A controller interrupted while polling may safely resume this
            # idempotent wait stage, but only for the same teacher inputs.
            if existing.get("command") != command or existing.get("input_artifacts") != inputs:
                raise RuntimeError("WAIT_TEACHER resume identity mismatch")
        elif not self.stage_start(name, command=command, inputs=inputs):
            return True
        while not _teacher_complete(self.teacher_root, self.teacher_log):
            self.state["stages"][name]["last_observation"] = {
                "checked_at": _utc(), "active_teacher_processes": _teacher_processes(self.teacher_root),
                "logged_steps": len(set(_logged_steps(self.teacher_log))),
            }
            self.flush()
            time.sleep(60)
        outputs = [self.teacher_root / f"resume_{step:05d}.pt" for step in range(1000, 5001, 1000)]
        outputs += [self.teacher_root / f"teacher_deploy_{step:05d}.pt" for step in range(1000, 5001, 1000)]
        self.stage_finish(name, status="PASS", outputs=outputs,
                          extra={"observed_steps": 5000, "process_exited": True})
        return True

    def teacher_health(self):
        output = self.result_root / "teacher_health.json"
        command = [sys.executable, "scripts/report_poly_painn_teacher_health.py",
                   "--run", str(self.teacher_root), "--log", str(self.teacher_log),
                   "--output", str(output), "--expected-steps", "5000"]
        if not self.run_command("TEACHER_HEALTH", command, outputs=[output],
                                inputs={"run": str(self.teacher_root), "log": str(self.teacher_log)}):
            self.stop("STOPPED_TEACHER_HEALTH_FAIL", "TEACHER_HEALTH")
            return False
        report = json.loads(output.read_text(encoding="utf-8"))
        required = (
            report.get("status") == "PASS" and report.get("last500_stable_window") is True
            and int(report.get("observed_steps", -1)) == 5000
            and report.get("missing_steps") == [] and int(report.get("duplicate_step_records", -1)) == 0
            and report.get("all_logged_values_finite") is True
            and report.get("checkpoint_values_finite") is True
            and report.get("no_collapse_to_zero") is True
            and all(item.get("status") == "PASS" for item in report.get("checkpoints", []))
        )
        self.state["stages"]["TEACHER_HEALTH"]["health_gate"] = bool(required)
        self.flush()
        if not required:
            self.stop("STOPPED_TEACHER_HEALTH_FAIL", "TEACHER_HEALTH", "explicit health gate failed")
            return False
        return True

    def teacher_equivariance(self):
        deploy = self.teacher_root / "teacher_deploy_05000.pt"
        cohort = ROOT / "data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1"
        cache = ROOT / "data/processed/mips_trimer_scage"
        middle = 959588 // 2
        outputs = [self.result_root / f"teacher_equivariance_{index}.json" for index in (0, middle, 959587)]
        for label, index, output in zip(("index0", "middle", "last"), (0, middle, 959587), outputs):
            stage_name = f"TEACHER_EQUIVARIANCE_{label}"
            command = [sys.executable, "scripts/validate_poly_painn_teacher.py",
                       "--deployment", str(deploy), "--cohort-root", str(cohort),
                       "--cache-root", str(cache), "--output", str(output), "--index", str(index)]
            if not self.run_command(stage_name, command, env={"CUDA_VISIBLE_DEVICES": "3"}, outputs=[output],
                                    inputs={"deployment": str(deploy), "index": index}):
                self.stop("STOPPED_TEACHER_EQUIVARIANCE_FAIL", stage_name, "validator command failed")
                return False
            report = json.loads(output.read_text(encoding="utf-8"))
            if report.get("status") != "PASS" or report.get("finite") is not True \
                    or report.get("deployment_excludes_noise_head") is not True:
                self.stop("STOPPED_TEACHER_EQUIVARIANCE_FAIL", stage_name, f"sample {index} validator failed")
                return False
        # Consolidate independent reports on the stage record.
        self.state["stages"]["TEACHER_EQUIVARIANCE"] = {
            "stage": "TEACHER_EQUIVARIANCE", "status": "PASS", "started_at": _utc(),
            "finished_at": _utc(), "exit_code": 0, "reports": [str(path) for path in outputs],
            "artifact_hashes": {str(path): _sha256(path) for path in outputs},
            "git_commit": self.args.automation_implementation_commit,
        }
        self.flush()
        return True

    def teacher_probe(self):
        teacher = self.teacher_root / "teacher_deploy_05000.pt"
        o8 = _existing_arm_b_deploy(ROOT)
        output = self.result_root / "probe"
        command = [sys.executable, "scripts/probe_poly_painn_teacher.py",
                   "--teacher-deployment", str(teacher), "--o8-deployment", str(o8),
                   "--raw-root", str(ROOT / "data/raw"),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage_downstream"),
                   "--downstream-static-root", str(ROOT / "data/processed/glt_dual_v2/downstream/dual_static_v1"),
                   "--split-root", str(ROOT / "data/splits/mips_outer5_inner20"),
                   "--output", str(output), "--batch-size", "16", "--device", "cuda"]
        if not self.run_command("TEACHER_PROBE_7TASK", command,
                                env={"CUDA_VISIBLE_DEVICES": "3"}, outputs=[output / "summary.json", output / "feature_sidecar.npz"],
                                inputs={"teacher": str(teacher), "o8": str(o8)}):
            self.stop("STOPPED_TEACHER_PROBE_FAIL", "TEACHER_PROBE_7TASK", "probe command failed")
            return False
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary.get("outer_test_accessed") is not False or int(summary.get("fold_count", -1)) != 35:
            self.stop("STOPPED_TEACHER_PROBE_FAIL", "TEACHER_PROBE_7TASK", "probe integrity gate failed")
            return False
        self.state["stages"]["TEACHER_PROBE_7TASK"]["reference_o8_deployment"] = str(o8)
        self.flush()
        return True

    def teacher_gate(self):
        summary_path = self.result_root / "probe/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gate = summary.get("teacher_residual_gate", {})
        status = gate.get("status")
        self.stage_start("TEACHER_GATE", command=["read", str(summary_path)], inputs={"summary": str(summary_path)})
        self.stage_finish("TEACHER_GATE", status="PASS" if status == "PASS" else "FAIL",
                          outputs=[summary_path], extra={"gate": gate})
        if status != "PASS":
            self.stop("STOPPED_TEACHER_RESIDUAL_GATE_FAIL")
            return False
        return True

    def student_tests(self):
        command = [sys.executable, "-m", "pytest", "-q",
                   "tests/test_o8_dnd_student.py", "tests/test_glt_sci_o8ctrl.py"]
        if not self.run_command("STUDENT_TESTS", command, outputs=(), inputs={}):
            self.stop("FAILED_STUDENT_TESTS", "STUDENT_TESTS", "focused student tests failed")
            return False
        return True

    def student_input_parity(self):
        output = self.result_root / "student_input_parity.json"
        command = [sys.executable, "scripts/validate_o8_dnd_student_parity.py",
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/dual_static_v1"),
                   "--pretrain-target-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/pretrain_targets_v1"),
                   "--output", str(output), "--count", "32", "--seed", "42"]
        if not self.run_command("STUDENT_INPUT_PARITY", command, outputs=[output], inputs={}):
            self.stop("FAILED_STUDENT_INPUT_PARITY", "STUDENT_INPUT_PARITY", "32-position O8 parity failed")
            return False
        if json.loads(output.read_text(encoding="utf-8")).get("status") != "PASS":
            self.stop("FAILED_STUDENT_INPUT_PARITY", "STUDENT_INPUT_PARITY", "parity report failed")
            return False
        return True

    def student_smoke(self):
        config = ROOT / "configs/mts/eq3d_dnd_o8_student_smoke.json"
        teacher = self.teacher_root / "teacher_deploy_05000.pt"
        common = [sys.executable, "scripts/pretrain_o8_dnd_student.py",
                  "--config", str(config), "--teacher-deployment", str(teacher),
                  "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1"),
                  "--cache-root", str(ROOT / "data/processed/mips_trimer_scage"),
                  "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/dual_static_v1"),
                  "--pretrain-target-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/pretrain_targets_v1"),
                  "--prep-workers", "0"]
        continuous = self.result_root / "student_smoke_continuous"
        candidate = self.result_root / "student_smoke_resume"
        if not self.run_command("STUDENT_SMOKE", common + ["--output", str(continuous), "--stop-after-step", "20"],
                                env={"CUDA_VISIBLE_DEVICES": "1,2,3"},
                                outputs=[continuous / "resume_00020.pt", continuous / "deploy_00020.pt"], inputs={}):
            self.stop("FAILED_STUDENT_SMOKE", "STUDENT_SMOKE", "20-update smoke failed")
            return False
        if not self.run_command("STUDENT_SMOKE_RESUME_PART1", common + ["--output", str(candidate), "--stop-after-step", "2"],
                                env={"CUDA_VISIBLE_DEVICES": "1,2,3"},
                                outputs=[candidate / "resume_00002.pt"], inputs={}):
            self.stop("FAILED_STUDENT_SMOKE", "STUDENT_SMOKE_RESUME_PART1", "resume part1 failed")
            return False
        if not self.run_command("STUDENT_SMOKE_RESUME_PART2", common + ["--output", str(candidate), "--resume", str(candidate / "resume_00002.pt"), "--stop-after-step", "4"],
                                env={"CUDA_VISIBLE_DEVICES": "1,2,3"},
                                outputs=[candidate / "resume_00004.pt", candidate / "deploy_00004.pt"], inputs={}):
            self.stop("FAILED_STUDENT_SMOKE", "STUDENT_SMOKE_RESUME_PART2", "resume part2 failed")
            return False
        continuous4 = self.result_root / "student_smoke_continuous4"
        if not self.run_command("STUDENT_SMOKE_CONTINUOUS4", common + ["--output", str(continuous4), "--stop-after-step", "4"],
                                env={"CUDA_VISIBLE_DEVICES": "1,2,3"},
                                outputs=[continuous4 / "resume_00004.pt"], inputs={}):
            self.stop("FAILED_STUDENT_SMOKE", "STUDENT_SMOKE_CONTINUOUS4", "continuous exact candidate failed")
            return False
        left = torch_load(continuous4 / "resume_00004.pt")
        right = torch_load(candidate / "resume_00004.pt")
        exact = _compare_values(left.get("model"), right.get("model")) \
            and _compare_values(left.get("optimizer"), right.get("optimizer")) \
            and _compare_values(left.get("rng"), right.get("rng")) \
            and left.get("identity") == right.get("identity") \
            and left.get("ordered_keys") == right.get("ordered_keys") \
            and left.get("next_position") == right.get("next_position") \
            and left.get("scheduler") == right.get("scheduler")
        self.state["stages"]["STUDENT_SMOKE"]["resume_exact"] = bool(exact)
        self.flush()
        if not exact:
            self.stop("FAILED_STUDENT_SMOKE_RESUME_MISMATCH", "STUDENT_SMOKE", "student resume state differs")
            return False
        return True

    def student_formal(self):
        config = ROOT / "configs/mts/eq3d_dnd_o8_student.json"
        teacher = self.teacher_root / "teacher_deploy_05000.pt"
        command = [sys.executable, "scripts/pretrain_o8_dnd_student.py",
                   "--config", str(config), "--teacher-deployment", str(teacher),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/dual_static_v1"),
                   "--pretrain-target-root", str(ROOT / "data/processed/glt_dual_v2/pi1m/pretrain_targets_v1"),
                   "--output", str(self.result_root / "student_formal"), "--prep-workers", "2"]
        output = self.result_root / "student_formal"
        if not self.run_command("STUDENT_FORMAL_5K", command, env={"CUDA_VISIBLE_DEVICES": "1,2,3"},
                                outputs=[output / f"resume_{step:05d}.pt" for step in range(1000, 5001, 1000)] +
                                [output / f"deploy_{step:05d}.pt" for step in range(1000, 5001, 1000)], inputs={}):
            self.stop("FAILED_STUDENT_FORMAL_5K", "STUDENT_FORMAL_5K", "student formal pretraining failed")
            return False
        return True

    def student_deploy_validate(self):
        output = self.result_root / "student_deploy_validation.json"
        command = [sys.executable, "scripts/validate_o8_dnd_student.py",
                   "--deployment", str(self.result_root / "student_formal/deploy_05000.pt"),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage_downstream"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/downstream/dual_static_v1"),
                   "--output", str(output), "--index", "0"]
        if not self.run_command("STUDENT_DEPLOY_VALIDATE", command, env={"CUDA_VISIBLE_DEVICES": "3"}, outputs=[output], inputs={}):
            self.stop("FAILED_STUDENT_DEPLOY_VALIDATE", "STUDENT_DEPLOY_VALIDATE", "student deploy validation failed")
            return False
        if json.loads(output.read_text(encoding="utf-8")).get("status") != "PASS":
            self.stop("FAILED_STUDENT_DEPLOY_VALIDATE", "STUDENT_DEPLOY_VALIDATE", "student deploy report failed")
            return False
        return True

    def arm_b_egc(self):
        output = self.result_root / "arm_b_egc_grid"
        command = [sys.executable, "scripts/run_glt_sci_o8_control_grid.py",
                   "--config", str(ROOT / "configs/mts/glt_sci_o8ctrl_arm_b.json"),
                   "--checkpoint", str(_existing_arm_b_deploy(ROOT)), "--raw-root", str(ROOT / "data/raw"),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage_downstream"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/downstream/dual_static_v1"),
                   "--split-root", str(ROOT / "data/splits/mips_outer5_inner20"),
                   "--output", str(output), "--log-root", str(ROOT / "logs/eq3d_dnd_20260917/arm_b_egc"),
                   "--arm", "B", "--task", "egc", "--gpu", "0", "--gpu", "1", "--gpu", "2", "--gpu", "3",
                   "--clean-cache-gib", "4"]
        if not self.run_command("ARM_B_EGC_5FOLD", command, outputs=[output], inputs={}):
            self.stop("FAILED_ARM_B_EGC_5FOLD", "ARM_B_EGC_5FOLD", "Arm B egc grid failed")
            return False
        return True

    def student_finetune_smoke(self):
        output = self.result_root / "student_finetune_smoke"
        command = [sys.executable, "scripts/finetune_glt_o8_control.py",
                   "--config", str(ROOT / "configs/mts/eq3d_dnd_o8_student.json"),
                   "--checkpoint", str(self.result_root / "student_formal/deploy_05000.pt"),
                   "--raw-root", str(ROOT / "data/raw"),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage_downstream"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/downstream/dual_static_v1"),
                   "--split-root", str(ROOT / "data/splits/mips_outer5_inner20"),
                   "--output", str(output), "--arm", "DND", "--task", "eat", "--fold", "0",
                   "--smoke", "--clean-cache-gib", "4"]
        if not self.run_command("STUDENT_FINETUNE_SMOKE", command, env={"CUDA_VISIBLE_DEVICES": "0"},
                                outputs=[output / "summary.json"], inputs={}):
            self.stop("FAILED_STUDENT_FINETUNE_SMOKE", "STUDENT_FINETUNE_SMOKE", "student finetune smoke failed")
            return False
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if report.get("outer_test") != "NOT_RUN":
            self.stop("FAILED_STUDENT_FINETUNE_SMOKE", "STUDENT_FINETUNE_SMOKE", "smoke accessed outer-test")
            return False
        return True

    def student_grid(self):
        output = self.result_root / "student_finetune_grid"
        command = [sys.executable, "scripts/run_glt_sci_o8_control_grid.py",
                   "--config", str(ROOT / "configs/mts/eq3d_dnd_o8_student.json"),
                   "--checkpoint", str(self.result_root / "student_formal/deploy_05000.pt"),
                   "--raw-root", str(ROOT / "data/raw"),
                   "--cohort-root", str(ROOT / "data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1"),
                   "--cache-root", str(ROOT / "data/processed/mips_trimer_scage_downstream"),
                   "--dual-static-root", str(ROOT / "data/processed/glt_dual_v2/downstream/dual_static_v1"),
                   "--split-root", str(ROOT / "data/splits/mips_outer5_inner20"),
                   "--output", str(output), "--log-root", str(ROOT / "logs/eq3d_dnd_20260917/student_grid"),
                   "--arm", "DND", "--gpu", "0", "--gpu", "1", "--gpu", "2", "--gpu", "3",
                   "--clean-cache-gib", "4"]
        for task in ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"):
            command += ["--task", task]
        if not self.run_command("STUDENT_GRID_8X5", command, env={}, outputs=[output], inputs={}):
            self.stop("FAILED_STUDENT_GRID_8X5", "STUDENT_GRID_8X5", "student 8x5 grid failed")
            return False
        return True

    def aggregate(self):
        output = self.result_root / "aggregation_8task"
        command = [sys.executable, "scripts/aggregate_o8_dnd_8task.py",
                   "--arm-b-7-root", str(ROOT / "results/glt_sci_o8ctrl_20260917/formal/arm_b_grid"),
                   "--arm-b-egc-root", str(self.result_root / "arm_b_egc_grid"),
                   "--student-root", str(self.result_root / "student_finetune_grid"),
                   "--split-root", str(ROOT / "data/splits/mips_outer5_inner20"),
                   "--raw-root", str(ROOT / "data/raw"), "--output", str(output)]
        if not self.run_command("AGGREGATE_8TASK", command,
                                outputs=[output / "summary.json", output / "all_fold_metrics.csv", output / "paired_fold_deltas.csv"], inputs={}):
            self.stop("FAILED_AGGREGATE_8TASK", "AGGREGATE_8TASK", "8-task aggregation failed")
            return False
        return True

    def run(self):
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.load_state()
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not self.wait_teacher():
                return 1
            if not self.teacher_health():
                return 0
            if not self.teacher_equivariance():
                return 0
            if not self.teacher_probe():
                return 1
            if not self.teacher_gate():
                return 0
            for method in (self.student_tests, self.student_input_parity, self.student_smoke, self.student_formal,
                           self.student_deploy_validate, self.arm_b_egc,
                           self.student_finetune_smoke, self.student_grid, self.aggregate):
                if not method():
                    return 1
            self.state["status"] = "COMPLETE_WAITING_FOR_CODEX_REVIEW"
            self.state["final_status"] = "WAITING_FOR_CODEX_REVIEW"
            self.flush()
        return 0


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--teacher-root", default=str(DEFAULT_TEACHER_ROOT))
    parser.add_argument("--teacher-log", default=str(DEFAULT_TEACHER_LOG))
    parser.add_argument("--teacher-runtime-source-commit", required=True)
    parser.add_argument("--automation-implementation-commit", default="")
    args = parser.parse_args()
    if not args.automation_implementation_commit:
        args.automation_implementation_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    controller = Controller(args)
    try:
        return controller.run()
    except BlockingIOError:
        print("another POST-TEACHER controller is already running", file=sys.stderr)
        return 2
    except Exception as exc:
        if controller.state is not None:
            controller.state["status"] = "STOPPED"
            controller.state["final_status"] = "FAILED_CONTROLLER"
            controller.state["controller_error"] = f"{type(exc).__name__}: {exc}"
            try:
                controller.flush()
            except Exception:
                pass
        print(f"POST-TEACHER controller failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
