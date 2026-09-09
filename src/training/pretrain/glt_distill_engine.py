"""Two-stage N+1/N+2 teacher and O8+MD200 student pretraining."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import hashlib
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

from src.modules.mts_glt_distill import (
    AtomPredictor, DistillStudent, NPlusGLTTeacher, StudentLineProjection, fixed_rbf,
)
from src.training.pretrain.engine import _joint_canonical_mask


PAIR_CLASSES = 101 * 102 // 2


def atom_pair_label(a, b):
    low, high = torch.minimum(a.long(), b.long()), torch.maximum(a.long(), b.long())
    return high * (high + 1) // 2 + low


def exact_center_mask(data, ratio, generator):
    eligible = data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()
    selected = torch.zeros_like(eligible)
    for graph in range(int(data.glt3_geometry_valid.numel())):
        indices = torch.nonzero(eligible & (data.glt3_token_batch == graph), as_tuple=False).flatten()
        if indices.numel():
            count = min(indices.numel(), max(1, int(round(float(ratio) * indices.numel()))))
            order = torch.randperm(indices.numel(), generator=generator, device=indices.device)
            selected[indices[order[:count]]] = True
    return selected


def graph_reduced_sum(values, graph_ids, graph_count):
    """Return sum of per-graph target means and number of nonempty graphs."""
    sums = scatter(values, graph_ids, dim=0, dim_size=graph_count, reduce="sum")
    counts = scatter(torch.ones_like(values), graph_ids, dim=0, dim_size=graph_count, reduce="sum")
    valid = counts > 0
    return (sums[valid] / counts[valid]).sum(), int(valid.sum())


def angle_targets(data):
    relation = data.glt3_relation_valid.bool()
    targets = data.glt3_relation_target.long()[relation]
    encoded = fixed_rbf(data.glt3_relation_angle[relation], 32, 0.0, math.pi)
    count = data.glt3_token_atom_a.numel()
    total = scatter(encoded, targets, dim=0, dim_size=count, reduce="sum")
    denominator = scatter(
        torch.ones((targets.numel(), 1), device=targets.device), targets,
        dim=0, dim_size=count, reduce="sum",
    )
    return total / denominator.clamp_min(1.0), denominator.squeeze(-1) > 0


class TeacherContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = NPlusGLTTeacher()
        self.pair_head = nn.Linear(256, PAIR_CLASSES)
        self.bond_head = nn.Linear(256, 5)
        self.stereo_head = nn.Linear(256, 6)
        self.conjugated_head = nn.Linear(256, 2)
        self.length_head = nn.Linear(256, 64)
        self.angle_head = nn.Linear(256, 32)

    def forward(self, data, generator=None, selected=None):
        selected = exact_center_mask(data, 0.40, generator) if selected is None else selected
        angle_mask = selected[data.glt3_relation_source.long()] | selected[data.glt3_relation_target.long()]
        output = self.teacher(data, token_mask=selected, angle_mask=angle_mask)
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        graph_ids = data.glt3_token_batch.long()[indices]
        projected = output["projected_lines"][indices]
        pair = atom_pair_label(data.glt3_token_endpoint_z_a[indices], data.glt3_token_endpoint_z_b[indices])
        chem = 0.25 * (
            F.cross_entropy(self.pair_head(projected).float(), pair, reduction="none")
            + F.cross_entropy(self.bond_head(projected).float(), data.glt3_token_bond_type[indices].long(), reduction="none")
            + F.cross_entropy(self.stereo_head(projected).float(), data.glt3_token_stereo[indices].long(), reduction="none")
            + F.cross_entropy(self.conjugated_head(projected).float(), data.glt3_token_conjugated[indices].long(), reduction="none")
        )
        length = (self.length_head(projected).float() - fixed_rbf(data.glt3_token_distance[indices], 64, 0.0, 3.75)).square().mean(-1)
        clean_angles, angle_valid = angle_targets(data)
        has_angle = angle_valid[indices]
        angle = (self.angle_head(projected[has_angle]).float() - clean_angles[indices[has_angle]]).square().mean(-1)
        graph_count = int(data.glt3_geometry_valid.numel())
        chem_sum, graph_target_count = graph_reduced_sum(chem, graph_ids, graph_count)
        length_sum, _ = graph_reduced_sum(length, graph_ids, graph_count)
        angle_sum, angle_graph_count = graph_reduced_sum(angle, graph_ids[has_angle], graph_count)
        return {
            "chem_sum": chem_sum, "length_sum": length_sum, "angle_sum": angle_sum,
            "graph_count": graph_target_count, "angle_graph_count": angle_graph_count,
            "masked_lines": int(indices.numel()), "angle_relations_masked": int(angle_mask.sum()),
        }


def _gather_sizes(value):
    local = torch.tensor([value.size(0)], dtype=torch.long, device=value.device)
    outputs = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, local)
    return [int(item.item()) for item in outputs]


def _gather_with_grad(value):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    from torch.distributed.nn.functional import all_gather
    sizes = _gather_sizes(value)
    maximum = max(1, max(sizes))
    padded = F.pad(value, (0, 0, 0, maximum - value.size(0)))
    gathered = tuple(all_gather(padded))
    return torch.cat([item[:size] for item, size in zip(gathered, sizes)], dim=0)


@torch.no_grad()
def _gather_hash(value):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    sizes = _gather_sizes(value)
    maximum = max(1, max(sizes))
    padded = F.pad(value, (0, maximum - value.size(0)))
    outputs = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, padded)
    return torch.cat([item[:size] for item, size in zip(outputs, sizes)])


def multi_positive_infonce(student, teacher, identities, temperature=0.1):
    student, teacher = _gather_with_grad(student), _gather_with_grad(teacher)
    identities = _gather_hash(identities)
    if student.size(0) < 2:
        return student.sum() * 0.0, int(student.size(0))
    s, t = F.normalize(student.float(), dim=-1), F.normalize(teacher.float(), dim=-1)
    logits = s @ t.T / temperature
    positive = identities[:, None] == identities[None, :]
    loss = -(torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1) - torch.logsumexp(logits, dim=1)).mean()
    return loss, int(student.size(0))


class StudentContainer(nn.Module):
    def __init__(self, teacher=None):
        super().__init__()
        self.student = DistillStudent()
        for parameter in self.student.o8.md_residual.parameters():
            parameter.requires_grad = False
        self.atom_head = AtomPredictor()
        self.line_projection = StudentLineProjection()
        self.teacher = teacher
        if self.teacher is None:
            for parameter in self.line_projection.parameters():
                parameter.requires_grad = False
        if self.teacher is not None:
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, data, stream_step, generator=None, atom_mask=None):
        atom_mask = _joint_canonical_mask(data, 42, stream_step, 0.30) if atom_mask is None else atom_mask
        md = data.mips_md.clone()
        disturb = torch.rand(md.shape, device=md.device, generator=generator) < 0.30
        replacements = torch.rand(md.shape, device=md.device, generator=generator)
        md[disturb] = replacements[disturb]
        raw_masked, fused_masked = self.student.encode(data, atom_mask=atom_mask, md200=md)
        indices = torch.nonzero(atom_mask, as_tuple=False).flatten()
        targets = data.mips_x[indices, :101].argmax(-1).long()
        atom = 0.5 * (
            F.cross_entropy(self.atom_head(raw_masked[indices]).float(), targets, reduction="sum")
            + F.cross_entropy(self.atom_head(fused_masked[indices]).float(), targets, reduction="sum")
        )
        if self.teacher is None:
            zero = atom * 0.0
            return {
                "atom_sum": atom, "atom_count": int(indices.numel()),
                "local_sum": zero, "local_count": 0,
                "global": zero, "valid_graphs": 0, "global_pool": 0,
                "md_disturbed": int(disturb.sum()), "md_total": int(disturb.numel()),
            }
        # Distillation has a private RNG stream so the extra clean O8 pass does
        # not perturb the atom-task dropout sequence shared with C0.
        devices = [data.mips_x.device.index] if data.mips_x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1_000_000_007 + int(stream_step))
            if data.mips_x.is_cuda:
                torch.cuda.manual_seed(1_000_000_007 + int(stream_step))
            _, _, clean = self.student.o8.canonical(data)
            student_lines, line_batch = self.line_projection(data, clean)
            with torch.no_grad():
                teacher_output = self.teacher(data)
                teacher_lines = teacher_output["center_projected"]
        if student_lines.shape != teacher_lines.shape:
            raise RuntimeError("student/teacher center-line mapping mismatch")
        cosine = 1.0 - (F.normalize(student_lines.float(), dim=-1) * F.normalize(teacher_lines.float(), dim=-1)).sum(-1)
        graph_count = int(data.glt3_geometry_valid.numel())
        local_sum, valid_graphs = graph_reduced_sum(cosine, line_batch, graph_count)
        student_graph = scatter(student_lines, line_batch, dim=0, dim_size=graph_count, reduce="mean")
        teacher_graph = scatter(teacher_lines, line_batch, dim=0, dim_size=graph_count, reduce="mean")
        valid = data.glt3_geometry_valid.bool() & (scatter(torch.ones_like(line_batch), line_batch, dim=0, dim_size=graph_count, reduce="sum") > 0)
        # Every rank enters the same gather collectives, including empty ranks.
        global_loss, global_pool = multi_positive_infonce(
            student_graph[valid], teacher_graph[valid], data.mts_sample_hash64.long()[valid]
        )
        return {
            "atom_sum": atom, "atom_count": int(indices.numel()),
            "local_sum": local_sum, "local_count": valid_graphs,
            "global": global_loss, "valid_graphs": int(valid.sum()),
            "global_pool": global_pool,
            "md_disturbed": int(disturb.sum()), "md_total": int(disturb.numel()),
        }


def _distributed_mean(local_sum, local_count, device):
    stats = torch.tensor([float(local_count)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
        world = dist.get_world_size()
    else:
        world = 1
    count = int(stats.item())
    return local_sum * (world / max(1, count)), count


def _global_count(local_count, device):
    value = torch.tensor(int(local_count), dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
    return int(value.item())


def _dataset(config):
    from types import SimpleNamespace
    from src.dataset import UniDataset
    from src.dataset.md200_sidecar import DatasetWithMD200
    from src.training.pretrain.config import _apply_glt_v3_config, dataset_kwargs_from_args
    args = SimpleNamespace(resume_state=None)
    template = {
        "schema": "mts-glt-v3-galformer-20k", "experiment_id": config["experiment_id"],
        "dataset_name": "PI1M_v2", "cache_root": "data",
        "line_sidecar_root": config.get("line_sidecar_root") or
            "data/processed/mips_trimer_scage/periodic_line_glt_distill_v2/n_plus_1",
        "md200_sidecar_root": config["md200_sidecar_root"], "result_root": config["result_root"],
        "output_path": config["output_path"], "atom_mask_ratio": 0.3, "line_mask_ratio": 0.4,
        "infonce_temperature": 0.1, "batch_size": config["microbatch"], "loader_workers": config["loader_workers"],
        "prefetch_factor": 2, "global_batch_size": 3 * config["microbatch"], "max_optimizer_steps": 20000,
        "stop_after_steps": 20000, "probe_steps": [5000,10000,20000], "amp_dtype": "bf16",
        "seed": 42, "lr": 2e-4, "warmup_steps": 2000, "end_lr": 1e-9, "weight_decay": 0,
        "cache_layers": "ru_base,topology,trimer", "glt_readout_mode": "galformer",
    }
    args = _apply_glt_v3_config(args, Path(config["config_path"]), template)
    kwargs = dataset_kwargs_from_args(args)
    if config["version"] == "none":
        kwargs["periodic_line_glt_sidecar"] = None
    return DatasetWithMD200(UniDataset(**kwargs), config["md200_sidecar_root"])


def _checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(bytes(tensor.numpy()))
    return digest.hexdigest()


def _deploy_student(container, step, version, geometry_revision, *, repair_experiment=False):
    return {
        "schema": (
            "mts-glt-distill-repair-student-deploy-v1" if repair_experiment
            else "mts-glt-distill-student-deploy-v1"
        ), "version": version, "step": int(step),
        "geometry_revision": geometry_revision,
        "state_dict": {k: v.detach().cpu() for k, v in container.student.state_dict().items()},
        "architecture": "PreLN-sourceQ-O8+atomic-conditioned-MD200",
    }


def run_stage(
    config_path, stage, *, stop_after=None, result_root=None,
    teacher_checkpoint=None, resume=None, line_sidecar_root=None,
):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config["config_path"] = str(config_path)
    repair_experiment = config.get("schema") == "mts-glt-distill-repair-control-v1"
    if result_root is not None:
        config["result_root"] = str(Path(result_root).resolve())
    if line_sidecar_root is not None:
        config["line_sidecar_root"] = str(Path(line_sidecar_root).resolve())
    if stage not in {"teacher", "student"}:
        raise ValueError(stage)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=torch.device("cuda", local_rank))
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world = 0, 0, 1
    try:
        if world != 3 or not torch.cuda.is_available():
            raise RuntimeError("formal N+ distillation requires three CUDA ranks")
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        device = torch.device("cuda", local_rank)
        dataset = _dataset(config)
        if config.get("geometry_revision") == 2:
            sidecar = dataset.dataset._periodic_line_glt_sidecar
            metadata = sidecar.metadata
            if (
                int(metadata.get("geometry_revision", -1)) != 2
                or metadata.get("schema") != f"mts-periodic-line-distill-v2-{config['version']}"
            ):
                raise RuntimeError("line sidecar version/geometry revision mismatch")
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=True)
        from src.utils import get_data_loader
        loader = get_data_loader(
            dataset, indices=None, batch_size=int(config["microbatch"]), sampler=sampler,
            shuffle=False, drop_last=True, num_workers=int(config["loader_workers"]), pin_memory=True,
            persistent_workers=int(config["loader_workers"]) > 0, prefetch_factor=2,
            generator=torch.Generator().manual_seed(42 + rank),
        )
        version_root = Path(config["result_root"])
        root = version_root / stage
        resume_path = Path(resume).resolve() if resume else None
        root_error = None
        if rank == 0:
            try:
                if resume_path is None:
                    if root.exists():
                        raise FileExistsError(root)
                    root.mkdir(parents=True)
                    (root / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
                elif not resume_path.is_file() or resume_path.parent != root.resolve():
                    raise RuntimeError("resume checkpoint must exist inside the selected stage directory")
            except Exception as error:
                root_error = f"{type(error).__name__}: {error}"
        coordinated = [root_error]
        dist.broadcast_object_list(coordinated, src=0)
        if coordinated[0] is not None:
            raise RuntimeError(f"coordinated stage preflight failure: {coordinated[0]}")
        geometry_revision = config.get("geometry_revision")
        if stage == "teacher":
            if config["version"] == "none":
                raise RuntimeError("version=none has no teacher stage")
            container = TeacherContainer().to(device)
            total_steps, warmup = 5000, 500
        else:
            teacher = None
            if config["version"] != "none":
                teacher_path = Path(teacher_checkpoint) if teacher_checkpoint else version_root / "teacher/teacher_005k.pt"
                teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
                expected_teacher_schema = (
                    "mts-glt-distill-repair-teacher-state-v1" if geometry_revision == 2
                    else "mts-glt-distill-teacher-state-v1"
                )
                if (
                    teacher_payload.get("schema") != expected_teacher_schema
                    or teacher_payload.get("version") != config["version"]
                    or int(teacher_payload.get("step", -1)) != (
                        int(stop_after) if stop_after is not None and int(stop_after) < 5000 else 5000
                    )
                    or (geometry_revision == 2 and int(teacher_payload.get("geometry_revision", -1)) != 2)
                ):
                    raise RuntimeError("teacher checkpoint version/step/geometry revision mismatch")
                # Teacher construction must not advance the public student
                # initialization stream shared with the no-distillation C0.
                with torch.random.fork_rng(devices=[]):
                    teacher = NPlusGLTTeacher()
                teacher.load_state_dict(teacher_payload["teacher_state"], strict=True)
            container = StudentContainer(teacher).to(device)
            total_steps, warmup = 20000, 2000
        run_until = total_steps
        if stop_after is not None:
            run_until = min(total_steps, int(stop_after))
            if run_until < 1:
                raise ValueError("stop_after must be positive")
        module = torch.nn.parallel.DistributedDataParallel(container, device_ids=[local_rank], find_unused_parameters=False)
        if stage == "student" and resume_path is None and rank == 0:
            common_state = {
                key: value.detach().cpu()
                for key, value in container.state_dict().items()
                if key.startswith(("student.", "atom_head."))
            }
            _checkpoint(root / "student_step0_common.pt", {
                "schema": "mts-glt-distill-repair-common-init-v1",
                "seed": 42, "version": config["version"],
                "state_dict": common_state,
                "sha256": _tensor_state_sha256(common_state),
            })
        parameters = [p for p in container.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, betas=(0.9,0.98), weight_decay=0.0)
        floor = 1e-9 / 2e-4
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda k: (k + 1) / warmup if k < warmup else floor + (1-floor) * (1-min(1.0,(k-warmup)/max(1,total_steps-warmup)))
        )
        metrics_path = root / "training_metrics.jsonl"
        accumulation = int(config["accumulation"])
        start_step = 0
        resume_payload = None
        if resume_path is not None:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            schema_stem = "mts-glt-distill-repair" if repair_experiment else "mts-glt-distill"
            expected_schema = f"{schema_stem}-{stage}-state-v1"
            if (
                resume_payload.get("schema") != expected_schema
                or resume_payload.get("version") != config["version"]
                or resume_payload.get("geometry_revision") != geometry_revision
            ):
                raise RuntimeError("resume checkpoint identity mismatch")
            start_step = int(resume_payload["step"])
            if not 0 < start_step < run_until:
                raise RuntimeError("resume step must precede the requested stop step")
            container.load_state_dict(resume_payload["container_state"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            if len(resume_payload.get("rng_states", ())) != world:
                raise RuntimeError("resume checkpoint does not contain per-rank RNG states")
            resume_log_error = None
            if rank == 0:
                try:
                    lines = metrics_path.read_text().splitlines()
                    existing = [json.loads(line) for line in lines]
                    if len(existing) < start_step or int(existing[start_step - 1]["step"]) != start_step:
                        raise RuntimeError("resume metrics do not contain the checkpoint step")
                    if len(existing) > start_step:
                        abandoned = root / f"training_metrics.abandoned_after_{start_step:06d}.jsonl"
                        abandoned.write_text("\n".join(lines[start_step:]) + "\n")
                        metrics_path.write_text("\n".join(lines[:start_step]) + "\n")
                except Exception as error:
                    resume_log_error = f"{type(error).__name__}: {error}"
            coordinated = [resume_log_error]
            dist.broadcast_object_list(coordinated, src=0)
            if coordinated[0] is not None:
                raise RuntimeError(f"coordinated resume log failure: {coordinated[0]}")
        batches_per_epoch = len(loader)
        total_batches = start_step * accumulation
        epoch, batch_offset = divmod(total_batches, batches_per_epoch)
        if resume_payload is not None and resume_payload.get("data_position") != {
            "epoch": epoch, "batch_offset": batch_offset,
        }:
            raise RuntimeError("resume data position does not match the completed optimizer step")
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(batch_offset):
            next(iterator)
        if resume_payload is not None:
            rank_rng = resume_payload["rng_states"][rank]
            torch.set_rng_state(rank_rng["cpu"])
            torch.cuda.set_rng_state(rank_rng["cuda"], device=device)
        for step in range(start_step, run_until):
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            prepared = []
            for micro in range(accumulation):
                try: data = next(iterator)
                except StopIteration:
                    epoch += 1; sampler.set_epoch(epoch); iterator = iter(loader); data = next(iterator)
                data = data.to(device, non_blocking=True)
                seed = 42 + (step * accumulation + micro) * world + rank
                generator = torch.Generator(device=device).manual_seed(seed)
                if stage == "teacher":
                    selected = exact_center_mask(data, 0.40, generator)
                    graph_ids = data.glt3_token_batch.long()[selected]
                    graph_count = int(torch.unique(graph_ids).numel())
                    _, has_angle = angle_targets(data)
                    angle_graph_count = int(torch.unique(graph_ids[has_angle[selected]]).numel())
                    prepared.append((data, seed, selected, graph_count, angle_graph_count))
                else:
                    atom_mask = _joint_canonical_mask(
                        data, 42, step * accumulation + micro, 0.30
                    )
                    if config["version"] == "none":
                        valid_graphs = 0
                    else:
                        center = data.glt3_token_center_internal.bool() & data.glt3_token_valid.bool()
                        line_graph = data.glt3_token_batch.long()[center]
                        valid_graphs = int(torch.unique(line_graph).numel())
                    prepared.append((data, seed, atom_mask, int(atom_mask.sum()), valid_graphs))
            if stage == "teacher":
                total_main = _global_count(sum(item[3] for item in prepared), device)
                total_angle = _global_count(sum(item[4] for item in prepared), device)
            else:
                total_atom = _global_count(sum(item[3] for item in prepared), device)
                total_local = _global_count(sum(item[4] for item in prepared), device)
            records = []
            for micro, item in enumerate(prepared):
                data, seed, fixed_mask = item[:3]
                generator = torch.Generator(device=device).manual_seed(seed)
                sync = module.no_sync() if micro + 1 < accumulation else nullcontext()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(data, generator=generator, selected=fixed_mask) if stage == "teacher" else module(data, step, generator=generator, atom_mask=fixed_mask)
                    if stage == "teacher":
                        loss = world * (
                            (output["chem_sum"] + output["length_sum"]) / max(1, total_main)
                            + output["angle_sum"] / max(1, total_angle)
                        )
                    else:
                        ramp = min(1.0, step / 2000.0)
                        global_total_pool = _global_count(sum(value[4] for value in prepared), device)
                        global_weight = output["global_pool"] / max(1, global_total_pool)
                        loss = world * (
                            output["atom_sum"] / max(1, total_atom)
                            + ramp * 0.1 * output["local_sum"] / max(1, total_local)
                        ) + ramp * 0.1 * global_weight * output["global"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {stage} loss at step {step + 1}")
                loss.backward(); records.append({k: float(v.detach()) if torch.is_tensor(v) and v.numel()==1 else v for k,v in output.items()})
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); scheduler.step()
            completed = step + 1
            if stage == "teacher":
                diagnostics = torch.tensor([
                    sum(item["chem_sum"] for item in records),
                    sum(item["length_sum"] for item in records),
                    sum(item["angle_sum"] for item in records),
                    sum(item["graph_count"] for item in records),
                    sum(item["angle_graph_count"] for item in records),
                    sum(item["masked_lines"] for item in records),
                ], dtype=torch.float64, device=device)
            else:
                diagnostics = torch.tensor([
                    sum(item["atom_sum"] for item in records),
                    sum(item["local_sum"] for item in records),
                    sum(item["atom_count"] for item in records),
                    sum(item["local_count"] for item in records),
                    sum(item["valid_graphs"] for item in records),
                    sum(item["md_disturbed"] for item in records),
                    sum(item["md_total"] for item in records),
                ], dtype=torch.float64, device=device)
            dist.all_reduce(diagnostics)
            if rank == 0:
                record = {
                    "step": completed, "stage": stage, "lr": optimizer.param_groups[0]["lr"],
                    "step_seconds": time.perf_counter()-started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                if stage == "teacher":
                    record.update({
                        "chem_loss": float(diagnostics[0] / diagnostics[3].clamp_min(1)),
                        "length_loss": float(diagnostics[1] / diagnostics[3].clamp_min(1)),
                        "angle_loss": float(diagnostics[2] / diagnostics[4].clamp_min(1)),
                        "valid_graph_targets": int(diagnostics[3]),
                        "masked_lines": int(diagnostics[5]),
                    })
                    record["loss"] = record["chem_loss"] + record["length_loss"] + record["angle_loss"]
                else:
                    record.update({
                        "atom_loss": float(diagnostics[0] / diagnostics[2].clamp_min(1)),
                        "local_distill_loss": float(diagnostics[1] / diagnostics[3].clamp_min(1)),
                        "valid_distill_graphs": int(diagnostics[4]),
                        "md_disturbance_fraction": float(diagnostics[5] / diagnostics[6].clamp_min(1)),
                        "global_distill_loss": sum(item["global"] for item in records) / accumulation,
                    })
                    ramp = min(1.0, step / 2000.0)
                    record["loss"] = record["atom_loss"] + ramp * 0.1 * (record["local_distill_loss"] + record["global_distill_loss"])
                with metrics_path.open("a") as handle: handle.write(json.dumps(record, sort_keys=True)+"\n")
                if completed % 20 == 0 or completed == 1:
                    print(f"{config['version']} {stage} step={completed}/{total_steps} loss={record['loss']:.5f} lr={record['lr']:.3e}", flush=True)
            should_checkpoint = (
                completed % 1000 == 0
                or completed == run_until
                or (repair_experiment and completed % 250 == 0)
            )
            if should_checkpoint:
                local_rng = {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device),
                }
                rng_states = [None for _ in range(world)]
                dist.all_gather_object(rng_states, local_rng)
                if rank == 0:
                    next_batches = completed * accumulation
                    next_epoch, next_offset = divmod(next_batches, batches_per_epoch)
                    schema_stem = "mts-glt-distill-repair" if repair_experiment else "mts-glt-distill"
                    payload = {
                        "schema": f"{schema_stem}-{stage}-state-v1", "version": config["version"], "step": completed,
                        "geometry_revision": geometry_revision,
                        "teacher_state": container.teacher.state_dict() if stage == "teacher" else None,
                        "container_state": container.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "rng_states": rng_states,
                        "data_position": {"epoch": next_epoch, "batch_offset": next_offset},
                    }
                    if completed % 1000 == 0:
                        checkpoint_name = f"{stage}_{completed//1000:03d}k.pt"
                    elif completed == run_until and run_until < 1000:
                        checkpoint_name = f"{stage}_step_{completed:06d}.pt"
                    else:
                        checkpoint_name = f"{stage}_resume_latest.pt"
                    _checkpoint(root / checkpoint_name, payload)
                    if stage == "student" and completed in {5000,10000,20000}:
                        _checkpoint(root / f"student_deploy_{completed//1000:03d}k.pt", _deploy_student(
                            container, completed, config["version"], geometry_revision,
                            repair_experiment=repair_experiment,
                        ))
                    elif stage == "student" and completed == run_until and run_until < 5000:
                        _checkpoint(root / f"student_deploy_step_{completed:06d}.pt", _deploy_student(
                            container, completed, config["version"], geometry_revision,
                            repair_experiment=repair_experiment,
                        ))
                dist.barrier()
        dist.barrier()
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


__all__ = ["StudentContainer", "TeacherContainer", "run_stage"]
