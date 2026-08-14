#!/usr/bin/env python3
"""Read-only MTS runtime doctor.

The doctor intentionally performs only local schema/readability checks and a
strict model smoke.  Integrity and identity digests belong to the explicit
offline audit tools; they are not recomputed during training startup or by
this command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.canonical_periodic import build_canonical_periodic_topology  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore  # noqa: E402
from src.dataset.trimer_mcl import attach_finite_trimer_mcl  # noqa: E402
from src.modules.mips_local_graph import MIPSLocalGraphEncoder  # noqa: E402
from src.modules.uni_encoder import UniEncoderAttention  # noqa: E402
from src.dataset.mips_trimer_contract import (  # noqa: E402
    CACHE_BOND_ANGLE_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA,
    ROUTE_NAME,
    ROUTE_SHORT_NAME,
)

TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def _specs() -> dict:
    """Resolve the existing cache roots without importing the audit tool."""

    dataset = object.__new__(UniDataset)
    dataset.root = str(PROJECT_ROOT / "data")
    dataset.mips_max_hops = 2
    dataset.trimer_num_candidates = 4
    dataset.trimer_max_heavy_atoms = 384
    dataset.feature_cache_item_timeout = 240
    dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
    return dataset._lmdb_cache_specs({})


def _read_layer(name: str, spec: dict) -> dict:
    root = Path(spec["root"])
    for filename in ("metadata.json", "manifest.json", ".done"):
        if not (root / filename).is_file():
            raise RuntimeError(f"{name} cache is missing {filename}: {root}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(manifest, dict):
        raise RuntimeError(f"{name} cache metadata/manifest is not JSON object")
    expected_schema = spec.get("meta", {}).get("schema")
    if expected_schema is not None and metadata.get("schema") != expected_schema:
        raise RuntimeError(
            f"{name} cache schema mismatch: {metadata.get('schema')!r} != {expected_schema!r}"
        )
    count = int(metadata.get("count", manifest.get("count", -1)))
    if count < 0 or int(manifest.get("count", count)) != count:
        raise RuntimeError(f"{name} cache record count is invalid")
    store = LmdbLayerStore(root, expected_meta=spec.get("meta"))
    if len(store) != count:
        raise RuntimeError(f"{name} LMDB count mismatch: {len(store)} != {count}")
    return {"root": str(root), "schema": metadata.get("schema"), "count": count}


def _find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RuntimeError(f"checkpoint is missing: {path}")
        return path.resolve()
    candidates = sorted(
        (PROJECT_ROOT / "pretrained_models/mts").glob(
            "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one canonical checkpoint, found {len(candidates)}")
    return candidates[0].resolve()


def _build_model(meta: dict) -> UniEncoderAttention:
    model_cfg = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    return UniEncoderAttention(
        joint_embedding_dim=256,
        smiles_model_name=None,
        gnn_model_name=None,
        modality_list=["graph"],
        graph_num_layers=int(model_cfg.get("layers", 6)),
        graph_emb_dim=int(model_cfg.get("embedding_dim", 512)),
        graph_dropout=0.1,
        graph_encoder_type="mips_trimer_scage",
        scage_num_heads=int(model_cfg.get("heads", 8)),
        scage_ffn_hidden_dim=int(model_cfg.get("ffn_hidden_dim", 2048)),
        scage_use_pbc_distance=False,
        scage_use_descriptors=False,
        mips_core=meta.get("mips_core", "paper_corrected"),
        mips_max_hops=int(meta.get("mips_max_hops", 2)),
        mips_use_descriptors=bool(meta.get("mips_use_descriptors", True)),
        spatial_mode=meta.get("spatial_mode", "trimer_scage"),
        graph_geometry_mode=meta.get("graph_geometry_mode", "trimer_scage_mcl"),
        mcl_distance_percentiles=tuple(meta.get("mcl_distance_percentiles", (0.20, 0.50))),
        trimer_num_candidates=int(meta.get("trimer_num_candidates", 4)),
        trimer_max_heavy_atoms=int(meta.get("trimer_max_heavy_atoms", 384)),
        mips_variant=meta.get("mips_variant", "O8"),
        mips_fusion_mode="none",
        projection_mode="plain",
        modality_control="real",
        mips_atom_feature_mode="mips137",
        mips_attention_scale="head_dim",
        mips_norm_mode="post",
        mips_activation="relu",
        mips_spd_bias_mode="per_head",
        mips_path_bias_mode="per_head_single_path_node",
        mips_multi_scale_hop_gate=False,
        mips_semantics=meta.get("mips_semantics", "paper_semantic"),
        mips_descriptor_fusion_mode=meta.get("mips_descriptor_fusion_mode", "graph_md_residual"),
        mips_descriptor_components=meta.get("mips_descriptor_components", "md200"),
        mips_descriptor_disturbance=float(meta.get("mips_descriptor_disturbance", 0.0)),
        mips_backbone_mode=meta.get("mips_backbone_mode", "independent"),
        mips_input_norm=bool(meta.get("mips_input_norm", False)),
        mips_mask_mode=meta.get("mips_mask_mode", "zero"),
        mips_mask_policy=meta.get("mips_mask_policy", "canonical_exact"),
        mips_masked_loss_reduction=meta.get("mips_masked_loss_reduction", "atom_mean"),
        fusion_type="none",
        fp_mode="disabled",
    )


def _strict_checkpoint_load(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint does not contain a state_dict mapping")
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint.get("meta"), dict) else {}
    model = _build_model(meta)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint state_dict key mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise RuntimeError(f"checkpoint state_dict shape mismatch: {key}")
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"checkpoint state_dict contains non-finite tensor: {key}")
    model.load_state_dict(state, strict=True)
    return checkpoint, model


def _two_sample_forward_backward(model: UniEncoderAttention) -> dict:
    samples = []
    for smiles in ("*CCO*", "*CCC*"):
        topology = build_canonical_periodic_topology(smiles)
        attach_finite_trimer_mcl(topology, smiles)
        samples.append(topology)
    batch = mips_trimer_collate(samples)
    encoder = model.encoders["graph"].encoder
    encoder.train()
    mask = torch.zeros(batch.mips_x.size(0), dtype=torch.bool)
    mask[: min(2, mask.numel())] = True
    graph, nodes, _aux = encoder.forward_joint_pretrain(batch, mask)
    loss = graph.square().mean() + nodes.square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("doctor forward/backward loss is non-finite")
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("doctor forward/backward gradient is non-finite")
    return {"samples": 2, "loss": float(loss.detach())}


def _check_cli(python: str) -> None:
    for script in ("scripts/pretrain.py", "scripts/train.py"):
        result = subprocess.run(
            [python, script, "--help"], cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"{script} --help failed: {result.stderr[-500:]}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("production", "pretrain"), default="production")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1,2,3":
        raise RuntimeError(f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) requires CUDA_VISIBLE_DEVICES=1,2,3; got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError(f"doctor requires three visible CUDA devices; available={torch.cuda.device_count()}")
    _check_cli(args.python)

    encoder = MIPSLocalGraphEncoder()
    if not (encoder.max_hops == 2 and len(encoder.layers) == 6 and encoder.emb_dim == 512):
        raise RuntimeError("MTS local graph encoder configuration is not canonical")

    report = {"mode": args.mode, "cache": [], "checkpoint": None}
    if not args.skip_cache:
        specs = _specs()
        for name, spec in specs.items():
            report["cache"].append({"name": name, **_read_layer(name, spec)})
        split_root = PROJECT_ROOT / "data/splits/mips_shared5"
        missing = [task for task in TASKS if not (split_root / f"{task}.json").is_file()]
        if missing:
            raise RuntimeError("missing shared split manifests: " + ", ".join(missing))

    if args.mode == "pretrain" and not args.profile:
        raise RuntimeError("--mode pretrain requires --profile")
    checkpoint_path = _find_checkpoint(args.checkpoint)
    checkpoint, model = _strict_checkpoint_load(checkpoint_path)
    report["checkpoint"] = {"path": str(checkpoint_path), "state_dict_strict": True}
    report["forward_backward"] = _two_sample_forward_backward(model)
    print(json.dumps(report, sort_keys=True))
    print(f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) doctor: ready (local schema, strict state_dict, two-sample forward/backward)")


if __name__ == "__main__":
    main()
