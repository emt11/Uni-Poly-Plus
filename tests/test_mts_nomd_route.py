"""Targeted contract checks for GLT-V2 revision-2 pure O8/MIPS-CE route."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.modules.mts_glt_distill import DistillStudent
from src.training.finetune.config import parse_arguments
from src.training.finetune.engine import build_mts_downstream_model, select_mts_glt_distill_state
from src.training.pretrain.config import NOMD_SCHEMA, parse_arguments as parse_pretrain
from src.training.pretrain.glt_distill_engine import StudentContainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_r2_o8_nomd_mipsloss_005k.json"


def _batch():
    # Two small canonical graphs with topology-only fields; no MD/Trimer or
    # line sidecar attributes are provided.
    data = Data()
    data.mips_x = torch.zeros(4, 137)
    data.mips_x[:, :101] = torch.eye(101)[:4]
    data.mips_backbone_mask = torch.tensor([True, False, True, False])
    data.lga_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    data.lga_spd = torch.tensor([1, 1, 1, 1])
    data.lga_path_index = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])
    data.lga_path_mask = torch.ones(4, 2, dtype=torch.bool)
    data.lga_star_edge_mask = torch.ones(4, dtype=torch.bool)
    data.lga_source_image_shift = torch.zeros(4, dtype=torch.long)
    data.lga_path_shift = torch.zeros(4, 2, dtype=torch.long)
    data.polymer_link_mask = torch.ones(4, dtype=torch.bool)
    data.feature_schema = "mts-canonical-periodic-feature-v3"
    data.mips_local_lga_schema_version = 2
    data.topology_representation = "canonical_lifted"
    data.canonical_ru_atom_index = torch.arange(4)
    data.canonical_graph_index = torch.tensor([0, 0, 1, 1])
    data.canonical_local_index = torch.tensor([0, 1, 0, 1])
    data.canonical_first_node_index = torch.arange(4)
    data.canonical_graph_index = torch.tensor([0, 0, 1, 1])
    data.graph_available = torch.tensor([True, True])
    data.mips_boundary_distance = torch.tensor([0, 0])
    data.mips_condition_valid = torch.tensor([True, True])
    data.batch = torch.tensor([0, 0, 1, 1])
    data.mts_canonical_periodic = True
    return data


def test_nomd_config_is_strict_and_has_no_md_sidecar():
    payload = json.loads(CONFIG.read_text())
    assert payload["schema"] == NOMD_SCHEMA
    args = parse_pretrain(["--experiment_config", str(CONFIG)])
    assert args.use_md200 is False
    assert args.periodic_line_glt_sidecar is None
    assert args.cache_layers == "ru_base,topology"


def test_nomd_student_has_no_md_parameters_and_canonical_alias():
    model = DistillStudent(use_md200=False)
    assert isinstance(model.md_residual, nn.Identity)
    assert not any("md_residual" in name for name, _ in model.named_parameters())
    data = _batch()
    mask = torch.tensor([True, False, False, True])
    clean, fused = model.encode(data, atom_mask=mask)
    torch.testing.assert_close(clean, fused)
    assert clean.shape == (4, 512)


def test_nomd_container_uses_single_masked_ce_without_md_fields():
    container = StudentContainer(None, use_md200=False)
    out = container(_batch(), 0, atom_mask=torch.tensor([True, False, False, True]))
    assert out["atom_count"] == 2
    assert out["md_disturbed"] == 0 and out["md_total"] == 0
    assert torch.isfinite(out["atom_sum"])
    out["atom_sum"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in container.student.o8.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in container.atom_head.parameters())


def test_nomd_downstream_strict_state_and_mse_contract():
    args = parse_arguments([
        "--config_schema", "mts-glt-v2-r2-nomd-downstream",
        "--mts_glt_version", "distill_nomd", "--distill_repair_version", "none",
        "--no-use_md200", "--mips_norm_mode", "pre",
        "--target_transform", "standard", "--regression_loss", "mse",
    ])
    model = build_mts_downstream_model(args)
    state = DistillStudent(use_md200=False).state_dict()
    ck = {
        "schema": "mts-glt-v2-r2-o8-nomd-student-deploy-v1",
        "version": "none", "step": 5000, "geometry_revision": None,
        "use_md200": False, "state_dict": state,
        "o8_ffn_activation": "GELU(approximate='none')",
        "o8_ffn_hidden": "512->2048->512",
        "downstream_graph_adapter": "identity",
        "downstream_predictor": "512->512->1",
        "downstream_predictor_dropout": 0.1,
    }
    mapped = select_mts_glt_distill_state(
        model.state_dict(), ck, expected_version="none", expected_step=5000,
    )
    merged = model.state_dict(); merged.update(mapped)
    model.load_state_dict(merged, strict=True)
    assert not any("md_residual" in key for key in mapped)
