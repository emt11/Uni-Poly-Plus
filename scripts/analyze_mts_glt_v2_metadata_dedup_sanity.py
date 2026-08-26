#!/usr/bin/env python3
"""Build and sanity-check the explicit GLT-v2 FULL/DEDUP modes.

This is a short CPU sanity only.  It never launches a data loader, trainer,
GPU job, or long trajectory.  The output paths are isolated from all existing
GLT-v2 artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.dataset.periodic_line_glt import build_periodic_line_sample
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.periodic_line_glt_v2 import GLTMaskedLineHeadV2, NUM_LINE_LABELS
from src.training.pretrain.glt_v2_engine import (
    MTSGLTV2PretrainContainer,
    _projection,
)
from src.training.pretrain.engine import _joint_canonical_mask
from src.training.pretrain.glt_v2_objectives import make_masked_line_inputs_v2
from src.training.pretrain.metadata_dedup import (
    build_matched_pretrain_containers,
    compare_shared_state_by_name_and_shape,
    parameter_accounting,
)


def _sample(smiles):
    topology = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(topology, smiles, num_candidates=1)
    return topology


def _attach(data, record):
    tokens, relations = record["tokens"], record["relations"]
    data.glt_geometry_valid = bool(record["geometry_valid"])
    token_fields = {
        "token_atom_a": [item["atom_a"] for item in tokens],
        "token_atom_b": [item["atom_b"] for item in tokens],
        "token_shift": [item["shift"] for item in tokens],
        "token_endpoint_z_a": [item["z_a"] for item in tokens],
        "token_endpoint_z_b": [item["z_b"] for item in tokens],
        "token_bond_type": [item["bond_type"] for item in tokens],
        "token_label": [item["label"] for item in tokens],
        "token_observation_count": [item["observation_count"] for item in tokens],
        "token_valid": [item["valid"] for item in tokens],
    }
    for name, values in token_fields.items():
        dtype = torch.bool if name == "token_valid" else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_token_observation_distances = torch.tensor(
        [item["distances"] for item in tokens], dtype=torch.float32
    )
    relation_fields = {
        "relation_source": [item["source"] for item in relations],
        "relation_target": [item["target"] for item in relations],
        "relation_center_atom": [item["center_atom"] for item in relations],
        "relation_multiplicity": [item["multiplicity"] for item in relations],
        "relation_observation_count": [item["observation_count"] for item in relations],
        "relation_valid": [item["valid"] for item in relations],
        "relation_is_fallback": [item["fallback"] for item in relations],
    }
    for name, values in relation_fields.items():
        dtype = torch.bool if name in {"relation_valid", "relation_is_fallback"} else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_relation_observation_angles = torch.tensor(
        [item["angles"] for item in relations], dtype=torch.float32
    )
    data.mips_md = torch.zeros(200)
    data.mips_md_valid = torch.tensor(False)
    return data


def _batch():
    smiles = ("*CCO*", "*COC*")
    samples = [_sample(item) for item in smiles]
    records = [
        build_periodic_line_sample(sample_key_from_smiles(item), sample, sample)
        for item, sample in zip(smiles, samples)
    ]
    return mips_trimer_collate([_attach(sample, record) for sample, record in zip(samples, records)])


def _same_masks(data):
    frequencies = torch.ones(200000, dtype=torch.long)
    atom_a = _joint_canonical_mask(data, 42, 0, 0.30)
    atom_b = _joint_canonical_mask(data, 42, 0, 0.30)
    line_a = make_masked_line_inputs_v2(
        data, 0.40, frequencies, torch.Generator().manual_seed(42)
    )
    line_b = make_masked_line_inputs_v2(
        data, 0.40, frequencies, torch.Generator().manual_seed(42)
    )
    return bool(torch.equal(atom_a, atom_b)), bool(
        all(torch.equal(line_a[key], line_b[key]) for key in line_a)
    )


def _constant_observations(data):
    """Make count changes metadata-only by repeating each first observation."""
    base = copy.deepcopy(data)
    distance = base.glt_token_observation_distances
    base.glt_token_observation_distances = distance[:, :1].expand(-1, 3).clone()
    valid = base.glt_token_valid.bool()
    base.glt_token_observation_count = torch.where(
        valid, torch.ones_like(base.glt_token_observation_count),
        torch.zeros_like(base.glt_token_observation_count),
    )
    changed = copy.deepcopy(base)
    changed.glt_token_observation_count = torch.where(
        valid, torch.full_like(base.glt_token_observation_count, 3),
        torch.zeros_like(base.glt_token_observation_count),
    )
    # Keep the angle moments identical while changing only multiplicity.
    angles = base.glt_relation_observation_angles
    base.glt_relation_observation_angles = angles[:, :1].expand(-1, 3).clone()
    changed.glt_relation_observation_angles = base.glt_relation_observation_angles.clone()
    base.glt_relation_multiplicity = torch.ones_like(base.glt_relation_multiplicity)
    changed.glt_relation_multiplicity = torch.full_like(
        base.glt_relation_multiplicity, 3
    )
    return base, changed


def _forward(model, data, *, backward=False):
    model.zero_grad(set_to_none=True)
    output = model(data)["graph_geometry"]
    if backward:
        output.square().mean().backward()
    return output.detach()


def run(output_path):
    torch.set_num_threads(1)
    data = _batch()
    rng_before = torch.get_rng_state()
    def make_container(metadata_mode):
        model = MTSGraphLineModelV2(
            glt_layers=6, glt_attention_variant="mips",
            glt_metadata_mode=metadata_mode, use_compact19=False,
        )
        for module in (
            model.atom_fusion_norm, model.atom_fusion_projection,
            model.compact19_residual, model.o8.md_residual,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        model.atom_channel_gate.requires_grad = False
        return MTSGLTV2PretrainContainer(
            model,
            torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
            GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
            torch.ones(NUM_LINE_LABELS, dtype=torch.long),
        )

    full_container, reference_container, init_report = (
        build_matched_pretrain_containers(
            make_container, "full", seed=1729
        )
    )
    # Build the actual DEDUP container from the same FULL reference separately
    # so the artifact reports the real complete-container namespace delta.
    dedup_container, _, dedup_init_report = build_matched_pretrain_containers(
        make_container, "dedup", seed=1729
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    assert init_report["source_only"] == []
    assert dedup_init_report["source_only"] == [
        "model.glt.relation_multiplicity_embedding.weight",
        "model.glt.token_count_embedding.weight",
    ]
    full = full_container.model.glt
    dedup = dedup_container.model.glt
    actual_cross_report = compare_shared_state_by_name_and_shape(
        full_container, dedup_container
    )
    assert actual_cross_report["mismatch"] == []
    assert len(actual_cross_report["shared_tensors"]) == len(
        actual_cross_report["equal_tensors"]
    )
    accounting = parameter_accounting(full_container, dedup_container)

    full.eval()
    dedup.eval()
    base, metadata_changed = _constant_observations(data)
    count_changed = copy.deepcopy(base)
    count_changed.glt_token_observation_count = torch.where(
        count_changed.glt_token_valid.bool(),
        torch.full_like(count_changed.glt_token_observation_count, 3),
        torch.zeros_like(count_changed.glt_token_observation_count),
    )
    multiplicity_changed = copy.deepcopy(base)
    multiplicity_changed.glt_relation_multiplicity = torch.full_like(
        multiplicity_changed.glt_relation_multiplicity, 3
    )
    with torch.no_grad():
        full_base = _forward(full, base)
        full_changed = _forward(full, metadata_changed)
        dedup_base = _forward(dedup, base)
        dedup_changed = _forward(dedup, metadata_changed)
        full_count = _forward(full, count_changed)
        full_multiplicity = _forward(full, multiplicity_changed)
        dedup_count = _forward(dedup, count_changed)
        dedup_multiplicity = _forward(dedup, multiplicity_changed)
    # DEDUP removes only the two additive metadata embeddings.  Repeated
    # observations keep mean/variance fixed, so this is a direct controlled
    # test of that contract.
    dedup_metadata_invariant = bool(torch.allclose(
        dedup_base, dedup_changed, atol=1e-6, rtol=1e-6
    ))
    full_metadata_sensitive = bool(not torch.allclose(
        full_base, full_changed, atol=1e-6, rtol=1e-6
    ))
    full_count_sensitive = bool(not torch.allclose(
        full_base, full_count, atol=1e-6, rtol=1e-6
    ))
    full_multiplicity_sensitive = bool(not torch.allclose(
        full_base, full_multiplicity, atol=1e-6, rtol=1e-6
    ))
    dedup_count_invariant = bool(torch.allclose(
        dedup_base, dedup_count, atol=1e-6, rtol=1e-6
    ))
    dedup_multiplicity_forward_invariant = bool(torch.allclose(
        dedup_base, dedup_multiplicity, atol=1e-6, rtol=1e-6
    ))

    shift_changed = copy.deepcopy(base)
    valid = shift_changed.glt_token_valid.bool()
    shift_changed.glt_token_shift = shift_changed.glt_token_shift.clone()
    shift_changed.glt_token_shift[valid] = (
        shift_changed.glt_token_shift[valid] + 1
    ).clamp(-1, 1)
    with torch.no_grad():
        shift_base = dedup.line_inputs(base)
        shift_alt = dedup.line_inputs(shift_changed)
        rel_base = dedup._relations(base, torch.float32)[2]
        rel_mult_alt = dedup._relations(metadata_changed, torch.float32)[2]
    angle_count_changed = copy.deepcopy(base)
    relation_valid = angle_count_changed.glt_relation_valid.bool()
    angle_count_changed.glt_relation_observation_angles = (
        angle_count_changed.glt_relation_observation_angles[:, :1]
        .expand(-1, 3).clone()
    )
    angle_count_changed.glt_relation_observation_count = torch.where(
        relation_valid,
        torch.full_like(angle_count_changed.glt_relation_observation_count, 3),
        torch.zeros_like(angle_count_changed.glt_relation_observation_count),
    )
    with torch.no_grad():
        rel_count_alt = dedup._relations(angle_count_changed, torch.float32)[2]
    abs_shift_sensitive = bool(not torch.allclose(
        shift_base, shift_alt, atol=1e-6, rtol=1e-6
    ))
    multiplicity_invariant = bool(torch.allclose(
        rel_base, rel_mult_alt, atol=1e-6, rtol=1e-6
    ))
    angle_count_sensitive = bool(not torch.allclose(
        rel_base, rel_count_alt, atol=1e-6, rtol=1e-6
    ))

    # One small backward pass per mode establishes finite gradients without
    # invoking the pretraining engine.
    full.train()
    dedup.train()
    full_fb = _forward(full, base, backward=True)
    dedup_fb = _forward(dedup, base, backward=True)
    finite_forward_backward = bool(
        torch.isfinite(full_fb).all()
        and torch.isfinite(dedup_fb).all()
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in list(full.parameters()) + list(dedup.parameters())
        )
    )
    mask_atom_equal, mask_line_equal = _same_masks(data)
    sample_ids = ["*CCO*", "*COC*"]
    identity_fields_equal = bool(
        torch.equal(data.glt_token_label, _batch().glt_token_label)
        and torch.equal(data.glt_token_endpoint_z_a, _batch().glt_token_endpoint_z_a)
    )

    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "shared_init_sanity.json").write_text(json.dumps({
        **dedup_init_report,
        "full_reference_container_source_only": init_report["source_only"],
        "full_reference_container_equal_tensors": len(init_report["equal_tensors"]),
        "actual_full_dedup_shared_tensors": len(
            actual_cross_report["shared_tensors"]
        ),
        "actual_full_dedup_equal_tensors": len(
            actual_cross_report["equal_tensors"]
        ),
        "finite_forward_backward": finite_forward_backward,
    }, indent=2) + "\n", encoding="utf-8")
    (root / "metadata_usage_sanity.json").write_text(json.dumps({
        "metadata_mode": ["full", "dedup"],
        "dedup_distance_count_and_angle_multiplicity_invariant": dedup_metadata_invariant,
        "dedup_angle_multiplicity_invariant": multiplicity_invariant,
        "full_uses_distance_count_or_angle_multiplicity": full_metadata_sensitive,
        "full_distance_count_sensitive": full_count_sensitive,
        "full_angle_multiplicity_sensitive": full_multiplicity_sensitive,
        "dedup_distance_count_invariant": dedup_count_invariant,
        "dedup_angle_multiplicity_forward_invariant": dedup_multiplicity_forward_invariant,
        "dedup_abs_shift_sensitive": abs_shift_sensitive,
        "dedup_angle_count_sensitive": angle_count_sensitive,
        "same_sample_ids": sample_ids,
        "sample_identity_fields_equal": identity_fields_equal,
        "masked_atom_masks_equal": mask_atom_equal,
        "masked_line_inputs_equal": mask_line_equal,
        "controlled_observation_note": (
            "Distance observations and angle observations were repeated so changing "
            "their count leaves mean/variance unchanged; only FULL additive count "
            "and multiplicity embeddings are being tested."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    (root / "parameter_accounting.json").write_text(
        json.dumps(accounting, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_root": str(root),
        "shared_tensors": len(init_report["shared_tensors"]),
        "equal_tensors": len(init_report["equal_tensors"]),
        "mismatch": len(init_report["mismatch"]),
        "removed_parameters": accounting["removed_parameter_names"],
        "parameter_delta": accounting["parameter_delta_full_minus_dedup"],
        "metadata_invariant": dedup_metadata_invariant,
        "full_sensitive": full_metadata_sensitive,
        "multiplicity_invariant": multiplicity_invariant,
        "abs_shift_sensitive": abs_shift_sensitive,
        "angle_count_sensitive": angle_count_sensitive,
        "mask_equal": mask_atom_equal and mask_line_equal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/mts_glt_v2/metadata_dedup_matched5k_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
