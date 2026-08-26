#!/usr/bin/env python3
"""Real-batch parity and gradient checks for the O8 BC/BT screening arms."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.canonical_periodic import build_canonical_periodic_topology  # noqa: E402
from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.periodic_line_glt import build_periodic_line_sample  # noqa: E402
from src.dataset.trimer_mcl import attach_finite_trimer_mcl  # noqa: E402
from src.modules.mts_glt_v2 import MTSGraphLineModelV2  # noqa: E402


def _attach(data, record):
    tokens, relations = record["tokens"], record["relations"]
    data.glt_geometry_valid = bool(record["geometry_valid"])
    token_fields = {
        "token_atom_a": [x["atom_a"] for x in tokens],
        "token_atom_b": [x["atom_b"] for x in tokens],
        "token_shift": [x["shift"] for x in tokens],
        "token_endpoint_z_a": [x["z_a"] for x in tokens],
        "token_endpoint_z_b": [x["z_b"] for x in tokens],
        "token_bond_type": [x["bond_type"] for x in tokens],
        "token_label": [x["label"] for x in tokens],
        "token_observation_count": [x["observation_count"] for x in tokens],
        "token_valid": [x["valid"] for x in tokens],
    }
    for name, values in token_fields.items():
        dtype = torch.bool if name == "token_valid" else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_token_observation_distances = torch.tensor(
        [x["distances"] for x in tokens], dtype=torch.float32
    )
    relation_fields = {
        "relation_source": [x["source"] for x in relations],
        "relation_target": [x["target"] for x in relations],
        "relation_center_atom": [x["center_atom"] for x in relations],
        "relation_multiplicity": [x["multiplicity"] for x in relations],
        "relation_observation_count": [x["observation_count"] for x in relations],
        "relation_valid": [x["valid"] for x in relations],
        "relation_is_fallback": [x["fallback"] for x in relations],
    }
    for name, values in relation_fields.items():
        dtype = torch.bool if name in {
            "relation_valid", "relation_is_fallback"
        } else torch.long
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=dtype))
    data.glt_relation_observation_angles = torch.tensor(
        [x["angles"] for x in relations], dtype=torch.float32
    )
    data.mips_md = torch.zeros(200)
    data.mips_md_valid = torch.tensor(False)
    return data


def _batch():
    samples = []
    for smiles in ("*CCO*", "*C=C*"):
        data = build_canonical_periodic_topology(smiles)
        attach_finite_trimer_mcl(data, smiles, num_candidates=1)
        record = build_periodic_line_sample(
            sample_key_from_smiles(smiles), data, data
        )
        if not record["geometry_valid"]:
            raise RuntimeError(f"sanity geometry failed: {smiles}")
        samples.append(_attach(data, record))
    return mips_trimer_collate(samples)


def _load_model(checkpoint, mode):
    model = MTSGraphLineModelV2(
        glt_layers=6, glt_attention_variant="mips",
        o8_bond_bias_mode=mode,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = {
        key[len("model."):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    state = model.state_dict()
    missing = sorted(set(state) - set(source))
    expected_missing = (
        ["o8.direct_bond_bias.projection.weight"] if mode != "none" else []
    )
    if missing != expected_missing or sorted(set(source) - set(state)):
        raise RuntimeError(
            f"unexpected checkpoint mapping for {mode}: missing={missing}"
        )
    state.update(source)
    model.load_state_dict(state, strict=True)
    model.downstream_mode = "o8_glt_atom"
    return model


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=ROOT / "results/mts_glt_v2/o8_bond_type_screen_v1/gradient_sanity.json",
        type=Path,
    )
    args = parser.parse_args(argv)
    torch.manual_seed(42)
    batch = _batch()
    baseline = _load_model(args.checkpoint, "none").eval()
    control = _load_model(args.checkpoint, "control").eval()
    bond_type = _load_model(args.checkpoint, "type").eval()
    with torch.no_grad():
        prediction_b = baseline(batch)
        prediction_bc = control(batch)
        prediction_bt = bond_type(batch)
        torch.testing.assert_close(prediction_b, prediction_bc, rtol=0, atol=0)
        torch.testing.assert_close(prediction_b, prediction_bt, rtol=0, atol=0)
        initial = baseline.o8.atom_embedding(batch)
        spd_b = baseline.o8.spd_embedding(batch.lga_spd)
        spd_bc = control.o8.spd_embedding(batch.lga_spd)
        path_b = baseline.o8.path_bias(initial, batch)
        path_bc = control.o8.path_bias(initial, batch)
        torch.testing.assert_close(spd_b, spd_bc, rtol=0, atol=0)
        torch.testing.assert_close(path_b, path_bc, rtol=0, atol=0)
        _, canonical = baseline._o8_canonical_states(batch)
        glt_b = baseline.glt(batch, canonical_atom_states=canonical)["line_states"]
        glt_bc = control.glt(batch, canonical_atom_states=canonical)["line_states"]
        torch.testing.assert_close(glt_b, glt_bc, rtol=0, atol=0)

    gradient = {}
    for name, model in (("BC", control), ("BT", bond_type)):
        model.train()
        model.zero_grad(set_to_none=True)
        model(batch).square().mean().backward()
        grad = model.o8.direct_bond_bias.projection.weight.grad
        gradient[name] = {
            "finite": bool(grad is not None and torch.isfinite(grad).all()),
            "absolute_sum": float(grad.abs().sum()) if grad is not None else 0.0,
        }
        if not gradient[name]["finite"] or gradient[name]["absolute_sum"] <= 0:
            raise RuntimeError(f"{name} first gradient is not finite/nonzero")

    module_bc = control.o8.direct_bond_bias
    module_bt = bond_type.o8.direct_bond_bias
    with torch.no_grad():
        module_bc.projection.weight.copy_(
            torch.arange(48, dtype=torch.float32).reshape(8, 6) / 100
        )
        module_bt.projection.weight.zero_()
        module_bt.projection.weight[:, 1] = 0.1
        module_bt.projection.weight[:, 2] = 0.3
        bc_categories = module_bc.category_bias(torch.tensor([1, 2]))
        bt_categories = module_bt.category_bias(torch.tensor([1, 2]))
        if not torch.equal(bc_categories[0], bc_categories[1]):
            raise RuntimeError("BC unexpectedly distinguishes categories")
        if torch.equal(bt_categories[0], bt_categories[1]):
            raise RuntimeError("BT cannot distinguish categories")
        relation_bias = module_bt(batch)
        categories = module_bt.relation_categories(batch)
        if not torch.equal(
            relation_bias[categories < 0],
            torch.zeros_like(relation_bias[categories < 0]),
        ):
            raise RuntimeError("non-bond bias is nonzero")
        edge = batch.lga_edge_index.long()
        shifts = batch.lga_source_image_shift.long()
        lookup = {
            (int(edge[0, i]), int(edge[1, i]), int(shifts[i])): i
            for i in range(edge.size(1)) if int(batch.lga_spd[i]) == 1
        }
        for (source, target, shift), index in lookup.items():
            inverse = lookup[(target, source, -shift)]
            torch.testing.assert_close(
                relation_bias[index], relation_bias[inverse], rtol=0, atol=0
            )

    parameter_count = {
        "BC": sum(p.numel() for p in module_bc.parameters()),
        "BT": sum(p.numel() for p in module_bt.parameters()),
    }
    if parameter_count["BC"] != parameter_count["BT"]:
        raise RuntimeError("BC/BT parameter counts differ")
    payload = {
        "schema": "mts-glt-v2-o8-bond-type-gradient-sanity-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "step0_prediction_parity": True,
        "first_backward": gradient,
        "category_diagnostic": {
            "BC_categories_equal": True,
            "BT_categories_different": True,
        },
        "endpoint_swap_symmetry": True,
        "nonbond_exact_zero": True,
        "spd_bias_unchanged": True,
        "path_bias_unchanged": True,
        "glt_line_states_unchanged": True,
        "new_parameter_count": parameter_count,
        "all_finite": True,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
