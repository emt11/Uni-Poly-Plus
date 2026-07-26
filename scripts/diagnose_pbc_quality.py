#!/usr/bin/env python3
"""Audit translational or screw-periodic RU geometry without training a model.

The script intentionally operates on unique P-SMILES only. It is useful before
an expensive cache rebuild: every row records whether the selected quality gate
accepted the RDKit trimer geometry or selected a non-periodic fallback.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
from collections import Counter

import pandas as pd
import torch
from rdkit import Chem

# Allow direct ``python scripts/diagnose_pbc_quality.py`` execution without
# requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.geom_data import (
    mol2periodic_pbc_coords,
    mol2polygen_periodic_coords,
    mol2screw_periodic_coords,
    mol2smer_context_coords,
    set_conformer_generation_config,
    set_screw_quality_config,
)


def _first(value, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _as_bool(value, default=False):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return bool(value.detach().cpu().flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return _as_bool(value[0], default) if value else default
    return bool(value)


def _summary(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _evaluate_smiles(task):
    """Run one quality evaluation in an isolated worker process."""
    torch.set_num_threads(1)
    smiles, profile, candidates, keep, periodic_mode, quality_config = task
    set_conformer_generation_config(
        conformer_3d_count=candidates,
        conformer_keep_count=keep,
        embed_tries_multiplier=8,
        conformer_profile=profile,
    )
    set_screw_quality_config(**quality_config)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "pbc_status": "invalid_smiles", "failure_reason": "rdkit_parse_failed"}
    try:
        builder = {
            "translation": mol2periodic_pbc_coords,
            "polygen": mol2polygen_periodic_coords,
            "screw": mol2screw_periodic_coords,
            "smer": mol2smer_context_coords,
        }[periodic_mode]
        data = builder(mol, optimizer="auto")
        lengths = getattr(data, "geom_attachment_bond_lengths", None)
        ratios = getattr(data, "geom_attachment_bond_ratios", None)
        t_norm = None
        if hasattr(data, "cell") and hasattr(data, "pbc") and bool(data.pbc.bool().any().item()):
            active_axis = int(torch.nonzero(data.pbc.bool(), as_tuple=False).flatten()[0])
            t_norm = float(torch.linalg.vector_norm(data.cell[active_axis]).item())
        return {
            "smiles": smiles,
            "pbc_status": str(getattr(data, "geom_pbc_status", "unknown")),
            "geom_input": str(getattr(data, "geom_input", "unknown")),
            "geom_build_ok": _as_bool(getattr(data, "geom_build_ok", False)),
            "geom_coordinate_ok": _as_bool(getattr(data, "geom_coordinate_ok", False)),
            "failure_reason": str(getattr(data, "geom_failed_reason", "")),
            "optimizer": str(getattr(data, "geom_optimizer_used", "unknown")),
            "converged": int(getattr(data, "geom_conformer_converged_count", 0)) > 0,
            "force_quality_accepted": _as_bool(getattr(data, "geom_force_quality_accepted", False)),
            "energy": _first(getattr(data, "geom_conformer_energies", None), None),
            "gradient_rms": _first(getattr(data, "geom_gradient_rms", None), None),
            "gradient_max": _first(getattr(data, "geom_gradient_max", None), None),
            "probe_energy_delta_per_atom": _first(
                getattr(data, "geom_probe_energy_delta_per_atom", None), None
            ),
            "T_norm": t_norm,
            "T_left_norm": _first(getattr(data, "geom_t_left_norm", None), None),
            "T_right_norm": _first(getattr(data, "geom_t_right_norm", None), None),
            "T_cosine": _first(getattr(data, "geom_t_cosine_similarity", None), None),
            "T_relative_length_difference": _first(getattr(data, "geom_t_relative_length_difference", None), None),
            "left_attachment_bond_length": _first(lengths, None),
            "right_attachment_bond_length": (
                float(lengths.detach().cpu().flatten()[1].item())
                if torch.is_tensor(lengths) and lengths.numel() >= 2 else None
            ),
            "left_attachment_bond_ratio": _first(ratios, None),
            "right_attachment_bond_ratio": (
                float(ratios.detach().cpu().flatten()[1].item())
                if torch.is_tensor(ratios) and ratios.numel() >= 2 else None
            ),
            "kabsch_rmsd_left": _first(getattr(data, "geom_kabsch_rmsd_left", None), None),
            "kabsch_rmsd_right": _first(getattr(data, "geom_kabsch_rmsd_right", None), None),
            "final_screw_rmsd_left": _first(getattr(data, "geom_final_screw_rmsd_left", None), None),
            "final_screw_rmsd_right": _first(getattr(data, "geom_final_screw_rmsd_right", None), None),
            "rotation_consistency_deg": _first(
                getattr(data, "geom_rotation_consistency_deg", None), None
            ),
            "translation_relative_difference": _first(
                getattr(data, "geom_translation_relative_difference", None), None
            ),
            "joint_screw_rmsd": _first(getattr(data, "geom_joint_screw_rmsd", None), None),
            "screw_angle_deg": _first(getattr(data, "geom_screw_angle", None), None),
            "screw_axial_rise": _first(getattr(data, "geom_screw_axial_rise", None), None),
            "screw_torsion_deg": _first(getattr(data, "geom_screw_torsion_degrees", None), None),
            "screw_energy_per_atom": _first(getattr(data, "geom_screw_energy_per_atom", None), None),
            "screw_symmetry_rmsd": _first(getattr(data, "geom_screw_symmetry_rmsd", None), None),
            "screw_source": str(getattr(
                data, "geom_screw_source",
                "polygen_periodic" if _as_bool(getattr(data, "polygen_periodic_valid", False)) else "unknown"
            )),
            "screw_source_id": int(getattr(data, "geom_screw_source_id", -1)),
            "geometry_source_id": int(getattr(data, "geometry_source_id", getattr(data, "geom_screw_source_id", -1))),
            "screw_fit_point_count": _first(getattr(data, "geom_screw_fit_point_count", None), None),
            "primary_failure_reason": str(getattr(data, "geom_primary_failed_reason", "")),
            "five_cell_minimum_distance": _first(
                getattr(data, "geom_five_cell_minimum_distance", None), None
            ),
            "periodic_ru_count": (
                int(getattr(data, "periodic_ru_count", 0))
                if _as_bool(getattr(data, "polygen_periodic_valid", False)) else None
            ),
            "periodic_cell_length": _first(getattr(data, "periodic_cell_length", None), None),
            "periodic_closure_error": _first(
                getattr(data, "periodic_closure_error", None), None
            ),
            "periodic_optimization_loss": _first(
                getattr(data, "periodic_optimization_loss", None), None
            ),
            "boundary_bond_error": _first(
                getattr(data, "periodic_boundary_bond_error", None), None
            ),
            "boundary_angle_error_deg": _first(
                getattr(data, "periodic_boundary_angle_error_deg", None), None
            ),
            "boundary_torsion_error_deg": _first(
                getattr(data, "periodic_boundary_torsion_error_deg", None), None
            ),
            "minimum_nonbonded_distance": _first(
                getattr(data, "periodic_minimum_nonbonded_distance", None), None
            ),
            "candidate_count": int(getattr(data, "periodic_candidate_count", 0)),
            "periodic_failure_counts": json.dumps(
                getattr(data, "periodic_failure_counts", {}), sort_keys=True
            ),
        }
    except Exception as exc:
        return {
            "smiles": smiles,
            "pbc_status": "diagnostic_error",
            "geom_build_ok": False,
            "failure_reason": f"{type(exc).__name__}: {str(exc)[:220]}",
        }


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Diagnose quality-gated 1D PBC geometry on unique P-SMILES.")
    parser.add_argument("--dataset", default="smi_all", help="CSV stem under data/raw, without .csv")
    parser.add_argument("--root", default="./data", help="Data root containing raw/ and processed/")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conformer-profile", choices=["quality"], default="quality")
    parser.add_argument("--conformer-candidates", type=int, default=4)
    parser.add_argument("--conformer-keep", type=int, default=2)
    parser.add_argument(
        "--periodic-mode",
        choices=["translation", "polygen", "screw", "smer"],
        default="polygen",
        help="polygen tests deterministic fractional-coordinate periodic optimization.",
    )
    parser.add_argument("--output-tag", default="", help="Optional suffix for side-by-side diagnostics.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ff-gradient-rms-max", type=float, default=0.05)
    parser.add_argument("--ff-gradient-max", type=float, default=0.25)
    parser.add_argument("--ff-probe-steps", type=int, default=20)
    parser.add_argument("--ff-probe-energy-delta-per-atom-max", type=float, default=5e-5)
    parser.add_argument("--screw-kabsch-rmsd-max", type=float, default=1.5)
    parser.add_argument("--screw-rotation-consistency-deg", type=float, default=30.0)
    parser.add_argument("--screw-translation-relative-max", type=float, default=0.30)
    parser.add_argument("--screw-final-rmsd-max", type=float, default=1.5)
    parser.add_argument("--screw-energy-per-atom-max", type=float, default=5.0)
    parser.add_argument("--screw-center-gradient-rms-max", type=float, default=10.0)
    parser.add_argument(
        "--isolate-every",
        type=int,
        default=10,
        help="Restart the RDKit worker after this many samples; 0 disables process isolation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of isolated RDKit workers. Use 2-4 when memory permits.",
    )
    args = parser.parse_args()

    source = os.path.join(args.root, "raw", f"{args.dataset}.csv")
    frame = pd.read_csv(source)
    unique_smiles = list(dict.fromkeys(str(value).strip() for value in frame.iloc[:, 0]))
    if not unique_smiles:
        raise ValueError(f"No SMILES found in {source}")
    sample_size = min(int(args.sample_size), len(unique_smiles))
    sampled = random.Random(args.seed).sample(unique_smiles, sample_size)

    rows = []
    quality_config = {
        "gradient_rms_max": args.ff_gradient_rms_max,
        "gradient_max": args.ff_gradient_max,
        "probe_steps": args.ff_probe_steps,
        "probe_energy_delta_per_atom_max": args.ff_probe_energy_delta_per_atom_max,
        "kabsch_rmsd_max": args.screw_kabsch_rmsd_max,
        "rotation_consistency_deg": args.screw_rotation_consistency_deg,
        "translation_relative_max": args.screw_translation_relative_max,
        "final_rmsd_max": args.screw_final_rmsd_max,
        "screw_energy_per_atom_max": args.screw_energy_per_atom_max,
        "screw_center_gradient_rms_max": args.screw_center_gradient_rms_max,
    }
    tasks = [
        (smiles, args.conformer_profile, args.conformer_candidates, args.conformer_keep,
         args.periodic_mode, quality_config)
        for smiles in sampled
    ]
    if args.isolate_every > 0:
        # RDKit force-field allocations can grow across hundreds of difficult
        # trimers. A single process and bounded maxtasksperchild make this
        # diagnostic deterministic while releasing native memory regularly.
        context = mp.get_context("spawn")
        with context.Pool(
            processes=max(1, int(args.workers)),
            maxtasksperchild=args.isolate_every,
        ) as pool:
            iterator = pool.imap(_evaluate_smiles, tasks, chunksize=1)
            for index, row in enumerate(iterator, start=1):
                rows.append(row)
                if index % 25 == 0 or index == sample_size:
                    print(f"[pbc-quality] {index}/{sample_size}")
    else:
        for index, task in enumerate(tasks, start=1):
            rows.append(_evaluate_smiles(task))
            if index % 25 == 0 or index == sample_size:
                print(f"[pbc-quality] {index}/{sample_size}")

    output = pd.DataFrame(rows)
    output_dir = args.output_dir or (
        "data/processed/scage" if args.periodic_mode in {"polygen", "screw", "smer"} else "data/processed/painn"
    )
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    stem = f"{args.periodic_mode}_quality_{args.dataset}_seed{args.seed}_n{sample_size}{suffix}"
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    json_path = os.path.join(output_dir, f"{stem}.json")
    output.to_csv(csv_path, index=False)

    status_counts = Counter(output["pbc_status"].fillna("unknown"))
    accepted_status = {
        "translation": "pbc_3d_converged",
        "polygen": "polygen_periodic_valid",
        "screw": "screw_periodic_converged",
        "smer": "smer_3d_valid",
    }[args.periodic_mode]
    accepted = int(status_counts[accepted_status])
    accepted_rows = output[output["pbc_status"] == accepted_status]
    summary = {
        "dataset": args.dataset,
        "source": source,
        "seed": args.seed,
        "sample_size": sample_size,
        "conformer_profile": args.conformer_profile,
        "conformer_candidates": args.conformer_candidates,
        "conformer_keep": args.conformer_keep,
        "periodic_mode": args.periodic_mode,
        "quality_config": quality_config,
        "force_field_iterations": 500,
        "pbc_status_counts": dict(status_counts),
        "success_status": accepted_status,
        "geometry_context_success_count": accepted,
        "geometry_context_success_rate": accepted / sample_size if sample_size else None,
        "pbc_quality_pass_count": accepted,
        "pbc_quality_pass_rate": accepted / sample_size if sample_size else None,
        "geometry_source_counts": {
            str(key): int(value)
            for key, value in Counter(output.get("screw_source", pd.Series(dtype=str)).fillna("unknown")).items()
        },
        "euclidean_fallback_usable_count": int(
            ((output["pbc_status"] != accepted_status) & output["geom_coordinate_ok"].fillna(False)).sum()
        ),
        "T_norm": _summary(output.get("T_norm", pd.Series(dtype=float)).tolist()),
        "T_left_norm": _summary(output.get("T_left_norm", pd.Series(dtype=float)).tolist()),
        "T_right_norm": _summary(output.get("T_right_norm", pd.Series(dtype=float)).tolist()),
        "T_cosine": _summary(output.get("T_cosine", pd.Series(dtype=float)).tolist()),
        "T_relative_length_difference": _summary(
            output.get("T_relative_length_difference", pd.Series(dtype=float)).tolist()
        ),
        "attachment_bond_length": _summary(
            output.get("left_attachment_bond_length", pd.Series(dtype=float)).tolist()
            + output.get("right_attachment_bond_length", pd.Series(dtype=float)).tolist()
        ),
        "gradient_rms": _summary(output.get("gradient_rms", pd.Series(dtype=float)).tolist()),
        "gradient_max": _summary(output.get("gradient_max", pd.Series(dtype=float)).tolist()),
        "kabsch_rmsd": _summary(
            output.get("kabsch_rmsd_left", pd.Series(dtype=float)).tolist()
            + output.get("kabsch_rmsd_right", pd.Series(dtype=float)).tolist()
        ),
        "final_screw_rmsd": _summary(
            output.get("final_screw_rmsd_left", pd.Series(dtype=float)).tolist()
            + output.get("final_screw_rmsd_right", pd.Series(dtype=float)).tolist()
        ),
        "rotation_consistency_deg": _summary(
            output.get("rotation_consistency_deg", pd.Series(dtype=float)).tolist()
        ),
        "translation_relative_difference": _summary(
            output.get("translation_relative_difference", pd.Series(dtype=float)).tolist()
        ),
        "screw_angle_deg": _summary(output.get("screw_angle_deg", pd.Series(dtype=float)).tolist()),
        "screw_axial_rise": _summary(output.get("screw_axial_rise", pd.Series(dtype=float)).tolist()),
        "screw_torsion_deg": _summary(output.get("screw_torsion_deg", pd.Series(dtype=float)).tolist()),
        "screw_energy_per_atom": _summary(
            output.get("screw_energy_per_atom", pd.Series(dtype=float)).tolist()
        ),
        "screw_symmetry_rmsd": _summary(
            output.get("screw_symmetry_rmsd", pd.Series(dtype=float)).tolist()
        ),
        "screw_source_counts": dict(Counter(
            output.get("screw_source", pd.Series(dtype=str)).fillna("unknown")
        )),
        "five_cell_minimum_distance": _summary(
            output.get("five_cell_minimum_distance", pd.Series(dtype=float)).tolist()
        ),
        "periodic_ru_count_counts": dict(Counter(
            accepted_rows.get("periodic_ru_count", pd.Series(dtype=int)).dropna().astype(int)
        )),
        "periodic_cell_length": _summary(
            accepted_rows.get("periodic_cell_length", pd.Series(dtype=float)).tolist()
        ),
        "periodic_optimization_loss": _summary(
            accepted_rows.get("periodic_optimization_loss", pd.Series(dtype=float)).tolist()
        ),
        "periodic_closure_error": _summary(
            accepted_rows.get("periodic_closure_error", pd.Series(dtype=float)).tolist()
        ),
        "boundary_bond_error": _summary(
            accepted_rows.get("boundary_bond_error", pd.Series(dtype=float)).tolist()
        ),
        "boundary_angle_error_deg": _summary(
            accepted_rows.get("boundary_angle_error_deg", pd.Series(dtype=float)).tolist()
        ),
        "boundary_torsion_error_deg": _summary(
            accepted_rows.get("boundary_torsion_error_deg", pd.Series(dtype=float)).tolist()
        ),
        "minimum_nonbonded_distance": _summary(
            accepted_rows.get("minimum_nonbonded_distance", pd.Series(dtype=float)).tolist()
        ),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[pbc-quality] wrote {csv_path}")
    print(f"[pbc-quality] wrote {json_path}")
    print(f"[pbc-quality] quality pass: {accepted}/{sample_size} = {summary['pbc_quality_pass_rate']:.2%}")


if __name__ == "__main__":
    main()
