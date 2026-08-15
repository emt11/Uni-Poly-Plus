"""Sym-MCL v1 mechanism statistics on the real eat cohort.

Counts, per MCL-eligible sample, how the inverse-OR symmetric mask changes
visibility relative to the plain single-observation hard mask, for both
q20 and q50.  Uses the exact downstream dataset construction of
``run_finetune_job`` so the population matches the screening cohort.

Run: PYTHONPATH=. python scripts/sym_mcl_mechanism_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.trimer_mcl import build_sym_mcl_mask  # noqa: E402
from src.training.finetune.config import parse_arguments  # noqa: E402


def main(argv=None):
    sidecar = (
        "data/processed/mips_trimer_scage/star_rbf_v2/"
        "fb6a23c6bc3dc9f85193d7dad6cc30aca57119626778beca10689ffe89e7909c/"
        "downstream_union/ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2"
    )
    sys.argv = [sys.argv[0], *[
        "--config_schema", "mts-config-v3",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--use_star_rbf", "--use_mcl", "--use_md200",
        "--star_rbf_upper", "3.75",
        "--star_rbf_v2_sidecar", sidecar,
        "--tasks", "eat",
        "--batch_size", "32", "--eval_batch_size", "64",
        "--amp_dtype", "fp32", "--loader_workers", "0",
    ]]
    args = parse_arguments()
    from src.dataset import UniDataset

    dataset = UniDataset(
        root=args.root,
        dataset="smi_eat",
        smiles_model_name=args.smiles_model_name,
        graph_encoder_type=args.graph_encoder_type,
        graph_input=args.graph_input,
        use_feature_cache=not args.disable_feature_cache,
        feature_source_dataset=args.feature_source_dataset,
        rebuild_feature_cache=args.rebuild_feature_cache,
        max_smiles_length=args.max_smiles_length,
        max_smiles_length_cap=args.max_smiles_length_cap,
        fp_mode=args.fp_mode,
        feature_cache_workers=args.feature_cache_workers,
        feature_cache_chunksize=args.feature_cache_chunksize,
        feature_cache_partial_every=args.feature_cache_partial_every,
        feature_cache_item_timeout=args.feature_cache_item_timeout,
        cache_layers=args.cache_layers,
        cache_validate=args.cache_validate,
        cache_commit_size=args.cache_commit_size,
        embed_tries_multiplier=args.embed_tries_multiplier,
        conformer_3d_count=args.conformer_3d_count,
        conformer_keep_count=args.conformer_keep_count,
        conformer_profile=args.conformer_profile,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        mips_core=args.mips_core,
        mips_max_hops=args.mips_max_hops,
        mips_use_descriptors=args.mips_use_descriptors,
        mips_descriptor_protocol=args.mips_descriptor_protocol,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        topology_representation=args.topology_representation,
        mcl_distance_percentiles=args.mcl_distance_percentiles,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        finite_variant=args.finite_variant,
        conformer_mode=args.conformer_mode,
        field_layout=args.field_layout,
        field_channels=args.field_channels,
        experiment_id=args.experiment_id,
        feature_config_hash="manual",
        modalities=args.modalities,
        star_rbf_v2_sidecar=args.star_rbf_v2_sidecar,
    )
    stats = {scale: {
        "old_visible": 0, "new_visible": 0,
        "symmetry_added": 0, "old_asymmetric_pairs": 0,
        "new_asymmetric_pairs": 0, "total_pairs": 0,
    } for scale in ("q20", "q50")}
    eligible = 0
    affected = {scale: 0 for scale in ("q20", "q50")}
    for index in range(len(dataset)):
        item = dataset[index]
        if not (
            bool(getattr(item, "graph_available", False))
            and bool(getattr(item, "trimer_geometry_valid", False))
            and bool(getattr(item, "trimer_geometry_is_3d", False))
            and not bool(getattr(item, "trimer_2d_fallback", False))
        ):
            continue
        positions = torch.as_tensor(item.trimer_pos).float()
        central_mask = torch.as_tensor(
            item.trimer_central_ru_mask
        ).bool().reshape(-1)
        central_count = int(central_mask.sum().item())
        if central_count < 1 or positions.size(0) != 3 * central_count:
            continue
        eligible += 1
        distances = torch.cdist(
            positions[central_mask], positions
        )  # [C, 3C]
        thresholds = getattr(item, "trimer_mcl_thresholds", None)
        if thresholds is None or not bool(
            torch.isfinite(torch.as_tensor(thresholds)).all()
        ):
            pair_distances = torch.pdist(positions)
            thresholds = torch.quantile(
                pair_distances, pair_distances.new_tensor((0.20, 0.50))
            )
        thresholds = torch.as_tensor(thresholds).float().reshape(2)
        for scale_index, scale in enumerate(("q20", "q50")):
            q = float(thresholds[scale_index].item())
            old = distances <= q
            new = build_sym_mcl_mask(
                distances.unsqueeze(0), q, torch.tensor([central_count]), None
            )[0]
            old_plus = old[:, 2 * central_count : 3 * central_count]
            old_minus = old[:, 0:central_count]
            new_plus = new[:, 2 * central_count : 3 * central_count]
            new_minus = new[:, 0:central_count]
            added = (new & ~old).sum().item()
            stats[scale]["old_visible"] += int(old.sum().item())
            stats[scale]["new_visible"] += int(new.sum().item())
            stats[scale]["symmetry_added"] += int(added)
            stats[scale]["old_asymmetric_pairs"] += int(
                (old_plus != old_minus.transpose(-1, -2)).sum().item()
            )
            stats[scale]["new_asymmetric_pairs"] += int(
                (new_plus != new_minus.transpose(-1, -2)).sum().item()
            )
            stats[scale]["total_pairs"] += int(new.numel())
            if added:
                affected[scale] += 1

    report = {
        "cohort": "smi_eat",
        "eligible_graphs": eligible,
        "dataset_size": len(dataset),
    }
    for scale in ("q20", "q50"):
        entry = stats[scale]
        report[scale] = {
            "old_visible": entry["old_visible"],
            "new_visible": entry["new_visible"],
            "total_pairs": entry["total_pairs"],
            "old_visible_ratio": round(entry["old_visible"] / entry["total_pairs"], 6),
            "new_visible_ratio": round(entry["new_visible"] / entry["total_pairs"], 6),
            "symmetry_added_pairs": entry["symmetry_added"],
            "symmetry_added_relative": round(
                entry["symmetry_added"] / max(1, entry["old_visible"]), 6
            ),
            "old_asymmetric_pairs": entry["old_asymmetric_pairs"],
            "new_asymmetric_pairs": entry["new_asymmetric_pairs"],
            "affected_graphs": affected[scale],
            "affected_graph_fraction": round(affected[scale] / max(1, eligible), 6),
        }
    output = Path(
        "results/mts_b0_periodic_coordinate_denoising_v2/sym_mcl_v1_screening/"
        "mechanism_stats.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
