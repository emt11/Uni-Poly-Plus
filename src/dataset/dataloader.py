import torch
from torch_geometric.data import Batch
from .mips_trimer_contract import (
    FEATURE_SCHEMA,
    TOPOLOGY_CANONICAL,
)


def mips_trimer_collate(data_list):
    """Small collator for the production graph-only MIPS route.

    The historical collator deliberately supports legacy geometry and dense
    multimodal batches and therefore allocates many unrelated tensors (SMILES tokens, fingerprints,
    PBC metadata and finite-geometry placeholders).  MTS never
    consumes those fields.  Keeping this path separate makes Stage 1/1.5 and
    Stage 3 read only the sparse O8, Trimer and MD200 payloads and avoids a
    second per-batch copy of legacy geometry data.
    """
    if not data_list:
        raise ValueError("cannot collate an empty MIPS batch")

    canonical_periodic_batch = all(
        bool(getattr(item, "mts_canonical_periodic", False))
        or int(getattr(item, "mips_local_lga_schema_version", 0)) == 2
        for item in data_list
    )
    representations = {
        str(getattr(
            item,
            "topology_representation",
            TOPOLOGY_CANONICAL,
        ))
        for item in data_list
    }
    if len(representations) != 1:
        raise ValueError("cannot mix MTS topology representations in one batch")
    topology_representation = next(iter(representations))
    if topology_representation != TOPOLOGY_CANONICAL:
        raise ValueError("MTS-GLT-v2 only supports canonical_lifted topology")
    if any(
        bool(getattr(item, "mts_canonical_periodic", False))
        or int(getattr(item, "mips_local_lga_schema_version", 0)) == 2
        for item in data_list
    ) and not canonical_periodic_batch:
        raise ValueError(
            "MTS-GLT-v2 batches must contain canonical-periodic topology records"
        )

    batch = Batch()
    x_parts, edge_parts, edge_attr_parts, graph_parts = [], [], [], []
    lga_edges, lga_spd, lga_paths, lga_masks = [], [], [], []
    lga_path_shifts = []
    lga_hist, lga_star = [], []
    canonical_parts, copy_parts, pair_parts = [], [], []
    canonical_to_trimer_parts = []
    canonical_id_parts, relation_shift_parts, polymer_link_parts = [], [], []
    canonical_graph_parts, canonical_local_parts, canonical_first_parts = [], [], []
    sample_hash64 = []
    mips_x_parts, backbone_parts = [], []
    smiles, ys = [], []
    graph_available, boundary, condition = [], [], []
    md_parts, md_valid = [], []
    star_v2_relation_rows, star_v2_relation_pairs = [], []
    star_v2_pair_key_src, star_v2_pair_key_dst, star_v2_pair_key_shift = [], [], []
    star_v2_pair_distances, star_v2_pair_counts = [], []
    star_v2_pair_valid, star_v2_pair_sources = [], []
    star_v2_uppers = set()
    glt_fields_present = all(hasattr(item, "glt_token_atom_a") for item in data_list)
    glt3_fields_present = all(hasattr(item, "glt3_token_atom_a") for item in data_list)
    glt_token_atom_a, glt_token_atom_b, glt_token_shift = [], [], []
    glt_token_z_a, glt_token_z_b, glt_token_bond_type, glt_token_label = [], [], [], []
    glt_token_distances, glt_token_counts, glt_token_valid = [], [], []
    glt_token_batch = []
    glt_relation_source, glt_relation_target, glt_relation_center = [], [], []
    glt_relation_multiplicity, glt_relation_angles = [], []
    glt_relation_counts, glt_relation_valid, glt_relation_fallback = [], [], []
    glt_geometry_valid = []
    glt_query_valid = []
    glt_token_observation_valid, glt_token_observation_translation = [], []
    glt_token_observation_q_a, glt_token_observation_q_b = [], []
    glt_relation_outer_offset_a, glt_relation_outer_offset_b = [], []
    glt_relation_span = []
    glt_relation_observation_valid, glt_relation_observation_translation = [], []
    glt_relation_source_distances, glt_relation_source_distance_valid = [], []
    glt_relation_source_distance_slot, glt_relation_source_cross_ru = [], []
    glt_torsion_values, glt_torsion_relations = [], []
    glt_torsion_counts, glt_torsion_covered = [], []
    glt_torsion_source_cross, glt_graph_torsion_covered = [], []
    glt3_token_atom_a, glt3_token_atom_b, glt3_token_shift = [], [], []
    glt3_token_z_a, glt3_token_z_b, glt3_token_distance = [], [], []
    glt3_token_bond_type, glt3_token_stereo, glt3_token_conjugated = [], [], []
    glt3_token_anchor_q_a, glt3_token_anchor_q_b, glt3_token_valid = [], [], []
    glt3_token_center_internal = []
    glt3_token_batch, glt3_geometry_valid = [], []
    glt3_relation_source, glt3_relation_target, glt3_relation_center = [], [], []
    glt3_relation_shift, glt3_relation_angle, glt3_relation_valid = [], [], []
    spatial_fields_present = all(
        hasattr(item, "spatial_pair_index") for item in data_list
    )
    spatial_pair_index, spatial_pair_shift = [], []
    spatial_obs_distances, spatial_obs_mask, spatial_obs_count = [], [], []
    spatial_shell_id, spatial_periodic_self, spatial_pair_valid = [], [], []
    spatial_pair_batch, spatial_graph_valid = [], []
    star_v2_pair_keys_present = all(
        all(hasattr(item, name) for name in (
            "mts_star_v2_pair_key_src", "mts_star_v2_pair_key_dst",
            "mts_star_v2_pair_key_shift",
        )) for item in data_list
    )
    smiles_ids, smiles_masks, fp_parts = [], [], []
    smiles_available, fp_available = [], []
    smiles_fields_present = all(
        hasattr(item, "input_ids_smiles") for item in data_list
    )
    fp_fields_present = all(hasattr(item, "fp") for item in data_list)
    md_fields_present = all(hasattr(item, "mips_md") for item in data_list)
    multitask_fields_present = all(
        hasattr(item, "mts_task_index") for item in data_list
    )
    trimer_fields_present = all(hasattr(item, "trimer_pos") for item in data_list)
    star_enabled = True
    trimer_pos, trimer_z, trimer_edges, trimer_bonds = [], [], [], []
    trimer_base, trimer_offset, trimer_central = [], [], []
    trimer_central_index, trimer_mapping, trimer_batch = [], [], []
    trimer_valid, trimer_is_3d, trimer_2d = [], [], []
    star_distance, star_asymmetry, star_valid = [], [], []
    node_offset = 0
    canonical_offset = 0
    pair_offset = 0
    trimer_offset_global = 0
    trimer_ptr = [0]
    lga_relation_offset = 0
    star_v2_pair_offset = 0
    glt_token_offset = 0
    glt_relation_offset = 0

    for graph_id, item in enumerate(data_list):
        # A legacy single-node canonical placeholder carries a one-column path
        # chain (lga_path_index/lga_path_mask/lga_path_shift of width 1) and no
        # bond histogram.  Normalize it to the canonical 3-column chain width
        # with inert -1/False/zero padding so the batch stays collatable.  Such
        # graphs never enter MCL (their Trimer geometry is invalid) or the
        # masked objective, so the padding is numerically inert.
        if (
            hasattr(item, "lga_path_index")
            and int(item.lga_path_index.size(1)) < 3
        ):
            chain = item.lga_path_index
            item.lga_path_index = torch.nn.functional.pad(
                chain, (0, 3 - int(chain.size(1))), value=-1
            )
            for name in ("lga_path_mask", "lga_path_shift"):
                if hasattr(item, name):
                    field = getattr(item, name)
                    setattr(item, name, torch.nn.functional.pad(
                        field, (0, 3 - int(field.size(1))),
                        value=(False if name == "lga_path_mask" else 0),
                    ))
            if not hasattr(item, "lga_path_bond_hist"):
                item.lga_path_bond_hist = torch.zeros(
                    (int(item.lga_path_index.size(0)), 2, 6),
                    dtype=torch.float32,
                )
        num_nodes = int(item.x.size(0))
        x_parts.append(item.x)
        graph_parts.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        edge_parts.append(item.edge_index.long() + node_offset)
        if getattr(item, "edge_attr", None) is not None:
            edge_attr_parts.append(item.edge_attr)
        node_offset += num_nodes

        if not hasattr(item, "lga_edge_index"):
            raise ValueError("MIPS batch item is missing lga_edge_index")
        lga_edges.append(item.lga_edge_index.long() + (node_offset - num_nodes))
        lga_spd.append(item.lga_spd.long())
        local_path = item.lga_path_index.long().clone()
        local_path[local_path >= 0] += node_offset - num_nodes
        lga_paths.append(local_path)
        if canonical_periodic_batch:
            lga_path_shifts.append(
                torch.as_tensor(
                    getattr(
                        item, "lga_path_shift",
                        torch.zeros_like(item.lga_path_index),
                    )
                ).long()
            )
        lga_masks.append(item.lga_path_mask.bool())
        lga_hist.append(item.lga_path_bond_hist.float())
        lga_star.append(item.lga_star_edge_mask.bool())

        if hasattr(item, "mts_star_v2_relation_row"):
            rows = item.mts_star_v2_relation_row.long()
            pairs = item.mts_star_v2_relation_pair_index.long()
            if rows.numel() != pairs.numel():
                raise ValueError("Star-RBF v2 relation/pair length mismatch")
            if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(item.lga_spd.numel())):
                raise ValueError("Star-RBF v2 relation row out of bounds")
            if not torch.equal(item.mts_star_v2_relation_spd.long(), item.lga_spd[rows].long()):
                raise ValueError("Star-RBF v2 relation SPD mismatch")
            pair_count = int(item.mts_star_v2_pair_valid.numel())
            if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= pair_count):
                raise ValueError("Star-RBF v2 pair index out of bounds")
            star_v2_relation_rows.append(rows + lga_relation_offset)
            star_v2_relation_pairs.append(pairs + star_v2_pair_offset)
            if star_v2_pair_keys_present:
                if any(int(getattr(item, name).numel()) != pair_count for name in (
                    "mts_star_v2_pair_key_src", "mts_star_v2_pair_key_dst",
                    "mts_star_v2_pair_key_shift",
                )):
                    raise ValueError("Star-RBF v2 pair key length mismatch")
                canonical_base = canonical_offset
                star_v2_pair_key_src.append(
                    item.mts_star_v2_pair_key_src.long() + canonical_base
                )
                star_v2_pair_key_dst.append(
                    item.mts_star_v2_pair_key_dst.long() + canonical_base
                )
                star_v2_pair_key_shift.append(item.mts_star_v2_pair_key_shift.long())
            star_v2_pair_distances.append(item.mts_star_v2_pair_observation_distances.float())
            star_v2_pair_counts.append(item.mts_star_v2_pair_observation_count.long())
            star_v2_pair_valid.append(item.mts_star_v2_pair_valid.bool())
            star_v2_pair_sources.append(item.mts_star_v2_pair_geometry_source.long())
            star_v2_uppers.add(float(item.mts_star_v2_rbf_upper))
            star_v2_pair_offset += pair_count

        if glt_fields_present:
            token_count = int(item.glt_token_atom_a.numel())
            relation_count = int(item.glt_relation_source.numel())
            for name in (
                "glt_token_atom_b", "glt_token_shift", "glt_token_endpoint_z_a",
                "glt_token_endpoint_z_b", "glt_token_bond_type", "glt_token_label",
                "glt_token_observation_count", "glt_token_valid",
            ):
                if int(getattr(item, name).numel()) != token_count:
                    raise ValueError(f"periodic line GLT token length mismatch: {name}")
            if tuple(item.glt_token_observation_distances.shape) != (token_count, 3):
                raise ValueError("periodic line GLT distance shape mismatch")
            for name in (
                "glt_relation_target", "glt_relation_center_atom",
                "glt_relation_multiplicity", "glt_relation_observation_count",
                "glt_relation_valid", "glt_relation_is_fallback",
            ):
                if int(getattr(item, name).numel()) != relation_count:
                    raise ValueError(f"periodic line GLT relation length mismatch: {name}")
            if tuple(item.glt_relation_observation_angles.shape) != (relation_count, 3):
                raise ValueError("periodic line GLT angle shape mismatch")
            source = item.glt_relation_source.long()
            target = item.glt_relation_target.long()
            if relation_count and (
                int(source.min()) < 0 or int(source.max()) >= token_count
                or int(target.min()) < 0 or int(target.max()) >= token_count
            ):
                raise ValueError("periodic line GLT relation endpoint out of bounds")
            glt_token_atom_a.append(item.glt_token_atom_a.long() + canonical_offset)
            glt_token_atom_b.append(item.glt_token_atom_b.long() + canonical_offset)
            glt_token_shift.append(item.glt_token_shift.long())
            glt_token_z_a.append(item.glt_token_endpoint_z_a.long())
            glt_token_z_b.append(item.glt_token_endpoint_z_b.long())
            glt_token_bond_type.append(item.glt_token_bond_type.long())
            glt_token_label.append(item.glt_token_label.long())
            glt_token_distances.append(item.glt_token_observation_distances.float())
            glt_token_counts.append(item.glt_token_observation_count.long())
            glt_token_valid.append(item.glt_token_valid.bool())
            glt_token_batch.append(torch.full((token_count,), graph_id, dtype=torch.long))
            glt_relation_source.append(source + glt_token_offset)
            glt_relation_target.append(target + glt_token_offset)
            center = item.glt_relation_center_atom.long().clone()
            center[center >= 0] += canonical_offset
            glt_relation_center.append(center)
            glt_relation_multiplicity.append(item.glt_relation_multiplicity.long())
            glt_relation_angles.append(item.glt_relation_observation_angles.float())
            glt_relation_counts.append(item.glt_relation_observation_count.long())
            glt_relation_valid.append(item.glt_relation_valid.bool())
            glt_relation_fallback.append(item.glt_relation_is_fallback.bool())
            glt_geometry_valid.append(bool(item.glt_geometry_valid))
            glt_query_valid.append(bool(getattr(item, "glt_query_valid", item.glt_geometry_valid)))
            if hasattr(item, "glt_token_observation_valid"):
                glt_token_observation_valid.append(item.glt_token_observation_valid.bool())
                glt_token_observation_translation.append(item.glt_token_observation_translation.long())
                glt_token_observation_q_a.append(item.glt_token_observation_q_a.long())
                glt_token_observation_q_b.append(item.glt_token_observation_q_b.long())
                glt_relation_outer_offset_a.append(item.glt_relation_outer_offset_a.long())
                glt_relation_outer_offset_b.append(item.glt_relation_outer_offset_b.long())
                glt_relation_span.append(item.glt_relation_span.long())
                glt_relation_observation_valid.append(item.glt_relation_observation_valid.bool())
                glt_relation_observation_translation.append(item.glt_relation_observation_translation.long())
            if hasattr(item, "glt_relation_torsion_count"):
                glt_torsion_values.append(item.glt_torsion_observation_value.float())
                glt_torsion_relations.append(
                    item.glt_torsion_observation_relation.long()
                    + glt_relation_offset
                )
                glt_torsion_counts.append(item.glt_relation_torsion_count.long())
                glt_torsion_covered.append(item.glt_relation_torsion_covered.bool())
                glt_torsion_source_cross.append(
                    item.glt_relation_torsion_source_cross_ru.bool()
                )
                glt_graph_torsion_covered.append(
                    bool(item.glt_graph_torsion_covered)
                )
            if hasattr(item, "glt_relation_source_distances"):
                if tuple(item.glt_relation_source_distances.shape) != (relation_count, 3):
                    raise ValueError("joint radial relation-distance shape mismatch")
                if tuple(item.glt_relation_source_distance_valid.shape) != (relation_count, 3):
                    raise ValueError("joint radial relation-valid shape mismatch")
                if tuple(item.glt_relation_source_distance_slot.shape) != (relation_count, 3):
                    raise ValueError("joint radial relation-slot shape mismatch")
                if int(item.glt_relation_source_cross_ru.numel()) != relation_count:
                    raise ValueError("joint radial source-shift length mismatch")
                glt_relation_source_distances.append(
                    item.glt_relation_source_distances.float()
                )
                glt_relation_source_distance_valid.append(
                    item.glt_relation_source_distance_valid.bool()
                )
                glt_relation_source_distance_slot.append(
                    item.glt_relation_source_distance_slot.long()
                )
                glt_relation_source_cross_ru.append(
                    item.glt_relation_source_cross_ru.bool()
                )
            glt_token_offset += token_count
            glt_relation_offset += relation_count

        if glt3_fields_present:
            token_count = int(item.glt3_token_atom_a.numel())
            relation_count = int(item.glt3_relation_source.numel())
            for name in (
                "glt3_token_atom_b", "glt3_token_shift",
                "glt3_token_endpoint_z_a", "glt3_token_endpoint_z_b",
                "glt3_token_distance", "glt3_token_bond_type",
                "glt3_token_stereo", "glt3_token_conjugated",
                "glt3_token_anchor_q_a", "glt3_token_anchor_q_b",
                "glt3_token_valid",
            ):
                if int(getattr(item, name).numel()) != token_count:
                    raise ValueError(f"image GLT token length mismatch: {name}")
            for name in (
                "glt3_relation_target", "glt3_relation_center_atom",
                "glt3_relation_source_image_shift", "glt3_relation_angle",
                "glt3_relation_valid",
            ):
                if int(getattr(item, name).numel()) != relation_count:
                    raise ValueError(f"image GLT relation length mismatch: {name}")
            source = item.glt3_relation_source.long()
            target = item.glt3_relation_target.long()
            if relation_count and (
                int(source.min()) < 0 or int(source.max()) >= token_count
                or int(target.min()) < 0 or int(target.max()) >= token_count
            ):
                raise ValueError("image GLT relation endpoint out of bounds")
            glt3_token_atom_a.append(item.glt3_token_atom_a.long() + canonical_offset)
            glt3_token_atom_b.append(item.glt3_token_atom_b.long() + canonical_offset)
            glt3_token_shift.append(item.glt3_token_shift.long())
            glt3_token_z_a.append(item.glt3_token_endpoint_z_a.long())
            glt3_token_z_b.append(item.glt3_token_endpoint_z_b.long())
            glt3_token_distance.append(item.glt3_token_distance.float())
            glt3_token_bond_type.append(item.glt3_token_bond_type.long())
            glt3_token_stereo.append(item.glt3_token_stereo.long())
            glt3_token_conjugated.append(item.glt3_token_conjugated.long())
            glt3_token_anchor_q_a.append(item.glt3_token_anchor_q_a.long())
            glt3_token_anchor_q_b.append(item.glt3_token_anchor_q_b.long())
            glt3_token_valid.append(item.glt3_token_valid.bool())
            glt3_token_center_internal.append(
                getattr(item, "glt3_token_center_internal", item.glt3_token_shift == 0).bool()
            )
            glt3_token_batch.append(torch.full((token_count,), graph_id, dtype=torch.long))
            glt3_relation_source.append(source + glt_token_offset)
            glt3_relation_target.append(target + glt_token_offset)
            center = item.glt3_relation_center_atom.long().clone()
            center[center >= 0] += canonical_offset
            glt3_relation_center.append(center)
            glt3_relation_shift.append(item.glt3_relation_source_image_shift.long())
            glt3_relation_angle.append(item.glt3_relation_angle.float())
            glt3_relation_valid.append(item.glt3_relation_valid.bool())
            glt3_geometry_valid.append(bool(item.glt3_geometry_valid))
            glt_token_offset += token_count

        if spatial_fields_present:
            spatial_index = item.spatial_pair_index.long()
            if spatial_index.ndim != 2 or int(spatial_index.size(0)) != 2:
                raise ValueError("spatial_pair_index must have shape [2,E]")
            spatial_count = int(spatial_index.size(1))
            if spatial_count and (
                int(spatial_index.min()) < 0
                or int(spatial_index.max()) >= num_nodes
            ):
                raise ValueError("spatial contact endpoint out of canonical bounds")
            for name in (
                "spatial_pair_shift", "spatial_obs_count", "spatial_shell_id",
                "spatial_periodic_self", "spatial_pair_valid",
            ):
                if int(getattr(item, name).numel()) != spatial_count:
                    raise ValueError(f"spatial contact length mismatch: {name}")
            if tuple(item.spatial_obs_distances.shape) != (spatial_count, 3):
                raise ValueError("spatial observation distance shape mismatch")
            if tuple(item.spatial_obs_mask.shape) != (spatial_count, 3):
                raise ValueError("spatial observation mask shape mismatch")
            spatial_pair_index.append(spatial_index + canonical_offset)
            spatial_pair_shift.append(item.spatial_pair_shift.long())
            spatial_obs_distances.append(item.spatial_obs_distances.float())
            spatial_obs_mask.append(item.spatial_obs_mask.bool())
            spatial_obs_count.append(item.spatial_obs_count.long())
            spatial_shell_id.append(item.spatial_shell_id.long())
            spatial_periodic_self.append(item.spatial_periodic_self.bool())
            spatial_pair_valid.append(item.spatial_pair_valid.bool())
            spatial_pair_batch.append(
                torch.full((spatial_count,), graph_id, dtype=torch.long)
            )
            spatial_graph_valid.append(bool(item.spatial_graph_valid))

        lga_relation_offset += int(item.lga_edge_index.size(1))

        if canonical_periodic_batch:
            local_canonical = torch.arange(num_nodes, dtype=torch.long)
            canonical_to_trimer_parts.append(torch.as_tensor(
                getattr(
                    item, "canonical_to_trimer_base_atom_id", local_canonical
                )
            ).long())
        else:
            local_canonical = item.canonical_ru_atom_index.long()
        canonical_parts.append(local_canonical + canonical_offset)
        if canonical_periodic_batch:
            canonical_id_parts.append(local_canonical + canonical_offset)
            relation_shift_parts.append(
                torch.as_tensor(item.lga_source_image_shift).long()
            )
            polymer_link_parts.append(
                torch.as_tensor(
                    getattr(item, "polymer_link_mask", item.lga_star_edge_mask)
                ).bool()
            )
            local_pairs = torch.empty(0, dtype=torch.long)
        else:
            copy_parts.append(item.ru_copy_index.long())
            local_pairs = item.canonical_pair_index.long()
            pair_parts.append(local_pairs + pair_offset)
        canonical_count = int(local_canonical.max().item()) + 1 if local_canonical.numel() else 0
        pair_count = int(local_pairs.max().item()) + 1 if local_pairs.numel() else 0
        canonical_offset += canonical_count
        pair_offset += pair_count

        mips_x_parts.append(item.mips_x.float())
        backbone_parts.append(item.mips_backbone_mask.long())
        smiles.append(str(getattr(item, "smiles", "")))
        sample_hash64.append(int(torch.as_tensor(
            getattr(item, "mts_sample_hash64", 0)
        ).item()))
        canonical_graph_parts.append(torch.full(
            (canonical_count,), graph_id, dtype=torch.long
        ))
        canonical_local_parts.append(torch.arange(canonical_count, dtype=torch.long))
        if canonical_count:
            first = torch.full(
                (canonical_count,), num_nodes, dtype=torch.long
            )
            first.scatter_reduce_(
                0,
                local_canonical,
                torch.arange(num_nodes, dtype=torch.long),
                reduce="amin",
                include_self=True,
            )
            canonical_first_parts.append(first + node_offset - num_nodes)
        else:
            canonical_first_parts.append(torch.empty(0, dtype=torch.long))
        item_y = getattr(item, "y", None)
        if item_y is None:
            item_y = torch.zeros(1, dtype=torch.float)
        ys.append(torch.as_tensor(item_y).reshape(-1))
        if smiles_fields_present:
            smiles_ids.append(item.input_ids_smiles.long())
            smiles_masks.append(item.attention_mask_smiles.long())
            smiles_available.append(bool(getattr(item, "smiles_available", True)))
        if fp_fields_present:
            fp_parts.append(item.fp.float().reshape(-1))
            fp_available.append(bool(getattr(item, "fp_available", True)))
        graph_available.append(bool(getattr(item, "graph_available", True)))
        boundary.append(int(getattr(item, "mips_boundary_distance", -1)))
        condition.append(bool(getattr(item, "mips_condition_valid", True)))
        if md_fields_present:
            md_parts.append(item.mips_md.float().reshape(-1))
            md_valid.append(bool(getattr(item, "mips_md_valid", False)))

        if trimer_fields_present:
            n_trimer = int(item.trimer_pos.size(0))
            trimer_pos.append(item.trimer_pos.float())
            trimer_z.append(item.trimer_atomic_number.long())
            trimer_edges.append(item.trimer_edge_index.long() + trimer_offset_global)
            trimer_bonds.append(item.trimer_bond_type.long())
            base_name = "trimer_base_ru_atom_id" if hasattr(item, "trimer_base_ru_atom_id") else "trimer_base_ru_atom_index"
            trimer_base.append(getattr(item, base_name).long() + canonical_offset - canonical_count)
            trimer_offset.append(item.trimer_ru_offset.long())
            trimer_central.append(item.trimer_central_ru_mask.bool())
            trimer_central_index.append(item.trimer_central_atom_index.long() + trimer_offset_global)
            mapping = item.mips_to_trimer_central_index.long().clone()
            mapping[mapping >= 0] += trimer_offset_global
            trimer_mapping.append(mapping)
            trimer_batch.append(torch.full((n_trimer,), graph_id, dtype=torch.long))
            trimer_valid.append(bool(getattr(item, "trimer_geometry_valid", False)))
            trimer_is_3d.append(bool(getattr(item, "trimer_geometry_is_3d", False)))
            trimer_2d.append(bool(getattr(item, "trimer_2d_fallback", False)))
            star_distance.append(float(getattr(item, "star_3d_distance", 0.0)))
            star_asymmetry.append(float(getattr(item, "star_3d_asymmetry", 0.0)))
            star_valid.append(bool(getattr(item, "star_3d_valid", False)))
            trimer_offset_global += n_trimer
            trimer_ptr.append(trimer_offset_global)

    batch.x = torch.cat(x_parts, dim=0)
    batch.edge_index = torch.cat(edge_parts, dim=1)
    if edge_attr_parts:
        batch.edge_attr = torch.cat(edge_attr_parts, dim=0)
    batch.batch = torch.cat(graph_parts, dim=0)
    batch.lga_edge_index = torch.cat(lga_edges, dim=1)
    if canonical_periodic_batch:
        batch.canonical_lga_edge_index = batch.lga_edge_index.clone()
    batch.lga_spd = torch.cat(lga_spd, dim=0)
    batch.lga_path_index = torch.cat(lga_paths, dim=0)
    batch.lga_path_mask = torch.cat(lga_masks, dim=0)
    if canonical_periodic_batch:
        batch.lga_path_shift = torch.cat(lga_path_shifts, dim=0)
        batch.lga_path_shifts = batch.lga_path_shift
        batch.lifted_single_path_index = batch.lga_path_index
        batch.lifted_single_path_shift = batch.lga_path_shift
        batch.lifted_single_path_mask = batch.lga_path_mask
    batch.lga_path_bond_hist = torch.cat(lga_hist, dim=0)
    batch.lga_star_edge_mask = torch.cat(lga_star, dim=0)
    if star_v2_relation_rows:
        if len(star_v2_uppers) != 1:
            raise ValueError("Star-RBF v2 batch RBF upper mismatch")
        batch.mts_star_v2_relation_row = torch.cat(star_v2_relation_rows)
        batch.mts_star_v2_relation_pair_index = torch.cat(star_v2_relation_pairs)
        if star_v2_pair_keys_present:
            batch.mts_star_v2_pair_key_src = torch.cat(star_v2_pair_key_src)
            batch.mts_star_v2_pair_key_dst = torch.cat(star_v2_pair_key_dst)
            batch.mts_star_v2_pair_key_shift = torch.cat(star_v2_pair_key_shift)
        batch.mts_star_v2_pair_observation_distances = torch.cat(star_v2_pair_distances)
        batch.mts_star_v2_pair_observation_count = torch.cat(star_v2_pair_counts)
        batch.mts_star_v2_pair_valid = torch.cat(star_v2_pair_valid)
        batch.mts_star_v2_pair_geometry_source = torch.cat(star_v2_pair_sources)
        batch.mts_star_v2_rbf_upper = next(iter(star_v2_uppers))
    if glt_fields_present:
        batch.glt_token_atom_a = torch.cat(glt_token_atom_a)
        batch.glt_token_atom_b = torch.cat(glt_token_atom_b)
        batch.glt_token_shift = torch.cat(glt_token_shift)
        batch.glt_token_endpoint_z_a = torch.cat(glt_token_z_a)
        batch.glt_token_endpoint_z_b = torch.cat(glt_token_z_b)
        batch.glt_token_bond_type = torch.cat(glt_token_bond_type)
        batch.glt_token_label = torch.cat(glt_token_label)
        batch.glt_token_observation_distances = torch.cat(glt_token_distances)
        batch.glt_token_observation_count = torch.cat(glt_token_counts)
        batch.glt_token_valid = torch.cat(glt_token_valid)
        batch.glt_token_batch = torch.cat(glt_token_batch)
        batch.glt_relation_source = torch.cat(glt_relation_source)
        batch.glt_relation_target = torch.cat(glt_relation_target)
        batch.glt_relation_center_atom = torch.cat(glt_relation_center)
        batch.glt_relation_multiplicity = torch.cat(glt_relation_multiplicity)
        batch.glt_relation_observation_angles = torch.cat(glt_relation_angles)
        batch.glt_relation_observation_count = torch.cat(glt_relation_counts)
        batch.glt_relation_valid = torch.cat(glt_relation_valid)
        batch.glt_relation_is_fallback = torch.cat(glt_relation_fallback)
        batch.glt_geometry_valid = torch.tensor(glt_geometry_valid, dtype=torch.bool)
        batch.glt_query_valid = torch.tensor(glt_query_valid, dtype=torch.bool)
        if glt_token_observation_valid:
            batch.glt_token_observation_valid = torch.cat(glt_token_observation_valid)
            batch.glt_token_observation_translation = torch.cat(glt_token_observation_translation)
            batch.glt_token_observation_q_a = torch.cat(glt_token_observation_q_a)
            batch.glt_token_observation_q_b = torch.cat(glt_token_observation_q_b)
            batch.glt_relation_outer_offset_a = torch.cat(glt_relation_outer_offset_a)
            batch.glt_relation_outer_offset_b = torch.cat(glt_relation_outer_offset_b)
            batch.glt_relation_span = torch.cat(glt_relation_span)
            batch.glt_relation_observation_valid = torch.cat(glt_relation_observation_valid)
            batch.glt_relation_observation_translation = torch.cat(glt_relation_observation_translation)
        if glt_torsion_counts:
            batch.glt_torsion_observation_value = torch.cat(glt_torsion_values)
            batch.glt_torsion_observation_relation = torch.cat(glt_torsion_relations)
            batch.glt_relation_torsion_count = torch.cat(glt_torsion_counts)
            batch.glt_relation_torsion_covered = torch.cat(glt_torsion_covered)
            batch.glt_relation_torsion_source_cross_ru = torch.cat(glt_torsion_source_cross)
            batch.glt_graph_torsion_covered = torch.tensor(
                glt_graph_torsion_covered, dtype=torch.bool
            )
        if glt_relation_source_distances:
            if len(glt_relation_source_distances) != len(data_list):
                raise ValueError("joint radial fields must be present for the full batch")
            batch.glt_relation_source_distances = torch.cat(
                glt_relation_source_distances
            )
            batch.glt_relation_source_distance_valid = torch.cat(
                glt_relation_source_distance_valid
            )
            batch.glt_relation_source_distance_slot = torch.cat(
                glt_relation_source_distance_slot
            )
            batch.glt_relation_source_cross_ru = torch.cat(
                glt_relation_source_cross_ru
            )
    if glt3_fields_present:
        batch.glt3_token_atom_a = torch.cat(glt3_token_atom_a)
        batch.glt3_token_atom_b = torch.cat(glt3_token_atom_b)
        batch.glt3_token_shift = torch.cat(glt3_token_shift)
        batch.glt3_token_endpoint_z_a = torch.cat(glt3_token_z_a)
        batch.glt3_token_endpoint_z_b = torch.cat(glt3_token_z_b)
        batch.glt3_token_distance = torch.cat(glt3_token_distance)
        batch.glt3_token_bond_type = torch.cat(glt3_token_bond_type)
        batch.glt3_token_stereo = torch.cat(glt3_token_stereo)
        batch.glt3_token_conjugated = torch.cat(glt3_token_conjugated)
        batch.glt3_token_anchor_q_a = torch.cat(glt3_token_anchor_q_a)
        batch.glt3_token_anchor_q_b = torch.cat(glt3_token_anchor_q_b)
        batch.glt3_token_valid = torch.cat(glt3_token_valid)
        batch.glt3_token_center_internal = torch.cat(glt3_token_center_internal)
        batch.glt3_token_batch = torch.cat(glt3_token_batch)
        batch.glt3_relation_source = torch.cat(glt3_relation_source)
        batch.glt3_relation_target = torch.cat(glt3_relation_target)
        batch.glt3_relation_center_atom = torch.cat(glt3_relation_center)
        batch.glt3_relation_source_image_shift = torch.cat(glt3_relation_shift)
        batch.glt3_relation_angle = torch.cat(glt3_relation_angle)
        batch.glt3_relation_valid = torch.cat(glt3_relation_valid)
        batch.glt3_geometry_valid = torch.tensor(glt3_geometry_valid, dtype=torch.bool)
    if spatial_fields_present:
        batch.spatial_pair_index = torch.cat(spatial_pair_index, dim=1)
        batch.spatial_pair_shift = torch.cat(spatial_pair_shift)
        batch.spatial_obs_distances = torch.cat(spatial_obs_distances)
        batch.spatial_obs_mask = torch.cat(spatial_obs_mask)
        batch.spatial_obs_count = torch.cat(spatial_obs_count)
        batch.spatial_shell_id = torch.cat(spatial_shell_id)
        batch.spatial_periodic_self = torch.cat(spatial_periodic_self)
        batch.spatial_pair_valid = torch.cat(spatial_pair_valid)
        batch.spatial_pair_batch = torch.cat(spatial_pair_batch)
        batch.spatial_graph_valid = torch.tensor(
            spatial_graph_valid, dtype=torch.bool
        )
    batch.canonical_ru_atom_index = torch.cat(canonical_parts, dim=0)
    if canonical_periodic_batch:
        batch.canonical_atom_id = torch.cat(canonical_id_parts, dim=0)
        batch.canonical_to_trimer_base_atom_id = torch.cat(
            canonical_to_trimer_parts, dim=0
        )
        batch.canonical_to_trimer_base_atom_index = (
            batch.canonical_to_trimer_base_atom_id
        )
    batch.canonical_graph_index = torch.cat(canonical_graph_parts, dim=0)
    batch.canonical_local_index = torch.cat(canonical_local_parts, dim=0)
    batch.canonical_first_node_index = torch.cat(
        canonical_first_parts, dim=0
    )
    batch.mts_sample_hash64 = torch.tensor(sample_hash64, dtype=torch.long)
    if not canonical_periodic_batch:
        batch.ru_copy_index = torch.cat(copy_parts, dim=0)
        batch.canonical_pair_index = torch.cat(pair_parts, dim=0)
    else:
        batch.lga_source_image_shift = torch.cat(
            relation_shift_parts, dim=0
        )
        batch.lga_relation_shift = batch.lga_source_image_shift
        batch.canonical_lga_source_image_shift = batch.lga_source_image_shift
        batch.source_image_shift = batch.lga_source_image_shift
        batch.spd = batch.lga_spd
        batch.polymer_link_mask = torch.cat(polymer_link_parts, dim=0)
        batch.lga_polymer_link_mask = batch.polymer_link_mask
    batch.mips_x = torch.cat(mips_x_parts, dim=0)
    batch.mips_backbone_mask = torch.cat(backbone_parts, dim=0)
    if canonical_periodic_batch:
        atomic_parts = []
        for item in data_list:
            atomic_numbers = getattr(item, "atomic_numbers", None)
            if atomic_numbers is None:
                atomic_numbers = getattr(
                    item, "z", torch.empty(0, dtype=torch.long)
                )
            atomic_parts.append(torch.as_tensor(atomic_numbers).long())
        batch.atomic_numbers = torch.cat(atomic_parts, dim=0)
        batch.canonical_atomic_numbers = batch.atomic_numbers
        batch.atomic_number = batch.atomic_numbers
        batch.z = batch.atomic_numbers
        batch.backbone_mask = batch.mips_backbone_mask
    batch.smiles = smiles
    batch.y = torch.stack(ys, dim=0)
    if smiles_fields_present:
        batch.input_ids_smiles = torch.stack(smiles_ids, dim=0)
        batch.attention_mask_smiles = torch.stack(smiles_masks, dim=0)
        batch.smiles_available = torch.tensor(smiles_available, dtype=torch.bool)
    if fp_fields_present:
        batch.fp = torch.stack(fp_parts, dim=0)
        batch.fp_available = torch.tensor(fp_available, dtype=torch.bool)
    if multitask_fields_present:
        batch.mts_task_index = torch.tensor([
            int(torch.as_tensor(item.mts_task_index).item()) for item in data_list
        ], dtype=torch.long)
    batch.graph_available = torch.tensor(graph_available, dtype=torch.bool)
    batch.mips_boundary_distance = torch.tensor(boundary, dtype=torch.long)
    batch.mips_condition_valid = torch.tensor(condition, dtype=torch.bool)
    if md_fields_present:
        batch.mips_md = torch.stack(md_parts, dim=0)
        batch.mips_md_valid = torch.tensor(md_valid, dtype=torch.bool)
    batch.feature_schema = FEATURE_SCHEMA
    batch.mips_local_lga_schema_version = 2
    batch.mts_canonical_periodic = bool(canonical_periodic_batch)
    batch.topology_representation = topology_representation
    batch.mts_topology_representation = topology_representation

    if trimer_fields_present:
        batch.trimer_pos = torch.cat(trimer_pos, dim=0)
        batch.trimer_atomic_number = torch.cat(trimer_z, dim=0)
        batch.trimer_edge_index = torch.cat(trimer_edges, dim=1)
        batch.trimer_bond_type = torch.cat(trimer_bonds, dim=0)
        batch.trimer_base_ru_atom_index = torch.cat(trimer_base, dim=0)
        batch.trimer_base_ru_atom_id = batch.trimer_base_ru_atom_index
        batch.trimer_ru_offset = torch.cat(trimer_offset, dim=0)
        batch.trimer_central_ru_mask = torch.cat(trimer_central, dim=0)
        batch.trimer_central_atom_index = torch.cat(trimer_central_index, dim=0)
        batch.trimer_central_ru_atom_index = batch.trimer_central_atom_index
        batch.mips_to_trimer_central_index = torch.cat(trimer_mapping, dim=0)
        batch.trimer_batch = torch.cat(trimer_batch, dim=0)
        batch.trimer_ptr = torch.tensor(trimer_ptr, dtype=torch.long)
        batch.trimer_geometry_valid = torch.tensor(trimer_valid, dtype=torch.bool)
        batch.trimer_geometry_is_3d = torch.tensor(trimer_is_3d, dtype=torch.bool)
        batch.trimer_2d_fallback = torch.tensor(trimer_2d, dtype=torch.bool)
        if star_enabled:
            batch.star_3d_distance = torch.tensor(star_distance, dtype=torch.float)
            batch.star_3d_asymmetry = torch.tensor(star_asymmetry, dtype=torch.float)
            batch.star_3d_valid = torch.tensor(star_valid, dtype=torch.bool)
    return batch


def custom_collate(data_list, random_conformer=False):
    """Compatibility spelling for the production MTS collator."""
    return mips_trimer_collate(data_list)
