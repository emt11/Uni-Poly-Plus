import torch
from torch_geometric.data import Batch
from .mips_trimer_contract import (
    FEATURE_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION,
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_SCHEMA_VERSION,
)
from .mips_cache_validation import trimer_can_enter_mcl


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
            TOPOLOGY_CANONICAL
            if (
                bool(getattr(item, "mts_canonical_periodic", False))
                or int(getattr(item, "mips_local_lga_schema_version", 0)) == 2
            )
            else TOPOLOGY_EXPLICIT,
        ))
        for item in data_list
    }
    if len(representations) != 1:
        raise ValueError("cannot mix MTS topology representations in one batch")
    topology_representation = next(iter(representations))
    if topology_representation not in {TOPOLOGY_CANONICAL, TOPOLOGY_EXPLICIT}:
        raise ValueError("unsupported MTS topology representation")
    if any(
        bool(getattr(item, "mts_canonical_periodic", False))
        or int(getattr(item, "mips_local_lga_schema_version", 0)) == 2
        for item in data_list
    ) and not canonical_periodic_batch:
        raise ValueError(
            "cannot mix canonical-periodic and explicit MTS topology records"
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
    ablation_ids = {
        getattr(item, "mts_ablation_id", None) for item in data_list
    }
    if len(ablation_ids) > 1:
        raise ValueError("cannot mix MTS geometry ablations in one batch")
    ablation_id = next(iter(ablation_ids))
    mcl_enabled = all(
        bool(getattr(item, "mts_use_mcl", True)) for item in data_list
    )
    star_enabled = all(
        bool(getattr(item, "mts_use_star_rbf", True)) for item in data_list
    )
    random_mask_enabled = ablation_id == "A4_star_mcl_random_mask"
    trimer_pos, trimer_z, trimer_edges, trimer_bonds = [], [], [], []
    trimer_base, trimer_offset, trimer_central = [], [], []
    trimer_central_index, trimer_mapping, trimer_batch = [], [], []
    trimer_valid, trimer_is_3d, trimer_2d = [], [], []
    mcl_valid = []
    star_distance, star_asymmetry, star_valid = [], [], []
    mcl_thresholds = []
    angle_indices = []
    angle_bins = []
    angle_cos = []
    angle_ptr = [0]
    angle_valid = []
    # Fixed-shape lookup tables let the CUDA MCL path gather ragged Trimer
    # records without a Python loop or per-graph synchronisation.  They are
    # inexpensive (512 int32 values per graph at the largest bucket) and are
    # derived from already validated cache records; no cache schema change is
    # required.
    mcl_bucket_sizes = []
    mcl_key_rows = []
    mcl_query_rows = []
    mcl_query_local_rows = []
    mcl_query_canonical_rows = []
    node_offset = 0
    canonical_offset = 0
    pair_offset = 0
    trimer_offset_global = 0
    trimer_ptr = [0]
    # A4 count-matched random MCL sidecar rows (Plan mts_geometry_injection
    # A4): per-graph visible-key tables and graph query/trimer start offsets.
    mcl_random_visible20_rows = []
    mcl_random_visible50_rows = []
    mcl_query_start = []
    mcl_trimer_start = []
    mcl_random_mask_valid = []
    query_row_offset = 0

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
            item_mcl_valid = bool(trimer_can_enter_mcl(item, 0))
            mcl_valid.append(item_mcl_valid)
            star_distance.append(float(getattr(item, "star_3d_distance", 0.0)))
            star_asymmetry.append(float(getattr(item, "star_3d_asymmetry", 0.0)))
            star_valid.append(bool(getattr(item, "star_3d_valid", False)))
            raw_thresholds = getattr(item, "trimer_mcl_thresholds", None)
            if raw_thresholds is None:
                raw_thresholds = torch.full((2,), float("nan"))
            raw_thresholds = torch.as_tensor(raw_thresholds).float().reshape(2)
            if (
                bool(item_mcl_valid)
                and not bool(torch.isfinite(raw_thresholds).all())
                and n_trimer > 1
            ):
                pair_distances = torch.pdist(item.trimer_pos.float())
                if pair_distances.numel() and bool(torch.isfinite(pair_distances).all()):
                    raw_thresholds = torch.quantile(
                        pair_distances, pair_distances.new_tensor((0.20, 0.50))
                    ).float()
            mcl_thresholds.append(raw_thresholds)
            local_angles = getattr(
                item, "trimer_angle_index", torch.empty((0, 3), dtype=torch.long)
            ).long().reshape(-1, 3).clone()
            if local_angles.numel():
                local_angles += trimer_offset_global
            angle_indices.append(local_angles)
            angle_bins.append(getattr(
                item, "trimer_angle_bins", torch.empty((0,), dtype=torch.long)
            ).long().reshape(-1))
            angle_cos.append(getattr(
                item, "trimer_angle_cos", torch.empty((0,), dtype=torch.float)
            ).float().reshape(-1))
            angle_ptr.append(angle_ptr[-1] + int(local_angles.size(0)))
            angle_valid.append(bool(getattr(item, "trimer_angle_valid", False)))

            central_local = torch.nonzero(
                item.trimer_central_ru_mask.bool(), as_tuple=False
            ).flatten()
            central_count = int(central_local.numel())
            bucket_size = next(
                (
                    size for size in (24, 48, 96, 192, 384)
                    if n_trimer <= size and central_count <= size // 3
                ),
                0,
            )
            # Records outside the production 384-heavy-atom contract are
            # unavailable geometry.  Keep their rows empty so they retain the
            # exact O8 fallback.
            if not item_mcl_valid or bucket_size == 0:
                bucket_size = 0
                key_row = torch.full((384,), -1, dtype=torch.int32)
                query_row = torch.full((128,), -1, dtype=torch.int32)
                query_local_row = torch.full((128,), -1, dtype=torch.int32)
                query_canonical_row = torch.full((128,), -1, dtype=torch.int32)
            else:
                key_row = torch.full((384,), -1, dtype=torch.int32)
                query_row = torch.full((128,), -1, dtype=torch.int32)
                query_local_row = torch.full((128,), -1, dtype=torch.int32)
                query_canonical_row = torch.full((128,), -1, dtype=torch.int32)
                key_row[:n_trimer] = torch.arange(
                    trimer_offset_global,
                    trimer_offset_global + n_trimer,
                    dtype=torch.int32,
                )
                query_row[:central_count] = (
                    central_local.to(torch.int32) + trimer_offset_global
                )
                query_local_row[:central_count] = central_local.to(torch.int32)
                query_canonical_row[:central_count] = (
                    getattr(item, base_name)[central_local].to(torch.int32)
                    + canonical_offset - canonical_count
                )
                if not bool((
                    query_canonical_row[:central_count] >= 0
                ).all()) or not bool((
                    query_canonical_row[:central_count] < canonical_offset
                ).all()):
                    raise ValueError(
                        "MCL padded canonical index is outside the current "
                        "batched canonical range"
                    )
                if not bool((central_local < n_trimer).all()):
                    raise ValueError(
                        "MCL central-RU local index exceeds Trimer atom count"
                    )
            # A4 random-mask sidecar rows (Plan mts_geometry_injection A4):
            # expand the compact per-query visible key sets into [Q, 384] bool
            # tables aligned to the batched query index, and record this
            # graph's query/trimer start offsets.
            if random_mask_enabled:
                if not all(hasattr(item, name) for name in (
                    "mcl_random_ptr20", "mcl_random_ptr50",
                    "mcl_random_keys20", "mcl_random_keys50",
                    "mcl_random_mask_valid",
                )):
                    raise ValueError("A4 item is missing random-mask placeholder fields")
                n_q = int(central_count)
                vis20 = torch.zeros((n_q, 384), dtype=torch.bool)
                vis50 = torch.zeros((n_q, 384), dtype=torch.bool)
                random_valid = bool(getattr(item, "mcl_random_mask_valid", False))
                if random_valid and item_mcl_valid and bucket_size and n_q:
                    p20 = item.mcl_random_ptr20
                    p50 = item.mcl_random_ptr50
                    if len(p20) != n_q + 1 or len(p50) != n_q + 1:
                        raise ValueError("A4 compact mask pointer/query mismatch")
                    k20 = torch.as_tensor(
                        item.mcl_random_keys20, dtype=torch.long
                    )
                    k50 = torch.as_tensor(
                        item.mcl_random_keys50, dtype=torch.long
                    )
                    for q in range(n_q):
                        a, b = int(p20[q]), int(p20[q + 1])
                        a5, b5 = int(p50[q]), int(p50[q + 1])
                        if not (0 <= a <= b <= k20.numel() and 0 <= a5 <= b5 <= k50.numel()):
                            raise ValueError("A4 compact mask pointer is out of bounds")
                        if k20[a:b].numel() and bool(((k20[a:b] < 0) | (k20[a:b] >= n_trimer)).any()):
                            raise ValueError("A4 random 20% mask selects invalid Trimer atom")
                        if k50[a5:b5].numel() and bool(((k50[a5:b5] < 0) | (k50[a5:b5] >= n_trimer)).any()):
                            raise ValueError("A4 random 50% mask selects invalid Trimer atom")
                        vis20[q, k20[a:b]] = True
                        vis50[q, k50[a5:b5]] = True
                mcl_random_visible20_rows.append(vis20)
                mcl_random_visible50_rows.append(vis50)
                mcl_query_start.append(query_row_offset)
                mcl_trimer_start.append(trimer_offset_global)
                mcl_random_mask_valid.append(random_valid and item_mcl_valid and bool(bucket_size))
                query_row_offset += n_q
            mcl_bucket_sizes.append(bucket_size)
            mcl_key_rows.append(key_row)
            mcl_query_rows.append(query_row)
            mcl_query_local_rows.append(query_local_row)
            mcl_query_canonical_rows.append(query_canonical_row)
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
        atomic_parts = [
            torch.as_tensor(
                getattr(item, "atomic_numbers", getattr(item, "z"))
            ).long()
            for item in data_list
        ]
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
    batch.feature_schema = (
        FEATURE_SCHEMA
        if topology_representation == TOPOLOGY_CANONICAL
        else EXPLICIT_FEATURE_SCHEMA
    )
    batch.mips_local_lga_schema_version = (
        2 if topology_representation == TOPOLOGY_CANONICAL
        else EXPLICIT_LGA_SCHEMA_VERSION
    )
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
        if mcl_enabled:
            batch.mcl_valid = torch.tensor(mcl_valid, dtype=torch.bool)
        if star_enabled:
            batch.star_3d_distance = torch.tensor(star_distance, dtype=torch.float)
            batch.star_3d_asymmetry = torch.tensor(star_asymmetry, dtype=torch.float)
            batch.star_3d_valid = torch.tensor(star_valid, dtype=torch.bool)
        if mcl_enabled:
            batch.trimer_mcl_thresholds = torch.stack(mcl_thresholds, dim=0)
        if ablation_id is None:
            batch.trimer_angle_index = torch.cat(angle_indices, dim=0)
            batch.trimer_angle_bins = torch.cat(angle_bins, dim=0)
            batch.trimer_angle_cos = torch.cat(angle_cos, dim=0)
            batch.trimer_angle_ptr = torch.tensor(angle_ptr, dtype=torch.long)
            batch.trimer_angle_valid = torch.tensor(angle_valid, dtype=torch.bool)
            schemas = {
                str(getattr(item, "trimer_angle_cache_schema", ""))
                for item in data_list if hasattr(item, "trimer_angle_cache_schema")
            }
            batch.trimer_angle_cache_schema = (
                next(iter(schemas)) if len(schemas) == 1 else "mixed-or-missing"
            )
        if mcl_enabled:
            batch.mcl_bucket_size = torch.tensor(mcl_bucket_sizes, dtype=torch.int16)
            batch.mcl_key_index_padded = torch.stack(mcl_key_rows, dim=0)
            batch.mcl_query_index_padded = torch.stack(mcl_query_rows, dim=0)
            batch.mcl_query_local_index_padded = torch.stack(
                mcl_query_local_rows, dim=0
            )
            batch.mcl_query_canonical_index_padded = torch.stack(
                mcl_query_canonical_rows, dim=0
            )
        if random_mask_enabled:
            if len(mcl_random_visible20_rows) != len(data_list):
                raise ValueError("A4 random visibility rows are not graph-aligned")
            batch.mcl_random_visible20 = torch.cat(
                mcl_random_visible20_rows, dim=0
            )
            batch.mcl_random_visible50 = torch.cat(
                mcl_random_visible50_rows, dim=0
            )
            batch.mcl_query_start = torch.tensor(
                mcl_query_start, dtype=torch.int32
            )
            batch.mcl_trimer_start = torch.tensor(
                mcl_trimer_start, dtype=torch.int32
            )
            batch.mcl_random_mask_valid = torch.tensor(
                mcl_random_mask_valid, dtype=torch.bool
            )
        if mcl_enabled:
            batch.trimer_mcl_schema = TRIMER_CONTENT_SCHEMA
            batch.trimer_mcl_schema_version = TRIMER_SCHEMA_VERSION
    return batch


def custom_collate(data_list, random_conformer=False):
    """Compatibility spelling for the production MTS collator.

    The former implementation assembled a retired dense batch
    (including cell, PBC and geometry fields).  Keeping a second collator was
    both a maintenance hazard and an easy way to reintroduce that route, so
    all callers now use the sparse MIPS-Trimer contract.
    """
    return mips_trimer_collate(data_list)

''' retired dense collator body kept out of the active module
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    smiles_list = []
    input_ids_smiles_list = []
    attention_mask_smiles_list = []
    fp_list = []
    batch2d_list = []
    x3d_list = []
    pos3d_list = []
    batch3d_list = []
    cell_list = []
    pbc_list = []
    screw_rotation_list = []
    screw_translation_list = []
    screw_valid_list = []
    screw_source_id_list = []
    smer_valid_list = []
    smer_image_pos3d_list = []
    geom_build_ok_list = []
    geom_coordinate_ok_list = []
    geom_pool_mask_list = []
    geom_context_id_list = []
    graph_to_geom_index_list = []
    attachment_pair_list = []
    star_link_edge_list = []
    ordered_backbone_path_list = []
    ordered_backbone_ptr = [0]
    star_link_metadata_valid_list = []
    polymer_ecfp_target_list = []
    polymer_ecfp_valid_list = []
    polymer_ecfp_source_list = []
    scage_backbone_role_list = []
    scage_ru_index_list = []
    periodic_ru_count_list = []
    polygen_periodic_valid_list = []
    periodic_valid_list = []
    periodic_closure_error_list = []
    periodic_cell_length_list = []
    scage_spd_list = []
    scage_path_bond_fields_list = []
    lga_edge_index_list = []
    lga_spd_list = []
    lga_path_index_list = []
    lga_path_mask_list = []
    lga_path_bond_hist_list = []
    lga_star_edge_mask_list = []
    canonical_ru_atom_index_list = []
    canonical_ru_atom_local_index_list = []
    ru_copy_index_list = []
    canonical_pair_index_list = []
    graph_available_list = []
    periodic_geometry_valid_list = []
    geometry_period_ru_list = []
    mips_repeat_factor_list = []
    model_cell_ru_list = []
    mips_boundary_distance_list = []
    mips_condition_valid_list = []
    mips_alias_free_list = []
    mips_x_list = []
    mips_backbone_mask_list = []
    mips_path_nodes_list = []
    mips_md_list = []
    mips_md_valid_list = []
    trimer_pos_list = []
    trimer_atomic_number_list = []
    trimer_edge_index_list = []
    trimer_bond_type_list = []
    trimer_base_ru_atom_index_list = []
    trimer_ru_offset_list = []
    trimer_central_ru_mask_list = []
    trimer_central_atom_index_list = []
    mips_to_trimer_central_index_list = []
    trimer_batch_list = []
    trimer_geometry_valid_list = []
    trimer_geometry_is_3d_list = []
    trimer_2d_fallback_list = []
    trimer_geometry_source_list = []
    trimer_failure_code_list = []
    trimer_conformer_energy_list = []
    trimer_conformer_seed_list = []
    star_3d_distance_list = []
    star_3d_asymmetry_list = []
    star_3d_valid_list = []
    trimer_ptr = [0]
    descriptor_names = ("shape", "usrcat", "autocorr3d", "rdf", "morse", "whim")
    descriptor_dims = (11, 60, 80, 210, 224, 114)
    scage_descriptor_lists = {name: [] for name in descriptor_names}
    scage_descriptor_valid_list = []
    y_list = []
    cross_task_aux_y_list = []
    cross_task_aux_mask_list = []
    scage_feature_lists = {
        name: [] for name in SCAGE_CATEGORICAL_FEATURES + SCAGE_CONTINUOUS_FEATURES
    }

    num_nodes2d_cum = 0
    num_nodes3d_cum = 0
    canonical_atoms_cum = 0
    canonical_pairs_cum = 0
    trimer_nodes_cum = 0

    for batch_idx, data in enumerate(data_list):
        # smiles
        smiles_list.append(data.smiles)
        if hasattr(data, "input_ids_smiles"):
            input_ids_smiles_list.append(data.input_ids_smiles)
            attention_mask_smiles_list.append(data.attention_mask_smiles)

        # 2D graph data — one graph per sample
        num_nodes2d = data.x.size(0)
        node_offset = num_nodes2d_cum
        x_list.append(data.x)
        edge_index_list.append(data.edge_index + num_nodes2d_cum)
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)
        batch2d_list.append(torch.full((num_nodes2d,), batch_idx, dtype=torch.long))
        for name in scage_feature_lists:
            if hasattr(data, name):
                scage_feature_lists[name].append(getattr(data, name))
        scage_backbone_role_list.append(
            getattr(data, 'scage_backbone_role', torch.zeros(num_nodes2d, dtype=torch.long))
        )
        scage_ru_index_list.append(
            getattr(data, 'scage_ru_index', torch.zeros(num_nodes2d, dtype=torch.long))
        )
        periodic_ru_count_list.append(torch.tensor(
            int(getattr(data, 'periodic_ru_count', 1)), dtype=torch.long
        ))
        polygen_periodic_valid_list.append(torch.tensor(
            bool(getattr(data, 'polygen_periodic_valid', False)), dtype=torch.bool
        ))
        periodic_valid_list.append(torch.tensor(
            bool(getattr(data, 'periodic_valid', False)), dtype=torch.bool
        ))
        periodic_closure_error_list.append(torch.tensor(
            float(getattr(data, 'periodic_closure_error', float('inf'))), dtype=torch.float
        ))
        periodic_cell_length_list.append(torch.tensor(
            float(getattr(data, 'periodic_cell_length', 0.0)), dtype=torch.float
        ))
        scage_spd_list.append(getattr(data, 'scage_spd', None))
        scage_path_bond_fields_list.append(getattr(data, 'scage_path_bond_fields', None))
        mips_x_list.append(getattr(data, 'mips_x', torch.zeros(num_nodes2d, 137)))
        mips_backbone_mask_list.append(
            getattr(data, 'mips_backbone_mask', torch.zeros(num_nodes2d, dtype=torch.long))
        )
        mips_path_nodes_list.append(getattr(data, 'mips_path_nodes', None))
        metadata_valid = bool(getattr(data, 'star_link_metadata_valid', False))
        pair = getattr(data, 'attachment_pair', torch.tensor([-1, -1], dtype=torch.long)).long()
        star_edge = getattr(data, 'star_link_edge', torch.tensor([-1, -1], dtype=torch.long)).long()
        if metadata_valid and pair.numel() == 2 and (pair >= 0).all():
            attachment_pair_list.append(pair + node_offset)
        else:
            attachment_pair_list.append(torch.tensor([-1, -1], dtype=torch.long))
        if metadata_valid and star_edge.numel() == 2 and (star_edge >= 0).all():
            star_link_edge_list.append(star_edge + node_offset)
        else:
            star_link_edge_list.append(torch.tensor([-1, -1], dtype=torch.long))
        path = getattr(data, 'ordered_backbone_path', torch.empty(0, dtype=torch.long)).long()
        if metadata_valid and path.numel():
            ordered_backbone_path_list.append(path + node_offset)
            ordered_backbone_ptr.append(ordered_backbone_ptr[-1] + int(path.numel()))
        else:
            ordered_backbone_ptr.append(ordered_backbone_ptr[-1])
        star_link_metadata_valid_list.append(torch.tensor(metadata_valid, dtype=torch.bool))
        num_nodes2d_cum += num_nodes2d

        # 3D graph data. Training loaders can sample one conformer per epoch;
        # evaluation keeps the deterministic first conformer stored in data.pos.
        z3d = data.z
        conf_idx = None
        if random_conformer and hasattr(data, 'pos_confs') and data.pos_confs.dim() == 3:
            conf_idx = torch.randint(data.pos_confs.size(0), (1,)).item()
            pos3d = data.pos_confs[conf_idx]
        else:
            pos3d = data.pos
        num_nodes3d = pos3d.size(0)
        x3d_list.append(z3d)
        pos3d_list.append(pos3d)
        batch3d_list.append(torch.full((num_nodes3d,), batch_idx, dtype=torch.long))
        geom_context_id_list.append(torch.tensor(int(getattr(data, 'geom_context_id', 2)), dtype=torch.long))
        if hasattr(data, 'graph_to_geom_index') and data.graph_to_geom_index.numel() == num_nodes2d:
            graph_to_geom_index_list.append(data.graph_to_geom_index.long() + num_nodes3d_cum)
        else:
            graph_to_geom_index_list.append(torch.full((num_nodes2d,), -1, dtype=torch.long))
        num_nodes3d_cum += num_nodes3d

        canonical_offset = canonical_atoms_cum
        if hasattr(data, "lga_edge_index"):
            local_edges = data.lga_edge_index.long()
            lga_edge_index_list.append(local_edges + node_offset)
            local_path = data.lga_path_index.long().clone()
            local_path[local_path >= 0] += node_offset
            lga_path_index_list.append(local_path)
            lga_path_mask_list.append(data.lga_path_mask.bool())
            lga_spd_list.append(data.lga_spd.long())
            lga_path_bond_hist_list.append(data.lga_path_bond_hist.float())
            lga_star_edge_mask_list.append(data.lga_star_edge_mask.bool())
            local_canonical = data.canonical_ru_atom_index.long()
            canonical_ru_atom_local_index_list.append(local_canonical.clone())
            canonical_ru_atom_index_list.append(
                local_canonical + canonical_atoms_cum
            )
            ru_copy_index_list.append(data.ru_copy_index.long())
            local_pair = data.canonical_pair_index.long()
            canonical_pair_index_list.append(local_pair + canonical_pairs_cum)
            canonical_atoms_cum += int(local_canonical.max().item()) + 1
            canonical_pairs_cum += int(local_pair.max().item()) + 1
        if hasattr(data, "trimer_geometry_valid"):
            trimer_nodes = int(data.trimer_pos.size(0))
            trimer_pos_list.append(data.trimer_pos.float())
            trimer_atomic_number_list.append(
                data.trimer_atomic_number.long()
            )
            local_trimer_edges = data.trimer_edge_index.long()
            trimer_edge_index_list.append(
                local_trimer_edges + trimer_nodes_cum
            )
            trimer_bond_type_list.append(data.trimer_bond_type.long())
            trimer_base_ru_atom_index_list.append(
                data.trimer_base_ru_atom_id.long() + canonical_offset
            )
            trimer_ru_offset_list.append(data.trimer_ru_offset.long())
            trimer_central_ru_mask_list.append(
                data.trimer_central_ru_mask.bool()
            )
            trimer_central_atom_index_list.append(
                data.trimer_central_atom_index.long() + trimer_nodes_cum
            )
            local_mapping = data.mips_to_trimer_central_index.long().clone()
            local_mapping[local_mapping >= 0] += trimer_nodes_cum
            mips_to_trimer_central_index_list.append(local_mapping)
            trimer_batch_list.append(torch.full(
                (trimer_nodes,), batch_idx, dtype=torch.long
            ))
            trimer_geometry_valid_list.append(torch.tensor(
                bool(data.trimer_geometry_valid), dtype=torch.bool
            ))
            trimer_geometry_is_3d_list.append(torch.as_tensor(
                getattr(data, "trimer_geometry_is_3d", False)
            ).bool().reshape(()))
            trimer_2d_fallback_list.append(torch.as_tensor(
                getattr(data, "trimer_2d_fallback", False)
            ).bool().reshape(()))
            trimer_geometry_source_list.append(
                str(getattr(data, "trimer_geometry_source", "unavailable"))
            )
            trimer_failure_code_list.append(
                str(getattr(data, "trimer_failure_code", ""))
            )
            trimer_conformer_energy_list.append(torch.tensor(
                float(getattr(
                    data, "trimer_conformer_energy", float("inf")
                )),
                dtype=torch.float,
            ))
            trimer_conformer_seed_list.append(torch.tensor(
                int(getattr(data, "trimer_conformer_seed", 0)),
                dtype=torch.long,
            ))
            star_3d_distance_list.append(
                torch.as_tensor(data.star_3d_distance).float().reshape(())
            )
            star_3d_asymmetry_list.append(
                torch.as_tensor(data.star_3d_asymmetry).float().reshape(())
            )
            star_3d_valid_list.append(
                torch.as_tensor(data.star_3d_valid).bool().reshape(())
            )
            trimer_nodes_cum += trimer_nodes
            trimer_ptr.append(trimer_nodes_cum)
        graph_available_list.append(torch.tensor(
            bool(getattr(data, "graph_available", True)), dtype=torch.bool
        ))
        periodic_geometry_valid_list.append(torch.tensor(
            bool(getattr(data, "periodic_geometry_valid", False)), dtype=torch.bool
        ))
        geometry_period_ru_list.append(torch.tensor(
            int(getattr(data, "geometry_period_ru", 0)), dtype=torch.long
        ))
        mips_repeat_factor_list.append(torch.tensor(
            int(getattr(data, "mips_repeat_factor", 1)), dtype=torch.long
        ))
        model_cell_ru_list.append(torch.tensor(
            int(getattr(data, "model_cell_ru", 1)), dtype=torch.long
        ))
        mips_boundary_distance_list.append(torch.tensor(
            int(getattr(data, "mips_boundary_distance", -1)), dtype=torch.long
        ))
        mips_condition_valid_list.append(torch.tensor(
            bool(getattr(data, "mips_condition_valid", True)), dtype=torch.bool
        ))
        mips_alias_free_list.append(torch.tensor(
            bool(getattr(data, "mips_alias_free", True)), dtype=torch.bool
        ))

        geom_ok = bool(getattr(data, 'geom_build_ok', True))
        coordinate_ok = bool(getattr(data, 'geom_coordinate_ok', geom_ok))
        geom_build_ok_list.append(torch.tensor(geom_ok, dtype=torch.bool))
        geom_coordinate_ok_list.append(torch.tensor(coordinate_ok, dtype=torch.bool))
        has_valid_pbc = (
            geom_ok
            and hasattr(data, 'cell')
            and hasattr(data, 'pbc')
            and bool(getattr(data, 'pbc').bool().any().item())
        )
        if has_valid_pbc:
            if conf_idx is not None and hasattr(data, 'cell_confs') and data.cell_confs.dim() == 3:
                cell_list.append(data.cell_confs[conf_idx])
            else:
                cell_list.append(data.cell)
            pbc_list.append(data.pbc)
        else:
            # Keep per-sample geometry metadata aligned even for mixed batches.
            # A fallback must never inherit another sample's periodic cell.
            cell_list.append(torch.zeros((3, 3), dtype=torch.float))
            pbc_list.append(torch.zeros(3, dtype=torch.bool))
        screw_valid = bool(getattr(data, 'screw_valid', False))
        if screw_valid:
            if conf_idx is not None and hasattr(data, 'screw_rotation_confs'):
                screw_rotation_list.append(data.screw_rotation_confs[conf_idx])
                screw_translation_list.append(data.screw_translation_confs[conf_idx])
            else:
                screw_rotation_list.append(data.screw_rotation)
                screw_translation_list.append(data.screw_translation)
        else:
            screw_rotation_list.append(torch.eye(3, dtype=torch.float))
            screw_translation_list.append(torch.zeros(3, dtype=torch.float))
        screw_valid_list.append(torch.tensor(screw_valid, dtype=torch.bool))
        screw_source_id_list.append(torch.tensor(
            int(getattr(
                data, 'geometry_source_id',
                getattr(data, 'geom_screw_source_id', 3 if screw_valid else (1 if coordinate_ok else 0))
            )),
            dtype=torch.long,
        ))
        smer_valid_list.append(torch.tensor(bool(getattr(data, 'smer_valid', False)), dtype=torch.bool))
        smer_images = None
        if bool(getattr(data, 'smer_valid', False)):
            if conf_idx is not None and hasattr(data, 'smer_image_pos_confs'):
                smer_images = data.smer_image_pos_confs[conf_idx]
            elif hasattr(data, 'smer_image_pos'):
                smer_images = data.smer_image_pos
        if smer_images is not None and smer_images.shape[:2] == (3, num_nodes2d):
            smer_image_pos3d_list.append(smer_images.permute(1, 0, 2).float())
        else:
            smer_image_pos3d_list.append(torch.zeros((num_nodes2d, 3, 3), dtype=torch.float))
        if hasattr(data, 'geom_pool_mask'):
            geom_pool_mask_list.append(data.geom_pool_mask.bool())
        descriptor_conf_idx = int(conf_idx) if conf_idx is not None else 0
        descriptor_valid = getattr(data, 'scage_descriptor_valid_confs', torch.zeros(0, dtype=torch.bool))
        selected_valid = bool(
            descriptor_conf_idx < descriptor_valid.numel()
            and descriptor_valid[descriptor_conf_idx].item()
        )
        scage_descriptor_valid_list.append(torch.tensor(selected_valid, dtype=torch.bool))
        for name, dim in zip(descriptor_names, descriptor_dims):
            conformers = getattr(data, f'scage_descriptor_{name}_confs', None)
            if conformers is not None and descriptor_conf_idx < conformers.size(0):
                scage_descriptor_lists[name].append(conformers[descriptor_conf_idx].float())
            else:
                scage_descriptor_lists[name].append(torch.zeros(dim, dtype=torch.float))
        mips_md_list.append(getattr(data, 'mips_md', torch.zeros(200, dtype=torch.float)))
        mips_md_valid_list.append(torch.tensor(
            bool(getattr(data, 'mips_md_valid', False)), dtype=torch.bool
        ))

        # fingerprint
        if hasattr(data, "fp"):
            fp_list.append(data.fp)
        polymer_ecfp_target_list.append(
            getattr(data, 'polymer_ecfp_target', torch.zeros(2048, dtype=torch.float)).reshape(-1)
        )
        polymer_ecfp_valid_list.append(
            torch.tensor(bool(getattr(data, 'polymer_ecfp_valid', False)), dtype=torch.bool)
        )
        polymer_ecfp_source_list.append(str(getattr(data, 'polymer_ecfp_source', 'invalid')))

        # Labels
        y_list.append(data.y.reshape(-1))
        cross_task_aux_y_list.append(
            getattr(data, 'cross_task_aux_y', torch.empty(0, dtype=torch.float)).reshape(-1)
        )
        cross_task_aux_mask_list.append(
            getattr(
                data, 'cross_task_aux_mask', torch.empty(0, dtype=torch.bool)
            ).reshape(-1)
        )

    # Create batched data
    batch = Batch()
    batch.x = torch.cat(x_list, dim=0)
    batch.edge_index = torch.cat(edge_index_list, dim=1)
    if edge_attr_list:
        batch.edge_attr = torch.cat(edge_attr_list, dim=0)
    batch.batch = torch.cat(batch2d_list, dim=0)
    for name, values in scage_feature_lists.items():
        if len(values) == len(data_list):
            setattr(batch, name, torch.cat(values, dim=0))
    batch.scage_backbone_role = torch.cat(scage_backbone_role_list, dim=0)
    batch.scage_ru_index = torch.cat(scage_ru_index_list, dim=0)
    if len(lga_edge_index_list) == len(data_list):
        batch.lga_edge_index = torch.cat(lga_edge_index_list, dim=1)
        batch.lga_spd = torch.cat(lga_spd_list, dim=0)
        batch.lga_path_index = torch.cat(lga_path_index_list, dim=0)
        batch.lga_path_mask = torch.cat(lga_path_mask_list, dim=0)
        batch.lga_star_edge_mask = torch.cat(
            lga_star_edge_mask_list, dim=0
        )
        if len(lga_path_bond_hist_list) == len(data_list):
            batch.lga_path_bond_hist = torch.cat(
                lga_path_bond_hist_list, dim=0
            )
            batch.canonical_ru_atom_index = torch.cat(
                canonical_ru_atom_index_list, dim=0
            )
            batch.canonical_ru_atom_local_index = torch.cat(
                canonical_ru_atom_local_index_list, dim=0
            )
            batch.ru_copy_index = torch.cat(ru_copy_index_list, dim=0)
            batch.canonical_pair_index = torch.cat(
                canonical_pair_index_list, dim=0
            )
            batch.mips_local_lga_schema_version = 1
            batch.feature_schema = FEATURE_SCHEMA
    if len(trimer_geometry_valid_list) == len(data_list):
        batch.trimer_pos = torch.cat(trimer_pos_list, dim=0)
        batch.trimer_atomic_number = torch.cat(
            trimer_atomic_number_list, dim=0
        )
        batch.trimer_edge_index = torch.cat(
            trimer_edge_index_list, dim=1
        )
        batch.trimer_bond_type = torch.cat(
            trimer_bond_type_list, dim=0
        )
        batch.trimer_base_ru_atom_index = torch.cat(
            trimer_base_ru_atom_index_list, dim=0
        )
        batch.trimer_ru_offset = torch.cat(trimer_ru_offset_list, dim=0)
        batch.trimer_central_ru_mask = torch.cat(
            trimer_central_ru_mask_list, dim=0
        )
        batch.trimer_central_atom_index = torch.cat(
            trimer_central_atom_index_list, dim=0
        )
        batch.mips_to_trimer_central_index = torch.cat(
            mips_to_trimer_central_index_list, dim=0
        )
        batch.trimer_batch = torch.cat(trimer_batch_list, dim=0)
        batch.trimer_ptr = torch.tensor(trimer_ptr, dtype=torch.long)
        batch.trimer_geometry_valid = torch.stack(
            trimer_geometry_valid_list, dim=0
        )
        batch.trimer_geometry_is_3d = torch.stack(
            trimer_geometry_is_3d_list, dim=0
        )
        batch.trimer_2d_fallback = torch.stack(
            trimer_2d_fallback_list, dim=0
        )
        batch.trimer_geometry_source = trimer_geometry_source_list
        batch.trimer_failure_code = trimer_failure_code_list
        batch.trimer_conformer_energy = torch.stack(
            trimer_conformer_energy_list, dim=0
        )
        batch.trimer_conformer_seed = torch.stack(
            trimer_conformer_seed_list, dim=0
        )
        batch.star_3d_distance = torch.stack(star_3d_distance_list, dim=0)
        batch.star_3d_asymmetry = torch.stack(
            star_3d_asymmetry_list, dim=0
        )
        batch.star_3d_valid = torch.stack(star_3d_valid_list, dim=0)
        batch.trimer_mcl_schema = TRIMER_CONTENT_SCHEMA
        batch.trimer_mcl_schema_version = TRIMER_SCHEMA_VERSION
        batch.feature_schema = FEATURE_SCHEMA
    sparse_lga_batch = len(lga_edge_index_list) == len(data_list)
    batch.graph_available = torch.stack(graph_available_list, dim=0)
    batch.mips_repeat_factor = torch.stack(mips_repeat_factor_list, dim=0)
    batch.mips_boundary_distance = torch.stack(mips_boundary_distance_list, dim=0)
    batch.mips_condition_valid = torch.stack(mips_condition_valid_list, dim=0)
    batch.mips_alias_free = torch.stack(mips_alias_free_list, dim=0)
    if not sparse_lga_batch:
        batch.periodic_geometry_valid = torch.stack(
            periodic_geometry_valid_list, dim=0
        )
        batch.geometry_period_ru = torch.stack(geometry_period_ru_list, dim=0)
        batch.model_cell_ru = torch.stack(model_cell_ru_list, dim=0)
        batch.periodic_ru_count = torch.stack(periodic_ru_count_list, dim=0)
        batch.polygen_periodic_valid = torch.stack(
            polygen_periodic_valid_list, dim=0
        )
        batch.periodic_valid = torch.stack(periodic_valid_list, dim=0)
        batch.periodic_closure_error = torch.stack(
            periodic_closure_error_list, dim=0
        )
        batch.periodic_cell_length = torch.stack(
            periodic_cell_length_list, dim=0
        )
        max_nodes = max(int(data.x.size(0)) for data in data_list)
        batch.scage_spd = torch.full(
            (len(data_list), max_nodes, max_nodes), 21, dtype=torch.uint8
        )
        batch.scage_path_bond_fields = torch.zeros(
            (len(data_list), max_nodes, max_nodes, 5, 5), dtype=torch.uint8
        )
        batch.mips_path_nodes = torch.full(
            (len(data_list), max_nodes, max_nodes, 3), -1, dtype=torch.long
        )
        for graph_idx, data in enumerate(data_list):
            count = int(data.x.size(0))
            if scage_spd_list[graph_idx] is not None:
                batch.scage_spd[graph_idx, :count, :count] = scage_spd_list[graph_idx]
            if scage_path_bond_fields_list[graph_idx] is not None:
                batch.scage_path_bond_fields[graph_idx, :count, :count] = (
                    scage_path_bond_fields_list[graph_idx]
                )
            if mips_path_nodes_list[graph_idx] is not None:
                batch.mips_path_nodes[graph_idx, :count, :count] = mips_path_nodes_list[graph_idx]
    batch.smiles = smiles_list
    if len(input_ids_smiles_list) == len(data_list):
        batch.input_ids_smiles = torch.cat(input_ids_smiles_list, dim=0)
        batch.attention_mask_smiles = torch.cat(
            attention_mask_smiles_list, dim=0
        )
    if len(fp_list) == len(data_list):
        batch.fp = torch.cat(fp_list, dim=0)
    batch.x3d = torch.cat(x3d_list, dim=0)
    batch.pos3d = torch.cat(pos3d_list, dim=0)
    batch.batch3d = torch.cat(batch3d_list, dim=0)
    batch.geom_context_id = torch.stack(geom_context_id_list, dim=0)
    batch.geom_build_ok = torch.stack(geom_build_ok_list, dim=0)
    batch.geom_coordinate_ok = torch.stack(geom_coordinate_ok_list, dim=0)
    batch.graph_to_geom_index = torch.cat(graph_to_geom_index_list, dim=0)
    batch.attachment_pair = torch.stack(attachment_pair_list, dim=0)
    batch.star_link_edge = torch.stack(star_link_edge_list, dim=0)
    batch.ordered_backbone_path = (
        torch.cat(ordered_backbone_path_list, dim=0)
        if ordered_backbone_path_list else torch.empty(0, dtype=torch.long)
    )
    batch.ordered_backbone_ptr = torch.tensor(ordered_backbone_ptr, dtype=torch.long)
    batch.star_link_metadata_valid = torch.stack(star_link_metadata_valid_list, dim=0)
    batch.polymer_ecfp_target = torch.stack(polymer_ecfp_target_list, dim=0)
    batch.polymer_ecfp_valid = torch.stack(polymer_ecfp_valid_list, dim=0)
    batch.polymer_ecfp_source = polymer_ecfp_source_list
    for name, values in scage_descriptor_lists.items():
        setattr(batch, f'scage_descriptor_{name}', torch.stack(values, dim=0))
    batch.scage_descriptor_valid = torch.stack(scage_descriptor_valid_list, dim=0)
    batch.mips_x = torch.cat(mips_x_list, dim=0)
    batch.mips_backbone_mask = torch.cat(mips_backbone_mask_list, dim=0)
    batch.mips_md = torch.stack(mips_md_list, dim=0)
    batch.mips_md_valid = torch.stack(mips_md_valid_list, dim=0)
    # The sparse non-PBC MIPS route must not expose cell/PBC/image data.
    # Legacy dense batches retain these fields for historical diagnostics;
    # they are not part of the active MTS route.
    if not sparse_lga_batch:
        batch.cell = torch.stack(cell_list, dim=0)
        batch.pbc = torch.stack(pbc_list, dim=0)
        batch.screw_rotation = torch.stack(screw_rotation_list, dim=0)
        batch.screw_translation = torch.stack(screw_translation_list, dim=0)
        batch.screw_valid = torch.stack(screw_valid_list, dim=0)
        batch.screw_source_id = torch.stack(screw_source_id_list, dim=0)
        batch.geometry_source_id = batch.screw_source_id
        batch.smer_valid = torch.stack(smer_valid_list, dim=0)
        batch.smer_image_pos3d = torch.cat(smer_image_pos3d_list, dim=0)
    if geom_pool_mask_list and len(geom_pool_mask_list) == len(data_list):
        batch.geom_pool_mask = torch.cat(geom_pool_mask_list, dim=0)

    batch.y = torch.stack(y_list, dim=0)
    aux_widths = {int(value.numel()) for value in cross_task_aux_y_list}
    mask_widths = {int(value.numel()) for value in cross_task_aux_mask_list}
    if len(aux_widths) != 1 or aux_widths != mask_widths:
        raise ValueError(
            'cross_task_aux_y and cross_task_aux_mask must have one '
            'consistent width per batch'
        )
    aux_width = next(iter(aux_widths))
    if aux_width:
        batch.cross_task_aux_y = torch.stack(cross_task_aux_y_list, dim=0)
        batch.cross_task_aux_mask = torch.stack(cross_task_aux_mask_list, dim=0)
    else:
        batch.cross_task_aux_y = torch.empty(len(data_list), 0, dtype=torch.float)
        batch.cross_task_aux_mask = torch.empty(len(data_list), 0, dtype=torch.bool)

    return batch
'''
