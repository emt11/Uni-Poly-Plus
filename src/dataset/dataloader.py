import torch
from torch_geometric.data import Batch
from .graph_data import SCAGE_CATEGORICAL_FEATURES, SCAGE_CONTINUOUS_FEATURES


def custom_collate(data_list, random_conformer=False):
    # Initialize lists to hold batched data
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
    periodic_aug_smiles_list = []
    periodic_aug_cut_identities_list = []
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
    lga_image_shift_list = []
    lga_pbc_distance_list = []
    lga_geometry_valid_list = []
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
    mips_atom_pair_3d_list = []
    mips_descriptor_valid_list = []
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

    for batch_idx, data in enumerate(data_list):
        # smiles
        smiles_list.append(data.smiles)
        periodic_aug_smiles_list.append(list(getattr(data, 'periodic_aug_smiles', [])))
        periodic_aug_cut_identities_list.append(
            list(getattr(data, 'periodic_aug_cut_identities', []))
        )
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

        if hasattr(data, "lga_edge_index"):
            local_edges = data.lga_edge_index.long()
            lga_edge_index_list.append(local_edges + node_offset)
            local_path = data.lga_path_index.long().clone()
            local_path[local_path >= 0] += node_offset
            lga_path_index_list.append(local_path)
            lga_path_mask_list.append(data.lga_path_mask.bool())
            lga_spd_list.append(data.lga_spd.long())
            lga_image_shift_list.append(data.lga_image_shift.long())
            distance_confs = data.lga_pbc_distance_confs.float()
            distance_idx = 0 if conf_idx is None else min(conf_idx, distance_confs.size(0) - 1)
            lga_pbc_distance_list.append(distance_confs[distance_idx])
            lga_geometry_valid_list.append(data.lga_geometry_valid.bool())
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
        mips_atom_pair_3d_list.append(
            getattr(data, 'mips_atom_pair_3d', torch.zeros(512, dtype=torch.float))
        )
        mips_descriptor_valid_list.append(torch.tensor(
            bool(getattr(data, 'mips_descriptor_valid', False)), dtype=torch.bool
        ))

        # fingerprint
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
        batch.lga_image_shift = torch.cat(lga_image_shift_list, dim=0)
        batch.lga_pbc_distance = torch.cat(lga_pbc_distance_list, dim=0)
        batch.lga_geometry_valid = torch.cat(lga_geometry_valid_list, dim=0)
        batch.periodic_lga_schema_version = 1
    batch.graph_available = torch.stack(graph_available_list, dim=0)
    batch.periodic_geometry_valid = torch.stack(periodic_geometry_valid_list, dim=0)
    batch.geometry_period_ru = torch.stack(geometry_period_ru_list, dim=0)
    batch.mips_repeat_factor = torch.stack(mips_repeat_factor_list, dim=0)
    batch.model_cell_ru = torch.stack(model_cell_ru_list, dim=0)
    batch.mips_boundary_distance = torch.stack(mips_boundary_distance_list, dim=0)
    batch.mips_condition_valid = torch.stack(mips_condition_valid_list, dim=0)
    batch.mips_alias_free = torch.stack(mips_alias_free_list, dim=0)
    batch.periodic_ru_count = torch.stack(periodic_ru_count_list, dim=0)
    batch.polygen_periodic_valid = torch.stack(polygen_periodic_valid_list, dim=0)
    batch.periodic_valid = torch.stack(periodic_valid_list, dim=0)
    batch.periodic_closure_error = torch.stack(periodic_closure_error_list, dim=0)
    batch.periodic_cell_length = torch.stack(periodic_cell_length_list, dim=0)
    sparse_lga_batch = len(lga_edge_index_list) == len(data_list)
    if not sparse_lga_batch:
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
    batch.periodic_aug_smiles = periodic_aug_smiles_list
    batch.periodic_aug_cut_identities = periodic_aug_cut_identities_list
    batch.input_ids_smiles = torch.cat(input_ids_smiles_list, dim=0)
    batch.attention_mask_smiles = torch.cat(attention_mask_smiles_list, dim=0)
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
    batch.mips_atom_pair_3d = torch.stack(mips_atom_pair_3d_list, dim=0)
    batch.mips_descriptor_valid = torch.stack(mips_descriptor_valid_list, dim=0)
    batch.cell = torch.stack(cell_list, dim=0)
    batch.pbc = torch.stack(pbc_list, dim=0)
    batch.screw_rotation = torch.stack(screw_rotation_list, dim=0)
    batch.screw_translation = torch.stack(screw_translation_list, dim=0)
    batch.screw_valid = torch.stack(screw_valid_list, dim=0)
    batch.screw_source_id = torch.stack(screw_source_id_list, dim=0)
    # Canonical name used by SCAGE and diagnostics.  Keep screw_source_id as a
    # compatibility alias for older checkpoints and analysis scripts.
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
