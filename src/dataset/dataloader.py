import torch
from torch_geometric.data import Batch


def custom_collate(data_list):
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
    y_list = []
    kg_embedding_index_list = []
    kg_mask_list = []
    has_literature_link_list = []

    num_nodes2d_cum = 0

    for batch_idx, data in enumerate(data_list):
        # smiles
        smiles_list.append(data.smiles)
        input_ids_smiles_list.append(data.input_ids_smiles)
        attention_mask_smiles_list.append(data.attention_mask_smiles)

        # 2D graph data
        num_nodes2d = data.x.size(0)
        x_list.append(data.x)
        edge_index_list.append(data.edge_index + num_nodes2d_cum)
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)
        batch2d_list.append(torch.full((num_nodes2d,), batch_idx, dtype=torch.long))
        num_nodes2d_cum += num_nodes2d

        # 3D graph data
        z3d = data.z
        pos3d = data.pos
        num_nodes3d = pos3d.size(0)
        x3d_list.append(z3d)
        pos3d_list.append(pos3d)
        batch3d_list.append(torch.full((num_nodes3d,), batch_idx, dtype=torch.long))

        # fingerprint
        fp_list.append(data.fp)

        # Labels
        y_list.append(data.y.reshape(-1))
        if hasattr(data, "kg_embedding_index"):
            kg_embedding_index_list.append(data.kg_embedding_index.reshape(-1))
            kg_mask_list.append(data.kg_mask.reshape(-1))
            has_literature_link_list.append(data.has_literature_link.reshape(-1))

    # Create batched data
    batch = Batch()
    batch.x = torch.cat(x_list, dim=0)
    batch.edge_index = torch.cat(edge_index_list, dim=1)
    if edge_attr_list:
        batch.edge_attr = torch.cat(edge_attr_list, dim=0)
    batch.batch = torch.cat(batch2d_list, dim=0)
    batch.smiles = smiles_list
    batch.input_ids_smiles = torch.cat(input_ids_smiles_list, dim=0)
    batch.attention_mask_smiles = torch.cat(attention_mask_smiles_list, dim=0)
    batch.fp = torch.cat(fp_list, dim=0)
    batch.x3d = torch.cat(x3d_list, dim=0)
    batch.pos3d = torch.cat(pos3d_list, dim=0)
    batch.batch3d = torch.cat(batch3d_list, dim=0)

    batch.y = torch.stack(y_list, dim=0)
    if kg_embedding_index_list:
        batch.kg_embedding_index = torch.cat(kg_embedding_index_list, dim=0)
        batch.kg_mask = torch.cat(kg_mask_list, dim=0)
        batch.has_literature_link = torch.cat(has_literature_link_list, dim=0)
    
    return batch
