import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom')


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain UniEncoderAttention Model")
    parser.add_argument(
        '--modalities',
        nargs='+',
        type=parse_modality,
        default=['smiles', 'graph', 'fp', 'geom'],
        help="Modalities to use. Supported: smiles, graph, fp, geom."
    )
    parser.add_argument(
        '--geometry_encoder',
        type=str,
        choices=['painn', 'schnet'],
        default='painn',
        help="Geometry encoder backend."
    )
    parser.add_argument(
        '--graph_input',
        type=str,
        choices=['repeat_unit', 'star_linking'],
        default='star_linking',
        help="Graph input type. 'repeat_unit' keeps the original graph; 'star_linking' removes two attachment atoms and connects their boundary atoms for graph-only topology input."
    )
    parser.add_argument(
        '--smiles_model_name',
        type=str,
        default="./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        help="Pretrained model name or path for SMILES"
    )
    parser.add_argument(
        '--gnn_model_name',
        type=str,
        default="",
        help="Pretrained GNN model path"
    )
    parser.add_argument(
        '--geom_model_name',
        type=str,
        default="",
        help="Pretrained Geometry model path"
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="If set, freeze the pretrained model weights."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='smi_all',
        help="Name of the dataset for pretraining (unlabeled or labeled, but labels unused here)"
    )
    parser.add_argument(
        '--pretrain_stage',
        type=str,
        choices=['graph_geom', 'alignment', 'joint'],
        default='joint',
        help=(
            "Pretraining stage. graph_geom trains graph/geometry auxiliary tasks only; "
            "alignment trains multi-modal contrastive alignment, optionally with small auxiliary losses; "
            "joint keeps the previous behavior and sums all enabled losses."
        )
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='',
        help="Optional checkpoint to initialize this pretraining stage, typically the graph_geom checkpoint before alignment."
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help="Root directory of the dataset."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for pretraining."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pretrained_models/saved_pretrained_model.pth',
        help="Path to save the pretrained model."
    )
    parser.add_argument(
        '--graph_num_layers',
        type=int,
        default=6,
        help="Number of GIN/GINE graph layers."
    )
    parser.add_argument(
        '--graph_emb_dim',
        type=int,
        default=256,
        help="Hidden dimension for GIN/GINE graph encoder."
    )
    parser.add_argument(
        '--graph_dropout',
        type=float,
        default=0.1,
        help="Dropout ratio for GIN/GINE graph encoder."
    )
    parser.add_argument(
        '--graph_pooling',
        type=str,
        choices=['sum', 'mean', 'max', 'attention', 'set2set', 'set2set1', 'set2set2'],
        default='attention',
        help="Graph-level pooling for GIN/GINE encoder."
    )
    parser.add_argument(
        '--joint_embedding_dim',
        type=int,
        default=256,
        help="Shared projection dimension for modality fusion."
    )
    parser.add_argument(
        '--feature_source_dataset',
        type=str,
        default=None,
        help="Dataset used to build the SMILES-level feature cache. Defaults to --dataset_name."
    )
    parser.add_argument(
        '--disable_feature_cache',
        action='store_true',
        help="Disable SMILES-level feature cache and use legacy per-dataset caching."
    )
    parser.add_argument(
        '--rebuild_feature_cache',
        action='store_true',
        help="Force rebuild the feature cache even if it already exists."
    )
    parser.add_argument(
        '--max_smiles_length',
        type=int,
        default=None,
        help="Override SMILES token max length. Computed from feature_source_dataset when not set."
    )
    parser.add_argument(
        '--geom_denoise_weight',
        type=float,
        default=1.0,
        help="Weight for geometry coordinate/distance denoising pretraining loss."
    )
    parser.add_argument(
        '--geom_noise_std',
        type=float,
        default=0.2,
        help="Gaussian coordinate noise std for geometry denoising pretraining."
    )
    parser.add_argument(
        '--geom_distance_loss_weight',
        type=float,
        default=1.0,
        help="Relative weight of pairwise distance reconstruction inside geometry loss."
    )
    parser.add_argument(
        '--graph_pretrain_weight',
        type=float,
        default=1.0,
        help="Weight for graph-specific pretraining losses."
    )
    parser.add_argument(
        '--graph_mask_atom_weight',
        type=float,
        default=1.0,
        help="Weight for masked graph atom-feature reconstruction."
    )
    parser.add_argument(
        '--graph_starlink_consistency_weight',
        type=float,
        default=0.5,
        help="Weight for repeat-unit vs star-linking graph consistency loss."
    )
    parser.add_argument(
        '--graph_mask_ratio',
        type=float,
        default=0.15,
        help="Node masking ratio for graph atom-feature reconstruction."
    )
    parser.add_argument(
        '--max_smiles_length_cap',
        type=int,
        default=256,
        help="Cap for auto-computed max SMILES token length (default: 256)."
    )
    return parser.parse_args()

def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _stage_loss_weights(args):
    if args.pretrain_stage == 'graph_geom':
        return {
            'contrastive': 0.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    if args.pretrain_stage == 'alignment':
        return {
            'contrastive': 1.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    return {
        'contrastive': 1.0,
        'graph': float(args.graph_pretrain_weight),
        'geom': float(args.geom_denoise_weight),
    }


def _pairwise_distance_loss(pred_pos, target_pos, batch, weight=1.0):
    coord_loss = F.smooth_l1_loss(pred_pos, target_pos)
    unique_batches = torch.unique(batch)
    dist_losses = []
    for batch_id in unique_batches:
        mask = batch == batch_id
        if int(mask.sum().item()) < 2:
            continue
        pred_dist = torch.cdist(pred_pos[mask], pred_pos[mask])
        target_dist = torch.cdist(target_pos[mask], target_pos[mask])
        dist_losses.append(F.smooth_l1_loss(pred_dist, target_dist))
    if dist_losses:
        dist_loss = torch.stack(dist_losses).mean()
    else:
        dist_loss = coord_loss.new_tensor(0.0)
    return coord_loss + float(weight) * dist_loss


def _build_repeat_unit_batch(smiles_list, device):
    from torch_geometric.data import Batch
    from src.dataset.graph_data import build_graph_for_input
    graphs = [build_graph_for_input(smiles, graph_input='repeat_unit') for smiles in smiles_list]
    return Batch.from_data_list(graphs).to(device)


def _graph_pretrain_loss(base_model, data, graph_atom_head, mask_ratio, mask_weight, consistency_weight):
    if 'graph' not in base_model.encoders:
        return data.x.new_tensor(0.0)

    graph_module = base_model.encoders['graph']
    graph_encoder = graph_module.encoder
    from src.dataset.graph_data import allowable_features
    num_atom_symbols = len(allowable_features['possible_atom_symbols'])
    atom_symbol_targets = data.x[:, :num_atom_symbols].argmax(dim=1).long()
    mask = torch.rand(data.x.size(0), device=data.x.device) < float(mask_ratio)
    if not mask.any():
        mask[torch.randint(data.x.size(0), (1,), device=data.x.device)] = True

    masked_x = data.x.clone()
    masked_x[mask] = 0.0
    _, node_rep = graph_encoder(masked_x, data.edge_index, data.edge_attr, data.batch)
    mask_loss = F.cross_entropy(graph_atom_head(node_rep[mask]), atom_symbol_targets[mask])

    consistency_loss = data.x.new_tensor(0.0)
    if consistency_weight > 0 and hasattr(data, 'smiles'):
        repeat_batch = _build_repeat_unit_batch(data.smiles, data.x.device)
        repeat_graph, _ = graph_encoder(
            repeat_batch.x, repeat_batch.edge_index, repeat_batch.edge_attr, repeat_batch.batch
        )
        star_graph, _ = graph_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        consistency_loss = 1.0 - F.cosine_similarity(star_graph, repeat_graph, dim=1).mean()

    return float(mask_weight) * mask_loss + float(consistency_weight) * consistency_loss


def _geom_denoise_loss(base_model, data, geom_coord_head, noise_std, distance_weight):
    if 'geom' not in base_model.encoders:
        return data.pos3d.new_tensor(0.0)

    geom_module = base_model.encoders['geom']
    clean_pos = data.pos3d
    target_pos = clean_pos.detach().clone()
    noise = torch.randn_like(target_pos) * float(noise_std)
    try:
        data.pos3d = target_pos + noise
        node_rep, batch = geom_module.encoder.encode_nodes(data)
        pred_delta = geom_coord_head(node_rep)
        pred_pos = data.pos3d + pred_delta
        return _pairwise_distance_loss(pred_pos, target_pos, batch, weight=distance_weight)
    finally:
        data.pos3d = clean_pos

def main():
    args = parse_arguments()

    from src.dataset import UniDataset
    from src.modules import UniEncoderAttention
    from src.utils import compute_contrastive_loss, get_data_loader

    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs available")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    dataset = UniDataset(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
        geometry_encoder=args.geometry_encoder,
        graph_input=args.graph_input,
        use_feature_cache=not args.disable_feature_cache,
        feature_source_dataset=args.feature_source_dataset,
        rebuild_feature_cache=args.rebuild_feature_cache,
        max_smiles_length=args.max_smiles_length,
        max_smiles_length_cap=args.max_smiles_length_cap,
    )
    indices = np.arange(len(dataset))
    dataloader = get_data_loader(dataset, indices=indices, batch_size=args.batch_size, shuffle=True, random_conformer=True)

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=args.joint_embedding_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name=args.gnn_model_name,
        geom_model_name=args.geom_model_name,
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder,
        geometry_encoder=args.geometry_encoder,
        graph_num_layers=args.graph_num_layers,
        graph_emb_dim=args.graph_emb_dim,
        graph_dropout=args.graph_dropout,
        graph_pooling=args.graph_pooling,
    )

    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretraining checkpoint from {args.pretrained_model_path}")
        print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)
    model = model.to(device)
    base_model = _base_model(model)
    loss_weights = _stage_loss_weights(args)
    print(
        "Pretraining stage: "
        f"{args.pretrain_stage} "
        f"(contrastive={loss_weights['contrastive']}, graph={loss_weights['graph']}, geom={loss_weights['geom']})"
    )

    aux_modules = nn.ModuleList()
    graph_atom_head = None
    geom_coord_head = None
    if 'graph' in args.modalities and loss_weights['graph'] > 0:
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import allowable_features
        graph_atom_head = nn.Linear(graph_dim, len(allowable_features['possible_atom_symbols'])).to(device)
        aux_modules.append(graph_atom_head)
    if 'geom' in args.modalities and loss_weights['geom'] > 0:
        geom_dim = base_model.encoders['geom'].encoder.hidden_channels
        geom_coord_head = nn.Linear(geom_dim, 3).to(device)
        aux_modules.append(geom_coord_head)

    optimizer = optim.Adam(list(model.parameters()) + list(aux_modules.parameters()), lr=args.lr)

    # Create directory for saving loss curves and data
    os.makedirs('./plots/pretrain', exist_ok=True)

    # Record loss for each epoch
    losses = []

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Pretraining Epoch {epoch + 1}/{args.epochs}")
        for data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            base_model = _base_model(model)
            loss_terms = {}

            geom_loss = data.y.new_tensor(0.0)
            if geom_coord_head is not None:
                geom_loss = _geom_denoise_loss(
                    base_model,
                    data,
                    geom_coord_head,
                    noise_std=args.geom_noise_std,
                    distance_weight=args.geom_distance_loss_weight,
                )
                loss_terms['geom'] = geom_loss

            contrastive_loss = data.y.new_tensor(0.0)
            if loss_weights['contrastive'] > 0:
                _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
                contrastive_loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
                loss_terms['contrastive'] = contrastive_loss

            graph_loss = data.y.new_tensor(0.0)
            if graph_atom_head is not None:
                graph_loss = _graph_pretrain_loss(
                    base_model,
                    data,
                    graph_atom_head,
                    mask_ratio=args.graph_mask_ratio,
                    mask_weight=args.graph_mask_atom_weight,
                    consistency_weight=args.graph_starlink_consistency_weight,
                )
                loss_terms['graph'] = graph_loss

            loss = loss_weights['contrastive'] * contrastive_loss
            if graph_atom_head is not None:
                loss = loss + loss_weights['graph'] * graph_loss
            if geom_coord_head is not None:
                loss = loss + loss_weights['geom'] * geom_loss

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite pretraining loss for batch smiles={getattr(data, 'smiles', [])}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_modules.parameters()), max_norm=args.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                con=f"{contrastive_loss.item():.4f}",
                graph=f"{graph_loss.item():.4f}",
                geom=f"{geom_loss.item():.4f}",
            )
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f}")

    # Save original loss data
    loss_data = {
        'pretrain_stage': args.pretrain_stage,
        'epochs': list(range(1, args.epochs + 1)),
        'losses': losses
    }
    with open('./plots/pretrain/loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=4)
    print("Loss data saved at ./plots/pretrain/loss_data.json")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), losses, marker='o')
    plt.title('Pretraining Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Total Pretraining Loss')
    plt.grid(True)
    plt.savefig('./plots/pretrain/loss_curve.png')
    plt.close()
    print("Loss curve saved at ./plots/pretrain/loss_curve.png")

    # If using DataParallel, need to handle module prefix when saving
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), args.save_path)
    else:
        torch.save(model.state_dict(), args.save_path)
    print(f"Pretrained model saved at {args.save_path}")

if __name__ == "__main__":
    main()
