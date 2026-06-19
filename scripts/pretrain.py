import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom', 'kg')


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


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
        help="Modalities to use. Supported: smiles, graph, fp, geom, kg."
    )
    parser.add_argument(
        '--geometry_encoder',
        type=str,
        choices=['painn', 'schnet'],
        default='painn',
        help="Geometry encoder backend."
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
        # default="./pretrained_models/encoders/Mole-BERT.pth",
        default="",
        help="Pretrained GNN model path"
    )
    parser.add_argument(
        '--geom_model_name',
        type=str,
        default="",
        # default="./pretrained_models/encoders/schnet_qm9_gap.pth",
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
    parser.add_argument("--kg_embedding_path", default="kg_work/features/kg_embedding.npy")
    parser.add_argument("--kg_mapping_path", default="kg_work/features/kg_entity_mapping.csv")
    parser.add_argument("--kg_embedding_dim", type=int, default=128)
    parser.add_argument("--kg_projection_dim", type=int, default=256)
    parser.add_argument("--kg_freeze_embedding", type=parse_bool, default=True, help="Freeze KG embedding table when true; fine-tune when false")
    return parser.parse_args()

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
        enable_kg="kg" in args.modalities,
        kg_mapping_path=args.kg_mapping_path,
        kg_embedding_path=args.kg_embedding_path
    )
    indices = np.arange(len(dataset))
    dataloader = get_data_loader(dataset, indices=indices, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=args.kg_projection_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name=args.gnn_model_name,
        geom_model_name=args.geom_model_name,
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder,
        geometry_encoder=args.geometry_encoder,
        kg_embedding_path=args.kg_embedding_path if "kg" in args.modalities else None,
        kg_embedding_dim=args.kg_embedding_dim,
        kg_freeze_embedding=args.kg_freeze_embedding
    )
    
    # Use all available GPUs for data parallel training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
            loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}] Contrastive Loss: {avg_loss:.4f}")

    # Save original loss data
    loss_data = {
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
    plt.ylabel('Contrastive Loss')
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
