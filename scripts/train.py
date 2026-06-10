import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import scale_targets, train_and_evaluate, get_data_loader


def collect_attention_pooling_weights(model, data_loader, device):
    """Collect final attention pooling weights over the whole test set."""
    model.eval()
    attention_batches = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            model(batch)
            attention_weights = model.attention_visual_weights.detach().cpu().numpy()
            if attention_weights.ndim != 2:
                raise ValueError(
                    "Expected attention pooling weights with shape "
                    f"[batch_size, num_modalities], got {attention_weights.shape}"
                )
            attention_batches.append(attention_weights)

    if not attention_batches:
        raise ValueError("Cannot compute attention statistics from an empty data loader.")

    all_attention = np.concatenate(attention_batches, axis=0)
    fold_attention = all_attention.mean(axis=0)
    return fold_attention, all_attention.shape


def format_attention_weights(modalities, attention_weights):
    if len(modalities) != len(attention_weights):
        raise ValueError(
            f"Modalities length ({len(modalities)}) does not match attention length "
            f"({len(attention_weights)})."
        )
    return ";".join(
        f"{modality}:{float(weight):.6f}"
        for modality, weight in zip(modalities, attention_weights)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train UniEncoderAttention Model")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['tg', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc', 'eat'],
        help="List of tasks to train on. Example: --tasks tg er de"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='UniEncoderAttention',
        help="Name of the model."
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'graph', 'fp', 'geom'],
        help="List of model modalities. Example: --modalities smiles text"
    )
    parser.add_argument(
        '--geometry_encoder',
        type=str,
        choices=['painn', 'schnet'],
        default='painn',
        help="Geometry encoder backend."
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="Freeze encoders weights if set."
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default="./pretrained_models/saved_pretrained_model.pth",
        help="Path to the pretrained model."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Early stopping patience."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/results.csv',
        help="Directory to save results CSV."
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./saved_models',
        help="Directory to save trained models."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    pre_trained_model_dict = {
        'smiles_model_name': "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        'text_model_name': "./pretrained_models/encoders/multitask-text-and-chemistry-t5-base-augm",
        # 'gnn_model_name': "./pretrained_models/encoders/Mole-BERT.pth",
        'gnn_model_name': "",
        # 'geom_model_name': "./pretrained_models/encoders/schnet_qm9_cv.pth"
        'geom_model_name': ""
    }
    
    result_output_dir = args.results_dir
    model_output_dir = args.models_dir
    model_modality_list = args.modalities
    use_kg = 'kg' in model_modality_list
    
    task_list = args.tasks
    dataset_name_list = ['smi_' + task for task in task_list]
    dataset_list = [
        UniDataset(
            root='./data',
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            geometry_encoder=args.geometry_encoder,
            use_kg=use_kg
        )
        for dataset_name in dataset_name_list
    ]
    
    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience
    
    result_file_initialized = False
    
    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_list[task_list.index(task)]
        raw_targets = np.array([data.y.item() for data in dataset], dtype=np.float64)

        print("Start 5-fold Cross Validation")
        splits = KFold(n_splits=5, shuffle=True, random_state=1)
        fold_metrics = []
        fold_attention_weights = []
        best_fold_r2 = -float('inf')
        best_model_state = None

        for fold, (train_indices, test_indices) in enumerate(splits.split(np.arange(len(dataset)))):
            print(f"\nFold {fold + 1}")
            scaler = scale_targets(dataset, task, train_indices=train_indices, raw_targets=raw_targets)

            train_loader = get_data_loader(
                dataset,
                indices=train_indices,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False
            )
            test_loader = get_data_loader(
                dataset,
                indices=test_indices,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False
            )

            model = UniEncoderAttention(
                joint_embedding_dim=256,
                smiles_model_name=pre_trained_model_dict['smiles_model_name'],
                text_model_name=pre_trained_model_dict['text_model_name'],
                gnn_model_name=pre_trained_model_dict['gnn_model_name'],
                geom_model_name=pre_trained_model_dict['geom_model_name'],
                modality_list=model_modality_list,
                freeze_encoder=freeze_encoder,
                geometry_encoder=args.geometry_encoder
            )

            if pretrained_model_path:
                missing, unexpected = model.load_state_dict(torch.load(pretrained_model_path), strict=False)
                print(f"Loaded pretrained model from {pretrained_model_path}")
                print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

            model.to(device)
            print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

            metrics = train_and_evaluate(
                model, scaler, train_loader, test_loader, test_loader,
                device, num_epochs=epochs, patience=patience, max_grad_norm=args.max_grad_norm
            )
            fold_attention, attention_shape = collect_attention_pooling_weights(model, test_loader, device)
            fold_attention_weights.append(fold_attention)

            fold_metrics.append(metrics)
            print(
                f"Fold {fold + 1} Test R2: {metrics['test_r2']:.3f}, "
                f"MAE: {metrics['test_mae']:.3f}, RMSE: {metrics['test_rmse']:.3f}"
            )
            print(f"Fold {fold + 1} attention shape: {attention_shape}")
            print(f"Fold {fold + 1} mean attention: {format_attention_weights(model_modality_list, fold_attention)}")

            if metrics['test_r2'] > best_fold_r2:
                best_fold_r2 = metrics['test_r2']
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            torch.cuda.empty_cache()

        cv_attention = np.mean(np.stack(fold_attention_weights, axis=0), axis=0)

        avg_test_r2 = np.mean([metric['test_r2'] for metric in fold_metrics])
        std_test_r2 = np.std([metric['test_r2'] for metric in fold_metrics])
        avg_test_mae = np.mean([metric['test_mae'] for metric in fold_metrics])
        std_test_mae = np.std([metric['test_mae'] for metric in fold_metrics])
        avg_test_rmse = np.mean([metric['test_rmse'] for metric in fold_metrics])
        std_test_rmse = np.std([metric['test_rmse'] for metric in fold_metrics])

        print("\nAverage of metrics over all folds")
        print(f"Test R2 = {avg_test_r2:.3f}")
        print(f"Test MAE = {avg_test_mae:.3f}")
        print(f"Test RMSE = {avg_test_rmse:.3f}")
        print(f"Standard Deviation of Test R2 = {std_test_r2:.3f}")
        print(f"Standard Deviation of Test MAE = {std_test_mae:.3f}")
        print(f"Standard Deviation of Test RMSE = {std_test_rmse:.3f}")
        print(f"5-fold mean attention: {format_attention_weights(model_modality_list, cv_attention)}")

        os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)
        torch.save(best_model_state, os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth'))
        print(f"Best fold model saved with R2: {best_fold_r2:.3f}")

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            'model_modality_list': model_modality_list,
            'avg_test_r2': f"{avg_test_r2:.3f}",
            'std_test_r2': f"{std_test_r2:.3f}",
            'avg_test_mae': f"{avg_test_mae:.3f}",
            'std_test_mae': f"{std_test_mae:.3f}",
            'avg_test_rmse': f"{avg_test_rmse:.3f}",
            'std_test_rmse': f"{std_test_rmse:.3f}",
            'attention': format_attention_weights(model_modality_list, cv_attention)
        }

        # Save to CSV
        os.makedirs(os.path.dirname(result_output_dir), exist_ok=True)
        results_df = pd.DataFrame([result])
        write_header = not result_file_initialized
        results_df.to_csv(
            result_output_dir,
            mode='w' if write_header else 'a',
            header=write_header,
            index=False
        )
        result_file_initialized = True
        print(f"Results have been appended to '{result_output_dir}'.")
        

if __name__ == "__main__":
    main()
