import os
import sys
import argparse
import ast
import json
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

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


def parse_modalities_for_plot(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if pd.isna(value):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw_value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]

    return None


def parse_attention_for_plot(value, modalities=None):
    if pd.isna(value):
        raise ValueError("Missing attention value.")

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Empty attention value.")

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw_value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

        if isinstance(parsed, dict):
            return {str(key): float(weight) for key, weight in parsed.items()}
        if isinstance(parsed, (list, tuple)):
            weights = np.asarray(parsed, dtype=float)
            if weights.ndim == 2:
                weights = weights.mean(axis=0)
            if weights.ndim != 1:
                raise ValueError(f"Unsupported attention array shape: {weights.shape}")
            if modalities is None:
                modalities = list(SUPPORTED_MODALITIES[:len(weights)])
            if len(modalities) != len(weights):
                raise ValueError(
                    f"Modalities length ({len(modalities)}) does not match attention length "
                    f"({len(weights)})."
                )
            return {modality: float(weight) for modality, weight in zip(modalities, weights)}

    if ":" in raw_value:
        parsed = {}
        for item in raw_value.split(";"):
            item = item.strip()
            if not item:
                continue
            name, raw_weight = item.split(":", 1)
            parsed[name.strip()] = float(raw_weight.strip())
        return parsed

    raise ValueError(f"Could not parse attention value: {raw_value}")


def build_attention_matrix(results_df):
    attention_column = "attention" if "attention" in results_df.columns else "attention_weights"
    if attention_column not in results_df.columns:
        raise ValueError("Results CSV must contain an 'attention' or 'attention_weights' column.")
    if "task" not in results_df.columns:
        raise ValueError("Results CSV must contain a 'task' column.")

    rows = []
    modality_order = []
    for _, row in results_df.iterrows():
        modalities = parse_modalities_for_plot(row.get("model_modality_list"))
        attention = parse_attention_for_plot(row[attention_column], modalities=modalities)
        rows.append((row["task"], attention))
        for modality in attention:
            if modality not in modality_order:
                modality_order.append(modality)

    matrix = pd.DataFrame(
        [
            [attention.get(modality, np.nan) for modality in modality_order]
            for _, attention in rows
        ],
        index=[task for task, _ in rows],
        columns=modality_order,
    )
    matrix.index.name = "task"
    return matrix


def default_attention_heatmap_path(results_csv_path):
    results_path = Path(results_csv_path)
    return results_path.with_name(f"{results_path.stem}_attention_heatmap.png")


def plot_attention_heatmap_from_results(results_csv_path, output_path=None):
    results_csv_path = Path(results_csv_path)
    if output_path is None:
        output_path = default_attention_heatmap_path(results_csv_path)
    else:
        output_path = Path(output_path)

    results_df = pd.read_csv(results_csv_path)
    matrix = build_attention_matrix(results_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height = max(4.0, 0.45 * len(matrix.index) + 1.5)
    width = max(5.0, 0.9 * len(matrix.columns) + 2.0)
    plt.figure(figsize=(width, height))
    sns.set_theme(style="white", font_scale=0.95)
    ax = sns.heatmap(
        matrix,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "5-fold mean attention"},
    )
    ax.set_xlabel("modalities")
    ax.set_ylabel("task")
    ax.set_title("Attention Pooling Weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


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
        type=parse_modality,
        default=['smiles', 'graph', 'fp', 'geom'],
        help="Model modalities. Supported: smiles, graph, fp, geom."
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
        '--geom_model_name',
        type=str,
        default='',
        help="Pretrained geometry encoder path for SchNet or PaiNN. Leave empty for random initialization."
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
        default='smi_all',
        help="Dataset used to build the SMILES-level feature cache (default: smi_all)."
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
        '--max_smiles_length_cap',
        type=int,
        default=256,
        help="Cap for auto-computed max SMILES token length (default: 256)."
    )
    parser.add_argument(
        '--attention_heatmap_path',
        type=str,
        default=None,
        help="Optional output path for attention heatmap. Defaults to <results_dir stem>_attention_heatmap.png."
    )
    parser.add_argument(
        '--disable_attention_heatmap',
        action='store_true',
        help="Disable automatic attention heatmap generation after writing training results."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    from src.dataset import UniDataset
    from src.modules import UniEncoderAttention
    from src.utils import get_data_loader, scale_targets, train_and_evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    pre_trained_model_dict = {
        'smiles_model_name': "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        'gnn_model_name': "",
        'geom_model_name': args.geom_model_name
    }

    result_output_dir = args.results_dir
    model_output_dir = args.models_dir
    model_modality_list = args.modalities

    task_list = args.tasks
    dataset_name_list = ['smi_' + task for task in task_list]
    dataset_list = [
        UniDataset(
            root='./data',
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            geometry_encoder=args.geometry_encoder,
            graph_input=args.graph_input,
            use_feature_cache=not args.disable_feature_cache,
            feature_source_dataset=args.feature_source_dataset,
            rebuild_feature_cache=args.rebuild_feature_cache,
            max_smiles_length=args.max_smiles_length,
            max_smiles_length_cap=args.max_smiles_length_cap,
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
                joint_embedding_dim=args.joint_embedding_dim,
                smiles_model_name=pre_trained_model_dict['smiles_model_name'],
                gnn_model_name=pre_trained_model_dict['gnn_model_name'],
                geom_model_name=pre_trained_model_dict['geom_model_name'],
                modality_list=model_modality_list,
                freeze_encoder=freeze_encoder,
                geometry_encoder=args.geometry_encoder,
                graph_num_layers=args.graph_num_layers,
                graph_emb_dim=args.graph_emb_dim,
                graph_dropout=args.graph_dropout,
                graph_pooling=args.graph_pooling,
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
            'graph_input': args.graph_input,
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


    if not args.disable_attention_heatmap and result_file_initialized:
        try:
            heatmap_path = plot_attention_heatmap_from_results(
                result_output_dir,
                output_path=args.attention_heatmap_path,
            )
            print(f"Attention heatmap saved to '{heatmap_path}'.")
        except Exception as exc:
            print(f"Warning: failed to generate attention heatmap: {exc}")


if __name__ == "__main__":
    main()
