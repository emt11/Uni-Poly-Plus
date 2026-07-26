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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom')
SUPPORTED_FUSION_TYPES = ('self_attention_pooling', 'parallel_attention')
CROSS_TASK_AUXILIARY_MAP = {
    'egb': ('egc',),
    'egc': ('egb',),
    'eps': ('nc',),
    'nc': ('eps',),
}


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value


def collect_attention_pooling_weights(model, data_loader, device):
    """Collect final attention/gate weights over the whole test set."""
    model.eval()
    attention_batches = []
    attention_labels = None
    named_batches = {}
    named_labels = {}

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            model(batch)
            attention_weights = model.attention_visual_weights.detach().cpu().numpy()
            if attention_weights.ndim != 2:
                raise ValueError(
                    "Expected attention/gate weights with shape "
                    f"[batch_size, num_inputs], got {attention_weights.shape}"
                )
            batch_labels = list(getattr(model, 'attention_visual_labels', model.modality_list))
            if attention_labels is None:
                attention_labels = batch_labels
            elif attention_labels != batch_labels:
                raise ValueError(f"Attention labels changed across batches: {attention_labels} vs {batch_labels}")
            attention_batches.append(attention_weights)

            for name, value in getattr(model, 'fusion_visual_weights', {}).items():
                weights, labels = value
                if weights is None or weights.numel() == 0:
                    continue
                labels = list(labels)
                if name in named_labels and named_labels[name] != labels:
                    raise ValueError(f"Fusion labels changed for {name}: {named_labels[name]} vs {labels}")
                named_labels[name] = labels
                named_batches.setdefault(name, []).append(weights.detach().cpu().numpy())

    if not attention_batches:
        raise ValueError("Cannot compute attention statistics from an empty data loader.")

    all_attention = np.concatenate(attention_batches, axis=0)
    fold_attention = all_attention.mean(axis=0)
    named_means = {}
    for name, batches in named_batches.items():
        values = np.concatenate(batches, axis=0).mean(axis=0)
        named_means[name] = (values, named_labels[name])
    return fold_attention, all_attention.shape, attention_labels, named_means


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
        modalities = parse_modalities_for_plot(row.get("fusion_inputs"))
        if modalities is None:
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
    parser.add_argument('--root', default='./data', help='Data root containing raw/ and processed/.')
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc'],
        help="List of tasks to train on. Default excludes tg: eat eea egb egc ei eps nc xc"
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
        '--fusion_type',
        type=str,
        choices=SUPPORTED_FUSION_TYPES,
        default='self_attention_pooling',
        help=(
            "Fusion layer type. self_attention_pooling keeps the original modality self-attention; "
            "parallel_attention fuses SMILES, SCAGE, and FP as three peer modality tokens."
        )
    )

    parser.add_argument(
        '--fp_mode',
        type=str,
        choices=['ecfp', 'mixfp'],
        default='ecfp',
        help="Fingerprint implementation. ecfp keeps the original Morgan/ECFP 1024-bit FP; mixfp uses MACCSKeys + PubChemFingerprints.",
    )
    parser.add_argument('--parallel_attention_layers', type=int, default=1)
    parser.add_argument('--fusion_dropout', type=float, default=0.20)
    parser.add_argument('--head_dropout', type=float, default=0.25)
    parser.add_argument('--fp_bit_dropout', type=float, default=0.15)
    parser.add_argument('--fp_modality_dropout', type=float, default=0.25)
    parser.add_argument('--smiles_modality_dropout', type=float, default=0.10)
    parser.add_argument('--graph_modality_dropout', type=float, default=0.05)
    parser.add_argument(
        '--geometry_encoder',
        type=str,
        choices=['painn'],
        default='painn',
        help="Geometry encoder backend (only PaiNN is supported)."
    )
    parser.add_argument(
        '--graph_input',
        type=str,
        choices=['repeat_unit', 'star_linking'],
        default='star_linking',
        help="Graph input type. 'repeat_unit' keeps the original graph; 'star_linking' removes two attachment atoms and connects their boundary atoms for graph-only topology input."
    )
    parser.add_argument(
        '--geom_input',
        type=str,
        choices=['repeat_unit', 'periodic_pbc', 'polygen_periodic', 'screw_periodic', 'smer_context'],
        default='repeat_unit',
        help="Geometry input: repeat_unit, legacy periodic_pbc, deterministic polygen_periodic, screw_periodic, or smer_context."
    )
    parser.add_argument('--screw_kabsch_rmsd_max', type=float, default=1.5)
    parser.add_argument('--screw_rotation_consistency_deg', type=float, default=30.0)
    parser.add_argument('--screw_translation_relative_max', type=float, default=0.30)
    parser.add_argument('--screw_final_rmsd_max', type=float, default=1.5)
    parser.add_argument('--ff_gradient_rms_max', type=float, default=0.05)
    parser.add_argument('--ff_gradient_max', type=float, default=0.25)
    parser.add_argument('--ff_probe_steps', type=int, default=20)
    parser.add_argument('--ff_probe_energy_delta_per_atom_max', type=float, default=5e-5)
    parser.add_argument('--screw_energy_per_atom_max', type=float, default=5.0)
    parser.add_argument('--screw_center_gradient_rms_max', type=float, default=10.0)
    parser.add_argument(
        '--geom_model_name',
        type=str,
        default='',
        help="Pretrained geometry encoder path for PaiNN. Leave empty for random initialization."
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
    parser.add_argument('--loader_workers', type=int, default=4)
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument('--smiles_lr', type=float, default=5e-6)
    parser.add_argument('--graph_lr', type=float, default=1e-5)
    parser.add_argument('--geom_lr', type=float, default=5e-5)
    parser.add_argument('--fp_lr', type=float, default=1e-4)
    parser.add_argument('--fusion_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--freeze_smiles_epochs', type=int, default=5)
    parser.add_argument('--deep_unfreeze_epoch', type=int, default=10)
    parser.add_argument(
        '--fp_unfreeze_epoch',
        type=int,
        default=-1,
        help='Stage-3 epoch index at which to unfreeze the FP encoder; -1 keeps it frozen.',
    )
    parser.add_argument('--regression_loss', choices=['mse', 'huber'], default='huber')
    parser.add_argument('--huber_beta', type=float, default=0.5)
    parser.add_argument(
        '--unimodal_aux_weight', type=float, default=0.0,
        help=(
            'Stage-3 deep-supervision weight for independent SMILES/SCAGE/FP '
            'regression heads. Zero preserves fused-head-only training.'
        ),
    )
    parser.add_argument(
        '--cross_task_aux_weight', type=float, default=0.0,
        help=(
            'Weight for paired-property auxiliary supervision. Only labels for '
            'the target fold-training SMILES are used; held-out SMILES '
            'are explicitly excluded.'
        ),
    )
    parser.add_argument(
        '--cross_task_aux_tasks', nargs='*',
        choices=sorted(CROSS_TASK_AUXILIARY_MAP), default=[],
        help=(
            'Target tasks that receive paired-property auxiliary supervision. '
            'An empty list preserves the existing behavior and enables every '
            'mapped target when cross_task_aux_weight is positive.'
        ),
    )
    parser.add_argument('--fusion_prior_kl_weight', type=float, default=0.0)
    parser.add_argument(
        '--fusion_prior', nargs=3, type=float, default=[0.30, 0.40, 0.30],
        metavar=('SMILES', 'GRAPH', 'FP'),
        help='Target mean pooling distribution for SMILES/Graph/FP.',
    )
    parser.add_argument(
        '--swa_start_epoch', type=int, default=-1,
        help=(
            'Zero-based epoch at which to start equal-weight checkpoint averaging. '
            'Negative values disable Stage-3 SWA.'
        ),
    )
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for model, dropout, and data order.')
    parser.add_argument(
        '--fold_ids', nargs='+', type=int, default=[0, 1, 2, 3, 4],
        help='Zero-based cross-validation folds to execute. Default runs all five folds.',
    )
    parser.add_argument(
        '--refit_full_train',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'After selecting the epoch on the shared held-out validation fold, restart from the '
            'same pretrained state and fit that many epochs on the complete '
            'fold-training partition before the final held-out evaluation.'
        ),
    )
    parser.add_argument(
        '--target_transform',
        choices=['recommended', 'auto', 'standard', 'log'],
        default='recommended',
        help='Target preprocessing: recommended logs eps/nc and standardizes other current tasks; auto keeps legacy transforms.',
    )
    parser.add_argument(
        '--graph_num_layers',
        type=int,
        default=4,
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
    parser.add_argument('--graph_jk', choices=['last', 'sum', 'concat'], default='sum')
    parser.add_argument('--graph_norm', choices=['batch', 'graph', 'layer'], default='graph')
    parser.add_argument('--graph_residual', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--graph_encoder_type',
        type=str,
        choices=['gin', 'scage'],
        default='gin',
        help="Graph encoder backend: GIN or the SCAGE-route sparse MIPS-PBC encoder."
    )
    parser.add_argument(
        '--scage_dist_bar',
        nargs='+',
        type=float,
        default=[20.0, 50.0],
        help="SCAGE multi-scale distance percentiles, e.g. 20 50."
    )
    parser.add_argument(
        '--scage_num_heads',
        type=int,
        default=16,
        help="Number of attention heads for SCAGE graph encoder."
    )
    parser.add_argument('--scage_ffn_hidden_dim', type=int, default=256)
    parser.add_argument('--scage_num_kernels', type=int, default=128)
    parser.add_argument('--scage_attention_dropout', type=float, default=0.1)
    parser.add_argument(
        '--scage_use_descriptors',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Inject explicit 3D descriptor tokens into SCAGE. Disabled by default.',
    )
    parser.add_argument(
        '--scage_distance_mode',
        choices=['bias', 'mask', 'multiscale_bias', 'mips_dual'],
        default='mips_dual',
    )
    parser.add_argument('--scage_distance_rbf', type=int, default=32)
    parser.add_argument('--scage_distance_cutoff', type=float, default=12.0)
    parser.add_argument('--scage_distance_scales', nargs='+', type=float, default=[4.0, 8.0, 12.0])
    parser.add_argument('--scage_distance_taus', nargs='+', type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument(
        '--scage_topology_bias', action=argparse.BooleanOptionalAction, default=True,
        help="Add shortest-path and direct-bond attention bias to every SCAGE layer.",
    )
    parser.add_argument('--scage_topology_max_distance', type=int, default=20)
    parser.add_argument('--scage_topology_locality_mode', choices=['hard', 'soft', 'none'], default='soft')
    parser.add_argument('--scage_topology_locality_threshold', type=int, default=5)
    parser.add_argument('--scage_topology_locality_tau', type=float, default=1.0)
    parser.add_argument(
        '--scage_periodic_image_mode',
        choices=['fixed_min', 'dynamic_nearest', 'dynamic_soft', 'explicit_images'],
        default='explicit_images',
    )
    parser.add_argument('--scage_periodic_image_cap', type=int, default=1)
    parser.add_argument('--scage_periodic_image_temperature', type=float, default=0.5)
    parser.add_argument(
        '--scage_force_topology_only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Disable coordinate/PBC attention inputs for a topology-only SCAGE ablation.',
    )
    parser.add_argument(
        '--scage_use_pbc_distance',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic cell vector when constructing SCAGE minimum-image distances."
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
        '--feature_cache_workers',
        type=int,
        default=0,
        help="Number of worker processes for feature cache construction. 0 or 1 keeps serial behavior."
    )
    parser.add_argument(
        '--feature_cache_chunksize',
        type=int,
        default=4,
        help="Chunksize for multiprocessing feature cache construction."
    )
    parser.add_argument(
        '--feature_cache_partial_every',
        type=int,
        default=200,
        help="Save feature cache .partial every N successful entries. Set 0 to disable partial writes."
    )
    parser.add_argument(
        '--feature_cache_item_timeout',
        type=int,
        default=45,
        help="Hard wall-clock seconds per cache item before topology-only fallback (0 disables)."
    )
    parser.add_argument(
        '--embed_tries_multiplier',
        type=int,
        default=8,
        help="Multiplier for RDKit 3D embedding attempts per requested conformer. Total tries = CONFORMER_3D_COUNT * multiplier."
    )
    parser.add_argument(
        '--conformer_3d_count',
        type=int,
        default=8,
        help='Number of ETKDG conformer candidates before energy ranking.'
    )
    parser.add_argument(
        '--conformer_keep_count',
        type=int,
        default=4,
        help='Number of lowest-energy optimized conformers retained in the feature cache.'
    )
    parser.add_argument(
        '--conformer_profile',
        choices=['fast', 'full', 'quality'],
        default='full',
        help='Conformer search budget. Must match the profile used to build the feature cache.'
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
    if args.geom_input in {"polygen_periodic", "screw_periodic", "smer_context"} and args.graph_encoder_type != "scage":
        raise ValueError("polygen_periodic, screw_periodic and smer_context require graph_encoder_type=scage")
    if args.graph_encoder_type == 'scage':
        fixed = {
            'graph_num_layers': (args.graph_num_layers, 6),
            'graph_emb_dim': (args.graph_emb_dim, 512),
            'scage_num_heads': (args.scage_num_heads, 8),
            'scage_ffn_hidden_dim': (args.scage_ffn_hidden_dim, 2048),
            'scage_num_kernels': (args.scage_num_kernels, 128),
            'scage_distance_rbf': (args.scage_distance_rbf, 64),
            'scage_distance_cutoff': (args.scage_distance_cutoff, 12.0),
        }
        mismatched = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in fixed.items()
            if actual != expected
        ]
        if (
            args.graph_input != 'star_linking'
            or args.geom_input != 'polygen_periodic'
            or args.scage_distance_mode != 'bias'
            or args.scage_use_descriptors
            or mismatched
        ):
            raise ValueError(
                "SCAGE is fixed to sparse MIPS-PBC LGA with star_linking, "
                "polygen_periodic, distance_mode=bias, no descriptors; "
                + ", ".join(mismatched)
            )
    if args.fusion_type == 'parallel_attention':
        if (
            args.graph_encoder_type != 'scage'
            or set(args.modalities) != {'smiles', 'graph', 'fp'}
        ):
            raise ValueError(
                "parallel_attention requires graph_encoder_type=scage "
                "and modalities=smiles graph fp"
            )

    from src.dataset.geom_data import set_screw_quality_config
    set_screw_quality_config(
        gradient_rms_max=args.ff_gradient_rms_max,
        gradient_max=args.ff_gradient_max,
        probe_steps=args.ff_probe_steps,
        probe_energy_delta_per_atom_max=args.ff_probe_energy_delta_per_atom_max,
        kabsch_rmsd_max=args.screw_kabsch_rmsd_max,
        rotation_consistency_deg=args.screw_rotation_consistency_deg,
        translation_relative_max=args.screw_translation_relative_max,
        final_rmsd_max=args.screw_final_rmsd_max,
        screw_energy_per_atom_max=args.screw_energy_per_atom_max,
        screw_center_gradient_rms_max=args.screw_center_gradient_rms_max,
    )

    from src.dataset import UniDataset
    from src.modules import UniEncoderAttention
    from src.utils import (
        TargetScaler, fit_fixed_epochs, get_data_loader, scale_targets,
        set_global_seed, test_model, train_and_evaluate,
    )
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
    dataset_task_list = list(task_list)
    selected_cross_task_aux = set(args.cross_task_aux_tasks)
    if float(args.cross_task_aux_weight) > 0:
        for task in task_list:
            if selected_cross_task_aux and task not in selected_cross_task_aux:
                continue
            for auxiliary_task in CROSS_TASK_AUXILIARY_MAP.get(task, ()):
                if auxiliary_task not in dataset_task_list:
                    dataset_task_list.append(auxiliary_task)
    dataset_name_list = ['smi_' + task for task in dataset_task_list]
    dataset_list = [
        UniDataset(
            root=args.root,
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            geometry_encoder=args.geometry_encoder,
            graph_encoder_type=args.graph_encoder_type,
            graph_input=args.graph_input,
            geom_input=args.geom_input,
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
            embed_tries_multiplier=args.embed_tries_multiplier,
            conformer_3d_count=args.conformer_3d_count,
            conformer_keep_count=args.conformer_keep_count,
            conformer_profile=args.conformer_profile,
            scage_distance_mode=args.scage_distance_mode,
            scage_distance_rbf=args.scage_distance_rbf,
            scage_distance_cutoff=args.scage_distance_cutoff,
        )
        for dataset_name in dataset_name_list
    ]
    dataset_by_task = dict(zip(dataset_task_list, dataset_list))
    raw_targets_by_task = {
        task: np.array([data.y.item() for data in dataset], dtype=np.float64)
        for task, dataset in dataset_by_task.items()
    }
    raw_target_maps = {}
    for task, dataset in dataset_by_task.items():
        target_map = {}
        for data, value in zip(dataset, raw_targets_by_task[task]):
            smiles = str(data.smiles)
            if smiles in target_map and not np.isclose(target_map[smiles], value):
                raise ValueError(f'Conflicting duplicate labels for {task}: {smiles}')
            target_map[smiles] = float(value)
        raw_target_maps[task] = target_map

    def configure_cross_task_targets(task, dataset, training_indices, scope):
        auxiliary_tasks = (
            CROSS_TASK_AUXILIARY_MAP.get(task, ())
            if (
                float(args.cross_task_aux_weight) > 0
                and (
                    not selected_cross_task_aux
                    or task in selected_cross_task_aux
                )
            ) else ()
        )
        auxiliary_tasks = tuple(
            name for name in auxiliary_tasks if name in raw_target_maps
        )
        for data in dataset:
            data.cross_task_aux_y = torch.zeros(
                len(auxiliary_tasks), dtype=torch.float
            )
            data.cross_task_aux_mask = torch.zeros(
                len(auxiliary_tasks), dtype=torch.bool
            )
        for aux_idx, auxiliary_task in enumerate(auxiliary_tasks):
            auxiliary_map = raw_target_maps[auxiliary_task]
            matched_train = [
                int(index) for index in training_indices
                if str(dataset[int(index)].smiles) in auxiliary_map
            ]
            if not matched_train:
                continue
            auxiliary_values = np.array([
                auxiliary_map[str(dataset[index].smiles)]
                for index in matched_train
            ], dtype=np.float64)
            auxiliary_scaler = TargetScaler(
                auxiliary_task,
                StandardScaler(),
                transform_mode=args.target_transform,
            )
            auxiliary_scaler.scaler.fit(
                auxiliary_scaler._pre_transform(
                    auxiliary_values.reshape(-1, 1)
                )
            )
            scaled_values = auxiliary_scaler.transform(
                auxiliary_values.reshape(-1, 1)
            ).reshape(-1)
            for index, value in zip(matched_train, scaled_values):
                data = dataset[index]
                data.cross_task_aux_y[aux_idx] = float(value)
                data.cross_task_aux_mask[aux_idx] = True
            print(
                f'Cross-task auxiliary {task} <- {auxiliary_task}: '
                f'{len(matched_train)}/{len(training_indices)} {scope} labels; '
                'all samples outside that training partition excluded'
            )
        return auxiliary_tasks

    # Feature-cache workers must be created before this process initializes
    # CUDA. Cache workers are CPU-only and PolyGen's CPU optimizer must not
    # inherit the downstream training CUDA context.
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience

    result_file_initialized = False

    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_by_task[task]
        raw_targets = raw_targets_by_task[task]

        print("Start 5-fold Cross Validation")
        splits = KFold(n_splits=5, shuffle=True, random_state=1)
        fold_metrics = []
        fold_attention_weights = []
        fold_named_attention_weights = {}
        best_fold_val_r2 = -float('inf')
        best_model_state = None

        selected_folds = set(args.fold_ids)
        if not selected_folds or any(fold < 0 or fold >= 5 for fold in selected_folds):
            raise ValueError("--fold_ids must contain one or more values from 0 to 4")
        for fold, (train_indices, test_indices) in enumerate(splits.split(np.arange(len(dataset)))):
            if fold not in selected_folds:
                continue
            print(f"\nFold {fold + 1}")
            task_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(task))
            fold_seed = int(args.seed) + 1009 * task_offset + fold
            set_global_seed(fold_seed)
            print(f"Fold seed: {fold_seed}")
            fold_train_indices = train_indices
            val_indices = test_indices
            print("Shared 5-fold protocol: validation and test use the same held-out fold")
            print(
                f"Fold partitions: train={len(fold_train_indices)}, "
                f"validation={len(val_indices)}, test={len(test_indices)}"
            )
            scaler = scale_targets(
                dataset,
                task,
                train_indices=fold_train_indices,
                raw_targets=raw_targets,
                transform_mode=args.target_transform,
            )
            auxiliary_tasks = configure_cross_task_targets(
                task, dataset, fold_train_indices, 'fold-train'
            )

            train_loader = get_data_loader(
                dataset,
                indices=fold_train_indices,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            test_loader = get_data_loader(
                dataset,
                indices=test_indices,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            val_loader = get_data_loader(
                dataset,
                indices=val_indices,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
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
                graph_jk=args.graph_jk,
                graph_norm=args.graph_norm,
                graph_residual=args.graph_residual,
                graph_encoder_type=args.graph_encoder_type,
                scage_dist_bar=args.scage_dist_bar,
                scage_num_heads=args.scage_num_heads,
                scage_ffn_hidden_dim=args.scage_ffn_hidden_dim,
                scage_num_kernels=args.scage_num_kernels,
                scage_attention_dropout=args.scage_attention_dropout,
                scage_use_pbc_distance=args.scage_use_pbc_distance,
                scage_use_descriptors=args.scage_use_descriptors,
                scage_distance_mode=args.scage_distance_mode,
                scage_distance_rbf=args.scage_distance_rbf,
                scage_distance_cutoff=args.scage_distance_cutoff,
                scage_distance_scales=args.scage_distance_scales,
                scage_distance_taus=args.scage_distance_taus,
                scage_topology_bias=args.scage_topology_bias,
                scage_topology_max_distance=args.scage_topology_max_distance,
                scage_topology_locality_mode=args.scage_topology_locality_mode,
                scage_topology_locality_threshold=args.scage_topology_locality_threshold,
                scage_topology_locality_tau=args.scage_topology_locality_tau,
                scage_periodic_image_mode=args.scage_periodic_image_mode,
                scage_periodic_image_cap=args.scage_periodic_image_cap,
                scage_periodic_image_temperature=args.scage_periodic_image_temperature,
                scage_force_topology_only=args.scage_force_topology_only,
                fusion_type=args.fusion_type,
                fp_mode=args.fp_mode,
                parallel_attention_layers=args.parallel_attention_layers,
                fusion_dropout=args.fusion_dropout,
                head_dropout=args.head_dropout,
                unimodal_auxiliary=args.unimodal_aux_weight > 0,
                cross_task_auxiliary_tasks=auxiliary_tasks,
                fp_bit_dropout=args.fp_bit_dropout,
                modality_dropout={
                    'fp': args.fp_modality_dropout,
                    'smiles': args.smiles_modality_dropout,
                    'graph': args.graph_modality_dropout,
                },
            )

            if pretrained_model_path:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                if args.graph_encoder_type == 'scage':
                    expected_schema = 'scage-mips-pbc-pyg-v1'
                    if not isinstance(checkpoint, dict) or checkpoint.get('meta', {}).get('schema') != expected_schema:
                        raise RuntimeError(
                            f"SCAGE requires a {expected_schema} alignment checkpoint. "
                            "Rerun both pretraining stages."
                        )
                    checkpoint_stage = checkpoint.get('meta', {}).get('stage')
                    if checkpoint_stage != 'alignment':
                        raise RuntimeError(
                            "SCAGE downstream training requires a Stage 2 alignment checkpoint; "
                            f"received stage={checkpoint_stage!r}."
                        )
                    checkpoint_fusion = checkpoint.get('meta', {}).get('fusion_type')
                    if checkpoint_fusion != 'parallel_attention':
                        raise RuntimeError(
                            "SCAGE downstream training requires a parallel_attention alignment checkpoint; "
                            f"received fusion_type={checkpoint_fusion!r}. Rerun Stage 2."
                        )
                    checkpoint_state = checkpoint['state_dict']
                else:
                    checkpoint_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
                if args.graph_encoder_type == 'scage':
                    allowed_unexpected = {
                        'alignment_mask_head.weight',
                        'alignment_mask_head.bias',
                    }
                    # Stage 3-only auxiliary heads are intentionally absent from
                    # the Stage 2 alignment checkpoint and start from a fresh
                    # initialization. Keep every backbone/fusion key strict.
                    allowed_missing = {
                        key for key in missing
                        if key.startswith('mlp.') or (
                            float(args.unimodal_aux_weight) > 0.0
                            and key.startswith('modality_heads.')
                        ) or (
                            float(args.cross_task_aux_weight) > 0.0
                            and key.startswith('cross_task_aux_heads.')
                        )
                    }
                    incompatible = [
                        key for key in missing if key not in allowed_missing
                    ] + [
                        key for key in unexpected if key not in allowed_unexpected
                    ]
                    if incompatible:
                        raise RuntimeError(
                            f"{args.graph_encoder_type.upper()} alignment checkpoint mismatch. "
                            "Re-run both pretraining stages "
                            "with the same run.sh model configuration. Mismatched keys: "
                            + ", ".join(incompatible[:10])
                        )
                    unexpected = [
                        key for key in unexpected if key not in allowed_unexpected
                    ]
                print(f"Loaded pretrained model from {pretrained_model_path}")
                print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
            initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            model.to(device)
            print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

            metrics = train_and_evaluate(
                model, scaler, train_loader, val_loader, test_loader,
                device, num_epochs=epochs, patience=patience, max_grad_norm=args.max_grad_norm,
                smiles_lr=args.smiles_lr, graph_lr=args.graph_lr,
                geom_lr=args.geom_lr, fp_lr=args.fp_lr, fusion_lr=args.fusion_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                freeze_smiles_epochs=args.freeze_smiles_epochs,
                deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                regression_loss=args.regression_loss,
                huber_beta=args.huber_beta,
                unimodal_aux_weight=args.unimodal_aux_weight,
                fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                fusion_prior=args.fusion_prior,
                cross_task_aux_weight=args.cross_task_aux_weight,
                swa_start_epoch=args.swa_start_epoch,
                evaluate_test=not args.refit_full_train,
            )
            if args.refit_full_train:
                refit_epochs = int(metrics.get('best_epoch', -1))
                if refit_epochs <= 0:
                    raise RuntimeError(
                        'Full-train refit requires a raw validation-selected '
                        'best_epoch; disable SWA or refit_full_train.'
                    )
                print(
                    f'Refitting fold {fold + 1} from the initial checkpoint on '
                    f'all {len(train_indices)} outer-train samples for '
                    f'{refit_epochs} selected epoch(s).'
                )
                refit_scaler = scale_targets(
                    dataset,
                    task,
                    train_indices=train_indices,
                    raw_targets=raw_targets,
                    transform_mode=args.target_transform,
                )
                auxiliary_tasks = configure_cross_task_targets(
                    task, dataset, train_indices, 'outer-train refit'
                )
                refit_train_loader = get_data_loader(
                    dataset,
                    indices=train_indices,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                test_loader = get_data_loader(
                    dataset,
                    indices=test_indices,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                set_global_seed(fold_seed)
                model.load_state_dict(initial_model_state)
                model.to(device)
                refit_details = fit_fixed_epochs(
                    model,
                    refit_train_loader,
                    device,
                    num_epochs=refit_epochs,
                    max_grad_norm=args.max_grad_norm,
                    smiles_lr=args.smiles_lr,
                    graph_lr=args.graph_lr,
                    geom_lr=args.geom_lr,
                    fp_lr=args.fp_lr,
                    fusion_lr=args.fusion_lr,
                    head_lr=args.head_lr,
                    weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs,
                    freeze_smiles_epochs=args.freeze_smiles_epochs,
                    deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                    fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                    regression_loss=args.regression_loss,
                    huber_beta=args.huber_beta,
                    unimodal_aux_weight=args.unimodal_aux_weight,
                    fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                    fusion_prior=args.fusion_prior,
                    cross_task_aux_weight=args.cross_task_aux_weight,
                )
                refit_test_metrics = test_model(
                    model, test_loader, refit_scaler, device
                )
                metrics.update(refit_test_metrics)
                metrics.update(refit_details)
                metrics['refit_full_train'] = True
            else:
                metrics['refit_full_train'] = False
                metrics['refit_epochs'] = 0
            fold_attention, attention_shape, attention_labels, fold_named_attention = collect_attention_pooling_weights(model, test_loader, device)
            fold_attention_weights.append(fold_attention)
            for name, (weights, labels) in fold_named_attention.items():
                fold_named_attention_weights.setdefault(name, {'labels': labels, 'weights': []})
                fold_named_attention_weights[name]['weights'].append(weights)

            fold_metrics.append(metrics)
            print(
                f"Fold {fold + 1} Test R2: {metrics['test_r2']:.3f}, "
                f"MAE: {metrics['test_mae']:.3f}, RMSE: {metrics['test_rmse']:.3f}"
            )
            print(f"Fold {fold + 1} attention shape: {attention_shape}")
            print(f"Fold {fold + 1} mean attention: {format_attention_weights(attention_labels, fold_attention)}")

            if best_model_state is None or metrics['best_val_r2'] > best_fold_val_r2:
                best_fold_val_r2 = metrics['best_val_r2']
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            torch.cuda.empty_cache()

        cv_attention = np.mean(np.stack(fold_attention_weights, axis=0), axis=0)

        avg_test_r2 = np.mean([metric['test_r2'] for metric in fold_metrics])
        std_test_r2 = np.std([metric['test_r2'] for metric in fold_metrics])
        avg_test_mae = np.mean([metric['test_mae'] for metric in fold_metrics])
        std_test_mae = np.std([metric['test_mae'] for metric in fold_metrics])
        avg_test_rmse = np.mean([metric['test_rmse'] for metric in fold_metrics])
        std_test_rmse = np.std([metric['test_rmse'] for metric in fold_metrics])
        avg_val_r2 = np.mean([metric['best_val_r2'] for metric in fold_metrics])
        std_val_r2 = np.std([metric['best_val_r2'] for metric in fold_metrics])

        print("\nAverage of metrics over all folds")
        print(f"Test R2 = {avg_test_r2:.3f}")
        print(f"Test MAE = {avg_test_mae:.3f}")
        print(f"Test RMSE = {avg_test_rmse:.3f}")
        print(f"Standard Deviation of Test R2 = {std_test_r2:.3f}")
        print(f"Standard Deviation of Test MAE = {std_test_mae:.3f}")
        print(f"Standard Deviation of Test RMSE = {std_test_rmse:.3f}")
        print(f"Best Validation R2 = {avg_val_r2:.3f} +/- {std_val_r2:.3f}")
        print(f"5-fold mean attention: {format_attention_weights(attention_labels, cv_attention)}")

        # Downstream model persistence is intentionally disabled. The best
        # fold state remains in memory for evaluation, but saved_models is not
        # populated after training.
        # os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)
        # torch.save(best_model_state, os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth'))
        # print(f"Best fold model saved by validation R2: {best_fold_val_r2:.3f}")

        named_attention_text = {}
        for name, payload in fold_named_attention_weights.items():
            if payload['weights']:
                weights = np.mean(np.stack(payload['weights'], axis=0), axis=0)
                named_attention_text[name] = format_attention_weights(payload['labels'], weights)

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            'model_modality_list': model_modality_list,
            'fusion_type': args.fusion_type,
            'fp_mode': args.fp_mode,
            'fp_dim': 1024 if args.fp_mode == 'ecfp' else 1048,
            'parallel_attention_layers': args.parallel_attention_layers,
            'fusion_dropout': args.fusion_dropout,
            'head_dropout': args.head_dropout,
            'fp_bit_dropout': args.fp_bit_dropout,
            'modality_dropout': (
                f"smiles={args.smiles_modality_dropout};graph={args.graph_modality_dropout};"
                f"fp={args.fp_modality_dropout}"
            ),
            'regression_loss': args.regression_loss,
            'huber_beta': args.huber_beta,
            'unimodal_aux_weight': args.unimodal_aux_weight,
            'cross_task_aux_weight': args.cross_task_aux_weight,
            'cross_task_auxiliary_tasks': (
                ';'.join(CROSS_TASK_AUXILIARY_MAP.get(task, ()))
                if (
                    args.cross_task_aux_weight > 0
                    and (
                        not selected_cross_task_aux
                        or task in selected_cross_task_aux
                    )
                ) else ''
            ),
            'cross_task_aux_target_tasks': ';'.join(
                args.cross_task_aux_tasks
            ),
            'fusion_prior_kl_weight': args.fusion_prior_kl_weight,
            'fusion_prior': args.fusion_prior,
            'swa_start_epoch': args.swa_start_epoch,
            'swa_selected_folds': sum(
                int(metric.get('swa_selected', False)) for metric in fold_metrics
            ),
            'swa_mean_snapshots': np.mean([
                metric.get('swa_snapshots', 0) for metric in fold_metrics
            ]),
            'seed': args.seed,
            'target_transform': args.target_transform,
            'fold_validation_protocol': 'shared_validation_test_fold',
            'refit_full_train': bool(args.refit_full_train),
            'avg_refit_epochs': np.mean([
                metric.get('refit_epochs', 0) for metric in fold_metrics
            ]),
            'avg_best_val_r2': f"{avg_val_r2:.3f}",
            'std_best_val_r2': f"{std_val_r2:.3f}",
            'optimizer_lrs': (
                f"smiles={args.smiles_lr};graph={args.graph_lr};"
                f"fp={'frozen' if args.fp_unfreeze_epoch < 0 else args.fp_lr};"
                f"fusion={args.fusion_lr};head={args.head_lr}"
            ),
            'fp_unfreeze_epoch': args.fp_unfreeze_epoch,
            'batch_size': args.batch_size,
            'fusion_inputs': attention_labels,
            'graph_input': args.graph_input,
            'geom_input': args.geom_input,
            'graph_encoder_type': args.graph_encoder_type,
            'scage_dist_bar': args.scage_dist_bar,
            'scage_backbone': (
                'sparse_mips_pyg_lga5_spd_path_pbc_distance'
                if args.graph_encoder_type == 'scage' else None
            ),
            'scage_input': (
                '9categorical_3continuous_binary_backbone'
                if args.graph_encoder_type == 'scage' else None
            ),
            'scage_checkpoint_schema': (
                'scage-mips-pbc-pyg-v1'
                if args.graph_encoder_type == 'scage' else None
            ),
            'scage_num_heads': args.scage_num_heads if args.graph_encoder_type == 'scage' else None,
            'scage_ffn_hidden_dim': args.scage_ffn_hidden_dim if args.graph_encoder_type == 'scage' else None,
            'scage_num_kernels': args.scage_num_kernels if args.graph_encoder_type == 'scage' else None,
            'scage_use_descriptors': (
                bool(args.scage_use_descriptors) if args.graph_encoder_type == 'scage' else None
            ),
            'scage_use_pbc_distance': (
                bool(args.scage_use_pbc_distance) if args.graph_encoder_type == 'scage' else None
            ),
            'scage_force_topology_only': (
                bool(args.scage_force_topology_only) if args.graph_encoder_type == 'scage' else None
            ),
            'scage_periodic_image_mode': (
                args.scage_periodic_image_mode if args.graph_encoder_type == 'scage' else None
            ),
            'scage_distance_mode': args.scage_distance_mode if args.graph_encoder_type == 'scage' else None,
            'scage_distance_rbf': args.scage_distance_rbf if args.graph_encoder_type == 'scage' else None,
            'scage_distance_cutoff': args.scage_distance_cutoff if args.graph_encoder_type == 'scage' else None,
            'scage_distance_scales': args.scage_distance_scales if args.graph_encoder_type == 'scage' else None,
            'scage_distance_taus': args.scage_distance_taus if args.graph_encoder_type == 'scage' else None,
            'scage_topology_bias': args.scage_topology_bias if args.graph_encoder_type == 'scage' else None,
            'scage_topology_max_distance': (
                args.scage_topology_max_distance if args.graph_encoder_type == 'scage' else None
            ),
            'scage_topology_locality_mode': (
                args.scage_topology_locality_mode if args.graph_encoder_type == 'scage' else None
            ),
            'scage_topology_locality_threshold': (
                args.scage_topology_locality_threshold if args.graph_encoder_type == 'scage' else None
            ),
            'avg_test_r2': f"{avg_test_r2:.3f}",
            'std_test_r2': f"{std_test_r2:.3f}",
            'avg_test_mae': f"{avg_test_mae:.3f}",
            'std_test_mae': f"{std_test_mae:.3f}",
            'avg_test_rmse': f"{avg_test_rmse:.3f}",
            'std_test_rmse': f"{std_test_rmse:.3f}",
            'attention': format_attention_weights(attention_labels, cv_attention),
            'flat_attention_weights': named_attention_text.get('flat', ''),
            'parallel_attention_weights': named_attention_text.get('parallel', ''),
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
