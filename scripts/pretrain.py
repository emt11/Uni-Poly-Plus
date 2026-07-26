import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import json
import hashlib
import math
from datetime import timedelta
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom')

# Periodic augmentation SMILES are deterministic cache fields, but their graph
# features were previously rebuilt with RDKit on every epoch. Keep immutable
# CPU templates per process and clone them before adding batch-specific fields.
_PERIODIC_AUG_GRAPH_CACHE = {}


def _cached_periodic_aug_graph(smiles, graph_input):
    from src.dataset.graph_data import build_graph_for_input

    key = (str(graph_input), str(smiles))
    template = _PERIODIC_AUG_GRAPH_CACHE.get(key)
    cache_hit = template is not None
    if template is None:
        template = build_graph_for_input(str(smiles), graph_input)
        _PERIODIC_AUG_GRAPH_CACHE[key] = template
    return template.clone(), cache_hit


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
        '--fusion_type',
        type=str,
        choices=['self_attention_pooling', 'parallel_attention'],
        default='self_attention_pooling',
        help="Fusion type. parallel_attention treats SMILES, SCAGE, and FP as peer modality tokens.",
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
        choices=['scage_m4p', 'graph_geom', 'alignment', 'joint'],
        default='joint',
        help=(
            "Pretraining stage. scage_m4p trains the Polymer-SCAGE M4P tasks; "
            "graph_geom trains legacy graph/geometry auxiliary tasks only; "
            "alignment trains SCAGE-SMILES and SCAGE-FP alignment on the SCAGE mainline; "
            "joint keeps the previous behavior and sums all enabled losses."
        )
    )
    parser.add_argument('--pretrain_profile', choices=['legacy', 'mips24h'], default='mips24h')
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='',
        help="Optional checkpoint to initialize this stage; SCAGE alignment requires a scage_m4p Stage 1 checkpoint."
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
    parser.add_argument('--loader_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--amp_dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--max_steps', type=int, default=0, help='Optional preflight step limit; 0 disables it.')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducible pretraining.')
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
        help=(
            'Disable all coordinate/PBC attention inputs and run SCAGE with '
            'topology attention only. Intended for controlled geometry ablations.'
        ),
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
        '--cache_only',
        action='store_true',
        help=(
            "Build/load and validate the CPU feature cache, then exit before "
            "CUDA/NCCL/model initialization."
        ),
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
        help='Conformer search budget. fast uses bounded 100/300 ETKDG and 50-step force-field optimization.'
    )
    parser.add_argument(
        '--geom_denoise_weight',
        type=float,
        default=1.0,
        help="Weight for geometry coordinate denoising pretraining loss."
    )
    parser.add_argument(
        '--geom_noise_std',
        type=float,
        default=0.2,
        help="Fixed Gaussian coordinate noise std for geometry denoising pretraining."
    )
    parser.add_argument(
        '--geom_noise_std_min',
        type=float,
        default=None,
        help="Minimum Gaussian coordinate noise std for random-range geometry denoising. "
             "When set with --geom_noise_std_max, overrides fixed --geom_noise_std per batch."
    )
    parser.add_argument(
        '--geom_noise_std_max',
        type=float,
        default=None,
        help="Maximum Gaussian coordinate noise std for random-range geometry denoising. "
             "When set with --geom_noise_std_min, overrides fixed --geom_noise_std per batch."
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
        '--graph_periodic_aug_weight',
        type=float,
        default=0.5,
        help="Weight for PerioGT-style periodicity augmentation contrastive loss."
    )
    parser.add_argument(
        '--graph_shortest_path_weight',
        type=float,
        default=1.0,
        help="Weight for SCAGE-style graph shortest-path distance prediction."
    )
    parser.add_argument(
        '--graph_angle_weight',
        type=float,
        default=1.0,
        help="Weight for SCAGE-style 3D angle prediction."
    )
    parser.add_argument(
        '--scage_sp_max_distance',
        type=int,
        default=20,
        help="Maximum shortest-path class; larger distances are clipped to this bucket."
    )
    parser.add_argument(
        '--scage_angle_bins',
        type=int,
        default=20,
        help="Number of angle bins for SCAGE-style angle prediction."
    )
    parser.add_argument('--scage_mips_mask_weight', type=float, default=1.0)
    parser.add_argument('--scage_ecfp_weight', type=float, default=0.0)
    parser.add_argument('--scage_periodic_sp_weight', type=float, default=0.0)
    parser.add_argument(
        '--scage_periodic_geometry_weight', '--scage_screw_geometry_weight',
        dest='scage_screw_geometry_weight', type=float, default=0.0,
        help="Weight for target-masked PBC LGA Euclidean-distance regression.",
    )
    parser.add_argument('--scage_periodic_contrast_weight', type=float, default=0.0)
    parser.add_argument('--scage_ecfp_effective_max', type=float, default=0.20)
    parser.add_argument('--scage_sp_max_pairs', type=int, default=256)
    parser.add_argument('--scage_focal_gamma', type=float, default=2.0)
    parser.add_argument('--scage_boundary_angle_weight', type=float, default=0.5)
    parser.add_argument('--scage_boundary_torsion_weight', type=float, default=0.40)
    parser.add_argument('--scage_torsion_bins', type=int, default=12)
    parser.add_argument(
        '--scage_torsion_objective',
        choices=['circular', 'categorical'],
        default='circular',
        help=(
            "Boundary torsion objective. 'circular' regresses sin/cos and avoids "
            "bin discontinuities; 'categorical' preserves the legacy binned focal loss."
        ),
    )
    parser.add_argument('--scage_geometry_max_pairs', type=int, default=32)
    parser.add_argument('--scage_geometry_mask_ratio', type=float, default=0.15)
    parser.add_argument('--scage_geometry_distance_weight', type=float, default=0.25)
    parser.add_argument('--scage_geometry_angle_weight', type=float, default=0.35)
    parser.add_argument('--scage_geometry_screw_weight', type=float, default=0.20)
    parser.add_argument(
        '--scage_shift_balance_power', type=float, default=0.5,
        help=(
            "Inverse-frequency exponent for periodic image-shift cross entropy. "
            "Zero disables class balancing; 0.5 uses square-root balancing."
        ),
    )
    parser.add_argument('--periodic_aug_views', type=int, default=2)
    parser.add_argument('--periodic_projection_dim', type=int, default=256)
    parser.add_argument(
        '--periodic_aug_max_mrus',
        type=int,
        default=3,
        help="Maximum MRU multiplier used by periodicity augmentation."
    )
    parser.add_argument(
        '--periodic_aug_retry',
        type=int,
        default=5,
        help="Number of augmentation attempts per SMILES before skipping it."
    )
    parser.add_argument(
        '--periodic_cl_temperature',
        type=float,
        default=0.2,
        help="Temperature for graph periodicity augmentation InfoNCE loss."
    )
    parser.add_argument(
        '--graph_mask_ratio',
        type=float,
        default=0.30,
        help="Node masking ratio for graph atom-feature reconstruction."
    )
    parser.add_argument(
        '--dynamic_pretrain_loss',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use SCAGE-style dynamic multi-task loss weighting for enabled "
            "pretraining losses. Disabled by default; enable explicitly when needed."
        ),
    )
    parser.add_argument(
        '--dynamic_loss_warmup_steps', type=int, default=200,
        help='Steps used to collect and then freeze dynamic-loss baselines.',
    )
    parser.add_argument('--dynamic_loss_recent_window', type=int, default=20)
    parser.add_argument('--dynamic_loss_temperature', type=float, default=0.5)
    parser.add_argument('--pretrain_unique_smiles', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--alignment_projection_dim', type=int, default=256)
    parser.add_argument('--alignment_fp_drop', type=float, default=0.40)
    parser.add_argument('--alignment_smiles_drop', type=float, default=0.15)
    parser.add_argument('--alignment_graph_drop', type=float, default=0.10)
    parser.add_argument('--alignment_fused_weight', type=float, default=1.0)
    parser.add_argument('--alignment_graph_smiles_weight', type=float, default=0.5)
    parser.add_argument('--alignment_graph_fp_weight', type=float, default=0.25)
    parser.add_argument('--alignment_lomo_weight', type=float, default=0.25)
    parser.add_argument('--alignment_pooling_kl_weight', type=float, default=0.01)
    parser.add_argument('--alignment_fused_mask_weight', type=float, default=0.25)
    parser.add_argument('--alignment_graph_lr', type=float, default=1e-5)
    parser.add_argument('--alignment_smiles_lr', type=float, default=1e-5)
    parser.add_argument('--alignment_fp_lr', type=float, default=2e-5)
    parser.add_argument('--alignment_projection_lr', type=float, default=1e-4)
    parser.add_argument('--alignment_fusion_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.10,
        help='Fraction of optimizer updates used for linear warm-up before cosine decay.',
    )
    return parser.parse_args()

def _base_model(model):
    return model


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_gradients(modules):
    if not _distributed_enabled():
        return
    world_size = dist.get_world_size()
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _distributed_sum_mapping(values, device):
    """Sum a scalar mapping across ranks, including keys absent on some ranks."""
    if not _distributed_enabled():
        return dict(values)
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, sorted(values))
    keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
    if not keys:
        return {}
    tensor = torch.tensor(
        [float(values.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def _gradient_vector(modules):
    values = [
        parameter.grad.detach().float().flatten()
        for module in modules for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(values) if values else torch.zeros(1)


def _bf16_parity_gate(base_model, data, atom_head, args):
    modules = (base_model, atom_head)
    for module in modules:
        module.zero_grad(set_to_none=True)
    fp32_loss, _ = _mips_masked_atom_loss(
        base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0
    )
    fp32_loss.backward()
    fp32_grad = _gradient_vector(modules)
    for module in modules:
        module.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss, _ = _mips_masked_atom_loss(
            base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector(modules)
    relative_delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    for module in modules:
        module.zero_grad(set_to_none=True)
    passed = finite and relative_delta <= 0.02 and cosine >= 0.98
    return passed, {"relative_loss_delta": relative_delta, "gradient_cosine": cosine, "finite": finite}


def _unique_smiles_indices(dataset):
    seen = set()
    indices = []
    for index in range(len(dataset)):
        smiles = str(dataset[index].smiles)
        if smiles in seen:
            continue
        seen.add(smiles)
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _polymer_ecfp_pos_weight(dataset, indices, device):
    positives = torch.zeros(2048, dtype=torch.float64)
    valid_count = 0
    for index in indices:
        sample = dataset[int(index)]
        valid = getattr(sample, 'polymer_ecfp_valid', False)
        valid = bool(valid.flatten()[0].item()) if torch.is_tensor(valid) else bool(valid)
        if not valid:
            continue
        target = getattr(sample, 'polymer_ecfp_target', None)
        if target is None or target.numel() != 2048:
            continue
        positives += target.detach().cpu().double().view(-1)
        valid_count += 1
    if valid_count == 0:
        raise ValueError("No valid 2048-bit Polymer ECFP targets were found")
    negatives = float(valid_count) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp_(1.0, 20.0).float().to(device)
    return pos_weight, valid_count


def _merge_periodic_aug_stats(total, update):
    for key in (
        'attempted', 'success', 'skipped', 'same_smiles', 'valid_contrastive',
        'graph_cache_hits', 'graph_cache_misses',
    ):
        total[key] = int(total.get(key, 0)) + int(update.get(key, 0))
    return total


def _finalize_periodic_aug_stats(epoch, stats):
    attempted = int(stats.get('attempted', 0))
    success = int(stats.get('success', 0))
    return {
        'epoch': int(epoch),
        'attempted': attempted,
        'success': success,
        'skipped': int(stats.get('skipped', 0)),
        'same_smiles': int(stats.get('same_smiles', 0)),
        'valid_contrastive': int(stats.get('valid_contrastive', 0)),
        'graph_cache_hits': int(stats.get('graph_cache_hits', 0)),
        'graph_cache_misses': int(stats.get('graph_cache_misses', 0)),
        'success_rate': float(success / attempted) if attempted else None,
        'same_smiles_rate': float(stats.get('same_smiles', 0) / attempted) if attempted else None,
    }


def _stage_loss_weights(args):
    if args.pretrain_stage == 'scage_m4p':
        return {'contrastive': 0.0, 'graph': 1.0, 'geom': 0.0}
    if args.pretrain_stage == 'graph_geom':
        return {
            'contrastive': 0.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    if args.pretrain_stage == 'alignment':
        if args.graph_encoder_type == 'scage':
            return {'contrastive': 1.0, 'graph': 0.0, 'geom': 0.0}
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


class DynamicPretrainLossWeighter:
    """SCAGE-style dynamic weighting with priors applied after normalization."""

    def __init__(
        self,
        task_names,
        init_window=200,
        recent_window=20,
        temperature=1.0,
        task_priors=None,
        effective_caps=None,
        device=None,
    ):
        self.task_names = list(task_names)
        self.init_window = int(init_window)
        self.recent_window = int(recent_window)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("dynamic loss temperature must be positive")
        self.device = device
        self.task_priors = {
            name: float((task_priors or {}).get(name, 1.0)) for name in self.task_names
        }
        self.effective_caps = {
            name: float(value) for name, value in (effective_caps or {}).items()
        }
        self.step = 0
        self.loss_init = {name: [] for name in self.task_names}
        self.loss_recent = {name: [] for name in self.task_names}
        self.loss_prev_recent = {name: [] for name in self.task_names}
        self.last_weights = {name: 1.0 / max(1, len(self.task_names)) for name in self.task_names}
        self.last_normalized = {name: 0.0 for name in self.task_names}
        self.last_effective = {name: 0.0 for name in self.task_names}
        self.last_baselines = {name: None for name in self.task_names}

    def _synchronized_statistics(self, loss_terms, reference=None):
        """Return detached per-task means shared by every distributed rank."""
        if not _distributed_enabled():
            return {
                name: loss_terms[name].detach()
                for name in self.task_names if name in loss_terms
            }

        reference = (
            next(iter(loss_terms.values()))
            if loss_terms else reference
        )
        if reference is None:
            raise ValueError("A differentiable reference is required for an empty local task set")
        values = reference.new_zeros(len(self.task_names))
        counts = reference.new_zeros(len(self.task_names))
        for idx, name in enumerate(self.task_names):
            if name in loss_terms:
                values[idx] = loss_terms[name].detach()
                counts[idx] = 1.0
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return {
            name: values[idx] / counts[idx].clamp_min(1.0)
            for idx, name in enumerate(self.task_names)
            if float(counts[idx].item()) > 0.0
        }

    def _mean_or_current(self, values, current_value):
        if not values:
            return current_value.detach().clamp_min(1e-8)
        return torch.stack(values).mean().to(current_value.device).clamp_min(1e-8)

    def _push(self, values, value, max_len):
        values.append(value.detach())
        if len(values) > max_len:
            values.pop(0)

    def _apply_effective_caps(self, effective, active_names):
        """Cap final coefficients and redistribute excess among uncapped tasks."""
        effective = effective / effective.sum().clamp_min(1e-8)
        for _ in range(len(active_names)):
            capped = []
            excess = effective.new_tensor(0.0)
            for idx, name in enumerate(active_names):
                cap = self.effective_caps.get(name)
                if cap is not None and float(effective[idx].item()) > cap:
                    excess = excess + effective[idx] - cap
                    effective[idx] = cap
                    capped.append(idx)
            if float(excess.item()) <= 0:
                break
            recipients = [idx for idx in range(len(active_names)) if idx not in capped]
            if not recipients:
                break
            recipient_values = effective[recipients]
            if float(recipient_values.sum().item()) <= 0:
                effective[recipients] += excess / len(recipients)
            else:
                effective[recipients] += excess * recipient_values / recipient_values.sum()
        return effective / effective.sum().clamp_min(1e-8)

    def __call__(self, loss_terms, reference=None):
        statistics = self._synchronized_statistics(loss_terms, reference=reference)
        active_names = [name for name in self.task_names if name in statistics]
        if not active_names:
            if reference is None:
                raise ValueError("dynamic pretrain loss received no active terms")
            self.last_weights = {}
            self.last_normalized = {}
            self.last_effective = {}
            return reference * 0.0
        reference = next(iter(loss_terms.values())) if loss_terms else reference

        def local_loss(name):
            # A task can be valid on other ranks but absent from this rank's
            # batch. Its local gradient contribution is correctly zero while
            # every rank still updates identical dynamic-weight state.
            return loss_terms.get(name, reference * 0.0)

        if len(active_names) == 1:
            only_name = active_names[0]
            self.last_weights = {only_name: 1.0}
            self.last_normalized = {
                only_name: float(statistics[only_name].detach().cpu().item())
            }
            self.last_effective = {only_name: 1.0}
            self._record(statistics)
            return local_loss(only_name)

        if self.step < self.init_window:
            priors = reference.new_tensor(
                [self.task_priors[name] for name in active_names]
            )
            prior_sum = priors.sum().clamp_min(1e-8)
            effective = self._apply_effective_caps(priors / prior_sum, active_names)
            total = torch.sum(torch.stack([local_loss(name) for name in active_names]) * effective)
            self._record(statistics)
            self.last_weights = {name: 1.0 / len(active_names) for name in active_names}
            self.last_normalized = {
                name: float(statistics[name].detach().cpu().item()) for name in active_names
            }
            self.last_effective = {
                name: float(effective[idx].detach().cpu().item())
                for idx, name in enumerate(active_names)
            }
            return total

        normalized = []
        normalized_statistics = []
        trend_ratios = []
        for name in active_names:
            statistic = statistics[name]
            init_mean = self._mean_or_current(self.loss_init[name], statistic)
            recent_mean = self._mean_or_current(self.loss_recent[name], statistic)
            prev_recent_mean = self._mean_or_current(self.loss_prev_recent[name], recent_mean)
            normalized.append(local_loss(name) / init_mean)
            normalized_statistics.append(statistic / init_mean)
            trend_ratios.append(recent_mean / prev_recent_mean)

        trend_tensor = torch.stack(trend_ratios)
        weights = F.softmax(trend_tensor / self.temperature, dim=0).detach()
        normalized_tensor = torch.stack(normalized)
        priors = normalized_tensor.new_tensor([self.task_priors[name] for name in active_names])
        effective = priors * weights
        effective = self._apply_effective_caps(effective, active_names)
        total = torch.sum(normalized_tensor * effective)
        self.last_weights = {name: float(weights[idx].detach().cpu().item()) for idx, name in enumerate(active_names)}
        self.last_normalized = {
            name: float(normalized_statistics[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_effective = {
            name: float(effective[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_baselines = {
            name: float(self._mean_or_current(self.loss_init[name], statistics[name]).cpu().item())
            for name in active_names
        }
        self._record(statistics)
        return total

    def _record(self, loss_terms):
        for name in self.task_names:
            if name not in loss_terms:
                continue
            value = loss_terms[name]
            # Baselines are collected only during warm-up and then frozen.
            if self.step < self.init_window:
                self._push(self.loss_init[name], value, self.init_window)
            if self.step % self.recent_window == 0 and self.loss_recent[name]:
                self.loss_prev_recent[name] = list(self.loss_recent[name])
                self.loss_recent[name] = []
            self._push(self.loss_recent[name], value, self.recent_window)
        self.step += 1


def _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio):
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
    _, node_rep = _graph_encode_nodes(base_model, data, x_override=masked_x)
    return F.cross_entropy(graph_atom_head(node_rep[mask]), atom_symbol_targets[mask])


def _empty_periodic_aug_stats():
    return {
        'attempted': 0,
        'success': 0,
        'skipped': 0,
        'same_smiles': 0,
        'valid_contrastive': 0,
        'graph_cache_hits': 0,
        'graph_cache_misses': 0,
    }


def _graph_periodic_aug_loss(base_model, data, graph_input, temperature, max_mrus, retry):
    stats = _empty_periodic_aug_stats()
    if 'graph' not in base_model.encoders or not hasattr(data, 'smiles'):
        return data.x.new_tensor(0.0), stats

    from torch_geometric.data import Batch
    from src.dataset.graph_data import build_graph_for_input, periodicity_augment_smiles

    graph_encoder = base_model.encoders['graph'].encoder
    aug_graphs = []
    valid_indices = []
    for sample_idx, smiles in enumerate(data.smiles):
        stats['attempted'] += 1
        aug_graph = None
        aug_smiles = None
        attempts = max(1, int(retry))
        for _ in range(attempts):
            try:
                aug_smiles, _ = periodicity_augment_smiles(
                    smiles,
                    max_mrus=max_mrus,
                    return_n=True,
                )
                aug_graph = build_graph_for_input(aug_smiles, graph_input=graph_input)
                break
            except Exception:
                aug_graph = None
        if aug_graph is None:
            stats['skipped'] += 1
            continue
        if str(aug_smiles) == str(smiles):
            stats['same_smiles'] += 1
            stats['skipped'] += 1
            continue
        stats['success'] += 1
        aug_graphs.append(aug_graph)
        valid_indices.append(sample_idx)

    stats['valid_contrastive'] = len(valid_indices)
    if len(valid_indices) < 2:
        return data.x.new_tensor(0.0), stats

    # PerioGT augmentation changes the repeat-unit graph but does not provide a
    # synchronized PBC conformer.  SCAGE therefore uses topology distance for
    # both views of this auxiliary task, avoiding asymmetric geometry inputs.
    if getattr(graph_encoder, 'uses_geometry', False):
        orig_graph, _ = graph_encoder.forward_topology_only(data)
    else:
        orig_graph, _ = _graph_encode_nodes(base_model, data)
    valid_index = torch.tensor(valid_indices, dtype=torch.long, device=orig_graph.device)
    z_orig = orig_graph.index_select(0, valid_index)

    aug_batch = Batch.from_data_list(aug_graphs).to(data.x.device)
    if getattr(graph_encoder, 'uses_geometry', False):
        z_aug, _ = graph_encoder.forward_topology_only(aug_batch)
    else:
        z_aug, _ = graph_encoder(aug_batch.x, aug_batch.edge_index, aug_batch.edge_attr, aug_batch.batch)

    z_orig = F.normalize(z_orig, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z_orig, z_aug.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss, stats


def _is_scage_graph_encoder(base_model):
    return 'graph' in base_model.encoders and getattr(base_model.encoders['graph'].encoder, 'uses_geometry', False)


def _graph_encode_nodes(base_model, data, x_override=None):
    graph_encoder = base_model.encoders['graph'].encoder
    if x_override is not None and hasattr(graph_encoder, 'forward_with_x'):
        return graph_encoder.forward_with_x(data, x_override)
    if getattr(graph_encoder, 'expects_data', False):
        return graph_encoder(data)
    if getattr(graph_encoder, 'uses_geometry', False):
        return graph_encoder(data)
    x = data.x if x_override is None else x_override
    return graph_encoder(x, data.edge_index, data.edge_attr, data.batch)


def _graph_encode_geometry_pretext(base_model, data):
    graph_encoder = base_model.encoders['graph'].encoder
    if not hasattr(graph_encoder, 'forward_geometry_pretext'):
        raise ValueError("geometry pretext encoding requires a SCAGE graph encoder")
    return graph_encoder.forward_geometry_pretext(data)


def _mips_masked_atom_loss(base_model, data, prediction_head, mask_ratio, seed, epoch):
    """MIPS fused masked-atom prediction with deterministic per-polymer masks."""
    masks = []
    for graph_idx, smiles in enumerate(data.smiles):
        if hasattr(data, "graph_available") and not bool(
            data.graph_available.flatten()[graph_idx].item()
        ):
            continue
        node_indices = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not node_indices.numel():
            continue
        count = max(1, int(round(float(mask_ratio) * node_indices.numel())))
        digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{smiles}".encode("utf-8")).digest()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2 ** 63 - 1))
        selected = torch.randperm(node_indices.numel(), generator=generator)[:count]
        masks.append(node_indices[selected.to(node_indices.device)])
    if not masks:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(masks)
    x_override = data.x.clone()
    x_override[mask_indices] = 0.0
    _, node_rep = _graph_encode_nodes(base_model, data, x_override=x_override)
    # Released MIPS derives labels from the first 101 entries of its 137-wide
    # atom feature vector (100 elements plus the unknown category).
    graph_encoder = base_model.encoders['graph'].encoder
    if getattr(graph_encoder, "architecture_name", "") == "mips_periodic_pyg_sparse_lga":
        targets = data.atomic_num[mask_indices].long()
    else:
        targets = data.mips_x[mask_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[mask_indices])
    return F.cross_entropy(logits.float(), targets), int(mask_indices.numel())


def _periodic_lga_sp_loss(data, node_rep, prediction_head, max_pairs):
    valid = data.graph_available[data.batch[data.lga_edge_index[1]]].bool()
    edge_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not edge_indices.numel():
        return node_rep.new_tensor(0.0), 0
    selection = _stratified_pair_selection(
        data.lga_spd[edge_indices], max_pairs=max_pairs
    )
    edge_indices = edge_indices[selection]
    source, target = data.lga_edge_index[:, edge_indices]
    left, right = node_rep[source], node_rep[target]
    representation = torch.cat([left, right, torch.abs(left - right)], dim=-1)
    logits = prediction_head(representation)
    targets = data.lga_spd[edge_indices].long()
    return F.cross_entropy(logits.float(), targets), int(targets.numel())


def _periodic_lga_distance_loss(
    base_model, data, prediction_head, max_pairs
):
    valid = (
        data.lga_geometry_valid.bool()
        & (data.lga_spd.long() > 0)
        & data.graph_available[data.batch[data.lga_edge_index[1]]].bool()
    )
    edge_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not edge_indices.numel():
        return data.x.new_tensor(0.0), 0, None, None
    selection = _stratified_pair_selection(
        data.lga_spd[edge_indices], max_pairs=max_pairs
    )
    edge_indices = edge_indices[selection]
    geometry_mask = torch.zeros_like(data.lga_geometry_valid, dtype=torch.bool)
    geometry_mask[edge_indices] = True
    data.lga_geometry_pair_mask = geometry_mask
    try:
        graph_rep, node_rep = _graph_encode_geometry_pretext(base_model, data)
    finally:
        delattr(data, "lga_geometry_pair_mask")
    source, target = data.lga_edge_index[:, edge_indices]
    left, right = node_rep[source], node_rep[target]
    representation = torch.cat([
        left, right, torch.abs(left - right), left * right
    ], dim=-1)
    prediction = prediction_head(representation).squeeze(-1)
    targets = data.lga_pbc_distance[edge_indices].float()
    loss = F.smooth_l1_loss(prediction.float(), targets, beta=0.5)
    return loss, int(targets.numel()), graph_rep, node_rep


def _multiclass_focal_loss(logits, targets, gamma=2.0):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    ce = F.cross_entropy(logits, targets.long(), reduction='none')
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def _scage_ecfp_loss(graph_rep, data, ecfp_head, pos_weight=None):
    valid = getattr(data, 'polymer_ecfp_valid', None)
    target = getattr(data, 'polymer_ecfp_target', None)
    if valid is None or target is None:
        return graph_rep.new_tensor(0.0), 0
    valid = valid.flatten().bool()
    if not valid.any():
        return graph_rep.new_tensor(0.0), 0
    logits = ecfp_head(graph_rep[valid])
    return F.binary_cross_entropy_with_logits(
        logits, target[valid].float(), pos_weight=pos_weight
    ), int(valid.sum().item())


def _stratified_pair_selection(distances, max_pairs):
    if distances.numel() <= int(max_pairs):
        return torch.arange(distances.numel(), device=distances.device)
    buckets = [
        distances == 1,
        distances == 2,
        (distances >= 3) & (distances <= 5),
        (distances >= 6) & (distances <= 10),
        distances > 10,
    ]
    quota = max(1, int(max_pairs) // len(buckets))
    selected = []
    for mask in buckets:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > quota:
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[:quota]]
        selected.append(indices)
    selected = torch.cat(selected) if selected else distances.new_empty(0, dtype=torch.long)
    remaining = int(max_pairs) - int(selected.numel())
    if remaining > 0:
        all_indices = torch.arange(distances.numel(), device=distances.device)
        available = all_indices[~torch.isin(all_indices, selected)]
        if available.numel() > remaining:
            available = available[torch.randperm(available.numel(), device=available.device)[:remaining]]
        selected = torch.cat([selected, available])
    return selected


def _scage_periodic_sp_loss(data, node_rep, sp_head, max_distance, max_pairs, gamma):
    reps, targets = [], []
    metadata_valid = getattr(data, 'star_link_metadata_valid', None)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        if metadata_valid is not None and not bool(metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        n = int(node_idx.numel())
        if n < 2:
            continue
        local = torch.full((data.batch.numel(),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        distance = torch.full((n, n), float(max_distance), device=data.x.device)
        diagonal = torch.arange(n, device=data.x.device)
        distance[diagonal, diagonal] = 0
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        ls, ld = local[src[keep]], local[dst[keep]]
        distance[ls, ld] = 1
        for k in range(n):
            distance = torch.minimum(distance, distance[:, k:k + 1] + distance[k:k + 1, :])
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        pair_targets = distance[row, col].clamp(max=float(max_distance)).long()
        selection = _stratified_pair_selection(pair_targets, max_pairs=max_pairs)
        row, col, pair_targets = row[selection], col[selection], pair_targets[selection]
        left, right = node_rep[node_idx[row]], node_rep[node_idx[col]]
        reps.append(torch.cat([left, right, torch.abs(left - right)], dim=-1))
        targets.append(pair_targets)
    if not reps:
        return node_rep.new_tensor(0.0), 0
    reps = torch.cat(reps, dim=0)
    targets = torch.cat(targets, dim=0)
    return _multiclass_focal_loss(sp_head(reps), targets, gamma=gamma), int(targets.numel())


def _angle_value(p0, center, p2):
    v1, v2 = p0 - center, p2 - center
    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0))


def _unsigned_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    denom = torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v, w) / denom, -1.0, 1.0))


def _signed_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    if float(torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)) <= 1e-8:
        return None
    return torch.atan2(torch.dot(torch.cross(b1u, v, dim=0), w), torch.dot(v, w))


def _angle_bin(value, bins):
    return torch.clamp((value / torch.pi * int(bins)).long(), max=int(bins) - 1)


def _circular_regression_loss(prediction, target_angles):
    """Regress a periodic angle through its unit-circle representation."""
    prediction = prediction.float()
    target_angles = target_angles.to(device=prediction.device, dtype=prediction.dtype)
    target = torch.stack([target_angles.sin(), target_angles.cos()], dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = prediction / prediction_norm
    direction_loss = 1.0 - (direction * target).sum(dim=-1)
    magnitude_loss = (prediction_norm.squeeze(-1) - 1.0).pow(2)
    return direction_loss.mean() + 0.05 * magnitude_loss.mean()


def _balanced_shift_cross_entropy(logits, targets, balance_power):
    """Balance signed periodic-image classes without unstable full inverse weighting."""
    balance_power = float(balance_power)
    if balance_power <= 0:
        return F.cross_entropy(logits, targets)
    class_count = int(logits.size(-1))
    counts = torch.bincount(targets, minlength=class_count).to(logits)
    present = counts > 0
    if int(present.sum().item()) <= 1:
        return F.cross_entropy(logits, targets)
    weights = torch.zeros_like(counts)
    present_counts = counts[present]
    weights[present] = (present_counts.mean() / present_counts).pow(balance_power)
    weights[present] = weights[present].clamp(0.25, 4.0)
    weights[present] /= weights[present].mean().clamp_min(1e-8)
    return F.cross_entropy(logits, targets, weight=weights)


def _scage_screw_geometry_loss(
    data,
    graph_rep,
    node_rep,
    angle_head,
    torsion_head,
    distance_head,
    shift_head,
    screw_head,
    angle_bins,
    torsion_bins,
    boundary_angle_weight,
    boundary_torsion_weight,
    torsion_objective,
    distance_component_weight,
    angle_component_weight,
    screw_component_weight,
    shift_balance_power,
    gamma,
    max_pairs=128,
    image_cap=1,
):
    internal_reps, internal_targets = [], []
    boundary_reps, boundary_targets = [], []
    torsion_reps, torsion_targets = [], []
    distance_reps, distance_targets, shift_targets = [], [], []
    screw_reps, screw_targets = [], []
    valid_samples = 0
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen):
            continue
        if not bool(data.star_link_metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != node_idx.numel() or (mapping < 0).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(node_idx.numel(), device=data.x.device)
        pair_global = data.attachment_pair[graph_idx].long()
        pair_local = global_to_local[pair_global]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = int(pair_local[0]), int(pair_local[1])
        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_global = data.ordered_backbone_path[ptr0:ptr1]
        path_local = global_to_local[path_global]
        if path_local.numel() < 2 or (path_local < 0).any():
            continue

        rotation = data.screw_rotation[graph_idx]
        translation = data.screw_translation[graph_idx]
        row, col = torch.triu_indices(pos.size(0), pos.size(0), offset=1, device=pos.device)
        pair_mask = getattr(data, 'scage_geometry_pair_mask', None)
        if pair_mask is not None:
            selected_mask = pair_mask[graph_idx, :pos.size(0), :pos.size(0)][row, col]
            row, col = row[selected_mask], col[selected_mask]
        same_distance = torch.linalg.vector_norm(pos[row] - pos[col], dim=-1)
        images = []
        shifts = []
        finite_images = None
        if has_screw:
            plus, minus = pos, pos
            for shift in range(1, int(image_cap) + 1):
                plus = plus @ rotation.transpose(0, 1) + translation
                minus = (minus - translation) @ rotation
                images.extend([plus, minus])
                shifts.extend([shift, -shift])
        elif has_polygen:
            active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
            if active_axes.numel() == 1:
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                for shift in range(1, int(image_cap) + 1):
                    images.extend([pos + shift * vector_t, pos - shift * vector_t])
                    shifts.extend([shift, -shift])
        elif hasattr(data, 'smer_image_pos3d'):
            finite_images = data.smer_image_pos3d[node_idx].permute(1, 0, 2)
            if finite_images.shape == (3, pos.size(0), 3) and torch.isfinite(finite_images).all():
                images = [finite_images[0], finite_images[2]]
                shifts = [-1, 1]
        if images and row.numel():
            cross = torch.stack([
                torch.linalg.vector_norm(pos[row] - image[col], dim=-1) for image in images
            ])
            nearest_cross = cross.min(dim=0).values
            all_distances = torch.cat([same_distance.unsqueeze(0), cross], dim=0)
            all_shifts = pos.new_tensor([0, *shifts], dtype=torch.long)
            nearest_all_idx = all_distances.min(dim=0).indices
            nearest_shift = all_shifts[nearest_all_idx]
            buckets = torch.clamp((nearest_cross / 2.0).floor().long() + 1, max=20)
            selection = _stratified_pair_selection(buckets, max_pairs=max_pairs)
            left_rep, right_rep = node_rep[node_idx[row[selection]]], node_rep[node_idx[col[selection]]]
            distance_reps.append(torch.cat([
                left_rep, right_rep, torch.abs(left_rep - right_rep), left_rep * right_rep
            ], dim=-1))
            distance_targets.append(torch.stack([
                same_distance[selection], nearest_cross[selection]
            ], dim=-1))
            shift_targets.append(nearest_shift[selection] + int(image_cap))

        if has_screw:
            trace = torch.trace(rotation)
            screw_angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
            axis = torch.stack([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ])
            axis_norm = axis.norm()
            if axis_norm < 1e-6:
                axis = translation / translation.norm().clamp_min(1e-8)
            else:
                axis = axis / axis_norm
            rise = torch.dot(axis, translation).abs()
            screw_reps.append(graph_rep[graph_idx])
            screw_targets.append(torch.stack([rise, screw_angle.sin(), screw_angle.cos()]))

        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(int(node_idx.numel()))]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()):
            if tuple(sorted((source, target))) in star_pair:
                continue
            neighbors[source].add(target)
        local_geometry_mask = None
        if pair_mask is not None:
            local_geometry_mask = pair_mask[
                graph_idx, :pos.size(0), :pos.size(0)
            ]

        def target_geometry_is_masked(atom_indices):
            if local_geometry_mask is None:
                return True
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    if not bool(local_geometry_mask[
                        atom_indices[first_idx], atom_indices[second_idx]
                    ].item()):
                        return False
            return True

        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    left, right = adjacent[first_idx], adjacent[second_idx]
                    if not target_geometry_is_masked((left, center, right)):
                        continue
                    angle = _angle_value(pos[left], pos[center], pos[right])
                    if angle is None:
                        continue
                    internal_reps.append(torch.cat([
                        node_rep[node_idx[left]], node_rep[node_idx[center]], node_rep[node_idx[right]]
                    ]))
                    internal_targets.append(_angle_bin(angle, angle_bins))

        if path_local.numel() >= 3:
            left_neighbor = int(path_local[1].item())
            right_neighbor = int(path_local[-2].item())
            if has_screw:
                previous_right = (pos[right_boundary] - translation) @ rotation
                next_left = pos[left_boundary] @ rotation.transpose(0, 1) + translation
                next_left_neighbor = pos[left_neighbor] @ rotation.transpose(0, 1) + translation
            elif has_polygen:
                active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
                if active_axes.numel() != 1:
                    continue
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                previous_right = pos[right_boundary] - vector_t
                next_left = pos[left_boundary] + vector_t
                next_left_neighbor = pos[left_neighbor] + vector_t
            elif finite_images is not None:
                previous_right = finite_images[0, right_boundary]
                next_left = finite_images[2, left_boundary]
                next_left_neighbor = finite_images[2, left_neighbor]
            else:
                continue
            left_angle = _angle_value(previous_right, pos[left_boundary], pos[left_neighbor])
            right_angle = _angle_value(pos[right_neighbor], pos[right_boundary], next_left)
            if left_angle is not None and target_geometry_is_masked(
                (right_boundary, left_boundary, left_neighbor)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_boundary]], node_rep[node_idx[left_boundary]],
                    node_rep[node_idx[left_neighbor]]
                ]))
                boundary_targets.append(_angle_bin(left_angle, angle_bins))
            if right_angle is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]]
                ]))
                boundary_targets.append(_angle_bin(right_angle, angle_bins))
            torsion = _signed_dihedral(
                pos[right_neighbor], pos[right_boundary], next_left, next_left_neighbor
            )
            if torsion is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary, left_neighbor)
            ):
                torsion_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]], node_rep[node_idx[left_neighbor]]
                ]))
                torsion_targets.append(torsion)
        valid_samples += 1

    geometry_components = {}
    if internal_targets:
        logits = angle_head(torch.stack(internal_reps))
        geometry_components['internal_angle'] = _multiclass_focal_loss(
            logits, torch.stack(internal_targets), gamma
        )
    if boundary_targets:
        logits = angle_head(torch.stack(boundary_reps))
        boundary_loss = _multiclass_focal_loss(logits, torch.stack(boundary_targets), gamma)
        if 'internal_angle' in geometry_components:
            geometry_components['angle'] = (
                geometry_components.pop('internal_angle') + float(boundary_angle_weight) * boundary_loss
            ) / (1.0 + float(boundary_angle_weight))
        else:
            geometry_components['angle'] = boundary_loss
    elif 'internal_angle' in geometry_components:
        geometry_components['angle'] = geometry_components.pop('internal_angle')
    if torsion_targets:
        logits = torsion_head(torch.stack(torsion_reps))
        torsion_values = torch.stack(torsion_targets)
        if torsion_objective == 'circular':
            geometry_components['torsion'] = _circular_regression_loss(
                logits, torsion_values
            )
        else:
            categorical_targets = torch.clamp(
                (((torsion_values + torch.pi) / (2.0 * torch.pi)) * int(torsion_bins)).long(),
                min=0, max=int(torsion_bins) - 1,
            )
            geometry_components['torsion'] = _multiclass_focal_loss(
                logits, categorical_targets, gamma
            )
    if distance_targets:
        pair_rep = torch.cat(distance_reps)
        distance_target = torch.cat(distance_targets)
        distance_loss = F.smooth_l1_loss(distance_head(pair_rep), distance_target, beta=0.5)
        shift_loss = _balanced_shift_cross_entropy(
            shift_head(pair_rep), torch.cat(shift_targets), shift_balance_power
        )
        geometry_components['distance'] = distance_loss + 0.2 * shift_loss
    if screw_targets:
        geometry_components['screw'] = F.smooth_l1_loss(
            screw_head(torch.stack(screw_reps)), torch.stack(screw_targets), beta=0.25
        )
    if not geometry_components:
        return node_rep.new_tensor(0.0), {
            'samples': 0, 'internal': 0, 'boundary': 0, 'torsion': 0,
            'distance_pairs': 0, 'shift_negative': 0, 'shift_center': 0,
            'shift_positive': 0, 'screw_ops': 0,
        }, {}
    component_priors = {
        'distance': float(distance_component_weight),
        'angle': float(angle_component_weight),
        'torsion': float(boundary_torsion_weight),
        'screw': float(screw_component_weight),
    }
    active_weight = sum(component_priors[name] for name in geometry_components)
    total = sum(
        component_priors[name] * value for name, value in geometry_components.items()
    ) / max(active_weight, 1e-8)
    signed_shift_targets = (
        torch.cat(shift_targets) - int(image_cap)
        if shift_targets else node_rep.new_empty(0, dtype=torch.long)
    )
    component_logs = {
        name: value.detach() for name, value in geometry_components.items()
    }
    if distance_targets:
        component_logs['distance_regression'] = distance_loss.detach()
        component_logs['image_shift'] = shift_loss.detach()
    return total, {
        'samples': valid_samples,
        'internal': len(internal_targets),
        'boundary': len(boundary_targets),
        'torsion': len(torsion_targets),
        'distance_pairs': sum(item.size(0) for item in distance_targets),
        'shift_negative': int((signed_shift_targets < 0).sum().item()),
        'shift_center': int((signed_shift_targets == 0).sum().item()),
        'shift_positive': int((signed_shift_targets > 0).sum().item()),
        'screw_ops': len(screw_targets),
    }, component_logs


def _make_scage_geometry_pair_mask(data, ratio):
    batch_size, max_nodes, _ = data.scage_spd.shape
    mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool, device=data.x.device)
    for graph_idx in range(batch_size):
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen) or float(ratio) <= 0:
            continue
        count = int((data.batch == graph_idx).sum().item())
        row, col = torch.triu_indices(count, count, offset=1, device=data.x.device)
        target_count = min(
            int(max(1, round(float(ratio) * row.numel()))),
            int(row.numel()),
        )
        if target_count <= 0:
            continue
        choice = torch.randperm(row.numel(), device=data.x.device)[:target_count]
        selected_row, selected_col = row[choice], col[choice]
        mask[graph_idx, selected_row, selected_col] = True
        mask[graph_idx, selected_col, selected_row] = True

        def mask_clique(atom_indices):
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    first = atom_indices[first_idx]
                    second = atom_indices[second_idx]
                    mask[graph_idx, first, second] = True
                    mask[graph_idx, second, first] = True

        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(count, device=data.x.device)
        pair_local = global_to_local[data.attachment_pair[graph_idx].long()]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = pair_local.tolist()
        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(count)]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(
            global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()
        ):
            if tuple(sorted((source, target))) not in star_pair:
                neighbors[source].add(target)
        angle_candidates = []
        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    angle_candidates.append((adjacent[first_idx], center, adjacent[second_idx]))
        if angle_candidates:
            angle_count = min(
                max(1, int(round(float(ratio) * len(angle_candidates)))),
                len(angle_candidates),
            )
            selected = torch.randperm(len(angle_candidates), device=data.x.device)[:angle_count]
            for candidate_idx in selected.tolist():
                mask_clique(angle_candidates[candidate_idx])

        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_local = global_to_local[data.ordered_backbone_path[ptr0:ptr1]]
        if path_local.numel() >= 3 and not (path_local < 0).any():
            mask_clique((
                int(path_local[-2]), right_boundary,
                left_boundary, int(path_local[1]),
            ))
    return mask


def _multi_positive_supcon_loss(embeddings, identities, temperature):
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.t() / float(temperature)
    identity_mask = identities[:, None] == identities[None, :]
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = identity_mask & ~self_mask
    logits = logits.masked_fill(self_mask, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count.clamp_min(1))
    return per_anchor[valid].mean()


def _periodic_cut_identity_key(value):
    """Normalize cached and online cut identities to stable hashable keys."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_periodic_cut_identity_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _periodic_cut_identity_key(item))
                for key, item in value.items()
            )
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _scage_multi_positive_periodic_loss(
    base_model, data, projection_head, graph_input, temperature, max_mrus, retry, views_per_polymer
):
    from src.dataset.graph_data import periodicity_augment_smiles
    from src.dataset.dataloader import custom_collate

    stats = {
        'identities_attempted': len(data.smiles), 'views_requested': len(data.smiles) * int(views_per_polymer),
        'views_valid': 0, 'duplicate_views': 0, 'failed_views': 0,
        'distinct_cut_views': 0, 'valid_identities': 0, 'positive_pairs': 0,
        'positive_count_sum': 0, 'graph_cache_hits': 0, 'graph_cache_misses': 0,
    }
    graph_encoder = base_model.encoders['graph'].encoder
    original_rep, _ = graph_encoder.forward_topology_only(data)
    augmented_graphs, augmented_ids, valid_original_ids = [], [], []
    for sample_idx, smiles in enumerate(data.smiles):
        unique_views = set()
        unique_cuts = set()
        cached_views = (
            list(data.periodic_aug_smiles[sample_idx])
            if hasattr(data, 'periodic_aug_smiles') else []
        )
        cached_cuts = (
            list(data.periodic_aug_cut_identities[sample_idx])
            if hasattr(data, 'periodic_aug_cut_identities') else []
        )
        for view_idx in range(int(views_per_polymer)):
            accepted = None
            # Feature-cache views are generated with max_mrus=3. Reusing them
            # under a different runtime cap silently violates the requested
            # augmentation policy and can reintroduce oversized 3-MRU graphs.
            if int(max_mrus) == 3 and view_idx < len(cached_views):
                try:
                    cached_cut = _periodic_cut_identity_key(
                        cached_cuts[view_idx] if view_idx < len(cached_cuts) else None
                    )
                    cached_graph, cache_hit = _cached_periodic_aug_graph(
                        cached_views[view_idx], graph_input
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(cached_views[view_idx]), cached_graph, cached_cut)
                except Exception:
                    accepted = None
            for _attempt in range(max(1, int(retry))):
                if accepted is not None:
                    break
                try:
                    aug_smiles, _, metadata = periodicity_augment_smiles(
                        smiles, max_mrus=max_mrus, return_n=True, return_metadata=True
                    )
                    cut_identity = _periodic_cut_identity_key(
                        metadata.get('cut_identity')
                    )
                    if (
                        str(aug_smiles) == str(smiles)
                        or str(aug_smiles) in unique_views
                        or (cut_identity is not None and cut_identity in unique_cuts)
                    ):
                        stats['duplicate_views'] += 1
                        continue
                    augmented_graph, cache_hit = _cached_periodic_aug_graph(
                        aug_smiles, graph_input
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(aug_smiles), augmented_graph, cut_identity)
                    break
                except Exception:
                    accepted = None
            if accepted is None:
                stats['failed_views'] += 1
                continue
            unique_views.add(accepted[0])
            if accepted[2] is not None:
                unique_cuts.add(accepted[2])
                stats['distinct_cut_views'] += 1
            augmented = accepted[1]
            augmented.smiles = accepted[0]
            augmented.input_ids_smiles = data.input_ids_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.attention_mask_smiles = data.attention_mask_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.fp = data.fp[sample_idx:sample_idx + 1].detach().cpu()
            augmented.y = data.y[sample_idx].detach().cpu().reshape(-1)
            augmented.z = torch.ones(1, dtype=torch.long)
            augmented.pos = torch.zeros(1, 3)
            augmented.pos_confs = torch.zeros(1, 1, 3)
            augmented.graph_to_geom_index = torch.full(
                (augmented.x.size(0),), -1, dtype=torch.long
            )
            augmented.geom_build_ok = False
            augmented.geom_coordinate_ok = False
            augmented.geom_context_id = 1
            augmented.screw_valid = False
            augmented.smer_valid = False
            augmented.polymer_ecfp_target = torch.zeros(2048)
            augmented.polymer_ecfp_valid = False
            augmented.polymer_ecfp_source = 'periodic_view'
            descriptor_valid = bool(data.scage_descriptor_valid[sample_idx].item())
            augmented.scage_descriptor_valid_confs = torch.tensor([descriptor_valid])
            for descriptor_name in ('shape', 'usrcat', 'autocorr3d', 'rdf', 'morse', 'whim'):
                selected = getattr(data, f'scage_descriptor_{descriptor_name}')[sample_idx]
                setattr(
                    augmented,
                    f'scage_descriptor_{descriptor_name}_confs',
                    selected.detach().cpu().unsqueeze(0),
                )
            augmented_graphs.append(augmented)
            augmented_ids.append(sample_idx)
            stats['views_valid'] += 1
        if unique_views:
            valid_original_ids.append(sample_idx)
    if len(valid_original_ids) < 2 or not augmented_graphs:
        return original_rep.new_tensor(0.0), stats
    aug_batch = custom_collate(augmented_graphs, random_conformer=False).to(data.x.device)
    augmented_rep, _ = graph_encoder.forward_topology_only(aug_batch)
    representations, identities = [], []
    for sample_idx in valid_original_ids:
        representations.append(original_rep[sample_idx])
        identities.append(sample_idx)
    representations.extend(list(augmented_rep))
    identities.extend(augmented_ids)
    identities = torch.tensor(identities, dtype=torch.long, device=data.x.device)
    projected = projection_head(torch.stack(representations))
    stats['valid_identities'] = len(valid_original_ids)
    counts = torch.bincount(identities, minlength=len(data.smiles))
    stats['positive_pairs'] = int((counts * (counts - 1)).sum().item())
    stats['positive_count_sum'] = int((counts[counts > 0] - 1).sum().item())
    return _multi_positive_supcon_loss(projected, identities, temperature), stats


def _symmetric_infonce(left, right, temperature, valid_mask=None):
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=left.device, dtype=torch.bool).view(-1)
        left = left[valid_mask]
        right = right[valid_mask]
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    if _distributed_enabled():
        from torch.distributed.nn.functional import all_gather

        local_count = torch.tensor([left.size(0)], device=left.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        global_count = sum(counts)
        if global_count < 2:
            return (left.sum() + right.sum()) * 0.0
        max_count = max(counts)
        if left.size(0) < max_count:
            padding = left.new_zeros((max_count - left.size(0), left.size(1)))
            left_padded = torch.cat([left, padding], dim=0)
            right_padded = torch.cat([right, padding], dim=0)
        else:
            left_padded, right_padded = left, right
        gathered_left_parts = list(all_gather(left_padded))
        gathered_right_parts = list(all_gather(right_padded))
        gathered_left = torch.cat([
            part[:count] for part, count in zip(gathered_left_parts, counts)
        ], dim=0)
        gathered_right = torch.cat([
            part[:count] for part, count in zip(gathered_right_parts, counts)
        ], dim=0)
        if left.size(0) == 0:
            return (gathered_left.sum() + gathered_right.sum()) * 0.0
        offset = sum(counts[:dist.get_rank()])
        labels = torch.arange(left.size(0), device=left.device) + offset
        left_logits = left @ gathered_right.t() / float(temperature)
        right_logits = right @ gathered_left.t() / float(temperature)
        local_loss = 0.5 * (
            F.cross_entropy(left_logits.float(), labels)
            + F.cross_entropy(right_logits.float(), labels)
        )
        # DDP averages gradients over ranks. Rescale the local mean so the
        # resulting gradient equals a global per-sample mean for unequal counts.
        return local_loss * (dist.get_world_size() * left.size(0) / global_count)
    if left.size(0) < 2:
        return (left.sum() + right.sum()) * 0.0
    logits = left @ right.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits.float(), labels) + F.cross_entropy(logits.t().float(), labels)
    )


def _scage_semantic_alignment_losses(base_model, data, args):
    required = {'graph', 'smiles', 'fp'}
    if not required.issubset(base_model.encoders):
        raise ValueError("Parallel semantic alignment requires graph, smiles, and fp encoders")
    embeddings = base_model.encode_modalities(data)
    modality_index = {name: idx for idx, name in enumerate(base_model.modality_list)}
    graph = embeddings[:, modality_index['graph']]
    smiles = embeddings[:, modality_index['smiles']]
    fp = embeddings[:, modality_index['fp']]
    projected_graph = base_model.project_alignment('graph', graph)
    projected_smiles = base_model.project_alignment('smiles', smiles)
    projected_fp = base_model.project_alignment('fp', fp)

    drop_probabilities = {
        'fp': args.alignment_fp_drop,
        'smiles': args.alignment_smiles_drop,
        'graph': args.alignment_graph_drop,
    }
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    masks = [
        base_model.sample_availability_mask(
            embeddings.size(0), drop_probabilities, min_available=2,
            device=embeddings.device, intrinsic_mask=intrinsic_mask,
        )
        for _ in range(2)
    ]
    fused_views = []
    for mask in masks:
        fused, _ = base_model.fuse_embeddings(embeddings, availability_mask=mask)
        fused_views.append(base_model.project_alignment('fusion', fused))
    fused_view_loss = _symmetric_infonce(fused_views[0], fused_views[1], args.temperature)

    full_mask = intrinsic_mask
    full_fused, full_weights = base_model.fuse_embeddings(embeddings, availability_mask=full_mask)
    teacher = F.normalize(base_model.project_alignment('fusion', full_fused), dim=-1).detach()
    lomo_losses = []
    for missing_name in ('fp', 'smiles', 'graph'):
        present = full_mask[:, modality_index[missing_name]]
        if not bool(present.any()):
            continue
        lomo_mask = full_mask.clone()
        lomo_mask[:, modality_index[missing_name]] = False
        student, _ = base_model.fuse_embeddings(embeddings, availability_mask=lomo_mask)
        student = F.normalize(base_model.project_alignment('fusion', student), dim=-1)
        lomo_losses.append(1.0 - (student[present] * teacher[present]).sum(dim=-1).mean())
    lomo_loss = (
        torch.stack(lomo_losses).mean()
        if lomo_losses else embeddings.sum() * 0.0
    )

    prior_by_name = {'smiles': 0.30, 'graph': 0.40, 'fp': 0.30}
    prior = full_weights.new_tensor([prior_by_name[name] for name in base_model.modality_list])
    sample_prior = prior.unsqueeze(0) * full_mask.to(dtype=full_weights.dtype)
    sample_prior = sample_prior / sample_prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    safe_weights = full_weights.clamp_min(1e-8)
    pooling_kl = (
        safe_weights
        * (safe_weights.log() - sample_prior.clamp_min(1e-8).log())
        * full_mask
    ).sum(dim=1).mean()
    mean_weights = full_weights.mean(dim=0)
    graph_valid = full_mask[:, modality_index['graph']]
    losses = {
        'fused_view': fused_view_loss,
        'graph_smiles': _symmetric_infonce(
            projected_graph, projected_smiles, args.temperature,
            valid_mask=graph_valid,
        ),
        'graph_fp': _symmetric_infonce(
            projected_graph, projected_fp, args.temperature,
            valid_mask=graph_valid,
        ),
        'lomo': lomo_loss,
        'pooling_kl': pooling_kl,
    }
    missing_rates = {
        name: float(torch.stack([~mask[:, idx] for mask in masks]).float().mean().item())
        for idx, name in enumerate(base_model.modality_list)
    }
    stats = {
        'missing_rates': missing_rates,
        'pooling_entropy': float(
            (-(full_weights.clamp_min(1e-8) * full_weights.clamp_min(1e-8).log()).sum(dim=1).mean()).item()
        ),
        'pooling_weights': {
            name: float(mean_weights[idx].item())
            for idx, name in enumerate(base_model.modality_list)
        },
    }
    return losses, stats


def _fusion_conditioned_masked_atom_loss(base_model, data, prediction_head, mask_ratio):
    mask_indices = []
    graph_available = getattr(data, 'graph_available', None)
    if graph_available is None:
        graph_available = torch.ones(
            len(data.smiles), dtype=torch.bool, device=data.x.device
        )
    else:
        graph_available = torch.as_tensor(
            graph_available, device=data.x.device
        ).view(-1).bool()
    for graph_idx in range(len(data.smiles)):
        if not bool(graph_available[graph_idx]):
            continue
        nodes = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not nodes.numel():
            continue
        count = max(1, int(round(float(mask_ratio) * nodes.numel())))
        mask_indices.append(nodes[torch.randperm(nodes.numel(), device=nodes.device)[:count]])
    if not mask_indices:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(mask_indices)
    masked_x = data.x.clone()
    masked_x[mask_indices] = 0.0
    graph_module = base_model.encoders['graph']
    raw_graph, raw_nodes = graph_module.encoder.forward_with_x(data, masked_x)
    masked_graph = graph_module.projection(graph_module.norm(raw_graph))
    modality_embeddings = []
    for name in base_model.modality_list:
        modality_embeddings.append(
            masked_graph if name == 'graph' else base_model.encoders[name](data)
        )
    embeddings = torch.stack(modality_embeddings, dim=1)
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    fused, _ = base_model.fuse_embeddings(
        embeddings, availability_mask=intrinsic_mask
    )
    context = fused[data.batch[mask_indices]]
    logits = prediction_head(torch.cat([raw_nodes[mask_indices], context], dim=-1))
    targets = data.atomic_num[mask_indices].long()
    return F.cross_entropy(logits.float(), targets), int(mask_indices.numel())


def _alignment_total(losses, args):
    return (
        args.alignment_fused_weight * losses['fused_view']
        + args.alignment_graph_smiles_weight * losses['graph_smiles']
        + args.alignment_graph_fp_weight * losses['graph_fp']
        + args.alignment_lomo_weight * losses['lomo']
        + args.alignment_pooling_kl_weight * losses['pooling_kl']
        + args.alignment_fused_mask_weight * losses.get('fused_mask', losses['fused_view'].new_tensor(0.0))
    )


def _bf16_alignment_parity_gate(base_model, data, args):
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    fp32_loss = _alignment_total(_scage_semantic_alignment_losses(base_model, data, args)[0], args)
    fp32_loss.backward()
    fp32_grad = _gradient_vector((base_model,))
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss = _alignment_total(
            _scage_semantic_alignment_losses(base_model, data, args)[0], args
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector((base_model,))
    delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    base_model.zero_grad(set_to_none=True)
    return finite and delta <= 0.02 and cosine >= 0.98, {
        'relative_loss_delta': delta, 'gradient_cosine': cosine, 'finite': finite,
    }


def _scage_alignment_optimizer(base_model, args):
    groups = []
    for name, lr in (
        ('graph', args.alignment_graph_lr),
        ('smiles', args.alignment_smiles_lr),
        ('fp', args.alignment_fp_lr),
    ):
        groups.append({
            'params': list(base_model.encoders[name].parameters()),
            'lr': float(lr),
            'name': name,
        })
    groups.extend([
        {
            'params': list(base_model.alignment_projections.parameters()),
            'lr': float(args.alignment_projection_lr),
            'name': 'alignment_projection',
        },
        {
            'params': list(base_model.parallel_attention_fusion.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'fusion',
        },
    ])
    if hasattr(base_model, 'alignment_mask_head'):
        groups.append({
            'params': list(base_model.alignment_mask_head.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'alignment_mask_head',
        })
    return optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _module_gradient_norm(module):
    squares = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu().item())


def _batched_shortest_path_targets(data, max_distance):
    targets = []
    pair_indices = []
    max_distance = int(max_distance)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 2:
            continue
        dist = torch.full((n, n), max_distance, device=data.x.device, dtype=torch.long)
        diag = torch.arange(n, device=data.x.device)
        dist[diag, diag] = 0
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        if keep.any():
            ls = local[src[keep]]
            ld = local[dst[keep]]
            valid = (ls >= 0) & (ld >= 0)
            dist[ls[valid], ld[valid]] = 1
        dist_float = dist.float()
        inf = float(max_distance)
        for k in range(n):
            dist_float = torch.minimum(dist_float, dist_float[:, k:k + 1] + dist_float[k:k + 1, :])
        dist = dist_float.clamp(max=inf).long()
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        if row.numel() == 0:
            continue
        pair_indices.append(torch.stack([node_idx[row], node_idx[col]], dim=1))
        targets.append(dist[row, col])
    if not targets:
        empty_pairs = data.edge_index.new_empty((0, 2))
        empty_targets = data.x.new_empty((0,), dtype=torch.long)
        return empty_pairs, empty_targets
    return torch.cat(pair_indices, dim=0), torch.cat(targets, dim=0)


def _graph_shortest_path_loss(base_model, data, sp_head, max_distance):
    if not _is_scage_graph_encoder(base_model):
        return data.x.new_tensor(0.0)
    _, node_rep = _graph_encode_nodes(base_model, data)
    pairs, targets = _batched_shortest_path_targets(data, max_distance=max_distance)
    if pairs.numel() == 0:
        return data.x.new_tensor(0.0)
    pair_rep = torch.cat([node_rep[pairs[:, 0]], node_rep[pairs[:, 1]], torch.abs(node_rep[pairs[:, 0]] - node_rep[pairs[:, 1]])], dim=-1)
    return F.cross_entropy(sp_head(pair_rep), targets)


def _graph_angle_loss(base_model, data, angle_head, angle_bins):
    if not _is_scage_graph_encoder(base_model) or not hasattr(data, 'pos3d') or not hasattr(data, 'batch3d'):
        return None
    _, node_rep = _graph_encode_nodes(base_model, data)
    reps = []
    targets = []
    angle_bins = int(angle_bins)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 3:
            continue
        coordinate_ok = getattr(data, 'geom_coordinate_ok', getattr(data, 'geom_build_ok', None))
        if coordinate_ok is not None and not bool(coordinate_ok.flatten()[graph_idx].item()):
            continue
        if not hasattr(data, 'graph_to_geom_index'):
            continue
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != n or (mapping < 0).any() or (mapping >= data.pos3d.size(0)).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        neigh = [[] for _ in range(n)]
        for s_idx, d_idx in zip(src[keep].tolist(), dst[keep].tolist()):
            center = int(local[s_idx].item())
            nb = int(local[d_idx].item())
            if center >= 0 and nb >= 0 and nb not in neigh[center]:
                neigh[center].append(nb)
        for center, ns in enumerate(neigh):
            if len(ns) < 2:
                continue
            for a_i in range(len(ns)):
                for b_i in range(a_i + 1, len(ns)):
                    left, right = ns[a_i], ns[b_i]
                    v1 = pos[left] - pos[center]
                    v2 = pos[right] - pos[center]
                    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
                    if float(denom.detach().cpu().item()) <= 1e-8:
                        continue
                    cos = torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0)
                    angle = torch.acos(cos)
                    target = torch.clamp((angle / torch.pi * angle_bins).long(), max=angle_bins - 1)
                    global_center = node_idx[center]
                    global_left = node_idx[left]
                    global_right = node_idx[right]
                    reps.append(torch.cat([node_rep[global_left], node_rep[global_center], node_rep[global_right]], dim=-1))
                    targets.append(target)
    if not reps:
        return None
    return F.cross_entropy(angle_head(torch.stack(reps, dim=0)), torch.stack(targets, dim=0).long())

def _graph_pretrain_loss(
    base_model,
    data,
    graph_atom_head,
    mask_ratio,
    mask_weight,
    periodic_aug_weight,
    graph_input,
    periodic_aug_max_mrus,
    periodic_aug_retry,
    periodic_cl_temperature,
):
    mask_loss = _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio)
    periodic_aug_loss = data.x.new_tensor(0.0)
    periodic_aug_stats = _empty_periodic_aug_stats()
    if float(periodic_aug_weight) > 0:
        periodic_aug_loss, periodic_aug_stats = _graph_periodic_aug_loss(
            base_model,
            data,
            graph_input=graph_input,
            temperature=periodic_cl_temperature,
            max_mrus=periodic_aug_max_mrus,
            retry=periodic_aug_retry,
        )
    total = float(mask_weight) * mask_loss + float(periodic_aug_weight) * periodic_aug_loss
    return total, mask_loss, periodic_aug_loss, periodic_aug_stats

def _geom_denoise_loss(
    base_model,
    data,
    geom_noise_head,
    noise_std,
    noise_std_min=None,
    noise_std_max=None,
):
    if 'geom' not in base_model.encoders:
        return data.pos3d.new_tensor(0.0)

    geom_module = base_model.encoders['geom']
    clean_pos = data.pos3d
    target_pos = clean_pos.detach().clone()
    if noise_std_min is not None or noise_std_max is not None:
        if noise_std_min is None or noise_std_max is None:
            raise ValueError("--geom_noise_std_min and --geom_noise_std_max must be set together")
        min_std = float(noise_std_min)
        max_std = float(noise_std_max)
        if min_std < 0.0 or max_std <= 0.0 or min_std > max_std:
            raise ValueError("Require 0 <= --geom_noise_std_min <= --geom_noise_std_max")
        std = torch.empty((), device=target_pos.device, dtype=target_pos.dtype).uniform_(min_std, max_std)
    else:
        std = torch.as_tensor(float(noise_std), device=target_pos.device, dtype=target_pos.dtype)
    noise = torch.randn_like(target_pos) * std
    try:
        data.pos3d = target_pos + noise
        node_rep, _ = geom_module.encoder.encode_nodes(data)
        pred_noise = geom_noise_head(node_rep)
        per_node = F.smooth_l1_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        context = getattr(data, 'geom_context_id', None)
        node_batch = getattr(data, 'batch3d', None)
        if context is None or node_batch is None:
            return per_node.mean()
        sample_weights = torch.where(context.to(per_node.device).long() == 1, 0.3, 1.0)
        weights = sample_weights[node_batch.to(per_node.device)]
        return (per_node * weights).sum() / weights.sum().clamp_min(1e-8)
    finally:
        data.pos3d = clean_pos

def _dataset_kwargs_from_args(args):
    return dict(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
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


def main():
    args = parse_arguments()
    stage1_weights = (
        args.scage_mips_mask_weight,
        args.scage_periodic_sp_weight,
        args.scage_screw_geometry_weight,
    )
    if any(weight < 0 for weight in stage1_weights):
        raise ValueError("SCAGE Stage 1 task weights must be non-negative")
    if args.scage_geometry_max_pairs < 1:
        raise ValueError("--scage_geometry_max_pairs must be positive")
    if not 0.0 <= args.scage_shift_balance_power <= 1.0:
        raise ValueError("--scage_shift_balance_power must be in [0, 1]")
    if args.cache_only:
        # Must happen before any CUDA query and before forking RDKit/torch
        # workers. CPU Adam otherwise performs a CUDA graph health check in a
        # bad fork and rejects valid periodic candidates.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    requested_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if args.cache_only and requested_distributed:
        raise ValueError("--cache_only must run as one CPU process, not through torchrun")
    distributed = requested_distributed
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=24),
            device_id=torch.device('cuda', local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
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
    if args.pretrain_stage == 'scage_m4p':
        if args.graph_encoder_type != 'scage' or args.graph_input != 'star_linking':
            raise ValueError("scage_m4p requires graph_encoder_type=scage and graph_input=star_linking")
        if args.geom_input != 'polygen_periodic' or set(args.modalities) != {'graph'}:
            raise ValueError(
                "scage_m4p requires geom_input=polygen_periodic and modalities=graph"
            )
        if args.scage_ecfp_weight != 0 or args.scage_periodic_contrast_weight != 0:
            raise ValueError(
                "The sparse MIPS-PBC Stage 1 supports only masked atom, LGA SPD, "
                "and PBC edge-distance tasks"
            )
    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage':
        if set(args.modalities) != {'smiles', 'graph', 'fp'}:
            raise ValueError("Parallel graph alignment requires modalities=smiles graph fp")
        if args.fusion_type != 'parallel_attention':
            raise ValueError("Parallel graph alignment requires fusion_type=parallel_attention")
    from src.dataset.geom_data import set_screw_quality_config
    set_screw_quality_config(
        kabsch_rmsd_max=args.screw_kabsch_rmsd_max,
        rotation_consistency_deg=args.screw_rotation_consistency_deg,
        translation_relative_max=args.screw_translation_relative_max,
        final_rmsd_max=args.screw_final_rmsd_max,
        gradient_rms_max=args.ff_gradient_rms_max,
        gradient_max=args.ff_gradient_max,
        probe_steps=args.ff_probe_steps,
        probe_energy_delta_per_atom_max=args.ff_probe_energy_delta_per_atom_max,
        screw_energy_per_atom_max=args.screw_energy_per_atom_max,
        screw_center_gradient_rms_max=args.screw_center_gradient_rms_max,
    )

    from src.dataset import UniDataset
    dataset_kwargs = _dataset_kwargs_from_args(args)
    if args.cache_only:
        warnings.filterwarnings("ignore")
        dataset = UniDataset(**dataset_kwargs)
        print(
            f"[feature_cache] cache-only complete: path={dataset.feature_cache_path}, "
            f"usable_rows={len(dataset)}"
        )
        return

    from src.modules import UniEncoderAttention
    from src.utils import compute_contrastive_loss, get_data_loader, set_global_seed
    set_global_seed(args.seed + rank)

    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if rank == 0:
            print(f"Found {n_gpus} GPUs available; distributed world_size={world_size}")
        device = torch.device("cuda", local_rank)
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    if distributed and args.rebuild_feature_cache:
        if rank == 0:
            dataset = UniDataset(**dataset_kwargs)
        dist.barrier()
        if rank != 0:
            dataset_kwargs['rebuild_feature_cache'] = False
            dataset = UniDataset(**dataset_kwargs)
    else:
        dataset = UniDataset(**dataset_kwargs)
    indices = np.arange(len(dataset))
    if args.pretrain_unique_smiles and args.pretrain_stage in {'scage_m4p', 'alignment'}:
        indices = _unique_smiles_indices(dataset)
        print(
            f"Pretraining identity deduplication: {len(dataset)} rows -> "
            f"{len(indices)} unique SMILES"
        )
    sampler = None
    if distributed:
        from torch.utils.data import Subset
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            Subset(dataset, [int(index) for index in indices]),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
    dataloader = get_data_loader(
        dataset, indices=indices, batch_size=args.batch_size, shuffle=True,
        drop_last=distributed, random_conformer=True,
        num_workers=args.loader_workers, pin_memory=True,
        persistent_workers=args.loader_workers > 0, sampler=sampler,
    )

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
        alignment_projection_dim=args.alignment_projection_dim,
    )

    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        if args.graph_encoder_type == 'scage':
            expected_schema = 'scage-mips-pbc-pyg-v1'
            if not isinstance(checkpoint, dict) or checkpoint.get('meta', {}).get('schema') != expected_schema:
                raise RuntimeError(
                    f"SCAGE requires a {expected_schema} checkpoint; rerun Stage 1 with the matching model."
                )
            checkpoint_stage = checkpoint.get('meta', {}).get('stage')
            if args.pretrain_stage == 'alignment' and checkpoint_stage != 'scage_m4p':
                raise RuntimeError(
                    "SCAGE semantic alignment requires a Stage 1 scage_m4p checkpoint; "
                    f"received stage={checkpoint_stage!r}."
                )
            checkpoint_state = {
                key: value for key, value in checkpoint['state_dict'].items()
                if key.startswith('encoders.graph.encoder.')
            }
            expected_graph_state = {
                key: value for key, value in model.state_dict().items()
                if key.startswith('encoders.graph.encoder.')
            }
            if set(checkpoint_state) != set(expected_graph_state):
                missing_graph = sorted(set(expected_graph_state) - set(checkpoint_state))
                unexpected_graph = sorted(set(checkpoint_state) - set(expected_graph_state))
                raise RuntimeError(
                    "SCAGE Stage 1 must transfer exactly the sparse Graph encoder "
                    f"into Stage 2; missing={missing_graph[:5]}, "
                    f"unexpected={unexpected_graph[:5]}"
                )
            shape_mismatch = [
                key for key in checkpoint_state
                if checkpoint_state[key].shape != expected_graph_state[key].shape
            ]
            if shape_mismatch:
                raise RuntimeError(
                    "SCAGE Stage 1 Graph tensor shape mismatch: "
                    + ", ".join(shape_mismatch[:10])
                )
        else:
            checkpoint_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        if args.graph_encoder_type == 'scage':
            graph_mismatch = [
                key for key in list(missing) + list(unexpected)
                if 'encoders.graph.encoder' in key
            ]
            if graph_mismatch:
                raise RuntimeError(
                    "SCAGE checkpoint architecture mismatch. The original-input SCAGE backbone "
                    "cannot load an older polymer-SCAGE checkpoint. Mismatched keys: "
                    + ", ".join(graph_mismatch[:10])
                )
        print(f"Loaded pretraining checkpoint from {args.pretrained_model_path}")
        print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    model = model.to(device)
    base_model = _base_model(model)
    loss_weights = _stage_loss_weights(args)
    dynamic_loss_weighter = None
    print(
        "Pretraining stage: "
        f"{args.pretrain_stage} "
        f"(contrastive={loss_weights['contrastive']}, graph={loss_weights['graph']}, geom={loss_weights['geom']}, "
        f"dynamic_loss={args.dynamic_pretrain_loss})"
    )

    aux_modules = nn.ModuleList()
    graph_atom_head = None
    geom_noise_head = None
    graph_sp_head = None
    graph_angle_head = None
    m4p_ecfp_head = None
    m4p_torsion_head = None
    m4p_distance_head = None
    m4p_shift_head = None
    m4p_screw_head = None
    m4p_projection_head = None
    mips_atom_head = None
    m4p_ecfp_pos_weight = None
    m4p_ecfp_valid_count = 0
    if args.pretrain_stage == 'scage_m4p':
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import SCAGE_ATOM_VOCABS
        mips_atom_head = nn.Linear(graph_dim, len(SCAGE_ATOM_VOCABS['atomic_num'])).to(device)
        graph_sp_head = nn.Linear(graph_dim * 3, 6).to(device)
        m4p_distance_head = nn.Sequential(
            nn.Linear(graph_dim * 4, graph_dim), nn.SiLU(), nn.Linear(graph_dim, 1),
        ).to(device)
        aux_modules.extend([
            mips_atom_head, graph_sp_head, m4p_distance_head,
        ])
    elif not (args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage') \
            and 'graph' in args.modalities and loss_weights['graph'] > 0:
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import allowable_features
        graph_atom_head = nn.Linear(graph_dim, len(allowable_features['possible_atom_symbols'])).to(device)
        aux_modules.append(graph_atom_head)
        if args.graph_encoder_type == 'scage':
            graph_sp_head = nn.Linear(graph_dim * 3, int(args.scage_sp_max_distance) + 1).to(device)
            graph_angle_head = nn.Linear(graph_dim * 3, int(args.scage_angle_bins)).to(device)
            aux_modules.extend([graph_sp_head, graph_angle_head])
    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage':
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import SCAGE_ATOM_VOCABS
        atom_classes = len(SCAGE_ATOM_VOCABS['atomic_num'])
        base_model.alignment_mask_head = nn.Linear(
            graph_dim + base_model.joint_embedding_dim,
            atom_classes,
        ).to(device)
    if args.pretrain_stage != 'scage_m4p' and 'geom' in args.modalities and loss_weights['geom'] > 0:
        geom_dim = base_model.encoders['geom'].encoder.hidden_channels
        geom_noise_head = nn.Linear(geom_dim, 3).to(device)
        aux_modules.append(geom_noise_head)

    if args.amp_dtype == 'bf16':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            args.amp_dtype = 'fp32'
            if rank == 0:
                print("BF16 disabled: CUDA BF16 support is unavailable")
        elif args.pretrain_stage == 'scage_m4p':
            parity_batch = next(iter(dataloader)).to(device)
            passed, parity = _bf16_parity_gate(base_model, parity_batch, mips_atom_head, args)
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'
        elif args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage':
            parity_batch = next(iter(dataloader)).to(device)
            passed, parity = _bf16_alignment_parity_gate(base_model, parity_batch, args)
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 alignment parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'

    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage':
        optimizer = _scage_alignment_optimizer(base_model, args)
    else:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(aux_modules.parameters()),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
        )

    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    available_batches = int(args.epochs) * len(dataloader)
    scheduled_batches = (
        min(available_batches, int(args.max_steps))
        if int(args.max_steps) > 0 else available_batches
    )
    total_optimizer_steps = max(1, math.ceil(scheduled_batches / accumulation_steps))
    warmup_steps = min(
        total_optimizer_steps - 1,
        max(0, int(round(total_optimizer_steps * float(args.warmup_ratio)))),
    )

    def lr_scale(update_step):
        if warmup_steps > 0 and update_step < warmup_steps:
            return float(update_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, (update_step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    if rank == 0:
        print(
            f"Optimizer schedule: updates={total_optimizer_steps}, "
            f"warmup={warmup_steps} ({float(args.warmup_ratio):.1%}), cosine decay"
        )

    if args.dynamic_pretrain_loss:
        active_loss_names = []
        task_priors = None
        if args.pretrain_stage == 'scage_m4p':
            active_loss_names = [
                name for name, weight in (
                    ('mips_mask', args.scage_mips_mask_weight),
                    ('ecfp', args.scage_ecfp_weight),
                    ('periodic_sp', args.scage_periodic_sp_weight),
                    ('periodic_geometry', args.scage_screw_geometry_weight),
                    ('periodic_contrast', args.scage_periodic_contrast_weight),
                ) if float(weight) > 0
            ]
            task_priors = {
                'mips_mask': args.scage_mips_mask_weight,
                'ecfp': args.scage_ecfp_weight,
                'periodic_sp': args.scage_periodic_sp_weight,
                'periodic_geometry': args.scage_screw_geometry_weight,
                'periodic_contrast': args.scage_periodic_contrast_weight,
            }
        elif args.pretrain_stage == 'graph_geom':
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                if float(args.graph_mask_atom_weight) > 0:
                    active_loss_names.append('graph_mask')
                if float(args.graph_periodic_aug_weight) > 0:
                    active_loss_names.append('graph_periodic_aug')
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    active_loss_names.append('shortest_path')
                if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                    active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        else:
            if loss_weights['contrastive'] > 0:
                active_loss_names.append('contrastive')
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                active_loss_names.append('graph')
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                active_loss_names.append('shortest_path')
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        dynamic_loss_weighter = DynamicPretrainLossWeighter(
            active_loss_names,
            init_window=args.dynamic_loss_warmup_steps,
            recent_window=args.dynamic_loss_recent_window,
            temperature=args.dynamic_loss_temperature,
            task_priors=task_priors,
            effective_caps={'ecfp': args.scage_ecfp_effective_max},
            device=device,
        )
        print(f"Dynamic pretraining loss terms: {active_loss_names}")


    # Create directory for saving loss curves and data
    os.makedirs('./plots/pretrain', exist_ok=True)

    # Record loss and periodic augmentation health for each epoch
    losses = []
    periodic_aug_epoch_stats = []
    dynamic_loss_epoch_weights = []
    dynamic_loss_epoch_stats = []
    m4p_epoch_target_stats = []
    m4p_geometry_epoch_stats = []
    gradient_norms = []
    alignment_epoch_stats = []

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_periodic_aug_stats = _empty_periodic_aug_stats()
        epoch_raw_terms = {}
        epoch_raw_term_counts = {}
        epoch_normalized_terms = {}
        epoch_dynamic_weights = {}
        epoch_effective_weights = {}
        epoch_baselines = {}
        epoch_target_counts = {}
        epoch_geometry_components = {}
        epoch_geometry_component_steps = {}
        epoch_alignment_stats = {
            'steps': 0,
            'fusion_gradient_norm': 0.0,
            'pooling_entropy': 0.0,
            'missing_rates': {},
            'pooling_weights': {},
        }
        epoch_term_steps = 0
        progress_bar = tqdm(
            dataloader, desc=f"Pretraining Epoch {epoch + 1}/{args.epochs}", disable=rank != 0
        )
        optimizer.zero_grad(set_to_none=True)
        for step_idx, data in enumerate(progress_bar):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            global_step += 1
            data = data.to(device)
            should_step = (
                (step_idx + 1) % accumulation_steps == 0
                or (step_idx + 1) == len(dataloader)
                or (args.max_steps > 0 and global_step >= args.max_steps)
            )
            base_model = _base_model(model)
            loss_terms = {}

            if args.pretrain_stage == 'scage_m4p':
                zero = data.x.new_tensor(0.0)
                with torch.autocast(
                    device_type='cuda', dtype=torch.bfloat16,
                    enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                ):
                    mask_loss, mask_count = _mips_masked_atom_loss(
                        base_model, data, mips_atom_head,
                        mask_ratio=args.graph_mask_ratio, seed=args.seed, epoch=epoch,
                    )
                if mask_count and float(args.scage_mips_mask_weight) > 0:
                    loss_terms['mips_mask'] = mask_loss
                    epoch_target_counts['masked_atoms'] = (
                        epoch_target_counts.get('masked_atoms', 0) + mask_count
                    )

                distance_loss, distance_count = zero, 0
                graph_rep, node_rep = None, None
                if float(args.scage_screw_geometry_weight) > 0:
                    distance_loss, distance_count, graph_rep, node_rep = (
                        _periodic_lga_distance_loss(
                            base_model, data, m4p_distance_head,
                            max_pairs=args.scage_geometry_max_pairs,
                        )
                    )
                if graph_rep is None or node_rep is None:
                    graph_rep, node_rep = _graph_encode_nodes(base_model, data)
                if distance_count:
                    loss_terms['periodic_geometry'] = distance_loss
                    epoch_target_counts['geometry_distance_pairs'] = (
                        epoch_target_counts.get('geometry_distance_pairs', 0)
                        + distance_count
                    )
                    epoch_geometry_components['distance'] = (
                        epoch_geometry_components.get('distance', 0.0)
                        + float(distance_loss.detach().cpu().item())
                    )
                    epoch_geometry_component_steps['distance'] = (
                        epoch_geometry_component_steps.get('distance', 0) + 1
                    )

                ecfp_loss, ecfp_count = zero, 0
                if float(args.scage_ecfp_weight) > 0:
                    ecfp_loss, ecfp_count = _scage_ecfp_loss(
                        graph_rep, data, m4p_ecfp_head, pos_weight=m4p_ecfp_pos_weight
                    )
                if ecfp_count and float(args.scage_ecfp_weight) > 0:
                    loss_terms['ecfp'] = ecfp_loss
                    epoch_target_counts['ecfp_samples'] = epoch_target_counts.get('ecfp_samples', 0) + ecfp_count

                sp_loss, sp_count = zero, 0
                if float(args.scage_periodic_sp_weight) > 0:
                    sp_loss, sp_count = _periodic_lga_sp_loss(
                        data, node_rep, graph_sp_head,
                        max_pairs=args.scage_sp_max_pairs,
                    )
                if sp_count:
                    loss_terms['periodic_sp'] = sp_loss
                    epoch_target_counts['sp_pairs'] = epoch_target_counts.get('sp_pairs', 0) + sp_count

                if float(args.scage_periodic_contrast_weight) > 0:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        paug_loss, m4p_aug_stats = _scage_multi_positive_periodic_loss(
                            base_model, data, m4p_projection_head,
                            graph_input=args.graph_input,
                            temperature=args.periodic_cl_temperature,
                            max_mrus=args.periodic_aug_max_mrus,
                            retry=args.periodic_aug_retry,
                            views_per_polymer=args.periodic_aug_views,
                        )
                else:
                    paug_loss = zero
                    m4p_aug_stats = {
                        'views_requested': 0, 'views_valid': 0, 'failed_views': 0,
                        'duplicate_views': 0, 'valid_identities': 0,
                        'positive_pairs': 0, 'positive_count_sum': 0,
                        'identities_attempted': 0, 'distinct_cut_views': 0,
                    }
                graph_paug_stats = {
                    'attempted': m4p_aug_stats['views_requested'],
                    'success': m4p_aug_stats['views_valid'],
                    'skipped': m4p_aug_stats['failed_views'],
                    'same_smiles': m4p_aug_stats['duplicate_views'],
                    'valid_contrastive': m4p_aug_stats['valid_identities'],
                    'graph_cache_hits': m4p_aug_stats.get('graph_cache_hits', 0),
                    'graph_cache_misses': m4p_aug_stats.get('graph_cache_misses', 0),
                }
                _merge_periodic_aug_stats(epoch_periodic_aug_stats, graph_paug_stats)
                if m4p_aug_stats['valid_identities'] >= 2:
                    loss_terms['periodic_contrast'] = paug_loss
                for key, value in m4p_aug_stats.items():
                    stat_key = f'paug_{key}'
                    epoch_target_counts[stat_key] = epoch_target_counts.get(stat_key, 0) + int(value)

                zero_reference = next(base_model.parameters()).sum() * 0.0
                if dynamic_loss_weighter is not None:
                    loss = dynamic_loss_weighter(
                        loss_terms, reference=zero_reference
                    )
                    dynamic_weights = dynamic_loss_weighter.last_weights
                else:
                    priors = {
                        'mips_mask': args.scage_mips_mask_weight,
                        'ecfp': args.scage_ecfp_weight,
                        'periodic_sp': args.scage_periodic_sp_weight,
                        'periodic_geometry': args.scage_screw_geometry_weight * min(1.0, (epoch + 1) / 2.0),
                        'periodic_contrast': args.scage_periodic_contrast_weight,
                    }
                    denom = sum(priors[name] for name in loss_terms)
                    loss = (
                        sum(priors[name] * value for name, value in loss_terms.items())
                        / max(denom, 1e-8)
                        if loss_terms else zero_reference
                    )
                    dynamic_weights = {}

                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite M4P loss for batch smiles={data.smiles}")
                (loss / accumulation_steps).backward()
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    _all_reduce_gradients((model, aux_modules))
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(aux_modules.parameters()), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.item())
                gradient_norms.append(float(gradient_norm.detach().cpu().item()))
                epoch_term_steps += 1
                for name, value in loss_terms.items():
                    epoch_raw_terms[name] = epoch_raw_terms.get(name, 0.0) + float(value.detach().cpu().item())
                    epoch_raw_term_counts[name] = epoch_raw_term_counts.get(name, 0) + 1
                if dynamic_loss_weighter is not None:
                    for name, value in dynamic_loss_weighter.last_normalized.items():
                        epoch_normalized_terms[name] = epoch_normalized_terms.get(name, 0.0) + float(value)
                    for name, value in dynamic_loss_weighter.last_weights.items():
                        epoch_dynamic_weights[name] = epoch_dynamic_weights.get(name, 0.0) + float(value)
                    for name, value in dynamic_loss_weighter.last_effective.items():
                        epoch_effective_weights[name] = epoch_effective_weights.get(name, 0.0) + float(value)
                    for name, value in dynamic_loss_weighter.last_baselines.items():
                        if value is not None:
                            epoch_baselines[name] = float(value)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mask=f"{mask_loss.item():.3f}",
                    sp=f"{sp_loss.item():.3f}",
                    distance=f"{distance_loss.item():.3f}",
                )
                continue

            if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage':
                with torch.autocast(
                    device_type='cuda', dtype=torch.bfloat16,
                    enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                ):
                    alignment_losses, alignment_stats = _scage_semantic_alignment_losses(
                        base_model, data, args
                    )
                    fused_mask_loss, fused_mask_count = _fusion_conditioned_masked_atom_loss(
                        base_model, data, base_model.alignment_mask_head, args.graph_mask_ratio
                    )
                    alignment_losses['fused_mask'] = fused_mask_loss
                loss = (
                    args.alignment_fused_weight * alignment_losses['fused_view']
                    + args.alignment_graph_smiles_weight * alignment_losses['graph_smiles']
                    + args.alignment_graph_fp_weight * alignment_losses['graph_fp']
                    + args.alignment_lomo_weight * alignment_losses['lomo']
                    + args.alignment_pooling_kl_weight * alignment_losses['pooling_kl']
                    + args.alignment_fused_mask_weight * alignment_losses['fused_mask']
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite parallel alignment loss for batch smiles={data.smiles}")
                (loss / accumulation_steps).backward()
                fusion_gradient = _module_gradient_norm(base_model.parallel_attention_fusion)
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    _all_reduce_gradients((model, aux_modules))
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.item())
                gradient_norms.append(float(gradient_norm.detach().cpu().item()))
                epoch_term_steps += 1
                for name, value in alignment_losses.items():
                    epoch_raw_terms[name] = epoch_raw_terms.get(name, 0.0) + float(value.item())
                epoch_alignment_stats['steps'] += 1
                epoch_alignment_stats['fusion_gradient_norm'] += fusion_gradient
                epoch_alignment_stats['pooling_entropy'] += alignment_stats['pooling_entropy']
                for group in ('missing_rates', 'pooling_weights'):
                    for name, value in alignment_stats[group].items():
                        target = epoch_alignment_stats[group]
                        target[name] = target.get(name, 0.0) + float(value)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    fusion=f"{alignment_losses['fused_view'].item():.3f}",
                    gs=f"{alignment_losses['graph_smiles'].item():.3f}",
                    gf=f"{alignment_losses['graph_fp'].item():.3f}",
                    lomo=f"{alignment_losses['lomo'].item():.3f}",
                )
                continue

            geom_loss = data.y.new_tensor(0.0)
            if geom_noise_head is not None:
                geom_loss = _geom_denoise_loss(
                    base_model,
                    data,
                    geom_noise_head,
                    noise_std=args.geom_noise_std,
                    noise_std_min=args.geom_noise_std_min,
                    noise_std_max=args.geom_noise_std_max,
                )
                loss_terms['geom'] = geom_loss

            contrastive_loss = data.y.new_tensor(0.0)
            if loss_weights['contrastive'] > 0:
                _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
                contrastive_loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
                loss_terms['contrastive'] = contrastive_loss

            graph_loss = data.y.new_tensor(0.0)
            if graph_atom_head is not None:
                graph_loss, graph_mask_loss, graph_paug_loss, graph_paug_stats = _graph_pretrain_loss(
                    base_model,
                    data,
                    graph_atom_head,
                    mask_ratio=args.graph_mask_ratio,
                    mask_weight=args.graph_mask_atom_weight,
                    periodic_aug_weight=args.graph_periodic_aug_weight,
                    graph_input=args.graph_input,
                    periodic_aug_max_mrus=args.periodic_aug_max_mrus,
                    periodic_aug_retry=args.periodic_aug_retry,
                    periodic_cl_temperature=args.periodic_cl_temperature,
                )
                _merge_periodic_aug_stats(epoch_periodic_aug_stats, graph_paug_stats)
                loss_terms['graph'] = graph_loss
            else:
                graph_mask_loss = data.y.new_tensor(0.0)
                graph_paug_loss = data.y.new_tensor(0.0)
                graph_paug_stats = _empty_periodic_aug_stats()

            graph_sp_loss = data.y.new_tensor(0.0)
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                graph_sp_loss = _graph_shortest_path_loss(
                    base_model,
                    data,
                    graph_sp_head,
                    max_distance=args.scage_sp_max_distance,
                )

            graph_angle_loss = None
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                graph_angle_loss = _graph_angle_loss(
                    base_model,
                    data,
                    graph_angle_head,
                    angle_bins=args.scage_angle_bins,
                )
            graph_angle_log = graph_angle_loss if graph_angle_loss is not None else data.y.new_tensor(0.0)

            weighted_loss_terms = {}
            if args.pretrain_stage == 'graph_geom':
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    if float(args.graph_mask_atom_weight) > 0:
                        weighted_loss_terms['graph_mask'] = float(args.graph_mask_atom_weight) * graph_mask_loss
                    if float(args.graph_periodic_aug_weight) > 0:
                        weighted_loss_terms['graph_periodic_aug'] = float(args.graph_periodic_aug_weight) * graph_paug_loss
                    if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                        weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                    if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                        weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss
            else:
                if loss_weights['contrastive'] > 0 and 'contrastive' in loss_terms:
                    weighted_loss_terms['contrastive'] = loss_weights['contrastive'] * contrastive_loss
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    weighted_loss_terms['graph'] = loss_weights['graph'] * graph_loss
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                    weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss

            if dynamic_loss_weighter is not None:
                loss = dynamic_loss_weighter(weighted_loss_terms)
                dynamic_weights = dynamic_loss_weighter.last_weights
            elif args.pretrain_stage == 'graph_geom' and weighted_loss_terms:
                loss = sum(weighted_loss_terms.values()) / len(weighted_loss_terms)
                dynamic_weights = {}
            else:
                loss = sum(weighted_loss_terms.values()) if weighted_loss_terms else data.y.new_tensor(0.0)
                dynamic_weights = {}


            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite pretraining loss for batch smiles={getattr(data, 'smiles', [])}")
            (loss / accumulation_steps).backward()
            gradient_norm = loss.new_tensor(0.0)
            if should_step:
                _all_reduce_gradients((model, aux_modules))
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(aux_modules.parameters()), max_norm=args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            gradient_norms.append(float(gradient_norm.detach().cpu().item()))
            epoch_term_steps += 1
            for name, value in weighted_loss_terms.items():
                epoch_raw_terms[name] = epoch_raw_terms.get(name, 0.0) + float(value.detach().cpu().item())
                epoch_raw_term_counts[name] = epoch_raw_term_counts.get(name, 0) + 1
            if dynamic_loss_weighter is not None:
                for name, value in dynamic_loss_weighter.last_normalized.items():
                    epoch_normalized_terms[name] = epoch_normalized_terms.get(name, 0.0) + float(value)
                for name, value in dynamic_weights.items():
                    epoch_dynamic_weights[name] = epoch_dynamic_weights.get(name, 0.0) + float(value)
            paug_attempted = int(graph_paug_stats.get('attempted', 0))
            paug_success = int(graph_paug_stats.get('success', 0))
            paug_rate = (paug_success / paug_attempted) if paug_attempted else 0.0
            postfix = {
                'loss': f"{loss.item():.4f}",
                'con': f"{contrastive_loss.item():.4f}",
                'g_mask': f"{graph_mask_loss.item():.4f}",
                'g_paug': f"{graph_paug_loss.item():.4f}",
                'paug_rate': f"{paug_rate:.2f}",
                'paug_valid': int(graph_paug_stats.get('valid_contrastive', 0)),
                'paug_same': int(graph_paug_stats.get('same_smiles', 0)),
                'geom': f"{geom_loss.item():.4f}",
                'sp': f"{graph_sp_loss.item():.4f}",
                'angle': f"{graph_angle_log.item():.4f}",
            }
            for name, value in dynamic_weights.items():
                postfix[f'w_{name}'] = f"{value:.2f}"
            progress_bar.set_postfix(**postfix)
        epoch_summary = torch.tensor(
            [float(epoch_loss), float(epoch_term_steps)], dtype=torch.float64, device=device
        )
        if distributed:
            dist.all_reduce(epoch_summary, op=dist.ReduceOp.SUM)
        global_epoch_steps = max(1, int(round(epoch_summary[1].item())))
        avg_loss = float(epoch_summary[0].item()) / global_epoch_steps
        epoch_raw_terms = _distributed_sum_mapping(epoch_raw_terms, device)
        epoch_raw_term_counts = _distributed_sum_mapping(epoch_raw_term_counts, device)
        epoch_normalized_terms = _distributed_sum_mapping(epoch_normalized_terms, device)
        epoch_dynamic_weights = _distributed_sum_mapping(epoch_dynamic_weights, device)
        epoch_effective_weights = _distributed_sum_mapping(epoch_effective_weights, device)
        epoch_target_counts = _distributed_sum_mapping(epoch_target_counts, device)
        epoch_geometry_components = _distributed_sum_mapping(epoch_geometry_components, device)
        epoch_geometry_component_steps = _distributed_sum_mapping(
            epoch_geometry_component_steps, device
        )
        epoch_periodic_aug_stats = _distributed_sum_mapping(
            epoch_periodic_aug_stats, device
        )
        alignment_summary = torch.tensor([
            float(epoch_alignment_stats['steps']),
            float(epoch_alignment_stats['fusion_gradient_norm']),
            float(epoch_alignment_stats['pooling_entropy']),
        ], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(alignment_summary, op=dist.ReduceOp.SUM)
        epoch_alignment_stats['steps'] = int(round(alignment_summary[0].item()))
        epoch_alignment_stats['fusion_gradient_norm'] = float(alignment_summary[1].item())
        epoch_alignment_stats['pooling_entropy'] = float(alignment_summary[2].item())
        epoch_alignment_stats['missing_rates'] = _distributed_sum_mapping(
            epoch_alignment_stats['missing_rates'], device
        )
        epoch_alignment_stats['pooling_weights'] = _distributed_sum_mapping(
            epoch_alignment_stats['pooling_weights'], device
        )
        losses.append(avg_loss)
        paug_summary = _finalize_periodic_aug_stats(epoch + 1, epoch_periodic_aug_stats)
        periodic_aug_epoch_stats.append(paug_summary)
        if dynamic_loss_weighter is not None:
            dynamic_loss_epoch_weights.append({
                'epoch': int(epoch + 1),
                'weights': dict(dynamic_loss_weighter.last_weights),
            })
        if global_epoch_steps:
            dynamic_loss_epoch_stats.append({
                'epoch': int(epoch + 1),
                'raw_weighted_losses': {
                    name: value / max(epoch_raw_term_counts.get(name, 1.0), 1.0)
                    for name, value in epoch_raw_terms.items()
                },
                'normalized_losses': {
                    name: value / global_epoch_steps for name, value in epoch_normalized_terms.items()
                },
                'dynamic_weights': {
                    name: value / global_epoch_steps for name, value in epoch_dynamic_weights.items()
                },
                'task_priors': (
                    dict(dynamic_loss_weighter.task_priors)
                    if dynamic_loss_weighter is not None else {}
                ),
                'effective_coefficients': {
                    name: value / global_epoch_steps for name, value in epoch_effective_weights.items()
                },
                'frozen_baselines': dict(epoch_baselines),
            })
        if args.pretrain_stage == 'scage_m4p':
            valid_ids = epoch_target_counts.get('paug_valid_identities', 0)
            epoch_target_counts['paug_average_positives_per_polymer'] = (
                epoch_target_counts.get('paug_positive_count_sum', 0) / valid_ids
                if valid_ids else 0.0
            )
            m4p_epoch_target_stats.append({'epoch': epoch + 1, **epoch_target_counts})
            m4p_geometry_epoch_stats.append({
                'epoch': epoch + 1,
                'component_losses': {
                    name: value / max(epoch_geometry_component_steps.get(name, 1), 1)
                    for name, value in epoch_geometry_components.items()
                },
                'component_steps': dict(epoch_geometry_component_steps),
            })
        if epoch_alignment_stats['steps']:
            steps = epoch_alignment_stats['steps']
            alignment_epoch_stats.append({
                'epoch': epoch + 1,
                'fusion_gradient_norm': epoch_alignment_stats['fusion_gradient_norm'] / steps,
                'pooling_entropy': epoch_alignment_stats['pooling_entropy'] / steps,
                'missing_rates': {
                    name: value / steps
                    for name, value in epoch_alignment_stats['missing_rates'].items()
                },
                'pooling_weights': {
                    name: value / steps
                    for name, value in epoch_alignment_stats['pooling_weights'].items()
                },
            })
        if rank == 0 and paug_summary['attempted'] > 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f} | "
                f"PerioGT aug success: {paug_summary['success']}/{paug_summary['attempted']} "
                f"= {paug_summary['success_rate']:.2%}, "
                f"same_smiles={paug_summary['same_smiles']}, skipped={paug_summary['skipped']}"
            )
        elif rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if distributed and rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    # Save original loss data
    loss_data = {
        'pretrain_stage': args.pretrain_stage,
        'fusion_type': args.fusion_type,
        'fp_mode': args.fp_mode,
        'parallel_attention_layers': args.parallel_attention_layers,
        'pretraining_rows': int(len(indices)),
        'pretraining_unique_smiles': bool(args.pretrain_unique_smiles),
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
        'scage_ffn_hidden_dim': args.scage_ffn_hidden_dim,
        'scage_num_kernels': args.scage_num_kernels,
        'scage_use_descriptors': bool(args.scage_use_descriptors),
        'scage_distance_mode': args.scage_distance_mode,
        'scage_distance_rbf': args.scage_distance_rbf,
        'scage_distance_cutoff': args.scage_distance_cutoff,
        'scage_distance_scales': args.scage_distance_scales,
        'scage_distance_taus': args.scage_distance_taus,
        'scage_topology_bias': bool(args.scage_topology_bias),
        'scage_topology_max_distance': args.scage_topology_max_distance,
        'scage_topology_locality_mode': args.scage_topology_locality_mode,
        'scage_topology_locality_threshold': args.scage_topology_locality_threshold,
        'scage_topology_locality_tau': args.scage_topology_locality_tau,
        'scage_periodic_image_mode': args.scage_periodic_image_mode,
        'scage_periodic_image_cap': args.scage_periodic_image_cap,
        'scage_periodic_image_temperature': args.scage_periodic_image_temperature,
        'scage_force_topology_only': bool(args.scage_force_topology_only),
        'scage_num_heads': args.scage_num_heads,
        'optimizer_schedule': {
            'type': 'linear_warmup_cosine_decay',
            'batch_size_per_rank': int(args.batch_size),
            'world_size': int(world_size),
            'effective_batch_size': int(args.batch_size) * int(world_size) * accumulation_steps,
            'total_optimizer_steps': total_optimizer_steps,
            'warmup_steps': warmup_steps,
            'warmup_ratio': float(args.warmup_ratio),
            'gradient_accumulation_steps': accumulation_steps,
        },
        'epochs': list(range(1, len(losses) + 1)),
        'losses': losses,
        'periodic_aug_stats': periodic_aug_epoch_stats,
        'dynamic_pretrain_loss': bool(args.dynamic_pretrain_loss),
        'dynamic_loss_config': ({
            'distributed_statistics': 'all_reduce_mean_over_valid_ranks',
            'warmup_steps': int(args.dynamic_loss_warmup_steps),
            'recent_window': int(args.dynamic_loss_recent_window),
            'temperature': float(args.dynamic_loss_temperature),
        } if args.dynamic_pretrain_loss else None),
        'dynamic_loss_weights': dynamic_loss_epoch_weights,
        'dynamic_loss_epoch_stats': dynamic_loss_epoch_stats,
        'm4p_target_stats': m4p_epoch_target_stats,
        'm4p_geometry_stats': m4p_geometry_epoch_stats,
        'm4p_geometry_objective': ({
            'version': 'pbc-lga-euclidean-distance-v1',
            'descriptor_shortcut_disabled': True,
            'geometry_max_pairs': int(args.scage_geometry_max_pairs),
            'target': 'topology-selected-valid-lga-edge-distance',
            'loss': 'smooth_l1',
            'target_edge_distance_bias_masked': True,
            'angle': False,
            'torsion': False,
            'image_shift_classification': False,
        } if args.pretrain_stage == 'scage_m4p' else None),
        'm4p_ecfp_pos_weight': ({
            'valid_samples': int(m4p_ecfp_valid_count),
            'min': float(m4p_ecfp_pos_weight.min().item()),
            'mean': float(m4p_ecfp_pos_weight.mean().item()),
            'max': float(m4p_ecfp_pos_weight.max().item()),
        } if m4p_ecfp_pos_weight is not None else None),
        'alignment_epoch_stats': alignment_epoch_stats,
        'alignment_objective': ({
            'version': 'fusion-view-lomo-v1',
            'weights': {
                'fused_view': args.alignment_fused_weight,
                'graph_smiles': args.alignment_graph_smiles_weight,
                'graph_fp': args.alignment_graph_fp_weight,
                'lomo': args.alignment_lomo_weight,
                'pooling_kl': args.alignment_pooling_kl_weight,
                'fused_mask': args.alignment_fused_mask_weight,
            },
            'dropout': {
                'fp': args.alignment_fp_drop,
                'smiles': args.alignment_smiles_drop,
                'graph': args.alignment_graph_drop,
            },
        } if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'scage' else None),
        'gradient_norm': {
            'mean': float(np.mean(gradient_norms)) if gradient_norms else None,
            'max': float(np.max(gradient_norms)) if gradient_norms else None,
        },
    }
    artifact_tag = os.path.splitext(os.path.basename(args.save_path))[0]
    loss_json_path = os.path.join('./plots/pretrain', f'{artifact_tag}_loss_data.json')
    loss_curve_path = os.path.join('./plots/pretrain', f'{artifact_tag}_loss_curve.png')
    with open(loss_json_path, 'w') as f:
        json.dump(loss_data, f, indent=4)
    print(f"Loss data saved at {loss_json_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('Pretraining Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Total Pretraining Loss')
    plt.grid(True)
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved at {loss_curve_path}")

    state_dict = model.state_dict()
    if args.graph_encoder_type == 'scage':
        torch.save({
            'state_dict': state_dict,
            'meta': {
                'schema': 'scage-mips-pbc-pyg-v1',
                'stage': args.pretrain_stage,
                'modalities': list(args.modalities),
                'graph_input': args.graph_input,
                'geom_input': args.geom_input,
                'fusion_type': args.fusion_type,
                'parallel_attention_layers': args.parallel_attention_layers,
                'pretraining_unique_smiles': bool(args.pretrain_unique_smiles),
                'feature_schema': 'scage-mips-pbc-lga-v1',
                'periodic_lga_schema_version': 1,
                'periodic_geometry_schema': 'polygen-pbc-v14-expanded-supercell',
                'strict_topology_fallback': True,
                'model': {
                    'architecture': 'mips_periodic_pyg_sparse_lga',
                    'layers': 6,
                    'embedding_dim': 512,
                    'heads': 8,
                    'ffn_hidden_dim': 2048,
                    'max_hops': 5,
                    'path_nodes': 6,
                    'distance_rbf': 64,
                    'distance_cutoff_angstrom': 12.0,
                    'readout': 'atom_mean',
                    'descriptors': False,
                    'graph_token': False,
                },
                'dynamic_loss_config': ({
                    'distributed_statistics': 'all_reduce_mean_over_valid_ranks',
                    'warmup_steps': int(args.dynamic_loss_warmup_steps),
                    'recent_window': int(args.dynamic_loss_recent_window),
                    'temperature': float(args.dynamic_loss_temperature),
                    'task_priors': dict(dynamic_loss_weighter.task_priors),
                } if dynamic_loss_weighter is not None else None),
                'optimizer_schedule': {
                    'type': 'linear_warmup_cosine_decay',
                    'batch_size_per_rank': int(args.batch_size),
                    'world_size': int(world_size),
                    'effective_batch_size': int(args.batch_size) * int(world_size) * accumulation_steps,
                    'warmup_ratio': float(args.warmup_ratio),
                    'total_optimizer_steps': total_optimizer_steps,
                },
                'amp_dtype': args.amp_dtype,
                'm4p_priors': {
                    'mips_mask': args.scage_mips_mask_weight,
                    'periodic_sp': args.scage_periodic_sp_weight,
                    'periodic_geometry': args.scage_screw_geometry_weight,
                },
                'm4p_geometry_objective': {
                    'version': 'pbc-lga-euclidean-distance-v1',
                    'distance_weight': float(args.scage_screw_geometry_weight),
                    'max_pairs': int(args.scage_geometry_max_pairs),
                    'target_edge_distance_bias_masked': True,
                },
                'alignment_objective': (
                    'fusion-view-lomo-fused-mask-v2'
                    if args.pretrain_stage == 'alignment' else None
                ),
                'fusion_dropout': args.fusion_dropout,
            },
        }, args.save_path)
    else:
        torch.save(state_dict, args.save_path)
    print(f"Pretrained model saved at {args.save_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
