"""Runtime configuration extraction for the standalone pretraining loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.dataset.mips_trimer_contract import ROUTE_INTERNAL as MTS_ROUTE_INTERNAL


@dataclass(frozen=True)
class PretrainRuntimeConfig:
    batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    scheduler: str
    amp_dtype: str
    seed: int
    world_size: int

    @classmethod
    def from_args(cls, args, *, world_size: int) -> "PretrainRuntimeConfig":
        return cls(
            batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            max_optimizer_steps=int(args.max_optimizer_steps),
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            warmup_steps=int(args.warmup_steps),
            scheduler=str(args.mips_scheduler),
            amp_dtype=str(args.amp_dtype),
            seed=int(args.seed),
            world_size=int(world_size),
        )


def dataset_kwargs_from_args(args):
    """Translate parsed CLI values into the Dataset constructor contract."""
    return dict(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
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
        cache_layers=args.cache_layers,
        cache_validate=args.cache_validate,
        cache_commit_size=args.cache_commit_size,
        embed_tries_multiplier=args.embed_tries_multiplier,
        conformer_3d_count=args.conformer_3d_count,
        conformer_keep_count=args.conformer_keep_count,
        conformer_profile=args.conformer_profile,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        mips_core=args.mips_core,
        mips_max_hops=args.mips_max_hops,
        mips_use_descriptors=args.mips_use_descriptors,
        mips_descriptor_protocol=args.mips_descriptor_protocol,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        topology_representation=args.topology_representation,
        mcl_distance_percentiles=args.mcl_distance_percentiles,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        finite_variant=args.finite_variant,
        conformer_mode=args.conformer_mode,
        field_layout=args.field_layout,
        field_channels=args.field_channels,
        experiment_id=args.experiment_id,
        feature_config_hash="manual",
        star_rbf_v2_sidecar=args.star_rbf_v2_sidecar,
        angle_cache_root_override=getattr(args, 'angle_cache_root_override', None),
    )
SUPPORTED_MODALITIES = ('graph', 'smiles', 'fp')



def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain UniEncoderAttention Model")
    parser.add_argument('--experiment_id', default='manual')
    parser.add_argument('--config_schema', default='manual')
    parser.add_argument('--config_source_schema', default='')
    parser.add_argument(
        '--modalities',
        nargs='+',
        type=parse_modality,
        default=['graph'],
        help="MTS pretraining uses the graph modality only."
    )

    parser.add_argument(
        '--fusion_type',
        type=str,
        choices=['none'],
        default='none',
        help="MTS joint pretraining has no multimodal fusion.",
    )
    parser.add_argument(
        '--fp_mode',
        type=str,
        choices=['disabled', 'ecfp', 'mixfp', 'attachment_count'],
        default='ecfp',
        help="Fingerprint implementation. ecfp keeps the original Morgan/ECFP 1024-bit FP; mixfp uses MACCSKeys + PubChemFingerprints.",
    )
    parser.add_argument('--fusion_dropout', type=float, default=0.0, help=argparse.SUPPRESS)
    # Accepted for the shared launcher COMMON vector; joint pretraining does
    # not consume per-modality dropout, the defaults mirror train.py.
    parser.add_argument('--smiles_modality_dropout', type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument('--fp_modality_dropout', type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument('--graph_modality_dropout', type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument(
        '--mips_fusion_mode',
        choices=['none'],
        default='none',
    )
    parser.add_argument(
        '--projection_mode',
        choices=['plain', 'shared_private'],
        default='shared_private',
    )
    parser.add_argument(
        '--modality_control',
        choices=['real', 'batch_shuffled', 'constant_zero'],
        default='real',
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
        '--dataset_name',
        type=str,
        default='smi_all',
        help="Name of the dataset for pretraining (unlabeled or labeled, but labels unused here)"
    )
    parser.add_argument(
        '--pretrain_stage',
        type=str,
        choices=['mts_joint_pretraining'],
        default='mts_joint_pretraining',
        help=(
            "MTS pretraining uses one joint masked-atom + Trimer bond-angle "
            "stage; property fine-tuning is handled by train.py."
        )
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='',
        help="Optional MTS joint-pretraining checkpoint to initialize or resume the fixed topology/Trimer stage."
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
    parser.add_argument('--loader_prefetch_factor', type=int, default=2)
    parser.add_argument(
        '--batch_balance', choices=['none', 'cost'], default='none',
        help='Optional topology cost-balanced distributed batches.',
    )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--amp_dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--max_steps', type=int, default=0, help='Optional preflight step limit; 0 disables it.')
    parser.add_argument(
        '--max_optimizer_steps', type=int, default=0,
        help='Exact optimizer-update budget; 0 uses the epoch budget.',
    )
    parser.add_argument(
        '--benchmark_only', action='store_true',
        help='Run a finite forward/backward throughput benchmark and exit.',
    )
    parser.add_argument(
        '--resume_smoke', action='store_true',
        help='Test-only short run; never writes a production checkpoint.',
    )
    parser.add_argument(
        '--resume_smoke_stop_steps', type=int, default=0,
        help='Optional early stop for a resume smoke while preserving the target schedule.',
    )
    parser.add_argument('--benchmark_batches', type=int, default=0)
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
        default=-1.0,
        help="Maximum gradient norm; values <=0 disable clipping for MTS."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pretrained_models/saved_pretrained_model.pth',
        help="Path to save the pretrained model."
    )
    parser.add_argument(
        '--resume_state', type=str, default='',
        help='Optional atomic pretraining train-state checkpoint to resume.'
    )
    parser.add_argument(
        '--checkpoint_interval_steps', type=int, default=2000,
        help='Optimizer steps between resumable train-state checkpoints.'
    )
    parser.add_argument(
        '--mts_num_layers',
        type=int,
        default=6,
        dest='graph_num_layers',
        help="Number of O8 MTS topology layers (fixed at 6)."
    )
    parser.add_argument(
        '--mts_hidden_dim',
        type=int,
        default=512,
        dest='graph_emb_dim',
        help="O8 MTS hidden dimension (fixed at 512)."
    )
    parser.add_argument(
        '--mts_dropout',
        type=float,
        default=0.1,
        dest='graph_dropout',
        help="O8 MTS attention dropout."
    )
    parser.add_argument(
        '--mts_num_heads',
        type=int,
        default=8,
        dest='scage_num_heads',
        help="Number of O8 MTS attention heads (fixed at 8)."
    )
    parser.add_argument(
        '--graph_encoder_type',
        type=str,
        choices=['mts', 'mips_trimer_scage'],
        default='mips_trimer_scage',
        help="Graph encoder backend: the production non-PBC MIPS-Trimer-SCAGE encoder."
    )
    parser.add_argument(
        '--mips_core',
        choices=['paper_corrected'],
        default='paper_corrected',
    )
    parser.add_argument('--mips_max_hops', type=int, default=None)
    parser.add_argument('--mips_atom_feature_mode', choices=['mips137'], default='mips137')
    parser.add_argument('--mips_attention_scale', choices=['head_dim'], default='head_dim')
    parser.add_argument('--mips_norm_mode', choices=['post'], default='post')
    parser.add_argument('--mips_activation', choices=['relu'], default='relu')
    parser.add_argument('--mips_spd_bias_mode', choices=['per_head'], default='per_head')
    parser.add_argument(
        '--mips_path_bias_mode',
        choices=['per_head_single_path_node'],
        default='per_head_single_path_node',
    )
    parser.add_argument(
        '--mips_multi_scale_hop_gate',
        action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        '--mips_semantics',
        choices=['paper_semantic'],
        default='paper_semantic',
    )
    parser.add_argument(
        '--mips_descriptor_fusion_mode',
        choices=['graph_md_residual'],
        default='graph_md_residual',
    )
    parser.add_argument(
        '--mips_descriptor_components',
        choices=['md200'],
        default='md200',
    )
    parser.add_argument(
        '--mips_descriptor_protocol',
        choices=['source_star_sub'],
        default='source_star_sub',
    )
    parser.add_argument('--mips_descriptor_disturbance', type=float, default=0.0)
    parser.add_argument(
        '--mips_backbone_mode',
        choices=['independent'],
        default='independent',
    )
    parser.add_argument(
        '--mips_input_norm',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        '--mips_mask_mode', choices=['zero'], default='zero',
    )
    parser.add_argument(
        '--mips_mask_policy',
        choices=['canonical_exact'],
        default='canonical_exact',
    )
    parser.add_argument(
        '--mips_masked_loss_reduction',
        choices=['atom_mean'],
        default='atom_mean',
    )
    parser.add_argument(
        '--mips_use_descriptors',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--spatial_mode',
        choices=['trimer_scage'],
        default='trimer_scage',
    )
    parser.add_argument(
        '--graph_geometry_mode',
        choices=['trimer_scage_mcl'],
        default='trimer_scage_mcl',
    )
    parser.add_argument(
        '--topology_attention_variant',
        choices=['o8', 'msta_last2'],
        default='msta_last2',
        help='O8 attention or MSTA in the final two layers.',
    )
    parser.add_argument('--msta_layer_indices', nargs=2, type=int, default=[4, 5])
    parser.add_argument('--msta_local_spd', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--msta_context_spd', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument(
        '--msta_share_relation_dropout',
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--msta_local_output_bias',
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument('--msta_local_output_init', choices=['zero'], default='zero')
    parser.add_argument('--star_rbf_definition', choices=['legacy_sample_direct_link_v1', 'trimer_periodic_relation_rbf_v2'], default='legacy_sample_direct_link_v1')
    parser.add_argument('--star_rbf_upper', type=float, default=3.0)
    parser.add_argument('--star_rbf_v2_sidecar', default=None)
    parser.add_argument('--pretraining_objective', choices=['joint', 'masked_atom_only'], default='joint')
    parser.add_argument('--angle_loss_weight', type=float, default=0.25)
    parser.add_argument(
        '--topology_representation',
        choices=['canonical_lifted', 'explicit_k_ru'],
        default='canonical_lifted',
    )
    parser.add_argument(
        '--mcl_distance_percentiles', nargs=2, type=float,
        default=[0.20, 0.50],
    )
    parser.add_argument('--trimer_num_candidates', type=int, default=4)
    parser.add_argument('--trimer_max_heavy_atoms', type=int, default=384)
    parser.set_defaults(
        finite_variant='none', conformer_mode='none',
        field_layout='none', field_channels='none',
    )
    parser.add_argument(
        '--mips_variant',
        choices=['O8'],
        default='O8',
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
        '--cache_layers',
        type=str,
        default='ru_base,topology,trimer,md200',
        help=(
            "Comma-separated MIPS LMDB layers to prepare/read: "
            "ru_base,topology,trimer,md200."
        ),
    )
    parser.add_argument(
        '--cache_validate',
        choices=['sample', 'full'],
        default='sample',
        help="Validate 128 records or the complete LMDB cohort after building.",
    )
    parser.add_argument(
        '--cache_commit_size',
        type=int,
        default=128,
        help="Maximum number of generated records per LMDB write transaction.",
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
        default=0.25,
        help="Weight for SCAGE-style 3D angle prediction."
    )
    parser.add_argument(
        '--angle_objective',
        choices=['categorical', 'cosine'],
        default='categorical',
        help=(
            'Trimer angle target. categorical preserves the completed '
            '20-bin checkpoint; cosine enables the optional Angle-v2 '
            'SmoothL1 experiment.'
        ),
    )
    parser.add_argument(
        '--angle_cache_root_override', default=None,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        '--mips_spd_weight',
        type=float, default=0.0,
    )
    parser.add_argument('--mips_path_bond_weight', type=float, default=0.0)
    parser.set_defaults(scage_screw_geometry_weight=0.0)
    parser.set_defaults(
        mips_repeat_consistency_weight=0.0,
        mips_distance_weight=0.0,
        mips_conformer_weight=0.0,
    )
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
    parser.add_argument('--repeat_cut_views', type=int, default=2)
    parser.add_argument('--repeat_cut_projection_dim', type=int, default=256)
    parser.add_argument(
        '--repeat_cut_max_mrus',
        type=int,
        default=3,
        help="Maximum MRU multiplier used by periodicity augmentation."
    )
    parser.add_argument(
        '--repeat_cut_retry',
        type=int,
        default=5,
        help="Number of augmentation attempts per SMILES before skipping it."
    )
    parser.add_argument(
        '--repeat_cut_temperature',
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
    parser.add_argument(
        '--alignment_shared_private_weight', type=float, default=0.01
    )
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
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--scheduler_power', type=float, default=1.0)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument(
        '--mips_scheduler', choices=('polynomial', 'cosine', 'linear'), default='polynomial',
        help='Learning-rate decay after warm-up. MTS uses polynomial power 1.',
    )
    args = parser.parse_args()
    # Fixed MTS route internals.  Retired geometry and peer-fusion options are
    # deliberately absent from the public CLI.
    args.gnn_model_name = ""
    args.freeze_encoder = False
    args.geom_input = 'repeat_unit'
    args.screw_kabsch_rmsd_max = 1.5
    args.screw_rotation_consistency_deg = 30.0
    args.screw_translation_relative_max = 0.30
    args.screw_final_rmsd_max = 1.5
    args.ff_gradient_rms_max = 0.05
    args.ff_gradient_max = 0.25
    args.ff_probe_steps = 20
    args.ff_probe_energy_delta_per_atom_max = 5e-5
    args.screw_energy_per_atom_max = 5.0
    args.screw_center_gradient_rms_max = 10.0
    args.scage_dist_bar = [20.0, 50.0]
    args.scage_num_heads = 8
    args.scage_ffn_hidden_dim = 2048
    args.scage_num_kernels = 128
    args.scage_attention_dropout = 0.1
    args.scage_use_descriptors = False
    args.scage_distance_mode = 'mips_dual'
    args.scage_distance_rbf = 32
    args.scage_distance_cutoff = 12.0
    args.scage_distance_scales = [4.0, 8.0, 12.0]
    args.scage_distance_taus = [0.5, 1.0, 1.5]
    args.scage_topology_bias = True
    args.scage_topology_max_distance = 20
    args.scage_topology_locality_mode = 'soft'
    args.scage_topology_locality_threshold = 5
    args.scage_topology_locality_tau = 1.0
    args.scage_periodic_image_mode = 'none'
    args.scage_periodic_image_cap = 0
    args.scage_periodic_image_temperature = 0.5
    args.scage_force_topology_only = True
    args.scage_use_pbc_distance = False
    # ``mts`` is the public backend spelling.  Keep the historical internal
    # selector so Dataset/model code and immutable cache metadata remain
    # compatible during the naming migration.
    if args.graph_encoder_type == "mts":
        args.graph_encoder_type = MTS_ROUTE_INTERNAL
    return args
