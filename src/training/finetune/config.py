"""Configuration parsing and normalization for one MTS fine-tune job."""

from __future__ import annotations

from dataclasses import dataclass
import argparse

from src.dataset.mips_trimer_contract import ROUTE_INTERNAL as MTS_ROUTE_INTERNAL

SUPPORTED_MODALITIES = ('graph', 'smiles', 'fp')
SUPPORTED_FUSION_TYPES = ('none', 'zero_gated_residual')
CROSS_TASK_AUXILIARY_MAP = {
    task: tuple(other for other in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc') if other != task)
    for task in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
}


@dataclass(frozen=True)
class FinetuneJobConfig:
    task: str
    seed: int
    fold: int
    experiment_id: str = "manual"

    def validate(self) -> None:
        if not self.task:
            raise ValueError("fine-tune task is required")
        if int(self.fold) not in range(5):
            raise ValueError("fine-tune fold must be in [0, 4]")


def normalize_job(task: str, seed: int, fold: int, experiment_id: str = "manual") -> FinetuneJobConfig:
    config = FinetuneJobConfig(str(task), int(seed), int(fold), str(experiment_id))
    config.validate()
    return config


def parse_modality(value):
    if value not in SUPPORTED_MODALITIES:
        raise argparse.ArgumentTypeError(
            f"Unsupported modality: {value}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )
    return value

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Train UniEncoderAttention Model")
    parser.add_argument('--experiment_id', default='manual')
    parser.add_argument('--predictions_dir', default='')
    parser.add_argument(
        '--checkpoint_seed', type=int, default=None,
        help='Pretraining seed recorded by the checkpoint; independent of the fine-tuning seed.',
    )
    parser.add_argument('--checkpoint_pretraining_dataset', default='')
    parser.add_argument('--checkpoint_tier', default='')
    parser.add_argument(
        '--config_schema', default='manual'
    )
    # The shared MTS launcher passes the source schema to both pretraining
    # and downstream entrypoints.  Downstream validation uses this metadata
    # to distinguish an experiment input from the resolved production
    # contract; it does not alter the fine-tuning objective.
    parser.add_argument('--config_source_schema', default='')
    parser.add_argument(
        '--split_manifest_dir', default='data/splits/mips_shared5'
    )
    parser.add_argument(
        '--cache_only', action='store_true',
        help='Build/validate downstream feature cache and exit before training.',
    )
    parser.add_argument(
        '--smiles_model_name',
        default="./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
    )
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
        default=['graph'],
        help="MTS modalities: graph, optionally smiles and/or fp."
    )
    parser.add_argument(
        '--fusion_type',
        type=str,
        choices=SUPPORTED_FUSION_TYPES,
        default='none',
        help="MTS fusion: none for graph-only or zero_gated_residual for optional views."
    )

    parser.add_argument(
        '--fp_mode',
        type=str,
        choices=['disabled', 'ecfp', 'mixfp', 'attachment_count'],
        default='ecfp',
        help="Fingerprint implementation. ecfp keeps the original Morgan/ECFP 1024-bit FP; mixfp uses MACCSKeys + PubChemFingerprints.",
    )
    parser.add_argument('--fusion_dropout', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--head_dropout', type=float, default=0.25)
    parser.add_argument('--fp_bit_dropout', type=float, default=0.15)
    parser.add_argument('--fp_modality_dropout', type=float, default=0.25)
    parser.add_argument('--smiles_modality_dropout', type=float, default=0.10)
    parser.add_argument('--graph_modality_dropout', type=float, default=0.05)
    parser.add_argument(
        '--graph_input',
        type=str,
        choices=['repeat_unit', 'star_linking'],
        default='star_linking',
        help="Graph input type. 'repeat_unit' keeps the original graph; 'star_linking' removes two attachment atoms and connects their boundary atoms for graph-only topology input."
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
        '--eval_batch_size', type=int, default=64,
        help='Validation/test batch size; training batch size is unchanged.',
    )
    parser.add_argument(
        '--amp_dtype', choices=['fp32', 'bf16'], default='fp32',
        help='Fine-tuning precision. BF16 is enabled only after its parity gate.',
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument('--smiles_lr', type=float, default=5e-6)
    parser.add_argument('--graph_lr', type=float, default=1e-5)
    parser.add_argument('--fp_lr', type=float, default=1e-4)
    parser.add_argument('--fusion_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument(
        '--regression_loss', choices=['huber'], default='huber',
        help='MTS production regression objective: SmoothL1/Huber(beta=0.5).'
    )
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
    # Validation-selected SWA is fixed off for the current MTS downstream
    # runtime.
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
        help='Target preprocessing: recommended logs eps/nc and standardizes other current tasks.',
    )
    parser.add_argument(
        '--evaluation_protocol',
        choices=['historical_shared5', 'nested5'],
        default='historical_shared5',
    )
    parser.add_argument(
        '--finetune_mode', choices=['single_task', 'multitask_pcgrad'],
        default='single_task',
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
    parser.add_argument(
        '--mips_descriptor_disturbance', type=float, default=0.0,
    )
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
        '--mips_downstream_head',
        choices=['unipoly'],
        default='unipoly',
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
        choices=['o8'],
        default='o8',
        help='B0-v2 uses the original O8 attention stack.',
    )
    parser.add_argument('--star_rbf_upper', type=float, default=3.0)
    parser.add_argument('--star_rbf_v2_sidecar', default=None)
    parser.add_argument('--periodic_line_glt_sidecar', default=None)
    parser.add_argument(
        '--mts_glt_mode',
        choices=['none', 'o8_only', 'o8_glt', 'o8_glt_atom', 'o8_glt_atom_desc', 'o8_glt_graph'],
        default='none',
    )
    parser.add_argument('--mts_glt_version', choices=['v1', 'v2', 'graphgate_v1'], default='v1')
    parser.add_argument('--mts_glt_geometry_mode', choices=['full', 'off'], default='full')
    parser.add_argument('--mts_glt_layers', type=int, choices=[6, 12], default=6)
    parser.add_argument(
        '--mts_glt_attention_variant', choices=['mips', 'paper'], default='mips'
    )
    parser.add_argument('--mts_glt_use_compact19', action='store_true')
    parser.add_argument(
        '--mts_glt_postmortem_dir', default='',
        help=(
            'Optional isolated directory for MTS-GLT fusion diagnostics. '
            'Empty keeps the production fine-tune path unchanged.'
        ),
    )
    parser.add_argument(
        '--mts_glt_fusion_strategy',
        choices=['legacy_zero', 'fusion_warm'],
        default='legacy_zero',
        help='Downstream GLT fusion schedule. legacy_zero preserves MTS-GLT-v1.',
    )
    parser.add_argument(
        '--mts_glt_fusion_warm_epochs', type=int, default=5,
        help=(
            'Constant-LR fusion-only epochs before joint GLT/O8 fine-tuning; '
            'zero starts joint fine-tuning at the first optimizer step.'
        ),
    )
    parser.add_argument(
        '--mts_glt_initial_alpha', type=float, default=0.1,
        help='Initial tanh(gate) value for fusion_warm after checkpoint loading.',
    )
    parser.add_argument(
        '--mts_glt_fusion_warm_dir', default='',
        help='Isolated per-fold FusionWarm audit output directory.',
    )
    parser.add_argument(
        '--mts_glt_fusion_stage2_trainability',
        choices=['both_frozen', 'o8_only', 'glt_query_only', 'joint'],
        default='joint',
        help=(
            'Encoder trainability during FusionWarm Stage 2. The default '
            'joint preserves the established FusionWarm behavior.'
        ),
    )
    parser.add_argument(
        '--use_star_rbf', action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--use_mcl', action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        '--use_md200', action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        '--topology_representation',
        choices=['canonical_lifted'],
        default='canonical_lifted',
    )
    parser.add_argument('--trimer_num_candidates', type=int, default=4)
    parser.add_argument('--trimer_max_heavy_atoms', type=int, default=384)
    parser.set_defaults(
        finite_variant='none', conformer_mode='none',
        field_layout='none', field_channels='none',
    )
    parser.add_argument(
        '--mips_fusion_mode',
        choices=['none'],
        default='none',
    )
    parser.add_argument(
        '--projection_mode', choices=['plain', 'shared_private'],
        default='plain',
    )
    parser.add_argument(
        '--modality_control',
        choices=['real', 'batch_shuffled', 'constant_zero'], default='real',
    )
    parser.add_argument('--controlled_modality', choices=['smiles', 'fp'], default=None)
    parser.add_argument(
        '--mips_variant',
        choices=['O8'],
        default='O8',
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
        help='Conformer search budget. Must match the profile used to build the feature cache.'
    )
    args = parser.parse_args(argv)
    if int(args.eval_batch_size) < int(args.batch_size):
        raise ValueError("--eval_batch_size must be >= --batch_size")
    if args.mts_glt_fusion_strategy == 'fusion_warm':
        if args.mts_glt_mode not in {'o8_glt', 'o8_glt_graph'}:
            parser.error(
                '--mts_glt_fusion_strategy=fusion_warm requires '
                '--mts_glt_mode=o8_glt or o8_glt_graph'
            )
        if (
            args.mts_glt_mode == 'o8_glt_graph'
            and args.mts_glt_version != 'graphgate_v1'
        ):
            parser.error(
                '--mts_glt_mode=o8_glt_graph requires '
                '--mts_glt_version=graphgate_v1'
            )
        if int(args.mts_glt_fusion_warm_epochs) < 0:
            parser.error('--mts_glt_fusion_warm_epochs must be non-negative')
        if not 0.0 < float(args.mts_glt_initial_alpha) < 1.0:
            parser.error('--mts_glt_initial_alpha must be strictly between 0 and 1')
        # FusionWarm owns its two-stage LR schedule: constant Stage 1 and a
        # fresh no-warmup cosine schedule after the Stage 2 optimizer rebuild.
        args.warmup_epochs = 0
    # These are fixed MTS topology values, not user-selectable route
    # parameters. Retired geometry and staged-finetuning controls are
    # intentionally absent from the runtime namespace.
    args.geom_input = 'repeat_unit'
    # MTS downstream constructs the encoder explicitly and applies its fixed
    # trainability policy in the shared training utilities.
    args.freeze_encoder = False
    # These names are internal call-compatibility slots only.  They are not
    # user-selectable geometry learning-rate controls: the complete MTS
    # graph wrapper always uses graph_lr=1e-5.
    args.geom_lr = 1e-5
    args.mts_o8_lr = 1e-5
    args.mts_geometry_lr = 1e-5
    args.mts_adapter_lr = 1e-5
    args.freeze_smiles_epochs = 0
    args.deep_unfreeze_epoch = 0
    args.fp_unfreeze_epoch = -1
    args.swa_start_epoch = -1
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
    if args.finetune_mode == 'multitask_pcgrad' and args.refit_full_train:
        parser.error(
            '--refit_full_train is not supported with multitask_pcgrad; '
            'nested validation already supplies leakage-safe model selection'
        )
    if args.graph_encoder_type == "mts":
        args.graph_encoder_type = MTS_ROUTE_INTERNAL
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL and args.mips_max_hops is None:
        args.mips_max_hops = 2
    return args
