"""Argument parsing for the MTS-GLT-v2-Base-5k downstream route."""

from __future__ import annotations

from dataclasses import dataclass
import argparse

from src.dataset.mips_trimer_contract import ROUTE_INTERNAL as MTS_ROUTE_INTERNAL
from src.training.finetune.mode_specs import MODE_SPECS


SUPPORTED_MODALITIES = ("graph",)
SUPPORTED_FUSION_TYPES = ("none",)


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


def parse_modality(value: str) -> str:
    if value != "graph":
        raise argparse.ArgumentTypeError("MTS-GLT-v2-Base-5k supports graph modality only")
    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run one graph-only MTS-GLT-v2 downstream fold")
    parser.add_argument("--experiment_id", default="manual")
    parser.add_argument("--predictions_dir", default="")
    parser.add_argument("--resolved_config_path", default="")
    parser.add_argument("--resolved_command_path", default="")
    parser.add_argument("--checkpoint_seed", type=int, default=None)
    parser.add_argument("--checkpoint_pretraining_dataset", default="")
    parser.add_argument("--checkpoint_tier", default="")
    parser.add_argument("--config_schema", default="manual")
    parser.add_argument("--config_source_schema", default="")
    parser.add_argument("--split_manifest_dir", default="data/splits/mips_shared5")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--tasks", nargs="+", default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    parser.add_argument("--model_name", default="UniEncoderAttention")
    parser.add_argument("--modalities", nargs="+", type=parse_modality, default=["graph"])
    parser.add_argument("--fusion_type", choices=SUPPORTED_FUSION_TYPES, default="none")
    parser.add_argument("--graph_input", choices=["repeat_unit", "star_linking"], default="repeat_unit")
    parser.add_argument("--pretrained_model_path", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--results_dir", default="./results/results.csv")
    parser.add_argument("--models_dir", default="./saved_models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--loader_workers", type=int, default=2)
    parser.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--graph_lr", type=float, default=1e-5)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--mts_o8_lr", type=float, default=1e-5)
    parser.add_argument("--mts_geometry_lr", type=float, default=1e-5)
    parser.add_argument("--mts_adapter_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--regression_loss", choices=["huber"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--target_transform", choices=["recommended", "standard", "log"], default="recommended")
    parser.add_argument("--evaluation_protocol", choices=["historical_shared5", "outer5_inner20"], default="historical_shared5")
    parser.add_argument("--fold_ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)

    # Fixed MIPS-Trimer-SCAGE input contract.
    parser.add_argument("--graph_encoder_type", choices=["mips_trimer_scage", "mts"], default="mips_trimer_scage")
    parser.add_argument("--topology_attention_variant", choices=["o8"], default="o8")
    parser.add_argument("--use_star_rbf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_mcl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_md200", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mips_core", choices=["paper_corrected"], default="paper_corrected")
    parser.add_argument("--mips_max_hops", type=int, default=2)
    parser.add_argument("--mips_atom_feature_mode", choices=["mips137"], default="mips137")
    parser.add_argument("--mips_attention_scale", choices=["head_dim"], default="head_dim")
    parser.add_argument("--mips_norm_mode", choices=["post", "pre"], default="post")
    parser.add_argument("--mips_activation", choices=["relu"], default="relu")
    parser.add_argument("--mips_spd_bias_mode", choices=["per_head"], default="per_head")
    parser.add_argument("--mips_path_bias_mode", choices=["per_head_single_path_node"], default="per_head_single_path_node")
    parser.add_argument("--mips_multi_scale_hop_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mips_semantics", choices=["paper_semantic"], default="paper_semantic")
    parser.add_argument("--mips_descriptor_fusion_mode", choices=["graph_md_residual"], default="graph_md_residual")
    parser.add_argument("--mips_descriptor_components", choices=["md200"], default="md200")
    parser.add_argument("--mips_descriptor_protocol", choices=["source_star_sub"], default="source_star_sub")
    parser.add_argument("--mips_backbone_mode", choices=["independent"], default="independent")
    parser.add_argument("--mips_mask_mode", choices=["zero"], default="zero")
    parser.add_argument("--mips_mask_policy", choices=["canonical_exact"], default="canonical_exact")
    parser.add_argument("--mips_masked_loss_reduction", choices=["atom_mean"], default="atom_mean")
    parser.add_argument("--mips_use_descriptors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mips_variant", choices=["O8"], default="O8")
    parser.add_argument("--mips_fusion_mode", choices=["none"], default="none")
    parser.add_argument("--projection_mode", choices=["plain"], default="plain")
    parser.add_argument("--spatial_mode", choices=["trimer_scage"], default="trimer_scage")
    parser.add_argument("--graph_geometry_mode", choices=["trimer_scage_mcl"], default="trimer_scage_mcl")
    parser.add_argument("--topology_representation", choices=["canonical_lifted"], default="canonical_lifted")
    parser.add_argument("--trimer_num_candidates", type=int, choices=[4], default=4)
    parser.add_argument("--trimer_max_heavy_atoms", type=int, choices=[384], default=384)
    parser.add_argument("--star_rbf_upper", type=float, default=3.0)
    parser.add_argument("--periodic_line_glt_sidecar", default=None)
    parser.add_argument("--mts_glt_mode", choices=sorted(MODE_SPECS), default="o8_glt_atom")
    parser.add_argument("--mts_glt_version", choices=["v2", "distill", "distill_repair"], default="v2")
    parser.add_argument("--distill_repair_version", choices=["none", "n_plus_1", "n_plus_2"])
    parser.add_argument("--allow_smoke_checkpoint", action="store_true")
    parser.add_argument("--glt_readout_mode", choices=["galformer", "mips_concat"], default="galformer")
    parser.add_argument("--mts_glt_geometry_mode", choices=["full"], default="full")
    parser.add_argument("--mts_glt_layers", type=int, choices=[6], default=6)
    parser.add_argument("--mts_glt_attention_variant", choices=["mips"], default="mips")
    parser.add_argument("--head_dropout", type=float, default=0.25)
    parser.add_argument(
        "--finetune_strategy",
        choices=["standard", "staged_head10"],
        default="standard",
    )
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=90)

    # Cache construction parameters are retained because the baseline Dataset
    # owns the immutable LMDB readers.
    parser.add_argument("--feature_source_dataset", default="smi_all")
    parser.add_argument("--disable_feature_cache", action="store_true")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--feature_cache_workers", type=int, default=0)
    parser.add_argument("--feature_cache_chunksize", type=int, default=4)
    parser.add_argument("--feature_cache_partial_every", type=int, default=200)
    parser.add_argument("--feature_cache_item_timeout", type=int, default=45)
    parser.add_argument("--cache_layers", default="ru_base,topology,trimer,md200")
    parser.add_argument("--cache_validate", choices=["sample", "full"], default="sample")
    parser.add_argument("--cache_commit_size", type=int, default=128)
    parser.add_argument("--embed_tries_multiplier", type=int, default=8)
    parser.add_argument("--conformer_3d_count", type=int, default=8)
    parser.add_argument("--conformer_keep_count", type=int, default=4)
    parser.add_argument("--conformer_profile", choices=["fast", "full", "quality"], default="full")

    args = parser.parse_args(argv)
    if args.graph_encoder_type == "mts":
        args.graph_encoder_type = MTS_ROUTE_INTERNAL
    if args.eval_batch_size < args.batch_size:
        parser.error("--eval_batch_size must be >= --batch_size")
    if args.modalities != ["graph"] or args.fusion_type != "none":
        parser.error("MTS-GLT-v2-Base-5k is graph-only with fusion_type=none")
    if args.mts_glt_mode not in MODE_SPECS:
        parser.error("unsupported MTS-GLT-v2 downstream mode")
    if args.mts_glt_version in {"distill", "distill_repair"}:
        # Revised DistillStudent uses the fixed Original-MIPS-style predictor
        # contract; do not let the legacy 0.25 CLI default reach its model or
        # resolved metadata.  Other routes retain their configured dropout.
        args.head_dropout = 0.1
    if args.mts_glt_version in {"distill", "distill_repair"}:
        args.mts_glt_mode = "none"
        args.graph_input = "star_linking"
    if args.mts_glt_version == "distill_repair" and args.distill_repair_version is None:
        parser.error("--distill_repair_version is required for distill_repair")
    args.mts_glt_use_compact19 = False
    args.geom_input = "repeat_unit"
    args.freeze_encoder = False
    args.geom_lr = args.graph_lr
    args.smiles_model_name = ""
    args.fp_mode = "disabled"
    args.fusion_dropout = 0.0
    args.fp_bit_dropout = 0.0
    args.smiles_modality_dropout = 0.0
    args.graph_modality_dropout = 0.0
    args.scage_use_pbc_distance = False
    args.scage_use_descriptors = False
    args.scage_distance_mode = "mips_dual"
    args.scage_distance_rbf = 32
    args.scage_distance_cutoff = 12.0
    args.scage_distance_scales = [4.0, 8.0, 12.0]
    args.scage_distance_taus = [0.5, 1.0, 1.5]
    args.scage_dist_bar = [20.0, 50.0]
    args.scage_num_heads = 8
    args.scage_ffn_hidden_dim = 2048
    args.scage_num_kernels = 128
    args.scage_attention_dropout = 0.1
    args.scage_topology_bias = True
    args.scage_topology_max_distance = 20
    args.scage_topology_locality_mode = "soft"
    args.scage_topology_locality_threshold = 5
    args.scage_topology_locality_tau = 1.0
    args.scage_periodic_image_mode = "none"
    args.scage_periodic_image_cap = 0
    args.scage_periodic_image_temperature = 0.5
    args.scage_force_topology_only = True
    args.mips_descriptor_disturbance = 0.0
    args.mips_input_norm = False
    args.controlled_modality = None
    args.modality_control = "real"
    args.mts_glt_mode = str(args.mts_glt_mode)
    return args
