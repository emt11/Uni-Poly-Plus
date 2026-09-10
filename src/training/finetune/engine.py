import os
import sys
import json
import hashlib
import warnings
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    CACHE_BUNDLE_SCHEMA as MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as MIPS_TRIMER_TOPOLOGY_SCHEMA,
    CHECKPOINT_SCHEMA as MIPS_TRIMER_CHECKPOINT_SCHEMA,
    ROUTE_NAME as MTS_ROUTE_NAME,
    ROUTE_SHORT_NAME as MTS_ROUTE_SHORT_NAME,
    validate_runtime_args as validate_mips_trimer_runtime,
)
from src.training.finetune.config import parse_arguments


class SingleFoldContext:
    """Validated task/seed/fold coordinates for one finite job."""

    __slots__ = ("task", "fold", "seed", "experiment_id")

    def __init__(self, task: str, fold: int, seed: int, experiment_id: str):
        self.task = str(task)
        self.fold = int(fold)
        self.seed = int(seed)
        self.experiment_id = str(experiment_id)

    def validate(self) -> None:
        if not self.task:
            raise ValueError("single-fold task is required")
        if self.fold not in range(5):
            raise ValueError("single-fold fold must be in [0, 4]")


def run_single_fold(train_and_evaluate, *args, context: SingleFoldContext, **kwargs):
    """Compatibility boundary for callers that already own the fold loop."""
    context.validate()
    return train_and_evaluate(*args, **kwargs)


def _cohort_hash_from_current_manifest(root, dataset_name):
    """Read an existing immutable cohort pointer without rebuilding it."""

    pointer = os.path.join(
        root, "processed", "mips_trimer_scage", "cohorts",
        str(dataset_name), "current.json",
    )
    try:
        with open(pointer, encoding="utf-8") as handle:
            value = json.load(handle)
        cohort_hash = str(value.get("cohort_hash", ""))
        return cohort_hash or None
    except (OSError, ValueError, TypeError):
        return None


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


def _student_architecture_metadata(glt_version):
    """Describe the revised DistillStudent O8 and predictor contract."""
    if str(glt_version) not in {"distill", "distill_repair"}:
        return {}
    return {
        "o8_ffn_activation": "GELU(approximate='none')",
        "o8_ffn_hidden": "512->2048->512",
        "graph_adapter": "identity",
        "predictor": "512->512->1",
        "predictor_dropout": 0.1,
    }


def _require_student_architecture_metadata(checkpoint):
    """Reject same-shaped student bundles produced before the GELU contract."""
    expected = {
        "o8_ffn_activation": "GELU(approximate='none')",
        "o8_ffn_hidden": "512->2048->512",
        "downstream_graph_adapter": "identity",
        "downstream_predictor": "512->512->1",
        "downstream_predictor_dropout": 0.1,
    }
    mismatched = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "student deployment architecture metadata mismatch; "
            "a pre-GELU/ReLU bundle cannot be loaded"
        )


def build_mts_downstream_model(args, auxiliary_tasks=()):
    """Construct the downstream MTS UniEncoder with the exact fixed switches.

    The publish step and every fold job share this one construction so the
    published final checkpoint always matches the training-time architecture.
    """
    from src.modules import UniEncoderAttention

    glt_version = str(getattr(args, "mts_glt_version", "v2"))
    is_new_student = glt_version in {"distill", "distill_repair"}
    joint_dim = 512 if is_new_student else 256
    actual_head_dropout = 0.1 if is_new_student else float(args.head_dropout)
    model = UniEncoderAttention(
        joint_embedding_dim=joint_dim,
        smiles_model_name="",
        gnn_model_name="",
        modality_list=("graph",),
        freeze_encoder=False,
        graph_num_layers=6,
        graph_emb_dim=512,
        graph_dropout=0.1,
        graph_encoder_type="mips_trimer_scage",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_variant="O8",
        mips_atom_feature_mode="mips137",
        mips_attention_scale="head_dim",
        mips_norm_mode="post",
        mips_activation="relu",
        mips_spd_bias_mode="per_head",
        mips_path_bias_mode="per_head_single_path_node",
        mips_multi_scale_hop_gate=False,
        mips_semantics="paper_semantic",
        mips_descriptor_fusion_mode="graph_md_residual",
        mips_descriptor_components="md200",
        mips_backbone_mode="independent",
        mips_mask_mode="zero",
        mips_mask_policy="canonical_exact",
        mips_masked_loss_reduction="atom_mean",
        topology_attention_variant="o8",
        use_star_rbf=False,
        use_mcl=False,
        fusion_type="none",
        head_dropout=actual_head_dropout,
    )
    if is_new_student:
        graph_wrapper = model.encoders["graph"]
        graph_wrapper.norm = nn.Identity()
        graph_wrapper.projection = nn.Identity()
        model.mlp = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(0.1),
            nn.Linear(512, 1, bias=True),
        )
    if glt_version in {"distill", "distill_repair"}:
        from src.modules.mts_glt_distill import DistillStudent
        model.encoders["graph"].encoder = DistillStudent()
        return model
    glt_mode = str(getattr(args, "mts_glt_mode", "none"))
    if glt_mode != "none":
        raise ValueError(
            "the MTS-GLT-v2 GLT downstream modes were retired with the v2/v3 "
            "routes; only mts_glt_mode=none remains"
        )
    return model


def select_mts_glt_graph_state(model_state, checkpoint_state):
    """Map a GLT pretrain container into the exact downstream graph module."""
    graph_prefix = 'encoders.graph.encoder.'
    mapped = {
        graph_prefix + str(key)[len('model.'):]: value
        for key, value in checkpoint_state.items()
        if str(key).startswith('model.')
    }
    expected = {key for key in model_state if key.startswith(graph_prefix)}
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        raise RuntimeError(
            'MTS-GLT graph checkpoint mismatch; missing='
            + ','.join(missing[:10]) + ' unexpected='
            + ','.join(unexpected[:10])
        )
    return mapped


def select_mts_glt_distill_state(
    model_state,
    checkpoint,
    expected_version=None,
    allow_smoke=False,
    expected_step=20000,
):
    expected_step = int(expected_step)
    if expected_step not in {5000, 20000}:
        raise ValueError("N+ student deployment supports 5k or 20k steps")
    if checkpoint.get("schema") not in {
        "mts-glt-distill-student-deploy-v1",
        "mts-glt-distill-repair-student-deploy-v1",
    } or (int(checkpoint.get("step", -1)) != expected_step and not allow_smoke):
        raise RuntimeError(
            f"N+ downstream requires a strict {expected_step // 1000}k student deploy bundle"
        )
    _require_student_architecture_metadata(checkpoint)
    if checkpoint.get("schema") == "mts-glt-distill-repair-student-deploy-v1":
        if checkpoint.get("version") != expected_version:
            raise RuntimeError("repair student deployment version mismatch")
        expected_revision = None if expected_version == "none" else 2
        if checkpoint.get("geometry_revision") != expected_revision:
            raise RuntimeError("repair student deployment geometry revision mismatch")
    graph_prefix = "encoders.graph.encoder."
    mapped = {graph_prefix + str(key): value for key, value in checkpoint["state_dict"].items()}
    expected = {key for key in model_state if key.startswith(graph_prefix)}
    if set(mapped) != expected:
        raise RuntimeError("N+ student deployment state does not strictly match downstream model")
    return mapped


def run_finetune_job(config=None, task=None, seed=None, fold=None):
    """Run exactly one task/seed/fold through the real MTS training path."""
    args = parse_arguments() if config is None else config
    if task is not None:
        args.tasks = [str(task)]
    if seed is not None:
        args.seed = int(seed)
    if fold is not None:
        args.fold_ids = [int(fold)]
    validate_mips_trimer_runtime(args)
    from src.dataset import UniDataset
    from src.modules import UniEncoderAttention
    from src.utils import (
        TargetScaler, get_data_loader, scale_targets,
        set_global_seed, test_model, train_and_evaluate,
        staged_train_and_evaluate,
    )
    # Frozen cache readers perform only their local schema/shape/index checks.
    # The old full-bundle hash audit was a separate offline concern and is no
    # longer part of a Finetune startup.
    # Ignore warnings
    warnings.filterwarnings("ignore")

    result_output_dir = args.results_dir

    task_list = list(args.tasks)
    dataset_task_list = list(task_list)
    dataset_name_list = ['smi_' + task for task in dataset_task_list]
    dataset_list = [
        UniDataset(
            root=args.root,
            dataset=dataset_name,
            smiles_model_name="",
            graph_encoder_type="mips_trimer_scage",
            graph_input=(
                "star_linking"
                if str(getattr(args, "mts_glt_version", "v2")) in {"v3", "distill_repair"}
                else "repeat_unit"
            ),
            use_feature_cache=not args.disable_feature_cache,
            feature_source_dataset=args.feature_source_dataset,
            rebuild_feature_cache=args.rebuild_feature_cache,
            max_smiles_length=None,
            max_smiles_length_cap=256,
            fp_mode="disabled",
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
            scage_distance_mode="mips_dual",
            scage_distance_rbf=32,
            scage_distance_cutoff=12.0,
            mips_core="paper_corrected",
            mips_max_hops=2,
            mips_use_descriptors=True,
            mips_descriptor_protocol="source_star_sub",
            spatial_mode="trimer_scage",
            graph_geometry_mode="trimer_scage_mcl",
            topology_representation="canonical_lifted",
            trimer_num_candidates=4,
            trimer_max_heavy_atoms=384,
            mips_variant="O8",
            finite_variant="none",
            conformer_mode="none",
            field_layout="none",
            field_channels="none",
            experiment_id=args.experiment_id,
            feature_config_hash="manual",
            modalities=("graph",),
            periodic_line_glt_sidecar=(
                None
                if str(getattr(args, "mts_glt_version", "v2")) == "distill_repair"
                else args.periodic_line_glt_sidecar
            ),
        )
        for dataset_name in dataset_name_list
    ]
    dataset_by_task = dict(zip(dataset_task_list, dataset_list))
    raw_targets_by_task = {
        task: np.asarray(dataset.raw_targets, dtype=np.float64)
        for task, dataset in dataset_by_task.items()
    }
    if args.graph_encoder_type == "mips_trimer_scage" and not args.cache_only:
        cache_dataset = dataset_list[0]
        original_layers = cache_dataset.cache_layers
        cache_dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
        specs = cache_dataset._lmdb_cache_specs({})
        cache_dataset.cache_layers = original_layers
        unfrozen = [
            name for name, spec in specs.items()
            if not os.path.isfile(os.path.join(spec["root"], ".frozen"))
        ]
        if unfrozen:
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} training requires frozen cache artifacts: "
                + ", ".join(unfrozen)
            )
    if args.cache_only:
        print(
            "Downstream feature-cache prebuild complete: "
            + ", ".join(
                f"{task}={len(dataset_by_task[task])}"
                for task in dataset_task_list
            )
        )
        return
    # Feature-cache workers must be created before this process initializes
    # CUDA. Cache workers are CPU-only and PolyGen's CPU optimizer must not
    # inherit the downstream training CUDA context.
    set_global_seed(args.seed)
    if args.graph_encoder_type == "mips_trimer_scage" and not torch.cuda.is_available():
        raise RuntimeError(f"{MTS_ROUTE_NAME} Stage 3 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience
    t1_init_artifact = False
    checkpoint_meta = {}
    pretraining_bundle_identity = {}

    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_by_task[task]
        raw_targets = raw_targets_by_task[task]
        # ``--results_dir`` historically had two meanings.  Treat an
        # existing directory (or a path without a CSV suffix) as a result
        # root and give every task its own file; this prevents concurrent
        # Stage-3 workers from writing the same CSV.
        result_root = Path(result_output_dir)
        if result_root.exists() and result_root.is_dir():
            task_result_output = result_root / f"{task}.csv"
        elif len(task_list) > 1 and result_root.suffix.lower() != ".csv":
            task_result_output = result_root / f"{task}.csv"
        else:
            task_result_output = result_root
        task_result_output = str(task_result_output)
        task_result_file_initialized = False

        print("Start 5-fold Cross Validation")
        manifest_path = Path(args.split_manifest_dir) / f"{task}.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Missing fixed split manifest: {manifest_path}. Run "
                "scripts/create_mips_split_manifests.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        task_csv = Path(args.root) / "raw" / f"smi_{task}.csv"
        if not task_csv.is_file():
            raise RuntimeError(f"Task CSV required for fixed split validation is missing: {task_csv}")
        ordered_smiles = (
            pd.read_csv(task_csv, usecols=[0])
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .tolist()
        )
        protocol = str(args.evaluation_protocol)
        expected_schema = (
            "mips-outer5-inner20-fold-v1" if protocol == "outer5_inner20"
            else "mips-shared-validation-test-fold-v1"
        )
        expected_validation_is_test = protocol == "historical_shared5"
        order_hash = hashlib.sha256("\n".join(ordered_smiles).encode("utf-8")).hexdigest()
        if (
            manifest.get("schema") != expected_schema
            or manifest.get("protocol") not in {protocol, "shared_validation_test_fold"}
            or int(manifest.get("sample_count", -1)) != len(dataset)
            or bool(manifest.get("validation_is_test", False)) != expected_validation_is_test
            or manifest.get("sample_order_sha256") != order_hash
        ):
            raise RuntimeError(f"Fixed split manifest does not match task cohort: {manifest_path}")
        split_identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        splits = [
            (
                np.asarray(item["train_indices"], dtype=np.int64),
                np.asarray(item["validation_indices"], dtype=np.int64),
                np.asarray(item["test_indices"], dtype=np.int64),
            )
            for item in manifest["folds"]
        ]
        fold_metrics = []
        fold_attention_weights = []
        fold_named_attention_weights = {}
        best_fold_val_r2 = -float('inf')
        best_model_state = None

        selected_folds = set(args.fold_ids)
        if not selected_folds or any(fold < 0 or fold >= 5 for fold in selected_folds):
            raise ValueError("--fold_ids must contain one or more values from 0 to 4")
        for fold, (train_indices, manifest_val_indices, test_indices) in enumerate(splits):
            if fold not in selected_folds:
                continue
            fold_started = time.monotonic()
            print(f"\nFold {fold + 1}")
            task_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(task))
            fold_seed = int(args.seed) + 1009 * task_offset + fold
            set_global_seed(fold_seed)
            print(f"Fold seed: {fold_seed}")
            fold_train_indices = train_indices
            val_indices = manifest_val_indices
            print(f"Evaluation protocol: {protocol}")
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
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            val_loader = get_data_loader(
                dataset,
                indices=val_indices,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )

            model = build_mts_downstream_model(args)
            if pretrained_model_path:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                if not isinstance(checkpoint, dict):
                    raise RuntimeError("MTS-GLT-v2 checkpoint must be a mapping")
                checkpoint_state = checkpoint.get("state_dict", checkpoint)
                if not isinstance(checkpoint_state, dict):
                    raise RuntimeError("MTS-GLT-v2 checkpoint does not contain a state_dict mapping")
                checkpoint_meta = (
                    checkpoint.get("meta", {})
                    if isinstance(checkpoint.get("meta"), dict) else {}
                )
                pretraining_bundle_identity = {
                    "pretraining_bundle_schema": checkpoint.get("schema"),
                    "pretraining_bundle_version": checkpoint.get("version"),
                    "pretraining_bundle_step": int(checkpoint.get("step", -1)),
                }
                merged_state = model.state_dict()
                glt_state = (
                    select_mts_glt_distill_state(
                        merged_state, checkpoint,
                        getattr(args, "distill_repair_version", None),
                        bool(getattr(args, "allow_smoke_checkpoint", False)),
                        expected_step=(
                            5000
                            if str(getattr(args, "checkpoint_tier", ""))
                            in {"student-5k", "student_005k", "5k"}
                            else 20000
                        ),
                    )
                    if str(getattr(args, "mts_glt_version", "v2")) in {"distill", "distill_repair"}
                    else select_mts_glt_graph_state(merged_state, checkpoint_state)
                )
                merged_state.update(glt_state)
                print(
                    f'MTS-GLT-v2 checkpoint load: strictly loaded '
                    f'{len(glt_state)} graph tensors for {args.mts_glt_mode}.'
                )
                model.load_state_dict(merged_state, strict=True)
                print(f"Loaded pretrained model from {pretrained_model_path}")
            initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            model.to(device)
            print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

            fold_context = SingleFoldContext(
                task=task,
                fold=int(fold),
                seed=int(fold_seed),
                experiment_id=str(args.experiment_id),
            )
            train_function = (
                staged_train_and_evaluate
                if str(getattr(args, "finetune_strategy", "standard")) == "staged_head10"
                else train_and_evaluate
            )
            resume_path = Path(task_result_output).with_suffix(".resume.pt")
            metrics = run_single_fold(
                train_function,
                model, scaler, train_loader, val_loader, test_loader,
                device, num_epochs=epochs, patience=patience, max_grad_norm=args.max_grad_norm,
                graph_lr=args.graph_lr,
                fusion_lr=args.fusion_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                regression_loss=args.regression_loss,
                huber_beta=args.huber_beta,
                evaluate_test=True,
                return_predictions=True,
                mts_o8_lr=args.mts_o8_lr,
                mts_geometry_lr=args.mts_geometry_lr,
                mts_adapter_lr=args.mts_adapter_lr,
                amp_dtype=args.amp_dtype,
                stage1_epochs=getattr(args, "stage1_epochs", 10),
                stage2_epochs=getattr(args, "stage2_epochs", 90),
                resume_path=str(resume_path) if train_function is staged_train_and_evaluate else None,
                context=fold_context,
            )
            if train_function is staged_train_and_evaluate and resume_path.exists():
                resume_path.unlink()
            metrics['refit_full_train'] = False
            metrics['refit_epochs'] = 0
            prediction_true = metrics.pop('_y_true', None)
            prediction_values = metrics.pop('_y_pred', None)
            prediction_path = None
            if args.graph_encoder_type == 'mips_trimer_scage':
                if prediction_true is None or prediction_values is None:
                    raise RuntimeError('MTS fold did not produce raw-space predictions')
                if not args.predictions_dir:
                    raise RuntimeError('MTS Stage 3 requires --predictions_dir')
                prediction_path = (
                    Path(args.predictions_dir) / task / f'fold_{fold}.npz'
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_tmp = prediction_path.with_name(
                    prediction_path.name + f'.tmp.{os.getpid()}'
                )
                metadata = {
                    'task': task,
                    'fold': int(fold),
                    'seed': int(args.seed),
                    'fold_seed': int(fold_seed),
                    'fold_validation_protocol': protocol,
                    'split_manifest_sha256': split_identity,
                    'independent_blind_test': False,
                    'amp_dtype': args.amp_dtype,
                    'train_batch_size': int(args.batch_size),
                    'eval_batch_size': int(args.eval_batch_size),
                    'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    'finetune_strategy': str(getattr(args, 'finetune_strategy', 'standard')),
                }
                try:
                    with prediction_tmp.open('wb') as handle:
                        np.savez(
                            handle,
                            y_true=np.asarray(prediction_true, dtype=np.float64),
                            y_pred=np.asarray(prediction_values, dtype=np.float64),
                            sample_indices=np.asarray(test_indices, dtype=np.int64),
                            metadata=np.asarray(
                                json.dumps(metadata, sort_keys=True)
                            ),
                        )
                    os.replace(prediction_tmp, prediction_path)
                finally:
                    if prediction_tmp.exists():
                        prediction_tmp.unlink()
            metrics["fold_wall_seconds"] = float(
                time.monotonic() - fold_started
            )
            metrics["fold"] = int(fold)
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

        finetuned_checkpoint_path = None
        if str(getattr(args, "mts_glt_version", "v2")) in {"distill", "distill_repair"}:
            finetuned_checkpoint_path = Path(args.results_dir).with_name(
                Path(args.results_dir).stem + "_best.pt"
            )
            finetuned_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "mts-glt-distill-finetune-v1",
                "task": task, "fold": int(fold_metrics[0]["fold"]),
                "seed": int(args.seed), "state_dict": best_model_state,
                "best_validation_r2": float(best_fold_val_r2),
                "evaluation_protocol": protocol,
                "split_manifest_sha256": split_identity,
                "finetune_strategy": str(getattr(args, "finetune_strategy", "standard")),
                "best_stage": fold_metrics[0].get("best_stage"),
                "stage1_epochs": int(getattr(args, "stage1_epochs", 10)),
                "stage2_epochs": int(getattr(args, "stage2_epochs", 90)),
                **_student_architecture_metadata(
                    getattr(args, "mts_glt_version", "v2")
                ),
                **pretraining_bundle_identity,
            }, finetuned_checkpoint_path)

        architecture_metadata = _student_architecture_metadata(
            getattr(args, "mts_glt_version", "v2")
        )
        actual_head_dropout = architecture_metadata.get(
            "predictor_dropout", float(args.head_dropout)
        )

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            # Keep every result shard self-describing.  This is intentionally
            # redundant with the checkpoint metadata: a shard must be safe to
            # resume/merge without consulting a mutable command line or the
            # current cache directory.
            'experiment_id': args.experiment_id,
            'amp_dtype': args.amp_dtype,
            'train_batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'checkpoint_path': str(pretrained_model_path) if pretrained_model_path else None,
            'finetuned_checkpoint_path': (
                str(finetuned_checkpoint_path) if finetuned_checkpoint_path else None
            ),
            'model_modality_list': ['graph'],
            'fusion_type': 'none',
            'mips_core': args.mips_core,
            'mips_max_hops': args.mips_max_hops,
            'mips_use_descriptors': bool(args.mips_use_descriptors),
            'spatial_mode': args.spatial_mode,
            'graph_geometry_mode': args.graph_geometry_mode,
            'topology_attention_variant': args.topology_attention_variant,
            'mts_glt_mode': args.mts_glt_mode,
            'mts_glt_geometry_mode': args.mts_glt_geometry_mode,
            'star_rbf_upper': getattr(args, 'star_rbf_upper', None),
            'feature_cohort': getattr(dataset, 'feature_cohort_name', None),
            'feature_cache_item_timeout': int(
                args.feature_cache_item_timeout
            ),
            'evaluation_protocol': args.evaluation_protocol,
            'finetune_strategy': str(getattr(args, 'finetune_strategy', 'standard')),
            'stage1_epochs': int(getattr(args, 'stage1_epochs', 10)),
            'stage2_epochs': int(getattr(args, 'stage2_epochs', 90)),
            'head_dropout': actual_head_dropout,
            'regression_loss': args.regression_loss,
            'huber_beta': args.huber_beta,
            'seed': args.seed,
            'prediction_path': str(prediction_path) if prediction_path else None,
            'target_transform': args.target_transform,
            'fold_validation_protocol': protocol,
            'split_manifest_sha256': split_identity,
            'independent_blind_test': False,
            'refit_full_train': False,
            'avg_refit_epochs': 0,
            'avg_best_val_r2': float(avg_val_r2),
            'std_best_val_r2': float(std_val_r2),
            'optimizer_lrs': (
                f"o8={args.mts_o8_lr};glt={args.mts_geometry_lr};"
                f"adapter={args.mts_adapter_lr};head={args.head_lr}"
            ),
            'batch_size': args.batch_size,
            'total_fold_wall_seconds': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            )),
            'estimated_gpu_hours': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            ) / 3600.0),
            'fusion_inputs': attention_labels,
            'graph_input': args.graph_input,
            'geom_input': args.geom_input,
            'graph_encoder_type': args.graph_encoder_type,
            'topology_representation': args.topology_representation,
            'baseline': MTS_ROUTE_NAME,
            'route_short_name': MTS_ROUTE_SHORT_NAME,
            'scage_backbone': 'sparse_non_pbc_mips_starlink_spd_single_path_node',
            'scage_input': 'mips137_plus_backbone_column_138_linear',
            'scage_checkpoint_schema': MIPS_TRIMER_CHECKPOINT_SCHEMA,
            'cache_bundle_schema': MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
            'topology_lmdb_schema': MIPS_TRIMER_TOPOLOGY_SCHEMA,
            'pretraining_dataset': checkpoint_meta.get('pretraining_dataset'),
            'avg_test_r2': float(avg_test_r2),
            'std_test_r2': float(std_test_r2),
            'avg_test_mae': float(avg_test_mae),
            'std_test_mae': float(std_test_mae),
            'avg_test_rmse': float(avg_test_rmse),
            'std_test_rmse': float(std_test_rmse),
            'per_fold_metrics': json.dumps(fold_metrics, sort_keys=True),
            'attention': format_attention_weights(attention_labels, cv_attention),
        }
        result.update(architecture_metadata)

        # Save to CSV.  Stage-3 campaign units write exactly one fold per
        # shard.  Use an atomic replacement for those paths so an interrupted
        # process can never leave a partially written CSV that looks complete
        # to the resume logic.  Legacy aggregate outputs retain append mode.
        os.makedirs(os.path.dirname(task_result_output) or ".", exist_ok=True)
        results_df = pd.DataFrame([result])
        write_header = not task_result_file_initialized
        shard_path = str(task_result_output).replace("\\", "/")
        atomic_shard = "/shards/" in shard_path and write_header
        if atomic_shard:
            tmp_path = f"{task_result_output}.tmp.{os.getpid()}"
            try:
                results_df.to_csv(
                    tmp_path,
                    mode='w',
                    header=True,
                    index=False,
                    float_format="%.17g",
                )
                os.replace(tmp_path, task_result_output)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            results_df.to_csv(
                task_result_output,
                mode='w' if write_header else 'a',
                header=write_header,
                index=False,
                float_format="%.17g",
            )
        print(f"Results have been appended to '{task_result_output}'.")


def main():
    return run_finetune_job(parse_arguments())


if __name__ == "__main__":
    main()
