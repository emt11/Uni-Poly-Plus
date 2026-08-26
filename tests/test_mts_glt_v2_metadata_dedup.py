import torch
from pathlib import Path

from src.modules.periodic_line_glt_v2 import LocalPeriodicGraphLineTransformerV2
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.periodic_line_glt_v2 import GLTMaskedLineHeadV2, NUM_LINE_LABELS
from src.training.pretrain.glt_v2_engine import (
    MTSGLTV2PretrainContainer,
    _projection,
)
from src.training.pretrain.metadata_dedup import (
    build_matched_pretrain_containers,
    copy_shared_state_by_name_and_shape,
    parameter_accounting,
)


def test_metadata_modes_only_remove_two_additive_embeddings():
    torch.manual_seed(19)
    full = LocalPeriodicGraphLineTransformerV2(
        hidden_size=64, layers=1, heads=8, metadata_mode="full"
    )
    torch.manual_seed(19)
    dedup = LocalPeriodicGraphLineTransformerV2(
        hidden_size=64, layers=1, heads=8, metadata_mode="dedup"
    )
    report = copy_shared_state_by_name_and_shape(full, dedup)
    accounting = parameter_accounting(full, dedup)
    assert report["mismatch"] == []
    assert len(report["shared_tensors"]) == len(report["equal_tensors"])
    assert report["source_only"] == [
        "relation_multiplicity_embedding.weight",
        "token_count_embedding.weight",
    ]
    assert report["target_only"] == []
    assert full.token_count_embedding is not None
    assert full.relation_multiplicity_embedding is not None
    assert dedup.token_count_embedding is None
    assert dedup.relation_multiplicity_embedding is None
    assert accounting["parameter_delta_full_minus_dedup"] == (4 * 64 + 4 * 8)


def test_full_default_is_explicitly_full():
    model = LocalPeriodicGraphLineTransformerV2(hidden_size=64, layers=1, heads=8)
    assert model.metadata_mode == "full"


def test_complete_pretrain_container_shared_init_and_engine_hook():
    def factory(metadata_mode):
        model = MTSGraphLineModelV2(
            glt_layers=6, glt_attention_variant="mips",
            glt_metadata_mode=metadata_mode, use_compact19=False,
        )
        for module in (
            model.atom_fusion_norm, model.atom_fusion_projection,
            model.compact19_residual, model.o8.md_residual,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        model.atom_channel_gate.requires_grad = False
        return MTSGLTV2PretrainContainer(
            model,
            torch.nn.Linear(512, int(model.o8.masked_atom_classes)),
            GLTMaskedLineHeadV2(512), _projection(256), _projection(256),
            torch.ones(NUM_LINE_LABELS, dtype=torch.long),
        )

    actual, reference, report = build_matched_pretrain_containers(
        factory, "dedup", seed=1729
    )
    assert len(report["shared_tensors"]) == len(report["equal_tensors"])
    assert report["mismatch"] == []
    assert report["source_only"] == [
        "model.glt.relation_multiplicity_embedding.weight",
        "model.glt.token_count_embedding.weight",
    ]
    assert report["target_only"] == []
    assert report["caller_cpu_rng_unchanged"]
    accounting = parameter_accounting(reference, actual)
    assert accounting["parameter_delta_full_minus_dedup"] == 2080
    engine_source = Path("src/training/pretrain/glt_v2_engine.py").read_text()
    assert "build_matched_pretrain_containers(" in engine_source
