from types import SimpleNamespace

import torch

from src.training.pretrain.config import PretrainRuntimeConfig
from src.training.pretrain.engine import compose_joint_payload_loss
from src.training.pretrain.objective import joint_masked_atom_angle_loss
from src.training.finetune.engine import SingleFoldContext, run_single_fold
from src.training.finetune.scheduler import dispatch_units


def test_pretrain_objective_adapter_matches_direct_composition():
    args = SimpleNamespace(scage_mips_mask_weight=1.0, graph_angle_weight=0.25)
    payload = {
        "loss_terms": {
            "masked_atom_sum": torch.tensor(6.0),
            "angle_sum": torch.tensor(8.0),
        },
        "zero_reference": torch.tensor(0.125),
    }
    direct = joint_masked_atom_angle_loss(
        payload, args, atom_count=3, angle_count=4
    )
    adapted = compose_joint_payload_loss(
        payload, args, {"masked_atoms": 3, "angle_graphs": 4}
    )
    torch.testing.assert_close(direct, adapted)


def test_pretrain_runtime_config_is_explicit_and_frozen():
    args = SimpleNamespace(
        batch_size=32,
        gradient_accumulation_steps=2,
        max_optimizer_steps=8,
        lr=2e-4,
        weight_decay=0.02,
        warmup_steps=2,
        mips_scheduler="polynomial",
        amp_dtype="bf16",
        seed=42,
    )
    config = PretrainRuntimeConfig.from_args(args, world_size=3)
    assert config.batch_size == 32
    assert config.gradient_accumulation_steps == 2
    assert config.world_size == 3


def test_finetune_engine_is_single_fold_and_scheduler_is_deterministic():
    context = SingleFoldContext("eat", 2, 42, "exp")
    assert run_single_fold(lambda value: value + 1, 4, context=context) == 5
    assert dispatch_units(
        ["eea", "eat"], [1, 0], task_order=["eat", "eea"], fold_order=[0, 1]
    ) == [("eat", 0), ("eat", 1), ("eea", 0), ("eea", 1)]

