#!/usr/bin/env python
"""Synthetic 3->8 optimizer-step resume smoke for checkpoint lifecycle."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.training.common.checkpoint import TRAIN_STATE_SCHEMA, atomic_torch_save, load_resume_state
from src.training.common.rng import capture_rng_state, restore_rng_state


def _step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _payload(model, optimizer, step):
    return {
        "schema": TRAIN_STATE_SCHEMA,
        "meta": {"batch_size": 4, "seed": 42, "world_size": 1},
        "train_module": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {},
        "epoch": 0,
        "next_step_idx": int(step),
        "global_step": int(step),
        "optimizer_steps_completed": int(step),
        "rng_state_by_rank": [capture_rng_state()],
        "loader_generator_state_by_rank": [torch.Generator().manual_seed(42).get_state()],
        "sampler_state_by_rank": [{"world_size": 1, "next_batch_index": int(step)}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    y = x.sum(dim=1, keepdim=True)

    reference = torch.nn.Linear(4, 1)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    reference_losses = [_step(reference, reference_optimizer, x[i:i + 4], y[i:i + 4]) for i in range(8)]

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    resumed = torch.nn.Linear(4, 1)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    first_losses = [_step(resumed, resumed_optimizer, x[i:i + 4], y[i:i + 4]) for i in range(3)]
    atomic_torch_save(_payload(resumed, resumed_optimizer, 3), args.state)
    state = load_resume_state(args.state)
    resumed.load_state_dict(state["train_module"], strict=True)
    resumed_optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng_state_by_rank"][0])
    tail_losses = [_step(resumed, resumed_optimizer, x[i:i + 4], y[i:i + 4]) for i in range(3, 8)]
    max_delta = max(float((a - b).abs().max()) for a, b in zip(reference.parameters(), resumed.parameters()))
    report = {
        "schema": "mts-checkpoint-resume-smoke-v1",
        "steps_before_save": 3,
        "steps_after_resume": 5,
        "total_steps": 8,
        "max_parameter_delta": max_delta,
        "losses_match": bool(np.allclose(reference_losses, first_losses + tail_losses, rtol=0, atol=1e-7)),
        "passed": bool(max_delta <= 1e-7 and np.allclose(reference_losses, first_losses + tail_losses, rtol=0, atol=1e-7)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
