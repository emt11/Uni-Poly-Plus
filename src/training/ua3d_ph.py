"""PH campaign training, diagnostics, freeze, and TEST adapter.

Parent trainers and their output roots are not imported. This file records the
fixed seven-arm contract and performs one unit at a time.
"""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.modules import ua3d_ph as ph
from src.modules.ua3d_stability import configure, check_det
from src.modules.ua3d_base_opt_r6 import initialize_ft
from src.training.ua3m_paperfuse import read, write, exclusive, now, seed_all, metrics, transform, scaler_validate, require_tmux

ROOT = Path(__file__).resolve().parents[2]
PLAN = "UA3D-PH-20261028-01"
REVISION = "r1.1"
OUT = ROOT / "results/ua3d_ph_20261028/r1"
LOG = ROOT / "logs/ua3d_ph_20261028/r1"
REF = ROOT / "results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1"
REFERENCE = 0.804675462276949
TASKS = ("eea", "egb", "egc", "ei", "eps", "nc", "xc")
SPEC = json.loads((ROOT / "configs/ua3d_ph_r1.json").read_text())


def gpu_for(task, fold):
    return (5 * TASKS.index(task) + fold) % 2


def unit_dir(arm, task, fold, formal=False):
    if arm not in SPEC["arms"] or task not in TASKS or fold not in range(5):
        raise ValueError("outside fixed matrix")
    return OUT / ("formal" if formal else "finetune") / arm / task / f"fold{fold}"


def assert_budget(kind, amount=1):
    budget = read(OUT / "budget.json")
    if budget["reserved"].get(kind, 0) + amount > SPEC["limits"][kind]:
        raise RuntimeError(f"{kind} budget exhausted")
    budget["reserved"][kind] += amount
    write(OUT / "budget.json", budget)


def load_base(task, fold, arm):
    """Initialize the unchanged BASE first, then attach only the requested adapter."""
    recipe = read(REF / "registry/recipes.json")["recipes"]["F_PD_N1_F0"]
    package = torch.load(REF / "pretrain/F_PD_N1_F0/deploy.pt", map_location="cpu", weights_only=False)
    head_root = ROOT / "results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/initialization/heads"
    head = torch.load(head_root / task / f"fold{fold}.pt", map_location="cpu", weights_only=False)
    model = initialize_ft(package, head, recipe)
    if arm == "C0_BASE":
        return model
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(ph.ADAPTER_SEED)
        adapted = ph.PHFuse(arm, ft=recipe["ft"])
    prefixes = ph.allowed_new_prefixes(arm)
    base_state = model.state_dict(); adapted_state = adapted.state_dict()
    if set(adapted_state) - set(base_state) != {k for k in adapted_state if k.startswith(prefixes)}:
        raise RuntimeError("unexpected PH parameter")
    for name, module in adapted.named_children():
        if name in ("graph_ph", "spatial_ph"): continue
        module.load_state_dict({k[len(name)+1:]: v for k, v in base_state.items() if k.startswith(name+".")}, strict=True)
    return adapted
