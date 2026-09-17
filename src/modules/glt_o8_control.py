"""Matched O8-only control encoder and pretraining heads.

This module is intentionally separate from the dual GLT model.  It reuses the
already-defined O8 Bond-Path implementation, but has no GLT module, geometry
heads, or coordinate-dependent input path.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import BondPathO8, build_dual_glt_model, mean_pool
from .glt_dual_pretrain import AtomGraphDecoder, DualPretrainer, per_graph


def _clone_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_digest(*states):
    """Stable digest for matched initialization evidence."""

    digest = hashlib.sha256()
    for prefix, state in enumerate(states):
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            digest.update(f"{prefix}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class O8ControlModel(nn.Module):
    """O8 graph encoder with a strict zero-padded 1024-wide predictor."""

    architecture_name = "O8-BondPath-O8Control-ZeroPad1024"

    def __init__(self, dropout=0.1):
        super().__init__()
        self.fusion_mode = "o8_control"
        self.o8 = BondPathO8(dropout)
        self.norm2 = nn.LayerNorm(512)
        self.predictor = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(512, 1)
        )

    def encode(self, data, *, atom_mask=None):
        atoms, bias = self.o8(data, atom_mask=atom_mask)
        graph = mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel())
        return {"atom_states": atoms, "bond_path_attention_bias": bias, "graph_2d": graph}

    def fuse(self, encoded):
        graph = self.norm2(encoded["graph_2d"])
        # The second half is a non-learnable, exact zero.  Do not duplicate or
        # project the O8 representation a second time.
        return torch.cat((graph, torch.zeros_like(graph)), dim=-1)

    def forward(self, data):
        return self.predictor(self.fuse(self.encode(data))), None


def load_fixed_concat_o8(model, package, expected_step=5000):
    """Load only O8+norm2 from a Fixed Concat deployment package."""

    if package.get("architecture") != "O8-BondPath-GalformerTrimer-Hop2":
        raise ValueError("reference deployment architecture mismatch")
    if package.get("fusion_mode") != "concat" or package.get("use_md200") is not False:
        raise ValueError("reference deployment must be Fixed Concat without MD200")
    if int(package.get("step", -1)) != int(expected_step):
        raise ValueError("reference deployment step mismatch")
    state = package.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("reference deployment has no state_dict")
    required = {f"o8.{key}" for key in model.o8.state_dict()} | {
        f"norm2.{key}" for key in model.norm2.state_dict()
    }
    selected = {key: state[key] for key in required if key in state}
    if set(selected) != required:
        raise ValueError("reference deployment lacks exact O8/norm2 tensors")
    current = model.state_dict()
    for key, value in selected.items():
        if tuple(current[key].shape) != tuple(value.shape):
            raise ValueError(f"reference tensor shape mismatch: {key}")
        current[key] = value.detach().cpu().clone()
    model.load_state_dict(current, strict=True)


def load_o8_deployment(model, package, expected_step=5000):
    """Strictly load an O8-control deployment package."""

    if package.get("architecture") != model.architecture_name:
        raise ValueError("O8 deployment architecture mismatch")
    if package.get("fusion_mode") != "o8_control" or package.get("use_md200") is not False:
        raise ValueError("O8 deployment metadata mismatch")
    if int(package.get("step", -1)) != int(expected_step):
        raise ValueError("O8 deployment step mismatch")
    expected = {f"o8.{key}" for key in model.o8.state_dict()} | {
        f"norm2.{key}" for key in model.norm2.state_dict()
    }
    state = package.get("state_dict")
    if set(state or {}) != expected:
        raise ValueError("O8 deployment must contain exactly O8 and norm2 tensors")
    current = model.state_dict()
    current.update({key: state[key].detach().cpu().clone() for key in expected})
    model.load_state_dict(current, strict=True)


def matched_predictor_state(seed, fold, *, dropout=0.1):
    """Construct the full reference once to obtain an exact predictor init."""

    from src.utils import set_global_seed

    set_global_seed(int(seed) + int(fold))
    reference = build_dual_glt_model("concat", dropout=dropout)
    state = _clone_state(reference.predictor)
    digest = state_digest(state)
    del reference
    return state, digest


def apply_matched_predictor(model, seed, fold, *, dropout=0.1):
    state, digest = matched_predictor_state(seed, fold, dropout=dropout)
    model.predictor.load_state_dict(state, strict=True)
    return digest


class O8ControlPretrainer(nn.Module):
    """Masked chemistry + Morgan pretrainer with no geometry route."""

    def __init__(self, *, dropout=0.1):
        super().__init__()
        self.encoder = O8ControlModel(dropout=dropout)
        self.encoder.predictor = nn.Identity()
        self.atom_head = AtomGraphDecoder()
        self.fp_head = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(512, 2048)
        )
        self.common_init_digest = None

    def forward(self, data, labels):
        encoded = self.encoder.encode(data, atom_mask=labels["atom_mask"])
        atom_logits = self.atom_head(encoded["atom_states"], data.lga_edge_index)
        mask = labels["atom_mask"].bool()
        graphs = int(data.graph_available.numel())
        if bool(mask.any()):
            chem_values = F.cross_entropy(
                atom_logits[mask].float(), labels["atom_label"][mask].long(), reduction="none"
            )
            chem, chem_valid = per_graph(
                chem_values, data.canonical_graph_index[mask], graphs
            )
        else:
            chem = atom_logits.new_zeros(graphs)
            chem_valid = torch.zeros(graphs, dtype=torch.bool, device=atom_logits.device)
        fp_logits = self.fp_head(self.encoder.fuse(encoded)).float()
        fp_target = labels["fingerprint"].float()
        if fp_target.ndim == 1:
            fp_target = fp_target.unsqueeze(0)
        fp_values = F.binary_cross_entropy_with_logits(
            fp_logits, fp_target, reduction="none"
        ).mean(-1)
        fp_valid = data.graph_available.bool()
        sums = torch.stack((chem[chem_valid].sum(), fp_values[fp_valid].sum()))
        counts = torch.stack((chem_valid.sum(), fp_valid.sum()))
        targets = torch.stack((mask.sum(), fp_valid.sum() * 2048))
        return {"sums": sums, "counts": counts, "targets": targets,
                "atom_logits": atom_logits, "fingerprint_logits": fp_logits}


def common_initialized_o8_pretrainer(seed=42, *, dropout=0.1):
    """Copy the exact common modules from a freshly constructed DualPretrainer."""

    from src.utils import set_global_seed

    set_global_seed(int(seed))
    reference = DualPretrainer("concat", geometry_head_norm=True)
    model = O8ControlPretrainer(dropout=dropout)
    model.encoder.o8.load_state_dict(_clone_state(reference.encoder.o8), strict=True)
    model.encoder.norm2.load_state_dict(_clone_state(reference.encoder.norm2), strict=True)
    model.atom_head.load_state_dict(_clone_state(reference.atom_head), strict=True)
    model.fp_head.load_state_dict(_clone_state(reference.fp_head), strict=True)
    digest = state_digest(
        model.encoder.o8, model.encoder.norm2, model.atom_head, model.fp_head
    )
    model.common_init_digest = digest
    del reference
    return model


def o8_global_objective(sums, global_counts, world_size=1, weights=(1.0, 0.1)):
    """Match DDP's gradient averaging while reducing global sums/counts."""

    if sums.numel() != 2 or global_counts.numel() != 2:
        raise ValueError("O8 objective expects two task components")
    return (sums * sums.new_tensor(weights) * int(world_size)
            / global_counts.clamp_min(1)).sum()


def o8_deployment_package(pretrainer, step):
    state = {}
    for prefix, module in (("o8", pretrainer.encoder.o8), ("norm2", pretrainer.encoder.norm2)):
        state.update({f"{prefix}.{key}": value.detach().cpu().clone()
                      for key, value in module.state_dict().items()})
    return {
        "architecture": pretrainer.encoder.architecture_name,
        "fusion_mode": "o8_control", "step": int(step), "use_md200": False,
        "common_init_digest": pretrainer.common_init_digest,
        "state_dict": state,
    }


__all__ = [
    "O8ControlModel", "O8ControlPretrainer", "common_initialized_o8_pretrainer",
    "matched_predictor_state", "apply_matched_predictor", "load_fixed_concat_o8",
    "load_o8_deployment", "o8_deployment_package", "o8_global_objective",
    "state_digest",
]
