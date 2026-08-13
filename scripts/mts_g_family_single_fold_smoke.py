#!/usr/bin/env python3
"""Two-epoch same-task/same-fold training-path smoke for G0--G3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.mts_g_family_readiness_smoke import _sample
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/mts_multiscale_topology/g_family_readiness_v1/single_fold_smoke.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    results = {}
    for arm in ("g0", "g1", "g2", "g3"):
        torch.manual_seed(42)
        model = MIPSLocalGraphEncoder(
            topology_attention_variant="msta_last2",
            graph_geometry_mode=arm,
            g_family_arm=arm,
            use_star_rbf=False,
            use_mcl=False,
        ).to(device)
        head = torch.nn.Linear(512, 1).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-4)
        losses = []
        for epoch in range(2):
            batch = mips_trimer_collate([_sample(epoch + 1, arm), _sample(epoch + 3, arm)]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(batch)
            prediction = head(output)
            loss = prediction.float().square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        results[arm] = {"epochs": 2, "losses": losses, "finite": all(torch.isfinite(torch.tensor(losses)).tolist())}
        del model, head, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"schema": "mts-g-family-single-fold-smoke-v1", "task": "synthetic_same_task_fold", "arms": results, "device": str(device)}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
