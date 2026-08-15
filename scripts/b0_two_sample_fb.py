#!/usr/bin/env python3
"""B0-v2 two-sample forward/backward smoke: one geometry-valid + one invalid.

Runs one real forward, loss, backward and Adam step on GPU, and checks:
- losses/metrics finite, step-0 R_denoise ~= 1, displacement RMS ~= 0;
- the invalid geometry sample contributes zero coordinate displacement/loss;
- Star-RBF projection and coordinate decoder receive finite gradients.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataloader import mips_trimer_collate  # noqa: E402
from src.modules.periodic_coordinate_decoder import PeriodicCoordinateDecoder  # noqa: E402
from src.training.pretrain.config import parse_arguments  # noqa: E402
from src.training.pretrain.engine import (  # noqa: E402
    MIPSPretrainContainer,
    _b0_differentiable_mean,
    _b0_reduce_metrics,
    _build_b0_model,
    dataset_kwargs_from_args,
)


def main(argv=None):
    gpu_id = int(argv[1]) if len(argv or []) > 1 else 1
    args = parse_arguments(["--b0_config", str(PROJECT_ROOT / "configs/mts/b0_v2_probe.json")])
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    from src.dataset import UniDataset

    kwargs = dataset_kwargs_from_args(args)
    kwargs["rebuild_feature_cache"] = False
    dataset = UniDataset(**kwargs)
    if len(dataset) != 995799:
        raise RuntimeError(f"expected full PI1M_v2 cohort, got {len(dataset)}")
    if getattr(dataset, "_star_rbf_v2_sidecar", None) is None:
        raise RuntimeError("frozen Star-RBF v2 sidecar missing")

    valid_index = invalid_index = None
    scan_limit = 20000
    for index in range(min(len(dataset), scan_limit)):
        item = dataset[index]
        ok = bool(
            getattr(item, "graph_available", False)
            and getattr(item, "trimer_geometry_valid", False)
            and getattr(item, "trimer_geometry_is_3d", False)
            and not getattr(item, "trimer_2d_fallback", False)
        )
        if ok and valid_index is None:
            valid_index = index
        elif not ok and invalid_index is None:
            invalid_index = index
        if valid_index is not None and invalid_index is not None:
            break
    if valid_index is None or invalid_index is None:
        raise RuntimeError(
            f"scan of {scan_limit} samples found valid={valid_index} "
            f"invalid={invalid_index}"
        )
    print(f"selected valid index={valid_index} invalid index={invalid_index}", flush=True)

    items = [dataset[valid_index], dataset[invalid_index]]
    data = mips_trimer_collate(items).to(device)
    print(
        "graph_valid flags:",
        [
            bool(v)
            for v in zip(
                data.graph_available.bool().tolist(),
                data.trimer_geometry_valid.bool().tolist(),
            )
        ],
        flush=True,
    )

    model = _build_b0_model(args)
    graph_encoder = model.encoders["graph"].encoder
    graph_dim = int(graph_encoder.emb_dim)
    atom_head = nn.Linear(graph_dim, int(graph_encoder.masked_atom_classes))
    coordinate_decoder = PeriodicCoordinateDecoder(graph_dim)
    container = MIPSPretrainContainer(
        model, {"mips_atom": atom_head, "coordinate": coordinate_decoder}
    ).to(device)
    for parameter in container.parameters():
        parameter.requires_grad = False
    for module in (
        graph_encoder.atom_embedding, graph_encoder.spd_embedding,
        graph_encoder.path_bias, graph_encoder.layers,
        graph_encoder.star_distance_bias, atom_head, coordinate_decoder,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True

    # B0-v2 RBF contract check mirrors the training loop.
    sidecar_upper = float(dataset._star_rbf_v2_sidecar.rbf_upper)
    star_bias = graph_encoder.star_distance_bias
    assert abs(sidecar_upper - 3.75) < 1e-9, sidecar_upper
    assert abs(float(args.star_rbf_upper) - 3.75) < 1e-9
    assert int(star_bias.centers.numel()) == 32
    assert abs(float(star_bias.centers[-1]) - 3.75) < 1e-6
    assert abs(float(star_bias.centers[0])) < 1e-9
    print(
        f"RBF contract ok: upper={sidecar_upper} centers_last="
        f"{float(star_bias.centers[-1])}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in container.parameters() if p.requires_grad],
        lr=float(args.lr), betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
    )
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(args.seed))

    payload = container(
        "b0_periodic_coordinate_denoising", data, args, 0,
        noise_generator=noise_generator,
    )
    atom_loss, atom_count, _ = _b0_differentiable_mean(
        payload["loss_terms"]["masked_atom_sum"],
        payload["counts"]["masked_atoms"], device,
    )
    coord_loss, coord_count, _ = _b0_differentiable_mean(
        payload["loss_terms"]["coordinate_sum"],
        payload["counts"]["coordinate_graphs"], device,
    )
    metrics = _b0_reduce_metrics(payload, device)
    zero_loss = metrics["zero_sum"] / max(1, int(metrics["valid_graph_count"]))
    loss = (
        atom_loss + float(args.b0_coordinate_loss_weight) * coord_loss
        + payload["zero_reference"]
    )
    print(
        f"step0 atom_loss={float(atom_loss):.6f} coord_loss={float(coord_loss):.6f} "
        f"zero_loss={float(zero_loss):.6f} coord_count={int(coord_count)} "
        f"atom_count={int(atom_count)} graphs={payload['counts']['graphs']}",
        flush=True,
    )
    assert int(coord_count) == 1, "invalid geometry sample must not count"
    assert int(atom_count) == payload["counts"]["masked_atoms"]
    assert int(metrics["valid_graph_count"]) == 1
    r_denoise = float(coord_loss) / max(float(zero_loss), 1e-8)
    disp_rms = float(
        metrics["displacement_squared_sum"] / max(1, int(metrics["displacement_element_count"]))
    ) ** 0.5
    assert abs(r_denoise - 1.0) < 1e-3, f"step-0 R_denoise={r_denoise}"
    assert disp_rms < 1e-6, f"step-0 displacement RMS={disp_rms}"
    print(f"step0 R_denoise={r_denoise:.6f} displacement_rms={disp_rms:.6f}", flush=True)

    graph_ids = data.canonical_graph_index.long()
    invalid_nodes = graph_ids == 1
    with torch.no_grad():
        from src.training.pretrain.engine import _joint_canonical_mask
        from src.training.pretrain.periodic_denoising import (
            add_canonical_correlated_noise,
        )

        atom_mask = _joint_canonical_mask(
            data, int(args.seed), 0, float(args.graph_mask_ratio)
        )
        noisy, distances, observation_mask = add_canonical_correlated_noise(
            data, float(args.b0_noise_sigma),
            generator=torch.Generator(device=device).manual_seed(1234),
        )
        _, node_states, _ = graph_encoder.forward_b0_pretrain(
            data, atom_mask, distances, observation_mask
        )
        displacement, _ = coordinate_decoder(data, node_states, noisy)
    assert torch.equal(
        displacement[invalid_nodes], torch.zeros_like(displacement[invalid_nodes])
    ), "invalid geometry sample must have zero displacement"

    loss.backward()
    star_projection = graph_encoder.star_distance_bias.projection
    assert star_projection.weight.grad is not None
    assert torch.isfinite(star_projection.weight.grad).all()
    for name, parameter in coordinate_decoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(
        torch.isfinite(p.grad).all()
        for p in container.parameters()
        if p.requires_grad and p.grad is not None
    )
    print("gradients finite on Star-RBF projection and coordinate decoder", flush=True)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    payload = container(
        "b0_periodic_coordinate_denoising", data, args, 0,
        noise_generator=noise_generator,
    )
    atom_loss, atom_count, _ = _b0_differentiable_mean(
        payload["loss_terms"]["masked_atom_sum"],
        payload["counts"]["masked_atoms"], device,
    )
    coord_loss, coord_count, _ = _b0_differentiable_mean(
        payload["loss_terms"]["coordinate_sum"],
        payload["counts"]["coordinate_graphs"], device,
    )
    assert torch.isfinite(atom_loss) and torch.isfinite(coord_loss)
    assert int(coord_count) == 1
    print(
        f"post-step atom_loss={float(atom_loss):.6f} coord_loss={float(coord_loss):.6f}",
        flush=True,
    )

    output = Path(str(args.b0_result_root)) / "two_sample_fb_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "mts-b0-v2-two-sample-fb-v1",
        "valid_index": valid_index,
        "invalid_index": invalid_index,
        "step0_atom_loss": float(atom_loss),
        "step0_coord_loss": float(coord_loss),
        "step0_zero_loss": float(zero_loss),
        "step0_r_denoise": r_denoise,
        "step0_displacement_rms": disp_rms,
        "coord_count": int(coord_count),
        "star_rbf_upper": sidecar_upper,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
