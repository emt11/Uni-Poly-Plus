#!/usr/bin/env python3
"""At most two real frozen records, CPU only, no optimizer/cache writes.

Run on a supported environment only. Example:
python scripts/validate_dual_glt.py --topology-root PATH --trimer-root PATH \
    --sample HEXKEY 'P-SMILES' --sample HEXKEY_N0 'P-SMILES_N0'
"""
import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from src.dataset.glt_dual import FrozenDualLayerSource, DualGLTDataset, dual_glt_collate
from src.modules.glt_dual import build_dual_glt_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology-root', required=True)
    parser.add_argument('--trimer-root', required=True)
    parser.add_argument('--sample', action='append', nargs=2, metavar=('HEXKEY', 'PSMILES'), required=True)
    args = parser.parse_args()
    if len(args.sample) != 2:
        parser.error('provide exactly two records: one ordinary and one real N=0')
    torch.set_num_threads(1)
    samples = [(bytes.fromhex(key), smiles) for key, smiles in args.sample]
    source = FrozenDualLayerSource(args.topology_root, args.trimer_root, samples)
    try:
        dataset = DualGLTDataset(source)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=dual_glt_collate)))
        centers = torch.bincount(batch.bond_batch[batch.bond_center], minlength=2)
        if not batch.geometry_valid.all() or not (centers == 0).any() or not (centers > 0).any():
            raise RuntimeError('fixtures must contain valid ordinary and N=0 geometry')
        for mode in ('concat', 'kfuse'):
            torch.manual_seed(42)
            model = build_dual_glt_model(mode, dropout=0).cpu()
            prediction = model(batch)
            if prediction.shape != (2, 1) or not torch.isfinite(prediction).all():
                raise RuntimeError('invalid predictions')
            (prediction - torch.tensor([[0.3], [-0.7]])).square().mean().backward()
            names = ['o8.bond_bias.position', 'o8.layers.0.attention.qkv.weight',
                     'glt.endpoint.weight', 'glt.distance_projection.weight',
                     'glt.angle_bias.heads.2.weight', 'glt.layers.0.attention.qkv.weight',
                     'predictor.0.weight']
            if mode == 'kfuse':
                names.append('kfuse.v_proj.glt3d.weight')
                torch.testing.assert_close(model.kfuse.last_attention_weights,
                                           torch.ones_like(model.kfuse.last_attention_weights))
            gradients = {}
            for name in names:
                grad = model.get_parameter(name).grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                    raise RuntimeError(f'missing/nonfinite gradient: {name}')
                gradients[name] = float(grad.abs().sum())
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('nonfinite gradient')
            print({'mode': mode, 'status': 'CPU_FORWARD_BACKWARD_PASS',
                   'centers': centers.tolist(), 'gradients': gradients})
            del model, prediction
            gc.collect()
    finally:
        source.close()


if __name__ == '__main__':
    main()
