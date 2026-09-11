#!/usr/bin/env python3
"""One or two REAL records, CPU forward/backward and in-memory package loading.

No optimizer, no training loop, no conformer generation or cache writes.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.dataset.glt_dual import FrozenDualLayerSource, build_dual_sample, dual_glt_collate
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective, deployment_package, load_deployment
from scripts.validate_dual_glt import audit_record, fixture_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology-root', required=True)
    parser.add_argument('--trimer-root', required=True)
    parser.add_argument('--sample', action='append', nargs=2, metavar=('HEXKEY', 'PSMILES'), required=True)
    args = parser.parse_args()
    if not 1 <= len(args.sample) <= 2:
        parser.error('one or two real records; include a valid graph with center bonds')
    if len({key for key, _ in args.sample}) != len(args.sample):
        parser.error('provide distinct frozen records')
    torch.set_num_threads(1)
    source = FrozenDualLayerSource(args.topology_root, args.trimer_root,
                                   [(bytes.fromhex(k), s) for k, s in args.sample])
    try:
        records = [source[i] for i in range(len(args.sample))]
        audits = [audit_record(*record) for record in records]
        coverage = fixture_coverage(audits)
        print(json.dumps(dict(data_audits=audits, fixture_coverage=coverage)), file=sys.stderr)
        if coverage['status'] != 'PASS' or any(check['status'] == 'ANOMALY'
                for entry in audits for check in entry['checks'].values()):
            raise ValueError('real data audit failed; no model validation performed')
        clean = dual_glt_collate([build_dual_sample(*r) for r in records])
        centers = torch.bincount(clean.bond_batch[clean.bond_center], minlength=len(records))
        if not clean.geometry_valid.all() or not (centers > 0).any():
            raise ValueError('requires at least one real valid graph with center bonds')
        batch, labels = pretrain_collate([prepare_pretrain_sample(*r, seed=42,
            key=args.sample[i][0], position=i) for i, r in enumerate(records)])
        for mode in ('concat', 'kfuse'):
            torch.manual_seed(42)
            model = DualPretrainer(mode).cpu().eval()
            output = model(batch, labels)
            loss = global_objective(output['sums'], output['counts'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite three-task loss')
            loss.backward()
            for name in ('encoder.o8.atom_embedding.projection.weight', 'encoder.glt.endpoint.weight',
                         'fp_head.3.weight'):
                grad = model.get_parameter(name).grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                    raise RuntimeError(f'invalid required gradient: {name}')
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('nonfinite gradient')
            # A step marker exercises loader semantics; this is NOT a trained 5k artifact.
            package = deployment_package(model, 0)
            downstream = build_dual_glt_model(mode, dropout=0).cpu().eval()
            load_deployment(downstream, package, expected_step=0)
            with torch.no_grad():
                torch.testing.assert_close(model.encoder.fuse(model.encoder.encode(clean)),
                                           downstream.fuse(downstream.encode(clean)))
                prediction = downstream(clean)
                if prediction.shape != (len(records), 1) or not torch.isfinite(prediction).all():
                    raise RuntimeError('invalid downstream forward')
            print(json.dumps(dict(mode=mode, status='LOCAL_FORWARD_BACKWARD_LOAD_PASS',
                                  loss=float(loss.detach()), valid_graphs=output['counts'].tolist(),
                                  targets=output['targets'].tolist())), flush=True)
            del model, downstream, output, loss, package
            gc.collect()
    finally:
        source.close()


if __name__ == '__main__':
    main()
