#!/usr/bin/env python3
"""Independently check fitted source rows and optionally recompute a small probe."""
import argparse
import json
from pathlib import Path
import sys
import sqlite3
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset.mcl_ph_view import load_geometric_statistics, prepare_mcl_ph_sample
from src.training.glt_dual_runtime import load_sample_index_artifact, open_source



def validate_final_geometry(fit, valid, entries, ledger):
    with sqlite3.connect(Path(ledger).resolve().as_uri() + '?mode=ro', uri=True) as connection:
        final = {int(index): (key, status) for index, key, status in
                 connection.execute('SELECT source_index, sample_key, final_status FROM rows')}
    expected = []
    counts = Counter()
    for index in fit:
        key, status = final[int(index)]
        if key != entries[int(index)]['sample_key']:
            raise ValueError('statistics source key differs from final geometry ledger')
        if status not in ('PASS', 'TIMEOUT', 'NO_VALID_CONFORMER', 'ETKDG_NO_VALID_CONFORMER'):
            raise ValueError('unrecognized final geometry status')
        counts[status] += 1
        if status == 'PASS': expected.append(index)
    np.testing.assert_array_equal(valid, expected)
    return counts


def audit(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    metadata = json.loads(args.statistics.with_suffix('.json').read_text())
    statistics = load_geometric_statistics(args.statistics)
    expected = load_sample_index_artifact(args.sample_index, 'train')['indices']
    if args.probe_rows:
        expected = expected[:args.probe_rows]
    if metadata['scope'] != ('probe' if args.probe_rows else 'full'):
        raise ValueError('statistics scope differs from requested audit')
    if Path(metadata['source_root']).resolve() != args.source_root.resolve():
        raise ValueError('statistics source root differs')
    with np.load(args.statistics, allow_pickle=False) as payload:
        fit = payload['fit_source_indices'].copy()
        valid = payload['valid_source_indices'].copy()
    np.testing.assert_array_equal(fit, expected)
    source, _ = open_source(args.source_root, args.source_root)
    router, length, nonbond, observed_valid = [], [], [], []
    final_status_counts = Counter()
    try:
        for index in fit:
            if source.entries[int(index)]['pretrain_role'] != 'train':
                raise ValueError('validation or excluded source row fitted')
        if not args.probe_rows:
            final_status_counts = validate_final_geometry(fit, valid, source.entries,
                Path(source.manifest['geometry_source']) / 'rows.sqlite')
        if args.probe_rows:
            for index in fit:
                record = source.record(int(index))
                if not bool(record.static['geometry_valid']):
                    continue
                data, labels = prepare_mcl_ph_sample(
                    record.topology, record.trimer, source.entries[int(index)]['source_smiles'],
                    seed=statistics['seed'], key=source.samples[int(index)][0].hex(),
                    position=int(index), sigma=statistics['sigma_noise'],
                    ratio=metadata['mask_ratio'], static=record.static, statistics=None)
                router.append(data.mcl_trajectory.numpy().astype(np.float64))
                length.extend(labels['mcl_length_raw'].numpy().astype(np.float64))
                nonbond.extend(labels['mcl_nonbond_raw'].numpy().astype(np.float64))
                observed_valid.append(index)
            np.testing.assert_array_equal(valid, observed_valid)
            raw = np.asarray(router)
            np.testing.assert_allclose(statistics['router_mean'], raw.mean(axis=0).astype(np.float32), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(statistics['router_std'], np.maximum(raw.std(axis=0), 1e-6).astype(np.float32), rtol=1e-6, atol=1e-7)
            for name, values in [('length', length), ('nonbond', nonbond)]:
                raw = np.log1p(values)
                np.testing.assert_allclose(statistics[name]['mu'], raw.mean(), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(statistics[name]['sigma'], max(raw.std(), 1e-6), rtol=1e-12, atol=1e-12)
    finally:
        source.close()
    report = {'conclusion': 'PASS', 'scope': metadata['scope'], 'train_rows': len(fit),
              'valid_rows': len(valid), 'validation_rows_used': 0,
              'final_geometry_status_counts': dict(final_status_counts),
              'valid_indices_vs_final_geometry': 'exact' if not args.probe_rows else 'probe decoded static',
              'checks': 'exact index/roles and finite scales' + ('; independent raw-array recomputation' if args.probe_rows else ''),
              'statistics': str(args.statistics.resolve())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--sample-index', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--probe-rows', type=int, default=0)
    audit(parser.parse_args())
