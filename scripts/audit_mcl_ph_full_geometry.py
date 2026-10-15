#!/usr/bin/env python3
"""Read every geometry staging row against its source, durable ledger and LMDB."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

import lmdb
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mcl_ph_full_geometry import EXPECTED, source_rows
from src.dataset.cache_lifecycle import deserialize_record


def audit(kind, output, progress_every=10000):
    output = Path(output).resolve()
    metadata = json.loads((output / 'metadata.json').read_text())
    if metadata['kind'] != kind or metadata['schema'] != 'mcl-ph-next-full-trimer-v1':
        raise ValueError('staging source contract differs')
    target = min(EXPECTED[kind], metadata['limit']) if metadata['limit'] else EXPECTED[kind]
    db = sqlite3.connect(f'file:{output / "rows.sqlite"}?mode=ro', uri=True)
    env = lmdb.open(str(output / 'data.lmdb'), subdir=True, readonly=True,
                    lock=False, readahead=False, max_readers=512)
    counts = Counter()
    seen = set()
    try:
        with env.begin() as txn:
            for index, _, key_hex in source_rows(kind, metadata['limit']):
                if index in seen:
                    raise ValueError(f'duplicate source row: {index}')
                seen.add(index)
                row = db.execute('''SELECT sample_key, main_status, retry_status,
                                    final_status, coordinate_lane, main_detail, retry_detail
                                    FROM rows WHERE source_index=?''', (index,)).fetchone()
                if row is None or row[0] != key_hex:
                    raise ValueError(f'{index}: ledger row missing or wrong identity')
                _, main, retry, final, lane, main_json, retry_json = row
                if final == 'PASS':
                    if lane not in ('main', 'retry') or (lane == 'retry' and main != 'TIMEOUT'):
                        raise ValueError(f'{index}: invalid successful lane')
                    payload = txn.get(bytes.fromhex(key_hex))
                    if payload is None:
                        raise ValueError(f'{index}: PASS row absent from LMDB')
                    data = deserialize_record(bytes(payload), bytes.fromhex(key_hex))
                    pos = torch.as_tensor(data.trimer_pos)
                    atoms = torch.as_tensor(data.trimer_atomic_number).reshape(-1)
                    if (pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != atoms.numel()
                            or pos.shape[0] == 0 or not torch.isfinite(pos).all()
                            or not bool(data.trimer_geometry_valid)):
                        raise ValueError(f'{index}: invalid stored all-atom coordinates')
                    detail = json.loads(retry_json if lane == 'retry' else main_json)
                    if (detail['atoms'] != pos.shape[0] or
                            detail['explicit_h_atoms'] != int(atoms.eq(1).sum())):
                        raise ValueError(f'{index}: QC and all-atom record differ')
                    center = torch.as_tensor(data.mips_to_trimer_central_index).reshape(-1)
                    if center.numel() == 0 or center.min() < 0 or center.max() >= pos.shape[0]:
                        raise ValueError(f'{index}: invalid center atom mapping')
                    counts['PASS'] += 1
                    counts['explicit_h_atoms'] += detail['explicit_h_atoms']
                    counts[f'static_{detail["static_interface"]}'] += 1
                else:
                    if txn.get(bytes.fromhex(key_hex)) is not None:
                        raise ValueError(f'{index}: failed row has LMDB geometry')
                    if final == 'TIMEOUT' and retry is None:
                        raise ValueError(f'{index}: TIMEOUT was not retried')
                    counts[final] += 1
                if len(seen) % progress_every == 0:
                    print(json.dumps({'audited': len(seen), 'target': target,
                                      'statuses': dict(counts)}), flush=True)
        ledger_count = db.execute('SELECT COUNT(*) FROM rows').fetchone()[0]
        lmdb_count = env.stat()['entries']
        if len(seen) != target or ledger_count != target or lmdb_count != counts['PASS']:
            raise ValueError(f'count mismatch: source={len(seen)}, ledger={ledger_count}, '
                             f'LMDB={lmdb_count}, PASS={counts["PASS"]}, target={target}')
    finally:
        db.close()
        env.close()
    summary = {'kind': kind, 'source_rows': target, 'ledger_rows': ledger_count,
               'lmdb_records': lmdb_count, 'status_counts': dict(counts),
               'conclusion': 'PASS'}
    (output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=EXPECTED, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--progress-every', type=int, default=10000)
    args = parser.parse_args()
    audit(args.kind, args.output, args.progress_every)
