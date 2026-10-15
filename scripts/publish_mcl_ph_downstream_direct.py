#!/usr/bin/env python3
"""Publish active-fold downstream chemistry using only existing coordinates."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import sys

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mcl_ph_full_geometry import DOWNSTREAM, INDEX, source_rows
from scripts.build_mcl_ph_next_indices import TASKS, role_lists
from scripts.audit_mcl_ph_geometry_semantics import verify
from src.dataset.cache_lifecycle import StagingWriter, deserialize_record, select_record_fields
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.trimer_mcl import attach_unavailable_trimer_identity, get_etkdg_call_count


def equal(left, right):
    if isinstance(left, Data):
        equal(left.to_dict(), right.to_dict())
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError('roundtrip fields differ')
        for key in left:
            equal(left[key], right[key])
    elif torch.is_tensor(left):
        if not torch.equal(left, right):
            raise ValueError('roundtrip tensor differs')
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise ValueError('roundtrip array differs')
    elif left != right:
        raise ValueError('roundtrip value differs')


def publish(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    if (audit.get('conclusion') != 'PASS' or audit.get('scope') != 'full'
            or audit.get('kind') != 'downstream'
            or audit['counts'].get('source_rows') != 3655):
        raise ValueError('full downstream semantic audit is required')
    records = [json.loads(line) for line in DOWNSTREAM.open()]
    direct = [json.loads(line) for line in (INDEX / 'downstream_rows.jsonl').open()]
    if len(records) != len(direct) or len(records) != 6265:
        raise ValueError('downstream direct rows differ')
    raw, roles = {}, {}
    for task in TASKS:
        with (ROOT / f'data/raw/smi_{task}.csv').open(newline='') as stream:
            raw[task] = list(csv.reader(stream))[1:]
        manifest = json.loads((ROOT / f'data/splits/mips_paper5_outer/{task}.json').read_text())
        if manifest['task'] != task or manifest['protocol'] != 'paper5_outer':
            raise ValueError('activity split contract differs')
        roles[task] = role_lists(manifest, len(raw[task]))
    db = sqlite3.connect(f'file:{args.geometry / "rows.sqlite"}?mode=ro', uri=True)
    geometry = lmdb.open(str(args.geometry / 'data.lmdb'), readonly=True,
                         lock=False, readahead=False, max_readers=512)
    writer = StagingWriter(args.output, {'schema': 'mcl-ph-direct-records-v1',
        'kind': 'downstream', 'geometry': str(args.geometry.resolve()),
        'semantic_audit': str(args.audit.resolve()), 'coordinate_generation': False})
    counts = Counter()
    status_by_key = {}
    try:
        for index, smiles, key_hex in source_rows('downstream'):
            row = db.execute('SELECT sample_key, final_status FROM rows WHERE source_index=?',
                             (index,)).fetchone()
            if row is None or row[0] != key_hex:
                raise ValueError('geometry ledger identity differs')
            key = bytes.fromhex(key_hex)
            topology = build_canonical_periodic_topology(smiles)
            if topology.normalized_canonical_smiles != records[index]['normalized_smiles']:
                raise ValueError('normalized source identity differs')
            with geometry.begin() as txn:
                payload = txn.get(key)
            if row[1] == 'PASS':
                if payload is None:
                    raise ValueError('successful geometry absent')
                trimer = deserialize_record(bytes(payload), key)
                verify(smiles, trimer)
            else:
                if payload is not None:
                    raise ValueError('failed geometry unexpectedly exists')
                trimer = select_record_fields('trimer', attach_unavailable_trimer_identity(
                    topology.clone(), smiles, row[1]))
            static = build_dual_static(topology, trimer, smiles)
            sample = build_dual_sample(topology, trimer, smiles, static=static)
            if bool(sample.geometry_valid) != (row[1] == 'PASS'):
                raise ValueError(f'{index}: new static geometry validity differs')
            entry = Data(topology=select_record_fields('topology', topology),
                         trimer=trimer, static=static, normalized_smiles=topology.normalized_canonical_smiles,
                         final_geometry_status=row[1])
            inserted, _, _ = writer.put(key, entry)
            if not inserted:
                raise ValueError('duplicate direct identity')
            equal(entry, writer.get(key))
            status_by_key[key_hex] = row[1]
            counts[row[1]] += 1
            if sum(counts.values()) % 100 == 0:
                print(json.dumps(dict(counts)), flush=True)
        if writer.count() != 3655 or get_etkdg_call_count():
            raise ValueError('direct identity count or no-embedding contract differs')
        tasks = {task: [] for task in TASKS}
        for source_index, (old, identity) in enumerate(zip(records, direct)):
            task, original = old['task'], int(old['original_row'])
            if (identity['source_index'] != source_index or identity['task'] != task
                    or identity['original_row'] != original
                    or identity['normalized_smiles'] != old['normalized_smiles']
                    or old['source_smiles'].strip() != raw[task][original][0].strip()
                    or float(old['label']) != float(raw[task][original][1])):
                raise ValueError('direct task identity or label differs')
            fold_roles = [roles[task][fold][original] for fold in range(5)]
            if identity['fold_roles'] != fold_roles:
                raise ValueError('direct activity roles differ')
            tasks[task].append({name: old[name] for name in (
                'task', 'original_row', 'source_smiles', 'normalized_smiles', 'sample_key', 'label')})
            tasks[task][-1].update(source_index=source_index,
                fold_membership=[{'fold': fold, 'role': role} for fold, role in enumerate(fold_roles)],
                geometry_status=status_by_key[old['sample_key']])
        for task, rows in tasks.items():
            rows.sort(key=lambda row: row['original_row'])
            if [row['original_row'] for row in rows] != list(range(len(raw[task]))):
                raise ValueError('direct task rows incomplete or duplicated')
            offsets = []
            with (args.output / f'{task}.jsonl').open('xb') as stream:
                for row in rows:
                    offsets.append(stream.tell())
                    stream.write((json.dumps(row) + '\n').encode())
            np.save(args.output / f'{task}_offsets.npy', np.asarray(offsets, dtype=np.int64))
        report = {'schema': 'mcl-ph-direct-source-v1', 'kind': 'downstream',
            'protocol': 'paper5_outer', 'identity_records': writer.count(),
            'task_rows': {task: len(rows) for task, rows in tasks.items()},
            'geometry_status': dict(counts), 'geometry_source': str(args.geometry.resolve()),
            'coordinate_generation_calls': get_etkdg_call_count(), 'conclusion': 'PASS'}
        writer.sync()
    finally:
        writer.close()
        geometry.close()
        db.close()
    (args.output / 'direct_source.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / '.frozen').write_text('mcl-ph-direct-source-v1\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    publish(parser.parse_args())
