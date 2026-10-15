#!/usr/bin/env python3
"""Materialize PI1M topology/static/FP records around audited stored coordinates."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mcl_ph_full_geometry import PI1M, INDEX, source_rows
from scripts.publish_mcl_ph_downstream_direct import equal
from src.dataset.cache_lifecycle import StagingWriter, deserialize_record, select_record_fields
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import build_dual_static, build_pretrain_target
from src.dataset.trimer_mcl import attach_unavailable_trimer_identity, get_etkdg_call_count


def initialize(geometry):
    global SOURCE_DB, SOURCE_ENV
    torch.set_num_threads(1)
    SOURCE_DB = sqlite3.connect(f'file:{geometry / "rows.sqlite"}?mode=ro', uri=True)
    SOURCE_ENV = lmdb.open(str(geometry / 'data.lmdb'), readonly=True,
                           lock=False, readahead=False, max_readers=512)


def materialize(item):
    index, smiles, key_hex = item
    try:
        row = SOURCE_DB.execute('SELECT sample_key, final_status FROM rows WHERE source_index=?',
                                 (index,)).fetchone()
        if row is None or row[0] != key_hex:
            raise ValueError('geometry ledger identity differs')
        key = bytes.fromhex(key_hex)
        topology = build_canonical_periodic_topology(smiles)
        with SOURCE_ENV.begin() as txn:
            payload = txn.get(key)
        if row[1] == 'PASS':
            if payload is None:
                raise ValueError('successful stored geometry missing')
            trimer = deserialize_record(bytes(payload), key)
        else:
            if payload is not None:
                raise ValueError('failed identity unexpectedly has geometry')
            trimer = select_record_fields('trimer', attach_unavailable_trimer_identity(
                topology.clone(), smiles, row[1]))
        if not torch.equal(topology.z, trimer.trimer_atomic_number[trimer.mips_to_trimer_central_index]):
            raise ValueError('published center mapping identity differs')
        static = build_dual_static(topology, trimer, smiles)
        sample = build_dual_sample(topology, trimer, smiles, static=static)
        if bool(sample.geometry_valid) != (row[1] == 'PASS'):
            raise ValueError('published static validity differs from stored coordinates')
        target = build_pretrain_target(topology.normalized_canonical_smiles)
        if get_etkdg_call_count():
            raise ValueError('publication must not generate conformers')
        entry = Data(topology=select_record_fields('topology', topology), trimer=trimer,
            static=static, target=target, normalized_smiles=topology.normalized_canonical_smiles,
            final_geometry_status=row[1], source_index=index)
        return index, key, entry, None
    except Exception as error:
        return index, bytes.fromhex(key_hex), None, f'{type(error).__name__}: {error}'


def publish(args):
    audit = json.loads(args.audit.read_text())
    if (audit.get('conclusion') != 'PASS' or audit.get('scope') != 'full'
            or audit.get('kind') != 'pi1m' or audit['counts'].get('source_rows') != 959359):
        raise ValueError('full PI1M semantic audit is required')
    writer = StagingWriter(args.output, {'schema': 'mcl-ph-direct-records-v1',
        'kind': 'pi1m', 'geometry': str(args.geometry.resolve()),
        'semantic_audit': str(args.audit.resolve()), 'coordinate_generation': False,
        'target': 'existing normalized BRICS and 2048-bit Morgan FP contract'})
    counts = Counter()
    resumed = Counter()
    pool = None
    try:
        def pending():
            for item in source_rows('pi1m'):
                key = bytes.fromhex(item[2])
                if key in writer:
                    record = writer.get(key)
                    if record.source_index != item[0] or 'target' not in record:
                        raise ValueError('resumable published record identity differs')
                    resumed[record.final_geometry_status] += 1
                else:
                    yield item
        pool = mp.get_context('spawn').Pool(args.workers, initializer=initialize,
                                           initargs=(args.geometry,))
        for index, key, entry, error in pool.imap(materialize, pending(), chunksize=4):
            if error:
                raise ValueError(f'{index}/{key.hex()}: {error}')
            inserted, _, _ = writer.put(key, entry)
            if not inserted:
                raise ValueError('duplicate published PI1M identity')
            equal(entry, writer.get(key))
            counts[entry.final_geometry_status] += 1
            if sum(counts.values()) % 10000 == 0:
                print(json.dumps({'published': sum(counts.values()), 'status': dict(counts)}), flush=True)
        pool.close()
        pool.join()
        pool = None
        counts.update(resumed)
        if writer.count() != 959359 or sum(counts.values()) != 959359:
            raise ValueError('PI1M publication identity count differs')
        for status in ('PASS', 'TIMEOUT', 'NO_VALID_CONFORMER', 'ETKDG_NO_VALID_CONFORMER'):
            if counts[status] != audit['counts'].get(status, 0):
                raise ValueError('published geometry statuses differ from audited ledger')
        with np.load(INDEX / 'pretrain_indices.npz', allow_pickle=False) as archive:
            train = archive['train_source_indices']
            validation = archive['validation_source_indices']
        roles = np.zeros(959588, dtype=np.int8)
        roles[train] = 1
        roles[validation] = 2
        offsets = []
        with PI1M.open() as source, (args.output / 'records.jsonl').open('wb') as target:
            for index, line in enumerate(source):
                old = json.loads(line)
                row = {name: old[name] for name in ('sample_key', 'source_smiles', 'normalized_smiles')}
                row.update(source_index=index, pretrain_role=('excluded', 'train', 'validation')[roles[index]])
                offsets.append(target.tell())
                target.write((json.dumps(row) + '\n').encode())
        if len(offsets) != len(roles):
            raise ValueError('PI1M source row count differs')
        np.save(args.output / 'record_offsets.npy', np.asarray(offsets, dtype=np.int64))
        report = {'schema': 'mcl-ph-direct-source-v1', 'kind': 'pi1m',
            'identity_records': writer.count(), 'source_rows': len(offsets),
            'train_rows': len(train), 'validation_rows': len(validation),
            'excluded_rows': int((roles == 0).sum()), 'geometry_status': dict(counts),
            'geometry_source': str(args.geometry.resolve()),
            'pretrain_indices': str((INDEX / 'pretrain_indices.npz').resolve()),
            'coordinate_generation_calls': 0, 'conclusion': 'PASS'}
        writer.sync()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        writer.close()
    (args.output / 'direct_source.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / '.frozen').write_text('mcl-ph-direct-source-v1\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 113), default=24)
    publish(parser.parse_args())
