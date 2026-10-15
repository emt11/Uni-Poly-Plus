#!/usr/bin/env python3
"""Independent full readback of the published PI1M direct source.

The publisher's PASS and ``.frozen`` marker are inputs to this audit, never its
result.  Every source row is compared to the immutable PI1M JSONL and every
eligible identity is decoded from the published LMDB.  No conformer builder is
called and no source digest is used as an identity substitute.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys

import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mcl_ph_full_geometry import PI1M, INDEX
from src.dataset.cache_lifecycle import deserialize_record
from src.dataset.canonical_periodic import resolve_normalized_identity
from src.dataset.glt_dual_static import build_pretrain_target
from src.dataset.lmdb_cache import sample_key_from_normalized


def audit(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    root = args.source.resolve()
    manifest = json.loads((root / 'direct_source.json').read_text())
    if not (root / '.frozen').is_file() or manifest.get('kind') != 'pi1m':
        raise ValueError('PI1M direct source is not frozen with the expected kind')
    if manifest.get('conclusion') != 'PASS' or manifest.get('coordinate_generation_calls') != 0:
        raise ValueError('published PI1M source contract is not PASS/no-generation')
    if int(args.workers) > 1:
        return audit_parallel(args, manifest)
    with np.load(INDEX / 'pretrain_indices.npz', allow_pickle=False) as archive:
        train = np.asarray(archive['train_source_indices'], dtype=np.int64)
        validation = np.asarray(archive['validation_source_indices'], dtype=np.int64)
        fixed = np.asarray(archive['fixed_validation_source_indices'], dtype=np.int64)
    roles = np.zeros(959588, dtype=np.int8)
    roles[train] = 1
    roles[validation] = 2
    if not np.isin(fixed, validation).all():
        raise ValueError('fixed validation is not contained in validation')
    expected = {}
    with PI1M.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            expected[index] = json.loads(line)
    if len(expected) != 959588:
        raise ValueError('immutable PI1M source row count differs')
    offsets = np.load(root / 'record_offsets.npy', allow_pickle=False)
    if offsets.shape != (len(expected),) or offsets.dtype != np.int64:
        raise ValueError('published row offset table shape differs')
    env = lmdb.open(str(root / 'data.lmdb'), readonly=True, lock=False,
                    readahead=False, max_readers=512)
    status = Counter()
    role_counts = Counter()
    decoded = 0
    try:
        with (root / 'records.jsonl').open('rb') as published:
            for index in range(len(expected)):
                published.seek(int(offsets[index]))
                row = json.loads(published.readline())
                source = expected[index]
                role = ('excluded', 'train', 'validation')[int(roles[index])]
                if (int(row.get('source_index', -1)) != index
                        or row.get('pretrain_role') != role
                        or row.get('source_smiles') != source['source_smiles']
                        or row.get('normalized_smiles') != source['normalized_smiles']
                        or row.get('sample_key') != source['sample_key']
                        or row['sample_key'] != sample_key_from_normalized(
                            row['normalized_smiles']).hex()):
                    raise ValueError(f'row {index}: source identity or role differs')
                role_counts[role] += 1
                key = bytes.fromhex(row['sample_key'])
                with env.begin() as txn:
                    payload = txn.get(key)
                if role == 'excluded':
                    if payload is not None:
                        raise ValueError(f'row {index}: excluded identity has a published record')
                    continue
                if payload is None:
                    raise ValueError(f'row {index}: eligible identity is missing from LMDB')
                record = deserialize_record(bytes(payload), key)
                if (record.normalized_smiles != row['normalized_smiles']
                        or record.topology.normalized_canonical_smiles != row['normalized_smiles']
                        or record.source_index != index):
                    raise ValueError(f'row {index}: decoded record identity differs')
                resolve_normalized_identity(record.topology, row['source_smiles'], require_fields=True)
                mapping = torch.as_tensor(record.trimer.mips_to_trimer_central_index).long()
                atoms = torch.as_tensor(record.trimer.trimer_atomic_number).long()
                if (mapping.shape != record.topology.z.shape
                        or not torch.equal(record.topology.z.long(), atoms[mapping])
                        or not torch.equal(record.trimer.trimer_base_ru_atom_id[mapping],
                                           record.topology.canonical_to_trimer_base_atom_id)
                        or not bool(record.trimer.trimer_ru_offset[mapping].eq(0).all())):
                    raise ValueError(f'row {index}: center/atom mapping differs')
                valid = bool(record.final_geometry_status == 'PASS')
                if bool(record.trimer.trimer_geometry_valid) != valid \
                        or bool(record.static['geometry_valid']) != valid:
                    raise ValueError(f'row {index}: geometry state differs')
                positions = torch.as_tensor(record.trimer.trimer_pos)
                if valid and (positions.shape != (atoms.numel(), 3)
                              or not bool(torch.isfinite(positions).all())):
                    raise ValueError(f'row {index}: successful coordinates are invalid')
                if not valid and positions.shape != (0, 3):
                    raise ValueError(f'row {index}: failed identity carries coordinates')
                target = record.target
                expected_target = build_pretrain_target(row['normalized_smiles'])
                if target['brics_groups'] != expected_target['brics_groups']:
                    raise ValueError(f'row {index}: BRICS target differs')
                status[record.final_geometry_status] += 1
                decoded += 1
                if decoded % 10000 == 0:
                    print(json.dumps({'decoded': decoded, 'status': dict(status)}), flush=True)
    finally:
        env.close()
    if decoded != int(manifest['identity_records']) \
            or role_counts != Counter({'train': len(train), 'validation': len(validation),
                                       'excluded': 229}):
        raise ValueError('PI1M direct role or decoded count differs')
    result = {'conclusion': 'PASS', 'scope': 'full', 'kind': 'pi1m',
              'source_rows': len(expected), 'decoded_records': decoded,
              'role_counts': dict(role_counts), 'geometry_status': dict(status),
              'coordinate_generation_calls': 0,
              'identity_contract': 'row-by-row source_index/SMILES/key plus decoded mapping/static/target'}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def _audit_chunk(start, stop, root, offsets, train, validation):
    """Audit one contiguous source range in an isolated read-only worker."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    roles = np.zeros(959588, dtype=np.int8)
    roles[train] = 1; roles[validation] = 2
    env = lmdb.open(str(root / 'data.lmdb'), readonly=True, lock=False,
                    readahead=False, max_readers=512)
    status, role_counts = Counter(), Counter()
    decoded = 0
    try:
        with PI1M.open(encoding='utf-8') as source, (root / 'records.jsonl').open('rb') as published:
            for index, line in enumerate(source):
                if index < int(start):
                    continue
                if index >= int(stop):
                    break
                expected = json.loads(line)
                published.seek(int(offsets[index]))
                row = json.loads(published.readline())
                role = ('excluded', 'train', 'validation')[int(roles[index])]
                if (int(row.get('source_index', -1)) != index
                        or row.get('pretrain_role') != role
                        or row.get('source_smiles') != expected['source_smiles']
                        or row.get('normalized_smiles') != expected['normalized_smiles']
                        or row.get('sample_key') != expected['sample_key']
                        or row['sample_key'] != sample_key_from_normalized(
                            row['normalized_smiles']).hex()):
                    raise ValueError(f'row {index}: source identity or role differs')
                role_counts[role] += 1
                key = bytes.fromhex(row['sample_key'])
                with env.begin() as txn:
                    payload = txn.get(key)
                if role == 'excluded':
                    if payload is not None:
                        raise ValueError(f'row {index}: excluded identity has a published record')
                    continue
                if payload is None:
                    raise ValueError(f'row {index}: eligible identity is missing from LMDB')
                record = deserialize_record(bytes(payload), key)
                if (record.normalized_smiles != row['normalized_smiles']
                        or record.topology.normalized_canonical_smiles != row['normalized_smiles']
                        or record.source_index != index):
                    raise ValueError(f'row {index}: decoded record identity differs')
                resolve_normalized_identity(record.topology, row['source_smiles'], require_fields=True)
                mapping = torch.as_tensor(record.trimer.mips_to_trimer_central_index).long()
                atoms = torch.as_tensor(record.trimer.trimer_atomic_number).long()
                if (mapping.shape != record.topology.z.shape
                        or not torch.equal(record.topology.z.long(), atoms[mapping])
                        or not torch.equal(record.trimer.trimer_base_ru_atom_id[mapping],
                                           record.topology.canonical_to_trimer_base_atom_id)
                        or not bool(record.trimer.trimer_ru_offset[mapping].eq(0).all())):
                    raise ValueError(f'row {index}: center/atom mapping differs')
                valid = record.final_geometry_status == 'PASS'
                if bool(record.trimer.trimer_geometry_valid) != valid \
                        or bool(record.static['geometry_valid']) != valid:
                    raise ValueError(f'row {index}: geometry state differs')
                positions = torch.as_tensor(record.trimer.trimer_pos)
                if valid and (positions.shape != (atoms.numel(), 3)
                              or not bool(torch.isfinite(positions).all())):
                    raise ValueError(f'row {index}: successful coordinates are invalid')
                if not valid and positions.shape != (0, 3):
                    raise ValueError(f'row {index}: failed identity carries coordinates')
                target = record.target
                expected_target = build_pretrain_target(row['normalized_smiles'])
                if target['brics_groups'] != expected_target['brics_groups']:
                    raise ValueError(f'row {index}: BRICS target differs')
                status[record.final_geometry_status] += 1
                decoded += 1
    finally:
        env.close()
    return decoded, dict(status), dict(role_counts)


def audit_parallel(args, manifest):
    root = args.source.resolve()
    with np.load(INDEX / 'pretrain_indices.npz', allow_pickle=False) as archive:
        train = np.asarray(archive['train_source_indices'], dtype=np.int64)
        validation = np.asarray(archive['validation_source_indices'], dtype=np.int64)
    count = int(manifest.get('source_rows', 0))
    offsets = np.load(root / 'record_offsets.npy', allow_pickle=False)
    if count != 959588 or offsets.shape != (count,) or offsets.dtype != np.int64:
        raise ValueError('PI1M direct row offsets differ')
    with PI1M.open(encoding='utf-8') as source:
        if sum(1 for _ in source) != count:
            raise ValueError('immutable PI1M source row count differs')
    workers = max(1, int(args.workers))
    ranges = [(start, min(count, start + (count + workers - 1) // workers))
              for start in range(0, count, (count + workers - 1) // workers)]
    totals, status, role_counts = 0, Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_audit_chunk, start, stop, root, offsets, train, validation)
                   for start, stop in ranges]
        for future in futures:
            decoded, row_status, row_roles = future.result()
            totals += decoded; status.update(row_status); role_counts.update(row_roles)
    expected_roles = Counter({'train': len(train), 'validation': len(validation), 'excluded': 229})
    if totals != int(manifest['identity_records']) or role_counts != expected_roles:
        raise ValueError('PI1M direct role or decoded count differs')
    result = {'conclusion': 'PASS', 'scope': 'full', 'kind': 'pi1m',
              'source_rows': count, 'decoded_records': totals,
              'role_counts': dict(role_counts), 'geometry_status': dict(status),
              'workers': workers, 'coordinate_generation_calls': 0,
              'identity_contract': 'row-by-row source_index/SMILES/key plus decoded mapping/static/target'}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    audit(parser.parse_args())
