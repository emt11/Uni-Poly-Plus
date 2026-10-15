#!/usr/bin/env python3
"""Independently verify stored Trimer chemistry without generating conformers."""
import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys

import lmdb
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_mcl_ph_full_geometry import source_rows
from src.dataset.cache_lifecycle import deserialize_record
from src.dataset.cache_spec import RECORD_FIELDS
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.trimer_mcl import (
    _canonical_source_mol, build_periodic_multimer_mol,
    _validate_trimer_contract, _attach_all_atom_topology,
    _audit_declared_stereo, get_etkdg_call_count,
)


def verify(smiles, stored):
    expected = build_canonical_periodic_topology(smiles)
    molecule, metadata = build_periodic_multimer_mol(
        _canonical_source_mol(smiles), 3, close_periodic=False,
        terminal_capping='missing_seam_bond_order_hydrogen_equivalents')
    base, units, mapping = _validate_trimer_contract(expected, molecule, metadata)
    hydrogenated = Chem.AddHs(Chem.Mol(molecule))
    _attach_all_atom_topology(expected, molecule, hydrogenated,
                              metadata, base, units, mapping)
    checked = 0
    for field in RECORD_FIELDS['trimer']:
        # Geometry and QC are audited separately; compare chemistry fields
        # reconstructed solely from the declared original structure.
        if field not in expected:
            continue
        if field not in stored or not torch.equal(
                torch.as_tensor(expected[field]), torch.as_tensor(stored[field])):
            raise ValueError('stored chemistry differs: ' + field)
        checked += 1
    positions = torch.as_tensor(stored.trimer_pos)
    if (positions.shape != (hydrogenated.GetNumAtoms(), 3)
            or not bool(torch.isfinite(positions).all())
            or not bool(stored.trimer_geometry_valid)
            or not bool(stored.trimer_geometry_is_3d)
            or bool(stored.trimer_2d_fallback)):
        raise ValueError('stored coordinate validity differs')
    double, tetra = _audit_declared_stereo(molecule, hydrogenated, positions)
    if get_etkdg_call_count():
        raise RuntimeError('semantic audit must never embed coordinates')
    return checked, double, tetra


def worker_init(source):
    global WORKER_DB, WORKER_ENV
    torch.set_num_threads(1)
    WORKER_DB = sqlite3.connect(f'file:{source / "rows.sqlite"}?mode=ro', uri=True)
    WORKER_ENV = lmdb.open(str(source / 'data.lmdb'), readonly=True, lock=False,
                           readahead=False, max_readers=512)


def verify_row(item, db, env):
    index, smiles, key = item
    counts = Counter()
    row = db.execute('SELECT sample_key, final_status FROM rows '
                     'WHERE source_index=?', (index,)).fetchone()
    if row is None or row[0] != key:
        raise ValueError('ledger identity differs')
    with env.begin() as txn:
        payload = txn.get(bytes.fromhex(key))
    if row[1] == 'PASS':
        if payload is None:
            raise ValueError('successful geometry missing')
        fields, double, tetra = verify(
            smiles, deserialize_record(bytes(payload), bytes.fromhex(key)))
        counts.update(chemistry_fields=fields, declared_double_stereo=double,
                      declared_tetra_stereo=tetra)
    elif payload is not None:
        raise ValueError('failed identity has geometry')
    counts[row[1]] += 1
    counts['source_rows'] += 1
    return counts


def worker_verify(item):
    try:
        return item[0], item[2], verify_row(item, WORKER_DB, WORKER_ENV), None
    except Exception as error:
        return item[0], item[2], None, f'{type(error).__name__}: {error}'


def audit(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    metadata = json.loads((args.source / 'metadata.json').read_text())
    if metadata['kind'] != args.kind or metadata['limit'] != 0:
        raise ValueError('unexpected source contract')
    db = sqlite3.connect(f'file:{args.source / "rows.sqlite"}?mode=ro', uri=True)
    env = lmdb.open(str(args.source / 'data.lmdb'), readonly=True, lock=False,
                    readahead=False, max_readers=512)
    counts = Counter()
    current = None
    pool = None
    try:
        if args.workers == 1:
            results = ((index, key, verify_row((index, smiles, key), db, env), None)
                       for index, smiles, key in source_rows(args.kind))
        else:
            pool = mp.get_context('spawn').Pool(
                args.workers, initializer=worker_init, initargs=(args.source,))
            results = pool.imap(worker_verify, source_rows(args.kind), chunksize=16)
        for index, key, checked, error in results:
                current = {'source_index': index, 'sample_key': key}
                if error is not None:
                    raise ValueError(error)
                counts.update(checked)
                if counts['source_rows'] % args.progress_every == 0:
                    print(json.dumps(dict(counts)), flush=True)
                if args.limit and counts['source_rows'] >= args.limit:
                    break
        if not args.limit and counts['source_rows'] != metadata['source_count']:
            raise ValueError('semantic audit source count differs')
        report = {'conclusion': 'PASS', 'scope': 'probe' if args.limit else 'full',
                  'kind': args.kind, 'counts': dict(counts), 'etkdg_calls': 0,
                  'workers': args.workers}
    except Exception as error:
        report = {'conclusion': 'FAIL', 'kind': args.kind, 'at': current,
                  'counts': dict(counts), 'error': str(error)}
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        db.close()
        env.close()
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('pi1m', 'downstream'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, choices=range(1, 25), default=1)
    parser.add_argument('--progress-every', type=int, default=100)
    audit(parser.parse_args())
