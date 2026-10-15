#!/usr/bin/env python3
"""本周期一次性MD200准备：显式source行映射，不生成坐标或文件指纹。"""
import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.mips_descriptors.rdNormalizedDescriptors import RDKit2DNormalized


def descriptor(generator, normalized):
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        return np.zeros(200, np.float32), False, 'invalid_ru_smiles'
    if Chem.MolToSmiles(molecule, canonical=True) != normalized:
        raise ValueError('source normalized_smiles differs from canonical RU identity')
    stars = [a for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2 or any(a.GetDegree() != 1 for a in stars):
        return np.zeros(200, np.float32), False, 'star_sub_requires_two_terminal_stars'
    opposite = [stars[1].GetNeighbors()[0].GetAtomicNum(),
                stars[0].GetNeighbors()[0].GetAtomicNum()]
    changed = Chem.RWMol(molecule)
    for star, number in zip(stars, opposite):
        changed.GetAtomWithIdx(star.GetIdx()).SetAtomicNum(number)
    try:
        Chem.SanitizeMol(changed)
    except Chem.rdchem.MolSanitizeException as error:
        return np.zeros(200, np.float32), False, 'star_sub:' + str(error)
    smiles = Chem.MolToSmiles(changed, canonical=True)
    result = generator.process(smiles)
    if result is None or not result[0]:
        return np.zeros(200, np.float32), False, 'descriptor_calculation_failed'
    values = np.asarray(result[1:], dtype=np.float32)
    if values.shape != (200,):
        raise ValueError('MD200 descriptor schema/column count mismatch')
    if not np.isfinite(values).all():
        columns = [generator.columns[i][0] for i in np.flatnonzero(~np.isfinite(values))]
        # 2026-10-07用户授权修复：底层True只排除None，未检验NaN。
        # 已定位的partial-charge失败关闭整条MD，不修改任何描述符公式。
        known = {'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge'}
        if not set(columns) <= known:
            raise ValueError('unexpected nonfinite MD200 columns: ' + ','.join(columns))
        evidence = {generator.columns[i][0]: repr(float(values[i]))
                    for i in np.flatnonzero(~np.isfinite(values))}
        return np.zeros(200, np.float32), False, 'descriptor_nonfinite_partial_charge:' + json.dumps(evidence, sort_keys=True)
    return values, True, ''


@lru_cache(maxsize=1)
def worker_generator():
    return RDKit2DNormalized()


def calculate_chunk(rows):
    result = []
    generator = worker_generator()
    for index, row in rows:
        if row.get('pretrain_role') == 'excluded':
            value, available, reason = np.zeros(200, np.float32), False, 'excluded_from_pretraining'
        else:
            value, available, reason = descriptor(generator, row['normalized_smiles'])
        result.append((index, row, value, available, reason))
    return result


def calculated_rows(source, first, count, workers, row_field):
    def chunks():
        batch = []
        for index, line in enumerate(source, start=first):
            row = json.loads(line)
            if index >= count or row.get(row_field) != index:
                raise ValueError('direct source row mapping mismatch')
            batch.append((index, row))
            if len(batch) == 100:
                yield batch
                batch = []
        if batch:
            yield batch
    if workers == 1:
        for batch in chunks():
            yield from calculate_chunk(batch)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for batch in chunks():
            pending.append(executor.submit(calculate_chunk, batch))
            if len(pending) >= 2 * workers:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-partial', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    metadata = json.loads((args.source / 'direct_source.json').read_text())
    if metadata.get('schema') != 'mcl-ph-direct-source-v1':
        raise ValueError('direct source schema mismatch')
    generator = RDKit2DNormalized()
    columns = [name for name, _ in generator.columns]
    if len(columns) != 200 or len(set(columns)) != 200:
        raise ValueError('unknown MD200 descriptor column order')
    reuse_metadata = None
    if args.reuse_partial:
        reuse_metadata = json.loads((args.reuse_partial / 'metadata.json').read_text())
        if (metadata['kind'] != 'pi1m' or reuse_metadata.get('complete')
                or reuse_metadata.get('schema') != 'glt-fusion-md200-v1'
                or reuse_metadata.get('semantics') != 'RDKit2DNormalized200/star_sub'
                or Path(reuse_metadata['source']).resolve() != args.source.resolve()
                or reuse_metadata['columns'] != columns):
            raise ValueError('partial MD200 source/schema/columns mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metadata.json').write_text(json.dumps(dict(
        schema='glt-fusion-md200-v1', source=str(args.source.resolve()),
        semantics='RDKit2DNormalized200/star_sub', columns=columns,
        molecule_source='canonical RU from direct normalized_smiles; equivalent chemical structure to RU_base',
        complete=False, failure_policy='partial-charge NaN closes entire MD channel; other nonfinite stops',
        repair_authorization='user: 请修改当前错误并继续执行！ (2026-10-07)',
        reuse_partial=str(args.reuse_partial.resolve()) if args.reuse_partial else None), indent=2) + '\n')
    started = time.time()
    ledgers = [('records', metadata['source_rows'])] if metadata['kind'] == 'pi1m' else [
        (task, metadata['task_rows'][task]) for task in ('eea','egb','egc','ei','eps','nc','xc')]
    counts = Counter()
    row_field = 'source_index' if metadata['kind'] == 'pi1m' else 'original_row'
    for task, count in ledgers:
        values = np.lib.format.open_memmap(args.output / f'{task}_values.npy', mode='w+', dtype=np.float32, shape=(count,200))
        valid = np.lib.format.open_memmap(args.output / f'{task}_valid.npy', mode='w+', dtype=np.bool_, shape=(count,))
        with (args.source / f'{task}.jsonl').open() as source, (args.output / f'{task}_rows.jsonl').open('x') as target:
            completed = 0
            if reuse_metadata:
                previous_values = np.load(args.reuse_partial / f'{task}_values.npy', mmap_mode='r')
                previous_valid = np.load(args.reuse_partial / f'{task}_valid.npy', mmap_mode='r')
                if previous_values.shape != (count, 200) or previous_values.dtype != np.float32 or previous_valid.shape != (count,) or previous_valid.dtype != np.bool_:
                    raise ValueError('partial MD200 array schema mismatch')
                with (args.reuse_partial / f'{task}_rows.jsonl').open() as previous:
                    for index, line in enumerate(previous):
                        old = json.loads(line)
                        row = json.loads(next(source))
                        value = previous_values[index]
                        if (old['source_row'] != index or row.get('source_index') != index
                                or old['normalized_smiles'] != row['normalized_smiles']
                                or bool(old['valid']) != bool(previous_valid[index])
                                or not np.isfinite(value).all()
                                or (not old['valid'] and np.any(value != 0))):
                            raise ValueError('partial MD200 mapping/payload mismatch')
                        values[index], valid[index] = value, old['valid']
                        target.write(line)
                        counts['valid' if old['valid'] else old['reason']] += 1
                        completed += 1
                print(json.dumps(dict(task=task, reused_rows=completed)), flush=True)
            for index, row, value, available, reason in calculated_rows(source, completed, count, args.workers, row_field):
                values[index], valid[index] = value, available
                target.write(json.dumps(dict(source_row=index, normalized_smiles=row['normalized_smiles'], valid=available, reason=reason))+'\n')
                counts['valid' if available else reason] += 1
                completed = index + 1
                if (index + 1) % 1000 == 0:
                    values.flush(); valid.flush(); target.flush()
                    print(json.dumps(dict(task=task, rows=index+1, counts=dict(counts), seconds=time.time()-started)),flush=True)
            if completed != count:
                raise ValueError('direct source ledger row count mismatch')
        values.flush(); valid.flush()
    path = args.output / 'metadata.json'
    final = json.loads(path.read_text()); final.update(complete=True, counts=dict(counts), seconds=time.time()-started)
    path.write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps(final), flush=True)

if __name__ == '__main__':
    main()
