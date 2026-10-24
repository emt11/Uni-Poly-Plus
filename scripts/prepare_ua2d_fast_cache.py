#!/usr/bin/env python3
"""One shared CPU-only materialization of the fixed training source."""
import argparse
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_SOURCE = _GENERATOR = _BEGUN = None


def worker_init():
    import torch
    from src.training.xc_kernel import generator
    global _GENERATOR
    torch.set_num_threads(1)
    _SOURCE._open_environment()
    _GENERATOR = generator()


def prepare_chunk(indices):
    import numpy as np
    from src.training.glt_dual_runtime import pretrain_source_fields
    from src.training.xc_kernel import fingerprint
    from src.dataset.ua2d_fast_pretrain import build_template, encode
    outputs = []
    for index in indices:
        top, tri, text, static, _, identity = pretrain_source_fields(_SOURCE, index, include_identity=True)
        entry = _SOURCE.entries[index]
        if entry['pretrain_role'] != 'train' or entry['normalized_smiles'] != str(top.normalized_canonical_smiles):
            raise ValueError('prepared record train/identity mismatch')
        template = build_template(top, tri, text, static, identity)
        key = _SOURCE.samples[index][0]
        template['sample_key'] = key
        with _BEGUN.get_lock():_BEGUN.value += 1
        fp, _ = fingerprint(template['identity'], _GENERATOR)
        bits = np.asarray(list(fp), dtype=np.uint8)
        if not 0 < int(bits.sum()) < 2048:raise ValueError('invalid clean FP target')
        template['fp_bits'] = np.packbits(bits, bitorder='little')
        outputs.append((key, encode(template)))
    return outputs


def main():
    import lmdb
    import numpy as np
    import torch
    from rdkit import rdBase
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.dataset.ua2d_fast_pretrain import CACHE_SCHEMA
    from src.training.glt_dual_runtime import load_sample_index_artifact, require_tmux
    from src.training.ua2d_pretrain_fp import SOURCE, INDEX, FP_SPEC, write
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    require_tmux()
    if not 1 <= args.workers <= 16:raise ValueError('CPU worker bound')
    torch.set_num_threads(1)
    global _SOURCE, _BEGUN
    context = mp.get_context('fork')
    _BEGUN = context.Value('q', 0)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    meta = dict(schema=CACHE_SCHEMA, status='RUNNING', source=str(SOURCE), index=str(INDEX),
                train_rows=911391, generated=0, completed=0, workers=args.workers, spec=FP_SPEC,
                rdkit=rdBase.rdkitVersion, preprocessing_seconds=0., frozen_inputs=True,
                objective_inputs='FP labels only consumed by B_FP; no FP input or 3D-to-2D fusion')
    write(args.output / 'metadata.json', meta)
    environment = None
    try:
        indices = load_sample_index_artifact(INDEX, 'train')['indices']
        if len(indices) != 911391:raise ValueError('fixed training subset changed')
        _SOURCE = DirectMCLSource(SOURCE, task=None, selected_indices=None, split_manifest=None,
                                  verify_key_digest=False, readahead=True)
        keys = np.empty(len(indices), dtype='V32')
        unique = {}
        for logical, source_row in enumerate(indices):
            row = int(source_row); entry = _SOURCE.entries[row]
            if entry['pretrain_role'] != 'train':raise ValueError('nontrain row in train subset')
            key = _SOURCE.samples[row][0]; keys[logical] = key
            if key in unique:
                first = _SOURCE.entries[unique[key]]
                if entry['normalized_smiles'] != first['normalized_smiles']:
                    raise ValueError('same template key has different source identity')
            else:unique[key] = row
        np.save(args.output / 'keys.npy', keys, allow_pickle=False)
        np.save(args.output / 'source_rows.npy', indices, allow_pickle=False)
        meta['unique_templates'] = len(unique)
        # Fork only after closing the parent's LMDB handle. Each worker reopens
        # its own read handle, retaining the validated immutable row metadata.
        _SOURCE.close()
        rows = list(unique.values())
        chunks = [rows[i:i+64] for i in range(0, len(rows), 64)]
        environment = lmdb.open(str(args.output / 'templates.lmdb'), map_size=1024**4,
                                max_readers=128, subdir=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=context, initializer=worker_init) as pool:
            # Bounded submission keeps cached tensors/payloads out of the parent
            # until they can be written; no million-future queue.
            pending = set(); iterator = iter(chunks)
            for _ in range(args.workers * 2):
                chunk = next(iterator, None)
                if chunk is not None:pending.add(pool.submit(prepare_chunk, chunk))
            while pending:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    values = future.result()
                    with environment.begin(write=True) as txn:
                        for key, payload in values:
                            if not txn.put(key, payload, overwrite=False):raise ValueError('duplicate template writer')
                    meta['completed'] += len(values)
                    chunk = next(iterator, None)
                    if chunk is not None:pending.add(pool.submit(prepare_chunk, chunk))
                meta.update(generated=int(_BEGUN.value), preprocessing_seconds=time.monotonic()-start)
                write(args.output / 'metadata.json', meta)
                if meta['completed'] % 4096 < 64 * len(done):
                    print(__import__('json').dumps(meta), flush=True)
        if meta['completed'] != len(unique) or _BEGUN.value != len(unique):raise ValueError('cache coverage incomplete')
        environment.sync()
        meta.update(status='COMPLETE', generated=int(_BEGUN.value), preprocessing_seconds=time.monotonic()-start)
        write(args.output / 'metadata.json', meta)
        print(__import__('json').dumps(meta), flush=True)
    except BaseException:
        meta.update(status='FAILED', generated=int(_BEGUN.value), error=traceback.format_exc(),
                    preprocessing_seconds=time.monotonic()-start)
        write(args.output / 'metadata.json', meta)
        raise
    finally:
        if environment is not None:environment.close()
        if _SOURCE is not None:_SOURCE.close()


if __name__ == '__main__':main()
