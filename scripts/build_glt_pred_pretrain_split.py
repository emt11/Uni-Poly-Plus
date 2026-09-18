#!/usr/bin/env python3
"""Build the immutable PI1M/downstream identity split for GLT-PRED S3b.

The artifact is only an ordered source-index view.  It never copies or edits
the frozen cohort, LMDB layers, static cache, or target cache.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.dataset.glt_dual_cache import load_dual_cohort, ordered_key_hash
from src.dataset.cache_lifecycle import json_hash


def _text_hash(values):
    digest = hashlib.sha256()
    for value in values:
        raw = str(value).encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'little'))
        digest.update(raw)
    return digest.hexdigest()


def _index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def _identity_digest(identity, seed):
    return hashlib.sha256(
        f'GLT-PRED-PVAL-V1|seed={int(seed)}|{identity}'.encode('utf-8')
    ).digest()


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     sort_keys=True, allow_nan=False) + '\n',
                          encoding='utf-8')


def build(pi1m_root, downstream_root, pi1m_cache_root, downstream_cache_root,
          output_root, seed=42):
    pi1m = load_dual_cohort(pi1m_root, pi1m_cache_root)
    downstream = load_dual_cohort(downstream_root, downstream_cache_root)
    # The downstream cohort is a row-level artifact; normalized_smiles is the
    # only identity used for exclusion, never labels, task or fold membership.
    downstream_identities = {str(row['normalized_smiles']) for row in downstream['records']}
    pi_records = pi1m['records']
    if len(pi_records) != 959588:
        raise ValueError(f'PI1M record count changed: {len(pi_records)}')
    identities = [str(row['normalized_smiles']) for row in pi_records]
    if len(set(identities)) != len(identities):
        raise ValueError('PI1M cohort is not unique by normalized identity')
    overlap = [identity in downstream_identities for identity in identities]
    eligible = [identity for identity, is_overlap in zip(identities, overlap) if not is_overlap]
    overlap_identities = {identity for identity, is_overlap in zip(identities, overlap) if is_overlap}
    if len(overlap_identities) != 229 or len(eligible) != 959359:
        raise ValueError('unexpected downstream overlap/eligible count')
    values = sorted(set(eligible))
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(values))
    n_train = round(0.95 * len(values))
    train_ids = [values[int(index)] for index in order[:n_train]]
    validation_ids = [values[int(index)] for index in order[n_train:]]
    if len(train_ids) != 911391 or len(validation_ids) != 47968:
        raise ValueError('unexpected train/validation identity counts')
    train_set, validation_set = set(train_ids), set(validation_ids)
    if train_set & validation_set or train_set | validation_set != set(eligible):
        raise ValueError('train/validation identity split is not disjoint/exhaustive')
    train_indices = np.asarray([i for i, identity in enumerate(identities)
                                if identity in train_set], dtype=np.int64)
    validation_indices = np.asarray([i for i, identity in enumerate(identities)
                                     if identity in validation_set], dtype=np.int64)
    fixed_pairs = sorted(
        (( _identity_digest(identity, seed), identity) for identity in validation_ids),
        key=lambda item: item[0])[:min(1024, len(validation_ids))]
    fixed_ids = [identity for _, identity in fixed_pairs]
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    fixed_indices = np.asarray([identity_to_index[identity] for identity in fixed_ids], dtype=np.int64)
    if len(set(fixed_indices.tolist())) != len(fixed_indices):
        raise ValueError('fixed validation selection contains duplicate source rows')

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / 'pretrain_split_v1.npz'
    np.savez(npz_path, train_source_indices=train_indices,
             validation_source_indices=validation_indices,
             fixed_validation_source_indices=fixed_indices)
    npz_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    train_keys = [bytes.fromhex(pi_records[int(i)]['sample_key']) for i in train_indices]
    validation_keys = [bytes.fromhex(pi_records[int(i)]['sample_key']) for i in validation_indices]
    fixed_keys = [bytes.fromhex(pi_records[int(i)]['sample_key']) for i in fixed_indices]
    split_payload = {
        'schema_version': 'glt-pred-pretrain-split-v1',
        'seed': int(seed),
        'pi1m_cohort_root': str(Path(pi1m_root).resolve()),
        'pi1m_manifest_hash': pi1m['manifest_hash'],
        'pi1m_ordered_sample_key_hash': pi1m['manifest']['ordered_sample_key_hash'],
        'pi1m_record_count': len(pi_records),
        'downstream_cohort_root': str(Path(downstream_root).resolve()),
        'downstream_manifest_hash': downstream['manifest_hash'],
        'downstream_unique_identity_count': len(downstream_identities),
        'excluded_overlap_identity_count': len(overlap_identities),
        'eligible_identity_count': len(eligible),
        'eligible_identity_sha256': _text_hash(sorted(eligible)),
        'train_identity_count': len(train_ids),
        'validation_identity_count': len(validation_ids),
        'train_identity_sha256': _text_hash(train_ids),
        'validation_identity_sha256': _text_hash(validation_ids),
        'train_source_index_sha256': _index_hash(train_indices),
        'validation_source_index_sha256': _index_hash(validation_indices),
        'train_ordered_sample_key_sha256': ordered_key_hash(train_keys),
        'validation_ordered_sample_key_sha256': ordered_key_hash(validation_keys),
        'fixed_validation_count': len(fixed_indices),
        'fixed_validation_source_index_sha256': _index_hash(fixed_indices),
        'fixed_validation_identity_sha256': _text_hash(fixed_ids),
        'fixed_validation_ordered_sample_key_sha256': ordered_key_hash(fixed_keys),
        'npz_sha256': npz_hash,
        'npz_file': npz_path.name,
        'fixed_validation_file': 'fixed_pval_v1.json',
    }
    split_path = output_root / 'pretrain_split_v1.json'
    _write_json(split_path, split_payload)
    fixed_payload = {
        'schema_version': 'glt-pred-fixed-pval-v1',
        'seed': int(seed),
        'split_artifact': str(split_path.resolve()),
        'split_artifact_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'source_indices': fixed_indices.tolist(),
        'sample_keys': [key.hex() for key in fixed_keys],
        'normalized_identities': fixed_ids,
        'source_index_sha256': _index_hash(fixed_indices),
        'identity_sha256': _text_hash(fixed_ids),
        'ordered_sample_key_sha256': ordered_key_hash(fixed_keys),
    }
    _write_json(output_root / 'fixed_pval_v1.json', fixed_payload)
    return split_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--downstream-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True,
                        help='explicit frozen PI1M main cache root used for validation')
    parser.add_argument('--downstream-cache-root', required=True,
                        help='explicit frozen downstream main cache root used for validation')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    payload = build(args.pi1m_cohort_root, args.downstream_cohort_root,
                    args.cache_root, args.downstream_cache_root,
                    args.output_root, args.seed)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
