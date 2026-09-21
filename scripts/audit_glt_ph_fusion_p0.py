#!/usr/bin/env python3
"""P0 audit for GLT-PH-END2END-20260920-01 / r1 (family B).

Read-only.  Verifies the frozen sources, the split and identity records, the
real all-atom Trimer interface that family B consumes, the three spatial graphs'
size and cost, and the PH/STAT inputs.  Samples at most 256 P_train structures
and 16 downstream structures per task, all drawn from the fixed train or
validation indices; no outer-test label is read and nothing is written except
the audit report and the explicitly SMOKE_ONLY standardization statistics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.dataset.glt_galformer_ph import PHBettiReader
from src.dataset.glt_ph import PH_BINS, PH_CHANNELS, PH_SCHEMA, radius_grid
from src.dataset.glt_ph_fusion_inputs import (ARMS, SPATIAL_RADII, bond_row_alignment,
                                              physical_bond_table, prepare_fusion_sample,
                                              save_ph_stats, spatial_edges)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           load_sample_index_artifact, open_source,
                                           require_tmux, sha256_file, write_json)

DEFAULT_SIDECAR = 'results/glt_galph_20260920/p0/ph_sidecar_betti_v2'
DEFAULT_DOWNSTREAM_SIDECAR = 'results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream'
DEFAULT_CONST_PROFILE = (DEFAULT_DOWNSTREAM_SIDECAR + '/p_train_mean_profile.npy')
EXPECTED_SPLIT_SCHEMA = 'glt-pred-pretrain-split-v1'
MEMORY_BYTES_PER_EDGE = {'edge_index': 2 * 8, 'distance': 4, 'bonded': 1, 'scale': 1}


def _key_report(keys):
    lengths = {len(key) for key in keys}
    nul = sum(1 for key in keys if b'\x00' in key)
    return {'count': len(keys), 'lengths': sorted(lengths), 'keys_with_nul_byte': nul,
            'unique': len(set(keys))}


def _directed_edge_set(edge_index):
    return {(int(a), int(b)) for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist())}


def audit_pretrain(args, report):
    reader = PHBettiReader(args.ph_sidecar_root)
    sidecar_meta = json.loads((Path(args.ph_sidecar_root) / 'metadata.json').read_text())
    report['ph_sidecar'] = {
        'root': str(Path(args.ph_sidecar_root).resolve()), 'metadata': sidecar_meta,
        'rows': len(reader), 'schema_ok': sidecar_meta.get('schema_version') == PH_SCHEMA,
        'radius_endpoints': sidecar_meta.get('radius'),
        'channels': int(reader.profiles.shape[1]), 'bins': int(reader.profiles.shape[2])}
    sample_index = load_sample_index_artifact(args.sample_index_artifact,
                                              args.sample_index_split)
    if sample_index is None:
        raise SystemExit('the audit requires the fixed P_train sample-index artifact')
    split_payload = sample_index['payload']
    report['split'] = {
        'artifact': sample_index['path'], 'sha256': sample_index['sha256'],
        'split': sample_index['split'], 'schema_version': sample_index['schema_version'],
        'indices': int(sample_index['indices'].size),
        'train_identity_count': split_payload.get('train_identity_count'),
        'validation_identity_count': split_payload.get('validation_identity_count'),
        'fixed_validation_count': split_payload.get('fixed_validation_count'),
        'pi1m_record_count': split_payload.get('pi1m_record_count'),
        'pi1m_manifest_hash': split_payload.get('pi1m_manifest_hash'),
        'pi1m_ordered_sample_key_hash': split_payload.get('pi1m_ordered_sample_key_hash'),
        'excluded_overlap_identity_count': split_payload.get('excluded_overlap_identity_count'),
        'downstream_unique_identity_count': split_payload.get('downstream_unique_identity_count')}
    const_profile = np.load(args.const_profile)
    report['const_profile'] = {'path': str(Path(args.const_profile).resolve()),
                               'sha256': sha256_file(args.const_profile),
                               'shape': list(const_profile.shape),
                               'finite': bool(np.isfinite(const_profile).all())}
    if tuple(const_profile.shape) != (PH_CHANNELS, PH_BINS):
        raise SystemExit('the constant P_train mean profile must be [3,32]')

    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    try:
        report['cohort'] = {
            'root': str(Path(args.cohort_root).resolve()),
            'manifest_hash': source.cohort['manifest_hash'],
            'main_bundle_hash': source.bundle.bundle_hash,
            'records': len(source),
            'static_manifest_hash': (source.static_cache.manifest_hash
                                     if source.static_cache is not None else None),
            'expected_pi1m_manifest_hash': split_payload.get('pi1m_manifest_hash'),
            'expected_pi1m_record_count': split_payload.get('pi1m_record_count'),
            'cohort_manifest_matches_split': (source.cohort['manifest_hash']
                                              == split_payload.get('pi1m_manifest_hash'))}
        if len(source) != int(split_payload.get('pi1m_record_count', len(source))):
            raise SystemExit('cohort record count differs from the split artifact')
        # P_train membership is the split's own source-index array; the frozen
        # source is addressed by those base indices, exactly as the runner does
        # after it wraps the source with the same array.
        report['cohort']['base_records'] = len(source)
        report['cohort']['p_train_records'] = int(sample_index['indices'].size)
        report['cohort']['sidecar_covers_p_train'] = (
            len(reader) == int(sample_index['indices'].size))
        if not report['cohort']['sidecar_covers_p_train']:
            raise SystemExit('the PH sidecar does not cover the whole P_train split')
        indices = [int(value) for value in sample_index['indices'][:args.pretrain_samples]]
        keys = [source.samples[index][0] for index in indices]
        report['keys'] = _key_report(keys)
        stats = {name: [] for name in ('atoms_all', 'atoms_heavy', 'bonds', 'edges_2.5',
                                       'edges_4.0', 'edges_6.0')}
        profiles, geometry_valid, alignment_failures = [], 0, {}
        nesting_failures, symmetry_failures, self_failures, ph_invalid = 0, 0, 0, 0
        build_seconds, stat_seconds = [], []
        spatial_note, sample_failures = {}, []
        for position in indices:
            key = source.samples[position][0]
            started = time.perf_counter()
            try:
                data, _ = prepare_fusion_sample(
                    *source[position], static=source.static_for(position),
                    seed=0, key=key.hex(), position=position, ph_reader=reader,
                    arm='PH', const_profile=const_profile)
            except Exception as error:
                sample_failures.append({'position': int(position),
                                        'key': key.hex()[:16],
                                        'error': f'{type(error).__name__}: {error}'[:240]})
                continue
            build_seconds.append(time.perf_counter() - started)
            stats['atoms_all'].append(int(data.atom_count))
            stats['bonds'].append(int(data.bond_count))
            table = physical_bond_table(source[position][1], source[position][0])
            stats['atoms_heavy'].append(int(table['projection']['heavy_indices'].numel()))
            if bool(data.geometry_valid):
                geometry_valid += 1
                issues = bond_row_alignment(table, data.bond_z_a, data.bond_z_b,
                                            data.bond_distance)
                if issues:
                    alignment_failures[key.hex()[:16]] = issues
                graphs = spatial_edges(table['projection']['positions'])
                for name, graph in zip(('edges_2.5', 'edges_4.0', 'edges_6.0'), graphs):
                    stats[name].append(int(graph.size(1)))
                directed = [_directed_edge_set(graph) for graph in graphs]
                if not (directed[0] <= directed[1] <= directed[2]):
                    nesting_failures += 1
                for graph, edges in zip(graphs, directed):
                    if any(a == b for a, b in edges):
                        self_failures += 1
                    if any((b, a) not in edges for a, b in edges):
                        symmetry_failures += 1
                observed = _directed_edge_set(data.spatial_edge_index)
                if observed != set().union(*directed):
                    spatial_note['edge_mismatch'] = spatial_note.get('edge_mismatch', 0) + 1
            else:
                spatial_note.setdefault('geometry_invalid', 0)
                spatial_note['geometry_invalid'] += 1
            profile = np.asarray(data.ph_profile)
            valid = bool(data.ph_valid)
            profiles.append((profile, valid))
            ph_invalid += int(not valid)
            started = time.perf_counter()
            from src.dataset.glt_ph_fusion_inputs import stat_profile
            stat_profile(table['projection']['positions'], table['projection']['numbers'])
            stat_seconds.append(time.perf_counter() - started)
        report['pretrain_samples'] = {
            'sampled': len(indices), 'built': len(build_seconds),
            'geometry_valid': geometry_valid, 'ph_invalid': ph_invalid,
            'alignment_failures': alignment_failures,
            'sample_failures': sample_failures[:20],
            'sample_failure_count': len(sample_failures),
            'nesting_failures': nesting_failures, 'symmetry_failures': symmetry_failures,
            'self_loop_failures': self_failures, 'spatial_notes': spatial_note,
            'sizes': {name: _summarize(values) for name, values in stats.items()},
            'build_seconds': _summarize(build_seconds),
            'stat_seconds': _summarize(stat_seconds)}
        if (alignment_failures or nesting_failures or symmetry_failures or self_failures
                or sample_failures):
            report['status'] = 'BLOCKED'
            report.setdefault('blocked_by', []).append('family-B spatial interface checks failed')
        _write_smoke_stats(args, profiles, report)
    finally:
        source.close()


def _summarize(values):
    if not values:
        return {'count': 0}
    array = np.asarray(values, dtype=np.float64)
    return {'count': int(array.size), 'mean': float(array.mean()),
            'median': float(np.median(array)), 'min': float(array.min()),
            'max': float(array.max()),
            'p95': float(np.percentile(array, 95))}


def _write_smoke_stats(args, profiles, report):
    valid = [profile for profile, ok in profiles if ok]
    if not valid:
        report['smoke_stats'] = {'written': False, 'reason': 'no valid PH profile'}
        return
    stacked = np.stack(valid, 0)
    mean = stacked.mean(0).astype(np.float32)
    std = stacked.std(0).astype(np.float32)
    std = np.maximum(std, 1e-3).astype(np.float32)
    save_ph_stats(args.stats_output, mean, std, source='p0_audit_256_p_train',
                  scope='SMOKE_ONLY_256_P_TRAIN_NOT_FOR_P2', samples=len(valid))
    report['smoke_stats'] = {
        'written': True, 'path': str(Path(args.stats_output).resolve()),
        'sha256': sha256_file(args.stats_output), 'samples': len(valid),
        'scope': 'SMOKE_ONLY_256_P_TRAIN_NOT_FOR_P2',
        'mean_range': [float(mean.min()), float(mean.max())],
        'std_range': [float(std.min()), float(std.max())]}


def audit_downstream(args, report):
    reader = PHBettiReader(args.const_profile.rsplit('/', 1)[0])
    entry = {}
    for task in args.tasks:
        manifest_path = Path(args.downstream_split_root) / f'{task}.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        fold = [item for item in manifest['folds'] if int(item['fold']) == 0][0]
        train = [int(value) for value in fold['train_indices']]
        validation = [int(value) for value in fold['validation_indices']]
        test = [int(value) for value in fold['test_indices']]
        picked = train[:args.downstream_samples // 2] + validation[:args.downstream_samples // 2]
        source, _ = open_source(args.downstream_cohort_root, args.downstream_cache_root,
                                task=task, dual_static_root=args.downstream_dual_static_root)
        try:
            rows = []
            invalid = missing = 0
            edges = {f'edges_{radius}': [] for radius in SPATIAL_RADII}
            for index in picked:
                key = source.samples[index][0]
                # Label-free: only the frozen structure and its geometry flag are read.
                built = source[index]
                table = physical_bond_table(built[1], built[0])
                geometry_valid = bool(getattr(built[1], 'trimer_geometry_valid', False))
                row = {'index': index, 'key_length': len(key), 'geometry_valid': geometry_valid,
                       'atoms_all': int(table['projection']['positions'].size(0)),
                       'bonds': int(table['index'].size(1))}
                if geometry_valid:
                    graph = spatial_edges(table['projection']['positions'])[-1]
                    row['edges_6.0'] = int(graph.size(1))
                    edges['edges_6.0'].append(int(graph.size(1)))
                profile, ok = reader.get(-1, key.hex())
                row['ph_valid'] = bool(ok)
                invalid += int(not ok)
                rows.append(row)
            entry[task] = {
                'manifest': str(manifest_path.resolve()),
                'manifest_sha256': sha256_file(manifest_path),
                'splits': {'train': len(train), 'validation': len(validation), 'test': len(test)},
                'sampled': len(rows), 'ph_invalid': invalid, 'ph_missing': missing,
                'samples': rows, 'edge_stats': {name: _summarize(values)
                                                for name, values in edges.items()}}
        finally:
            source.close()
    report['downstream'] = entry
    report['downstream_sidecar'] = {
        'root': str(Path(args.downstream_sidecar).resolve()),
        'rows': len(PHBettiReader(args.downstream_sidecar))}


def audit_identity(args, report):
    identity = json.loads(Path(args.identity).read_text(encoding='utf-8'))
    checkpoint = Path(identity['checkpoint'])
    observed = sha256_file(checkpoint) if checkpoint.is_file() else None
    common = Path(identity['pretrain']['common_init_artifact'])
    common_observed = sha256_file(common) if common.is_file() else None
    report['current'] = {
        'identity_record': str(Path(args.identity).resolve()),
        'record': identity.get('record'), 'checkpoint': str(checkpoint),
        'sha256_expected': identity.get('sha256'), 'sha256_observed': observed,
        'matches': observed == identity.get('sha256'),
        'step': identity.get('step'), 'architecture': identity.get('architecture'),
        'summary_mode': identity.get('summary_mode'), 'ph_mode': identity.get('ph_mode'),
        'ph_encoder_version': identity.get('ph_encoder_version'),
        'common_init_artifact': str(common),
        'common_init_sha256_expected': identity['pretrain'].get('common_init_sha256'),
        'common_init_sha256_observed': common_observed,
        'common_init_matches': common_observed == identity['pretrain'].get('common_init_sha256')}
    payload = torch.load(common, map_location='cpu', weights_only=False)
    keys = sorted(payload['common_state_dict'])
    prefixes = {}
    for name in keys:
        prefixes[name.split('.')[0]] = prefixes.get(name.split('.')[0], 0) + 1
    report['common_init'] = {'path': str(common), 'sha256': common_observed,
                             'tensors': len(keys), 'prefixes': prefixes,
                             'schema_version': payload.get('schema_version'),
                             'seed': payload.get('seed'),
                             'has_cls_block': any(name.startswith('cls') for name in keys),
                             'has_ph_block': any(name.startswith(('ph_', 'alpha_ph'))
                                                 for name in keys)}
    if not report['current']['matches'] or not report['current']['common_init_matches']:
        report['status'] = 'BLOCKED'
        report.setdefault('blocked_by', []).append('CURRENT or common-init identity mismatch')


def audit_cost(args, report, samples):
    projected = {}
    total = int(report['split'].get('pi1m_record_count') or 0)
    for name, values in samples.items():
        if name.startswith('edges_'):
            mean = values['mean']
            bytes_per_sample = sum(MEMORY_BYTES_PER_EDGE[key] * mean
                                   for key in MEMORY_BYTES_PER_EDGE)
            projected[name] = {
                'directed_edges_mean': mean, 'edges_p95': values['p95'],
                'input_bytes_per_sample': bytes_per_sample,
                'projected_full_pretrain_bytes': bytes_per_sample * total,
                'projected_full_pretrain_gib': bytes_per_sample * total / 2 ** 30,
                'rbf_working_set_bytes_per_edge_fp32': 33 * 4}
    report['cost'] = {
        'p_train_structures': total,
        'spatial_radii': list(SPATIAL_RADII),
        'rbf_bins': 32, 'per_scale_edge_features': ['U_target', 'U_source', 'RBF32', 'bonded_bit',
                                                    'scale_embedding_128'],
        'message_input_dim': 417,
        'projected': projected,
        'note': ('edge tensors are per microbatch; the message MLP consumes edges in exact '
                 'chunks (no edge dropped, no radius changed)'),
        'stat_build_seconds_per_sample': report['pretrain_samples']['stat_seconds'],
        'stat_full_pretrain_single_process_hours': (
            report['pretrain_samples']['stat_seconds'].get('mean', 0.0) * total / 3600.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root',
                        default='data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1')
    parser.add_argument('--cache-root', default='data/processed/mips_trimer_scage')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    parser.add_argument('--split-artifact',
                        default='results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json')
    parser.add_argument('--sample-index-artifact')
    parser.add_argument('--sample-index-split', default='train')
    parser.add_argument('--ph-sidecar-root', default=DEFAULT_SIDECAR)
    parser.add_argument('--const-profile', default=DEFAULT_CONST_PROFILE)
    parser.add_argument('--downstream-cohort-root',
                        default='data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1')
    parser.add_argument('--downstream-cache-root',
                        default='data/processed/mips_trimer_scage_downstream')
    parser.add_argument('--downstream-dual-static-root',
                        default='data/processed/glt_dual_v2/downstream/dual_static_v1')
    parser.add_argument('--downstream-split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--downstream-sidecar', default=DEFAULT_DOWNSTREAM_SIDECAR)
    parser.add_argument('--identity', default='configs/mts/glt_galph_c1_repair_5k_identity.json')
    parser.add_argument('--tasks', nargs='*', default=['xc', 'eps', 'eat'])
    parser.add_argument('--pretrain-samples', type=int, default=256)
    parser.add_argument('--downstream-samples', type=int, default=16)
    parser.add_argument('--output',
                        default='results/glt_ph_end2end_20260920/p0/p0_audit.json')
    parser.add_argument('--stats-output',
                        default='results/glt_ph_end2end_20260920/p0/ph_stats_smoke_256.npz')
    parser.add_argument('--skip-downstream', action='store_true')
    args = parser.parse_args()
    require_tmux()
    args.sample_index_artifact = args.sample_index_artifact or args.split_artifact
    started = time.perf_counter()
    report = {'plan': 'GLT-PH-END2END-20260920-01 / r1', 'stage': 'P0',
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'status': 'PASS', 'radius_grid': radius_grid().tolist(),
              'radius_grid_sha256': hashlib.sha256(
                  radius_grid().astype('<f8').tobytes()).hexdigest()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    audit_identity(args, report)
    audit_pretrain(args, report)
    if not args.skip_downstream:
        audit_downstream(args, report)
    audit_cost(args, report, report['pretrain_samples']['sizes'])
    report['wall_seconds'] = time.perf_counter() - started
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'blocked_by': report.get('blocked_by', []),
                      'pretrain_sampled': report['pretrain_samples']['sampled'],
                      'geometry_valid': report['pretrain_samples']['geometry_valid'],
                      'sizes': report['pretrain_samples']['sizes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
