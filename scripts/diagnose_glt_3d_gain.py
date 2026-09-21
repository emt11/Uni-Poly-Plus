#!/usr/bin/env python3
"""D1 zero-update diagnosis for GLT-3D-GAIN-20260921-01.

Three modes, all with **no neural-network parameter update**:

``encode``        Frozen z2/z3/z23 extraction for the six development units
                  (xc/eps/eat x fold0/1) under one deployment checkpoint.
``interventions`` Bounded input/representation interventions on at most 256
                  XC/fold0 train samples.  Representation layers come from the
                  frozen B_FP deployment package; the scalar prediction comes
                  from that unit's trained S5 best checkpoint (never from the
                  randomly initialised deployment head).
``probes``        CPU-only frozen linear probes: four 5-alpha Ridge families
                  plus the cross-fitted residual probe.

Diagnostics are FP32.  DataLoader workers are 0; batches are built by hand so
the sample order is exactly the requested order.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_dual import build_dual_glt_model, triplet_type
from src.modules.glt_dual_pretrain import load_deployment
from src.training.glt_dual_runtime import CleanLabeledDataset, open_source

TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PERMUTATION_SEED = 20260921
REPEATS = 3
LAYERS = ('token', 'graph2d', 'graph3d', 'fused', 'prediction')
# ``triplet_type`` values live below 25858 (see PathAngleBias's virtual
# category), so a pair key of left*TRIPLET_VOCAB+right stays in int64 range.
TRIPLET_VOCAB = 25859


def log(message):
    print(message, flush=True)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


# --------------------------------------------------------------------------
# Model / representation helpers
# --------------------------------------------------------------------------

def split_fuse(model, encoded):
    """Reproduce DualGLTModel.fuse() while exposing z2 and z3 separately."""
    valid = encoded['geometry_valid']
    z2 = model.norm2(encoded['graph_2d'])
    z3 = model.norm3(encoded['graph_3d'])
    z3 = torch.where(valid.unsqueeze(-1), z3, torch.zeros_like(z3))
    return z2, z3


def check_fuse_equivalence(model, encoded):
    z2, z3 = split_fuse(model, encoded)
    reference = model.fuse(encoded)
    rebuilt = torch.cat([z2, z3], -1)
    if not torch.allclose(rebuilt, reference, rtol=1e-5, atol=1e-5):
        raise ValueError('split_fuse does not reproduce DualGLTModel.fuse')
    return float((rebuilt - reference).abs().max())


def build_model(fusion_mode, checkpoint, *, device):
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torsion = bool(package.get('torsion_modules', False))
    model = build_dual_glt_model(fusion_mode, torsion=torsion)
    if 'state_dict' in package and 'architecture' in package and package.get('step') is None:
        if package.get('fusion_mode') != model.fusion_mode:
            raise ValueError('checkpoint fusion mode mismatch')
        model.load_state_dict(package['state_dict'], strict=True)
        meta = dict(kind='finetuned', architecture=package.get('architecture'),
                    fusion_mode=package.get('fusion_mode'), task=package.get('task'),
                    fold=package.get('fold'), protocol=package.get('protocol'),
                    best_epoch=package.get('best_epoch'),
                    best_validation_r2=package.get('best_validation_r2'),
                    scaler_mean=package.get('scaler_mean'),
                    scaler_scale=package.get('scaler_scale'))
    else:
        load_deployment(model, package, expected_step=5000)
        meta = dict(kind='deployment', architecture=package.get('architecture'),
                    fusion_mode=package.get('fusion_mode'), step=package.get('step'))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, meta


def open_task(args, task):
    source, frame = open_source(args.cohort_root, args.cache_root, task=task,
                                dual_static_root=args.dual_static_root)
    targets = frame['label'].to_numpy(dtype=np.float64)
    dataset = CleanLabeledDataset(
        source, targets,
        cache_capacity_bytes=int(args.clean_cache_gib * (1024 ** 3)),
    )
    manifest = json.loads((Path(args.split_root) / f'{task}.json').read_text(encoding='utf-8'))
    return source, dataset, manifest


def fold_indices(manifest, fold):
    entry = next(item for item in manifest['folds'] if int(item['fold']) == int(fold))
    return ([int(v) for v in entry['train_indices']],
            [int(v) for v in entry['validation_indices']])


def per_graph_mean(states, index, count):
    sums = states.new_zeros((count, states.size(-1)))
    sums.index_add_(0, index, states)
    counts = torch.bincount(index, minlength=count).clamp_min(1)
    return sums / counts.unsqueeze(-1)


def batch_keys(dataset, indices):
    return np.stack([np.frombuffer(dataset.source.samples[int(index)][0], dtype=np.uint8)
                     for index in indices])


def encode_batches(model, dataset, indices, device, batch_size):
    """Encode dataset indices in order; every returned row is per-graph."""
    out = {key: [] for key in ('z2', 'z3', 'graph_2d', 'graph_3d', 'geometry_valid',
                               'center_bond_count', 'token_norm_mean')}
    for start in range(0, len(indices), batch_size):
        chunk = [int(value) for value in indices[start:start + batch_size]]
        samples = [dataset[index] for index in chunk]
        batch = dual_glt_collate(samples).to(device)
        with torch.no_grad():
            encoded = model.encode(batch)
            z2, z3 = split_fuse(model, encoded)
        graphs = int(batch.graph_available.numel())
        center = batch.bond_center.bool()
        counts = torch.bincount(batch.bond_batch[center], minlength=graphs)
        token_norm = encoded['bond_states'].float().norm(dim=-1)
        token_mean = per_graph_mean(token_norm.unsqueeze(-1), batch.bond_batch, graphs).squeeze(-1)
        out['z2'].append(z2.float().cpu().numpy())
        out['z3'].append(z3.float().cpu().numpy())
        out['graph_2d'].append(encoded['graph_2d'].float().cpu().numpy())
        out['graph_3d'].append(encoded['graph_3d'].float().cpu().numpy())
        out['geometry_valid'].append(encoded['geometry_valid'].bool().cpu().numpy())
        out['center_bond_count'].append(counts.cpu().numpy().astype(np.int32))
        out['token_norm_mean'].append(token_mean.cpu().numpy().astype(np.float32))
    for key in ('z2', 'z3', 'graph_2d', 'graph_3d', 'token_norm_mean'):
        out[key] = np.concatenate(out[key], axis=0)
    for key in ('geometry_valid', 'center_bond_count'):
        out[key] = np.concatenate(out[key], axis=0)
    out['sample_key'] = batch_keys(dataset, indices)
    out['label'] = np.asarray([float(dataset.targets[int(index)]) for index in indices],
                              dtype=np.float64)
    return out


def run_encode(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_unit = args.checkpoint_kind == 'finetuned'
    model = meta = None
    dtypes = None
    if per_unit:
        if not args.finetuned_root:
            raise ValueError('finetuned encoding requires --finetuned-root')
    else:
        model, meta = build_model('concat', args.checkpoint, device=device)
        log(f'[encode] checkpoint kind={meta["kind"]} arch={meta["architecture"]} device={device}')
        sanity_source, sanity_dataset, sanity_manifest = open_task(args, 'eat')
        try:
            train, _ = fold_indices(sanity_manifest, 0)
            batch = dual_glt_collate([sanity_dataset[i] for i in train[:3]]).to(device)
            with torch.no_grad():
                deviation = check_fuse_equivalence(model, model.encode(batch))
            log(f'[encode] split_fuse max|delta| vs fuse(): {deviation:.3e}')
            dtypes = {'bond_distance': str(batch.bond_distance.dtype),
                      'line_angle': str(batch.line_angle.dtype),
                      'graph_2d': str(model.encode(batch)['graph_2d'].dtype)}
        finally:
            sanity_source.close()

    tag = 'finetuned' if per_unit else 'deploy'
    output_root = Path(args.output) / f'feat_{tag}'
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {'checkpoint_kind': tag, 'dtypes': dtypes, 'units': {}}
    for task in TASKS:
        source, dataset, manifest = open_task(args, task)
        try:
            for fold in FOLDS:
                unit_model, unit_meta = model, meta
                if per_unit:
                    unit_path = (Path(args.finetuned_root) / task / f'fold{fold}'
                                 / task / f'fold{fold}' / 'best.pt')
                    unit_model, unit_meta = build_model('concat', str(unit_path), device=device)
                    if unit_meta['task'] != task or int(unit_meta['fold']) != fold:
                        raise ValueError(f'finetuned checkpoint identity mismatch: {unit_path}')
                entry = {'checkpoint': {k: v for k, v in unit_meta.items()
                                        if k not in ('scaler_mean', 'scaler_scale')}}
                train, validation = fold_indices(manifest, fold)
                for split_name, indices in (('train', train), ('validation', validation)):
                    encoded = encode_batches(unit_model, dataset, indices, device, args.encode_batch)
                    path = output_root / f'{task}_fold{fold}_{split_name}.npz'
                    np.savez_compressed(path, split=np.array(split_name),
                                        row_index=np.asarray(indices, dtype=np.int64), **encoded)
                    entry[split_name] = {
                        'rows': int(len(indices)),
                        'geometry_valid': int(encoded['geometry_valid'].sum()),
                        'center_bond_zero': int((encoded['center_bond_count'] == 0).sum()),
                        'z2_mean_norm': float(np.linalg.norm(encoded['z2'], axis=1).mean()),
                        'z3_mean_norm': float(np.linalg.norm(encoded['z3'], axis=1).mean()),
                        'z3_exact_zero_rows': int((np.abs(encoded['z3']).sum(1) == 0).sum()),
                        'path': str(path),
                    }
                    record = entry[split_name]
                    log(f'[encode:{tag}] {task}/fold{fold}/{split_name}: rows={record["rows"]} '
                        f'valid={record["geometry_valid"]} center0={record["center_bond_zero"]} '
                        f'z3_zero={record["z3_exact_zero_rows"]}')
                summary['units'][f'{task}/fold{fold}'] = entry
                if per_unit:
                    del unit_model
        finally:
            source.close()
    if per_unit:
        summary['per_unit_root'] = str(Path(args.finetuned_root).resolve())
    write_json(Path(args.output) / f'encode_{tag}.json', summary)
    log(f'[encode:{tag}] PASS')


# --------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------

def deterministic_derangement(size, seed=PERMUTATION_SEED, attempts=200):
    """Fixed, seed-derived permutation with no fixed point."""
    if size < 2:
        raise ValueError('a derangement needs at least two samples')
    generator = np.random.default_rng(seed)
    for _ in range(attempts):
        candidate = generator.permutation(size)
        if not np.any(candidate == np.arange(size)):
            return candidate
    raise RuntimeError('failed to draw a derangement')


def _relative_stats(changed, original):
    changed = np.asarray(changed, dtype=np.float64).reshape(len(changed), -1)
    original = np.asarray(original, dtype=np.float64).reshape(len(original), -1)
    difference = np.linalg.norm(changed - original, axis=1)
    denominator = np.maximum(np.linalg.norm(original, axis=1), 1e-8)
    return {'relative': _stats(difference / denominator), 'raw': _stats(difference)}


def _stats(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {'count': 0}
    return {'count': int(array.size), 'mean': float(array.mean()),
            'median': float(np.median(array)), 'max': float(array.max()),
            'min': float(array.min())}


def batch_token_type_keys(batch):
    """Per-relation, per-hop triplet-type token index table.

    ``triplet_type`` codes live in a fixed 25859-entry space (see
    ``PathAngleBias``), so the pair key fits comfortably in int64.
    """
    types = triplet_type(batch.bond_z_a, batch.bond_z_b, batch.bond_type).long()
    path = batch.line_path.long()
    filler = torch.full_like(path, TRIPLET_VOCAB - 1)
    if types.numel() == 0:
        return filler
    safe = path.clamp_min(0).clamp_max(types.numel() - 1)
    return torch.where(path >= 0, types[safe], filler)


def pair_keys(path_types, hop):
    left = path_types[:, hop]
    right = path_types[:, hop + 1]
    return (left * TRIPLET_VOCAB + right).numpy().astype(np.int64)


def collect_geometry_statistics(dataset, train_indices, batch_size, min_group):
    """Distance-by-bond-type and angle-by-type-pair means on the given pool."""
    distance_sums, distance_counts = {}, {}
    angle_sums, angle_counts = {}, {}
    global_distance = [0.0, 0]
    global_angle = [0.0, 0]

    for start in range(0, len(train_indices), batch_size):
        chunk = [int(value) for value in train_indices[start:start + batch_size]]
        batch = dual_glt_collate([dataset[index] for index in chunk])
        distance = batch.bond_distance.double().numpy()
        bond_type = batch.bond_type.long().numpy()
        global_distance[0] += float(distance.sum())
        global_distance[1] += int(distance.size)
        for code in np.unique(bond_type):
            selected = distance[bond_type == code]
            distance_sums[int(code)] = distance_sums.get(int(code), 0.0) + float(selected.sum())
            distance_counts[int(code)] = distance_counts.get(int(code), 0) + int(selected.size)

        path_types = batch_token_type_keys(batch)
        mask = batch.line_mask.bool().numpy()
        angle = batch.line_angle.double().numpy()
        for hop in (0, 1):
            valid = mask[:, hop]
            if not bool(valid.any()):
                continue
            keys = pair_keys(path_types, hop)[valid]
            values = angle[valid, hop]
            global_angle[0] += float(values.sum())
            global_angle[1] += int(values.size)
            unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
            sums = np.bincount(inverse, weights=values)
            for key, total, observed in zip(unique.tolist(), sums.tolist(), counts.tolist()):
                angle_sums[key] = angle_sums.get(key, 0.0) + total
                angle_counts[key] = angle_counts.get(key, 0) + int(observed)

    distance_global = global_distance[0] / max(1, global_distance[1])
    angle_global = global_angle[0] / max(1, global_angle[1])
    max_bond_types = 8
    distance_by_type = [distance_global] * max_bond_types
    for code, total in distance_sums.items():
        if code < max_bond_types and distance_counts[code] >= min_group:
            distance_by_type[code] = total / distance_counts[code]

    keys = np.asarray(sorted(angle_sums), dtype=np.int64)
    means = np.asarray([angle_sums[key] / angle_counts[key] for key in keys.tolist()],
                       dtype=np.float64) if keys.size else np.zeros(0, dtype=np.float64)
    own = int(sum(1 for key in angle_sums if angle_counts[key] >= min_group))
    with_own = keys[np.asarray([angle_counts[int(key)] >= min_group for key in keys.tolist()],
                               dtype=bool)] if keys.size else keys
    return {
        'distance_global': distance_global,
        'distance_by_type': distance_by_type,
        'angle_global': angle_global,
        'angle_keys_sorted': keys,
        'angle_means_sorted': means,
        'angle_keys_with_own_mean': with_own,
        'summary': {
            'distance_global': distance_global,
            'distance_groups': {str(code): int(count) for code, count in sorted(distance_counts.items())},
            'distance_groups_with_own_mean': int(sum(
                1 for code, count in distance_counts.items()
                if count >= min_group and code < max_bond_types)),
            'bond_observations': int(global_distance[1]),
            'angle_global': angle_global,
            'angle_observations': int(global_angle[1]),
            'angle_groups_total': int(len(angle_sums)),
            'angle_groups_with_own_mean': own,
            'angle_rare_groups': int(len(angle_sums) - own),
            'min_group': int(min_group),
            'note': ('statistics are fit on XC/fold0 train only; rare groups fall '
                     'back to that pool global mean; no topology, element or label '
                     'is modified'),
        },
    }


def lookup_angle_means(keys, statistics):
    """Vectorised lookup with the train global mean as the rare-group fallback."""
    table_keys = statistics['angle_keys_sorted']
    table_means = statistics['angle_means_sorted']
    if table_keys.size == 0:
        return np.full(keys.shape, statistics['angle_global'], dtype=np.float64)
    position = np.searchsorted(table_keys, keys)
    position = np.clip(position, 0, table_keys.size - 1)
    matched = table_keys[position] == keys
    return np.where(matched, table_means[position], statistics['angle_global'])


def replace_distance(batch, statistics):
    """Return a private CPU copy whose bond lengths are the train type means.

    The source batch is never modified: PyG's ``Data.to`` is in place, so every
    intervention must start from its own clone."""
    clone = batch.clone()
    by_type = torch.as_tensor(statistics['distance_by_type'], dtype=clone.bond_distance.dtype)
    global_mean = torch.as_tensor(statistics['distance_global'],
                                  dtype=clone.bond_distance.dtype)
    bond_type = clone.bond_type.long()
    known = bond_type < by_type.numel()
    clone.bond_distance = torch.where(known, by_type[bond_type.clamp(max=by_type.numel() - 1)],
                                      global_mean)
    return clone


def replace_angle(batch, statistics):
    """Replace every valid path angle with its train type-pair mean."""
    clone = batch.clone()
    path_types = batch_token_type_keys(clone)
    angle = clone.line_angle.clone()
    mask = clone.line_mask.bool()
    for hop in (0, 1):
        valid = mask[:, hop]
        if not bool(valid.any()):
            continue
        keys = pair_keys(path_types, hop)[valid.numpy()]
        replacement = torch.as_tensor(lookup_angle_means(keys, statistics),
                                      dtype=angle.dtype)
        angle[valid, hop] = replacement
    clone.line_angle = angle
    return clone


def layer_values(deploy_model, best_model, batch, condition, permutation):
    """Per-graph token/graph/fused/prediction for one intervention condition."""
    with torch.no_grad():
        encoded = deploy_model.encode(batch)
        z2_deploy, z3_deploy = split_fuse(deploy_model, encoded)
        best_encoded = best_model.encode(batch)
        z2_best, z3_best = split_fuse(best_model, best_encoded)

        if condition == 'no3d':
            z3_deploy = torch.zeros_like(z3_deploy)
            z3_best = torch.zeros_like(z3_best)
        elif condition == 'permuted3d' and permutation is not None:
            index = torch.as_tensor(permutation, device=z3_deploy.device)
            z3_deploy = z3_deploy[index]
            z3_best = z3_best[index]
        fused_deploy = torch.cat([z2_deploy, z3_deploy], -1)
        fused_best = torch.cat([z2_best, z3_best], -1)
        prediction = best_model.predictor(fused_best)
        count = int(z2_deploy.size(0))
        token = per_graph_mean(encoded['bond_states'].float(), batch.bond_batch, count)
    return {'token': token.cpu().numpy(),
            'graph2d': encoded['graph_2d'].float().cpu().numpy(),
            'graph3d': encoded['graph_3d'].float().cpu().numpy(),
            'fused': fused_deploy.float().cpu().numpy(),
            'prediction': prediction.float().cpu().numpy().reshape(-1, 1)}


def run_interventions(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task, fold, split_name, limit = args.probe_task, args.probe_fold, args.probe_split, args.limit

    deploy_model, deploy_meta = build_model('concat', args.checkpoint, device=device)
    best_model, best_meta = build_model('concat', args.best_checkpoint, device=device)
    if best_meta['kind'] != 'finetuned' or best_meta['task'] != task \
            or int(best_meta['fold']) != fold:
        raise ValueError('intervention scoring checkpoint is not the requested task/fold best')

    source, dataset, manifest = open_task(args, task)
    try:
        train, validation = fold_indices(manifest, fold)
        pool = train if split_name == 'train' else validation
        keys = [source.samples[int(index)][0] for index in pool]
        order = sorted(range(len(pool)), key=lambda position: keys[position])
        chosen = [int(pool[position]) for position in order[:limit]]
        log(f'[interv] {task}/fold{fold}/{split_name}: probing {len(chosen)} of {len(pool)} samples')
        if len(chosen) < 2:
            raise ValueError('intervention probe needs at least two samples for the shuffle control')

        statistics = collect_geometry_statistics(dataset, train, args.encode_batch, args.min_group)
        summary = statistics['summary']
        log(f'[interv] distance global={summary["distance_global"]:.6f} '
            f'groups={summary["distance_groups_with_own_mean"]}; '
            f'angle global={summary["angle_global"]:.6f} '
            f'groups={summary["angle_groups_with_own_mean"]}/{summary["angle_groups_total"]} '
            f'rare={summary["angle_rare_groups"]}')

        conditions = ('baseline', 'no3d', 'permuted3d', 'distance_mean', 'angle_mean')
        primary = {name: {layer: [] for layer in LAYERS} for name in conditions}
        repeated = {name: [{layer: [] for layer in LAYERS} for _ in range(REPEATS - 1)]
                    for name in conditions}
        permutation_report = []

        for start in range(0, len(chosen), args.encode_batch):
            chunk = chosen[start:start + args.encode_batch]
            base_batch = dual_glt_collate([dataset[index] for index in chunk])
            if len(chunk) >= 2:
                permutation = deterministic_derangement(len(chunk))
                permutation_report.append({'batch_start': int(start), 'size': int(len(chunk)),
                                           'fixed_points': int(np.sum(permutation == np.arange(len(chunk))))})
            else:
                permutation = None
                permutation_report.append({'batch_start': int(start), 'size': int(len(chunk)),
                                           'fixed_points': None, 'skipped': 'batch smaller than two'})
            for condition in conditions:
                if condition == 'distance_mean':
                    batch = replace_distance(base_batch, statistics)
                elif condition == 'angle_mean':
                    batch = replace_angle(base_batch, statistics)
                else:
                    batch = base_batch.clone()
                batch = batch.to(device)
                for repeat in range(REPEATS):
                    values = layer_values(deploy_model, best_model, batch, condition, permutation)
                    target = primary[condition] if repeat == 0 else repeated[condition][repeat - 1]
                    for layer, value in values.items():
                        target[layer].append(value)

        result = {
            'task': task, 'fold': int(fold), 'split': split_name,
            'probe_samples': len(chosen),
            'conditions': list(conditions),
            'repeats': REPEATS,
            'permutation': {'scheme': 'seed-derived derangement inside each probe batch',
                            'seed': PERMUTATION_SEED, 'batches': permutation_report},
            'statistics': summary,
            'deploy_checkpoint': deploy_meta,
            'scoring_checkpoint': best_meta,
            'layers': {},
        }
        scaler_mean = float(np.asarray(best_meta['scaler_mean']).reshape(-1)[0])
        scaler_scale = float(np.asarray(best_meta['scaler_scale']).reshape(-1)[0])
        for condition in conditions:
            entry = {layer: _relative_stats(np.concatenate(primary[condition][layer]),
                                            np.concatenate(primary['baseline'][layer]))
                     for layer in LAYERS}
            prediction = np.concatenate(primary[condition]['prediction']).reshape(-1)
            baseline_prediction = np.concatenate(primary['baseline']['prediction']).reshape(-1)
            entry['prediction_original_units'] = {
                'absolute_difference': _stats(np.abs(
                    (prediction - baseline_prediction) * scaler_scale)),
                'scaler_mean': scaler_mean, 'scaler_scale': scaler_scale}
            if condition != 'baseline':
                entry['repeat_floor'] = [
                    {layer: _relative_stats(np.concatenate(repeated[condition][index][layer]),
                                            np.concatenate(primary[condition][layer]))
                     for layer in LAYERS} for index in range(REPEATS - 1)]
            result['layers'][condition] = entry
        write_json(Path(args.output) / 'interventions.json', result)
        log('[interv] PASS')
    finally:
        source.close()


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def load_feature(base, task, fold, split):
    with np.load(Path(base) / f'{task}_fold{fold}_{split}.npz', allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def fit_ridge(x_train, y_train, x_validation, alpha):
    """Train-only feature and label standardisation, prediction in the original
    label units.  Nothing is fitted on the validation split."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    feature_scaler = StandardScaler().fit(x_train)
    label_scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(feature_scaler.transform(x_train),
              label_scaler.transform(np.asarray(y_train).reshape(-1, 1)).ravel())
    scaled = model.predict(feature_scaler.transform(x_validation)).reshape(-1, 1)
    return label_scaler.inverse_transform(scaled).reshape(-1)


def r2_original(predicted, actual):
    from sklearn.metrics import r2_score
    return float(r2_score(np.asarray(actual).reshape(-1, 1),
                          np.asarray(predicted).reshape(-1, 1)))


def run_probes(args):
    from sklearn.model_selection import KFold

    base = Path(args.output) / 'feat_deploy'
    report = {'alphas': list(ALPHAS), 'feature_root': str(base), 'units': {}, 'fit_budget': {},
              'permutation_note': ('z3 shuffle is drawn inside each split with '
                                   'seed 42 + fold; no label use, no cross-split exchange')}
    fits = 0
    for task in TASKS:
        for fold in FOLDS:
            unit_key = f'{task}/fold{fold}'
            train = load_feature(base, task, fold, 'train')
            validation = load_feature(base, task, fold, 'validation')
            y_train = train['label'].astype(np.float64)
            y_validation = validation['label'].astype(np.float64)

            perm_train = np.random.default_rng(42 + fold).permutation(len(y_train))
            perm_validation = np.random.default_rng(42 + fold).permutation(len(y_validation))

            families = {
                'L2': (train['z2'], validation['z2']),
                'L3': (train['z3'], validation['z3']),
                'L23': (np.concatenate([train['z2'], train['z3']], -1),
                        np.concatenate([validation['z2'], validation['z3']], -1)),
                'L23-S': (np.concatenate([train['z2'], train['z3'][perm_train]], -1),
                          np.concatenate([validation['z2'], validation['z3'][perm_validation]], -1)),
            }
            unit = {}
            for name, (x_train, x_validation) in families.items():
                candidates = {}
                for alpha in ALPHAS:
                    predicted = fit_ridge(x_train, y_train, x_validation, alpha)
                    candidates[str(alpha)] = {'alpha': float(alpha),
                                              'validation_r2': r2_original(predicted, y_validation)}
                    fits += 1
                best = max(candidates.values(), key=lambda item: item['validation_r2'])
                unit[name] = {'candidates': candidates, 'selected_alpha': best['alpha'],
                              'validation_r2': best['validation_r2'],
                              'feature_dim': int(x_train.shape[1])}
                log(f'[probe] {unit_key} {name}: r2={best["validation_r2"]:.6f} alpha={best["alpha"]}')

            oof = np.zeros(len(y_train))
            for inner_train, inner_test in KFold(3, shuffle=True,
                                                 random_state=42 + fold).split(train['z2']):
                predicted = fit_ridge(train['z2'][inner_train], y_train[inner_train],
                                    train['z2'][inner_test], 1.0)
                oof[inner_test] = predicted
                fits += 1
            residual = y_train - oof
            baseline_prediction = fit_ridge(train['z2'], y_train, validation['z2'], 1.0)
            fits += 1
            baseline_r2 = r2_original(baseline_prediction, y_validation)
            residual_candidates = {}
            for alpha in ALPHAS:
                residual_prediction = fit_ridge(train['z3'], residual, validation['z3'], alpha)
                residual_candidates[str(alpha)] = {
                    'alpha': float(alpha),
                    'validation_r2': r2_original(baseline_prediction + residual_prediction,
                                                 y_validation)}
                fits += 1
            best_residual = max(residual_candidates.values(), key=lambda item: item['validation_r2'])
            unit['residual'] = {
                'definition': ('KFold(3, shuffle=True, random_state=42+fold) OOF residual from an '
                               'alpha=1 2D Ridge; final validation prediction is the full-train '
                               'alpha=1 2D Ridge plus the 3D residual prediction'),
                'baseline_alpha': 1.0,
                'baseline_validation_r2': baseline_r2,
                'candidates': residual_candidates,
                'selected_alpha': best_residual['alpha'],
                'validation_r2': best_residual['validation_r2'],
                'delta_vs_alpha1_baseline': best_residual['validation_r2'] - baseline_r2,
            }
            log(f'[probe] {unit_key} residual: r2={best_residual["validation_r2"]:.6f} '
                f'delta={unit["residual"]["delta_vs_alpha1_baseline"]:+.6f}')
            report['units'][unit_key] = unit

    report['fit_budget'] = {'ridge_fits_executed': int(fits),
                            'ridge_fits_allowed': int(args.fit_budget)}
    if fits > int(args.fit_budget):
        raise RuntimeError(f'ridge fit budget exceeded: {fits} > {args.fit_budget}')
    write_json(Path(args.output) / 'probes.json', report)
    log(f'[probe] PASS fits={fits}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('encode', 'interventions', 'probes'))
    parser.add_argument('--cohort-root', default='data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1')
    parser.add_argument('--cache-root', default='data/processed/mips_trimer_scage_downstream')
    parser.add_argument('--dual-static-root', default='data/processed/glt_dual_v2/downstream/dual_static_v1')
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--best-checkpoint')
    parser.add_argument('--checkpoint-kind', default='deploy', choices=('deploy', 'finetuned'))
    parser.add_argument('--finetuned-root',
                        help='root holding {task}/fold{n}/{task}/fold{n}/best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--clean-cache-gib', type=float, default=1.0)
    parser.add_argument('--probe-task', default='xc')
    parser.add_argument('--probe-fold', type=int, default=0)
    parser.add_argument('--probe-split', default='train', choices=('train', 'validation'))
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--min-group', type=int, default=5)
    parser.add_argument('--fit-budget', type=int, default=174)
    args = parser.parse_args()
    if args.mode == 'encode':
        run_encode(args)
    elif args.mode == 'interventions':
        run_interventions(args)
    else:
        run_probes(args)


if __name__ == '__main__':
    main()
