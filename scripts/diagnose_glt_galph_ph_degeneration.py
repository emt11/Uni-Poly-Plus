#!/usr/bin/env python3
"""Read-only degeneration diagnostics for the GALPH PH input path.

Evidence for the r2 cycle (Plan.md sec.11): for each C1 resume checkpoint the PH
encoder is fed one fixed, traceable set of P_train profiles (the first N valid
rows of the frozen pretraining sidecar, addressed by row index *and* key), and
the full model is fed a fixed window of the P_train stream at fixed positions.

Reported per checkpoint, separately for the FP32 and the production BF16 path:

* whether the raw probe profiles differ from each other at all;
* the across-sample spread of the PH summary and its norm;
* the real-vs-constitive-mean output difference;
* the PH residual relative to the 3D summary it is added to;
* what replacing the PH input does to ``g3`` and to the heads that read it
  (``ph_head``, ``cl_proj3``) under a fixed model.

Only valid P_train rows are used, the selection never looks at any downstream
validation metric, and every value is compared against an explicit tolerance so
that a numerically meaningless non-zero is not reported as signal.  The script
writes one JSON (plus the probe tensors) and never modifies a checkpoint.

r3 corrections (Codex review of the r2 diagnostic):

* every condition runs on its own shallow batch view, so replacing the PH input
  for one condition cannot leak into the next one;
* the residual and the heads are taken *inside* the same autocast context as the
  forward that produced ``g3`` (the residual is read back as ``g3`` minus the
  pre-residual summary), so the bf16 and fp32 rows no longer mix precisions.
"""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.dataset.glt_galformer_ph import PHBettiReader
from src.dataset.glt_ph import PH_BINS, PH_CHANNELS
from src.modules.glt_galformer_ph import PH_PATCHES, PH_RADII_PER_PATCH, GLTGalPH
from src.modules.glt_galformer_ph_downstream import TRAINING_ONLY_HEADS
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           OrderedSampleStream,
                                           load_sample_index_artifact,
                                           move_labels, open_source)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain_glt_galformer_ph import _prepare_window, is_ph_path_parameter  # noqa: E402

# Machine epsilon is ~6e-8 (fp32) and ~7.8e-3 (bf16); the bf16 floor below is the
# conservative absolute level under which a difference is not called observable.
TOLERANCE = {'fp32': 1e-6, 'bf16': 1e-3}
PRECISIONS = ('fp32', 'bf16')


def _autocast(device, precision):
    if precision == 'bf16':
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def build_probe_set(reader, rows, profile_path, key_path):
    """First ``rows`` valid P_train rows, addressed by row index and 32-byte key."""
    keys = np.asarray(reader.keys)
    valid = np.asarray(reader.valid)
    selected, profiles = [], []
    for row in range(len(keys)):
        if len(selected) >= rows:
            break
        if not bool(valid[row]):
            continue
        profile, ok = reader.get(row, bytes(keys[row]).hex())
        if not ok:
            continue
        selected.append(row)
        profiles.append(np.asarray(profile, dtype=np.float32))
    if len(selected) < rows:
        raise ValueError(f'only {len(selected)} valid P_train rows available')
    stack = np.stack(profiles).astype(np.float32)
    np.save(profile_path, stack)
    key_path.write_text(json.dumps(
        {'sidecar_rows': selected,
         'keys': [bytes(keys[row]).hex() for row in selected],
         'selector': 'first valid rows of the frozen P_train sidecar, in row order',
         'tolerance': TOLERANCE}, indent=2), encoding='utf-8')
    return stack


def encoder_stats(encoder, profiles, const, device, precision):
    """Encoder-level degeneration numbers on the fixed probe set."""
    encoder.eval()
    mask = torch.zeros((profiles.size(0), PH_PATCHES), dtype=torch.bool,
                       device=profiles.device)
    const_batch = const.unsqueeze(0).expand(profiles.size(0), -1, -1).contiguous()
    with torch.no_grad(), _autocast(device, precision):
        summary = encoder.summarize(encoder(profiles, mask))
        average = encoder.summarize(encoder(const_batch, mask))
        shuffled = encoder.summarize(encoder(profiles.roll(1, 0), mask))
    profile_spread = float((profiles.max(0).values - profiles.min(0).values).max())
    summary_spread = float((summary.max(0).values - summary.min(0).values).max())
    real_vs_const = float((summary - average).abs().max())
    real_vs_shuffled = float((summary - shuffled).abs().max())
    tolerance = TOLERANCE[precision]
    return {
        'probe_rows': int(profiles.size(0)),
        'profile_spread': profile_spread,
        'profile_spread_resolvable': bool(profile_spread > tolerance),
        'ph_summary_norm_mean': float(summary.norm(dim=-1).mean()),
        'ph_summary_spread': summary_spread,
        'ph_summary_spread_resolvable': bool(summary_spread > tolerance),
        'real_vs_const_max_abs': real_vs_const,
        'real_vs_const_resolvable': bool(real_vs_const > tolerance),
        'real_vs_shuffled_max_abs': real_vs_shuffled,
        'real_vs_const_bit_identical': bool(torch.equal(summary, average)),
        'tolerance': tolerance,
    }


def mask_leakage(encoder, profiles, device, precision):
    """Masked patches must not carry their own raw content into the summary.

    Two inputs that differ *only* inside the masked patch must produce the same
    summary; anything else would be a direct target leak of the masked Betti
    patch the pretraining objective predicts.
    """
    encoder.eval()
    mask = torch.ones((profiles.size(0), PH_PATCHES), dtype=torch.bool,
                      device=profiles.device)
    mask[:, 0] = False                      # patch 0 (bins 0..3) stays visible
    perturbed = profiles.clone()
    # Patch p occupies bins [p*4, (p+1)*4); only the masked patches are perturbed.
    perturbed[:, :, PH_RADII_PER_PATCH:] += (
        torch.rand_like(perturbed[:, :, PH_RADII_PER_PATCH:]) * 10.0)
    with torch.no_grad(), _autocast(device, precision):
        left = encoder.summarize(encoder(profiles, mask))
        right = encoder.summarize(encoder(perturbed, mask))
    return {'changed_patches': int(mask[0].sum()),
            'max_abs_difference': float((left - right).abs().max()),
            'tolerance': TOLERANCE[precision],
            'no_observable_leakage': bool(torch.equal(left, right))}


def build_step0_model(common_init, device, summary_mode='cls', ph_mode='global', seed=42):
    """Rebuild the verified training start state from the common-init artifact.

    The same construction, seed and artifact produced a run whose first 256 steps
    reproduce the legacy C1 records bit for bit, so this reconstruction is the
    measured start state rather than an assumption about it.
    """
    from src.training.glt_dual_runtime import apply_common_initialization
    torch.manual_seed(seed)
    model = GLTGalPH(summary_mode, ph_mode)
    record = apply_common_initialization(model, common_init)
    return model.to(device).eval(), record


def compare_deploy_state(resume_state, deploy_state):
    """Deploy tensors must be exactly the resume tensors minus the training-only heads."""
    resume = {name[len('model.'):]: value for name, value in resume_state.items()
              if name.startswith('model.')}
    excluded = sorted(name for name in resume if name.startswith(TRAINING_ONLY_HEADS))
    expected = sorted(set(resume) - set(excluded))
    actual = sorted(deploy_state)
    identical = [name for name in set(expected) & set(actual)
                 if torch.equal(resume[name], deploy_state[name])]
    differing = sorted(set(expected) & set(actual) - set(identical))
    return {'deploy_tensors': len(actual), 'expected_tensors': len(expected),
            'expected_keys': expected, 'deploy_keys': actual,
            'keys_match': expected == actual,
            'excluded_training_only': excluded,
            'identical_tensors': len(identical), 'differing_tensors': differing,
            'all_identical': expected == actual and not differing}


def load_model(checkpoint, device):
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = package['model']
    model = GLTGalPH(str(package['identity']['summary_mode']),
                     package['identity']['ph_mode'])
    model.load_state_dict({name[len('model.'):]: value for name, value in state.items()},
                          strict=True)
    return model.to(device).eval(), int(package['step'])


def _forward(model, data, device, precision, include_heads=True):
    """Model outputs in one precision convention.

    The residual is read back from the model's own tensors (``g3`` minus the
    pre-residual summary it was added to) and the heads run *inside* the same
    autocast context as the forward, so no quantity is recomputed in a different
    precision than the pass that produced it.  ``include_heads=False`` is for
    deployments, which exclude the training-only heads by construction.
    """
    with torch.no_grad(), _autocast(device, precision):
        out = model(data)
        if out.get('cls3') is None:
            raise ValueError('the diagnostic requires the CLS summary route')
        summary = torch.where(out['line3d_valid'].bool().unsqueeze(-1), out['cls3'],
                              torch.zeros_like(out['cls3']))
        # Two views of the same quantity, both inside the forward's own precision:
        # the model's own tensors (exact convention, but a subtraction, so limited
        # by cancellation at ~eps_T of the summary) and the model's own expression
        # recomputed in the same context (cancellation-free).
        direct = torch.tanh(model.alpha_ph) * model.ph_to_summary(out['ph_summary'])
        direct = torch.where(data.ph_valid.bool().unsqueeze(-1), direct,
                             torch.zeros_like(direct))
        result = {'g3': out['g3'], 'summary': summary, 'residual': out['g3'] - summary,
                  'residual_direct': direct}
        if include_heads:
            result['ph_head'] = model.ph_head(out['g3'])
            result['cl_proj3'] = model.cl_proj3(out['g3'])
        return result


def _with_profile(data, profile):
    """A shallow batch view carrying a different PH profile.

    The batch handed in is never modified: every condition gets its own view, so
    one condition's input cannot leak into the next call.
    """
    if profile is None:
        return data
    view = copy.copy(data)
    view.ph_profile = profile
    return view


def _relative_change(left, right, reference):
    return float((left.float() - right.float()).norm()) / reference


def model_stats(model, data, const, device, precision, forced_gate=0.02,
                repeat_runs=3, include_heads=True):
    """Fixed-model effect of replacing the PH input, per gate condition.

    Two explicitly different conditions are reported: ``checkpoint``, which uses
    the trained ``alpha_ph`` tensor as it is (never rounded and rebuilt through
    ``atanh``), and ``forced``, a counterfactual that sets the gate to a stated
    value.  Every condition runs on its own batch view, and the model state is
    restored (``alpha_ph`` bit-exactly, training mode included) even if a pass
    raises.
    """
    handle = data.to(device)
    own = handle.ph_profile
    inputs = {'own': None,
              'const': const.unsqueeze(0).expand_as(own).contiguous(),
              'shuffled': own.roll(1, 0)}
    checkpoint_gate = float(torch.tanh(model.alpha_ph.detach().float()))
    original = model.alpha_ph.detach().clone()
    was_training = model.training
    rows = []
    try:
        model.eval()
        for condition, gate in (('checkpoint', None), ('forced', float(forced_gate))):
            with torch.no_grad():
                if gate is not None:
                    model.alpha_ph.fill_(math.atanh(gate))
            outputs = {name: _forward(model, _with_profile(handle, profile),
                                      device, precision, include_heads)
                       for name, profile in inputs.items()}
            reference = float(outputs['own']['summary'].float().norm().clamp_min(1e-30))
            residual_norm = float(outputs['own']['residual'].float().norm())
            direct_norm = float(outputs['own']['residual_direct'].float().norm())
            # The same input three times: the spread reported is the largest
            # difference actually observed, not a proven upper bound on noise.
            repeats = [_forward(model, handle, device, precision, include_heads)
                       for _ in range(int(repeat_runs))]
            observed_spread = max(_relative_change(repeats[i]['g3'], repeats[j]['g3'],
                                                   reference)
                                  for i in range(len(repeats))
                                  for j in range(i + 1, len(repeats)))
            row = {
                'condition': condition,
                'gate_source': ('the trained alpha_ph tensor, unmodified'
                                if gate is None else
                                f'counterfactual: tanh(alpha_ph) set to {gate}'),
                'forced_gate_requested': gate,
                'tanh_alpha': float(torch.tanh(model.alpha_ph.detach().float())),
                'precision': precision,
                'g3_dtype': str(outputs['own']['g3'].dtype),
                'residual_dtype': str(outputs['own']['residual'].dtype),
                'residual_direct_dtype': str(outputs['own']['residual_direct'].dtype),
                'head_dtype': (str(outputs['own']['ph_head'].dtype) if include_heads
                               else None),
                'reference_profile': 'own',
                'summary_norm': reference,
                # g3 - summary: the model's exact convention, cancellation-limited.
                'residual_norm': residual_norm,
                'residual_relative_norm': residual_norm / reference,
                # the model's own expression in the same context: cancellation-free.
                'residual_norm_direct': direct_norm,
                'residual_relative_norm_direct': direct_norm / reference,
                # How far the cancellation-limited view is from the exact one; a
                # number, not a verdict, because a degenerate checkpoint can put
                # them orders of magnitude apart while both stay negligible.
                'residual_views_ratio': (abs(residual_norm - direct_norm)
                                         / max(direct_norm, 1e-30)),
                'g3_real_norm': float(outputs['own']['g3'].float().norm()),
                'repeat_runs': int(repeat_runs),
                'repeat_max_observed_difference': observed_spread,
                'g3_relative_change_const': _relative_change(outputs['const']['g3'],
                                                             outputs['own']['g3'], reference),
                'g3_relative_change_shuffled': _relative_change(outputs['shuffled']['g3'],
                                                                outputs['own']['g3'], reference),
                'ph_head_relative_change_const': (_relative_change(
                    outputs['const']['ph_head'], outputs['own']['ph_head'],
                    float(outputs['own']['ph_head'].float().norm().clamp_min(1e-30)))
                    if include_heads else None),
                'cl_proj3_relative_change_const': (_relative_change(
                    outputs['const']['cl_proj3'], outputs['own']['cl_proj3'],
                    float(outputs['own']['cl_proj3'].float().norm().clamp_min(1e-30)))
                    if include_heads else None),
                'all_finite': bool(all(torch.isfinite(value).all() for value in
                                       ((outputs['own']['g3'], outputs['const']['g3'],
                                         outputs['own']['ph_head']) if include_heads
                                        else (outputs['own']['g3'], outputs['const']['g3'],
                                              outputs['own']['residual_direct'])))),
            }
            # A ten-fold margin over the observed repeat spread, or the stated
            # tolerance, whichever is larger.  The margin is a convention for
            # "not read as signal here", not a proven bound on the numerics.
            threshold = max(TOLERANCE[precision], 10.0 * observed_spread)
            row['resolution_threshold'] = threshold
            row['observable_at_tolerance'] = bool(
                row['residual_relative_norm_direct'] > threshold
                and row['g3_relative_change_const'] > threshold)
            rows.append(row)
    finally:
        with torch.no_grad():
            model.alpha_ph.copy_(original)
        model.train(was_training)
    return {'checkpoint_tanh_alpha': checkpoint_gate,
            'ph_valid_fraction': float(handle.ph_valid.float().mean()),
            'graph_count': int(handle.graph_available.numel()), 'rows': rows}


def step0_from_monitor(path):
    """The step-0 row of a pre-check monitor, as the verified initialization."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for row in rows:
        if int(row.get('step', -1)) == 0:
            return {'source': str(path), 'arm': row.get('arm'),
                    'probe': row.get('probe_sensitivity'),
                    'ph_parameter_norms': row.get('ph_parameter_norms'),
                    'tanh_alpha': row.get('tanh_alpha'),
                    'note': ('the legacy C1 run saved checkpoints every 1000 steps, so '
                             'step 0 has no checkpoint; this row is the measured start '
                             'state of a run constructed from the same common '
                             'initialization artifact with the same seed')}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-rows', type=int, default=64)
    parser.add_argument('--deployments', nargs='*',
                        default=['results/glt_galph_20260920/p2/c1/pretrain/resume_01000.pt',
                                 'results/glt_galph_20260920/p2/c1/pretrain/resume_02000.pt',
                                 'results/glt_galph_20260920/p2/c1/pretrain/resume_03000.pt',
                                 'results/glt_galph_20260920/p2/c1/pretrain/resume_04000.pt',
                                 'results/glt_galph_20260920/p2/c1/pretrain/resume_05000.pt'])
    parser.add_argument('--step0-monitor')
    parser.add_argument('--common-init',
                        default='results/glt_galph_20260920/p1/common_init_galph_v1.pt')
    parser.add_argument('--pretrain-sidecar',
                        default='results/glt_galph_20260920/p0/ph_sidecar_betti_v2')
    parser.add_argument('--const-profile',
                        default='results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream/'
                                'p_train_mean_profile.npy')
    parser.add_argument('--probe-dir', default='results/glt_galph_ph_retention_20260920/p1')
    parser.add_argument('--build-probe-only', action='store_true')
    parser.add_argument('--supersedes',
                        help='earlier diagnostic JSON this run replaces (recorded, never '
                             'overwritten)')
    parser.add_argument('--strict-load-pair', nargs=2, metavar=('RESUME', 'DEPLOY'),
                        help='also verify that the deployment package reproduces the '
                             'resume checkpoint state and diagnostics exactly')
    parser.add_argument('--replacement-scope',
                        help='which sections of the superseded file this run replaces')
    parser.add_argument('--window-steps', type=int, nargs='*', default=[0])
    parser.add_argument('--microbatch', type=int, default=84)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cohort-root')
    parser.add_argument('--cache-root')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    parser.add_argument('--sample-index-artifact')
    parser.add_argument('--sample-index-split', default='train')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    probe_dir = Path(args.probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    profile_path = probe_dir / 'ph_probe_profiles.npy'
    key_path = probe_dir / 'ph_probe_keys.json'
    reader = PHBettiReader(args.pretrain_sidecar)
    if profile_path.is_file() and key_path.is_file():
        profiles = torch.as_tensor(np.load(profile_path), dtype=torch.float32).to(device)
        probe_meta = json.loads(key_path.read_text(encoding='utf-8'))
    else:
        profiles = torch.as_tensor(build_probe_set(reader, args.probe_rows, profile_path,
                                                   key_path), dtype=torch.float32).to(device)
        probe_meta = json.loads(key_path.read_text(encoding='utf-8'))
    if args.build_probe_only:
        print(json.dumps({'probe_profiles': str(profile_path),
                          'rows': len(probe_meta['sidecar_rows']),
                          'first_row': probe_meta['sidecar_rows'][0],
                          'last_row': probe_meta['sidecar_rows'][-1]}, indent=2))
        return

    const = torch.as_tensor(np.load(args.const_profile), dtype=torch.float32).to(device)
    payload = {'probe_set': {'profiles': str(profile_path), 'keys': str(key_path),
                             'rows': len(probe_meta['sidecar_rows']),
                             'sidecar_rows_first_last': [probe_meta['sidecar_rows'][0],
                                                         probe_meta['sidecar_rows'][-1]],
                             'selector': probe_meta['selector'],
                             'tolerance': TOLERANCE},
               'checkpoints': []}
    if args.step0_monitor and Path(args.step0_monitor).is_file():
        payload['step0'] = step0_from_monitor(args.step0_monitor)

    run = json.loads(Path('results/glt_galph_20260920/p2/c1/pretrain/run.json')
                     .read_text(encoding='utf-8'))['identity']
    cohort_root = args.cohort_root or run['cohort_root']
    cache_root = args.cache_root or run['cache_root']
    dual_static_root = args.dual_static_root
    sample_index = (args.sample_index_artifact
                    or run['config'].get('sample_index_artifact'))
    source, _ = open_source(cohort_root, cache_root, dual_static_root=dual_static_root)
    try:
        index = load_sample_index_artifact(sample_index, args.sample_index_split)
        if index is not None:
            source = IndexedFrozenDualSource(source, index['indices'])
        stream = OrderedSampleStream(len(source), args.seed)
        windows = []
        for step in args.window_steps:
            window, positions = _prepare_window(source, stream, reader, step=step,
                                               micro=args.microbatch, accumulation=1,
                                               world=1, rank=0, seed=args.seed)
            windows.extend(window)
        payload['fixed_window'] = {
            'steps': args.window_steps, 'microbatch': args.microbatch, 'seed': args.seed,
            'world': 1, 'accumulation': 1,
            'samples': sum(int(data.graph_available.numel()) for data, _ in windows),
            'note': 'fixed positions of the P_train stream; no downstream metric involved'}
        window_data, window_labels = windows[0]
        window_labels = move_labels(window_labels, device)
        for path in args.deployments:
            model, step = load_model(path, device)
            entry = {'checkpoint': str(path), 'step': step, 'encoder': {}, 'model': {}}
            for precision in PRECISIONS:
                entry['encoder'][precision] = encoder_stats(
                    model.ph_encoder, profiles, const, device, precision)
                entry['encoder'][precision]['mask_leakage'] = mask_leakage(
                    model.ph_encoder, profiles, device, precision)
            for precision in PRECISIONS:
                entry['model'][precision] = model_stats(
                    model, window_data, const, device, precision)
            payload['checkpoints'].append(entry)
            print(json.dumps({'checkpoint': path, 'step': step,
                              'encoder_fp32': entry['encoder']['fp32'],
                              'model_fp32': entry['model']['fp32']['rows']}), flush=True)
            del model
        if args.strict_load_pair:
            resume_path, deploy_path = args.strict_load_pair
            package = torch.load(deploy_path, map_location='cpu', weights_only=False)
            state_report = compare_deploy_state(
                torch.load(resume_path, map_location='cpu',
                           weights_only=False)['model'], package['state_dict'])
            deploy_model = GLTGalPH(str(package['summary_mode']),
                                    package.get('ph_mode')).to(device)
            load_galformer_deployment(deploy_model, package, int(package['step']))
            resume_model, resume_step = load_model(resume_path, device)
            comparison = {}
            for precision in PRECISIONS:
                left_encoder = encoder_stats(resume_model.ph_encoder, profiles, const,
                                             device, precision)
                right_encoder = encoder_stats(deploy_model.ph_encoder, profiles, const,
                                              device, precision)
                encoder_fields = {
                    key: {'resume': left_encoder[key], 'deploy': right_encoder[key],
                          'identical': left_encoder[key] == right_encoder[key]}
                    for key in ('profile_spread', 'ph_summary_norm_mean',
                                'ph_summary_spread', 'real_vs_const_max_abs',
                                'real_vs_shuffled_max_abs', 'real_vs_const_bit_identical')}
                left = model_stats(resume_model, window_data, const, device, precision,
                                   include_heads=False)
                right = model_stats(deploy_model, window_data, const, device, precision,
                                    include_heads=False)
                rows = []
                for left_row, right_row in zip(left['rows'], right['rows']):
                    fields = {
                        key: {'resume': left_row[key], 'deploy': right_row[key],
                              'identical': left_row[key] == right_row[key],
                              'relative_difference': (
                                  abs(left_row[key] - right_row[key])
                                  / max(abs(left_row[key]), abs(right_row[key]), 1e-30)
                                  if isinstance(left_row[key], (int, float)) else None)}
                        for key in ('condition', 'tanh_alpha', 'summary_norm',
                                    'residual_norm_direct',
                                    'residual_relative_norm_direct',
                                    'g3_relative_change_const',
                                    'g3_relative_change_shuffled',
                                    'repeat_max_observed_difference',
                                    'resolution_threshold', 'observable_at_tolerance')}
                    # The two model objects hold identical tensors, so they can only
                    # differ through the forward's own run-to-run behaviour.  The
                    # effect fields are all "relative to |summary|", so they are
                    # compared as absolute differences in those units against the
                    # row's resolution threshold; a tiny effect divided by its own
                    # value would overstate a disagreement that is orders of
                    # magnitude below anything this diagnostic resolves.
                    effects = ('residual_relative_norm_direct', 'g3_relative_change_const',
                               'g3_relative_change_shuffled')
                    worst = max(abs(left_row[key] - right_row[key]) for key in effects)
                    threshold = min(left_row['resolution_threshold'],
                                    right_row['resolution_threshold'])
                    rows.append({
                        'condition': left_row['condition'],
                        'worst_effect_difference': worst,
                        'resolution_threshold': threshold,
                        'conclusions_identical': bool(
                            left_row['condition'] == right_row['condition']
                            and left_row['observable_at_tolerance']
                            == right_row['observable_at_tolerance']),
                        'within_resolution': bool(
                            worst <= threshold
                            and left_row['observable_at_tolerance']
                            == right_row['observable_at_tolerance']),
                        'fields': fields})
                comparison[precision] = {
                    'encoder': {'fields': encoder_fields,
                                'bit_identical': all(value['identical']
                                                     for value in encoder_fields.values())},
                    'model_rows': rows,
                    'within_resolution': all(row['within_resolution'] for row in rows)}
            payload['strict_load'] = {
                'resume': str(resume_path), 'deploy': str(deploy_path),
                'resume_step': resume_step, 'deploy_step': int(package['step']),
                'architecture': package.get('architecture'),
                'ph_encoder_version': package.get('ph_encoder_version'),
                'loaded_through': 'load_galformer_deployment (strict)',
                'state': state_report, 'comparison': comparison,
                'note': ('the deployment excludes the training-only heads by construction '
                         '(they are listed in state.excluded_training_only and frozen '
                         'downstream), so the head fields are not part of this check')}
            print(json.dumps({'strict_load': {
                'keys_match': state_report['keys_match'],
                'state_all_identical': state_report['all_identical'],
                'encoder_bit_identical': {precision: comparison[precision]['encoder'][
                    'bit_identical'] for precision in PRECISIONS},
                'model_within_resolution': {precision: comparison[precision][
                    'within_resolution'] for precision in PRECISIONS},
                'worst_effect_difference': {
                    precision: max(row['worst_effect_difference']
                                   for row in comparison[precision]['model_rows'])
                    for precision in PRECISIONS}}}), flush=True)
            del deploy_model, resume_model
        if args.common_init and Path(args.common_init).is_file():
            model, record = build_step0_model(args.common_init, device)
            entry = {
                'checkpoint': f'constructed from {args.common_init}', 'step': 0,
                'provenance': {
                    'common_init_sha256': record.get('sha256'),
                    'construction': ('set_global_seed(42) -> GLTGalPH(cls, global) -> '
                                     'apply_common_initialization'),
                    'verified_by': ('the R_LEGACY pre-check uses this construction and '
                                    'reproduces the legacy C1 records bit for bit for 256 '
                                    'steps, so this is the measured start state')},
                'encoder': {}, 'model': {}}
            for precision in PRECISIONS:
                entry['encoder'][precision] = encoder_stats(model.ph_encoder, profiles,
                                                            const, device, precision)
                entry['encoder'][precision]['mask_leakage'] = mask_leakage(
                    model.ph_encoder, profiles, device, precision)
            for precision in PRECISIONS:
                entry['model'][precision] = model_stats(model, window_data, const,
                                                        device, precision)
            observed = payload.get('step0')
            if observed:
                rebuilt = {name: float(parameter.detach().float().norm())
                           for name, parameter in model.named_parameters()
                           if is_ph_path_parameter(name)}
                compared = {name: {'constructed': rebuilt[name],
                                   'observed_legacy_step0': value,
                                   'equal': rebuilt[name] == value}
                            for name, value in (observed.get('ph_parameter_norms') or {}).items()
                            if name in rebuilt}
                probe = observed.get('probe') or {}
                compared['probe_summary_spread'] = {
                    'constructed': entry['encoder']['fp32']['ph_summary_spread'],
                    'observed_legacy_step0': probe.get('summary_spread'),
                    'equal': entry['encoder']['fp32']['ph_summary_spread']
                    == probe.get('summary_spread')}
                compared['probe_real_vs_const'] = {
                    'constructed': entry['encoder']['fp32']['real_vs_const_max_abs'],
                    'observed_legacy_step0': probe.get('real_vs_const_max_abs'),
                    'equal': entry['encoder']['fp32']['real_vs_const_max_abs']
                    == probe.get('real_vs_const_max_abs')}
                entry['provenance']['matches_observed_legacy_start_state'] = {
                    'all_equal': all(row['equal'] for row in compared.values()),
                    'items': compared}
            payload['checkpoints'].insert(0, entry)
            print(json.dumps({'checkpoint': 'constructed step 0', 'step': 0,
                              'matches_observed_start_state':
                                  entry['provenance'].get(
                                      'matches_observed_legacy_start_state', {}).get('all_equal')}),
                  flush=True)
            del model
    finally:
        source.close()
    payload['interpretation'] = (
        'A checkpoint whose ph_summary_spread, real_vs_const_max_abs and '
        'g3_relative_change_const are all at or below the reported tolerance no longer '
        'carries observable PH sample information in the checked precision.  Each row '
        'reports two explicitly different gate conditions: "checkpoint" uses the trained '
        'alpha_ph tensor unmodified, "forced" is a stated counterfactual.  Differences '
        'are resolved against max(tolerance, 10x the largest difference observed across '
        'three identical forwards); that ten-fold margin is a convention for "not read as '
        'signal here" and the observed spread is not a proven upper bound on the '
        'numerics, so this bounds what THIS measurement resolves in a given precision and '
        'says nothing about the precision itself.  The residual is the tensor the model '
        'actually added (g3 minus the pre-residual summary) and the heads are evaluated '
        'inside the same autocast context, so the reported dtypes and norms share one '
        'precision convention per row.')
    if args.supersedes:
        superseded = Path(args.supersedes)
        payload['supersedes'] = {
            'path': str(superseded),
            'sha256': (hashlib.sha256(superseded.read_bytes()).hexdigest()
                       if superseded.is_file() else None),
            'kept': True,
            'replacement_scope': args.replacement_scope or 'model sections only',
            'reason': ('r3: the earlier model-level rows used a batch whose ph_profile was '
                       'replaced in place (so one condition could leak into the next) and '
                       'recomputed the residual and heads in fp32 after a bf16 forward')}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
