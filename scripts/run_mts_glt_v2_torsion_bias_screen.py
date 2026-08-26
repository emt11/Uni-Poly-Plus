#!/usr/bin/env python3
"""Run TC/TG torsion-bias screening arms and strict report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/mts/glt_v2_torsion_bias_screen_v1.json'


def _command(config, arm, gpu_ids, smoke=False):
    mode = {
        'tc': 'o8_glt_atom_torsion_count',
        'tg': 'o8_glt_atom_torsion',
    }[arm]
    output, logs = ROOT / config['output_root'], ROOT / config['log_root']
    suffix = f'smoke/{arm}' if smoke else arm
    tasks = [config['tasks'][0]] if smoke else config['tasks']
    folds = ['0'] if smoke else [str(v) for v in config['folds']]
    return [
        sys.executable, 'scripts/run_mts_finetune_scheduler.py',
        '--python', sys.executable, '--gpu-ids', gpu_ids,
        '--results-dir', str(output / suffix), '--logs-dir', str(logs / suffix),
        '--tasks', *tasks, '--folds', *folds, '--seeds', str(config['seed']),
        '--checkpoint', str((ROOT / config['checkpoint']).resolve()),
        '--checkpoint-seed', str(config['seed']), '--pretrain-dataset', 'PI1M_v2',
        '--finetune-epochs', str(2 if smoke else config['epochs']),
        '--finetune-patience', str(config['patience']),
        '--batch-size', str(config['train_batch_size']),
        '--eval-batch-size', str(config['eval_batch_size']),
        '--amp-dtype', config['precision'], '--loader-workers', str(config['workers']),
        '--evaluation-protocol', config['evaluation_protocol'], '--train-args',
        '--experiment_id', config['experiment'] + '_' + suffix.replace('/', '_'),
        '--config_schema', 'mts-glt-v2-downstream',
        '--graph_encoder_type', 'mips_trimer_scage',
        '--topology_attention_variant', 'o8', '--no-use_star_rbf', '--no-use_mcl',
        '--use_md200', '--mts_glt_version', 'v2',
        '--mts_glt_layers', str(config['glt_layers']),
        '--mts_glt_attention_variant', config['glt_attention_variant'],
        '--mts_glt_mode', mode,
        '--periodic_line_glt_sidecar', str((ROOT / config['sidecar']).resolve()),
        '--mts_glt_fusion_strategy', 'legacy_zero', '--save_best_checkpoint',
        '--best_checkpoint_dir', str(output / 'checkpoints' / suffix),
        '--mts_glt_interaction_diagnostics_dir', str(output / 'torsion_units' / suffix),
        '--graph_lr', str(config['graph_lr']), '--fusion_lr', str(config['fusion_lr']),
        '--head_lr', str(config['head_lr']), '--warmup_epochs', str(config['warmup_epochs']),
        '--regression_loss', config['loss'], '--huber_beta', str(config['huber_beta']),
        '--head_dropout', str(config['head_dropout']),
        '--weight_decay', str(config['weight_decay']),
    ]


def _write_coverage(config):
    output = ROOT / config['output_root']
    unit = output / 'torsion_units/smoke/tc' / config['tasks'][0] / 'fold_0.json'
    payload = json.loads(unit.read_text(encoding='utf-8'))
    coverage = {
        key: payload[key] for key in (
            'relation_total', 'relation_covered', 'torsion_relation_coverage',
            'graph_total', 'graph_covered', 'torsion_graph_coverage',
            'mean_torsion_observations', 'median_torsion_observations',
            'p25_torsion_observations', 'p75_torsion_observations',
            'max_torsion_observations', 'internal_torsion_coverage',
            'cross_ru_torsion_coverage',
        )
    }
    coverage.update({
        'schema': 'mts-glt-v2-torsion-coverage-v1',
        'source': str(unit.relative_to(ROOT)),
        'task': config['tasks'][0], 'fold': 0,
    })
    if coverage['relation_covered'] <= 0:
        raise RuntimeError('torsion coverage is zero; formal screening is blocked')
    path = output / 'torsion_coverage.json'
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(coverage, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ids', default='0,1,2,3')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if not args.report_only:
        subprocess.run([
            sys.executable, 'scripts/check_mts_glt_v2_torsion_sanity.py',
            '--output', str(ROOT / config['output_root'] / 'geometry_sanity.json'),
        ], cwd=ROOT, check=True)
        if args.smoke:
            for arm in ('tc', 'tg'):
                status = subprocess.call(_command(config, arm, args.gpu_ids, True), cwd=ROOT)
                if status: return status
            _write_coverage(config)
            return 0
        if not (ROOT / config['output_root'] / 'torsion_coverage.json').is_file():
            raise RuntimeError('run --smoke first to establish nonzero torsion coverage')
        for arm in ('tc', 'tg'):
            status = subprocess.call(_command(config, arm, args.gpu_ids, False), cwd=ROOT)
            if status: return status
    return subprocess.call([
        sys.executable, 'scripts/report_mts_glt_v2_torsion_bias_screen.py'
    ], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
