"""One fresh U1 ALIGN cycle: 20k updates, checkpoints every 5k, two ranks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/glt_optimization_20261011/r4/U1'
OUT = ROOT / 'results/ua2d_fusion_20261017/p2_align20k_r7'
LOG = ROOT / 'logs/ua2d_fusion_20261017/p2_align20k_r7'
PI = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
CONFIG = OUT / 'initialization/P_ALIGN.json'
INITIAL = OUT / 'initialization/P_ALIGN_step0.pt'
STEPS = (5000, 10000, 15000, 20000)
PARENT = OUT / 'pretrain/P_ALIGN/deploy_20000.pt'
REVISION = 'ua2d_align20k_r7'
GPUS = (2, 3)


def make_config():
    original = json.loads((BASE / 'initialization/P_ALIGN.json').read_text())
    return dict(original, plan='UA2D-FUSION-20261017-01', revision='r7', cycle='P2',
                cycle_root=str(OUT), expected_world_size=2, max_optimizer_steps=20000,
                save_every=5000, initial_state=str(INITIAL), ua2d_pt_revision=REVISION)


def validate_config(config):
    if config != make_config():
        raise ValueError('P2 recipe differs from the fixed fresh ALIGN20k configuration')
    if config['global_batch'] != config['microbatch'] * 2 * 6:
        raise ValueError('P2 global batch/accumulation mismatch')
    return config


def bind_package(package, config):
    validate_config(config)
    from src.modules.glt_dual_pretrain import u1_alignment_options
    if package['step'] not in STEPS or not u1_alignment_options(package):
        raise ValueError('P2 deployment ALIGN schema/step mismatch')
    return dict(package, ua2d_pt_revision=REVISION, ua2d_pt_config=str(CONFIG),
                ua2d_pt_cycle='P2', ua2d_source_step0=str(BASE / 'initialization/P_ALIGN_step0.pt'))


def validate_parent(package):
    if (package.get('step') != 20000 or package.get('ua2d_pt_revision') != REVISION
            or package.get('ua2d_pt_config') != str(CONFIG) or package.get('ua2d_pt_cycle') != 'P2'
            or package.get('ua2d_source_step0') != str(BASE / 'initialization/P_ALIGN_step0.pt')):
        raise ValueError('G2 requires the fresh P2 step20000 deployment')
    return package


def require_complete():
    config = validate_config(json.loads(CONFIG.read_text()))
    folder = OUT / 'pretrain/P_ALIGN'
    runtime = json.loads((folder / 'runtime.json').read_text())
    consumption = json.loads((folder / 'consumption.json').read_text())
    if (runtime.get('status') != 'PASS' or runtime.get('completed_steps') != 20000
            or runtime.get('config') != config or runtime.get('world_size') != 2
            or consumption != dict(starts=1, attempted_updates=20000,
                                   completed_updates=20000, status='COMPLETED')):
        raise ValueError('P2 has not completed exactly 20000 updates')
    for step in STEPS:
        for prefix in ('resume', 'deploy'):
            if not (folder / f'{prefix}_{step:05d}.pt').is_file():
                raise FileNotFoundError(folder / f'{prefix}_{step:05d}.pt')
    validation = []
    for step in (1000, 2500, 5000):
        metrics = json.loads((OUT / f'pt_validation/P_ALIGN/step{step:05d}/metrics.json').read_text())
        if (metrics.get('status') != 'PASS' or metrics.get('presented') != 1024
                or metrics.get('step') != step or metrics.get('config') != config
                or metrics.get('checkpoint_selection') is not False):
            raise ValueError('P2 process validation record incomplete')
        validation.append(metrics)
    return dict(config=config, runtime=runtime, consumption=consumption,
                parent=str(PARENT), checkpoints=list(STEPS), process_validation=validation)
