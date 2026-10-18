"""P2 science stays r7; r8 permits exactly one full-state 15k->20k segment."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/glt_optimization_20261011/r4/U1'
ORIGINAL = ROOT / 'results/ua2d_fusion_20261017/p2_align20k_r7'
ORIGINAL_LOG = ROOT / 'logs/ua2d_fusion_20261017/p2_align20k_r7'
OUT = ROOT / 'results/ua2d_fusion_20261017/p2_align20k_r7_recovery1'
LOG = ROOT / 'logs/ua2d_fusion_20261017/p2_align20k_r7_recovery1'
PI = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
# Paths embedded in the scientific identity must retain their r7 values.
CONFIG = ORIGINAL / 'initialization/P_ALIGN.json'
INITIAL = ORIGINAL / 'initialization/P_ALIGN_step0.pt'
RESUME = ORIGINAL / 'pretrain/P_ALIGN/resume_15000.pt'
TRAIN_OUTPUT = OUT / 'pretrain/P_ALIGN'
STEPS = (5000, 10000, 15000, 20000)
PARENT = TRAIN_OUTPUT / 'deploy_20000.pt'
REVISION = 'ua2d_align20k_r7'
GPUS = (2, 3)
START = 15000
END = 20000
SEGMENT = END - START


def make_config():
    original = json.loads((BASE / 'initialization/P_ALIGN.json').read_text())
    return dict(original, plan='UA2D-FUSION-20261017-01', revision='r7', cycle='P2',
                cycle_root=str(ORIGINAL), expected_world_size=2, max_optimizer_steps=END,
                save_every=5000, initial_state=str(INITIAL), ua2d_pt_revision=REVISION)


def validate_config(config):
    if config != make_config() or config['global_batch'] != config['microbatch'] * 2 * 6:
        raise ValueError('P2 scientific r7 identity/batch changed')
    return config


def validate_recovery_request(resume, output, config_path):
    if (Path(resume).resolve() != RESUME or Path(output).resolve() != TRAIN_OUTPUT
            or Path(config_path).resolve() != CONFIG):
        raise ValueError('r8 allows only original resume_15000 and new recovery1 output')


def validate_resume_position(state, config):
    batch = config['global_batch']
    if (state['step'] != START or state['next_position'] != START * batch
            or state['scheduler']['step'] != START or len(state['rng']) != 2
            or state['identity']['config'] != config or state['identity']['world_size'] != 2):
        raise ValueError('P2 requires complete 15k model/optimizer/scheduler/two-rank RNG/data state')
    from src.training.glt_dual_runtime import scheduled_lr
    expected = scheduled_lr(START - 1, **{k: config[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')})
    if state['scheduler']['lr'] != expected or not state['optimizer']['state']:
        raise ValueError('15k optimizer/scheduler state missing or LR reset')


def segment_consumption(attempted, completed, status):
    if not (START <= completed <= attempted <= END and attempted - completed <= 1):
        raise ValueError('P2 logical segment budget/inflight mismatch')
    return dict(starts=1, attempted_updates=attempted - START, completed_updates=completed - START,
        logical_attempted=attempted, logical_completed=completed, segment_start=START,
        status=status, old_known=dict(starts=1, attempted_updates=18348, completed_updates=18347),
        old_interrupted_update='unknown: logical step18348 may have updated before bookkeeping',
        cumulative_starts=2, cumulative_known_completed=18347 + completed - START,
        cumulative_attempted_budget_occupancy=18348 + attempted - START)


def publish_rank(output, rank, world, phase, **fields):
    from src.training.glt_dual_runtime import write_json
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    value = dict(pid=os.getpid(), ppid=os.getppid(), start_ticks=int(stat[19]), rank=rank,
        local_rank=int(os.environ['LOCAL_RANK']), world=world, visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        argv=sys.argv, phase=phase, execution_revision='r8', **fields)
    p = Path(output) / f'rank_lifecycle_{rank}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    write_json(tmp, value)
    os.replace(tmp, p)


def classify_pt(snapshot, rank_records, launcher_pid, bound_nvml=(), allow_exited=False):
    """Two actual rank identities, group CUDA birth, cross-card PID union.

    No namespace arithmetic. Unassigned current contexts stay conservative.
    """
    proc = {r['pid']: r for r in snapshot['processes']}
    descendants = {launcher_pid}
    if allow_exited:
        descendants.update(r['pid'] for r in rank_records.values() if r['ppid'] == launcher_pid and r['phase'] == 'restored')
    changed = True
    while changed:
        previous = len(descendants)
        descendants.update(pid for pid, row in proc.items() if row['ppid'] in descendants)
        changed = previous != len(descendants)
    ready = []
    errors = []
    for rank, record in rank_records.items():
        row = proc.get(record['pid'])
        if (record['rank'] != rank or record['local_rank'] != rank or record['world'] != 2
                or record['visible'] != '2,3' or record['pid'] not in descendants
                or (row is None and not allow_exited)
                or (row is not None and row['start_ticks'] != record['start_ticks'])):
            errors.append('rank process birth/torchrun lineage mismatch: ' + str(rank))
        if record['phase'] == 'restored':
            ready.append(rank)
    nvml = set().union(*(set(snapshot['nvidia_pids'][str(g)]) for g in GPUS))
    # Proc-view external training reservations are retained without mixing its
    # namespace with NVML identifiers. CPU prefetch descendants are not ranks.
    external_proc = sorted(set().union(*(set(snapshot['proc_pids'][str(g)]) for g in GPUS)) - descendants)
    binding = set(bound_nvml)
    if not binding and len(ready) == 2 and len(nvml) == 2 and not external_proc and not errors:
        # Baseline was idle, exactly these two DDP ranks reached restored CUDA
        # state; bind the complete two-process birth group, not arbitrary ranks.
        binding = nvml.copy()
    unassigned = sorted(nvml - binding)
    total = len(nvml) + len(external_proc)
    return dict(raw=snapshot, ranks=rank_records, launcher_pid=launcher_pid,
        lineage_pids=sorted(descendants), registered_nvml=sorted(binding), unassigned_nvml=unassigned,
        external_proc=external_proc, unique_gpu_processes=total,
        per_card_contexts={str(g): len(set(snapshot['nvidia_pids'][str(g)])) for g in GPUS},
        method='idle-baseline required DDP CUDA birth group; union NVML across cards; proc external reservations conservative',
        errors=errors, ready_ranks=ready)


def bind_package(package, config):
    validate_config(config)
    from src.modules.glt_dual_pretrain import u1_alignment_options
    if package['step'] not in STEPS or not u1_alignment_options(package):
        raise ValueError('P2 deployment ALIGN schema/step mismatch')
    return dict(package, ua2d_pt_revision=REVISION, ua2d_pt_config=str(CONFIG),
                ua2d_pt_cycle='P2', ua2d_source_step0=str(BASE / 'initialization/P_ALIGN_step0.pt'))


def validate_parent(package):
    if (package.get('step') != END or package.get('ua2d_pt_revision') != REVISION
            or package.get('ua2d_pt_config') != str(CONFIG) or package.get('ua2d_pt_cycle') != 'P2'
            or package.get('ua2d_source_step0') != str(BASE / 'initialization/P_ALIGN_step0.pt')):
        raise ValueError('G2 requires original science identity P2 step20000 deployment')
    return package


def checkpoint_index():
    return [dict(step=s, **{prefix: str((ORIGINAL / 'pretrain/P_ALIGN' if s < END else TRAIN_OUTPUT)
        / f'{prefix}_{s:05d}.pt') for prefix in ('resume', 'deploy')}) for s in STEPS]


def require_complete():
    config = validate_config(json.loads(CONFIG.read_text()))
    runtime = json.loads((TRAIN_OUTPUT / 'runtime.json').read_text())
    consumption = json.loads((TRAIN_OUTPUT / 'consumption.json').read_text())
    if (runtime.get('status') != 'PASS' or runtime.get('completed_steps') != SEGMENT
            or runtime.get('segment_start') != START or runtime.get('logical_end') != END
            or runtime.get('config') != config or runtime.get('world_size') != 2
            or consumption != segment_consumption(END, END, 'COMPLETED')
            or (LOG / 'pretrain_exit.txt').read_text().strip() != '0'):
        raise ValueError('P2 recovery must finish exactly 5000 physical updates and logical20000 with true exit0')
    for rank in range(2):
        restored = json.loads((TRAIN_OUTPUT / f'recovery_rank{rank}.json').read_text())
        if (restored['start'] != START or restored['next_position'] != START * config['global_batch']
                or restored['resume'] != str(RESUME) or not restored['full_state_restored']):
            raise ValueError('15k model/optimizer/RNG/data inheritance evidence missing')
    checkpoints = checkpoint_index()
    for group in checkpoints:
        for prefix in ('resume', 'deploy'):
            if not Path(group[prefix]).is_file():
                raise FileNotFoundError(group[prefix])
    validation = []
    for step in (1000, 2500, 5000):
        path = ORIGINAL / f'pt_validation/P_ALIGN/step{step:05d}/metrics.json'
        metrics = json.loads(path.read_text())
        if (metrics.get('status') != 'PASS' or metrics.get('presented') != 1024
                or metrics.get('step') != step or metrics.get('config') != config
                or metrics.get('checkpoint_selection') is not False):
            raise ValueError('inherited P2 process validation record incomplete')
        validation.append(dict(path=str(path), step=step, reused=True))
    return dict(config=config, runtime=runtime, consumption=consumption,
        parent=str(PARENT), checkpoints=checkpoints, process_validation=validation,
        execution_revision='r8', resume=str(RESUME), old_failure=str(ORIGINAL / 'failure.json'))
