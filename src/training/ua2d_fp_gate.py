"""G1-only runtime: bindings, read-only input cache, budgets and slots."""
import copy
import datetime
import json
import os
from pathlib import Path
import sqlite3
import numpy as np
import torch
from torch import nn
from src.modules.ua2d_fp_gate import FPGate, GatedHead
from src.modules.glt_dual import mean_pool, restore_pure_2d, strict_module_state
from src.training.xc_kernel import FEATURE_SPEC, generator, fingerprint, metrics
from src.training.fce_kernel import TASKS, bind_branch_arrays, blend_arrays, validate_split

ROOT = Path(__file__).resolve().parents[2]
OLD_OUT = ROOT / 'results/ua2d_fusion_20261017/g1_fp_gate'
OLD_LOG = ROOT / 'logs/ua2d_fusion_20261017/g1_fp_gate'
PREVIOUS_OUT = ROOT / 'results/ua2d_fusion_20261017/g1_fp_gate_recovery1'
PREVIOUS_LOG = ROOT / 'logs/ua2d_fusion_20261017/g1_fp_gate_recovery1'
FAILED_OUT = ROOT / 'results/ua2d_fusion_20261017/g1_fp_gate_recovery2'
FAILED_LOG = ROOT / 'logs/ua2d_fusion_20261017/g1_fp_gate_recovery2'
OUT = ROOT / 'results/ua2d_fusion_20261017/g2_fp_gate_pt20k_r8'
LOG = ROOT / 'logs/ua2d_fusion_20261017/g2_fp_gate_pt20k_r8'
U1 = ROOT / 'results/glt_optimization_20261011/r4/U1'
R10 = ROOT / 'results/fce_tune_20261016/global_blend'
OLD_PARENT = U1 / 'pretrain/P_ALIGN/deploy_05000.pt'
from src.training.ua2d_pt20k import PARENT, GPUS
DOWN = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT / 'data/splits/mips_paper5_outer'
ARMS = ('GATE_REAL', 'GATE_NULL')
FP_SPEC = {k: v for k, v in FEATURE_SPEC.items() if k not in
           ('kernel', 'returnDistance', 'kernel_dtype', 'g05_weight', 'krr_weight')}
POLICY = dict(plan='UA2D-FUSION-20261017-01', revision='r8', cycle='G2', parent=str(PARENT), parent_step=20000,
    arms=list(ARMS), main='GATE_REAL', fp_spec=FP_SPEC,
    normalization='sqrt(512)*x/L2(x)', bottleneck=32, gate='sigmoid concat(a,q) 544->1',
    initial_seed=20261017, initial_g=.1, zero_v=True, weight=.75,
    alpha='unchanged r10/r9', feature_cache='disabled', recipe='uniform_u1',
    head_epochs=10, joint_epochs=60, batch=24, workers=0, precision='fp32')
LIMITS = dict(ft_starts=70, ft_completed=70, epochs=4900, attempted=112000,
    completed=112000, formal_starts=70, formal_completed=70, gate_arrays=140,
    reference_units=35, reference_arrays=35, fp_generated=2461,
    ft_graphs=47000, test_graphs=11750, final_cpu_starts=1,
    pt=0, teacher=0, krr_fit=0, krr_forward=0, conformers=0, extra_seed=0, workers=0)
EDITED = ('src/modules/ua2d_fp_gate.py', 'src/training/ua2d_fp_gate.py',
          'scripts/run_ua2d_fp_gate_unit.py', 'scripts/run_ua2d_fp_gate.py',
          'tests/test_ua2d_fp_gate.py', 'CURRENT_PLAN.md', 'src/training/ua2d_pt20k.py',
          'scripts/run_ua2d_pt20k.py', 'scripts/pretrain_glt_dual.py', 'scripts/evaluate_glt_pt_endpoint.py')
ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
PYTHON = '/opt/conda/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def guard_write(path):
    p = Path(path).resolve()
    if any(p == root.resolve() or root.resolve() in p.parents for root in (OLD_OUT, OLD_LOG, PREVIOUS_OUT, PREVIOUS_LOG, FAILED_OUT, FAILED_LOG,
        ROOT / 'results/ua2d_fusion_20261017/p2_align20k_r7', ROOT / 'logs/ua2d_fusion_20261017/p2_align20k_r7',
        ROOT / 'results/ua2d_fusion_20261017/g2_fp_gate_pt20k_r7', ROOT / 'logs/ua2d_fusion_20261017/g2_fp_gate_pt20k_r7')):
        raise ValueError('old G1 root is read-only')
    return p


def write(path, value, replace=False):
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with p.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    else:
        tmp = p.with_suffix(p.suffix + '.tmp')
        with tmp.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, p)


def check_budget(b):
    if set(b) != set(LIMITS) or any(type(v) is not int or v < 0 or v > LIMITS[k] for k, v in b.items()):
        raise RuntimeError('G1 budget exceeded')
    if b['gate_arrays'] + b['reference_arrays'] > 175:
        raise RuntimeError('array budget exceeded')


OLD_KNOWN = dict(ft_starts=43, ft_completed=19, attempted=30840, completed=30837, fp_generated=3414,
                 ft_graphs=None, actual_updates=None)
OLD_OCCUPANCY = dict(ft_starts=43, ft_completed=19, epochs=1573, attempted=30848,
    completed=30848, fp_generated=3414, ft_graphs=24135)


def consumption_record(new, unknown=None):
    check_budget(new)
    cumulative = dict(new)
    for key, value in OLD_OCCUPANCY.items():
        cumulative[key] += value
    if cumulative['ft_starts'] > 113 or cumulative['attempted'] > 142848 or cumulative['completed'] > 142848 or cumulative['ft_graphs'] > 71135 or cumulative['fp_generated'] > 5875:
        raise RuntimeError('cumulative recovery budget exceeded')
    return dict(old_known=OLD_KNOWN, old_budget_occupancy=OLD_OCCUPANCY,
                old_actual_unknown=['ft_graphs', 'exact_optimizer_updates', 'partial_epochs'],
                new=dict(new), new_incomplete_records=unknown or [], cumulative_budget_occupancy=cumulative)


def head_path(task, fold):
    return U1 / 'initialization/heads_2d' / task / f'fold{fold}.pt'


def unit_path(arm, task, fold, formal=False):
    return OUT / ('formal' if formal else 'ft') / arm / task / f'fold{fold}'


def module_initial_path(arm, task, fold):
    return OUT / 'initialization/gate_initial.pt'


def initial_state():
    return {k: v.detach().clone() for k, v in FPGate().state_dict().items()}


class GatedModel(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown G1 arm')
        self.o8, self.norm2 = base.o8, base.norm2
        self.head = GatedHead(base.head, constant=arm == 'GATE_NULL')
        # No feature_cache_context: trainable gate always stays online.

    def features(self, data):
        if hasattr(data, 'cached_features'):
            raise ValueError('G1 head feature cache forbidden')
        atoms, _ = self.o8(data)
        a = self.norm2(mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel()))
        return self.head.features(a, data.fp_bits)

    def forward(self, data):
        return self.head.regressor(self.features(data))


def build_model(package, head, arm, initial):
    from scripts.run_glt_align_2d_unit import validate_config
    cfg = validate_config(read(U1 / 'initialization/UA_2D.json'), 'UA_2D')
    from src.modules.glt_dual_pretrain import u1_alignment_options
    from src.training.ua2d_pt20k import validate_parent
    validate_parent(package)
    if (cfg['parent'] != str(OLD_PARENT) or package.get('step') != 20000 or
            not u1_alignment_options(package) or package.get('align_enabled') is not True):
        raise ValueError('UA P2 ALIGN step20000 mismatch')
    for key in ('architecture', 'fusion_mode', 'fusion_revision', 'geometry_revision',
                'geometry_mode', 'angle_value_enabled', 'readout', 'use_md200',
                'atom_ce_input', 'third_task', 'align_enabled', 'local_enabled', 'torsion_modules'):
        if package.get(key) != cfg[key]:
            raise ValueError('UA parent schema mismatch ' + key)
    base = restore_pure_2d(package, head, expected_step=20000)
    model = GatedModel(base, arm)
    expected = initial_state()
    if set(initial) != set(expected) or any(not torch.equal(initial[k], expected[k]) for k in expected):
        raise ValueError('common gate initialization mismatch')
    strict_module_state(model.head.fusion, initial)
    named = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError('duplicate parameter registration')
    counts = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.head.parameters()),
              sum(p.numel() for p in model.head.fusion.parameters()))
    if counts != (19345242, 346178, 83009):
        raise ValueError('G1 parameter count mismatch: ' + str(counts))
    return model


def stage_groups(model, stage, evidence):
    head = {id(p) for p in model.head.parameters()}
    named = dict(model.named_parameters())
    allowed = {id(p) for p in model.parameters() if stage == 'joint' or id(p) in head}
    if {id(p) for p in model.parameters() if p.requires_grad} != allowed:
        raise ValueError('GATE trainability mismatch')
    result = copy.deepcopy(evidence)
    seen = []
    if [g['name'] for g in result] != (['head'] if stage == 'head' else ['backbone', 'head']):
        raise ValueError('optimizer group schema mismatch')
    for g in result:
        members = [(n, p) for n, p in named.items() if p.requires_grad and
                   ((id(p) in head) == (g['name'] == 'head'))]
        if (g['num_parameters'], g['num_tensors']) != (sum(p.numel() for _, p in members), len(members)):
            raise ValueError('optimizer group count mismatch')
        g['parameter_names'] = [n for n, _ in members]
        seen.extend(id(p) for _, p in members)
    if len(seen) != len(set(seen)) or set(seen) != allowed:
        raise ValueError('optimizer omission/duplication')
    return result


class FPWriter:
    """The controller is the only writer, keyed by literal normalized SMILES."""
    def __init__(self, path, source=None, expected_old=3414):
        guard_write(path)
        if Path(path).exists():
            raise FileExistsError(path)
        self.db = sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE bits (smiles TEXT PRIMARY KEY, value BLOB NOT NULL)')
        self.gen = generator()
        self.generated = self.hits = 0
        self.inherited = 0
        if source is not None:
            old = sqlite3.connect(Path(source).resolve().as_uri() + '?mode=ro', uri=True)
            try:
                rows = old.execute('SELECT smiles,value FROM bits').fetchall()
            finally:
                old.close()
            if len(rows) != expected_old:
                raise ValueError('inherited FP row count changed')
            self.db.executemany('INSERT INTO bits VALUES (?,?)', rows)
            self.db.commit()
            self.inherited = len(rows)

    def ensure(self, smiles):
        row = self.db.execute('SELECT value FROM bits WHERE smiles=?', (smiles,)).fetchone()
        if row is None:
            if self.generated >= LIMITS['fp_generated'] or self.inherited + self.generated >= 5875:
                raise RuntimeError('FP budget exhausted')
            fp, _ = fingerprint(smiles, self.gen)
            bits = np.asarray(list(fp), dtype=np.uint8)
            self.db.execute('INSERT INTO bits VALUES (?,?)', (smiles, bits.tobytes()))
            self.db.commit()
            self.generated += 1
        else:
            bits = np.frombuffer(row[0], dtype=np.uint8)
            if bits.shape != (2048,) or not np.isin(bits, [0, 1]).all() or not bits.any():
                raise ValueError('invalid inherited/cached bits')
            self.hits += 1

    def close(self):
        self.db.close()


class FPReader:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        self.hits = 0

    def get(self, smiles):
        row = self.db.execute('SELECT value FROM bits WHERE smiles=?', (smiles,)).fetchone()
        if row is None:
            raise ValueError('controller did not prepare authorized FP')
        bits = np.frombuffer(row[0], dtype=np.uint8).copy()
        if bits.shape != (2048,) or not np.isin(bits, [0, 1]).all() or not bits.any():
            raise ValueError('invalid stored FP')
        self.hits += 1
        return torch.from_numpy(bits).float().reshape(1, 2048)

    def close(self):
        self.db.close()


def fp_collate(samples):
    from src.dataset.glt_dual import glt_2d_collate
    if not samples:
        raise ValueError('empty FP batch')
    for s in samples:
        x = s.fp_bits
        if x.shape != (1, 2048) or not torch.isfinite(x).all() or not ((x == 0) | (x == 1)).all() or not x.any():
            raise ValueError('invalid sample FP')
        if (s.fp_original_row, s.fp_sample_key, s.fp_smiles) != tuple(s.fp_binding):
            raise ValueError('FP sample identity mismatch')
    batch = glt_2d_collate(samples)
    batch.fp_bits = torch.cat([s.fp_bits for s in samples], dim=0)
    batch.fp_original_rows = [s.fp_original_row for s in samples]
    batch.fp_sample_keys = [s.fp_sample_key for s in samples]
    batch.fp_smiles = [s.fp_smiles for s in samples]
    return batch


def make_dataset(source, frame, indices, identities, cache, tracker=None, should_stop=None):
    from scripts.finetune_mcl_ph import CleanDualDataset, clone_clean_record
    from src.training.glt_dual_runtime import pretrain_source_fields
    from src.dataset.glt_dual import build_glt_2d_sample

    class GatedDataset(CleanDualDataset):
        def __init__(self):
            super().__init__(source, frame['label'].to_numpy(dtype=np.float64))
            self.materialized = set()
            self.rows = []

        def __getitem__(self, index):
            if should_stop and should_stop():
                raise InterruptedError('controlled stop at data boundary')
            row = indices[index]
            ident = identities[str(row)]
            key = source.samples[index][0].hex()
            smiles = frame.iloc[index]['normalized_smiles']
            if key != ident['sample_key'] or smiles != ident['normalized_smiles'] or int(frame.iloc[index]['original_row']) != row:
                raise ValueError('FP source identity mismatch')
            if index not in self._clean_cache:
                identity = dict(original_row=row, sample_key=key, normalized_smiles=smiles)
                if tracker:
                    tracker.begin(identity)
                top, _, text, static, _ = pretrain_source_fields(source, index)
                if top.normalized_canonical_smiles != smiles:
                    raise ValueError('FP/topology normalized_smiles mismatch')
                data = build_glt_2d_sample(top, text, static=static)
                data.fp_bits = cache.get(smiles)
                data.fp_original_row, data.fp_sample_key, data.fp_smiles = row, key, smiles
                data.fp_binding = (row, key, smiles)
                self._clean_cache[index] = data
                self.materialized.add(index)
                self.rows.append(dict(original_row=row, sample_key=key, normalized_smiles=smiles))
                if tracker:
                    tracker.complete(identity)
            data = clone_clean_record(self._clean_cache[index])
            data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
            return data
    return GatedDataset()


class InputTracker:
    def __init__(self, folder, role):
        self.folder, self.role = Path(folder), role
        self.attempted = self.completed = 0
        self.inflight = None
        self.publish()

    def publish(self):
        write(self.folder / 'input_consumption.json', dict(attempted=self.attempted,
            completed=self.completed, inflight=self.inflight, roles=self.role), True)

    def event(self, kind, identity):
        with (self.folder / 'input_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(kind=kind, identity=identity, time=now())) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def begin(self, identity):
        if self.inflight is not None:
            raise ValueError('concurrent dataset writer')
        self.attempted += 1
        self.inflight = dict(identity)
        self.publish()
        self.event('attempted', identity)

    def complete(self, identity):
        if self.inflight != identity:
            raise ValueError('materialization identity mismatch')
        self.event('completed', identity)
        self.completed += 1
        self.inflight = None
        self.publish()


def cuda_birth_pid(before, after, gpu):
    """Bind one real CUDA initialization, never an arithmetic PID offset."""
    prior = set(before['nvidia_pids'][str(gpu)])
    current = set(after['nvidia_pids'][str(gpu)])
    born = current - prior
    if len(born) != 1:
        raise RuntimeError('CUDA birth identity ambiguous: GPU%d new=%s' % (gpu, sorted(born)))
    return next(iter(born))


def classify_observation(snapshot, active):
    """Pure classification, allowing synthetic PID fixtures in the final suite."""
    proc = {int(p['pid']): p for p in snapshot['processes']}
    aliases = {}
    errors = []
    for pid, row in proc.items():
        for alias in (pid, *row.get('nspid', [])):
            if alias in aliases and aliases[alias] != pid:
                errors.append('ambiguous PID namespace alias: ' + str(alias))
            aliases[alias] = pid
    if snapshot.get('independent_nvml_pid_view'):
        # A single NSpid value in this container does not identify an NVML PID.
        aliases = {int(row['nspid'][0]): pid for pid, row in proc.items()
                   if len(row.get('nspid', [])) > 1}
    for unit in active.values():
        if unit.get('nvml_pid') is not None and unit.get('pid') is not None:
            host, local_pid = int(unit['nvml_pid']), int(unit['pid'])
            if host in aliases and aliases[host] != local_pid:
                errors.append('conflicting registered NVML identity: ' + str(host))
            aliases[host] = local_pid
    roots = {}
    for key, unit in active.items():
        for pid in (unit.get('pid'), unit.get('wrapper_pid')):
            if pid:
                roots[int(pid)] = key
        unit['aliases'] = []
    def owner(pid):
        seen = set()
        while pid and pid not in seen:
            if pid in roots:
                return roots[pid]
            seen.add(pid)
            pid = proc.get(pid, {}).get('ppid', 0)
        return None
    normalized = {2: set(), 3: set()}
    classified = []
    for gpu in (2, 3):
        nvml = set(snapshot['nvidia_pids'].get(str(gpu), []))
        local = set(snapshot['proc_pids'].get(str(gpu), []))
        local.update(aliases.get(p, -p if snapshot.get('independent_nvml_pid_view') else p) for p in nvml)
        for pid in sorted(local):
            key = owner(pid)
            if key is not None:
                if active[key]['gpu'] != gpu:
                    errors.append('owned PID GPU binding mismatch: ' + str(pid))
                active[key]['aliases'].append(pid)
                normalized[gpu].add(pid)
                classified.append(dict(gpu=gpu, pid=pid, owner=key, kind='own'))
            else:
                # An external parent experiment and its visible children share
                # one slot only when the sampled process relationship proves it.
                representative = pid
                seen = set()
                while representative in proc and representative not in seen:
                    seen.add(representative)
                    parent = proc[representative]['ppid']
                    if parent not in local or owner(parent) is not None:
                        break
                    representative = parent
                normalized[gpu].add(representative)
                classified.append(dict(gpu=gpu, pid=pid, owner=None, kind='external_or_unassigned',
                                       representative=representative, proc_present=pid in proc,
                                       nvml_pid=-pid if pid < 0 else None))
    detail = dict(raw=snapshot, active=copy.deepcopy(active), classified=classified,
                  normalized_pids={str(g): sorted(v) for g, v in normalized.items()}, errors=errors)
    return normalized, detail


class Slots:
    def __init__(self, gpus=GPUS, total_limit=4):
        self.gpus = tuple(gpus)
        self.total_limit = int(total_limit)
        self.active = {}
        self.peak = {g: 0 for g in self.gpus}
        self.total_peak = 0

    def occupancy(self, observed):
        counts = {}
        for gpu in self.gpus:
            own = [v for v in self.active.values() if v['gpu'] == gpu]
            pids = {p for v in own for p in (v.get('pid'), v.get('wrapper_pid'), *v.get('aliases', [])) if p}
            counts[gpu] = len(own) + len(set(observed.get(gpu, ())) - pids)
        return counts

    def total_occupancy(self, observed):
        aliases = {p for unit in self.active.values() for p in
                   (unit.get('pid'), unit.get('wrapper_pid'), *unit.get('aliases', [])) if p}
        external = set().union(*(set(observed.get(g, ())) for g in self.gpus)) - aliases
        return len(self.active) + len(external)

    def reserve(self, key, gpu, observed):
        counts = self.occupancy(observed)
        if key in self.active or gpu not in self.gpus or counts[gpu] >= 4 or self.total_occupancy(observed) >= self.total_limit:
            raise RuntimeError('GPU slot unavailable')
        self.active[key] = dict(gpu=gpu, pid=None, start=now())
        self.record(observed)

    def record(self, observed):
        counts = self.occupancy(observed)
        if any(v > 4 for v in counts.values()) or self.total_occupancy(observed) > self.total_limit:
            raise RuntimeError('GPU concurrency exceeded')
        for gpu, count in counts.items():
            self.peak[gpu] = max(self.peak[gpu], count)
        self.total_peak = max(self.total_peak, self.total_occupancy(observed))
        return counts

    def release(self, key, exit_record):
        if exit_record.get('exit') is None or exit_record.get('end') is None:
            raise ValueError('release requires true exit')
        return self.active.pop(key)


def validate_frozen(payload):
    expected = {(a, t, f) for a in ARMS for t in TASKS for f in range(5)}
    units = payload['units']
    if not payload.get('frozen') or payload.get('policy') != POLICY or len(units) != 70 or {(u['arm'], u['task'], u['fold']) for u in units} != expected:
        raise ValueError('all 70 bests must freeze before TEST')
    for u in units:
        validate_split(u, u['sample_count'])
        if u['parent'] != str(PARENT) or u['seed'] != 42 + u['fold'] or u['head_initial'] != str(head_path(u['task'], u['fold'])) or u['scaler'] != u['source']['scaler'] or u['alpha'] != u['source']['alpha'] or u['policy'] != POLICY:
            raise ValueError('frozen parent/head/scaler/alpha/policy mismatch')
    return payload


def require_freeze():
    return validate_frozen(read(OUT / 'selection_before_test.json'))


def load_arrays(path, unit, role, identities, value_keys=('y_pred',)):
    if role == 'test':
        require_freeze()
    elif role != 'validation':
        raise ValueError('invalid prediction role')
    with np.load(path, allow_pickle=False) as f:
        values = {k: f[k].copy() for k in f.files}
    return bind_branch_arrays(values, unit, role, identities, value_keys)


def fixed_blend(neural, krr):
    return blend_arrays(neural, krr, .75)
