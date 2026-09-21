"""D2 arm tests for GLT-3D-GAIN-20260921-01 (r3/r4).

Structural checks run on synthetic model instances; the geometry-independence
and random-stream checks reuse the frozen xc/fold0 fixture already used by the
D1 tests.  No training update is executed here: the smoke/development budgets
belong to the launcher, not to this file.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_glt_3d_gain_d2 import (  # noqa: E402
    ARM_LRS, ARMS, COMMON_INIT_SEED, FOLDS, O8OnlyArm, SCHEDULE_TOTAL_EPOCHS,
    STAGE_EPOCH_LIMIT, TASKS, ambient_rng_digest, build_arm, build_reference,
    effective_capacity, failure_record, optimizer_for_arm, prepare_unit_directory,
    resolve_fold, run_epochs, unit_directory,
)
from src.dataset.glt_dual import dual_glt_collate  # noqa: E402
from src.modules.glt_dual import build_dual_glt_model  # noqa: E402
from src.training.glt_dual_runtime import CleanLabeledDataset, open_source  # noqa: E402
from src.utils import _cosine_scheduler  # noqa: E402

COHORT = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1'
CACHE = ROOT / 'data/processed/mips_trimer_scage_downstream'
STATIC = ROOT / 'data/processed/glt_dual_v2/downstream/dual_static_v1'
SPLIT = Path('data/splits/mips_outer5_inner20')
LAUNCHER = ROOT / 'scripts/run_glt_3d_gain_d2_smoke.sh'
CENTRE_ZERO_ROW = 0
BATCH_ROWS = 4
STREAM_SEED = 13
GEOMETRY_FIELDS = (
    'bond_distance', 'bond_z_a', 'bond_z_b', 'bond_type', 'bond_center',
    'line_source', 'line_target', 'line_path', 'line_angle', 'line_mask',
    'line_path_group', 'line_is_self', 'geometry_valid', 'geometry_invalid_reason',
)


class _PassthroughStream:
    """The pre-r4 behaviour: the 3D dropout draws from the shared stream.

    Used only as the negative control that shows the stream comparison is
    actually sensitive to how much randomness the 3D branch consumes.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _ToyBatch:
    def __init__(self, features, target):
        self.features = features
        self.y = target

    def to(self, device, non_blocking=True):
        return self


class _ToyModel(nn.Module):
    """A one-parameter model whose validation R² falls monotonically.

    ``AdaptationAdapter`` supplies the ``(output, None)`` tuple, exactly as it
    does for the real arms, so ``forward`` returns the raw prediction.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, batch):
        return self.scale * batch.features.reshape(-1)


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return _ToyBatch(self.features[index], self.targets[index])


def _toy_collate(items):
    return _ToyBatch(torch.cat([item.features for item in items]),
                     torch.cat([item.y for item in items]))


def _xc_batches(xc_unit, count):
    """``count`` batches of ``BATCH_ROWS`` frozen xc/fold0 train rows each."""
    indices = xc_unit['train'][:count * BATCH_ROWS]
    samples = [xc_unit['dataset'][index] for index in indices]
    return [dual_glt_collate(samples[start:start + BATCH_ROWS])
            for start in range(0, len(samples), BATCH_ROWS)]


@pytest.fixture(scope='module')
def reference():
    """A common initialisation in the production shape, without the 150 MB
    deployment package: the arm relations under test are structural."""
    torch.manual_seed(COMMON_INIT_SEED)
    model = build_dual_glt_model('concat')
    model.eval()
    return model


@pytest.fixture(scope='module')
def arms(reference):
    return {arm: build_arm(arm, reference)[0] for arm in ARMS}


@pytest.fixture(scope='module')
def xc_unit():
    source, frame = open_source(str(COHORT), str(CACHE), task='xc',
                                dual_static_root=str(STATIC))
    dataset = CleanLabeledDataset(source, frame['label'].to_numpy(dtype=np.float64))
    manifest = json.loads((ROOT / SPLIT / 'xc.json').read_text(encoding='utf-8'))
    fold = manifest['folds'][0]
    train = [int(v) for v in fold['train_indices']]
    validation = [int(v) for v in fold['validation_indices']]
    samples = [dataset[index] for index in train[:4]]
    yield {'dataset': dataset, 'train': train, 'validation': validation,
           'samples': samples, 'manifest': manifest}
    source.close()


def test_arm_learning_rate_table_matches_the_plan():
    assert ARM_LRS['f2d'] == {'o8': 1e-5, 'glt': None, 'norm': 1e-4, 'head': 1e-4}
    assert ARM_LRS['fbase'] == {'o8': 1e-5, 'glt': 1e-5, 'norm': 1e-4, 'head': 1e-4}
    assert ARM_LRS['fnorm'] == {'o8': 1e-5, 'glt': 1e-5, 'norm': 1e-5, 'head': 1e-4}
    assert ARM_LRS['fstable'] == {'o8': 1e-5, 'glt': 3e-6, 'norm': 1e-5, 'head': 1e-4}


def test_arms_differ_only_in_the_declared_learning_rates():
    base, norm, stable = ARM_LRS['fbase'], ARM_LRS['fnorm'], ARM_LRS['fstable']
    assert {key for key in base if base[key] != norm[key]} == {'norm'}
    assert {key for key in norm if norm[key] != stable[key]} == {'glt'}
    assert 'glt' not in ARM_LRS['f2d'] or ARM_LRS['f2d']['glt'] is None


def test_f2d_does_not_construct_a_glt_branch(arms):
    model = arms['f2d']
    assert isinstance(model, O8OnlyArm)
    assert not hasattr(model, 'glt')
    assert not any(name.startswith('glt.') for name, _ in model.named_parameters())
    for arm in ('fbase', 'fnorm', 'fstable'):
        assert hasattr(arms[arm], 'glt')


def test_shared_initial_tensors_are_identical_across_arms(reference, arms):
    source = reference.state_dict()
    for arm, model in arms.items():
        target = model.state_dict()
        for name, value in target.items():
            assert name in source, f'{arm}: {name} has no reference source'
            assert torch.equal(value, source[name]), f'{arm}: {name} differs'
    # The 2D half, the normalisations and the head must agree between F2D and
    # FBASE by name, which is what makes the arms comparable.
    f2d, fbase = arms['f2d'].state_dict(), arms['fbase'].state_dict()
    for name in f2d:
        if name.startswith('o8.') or name.startswith('norm2.') or name.startswith('predictor.'):
            assert torch.equal(f2d[name], fbase[name]), name


def test_build_arm_reports_full_coverage(reference):
    for arm in ARMS:
        model, evidence = build_arm(arm, reference)
        assert evidence['unsourced'] == []
        assert set(evidence['copied']) == set(model.state_dict())
        if arm == 'f2d':
            assert any(name.startswith('glt.') for name in evidence['reference_only'])
        else:
            assert evidence['reference_only'] == []


def test_optimizer_groups_cover_every_trainable_tensor_exactly_once(arms):
    for arm, model in arms.items():
        optimizer, evidence = optimizer_for_arm(model, arm, weight_decay=0.02)
        grouped = [name for group in evidence for name in group['parameter_names']]
        trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
        assert sorted(grouped) == trainable, arm
        assert len(grouped) == len(set(grouped)), f'{arm}: a parameter is in two groups'
        assert {group['name'] for group in evidence} == {'o8', 'norm', 'head'} | (
            {'glt'} if arm != 'f2d' else set())
        for group in evidence:
            assert group['weight_decay'] == 0.02
        lrs = {group['name']: group['lr'] for group in evidence}
        assert lrs['o8'] == ARM_LRS[arm]['o8'] and lrs['norm'] == ARM_LRS[arm]['norm']
        assert lrs['head'] == ARM_LRS[arm]['head']
        if arm != 'f2d':
            assert lrs['glt'] == ARM_LRS[arm]['glt']
        optimizer.zero_grad(set_to_none=True)


def test_optimizer_rejects_declared_glt_lr_without_a_glt_branch(reference):
    model, _ = build_arm('f2d', reference)
    with pytest.raises(ValueError, match='GLT lr without a GLT branch'):
        optimizer_for_arm(model, 'fbase', weight_decay=0.02)


def test_fbase_matches_the_reference_forward(reference, arms, xc_unit):
    batch = dual_glt_collate(xc_unit['samples'])
    reference.eval()
    fbase = arms['fbase'].eval()
    with torch.no_grad():
        expected = reference(batch)
        observed = fbase(batch)
    assert torch.equal(observed, expected)


def test_f2d_reports_dead_head_capacity(reference, arms):
    f2d = effective_capacity(arms['f2d'])
    assert f2d['dead_head_parameters'] == 512 * 512
    assert f2d['effective_parameters'] == f2d['total_parameters'] - 512 * 512
    base = effective_capacity(arms['fbase'])
    assert base['dead_head_parameters'] == 0
    assert base['effective_parameters'] == base['total_parameters']


def test_f2d_predicts_when_every_geometry_field_is_absent(arms, xc_unit):
    batch = dual_glt_collate(xc_unit['samples'])
    for name in GEOMETRY_FIELDS:
        if hasattr(batch, name):
            delattr(batch, name)
    model = arms['f2d'].eval()
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (len(xc_unit['samples']), 1)
    assert torch.isfinite(output).all()


def test_fbase_requires_the_geometry_fields(arms, xc_unit):
    """The negative control for the F2D test above: the base arm does read 3D."""
    batch = dual_glt_collate(xc_unit['samples'])
    for name in GEOMETRY_FIELDS:
        if hasattr(batch, name):
            delattr(batch, name)
    with pytest.raises(AttributeError):
        with torch.no_grad():
            arms['fbase'](batch)


def test_centre_bond_zero_sample_keeps_a_finite_prediction(arms, xc_unit):
    model = arms['f2d'].eval()
    sample = xc_unit['samples'][CENTRE_ZERO_ROW]
    assert int(sample.bond_center.sum()) == 0
    assert bool(sample.geometry_valid) is True
    batch = dual_glt_collate([sample])
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()


def test_split_indices_are_disjoint_and_cover_the_fold(xc_unit):
    manifest = xc_unit['manifest']
    fold = manifest['folds'][0]
    train = set(int(v) for v in fold['train_indices'])
    validation = set(int(v) for v in fold['validation_indices'])
    test = set(int(v) for v in fold['test_indices'])
    assert not (train & validation) and not (train & test) and not (validation & test)
    assert len(train) + len(validation) + len(test) == int(manifest['sample_count'])


def test_best_checkpoint_round_trips_with_strict_loading(reference, arms):
    for arm in ARMS:
        payload = {'state_dict': {k: v.clone() for k, v in arms[arm].state_dict().items()},
                   'arm': arm}
        if arm == 'f2d':
            fresh = O8OnlyArm()
        else:
            fresh = build_dual_glt_model('concat')
        result = fresh.load_state_dict(payload['state_dict'], strict=True)
        assert result.missing_keys == [] and result.unexpected_keys == []


# --- r4: the shared O8/head stream versus the private 3D stream ---------------

def test_common_stream_is_shared_by_all_four_arms_in_train_mode(arms, xc_unit):
    """Two consecutive train-mode batches: the O8/head draws land on the same
    stream position in every arm, and that position advances between batches."""
    batches = _xc_batches(xc_unit, count=2)
    observed, entries = {}, {}
    for arm in ARMS:
        model = arms[arm]
        model.train()
        torch.manual_seed(STREAM_SEED)
        stream = getattr(model, 'glt_stream', None)
        begin = None if stream is None else stream.entries
        observed[arm] = []
        with torch.no_grad():
            for batch in batches:
                model(batch)
                observed[arm].append(ambient_rng_digest())
        entries[arm] = (begin, None if stream is None else stream.entries)
    for index in range(len(batches)):
        assert len({observed[arm][index] for arm in ARMS}) == 1, f'batch {index}'
    assert observed['fbase'][0] != observed['fbase'][1]
    # The comparison is only meaningful because the 3D branch really does draw.
    for arm in ('fbase', 'fnorm', 'fstable'):
        start, end = entries[arm]
        assert end == start + len(batches), arm
    assert entries['f2d'] == (None, None)   # no 3D branch, so no private stream


def test_glt_stream_is_shared_by_the_dual_arms_and_never_replays_a_batch(arms, xc_unit):
    batches = _xc_batches(xc_unit, count=2)
    digests = {}
    for arm in ('fbase', 'fnorm', 'fstable'):
        model = arms[arm]
        model.train()
        torch.manual_seed(STREAM_SEED)
        stream = model.glt_stream
        begin = stream.entries
        digests[arm] = []
        with torch.no_grad():
            for batch in batches:
                model(batch)
                digests[arm].append(stream.digest())
        assert stream.entries == begin + len(batches)
    for index in range(len(batches)):
        assert len({digests[arm][index] for arm in ('fbase', 'fnorm', 'fstable')}) == 1
    assert digests['fbase'][0] != digests['fbase'][1]


def test_control_without_isolation_the_3d_draws_shift_the_common_stream(arms, xc_unit):
    """Negative control: the measurement above does report a difference when the
    3D dropout draws from the shared stream, which is the pre-r4 behaviour."""
    batch = _xc_batches(xc_unit, count=1)[0]
    model = arms['fbase']
    original = model.glt_stream
    try:
        model.glt_stream = _PassthroughStream()
        model.train()
        torch.manual_seed(STREAM_SEED)
        with torch.no_grad():
            model(batch)
        control = ambient_rng_digest()
    finally:
        model.glt_stream = original
    arms['f2d'].train()
    torch.manual_seed(STREAM_SEED)
    with torch.no_grad():
        arms['f2d'](batch)
    assert control != ambient_rng_digest()


def test_glt_failure_inside_the_production_encode_restores_the_common_stream(arms, xc_unit,
                                                                            monkeypatch):
    """An exception raised by the 3D branch must leave the shared stream where
    the O8 branch alone would have left it."""
    batch = _xc_batches(xc_unit, count=1)[0]
    f2d, fbase = arms['f2d'], arms['fbase']
    f2d.train()
    fbase.train()
    torch.manual_seed(STREAM_SEED)
    with torch.no_grad():
        f2d.encode(batch)
    after_2d = ambient_rng_digest()

    def explode(*args, **kwargs):
        raise RuntimeError('synthetic 3D failure')

    monkeypatch.setattr(fbase.glt, 'forward', explode)
    torch.manual_seed(STREAM_SEED)
    with pytest.raises(RuntimeError, match='synthetic 3D failure'):
        with torch.no_grad():
            fbase.encode(batch)
    assert ambient_rng_digest() == after_2d


# --- r4: unit records, stage limits and the launcher failure contract --------

def test_unit_directories_are_distinct_and_never_reused(tmp_path):
    units = [(arm, task, fold) for arm in ARMS for task in TASKS for fold in FOLDS]
    paths = {unit_directory(tmp_path, *unit) for unit in units}
    assert len(paths) == len(units)
    first = prepare_unit_directory(unit_directory(tmp_path, 'fbase', 'xc', 0))
    marker = first / 'best.pt'
    marker.write_text('previous unit', encoding='utf-8')
    with pytest.raises(FileExistsError):
        prepare_unit_directory(unit_directory(tmp_path, 'fbase', 'xc', 0))
    assert marker.read_text(encoding='utf-8') == 'previous unit'
    sibling = prepare_unit_directory(unit_directory(tmp_path, 'fbase', 'xc', 1))
    assert sibling != first and sibling.is_dir()


def test_stage_limits_and_task_fold_restrictions_match_the_plan():
    assert STAGE_EPOCH_LIMIT == {'smoke': 1, 'development': SCHEDULE_TOTAL_EPOCHS}
    assert SCHEDULE_TOTAL_EPOCHS == 30
    assert TASKS == ('xc', 'eps', 'eat') and FOLDS == (0, 1)
    for task in TASKS:
        for fold in FOLDS:
            path = unit_directory('/tmp/units', 'fbase', task, fold)
            assert path.parts[-3:] == ('fbase', task, f'fold{fold}')
    with pytest.raises(ValueError, match='unknown task'):
        unit_directory('/tmp/units', 'fbase', 'egb', 0)
    with pytest.raises(ValueError, match='unknown fold'):
        unit_directory('/tmp/units', 'fbase', 'xc', 2)


def test_failure_record_is_never_a_pass():
    from types import SimpleNamespace
    args = SimpleNamespace(arm='fbase', task='xc', fold=0, stage='smoke', epochs=1)
    record = failure_record(RuntimeError('synthetic failure'), args, 0.0)
    assert record['status'] == 'FAILED' and record['exit_code'] != 0
    assert record['error'] == 'RuntimeError: synthetic failure'


def test_resolve_fold_checks_task_and_three_set_separation(xc_unit):
    manifest = xc_unit['manifest']
    train, validation, evidence = resolve_fold(manifest, 'xc', 0,
                                               cohort_rows=int(manifest['sample_count']))
    assert evidence['sets_disjoint'] is True and evidence['outer_test'] == 'NOT_RUN'
    assert len(train) == evidence['train_rows'] and len(validation) == evidence['validation_rows']
    broken = json.loads(json.dumps(manifest))
    first_train = broken['folds'][0]['train_indices'][0]
    broken['folds'][0]['validation_indices'] = list(broken['folds'][0]['validation_indices']) + [first_train]
    with pytest.raises(ValueError, match='overlap'):
        resolve_fold(broken, 'xc', 0, cohort_rows=int(manifest['sample_count']))
    with pytest.raises(ValueError, match='task'):
        resolve_fold(manifest, 'eat', 0, cohort_rows=int(manifest['sample_count']))
    with pytest.raises(ValueError, match='row count'):
        resolve_fold(manifest, 'xc', 0, cohort_rows=int(manifest['sample_count']) + 1)


def _synthetic_manifest():
    """A small but complete fold under the fixed split contract."""
    return {'protocol': 'outer5_inner20', 'task': 'xc', 'sample_count': 6,
            'validation_is_test': False,
            'folds': [{'fold': 0, 'train_indices': [0, 1, 2], 'validation_indices': [3],
                       'test_indices': [4, 5]}]}


def _broken_manifest(**changes):
    manifest = json.loads(json.dumps(_synthetic_manifest()))
    fold = manifest['folds'][0]
    if 'fold' in changes:
        fold.update(changes.pop('fold'))
    manifest.update(changes)
    return manifest


def test_resolve_fold_accepts_a_complete_split():
    train, validation, evidence = resolve_fold(_synthetic_manifest(), 'xc', 0, cohort_rows=6)
    assert train == [0, 1, 2] and validation == [3]
    assert evidence['protocol'] == 'outer5_inner20' and evidence['validation_is_test'] is False
    assert evidence['union_equals_full_cohort'] is True and evidence['sets_disjoint'] is True
    assert evidence['test_rows'] == 2 and evidence['outer_test'] == 'NOT_RUN'


def test_resolve_fold_rejects_a_manifest_that_is_not_the_fixed_protocol():
    with pytest.raises(ValueError, match='protocol'):
        resolve_fold(_broken_manifest(protocol='outer5_inner30'), 'xc', 0, cohort_rows=6)
    without = _synthetic_manifest()
    del without['protocol']
    with pytest.raises(ValueError, match='protocol'):
        resolve_fold(without, 'xc', 0, cohort_rows=6)


def test_resolve_fold_rejects_a_shared_validation_test_flag():
    with pytest.raises(ValueError, match='validation_is_test'):
        resolve_fold(_broken_manifest(validation_is_test=True), 'xc', 0, cohort_rows=6)
    without = _synthetic_manifest()
    del without['validation_is_test']
    with pytest.raises(ValueError, match='validation_is_test'):
        resolve_fold(without, 'xc', 0, cohort_rows=6)


def test_resolve_fold_rejects_broken_index_arrays():
    cases = (
        ('repeats', {'train_indices': [0, 0, 2]}),           # an index used twice
        ('repeats', {'train_indices': [0, 0, 1, 2]}),        # duplicate, union still complete
        ('out-of-range', {'train_indices': [-1, 1, 2]}),     # negative index
        ('out-of-range', {'test_indices': [4, 6]}),          # index past sample_count
        ('non-integer', {'train_indices': [0, 1, '2']}),     # not an integer
    )
    for message, change in cases:
        with pytest.raises(ValueError, match=message):
            resolve_fold(_broken_manifest(fold=change), 'xc', 0, cohort_rows=6)
    with pytest.raises(ValueError, match='usable train_indices'):
        resolve_fold(_broken_manifest(fold={'train_indices': None}), 'xc', 0, cohort_rows=6)


def test_resolve_fold_rejects_missing_and_overlapping_rows():
    with pytest.raises(ValueError, match='missing'):          # a row in no split at all
        resolve_fold(_broken_manifest(fold={'test_indices': [4]}), 'xc', 0, cohort_rows=6)
    with pytest.raises(ValueError, match='overlap'):          # a row in two splits
        resolve_fold(_broken_manifest(fold={'train_indices': [0, 1, 2, 3]}), 'xc', 0,
                     cohort_rows=6)
    with pytest.raises(ValueError, match='row count'):        # wrong cohort size
        resolve_fold(_synthetic_manifest(), 'xc', 0, cohort_rows=7)


def test_run_epochs_records_the_epochs_that_actually_ran():
    """Early stopping must be reflected in the executed-epoch count."""
    features = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    train_loader = DataLoader(_ToyDataset(features, torch.zeros_like(features)), batch_size=4,
                              shuffle=False, collate_fn=_toy_collate)
    validation_loader = DataLoader(_ToyDataset(features, features.clone()), batch_size=4,
                                   shuffle=False, collate_fn=_toy_collate)
    model = _ToyModel()
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 0.05}], weight_decay=0.0)
    scheduler = _cosine_scheduler(optimizer, 30 * len(train_loader), 5 * len(train_loader))
    result = run_epochs(model, train_loader, validation_loader, nn.MSELoss(), optimizer, scheduler,
                        torch.device('cpu'), epochs=5, patience=1, scaler=None)
    assert [record['epoch'] for record in result['history']] == [1, 2]
    assert result['best_epoch'] == 1
    assert result['optimizer_updates'] == 2 * len(train_loader)
    assert result['history'][0]['validation_r2'] > result['history'][1]['validation_r2']


def _run_launcher(tmp_path, arms, failing_exit_code):
    """Run the real launcher against a stub runner under a temporary log."""
    stub = tmp_path / 'stub_runner.py'
    stub.write_text('import sys\n'
                    "arm = sys.argv[sys.argv.index('--arm') + 1]\n"
                    "print('stub unit', arm, flush=True)\n"
                    f"sys.exit({failing_exit_code} if arm == 'b' else 0)\n", encoding='utf-8')
    log = tmp_path / 'units.log'
    log.write_text('earlier run kept\n', encoding='utf-8')
    environment = {**os.environ, 'ARMS': arms, 'PYTHON': sys.executable, 'RUNNER': str(stub),
                   'LOG': str(log), 'OUTPUT': str(tmp_path / 'out')}
    done = subprocess.run(['bash', str(LAUNCHER)], env=environment, capture_output=True, text=True)
    return done, log.read_text(encoding='utf-8')


def test_launcher_stops_at_the_first_failure_with_its_exit_code(tmp_path):
    done, log = _run_launcher(tmp_path, 'a b c', 3)
    assert done.returncode == 3
    assert 'earlier run kept' in log          # the log is appended, never truncated
    assert 'ARM=a EXIT=0' in log and 'ARM=b EXIT=3' in log
    assert 'ARM=c' not in log                 # the failing arm stops the loop
    assert 'ALL_ARMS_OK' not in log           # and no completion marker is written


def test_launcher_marks_completion_only_when_every_arm_succeeds(tmp_path):
    done, log = _run_launcher(tmp_path, 'a c', 3)
    assert done.returncode == 0
    assert 'ARM=a EXIT=0' in log and 'ARM=c EXIT=0' in log
    assert 'ALL_ARMS_OK' in log

