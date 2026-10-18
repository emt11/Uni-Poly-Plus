"""Final-only G1 CPU checks with synthetic tensors and identities."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.modules.ua2d_fp_gate import FPGate, GatedHead
import src.training.ua2d_fp_gate as runtime
from src.training.ua2d_fp_gate import GatedModel, Slots, stage_groups, fixed_blend


def test_initial_identity_gate_shape_broadcast_rng():
    torch.manual_seed(71)
    state = torch.get_rng_state().clone()
    gate = FPGate()
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in gate.parameters()) == 83009
    other = FPGate()
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in gate.state_dict().items())
    a, x = torch.randn(3, 512), torch.ones(3, 2048)
    out, g, delta = gate.components(a, x)
    assert g.shape == (3, 1) and delta.shape == (3, 512)
    assert torch.allclose(g, torch.full((3, 1), .1))
    assert torch.equal(out, a) and torch.count_nonzero(delta) == 0
    with torch.no_grad():
        gate.v.bias.fill_(2.)
    out, g, delta = gate.components(a, x)
    torch.testing.assert_close(out, a + .2)
    torch.testing.assert_close(out, a + g.expand_as(delta) * delta)


def test_real_null_same_capacity_and_initial_function():
    regressor = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 1))
    import copy
    real = GatedHead(copy.deepcopy(regressor)).eval()
    null = GatedHead(copy.deepcopy(regressor), constant=True).eval()
    a = torch.randn(4, 512)
    bits = torch.randint(0, 2, (4, 2048)).float()
    assert sum(p.numel() for p in real.parameters()) == sum(p.numel() for p in null.parameters()) == 346178
    torch.testing.assert_close(real(a, bits), regressor.eval()(a), rtol=0, atol=0)
    torch.testing.assert_close(null(a, bits), regressor(a), rtol=0, atol=0)
    with torch.no_grad():
        real.fusion.v.weight.fill_(.01)
        null.fusion.v.weight.fill_(.01)
    torch.testing.assert_close(null.features(a, bits), null.features(a, torch.ones_like(bits)))


def test_gradient_paths_before_and_after_v_update():
    gate = FPGate()
    a = torch.randn(5, 512)
    x = torch.randint(0, 2, (5, 2048)).float()
    opt = torch.optim.SGD(gate.parameters(), lr=.01)
    first = gate(a, x).square().mean()
    first.backward()
    assert gate.v.weight.grad.abs().sum() > 0
    assert gate.u.weight.grad is not None and gate.u.weight.grad.abs().sum() == 0
    assert gate.gate.weight.grad is not None and gate.gate.weight.grad.abs().sum() == 0
    opt.step()
    opt.zero_grad(set_to_none=True)
    gate(a, x).square().mean().backward()
    for module in (gate.u, gate.v, gate.gate):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())


class TinyO8(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 512)

    def forward(self, data):
        return self.proj(data.x), None


def tiny_model():
    base = nn.Module()
    base.o8, base.norm2, base.head = TinyO8(), nn.LayerNorm(512), nn.Linear(512, 1)
    return GatedModel(base, 'GATE_REAL')


def test_head_joint_groups_and_cache_disabled():
    from scripts.finetune_mcl_ph import periodic_tdl_trainability, periodic_tdl_optimizer
    model = tiny_model()
    assert not hasattr(model, 'feature_cache_context')
    for stage in ('head', 'joint'):
        periodic_tdl_trainability(model, head_only=stage == 'head')
        optimizer, evidence = periodic_tdl_optimizer(model, 'eea', head_only=stage == 'head', profile='uniform_u1')
        groups = stage_groups(model, stage, evidence)
        names = [n for g in groups for n in g['parameter_names']]
        assert len(names) == len(set(names))
        assert all(n in names for n, _ in model.named_parameters() if n.startswith('head.'))
        if stage == 'head':
            assert all(n.startswith('head.') for n in names)
        else:
            assert set(names) == set(dict(model.named_parameters()))
        assert sum(len(g['params']) for g in optimizer.param_groups) == len(names)
    with pytest.raises(ValueError, match='cache'):
        model(SimpleNamespace(cached_features=torch.ones(2, 512)))


def test_fp_binding_and_collate(monkeypatch):
    import src.dataset.glt_dual as glt
    monkeypatch.setattr(glt, 'glt_2d_collate', lambda samples: SimpleNamespace())
    samples = [SimpleNamespace(fp_bits=torch.ones(1, 2048), fp_original_row=i,
        fp_sample_key=f'key{i}', fp_smiles=f'smiles{i}', fp_binding=[i, f'key{i}', f'smiles{i}']) for i in (7, 3)]
    batch = runtime.fp_collate(samples)
    assert batch.fp_bits.shape == (2, 2048)
    assert batch.fp_original_rows == [7, 3]
    assert batch.fp_sample_keys == ['key7', 'key3']
    assert batch.fp_smiles == ['smiles7', 'smiles3']
    samples[0].fp_sample_key = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        runtime.fp_collate(samples)


def test_fixed_fusion_original_units_and_no_fit(monkeypatch):
    import src.training.xc_kernel as kernel
    monkeypatch.setattr(kernel, 'fit_once', lambda *args, **kwargs: pytest.fail('unexpected fit'))
    a, k = np.array([-100., 0., 200.]), np.array([8., 12., -5.])
    got = fixed_blend(a, k)
    np.testing.assert_array_equal(got, .75 * a + .25 * k)
    assert got[0] < 0
    with pytest.raises(FloatingPointError):
        fixed_blend(np.array([1., np.nan]), np.ones(2))


def test_slots_include_external_reservations_and_live_pids():
    slots = Slots(gpus=(2, 3), total_limit=8)
    observed = {2: {10, 11, 12}, 3: {20, 21, 22, 23}}
    slots.reserve('ours', 2, observed)
    assert slots.occupancy(observed) == {2: 4, 3: 4}
    assert slots.total_peak == 8
    with pytest.raises(RuntimeError):
        slots.reserve('too_many', 2, observed)
    slots.active['ours']['pid'] = 99
    observed[2].add(99)
    assert slots.record(observed) == {2: 4, 3: 4}
    with pytest.raises(ValueError):
        slots.release('ours', dict(exit=None, end=None))
    assert 'ours' in slots.active
    slots.release('ours', dict(exit=0, end='finished'))
    assert not slots.active


def test_freeze_before_test_array_read(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'OUT', tmp_path)
    called = []
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: called.append(args))
    with pytest.raises(FileNotFoundError):
        runtime.load_arrays(tmp_path / 'test.npz', {}, 'test', {})
    assert called == []


def test_invalid_gate_inputs():
    gate = FPGate()
    with pytest.raises(ValueError):
        gate(torch.ones(2, 512), torch.zeros(2, 2048))
    with pytest.raises(ValueError):
        gate(torch.ones(2, 512), torch.ones(2, 32))
    with pytest.raises(FloatingPointError):
        gate(torch.full((2, 512), float('inf')), torch.ones(2, 2048))


def fake_observation(nvidia=None, proc_pids=None, processes=None):
    return dict(start='start', end='end', nvidia_raw='fixture', gpu_mapping_raw='fixture',
                nvidia_pids=nvidia or {'2': [], '3': []},
                proc_pids=proc_pids or {'2': [], '3': []}, processes=processes or [])


def test_pid_handoff_alias_and_duplicate_sources():
    active = {'unit': dict(gpu=2, pid=None, wrapper_pid=10)}
    observed = fake_observation(nvidia={'2': [120, 120], '3': []}, proc_pids={'2': [20], '3': []},
        processes=[dict(pid=10, ppid=1, nspid=[110, 10]), dict(pid=20, ppid=10, nspid=[120, 20])])
    normalized, detail = runtime.classify_observation(observed, active)
    slots = Slots(gpus=(2, 3), total_limit=8)
    slots.active = active
    assert slots.occupancy(normalized) == {2: 1, 3: 0}
    assert detail['classified'][0]['owner'] == 'unit'
    assert not detail['errors']
    active['unit']['pid'] = 20
    normalized, _ = runtime.classify_observation(observed, active)
    assert slots.occupancy(normalized)[2] == 1


def test_external_processes_remain_counted():
    snapshot = fake_observation(nvidia={'2': [40, 50], '3': [90]},
        proc_pids={'2': [40, 41], '3': []}, processes=[dict(pid=40, ppid=1), dict(pid=41, ppid=40)])
    normalized, detail = runtime.classify_observation(snapshot, {})
    assert normalized == {2: {40, 50}, 3: {90}}
    assert all(r['kind'] == 'external_or_unassigned' for r in detail['classified'])
    assert any(r['pid'] == 50 and not r['proc_present'] for r in detail['classified'])


def test_overflow_snapshot_is_persisted_before_raise(tmp_path, monkeypatch):
    import scripts.run_ua2d_fp_gate as controller
    monkeypatch.setattr(controller, 'OUT', tmp_path)
    monkeypatch.setattr(controller, 'observe_gpus', lambda: fake_observation(nvidia={'2': [], '3': [30, 31, 32, 33]}))
    c = controller.Controller()
    c.slots.active['ours'] = dict(gpu=3, pid=10)
    monkeypatch.setattr(c, 'refresh_bindings', lambda *args: None)
    with pytest.raises(RuntimeError, match='concurrency'):
        c.sample()
    saved = runtime.read(tmp_path / 'concurrency.json')
    assert saved['counts']['3'] == 5
    assert saved['raw']['nvidia_pids']['3'] == [30, 31, 32, 33]
    assert 'ours' in saved['active']
    assert (tmp_path / 'concurrency_observations.jsonl').exists()


def test_input_inflight_and_completion_records(tmp_path):
    tracker = runtime.InputTracker(tmp_path, 'train_validation')
    identity = dict(original_row=9, sample_key='key', normalized_smiles='*C*')
    tracker.begin(identity)
    saved = runtime.read(tmp_path / 'input_consumption.json')
    assert saved == dict(attempted=1, completed=0, inflight=identity, roles='train_validation')
    with pytest.raises(ValueError, match='writer'):
        tracker.begin(identity)
    tracker.complete(identity)
    saved = runtime.read(tmp_path / 'input_consumption.json')
    assert saved['attempted'] == saved['completed'] == 1 and saved['inflight'] is None
    assert len((tmp_path / 'input_events.jsonl').read_text().splitlines()) == 2


def test_old_consumption_and_recovery_bounds():
    new = {k: 0 for k in runtime.LIMITS}
    new.update(ft_starts=70, attempted=112000, completed=112000, ft_graphs=47000, fp_generated=2461)
    record = runtime.consumption_record(new)
    assert record['old_known']['ft_graphs'] is None
    assert record['old_known']['attempted'] == 30840 and record['old_known']['completed'] == 30837
    total = record['cumulative_budget_occupancy']
    assert (total['ft_starts'], total['attempted'], total['completed'], total['ft_graphs'], total['fp_generated']) == (113, 142848, 142848, 71135, 5875)
    new['ft_starts'] += 1
    with pytest.raises(RuntimeError):
        runtime.consumption_record(new)


def test_cache_inherits_without_regeneration(tmp_path, monkeypatch):
    import sqlite3
    source = tmp_path / 'old.sqlite'
    bits = np.zeros(2048, dtype=np.uint8)
    bits[3] = 1
    db = sqlite3.connect(source)
    db.execute('CREATE TABLE bits (smiles TEXT PRIMARY KEY, value BLOB NOT NULL)')
    db.execute('INSERT INTO bits VALUES (?,?)', ('*C*', bits.tobytes()))
    db.commit()
    db.close()
    monkeypatch.setattr(runtime, 'fingerprint', lambda *args: pytest.fail('regenerated inherited FP'))
    cache = runtime.FPWriter(tmp_path / 'new.sqlite', source=source, expected_old=1)
    cache.ensure('*C*')
    assert cache.inherited == 1 and cache.generated == 0 and cache.hits == 1
    cache.close()


def test_recovery_paths_reject_old_writes(tmp_path, monkeypatch):
    assert runtime.OUT.name == 'g2_fp_gate_pt20k_r8'
    assert runtime.LOG.name == 'g2_fp_gate_pt20k_r8'
    old_out, old_log = tmp_path / 'old_results', tmp_path / 'old_logs'
    monkeypatch.setattr(runtime, 'OLD_OUT', old_out)
    monkeypatch.setattr(runtime, 'OLD_LOG', old_log)
    for root in (old_out, old_log):
        with pytest.raises(ValueError, match='read-only'):
            runtime.write(root / 'bad.json', {})
        assert not root.exists()


def test_missing_input_record_keeps_unknown(tmp_path, monkeypatch):
    import scripts.run_ua2d_fp_gate as controller
    monkeypatch.setattr(controller, 'OUT', tmp_path)
    unit = tmp_path / 'unit'
    unit.mkdir()
    monkeypatch.setattr(controller, 'unit_path', lambda *args: unit)
    c = controller.Controller()
    c.jobs = {'one': dict(arm='GATE_REAL', task='eea', fold=0, formal=False)}
    c.budget['ft_starts'] = 1
    c.ledger()
    record = runtime.read(tmp_path / 'consumption.json')
    assert any(r['kind'] == 'input_record_not_yet_created' for r in record['new_incomplete_records'])
    assert record['cumulative_budget_occupancy']['ft_starts'] == 44
    assert record['cumulative_budget_occupancy']['ft_graphs'] == 24135


def test_cuda_birth_requires_unique_new_nvml_identity():
    before = fake_observation(nvidia={'2': [100], '3': [200]})
    after = fake_observation(nvidia={'2': [100, 479642], '3': [200]})
    assert runtime.cuda_birth_pid(before, after, 2) == 479642
    with pytest.raises(RuntimeError, match='ambiguous'):
        runtime.cuda_birth_pid(before, before, 2)
    after['nvidia_pids']['2'].append(479644)
    with pytest.raises(RuntimeError, match='ambiguous'):
        runtime.cuda_birth_pid(before, after, 2)


def test_registered_separate_pid_views_count_one_experiment():
    active = {'real': dict(gpu=2, pid=2934905, wrapper_pid=2933400, nvml_pid=479642)}
    snapshot = fake_observation(nvidia={'2': [479642], '3': []},
        proc_pids={'2': [2934905], '3': []},
        processes=[dict(pid=2934905, ppid=2933400, nspid=[2934905])])
    snapshot['independent_nvml_pid_view'] = True
    normalized, detail = runtime.classify_observation(snapshot, active)
    slots = Slots(gpus=(2, 3), total_limit=8)
    slots.active = active
    assert slots.record(normalized) == {2: 1, 3: 0}
    assert not detail['errors']
    snapshot['nvidia_pids']['2'].append(777)
    snapshot['processes'].append(dict(pid=777, ppid=1, nspid=[777]))
    normalized, detail = runtime.classify_observation(snapshot, active)
    assert -777 in normalized[2]
    assert slots.occupancy(normalized)[2] == 2
    assert any(r.get('nvml_pid') == 777 and r['owner'] is None for r in detail['classified'])


def test_gpu_csv_and_device_errors_preserve_evidence():
    from scripts.run_ua2d_fp_gate import parse_gpu_csv
    assert parse_gpu_csv('\n  \n2, GPU-abc\n3, "GPU-def"\n') == [('2', 'GPU-abc'), ('3', 'GPU-def')]
    assert parse_gpu_csv('\n') == []
    raw = 'Unable to determine the device handle for GPU2: Unknown Error\n0, GPU-abc\n'
    with pytest.raises(RuntimeError, match='Unable to determine'):
        parse_gpu_csv(raw)


def test_pt20k_budget_and_fixed_parent():
    from src.training.ua2d_pt20k import make_config, validate_config, validate_parent, STEPS
    import copy
    config = make_config()
    validate_config(config)
    assert STEPS == (5000, 10000, 15000, 20000)
    assert config['expected_world_size'] == 2
    assert config['microbatch'] * 2 * 6 == config['global_batch'] == 1008
    assert config['max_optimizer_steps'] == config['schedule_total_steps'] == 20000
    assert config['save_every'] == 5000 and runtime.POLICY['parent_step'] == 20000
    for field, value in [('max_optimizer_steps', 5000), ('align_weight', .5), ('expected_world_size', 4)]:
        bad = copy.deepcopy(config)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    with pytest.raises(ValueError):
        validate_parent({'step': 5000})


def test_total_four_slots_and_targeted_queries(monkeypatch):
    from scripts import run_ua2d_fp_gate as controller
    slots = Slots()
    assert slots.gpus == (2, 3) and slots.total_limit == 4
    for i in range(4):
        slots.reserve(str(i), 2 + i % 2, {2: set(), 3: set()})
    assert slots.occupancy({2: set(), 3: set()}) == {2: 2, 3: 2}
    with pytest.raises(RuntimeError):
        slots.reserve('fifth', 3, {2: set(), 3: set()})
    with pytest.raises(RuntimeError):
        Slots().reserve('forbidden', 1, {})
    calls = []
    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='3, GPU-fixture\n', stderr='')
    monkeypatch.setattr(controller.subprocess, 'run', query)
    assert controller.gpu_query('gpu=index,uuid') == '3, GPU-fixture\n'
    assert calls[0][1:3] == ['-i', '2,3']


def pt_fixture_snapshot(contexts=None, external=None):
    return dict(nvidia_pids=contexts or {'2': [101], '3': [102]},
        proc_pids=external or {'2': [11], '3': [12]}, processes=[
            dict(pid=10, ppid=1, start_ticks=100), dict(pid=11, ppid=10, start_ticks=110),
            dict(pid=12, ppid=10, start_ticks=120)])


def pt_fixture_ranks():
    return {rank: dict(pid=11+rank, ppid=10, start_ticks=110+10*rank, rank=rank,
        local_rank=rank, world=2, visible='2,3', phase='restored') for rank in range(2)}


def test_pt_ddp_birth_and_cross_card_union():
    from src.training.ua2d_pt20k import classify_pt
    ranks = pt_fixture_ranks()
    snapshot = pt_fixture_snapshot({'2': [101, 102], '3': [101, 102]})
    result = classify_pt(snapshot, ranks, 10)
    assert result['registered_nvml'] == [101, 102]
    assert result['unique_gpu_processes'] == 2 and not result['errors']
    snapshot['nvidia_pids'] = {'2': [101, 900], '3': [102, 900]}
    result = classify_pt(snapshot, ranks, 10, [101, 102])
    assert result['unassigned_nvml'] == [900] and result['unique_gpu_processes'] == 3
    # Unknown external PID is neither silently attributed nor double-counted.
    assert result['registered_nvml'] == [101, 102]


def test_pt_rank_birth_identity_and_normal_exit():
    from src.training.ua2d_pt20k import classify_pt
    ranks = pt_fixture_ranks()
    wrong = pt_fixture_snapshot()
    wrong['processes'][1]['start_ticks'] += 1
    assert classify_pt(wrong, ranks, 10)['errors']
    exited = pt_fixture_snapshot()
    exited['processes'] = [dict(pid=10, ppid=1, start_ticks=100)]
    assert classify_pt(exited, ranks, 10, [101, 102])['errors']
    assert not classify_pt(exited, ranks, 10, [101, 102], allow_exited=True)['errors']


def test_p2_segment_physical_consumption_and_unknown():
    from src.training.ua2d_pt20k import segment_consumption
    first = segment_consumption(15001, 15000, 'RUNNING')
    assert first['attempted_updates'] == 1 and first['completed_updates'] == 0
    end = segment_consumption(20000, 20000, 'COMPLETED')
    assert end['completed_updates'] == 5000 and end['logical_completed'] == 20000
    assert end['cumulative_known_completed'] == 23347
    assert end['cumulative_attempted_budget_occupancy'] == 23348
    assert 'unknown' in end['old_interrupted_update']
    with pytest.raises(ValueError):
        segment_consumption(20001, 20000, 'RUNNING')


def test_restricted_p2_resume_location_and_full_state():
    from src.training import ua2d_pt20k as pt
    from src.training.glt_dual_runtime import scheduled_lr
    pt.validate_recovery_request(pt.RESUME, pt.TRAIN_OUTPUT, pt.CONFIG)
    with pytest.raises(ValueError):
        pt.validate_recovery_request(pt.RESUME.parent / 'deploy_15000.pt', pt.TRAIN_OUTPUT, pt.CONFIG)
    with pytest.raises(ValueError):
        pt.validate_recovery_request(pt.RESUME, pt.ORIGINAL / 'pretrain/P_ALIGN', pt.CONFIG)
    cfg = pt.make_config()
    state = dict(step=15000, next_position=15000*1008, rng=[{}, {}],
        identity=dict(config=cfg, world_size=2), optimizer=dict(state={0: 'fixture'}),
        scheduler=dict(step=15000, lr=scheduled_lr(14999, **{k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')})))
    pt.validate_resume_position(state, cfg)
    state['next_position'] = 0
    with pytest.raises(ValueError):
        pt.validate_resume_position(state, cfg)


def test_g2_uses_recovered_20k_parent_and_unique_external_slots():
    from src.training import ua2d_pt20k as pt
    assert runtime.PARENT == pt.TRAIN_OUTPUT / 'deploy_20000.pt'
    assert runtime.OUT.name == 'g2_fp_gate_pt20k_r8'
    assert pt.make_config()['cycle_root'] == str(pt.ORIGINAL)
    slots = Slots()
    # One external context visible on two GPUs still spends one global slot.
    observed = {2: {-123}, 3: {-123}}
    slots.reserve('one', 2, observed)
    slots.reserve('two', 3, observed)
    slots.reserve('three', 2, observed)
    assert slots.total_occupancy(observed) == 4
    with pytest.raises(RuntimeError):
        slots.reserve('four', 3, observed)


def test_p2_segment_completion_rejects_fake_20k_consumption(tmp_path, monkeypatch):
    import json
    from src.training import ua2d_pt20k as pt
    train, old, logs = tmp_path / 'train', tmp_path / 'old', tmp_path / 'logs'
    for path in (train, old, logs):
        path.mkdir()
    config = dict(global_batch=1008)
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps(config))
    monkeypatch.setattr(pt, 'CONFIG', cfg_path)
    monkeypatch.setattr(pt, 'TRAIN_OUTPUT', train)
    monkeypatch.setattr(pt, 'ORIGINAL', old)
    monkeypatch.setattr(pt, 'LOG', logs)
    monkeypatch.setattr(pt, 'validate_config', lambda cfg: cfg)
    (logs / 'pretrain_exit.txt').write_text('0\n')
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    runtime_record = dict(status='PASS', completed_steps=5000, segment_start=15000,
                          logical_end=20000, config=config, world_size=2)
    save(train / 'runtime.json', runtime_record)
    save(train / 'consumption.json', pt.segment_consumption(20000, 20000, 'COMPLETED'))
    for rank in range(2):
        save(train / f'recovery_rank{rank}.json', dict(start=15000, next_position=15000*1008,
            resume=str(pt.RESUME), full_state_restored=True))
    for group in pt.checkpoint_index():
        for key in ('resume', 'deploy'):
            path = Path(group[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for step in (1000, 2500, 5000):
        save(old / f'pt_validation/P_ALIGN/step{step:05d}/metrics.json', dict(status='PASS',
            presented=1024, step=step, config=config, checkpoint_selection=False))
    assert pt.require_complete()['runtime']['completed_steps'] == 5000
    runtime_record['completed_steps'] = 20000
    save(train / 'runtime.json', runtime_record)
    with pytest.raises(ValueError, match='5000 physical'):
        pt.require_complete()
