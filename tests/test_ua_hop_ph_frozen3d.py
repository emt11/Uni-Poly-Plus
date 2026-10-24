"""Single final suite for new freezing/restoration/gate behavior only."""
import copy
import json
import pytest
import torch
from src.dataset.ua_hop_ph import collate
from src.modules.ua_hop_ph_frozen3d import initialize, restore, optimizer, SCHEMA, ARCHITECTURE, POLICY, CANDIDATE
from src.training.ua_hop_ph_frozen3d import PARENT, HEADS, REFERENCE, SIDECAR, development_gate


@pytest.fixture(scope='module')
def parent():
    return torch.load(PARENT, map_location='cpu', weights_only=False, mmap=True)


@pytest.fixture
def model(parent):
    torch.manual_seed(42)
    head = torch.load(HEADS/'eea/fold0.pt', map_location='cpu', weights_only=False)
    m = initialize(parent, head)
    m.adapter.load_state_dict(torch.load(REFERENCE/'initialization/common_adapter.pt', map_location='cpu', weights_only=False))
    m.router.load_state_dict(torch.load(REFERENCE/'initialization/ph_router.pt', map_location='cpu', weights_only=False))
    return m


def test_initialization_and_complete_disjoint_optimizer_policy(model, parent):
    for name in ('o8', 'encoder3d', 'norm2'):
        for k, v in getattr(model, name).state_dict().items():
            assert torch.equal(v, parent['state_dict'][name+'.'+k])
    for stage in ('head', 'joint'):
        opt, evidence = optimizer(model, stage)
        model.train()
        assert not model.encoder3d.training
        assert all(not p.requires_grad for p in model.encoder3d.parameters())
        ids = [id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids) == len(set(ids))
        assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
        assert not (set(ids) & {id(p) for p in model.encoder3d.parameters()})
        assert [x['lr'] for x in evidence] == ([3e-4] if stage == 'head' else [1e-4, 1e-3, 1e-3])
        if stage == 'head': assert set(ids) == {id(p) for p in model.head.parameters()}
        else:
            assert all(p.requires_grad for p in model.o8.parameters())
            assert all(p.requires_grad for p in model.adapter.parameters())


def test_real_3d_forward_without_3d_gradients_but_joint_paths_train(model):
    rows = json.loads((SIDECAR/'tasks/eea.json').read_text())
    sample = None
    # A legal unavailable geometry is not a valid fixture for proving a real 3D call.
    # Only this final suite chooses a bounded existing fixture; no new geometry/cache.
    for row in rows[:16]:
        candidate = torch.load(SIDECAR/'samples'/f"{row['sample_key']}.pt", map_location='cpu', weights_only=False)['data']
        if candidate.geometry_valid and candidate.bond_distance.numel():
            sample = candidate
            break
    assert sample is not None, 'no valid real-3D fixture in the declared bounded input slice'
    batch = collate([sample])
    optimizer(model, 'joint'); model.train()
    with torch.no_grad(): model.adapter.W_out.weight.normal_(0., .01)
    calls = []
    hook = model.encoder3d.register_forward_hook(lambda module, args, out: calls.append(out[0].shape))
    model(batch).square().sum().backward(); hook.remove()
    assert calls and calls[0][0] == batch.bond_distance.numel()
    assert all(p.grad is None for p in model.encoder3d.parameters())
    assert model.o8.layers[-1].attention.qkv.weight.grad.abs().sum() > 0
    assert model.adapter.branches[0].qkv.weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_complete_restore_reapplies_freeze_policy(model, tmp_path):
    cfg = dict(arm=CANDIDATE, candidate_id=CANDIDATE, trainability_policy=POLICY)
    saved = dict(schema=SCHEMA, architecture=ARCHITECTURE, candidate_id=CANDIDATE,
        trainability_policy=POLICY, config=cfg, state_dict=model.state_dict(), best_stage='joint', best_epoch=20)
    path = tmp_path/'frozen.pt'; torch.save(saved, path)
    restored = restore(torch.load(path, map_location='cpu', weights_only=False), cfg)
    restored.train()
    assert not restored.encoder3d.training
    assert all(not p.requires_grad for p in restored.encoder3d.parameters())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    bad = copy.deepcopy(saved); bad['trainability_policy']['encoder3d'] = 'trainable'
    with pytest.raises(ValueError, match='policy'): restore(bad, cfg)


def test_full_precision_validation_gate():
    assert not development_gate(.5099999, .5)['passed']
    assert development_gate(.5100001, .5)['passed']
    assert not development_gate(.499, .5)['passed']
    assert development_gate(.52, .5)['threshold'] == .010


def test_test_dispatch_refused_before_positive_decision_and_freeze(tmp_path, monkeypatch):
    from scripts import run_ua_hop_ph_frozen3d as entry
    monkeypatch.setattr(entry, 'OUT', tmp_path)
    (tmp_path/'selection').mkdir()
    path = tmp_path/'selection/development_decision.json'
    path.write_text(json.dumps(dict(passed=False)))
    controller = entry.Controller.__new__(entry.Controller)
    controller.jobs = []; controller.refresh = lambda: dict(test_starts=0)
    with pytest.raises(RuntimeError, match='forbidden'):
        controller.launch('must_not_launch', 'test', [])
    path.write_text(json.dumps(dict(passed=True)))
    with pytest.raises(RuntimeError, match='forbidden'):
        controller.launch('still_not_frozen', 'test', [])
    assert controller.jobs == []
