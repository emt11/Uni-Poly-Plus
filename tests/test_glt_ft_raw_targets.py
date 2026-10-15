"""Only N1's final CPU correctness: raw labels, evaluation, contracts and GPU scope."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from scripts import run_glt_ft_raw_targets_unit as unit
from scripts import select_glt_ft_raw_targets as selection
from scripts import run_glt_ft_raw_targets as controller
from src.utils import evaluate


@pytest.mark.parametrize('path', ['fused', 'pure_2d'])
def test_original_labels_no_override_and_clean_cache_unchanged(monkeypatch, path):
    calls = []
    source = [object(), object()]
    template = Data(x=torch.tensor([[1.0]]))
    def fields(src, index):
        assert src is source
        return object(), object(), '*CC*', object(), None
    def builder(*args, **kwargs):
        calls.append(args)
        return template.clone()
    monkeypatch.setattr(unit, 'pretrain_source_fields', fields)
    monkeypatch.setattr(unit, 'build_dual_sample', builder)
    monkeypatch.setattr(unit, 'build_glt_2d_sample', builder)
    raw = np.array([-1.25, 57.123456789], dtype=np.float64)
    ds = unit.RawCleanDataset(source, raw, dict(prediction_path=path, geometry_mode='finite_incident'))
    for index in range(2):
        data = ds[index]
        assert data.y.dtype == torch.float32
        assert data.y.item() == np.float32(raw[index])
        assert 'y' not in ds._clean_cache[index]
        data.y.fill_(999)
        data.x.fill_(99)
        assert ds[index].y.item() == np.float32(raw[index])
        assert ds[index].x.item() == 1
    assert len(calls) == 2
    np.testing.assert_array_equal(ds.raw_targets, raw)
    np.testing.assert_array_equal(ds.targets, raw)
    with pytest.raises(ValueError, match='forbids'):
        ds.set_target_override([0, 1])
    assert 'y' not in template
    with pytest.raises(ValueError):
        unit.RawCleanDataset(source, np.array([np.nan, 1]), ds.cfg)


def test_original_unit_loss_gradient_and_prediction_without_inverse():
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(2.0))
        def forward(self, batch):
            return self.bias.expand_as(batch.y)
    batches = [Data(y=torch.tensor([10.0])), Data(y=torch.tensor([20.0]))]
    model = Predictor()
    loss, r2, truth, predicted = evaluate(model, batches, nn.MSELoss(), 'cpu', scaler=None)
    assert loss == pytest.approx(194.0)
    assert r2 == pytest.approx(-6.76)
    np.testing.assert_array_equal(truth, [10, 20])
    np.testing.assert_array_equal(predicted, [2, 2])
    direct_loss = nn.MSELoss()(model(Data(y=torch.tensor([10.0, 20.0]))), torch.tensor([10.0, 20.0]))
    direct_loss.backward()
    assert model.bias.grad.item() == pytest.approx(-26.0)


def make_best(arm, task='eea', fold=0):
    cfg = unit.make_config(arm)
    return dict(config=cfg, task=task, fold=fold, seed=42+fold,
                parent=str(unit.parent(arm)), head_initial=str(unit.head_path(arm, task, fold)),
                prediction_path=cfg['prediction_path'], input_dim=cfg['input_dim'],
                target_transform='identity', scaler_fit='none',
                target_contract=unit.target_contract(), best_epoch=7)


@pytest.mark.parametrize('arm', unit.ARMS)
def test_raw_best_rejects_scaled_or_incomplete_configuration(arm):
    cfg = unit.make_config(arm)
    assert cfg['target_transform'] == 'identity'
    assert cfg['ft_profile'] == unit.u1.PROFILE
    assert cfg['new_parameters'] == 0
    unit.validate_config(cfg, arm)
    best = make_best(arm)
    unit.validate_best(best, arm, 'eea', 0, cfg)
    for key, value in [('target_transform', 'standard'), ('scaler_fit', 'train'),
                       ('target_contract', None), ('input_dim', 12)]:
        wrong = copy.deepcopy(best)
        wrong[key] = value
        with pytest.raises(ValueError):
            unit.validate_best(wrong, arm, 'eea', 0, cfg)
    for key in ('scaler_mean', 'scaler_scale', 'scaler'):
        wrong = dict(best, **{key: [0]})
        with pytest.raises(ValueError):
            unit.validate_best(wrong, arm, 'eea', 0, cfg)
    for key in ('target_contract', 'parent', 'ft_profile'):
        wrong = copy.deepcopy(cfg)
        wrong.pop(key)
        with pytest.raises(ValueError):
            unit.validate_config(wrong, arm)
    folder = unit.ft_folder(arm, 'eea', 0)
    original = torch.load(unit.head_path(arm, 'eea', 0), map_location='cpu', weights_only=False)
    used = torch.load(folder/'head_initial.pt', map_location='cpu', weights_only=False)
    assert set(original) == set(used)
    assert all(torch.equal(original[k], used[k]) for k in original)
    actual = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    unit.validate_best(actual, arm, 'eea', 0, cfg)
    assert actual['target_contract']['scaler'] is None


def test_scaled_reference_roundtrip_bound_rejects_different_labels():
    raw = np.array([-1.25, 0.15, 57.123456789], dtype=np.float64)
    mean, scale = np.array([17.0]), np.array([20.0])
    truth = ((raw-mean)/scale).astype(np.float32).astype(np.float64)*scale+mean
    assert selection.check_reference_labels(raw, truth, mean, scale) >= 0
    with pytest.raises(ValueError, match='beyond'):
        selection.check_reference_labels(raw, truth+0.1, mean, scale)
    with pytest.raises(ValueError):
        selection.check_reference_labels(raw, truth, mean, [0])


def test_complete_freeze_and_formal_raw_target_binding(tmp_path, monkeypatch):
    out = tmp_path/'outputs'
    split_root = tmp_path/'splits'
    monkeypatch.setattr(unit, 'OUT', out)
    monkeypatch.setattr(selection, 'OUT', out)
    monkeypatch.setattr(selection, 'SPLIT', split_root)
    for arm in unit.ARMS:
        cfg = unit.make_config(arm)
        unit.write(out/'initialization'/f'{arm}.json', cfg)
        for task in unit.TASKS:
            folds = [dict(train_indices=[0, 1], validation_indices=[2], test_indices=[4, 3]) for _ in range(5)]
            unit.write(split_root/f'{task}.json', dict(folds=folds))
            for fold in range(5):
                folder = unit.ft_folder(arm, task, fold)
                history = [dict(epoch=e, stage='head' if e<=10 else 'joint', training_steps=1,
                                validation_rmse=0.5 if e==7 else 1.0,
                                target_transform='identity', ft_profile='uniform_u1') for e in range(1, 71)]
                best = make_best(arm, task, fold)
                best.update(history=history, optimizer_updates=70, state_dict={'w': torch.ones(1)})
                unit.write(folder/'metrics.json', dict(status='PASS', outer_test='NOT_RUN', epochs=70,
                           parameters=unit.PARAMETERS[arm], modes=cfg,
                           updates={'attempted':70, 'completed':70}))
                unit.write(folder/'target.json', unit.target_contract())
                torch.save(best, folder/'best.pt')
    missing = unit.ft_folder('UA_2D', 'xc', 4)/'metrics.json'
    saved = missing.read_text()
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        selection.freeze()
    assert not (out/'evaluation_before_test.json').exists()
    missing.write_text(saved)
    selection.freeze()
    frozen = unit.read(out/'evaluation_before_test.json')
    assert frozen['status']=='FROZEN' and len(frozen['units'])==70
    cfg = unit.make_config('UA_2D')
    best = make_best('UA_2D')
    binding = unit.frozen_binding(frozen, 'UA_2D', 'eea', 0, cfg, best)
    assert binding['test_indices']==[3, 4] and binding['target_contract']['fit']=='none'
    wrong = copy.deepcopy(frozen)
    wrong['validation_units']=69
    with pytest.raises(ValueError):
        unit.frozen_binding(wrong, 'UA_2D', 'eea', 0, cfg, best)
    wrong = copy.deepcopy(frozen)
    next(u for u in wrong['units'] if (u['arm'],u['task'],u['fold'])==('UA_2D','eea',0))['target_contract']['fit']='train'
    with pytest.raises(ValueError):
        unit.frozen_binding(wrong, 'UA_2D', 'eea', 0, cfg, best)
    with pytest.raises(FileExistsError):
        selection.freeze()


def test_only_authorized_gpu_single_unit_and_cpu_test_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'LOG', tmp_path)
    monkeypatch.setattr(controller, 'ACTIVE', [])
    launches=[]
    monkeypatch.setattr(controller.subprocess, 'run', lambda cmd, **kw: launches.append(cmd))
    for bad in ((0,), (1,), (4,), (2, 3), (5,)):
        with pytest.raises(ValueError):
            controller.launch('bad', ['python'], bad)
    controller.launch('ft_example', ['python', 'authorized.py'], [3])
    assert unit.read(tmp_path/'ft_example/environment.json')['CUDA_VISIBLE_DEVICES']=='3'
    assert unit.read(tmp_path/'ft_example/environment.json')['logical_device']=='cuda:0'
    controller.launch('final', ['python', '-m', 'pytest'], limit=300)
    assert unit.read(tmp_path/'final/environment.json')['CUDA_VISIBLE_DEVICES']==''
    assert 'timeout 300s' in (tmp_path/'final/command.sh').read_text()
    assert len(launches)==2


def test_resource_handoff_adopts_started_units_without_duplicate_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'LOG', tmp_path)
    monkeypatch.setattr(controller, 'ARMS', ('F_CE',))
    monkeypatch.setattr(controller, 'TASKS', ('eea',))
    monkeypatch.setattr(controller, 'ELIGIBLE', (2, 3))
    monkeypatch.setattr(controller, 'ACTIVE', [])
    launch_calls = []
    def finished_launch(name, command, gpus=(), limit=None):
        launch_calls.append((name, gpus))
        folder=tmp_path/name
        folder.mkdir()
        unit.write(folder/'environment.json', dict(CUDA_VISIBLE_DEVICES=str(gpus[0])))
        (folder/'exit.txt').write_text('0')
        return folder/'exit.txt'
    original=tmp_path/'ft_F_CE_eea_0'
    original.mkdir()
    unit.write(original/'environment.json', dict(CUDA_VISIBLE_DEVICES='3'))
    monkeypatch.setattr(controller, 'launch', finished_launch)
    def wait_existing(path):
        if path==original/'exit.txt':
            path.write_text('0')
        assert path.read_text()=='0'
    monkeypatch.setattr(controller, 'wait', wait_existing)
    monkeypatch.setattr(controller, 'write', lambda *args:None)
    monkeypatch.setattr(controller, 'consumption', lambda:{} )
    monkeypatch.setattr(controller.time, 'sleep', lambda *args:(original/'exit.txt').write_text('0'))
    controller.matrix(False, adopt=True)
    assert len(launch_calls)==4
    assert all(name!='ft_F_CE_eea_0' for name,_ in launch_calls)
    assert all(gpus[0] in (2,3) for _,gpus in launch_calls)


def test_handoff_waits_existing_exits_without_host_pid_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'LOG', tmp_path)
    folder=tmp_path/'ft_F_CE_eea_0'
    folder.mkdir()
    (folder/'pid.txt').write_text('123')
    waited=[]
    monkeypatch.setattr(controller, 'wait', lambda p:waited.append(p))
    controller.finish_existing_ft_for_handoff()
    assert waited==[folder/'exit.txt']
