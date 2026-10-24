"""Final r2 F0 CPU suite: synthetic fixtures only; no encoder or FP generation."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.training import ua2d_fp_film as f


def head_state():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(5)
        return nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1)).state_dict()


def test_morgan_definition_and_inverse_transform():
    assert f.FEATURE_SPEC['kind']=='MorganBinary'
    assert f.FEATURE_SPEC['radius']==2 and f.FEATURE_SPEC['fpSize']==2048
    assert f.FEATURE_SPEC['includeChirality'] and f.FEATURE_SPEC['useBondTypes'] and f.FEATURE_SPEC['includeRingMembership']
    assert not f.FEATURE_SPEC['countSimulation']
    scaler=dict(mean=[2.],scale=[3.],fit='train',transform='standard')
    assert np.allclose(f.inverse(np.asarray([-1.,0.,1.]),scaler),[-1.,2.,5.])


def test_zero_initialization_pretrained_head_and_rng():
    h=torch.randn(4,512);bits=torch.ones(4,2048);state=head_state()
    baseline=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1));baseline.load_state_dict(state);baseline.eval()
    for mode in f.MODES:
        before=torch.get_rng_state().clone();model=f.CachedFusion(mode,state).eval()
        assert torch.equal(before,torch.get_rng_state())
        assert sum(p.numel() for p in model.adapter.parameters())==99360
        assert torch.equal(model.adapter(h,bits),h)
        assert torch.equal(model(h,bits),baseline(h))
        assert {name.split('.')[0] for name,_ in model.named_parameters()}=={'adapter','head'}
        assert not hasattr(model,'o8') and not hasattr(model,'norm2') and not hasattr(model,'readout')


def test_null_molecule_independence_and_shift_disables_gamma():
    h=torch.randn(3,512);a=torch.ones(3,2048);b=torch.zeros_like(a);b[:,:20]=1
    null=f.FilmAdapter('NULL')
    with torch.no_grad(): null.output.weight.normal_();null.output.bias.normal_()
    assert torch.equal(null(h,a),null(h,b))
    shift=f.FilmAdapter('SHIFT')
    with torch.no_grad(): shift.output.bias[:512].fill_(4.)
    assert torch.equal(shift(h,a),h)
    with torch.no_grad(): shift.output.bias[512:].fill_(.5)
    assert torch.allclose(shift(h,a),h+.1*torch.tanh(torch.tensor(.5)))


def test_condition_bounds_and_first_then_later_gradient_paths():
    model=f.FilmAdapter('FILM');h=torch.randn(3,512);bits=torch.zeros(3,2048);bits[:,:32]=1
    loss=(model(h,bits)-torch.ones_like(h)).square().mean();loss.backward()
    assert model.input.weight.grad is not None and torch.count_nonzero(model.input.weight.grad)==0
    assert model.output.weight.grad.abs().sum()>0
    with torch.no_grad(): model.output.weight.add_(-.5*model.output.weight.grad)
    model.zero_grad();model(h,bits).square().mean().backward()
    assert model.input.weight.grad.abs().sum()>0
    with torch.no_grad(): model.output.weight.fill_(100.);model.output.bias.fill_(100.)
    gamma,beta=model.condition(bits)
    assert gamma.abs().max()<=.1 and beta.abs().max()<=.1
    with pytest.raises(ValueError,match='FP input'): model(h,torch.zeros_like(bits))


def test_raw_h_split_and_key_alignment():
    entry=dict(train_indices=[3,0],validation_indices=[2],test_indices=[1])
    h=np.arange(3*512,dtype=np.float64).reshape(3,512)
    arrays=dict(features=h,indices=np.asarray([3,0,2]),sample_keys=np.asarray(['a','b','c']),train_count=np.asarray(2))
    restored=f.validate_features(arrays,entry,['a','b','c'])
    assert restored.dtype==np.float32 and np.array_equal(restored,h.astype(np.float32))
    assert np.linalg.norm(restored[0])>1.
    wrong=dict(arrays,indices=np.asarray([0,3,2]))
    with pytest.raises(ValueError,match='split'): f.validate_features(wrong,entry,['a','b','c'])
    with pytest.raises(ValueError,match='sample_key'): f.validate_features(arrays,entry,['a','c','b'])
    wrong=dict(arrays);wrong['train_u']=wrong.pop('features')
    with pytest.raises(ValueError,match='raw h'): f.validate_features(wrong,entry,['a','b','c'])


def test_strict_adaptation_restore():
    model=f.CachedFusion('FILM',head_state());state=copy.deepcopy(model.state_dict())
    other=f.CachedFusion('FILM',head_state());f.strict_restore(other,state)
    assert all(torch.equal(state[k],other.state_dict()[k]) for k in state)
    wrong=dict(state);wrong['unexpected']=torch.zeros(1)
    with pytest.raises(ValueError,match='names'): f.strict_restore(other,wrong)
    wrong=dict(state);wrong['adapter.output.bias']=wrong['adapter.output.bias'].double()
    with pytest.raises(ValueError,match='tensor'): f.strict_restore(other,wrong)


def test_development_selects_rows_before_reading_labels(monkeypatch):
    from src.training import glt_dual_runtime as runtime
    seen=[]
    manifest=dict(folds=[dict(train_indices=[0,2],validation_indices=[1],test_indices=[3])])
    monkeypatch.setattr(f,'read',lambda _:manifest)
    def boundary(*args,**kwargs):
        seen.extend(kwargs['selected_indices']);raise RuntimeError('selected source boundary')
    monkeypatch.setattr(runtime,'open_source',boundary)
    with pytest.raises(RuntimeError,match='boundary'): f.load_unit('eea',0)
    assert seen==[0,1,2] and 3 not in seen
    with pytest.raises(ValueError): f.load_unit('eat',0)


def test_missing_fp_never_generates_replacement(monkeypatch):
    class Database:
        def execute(self,*args): return self
        def fetchone(self): return None
    reader=object.__new__(f.MorganReader);reader.old=Database();reader.side=Database()
    with pytest.raises(ValueError,match='missing read-only'): reader.get('*CC*')


def test_budget_and_unapproved_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'OUT',tmp_path)
    actual=f.budget()
    assert all(actual[k]==0 for k in ('pt','krr','new_fp','ph','conformers','encoder_forward','formal_test'))
    unit=tmp_path/'adaptation/F0/NULL/eea/fold0';unit.mkdir(parents=True);f.write(unit/'config.json',{})
    limits=dict(f.LIMITS,adaptation_starts=0);monkeypatch.setattr(f,'LIMITS',limits)
    with pytest.raises(RuntimeError,match='budget'): f.budget()
    import scripts.run_ua2d_fp_film as entry
    monkeypatch.setattr('sys.argv',['run_ua2d_fp_film.py','--revision','r2','--round','F1','--stage','formal'])
    with pytest.raises(SystemExit): entry.main()


def test_package_rejects_mode_recipe_core_and_schema_changes():
    model=f.CachedFusion('FILM',head_state());mode='FILM';task='eea';fold=0
    core=str(f.base_folder(task,fold)/'best.pt')
    package=dict(config=dict(plan='UA2D-SOTA-20261018-01',revision='r2',round='F0',mode=mode,task=task,fold=fold,seed=42,recipe=f.RECIPE),
        feature_spec=f.FEATURE_SPEC,outer_test='NOT_RUN',best_epoch=0,no_adaptation_selected=True,frozen_components=['O8','readout','norm2'],
        encoder_weights_reference=core,metadata=dict(core=core,scaler=dict(mean=[2.],scale=[3.],fit='train',transform='standard')),state_dict=model.state_dict())
    restored=f.restore_package(package,mode=mode,task=task,fold=fold)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match='configuration'): f.restore_package(package,mode='SHIFT',task=task,fold=fold)
    bad=copy.deepcopy(package);bad['config']['recipe']['epochs']=51
    with pytest.raises(ValueError,match='configuration'): f.restore_package(bad,mode=mode,task=task,fold=fold)
    bad=copy.deepcopy(package);bad['encoder_weights_reference']='another-core.pt'
    with pytest.raises(ValueError,match='core'): f.restore_package(bad,mode=mode,task=task,fold=fold)
