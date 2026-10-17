"""r11 correctness fixtures; no real-data fit or repeated prior test suite."""
import copy
import json
import numpy as np
import pytest
from src.training import fce_kernel as k
from src.training import fce_param as public
from scripts import run_fce_aligned_alpha as runner


def unit(task='eea',fold=0):
    scale=float(np.std([1.,2.,3.]))
    samples=lambda ids:[dict(original_row=i,sample_key=f'key{i}',normalized_smiles='*C*') for i in ids]
    return dict(task=task,fold=fold,sample_count=7,alpha=.1,train_indices=[0,1,2],validation_indices=[3,6],test_indices=[4,5],
        train_samples=samples([0,1,2]),validation_samples=samples([3,6]),scaler=dict(mean=[2.],scale=[scale],fit='train',transform='standard'),reference=dict(scaler=dict(mean=[2.],scale=[scale],fit='train',transform='standard')))


def caches(u):
    identities={str(i):dict(sample_key=f'key{i}',normalized_smiles='*C*') for i in range(7)}
    result={}
    for alpha,branch in zip(k.ALPHAS,([1.,1.],[3.,-1.],[6.,-4.])):
        result[alpha]=dict(indices=[3,6],sample_keys=['key3','key6'],y_true=[0.,2.],y_g05=[-1.,3.],y_krr=branch,
            alpha=alpha,scaler_mean=u['scaler']['mean'],scaler_scale=u['scaler']['scale'])
    return result,identities


def model(u):
    bits=np.zeros((3,2048),dtype=np.uint8);bits[0,0]=bits[1,1]=bits[2,2]=1
    return dict(alpha=np.float64(u['alpha']),dual_coef=np.ones(3),train_bits=bits,train_y=np.array([1.,2.,3.]),train_indices=np.array([0,1,2]),
        train_keys=np.array(['key0','key1','key2']),train_normalized_smiles=np.array(['*C*']*3),mean=np.float64(2.),scale=np.float64(u['scaler']['scale'][0]),feature_spec=np.asarray(json.dumps(k.FEATURE_SPEC)))


def r10_fixture():
    trials=[dict(alpha=a,fused=dict(rmse=1. if a==.1 else 2.)) for a in k.ALPHAS]
    prior=[]
    for t in k.TASKS:
        for f in range(5):
            u=unit(t,f);u.update(validation_trials=trials,chosen_validation=trials[0]);prior.append(u)
    scores=[dict(weight=w,mean_validation_r2=(1. if w==.75 else 0.),units=[dict(task=u['task'],fold=u['fold'],r2=1. if w==.75 else 0.) for u in prior]) for w in k.GLOBAL_WEIGHTS]
    return dict(policy=copy.deepcopy(k.GLOBAL_BLEND_POLICY),weight=.75,scores=scores,frozen=True,r9_freeze=dict(policy=copy.deepcopy(k.POLICY),frozen=True,units=prior))


def aligned_fixture():
    old=r10_fixture();units=[]
    for prior in old['r9_freeze']['units']:
        p,identity=caches(prior);choice=public.select_aligned_finetune_alpha(prior,p,identity)
        units.append(dict(prior,**choice,model_ready=True))
    return dict(policy=copy.deepcopy(k.ALIGNED_ALPHA_POLICY),r10_freeze=old,frozen=True,units=units,K=35)


def test_common_075_selection_differs_from_old_half_and_exact_tie():
    u=unit();p,identity=caches(u);choice=public.select_aligned_finetune_alpha(u,p,identity)
    assert choice['old_alpha']==.1 and choice['alpha']==1. and choice['changed']
    half=[dict(alpha=a,fused=k.metrics(p[a]['y_true'],k.blend_arrays(p[a]['y_g05'],p[a]['y_krr'],.5))) for a in k.ALPHAS]
    assert k.choose_alpha(half)['alpha']==.1
    for v in p.values():v['y_krr']=[3.,-1.]
    assert public.select_aligned_finetune_alpha(u,p,identity)['alpha']==10.
    assert public.ALIGNED_FINETUNE_POLICY['alpha_selection_weight']==.75
    assert k.FEATURE_SPEC['g05_weight']==.5


def test_cache_own_alpha_dynamic_order_and_reject_mismatch():
    u=unit();p,identity=caches(u)
    p[1.]=dict(p[1.],indices=[6,3],sample_keys=['key6','key3'],y_true=[2.,0.],y_g05=[3.,-1.],y_krr=[-1.,3.])
    assert public.select_aligned_finetune_alpha(u,p,identity)['alpha']==1.
    for change in (dict(alpha=.1),dict(sample_keys=['wrong','key3']),dict(scaler_scale=[9.]),dict(y_true=[99.,0.]),dict(y_g05=[9.,9.])):
        bad=copy.deepcopy(p);bad[1.].update(change)
        with pytest.raises(ValueError):public.select_aligned_finetune_alpha(u,bad,identity)


def test_unchanged_no_fit_no_features_no_predict(monkeypatch):
    def forbidden(*args,**kw):raise AssertionError('unchanged work forbidden')
    for name in ('fit_once','generator','fingerprint','predict_dual'):monkeypatch.setattr(k,name,forbidden)
    u=unit();saved=model(u);got,origin=public.restore_aligned_finetune_model(saved,u,.1,on_fit=forbidden)
    assert got is saved and origin=='unchanged'
    gate=k.AlignedAlphaGate();gate.payload={}
    with pytest.raises(ValueError,match='unchanged'):public.predict_aligned_finetune_predictions(saved,dict(u,changed=False),{}, {},gate,on_query=forbidden)


def test_changed_fit_once_selected_alpha_original_train_only(monkeypatch):
    u=unit();saved=model(u);calls=[];events=[]
    def fake_fit(kernel,z,alpha):
        calls.append(alpha);assert kernel.shape==(3,3)
        np.testing.assert_allclose(z,([-1.,0.,1.]/np.asarray(u['scaler']['scale'][0])))
        return np.array([.1,.2,.3])
    monkeypatch.setattr(k,'fit_once',fake_fit)
    monkeypatch.setattr(k,'generator',lambda:pytest.fail('no training Morgan generation'))
    result,origin=public.restore_aligned_finetune_model(saved,u,1.,on_fit=events.append)
    assert calls==[1.] and events==['attempted','completed'] and origin=='fitted'
    np.testing.assert_equal(result['train_y'],saved['train_y']);np.testing.assert_equal(result['train_indices'],[0,1,2])
    assert json.loads(str(result['feature_spec']))==k.FEATURE_SPEC
    bad=model(u);bad['train_y']=np.array([1.,2.,90.])
    with pytest.raises(ValueError):public.restore_aligned_finetune_model(bad,u,1.,on_fit=events.append)


def test_explicit_coefficients_reuse_and_scaler_column_rejection():
    u=unit();saved=model(u);candidate=dict(saved,alpha=np.float64(1.))
    def forbidden(*a):raise AssertionError('cached coefficient refit')
    got,origin=public.restore_aligned_finetune_model(saved,u,1.,on_fit=forbidden,cached_models=[candidate])
    assert got is candidate and origin=='reused_coefficients'
    for field,value in [('train_indices',np.array([1,0,2])),('mean',np.float64(99.)),('train_y',np.array([3.,2.,1.]))]:
        bad=dict(candidate);bad[field]=value
        with pytest.raises(ValueError):public.restore_aligned_finetune_model(saved,u,1.,on_fit=forbidden,cached_models=[bad])


def test_all_35_models_frozen_before_test_and_no_task_weight(tmp_path):
    gate=k.AlignedAlphaGate()
    with pytest.raises(ValueError):runner.load_npz(tmp_path/'missing.npz',gate,True)
    good=aligned_fixture();gate.freeze(good);assert gate.payload['K']==35
    for mutate in (lambda p:p['units'].pop(),lambda p:p['units'][0].update(prediction_weight=.5),lambda p:p['units'][0].update(model_ready=False),lambda p:p.update(K=0)):
        bad=aligned_fixture();mutate(bad)
        with pytest.raises(ValueError):k.AlignedAlphaGate().freeze(bad)
    with pytest.raises(FileNotFoundError):runner.load_npz(tmp_path/'missing.npz',gate,True)


def test_k_zero_and_changed_only_budget_and_output_conflicts(tmp_path):
    b={key:0 for key in runner.LIMITS};b.update(validation_scores=105,selected_units=35,reused_models=35,reused_formal=35);runner.check_budget(b)
    for key in ('fit_attempts','formal_starts','new_outputs'):
        bad=dict(b);bad[key]=1
        with pytest.raises(ValueError):runner.check_budget(bad)
    b.update(K=2,reused_models=33,reused_formal=33,fit_attempts=2,fit_completed=2,formal_starts=2,formal_completed=2,new_outputs=4);runner.check_budget(b)
    for key in runner.LIMITS:
        bad=dict(b);bad[key]=runner.LIMITS[key]+1
        with pytest.raises(ValueError):runner.check_budget(bad)
    p=k.new_directory(tmp_path/'owned')
    with pytest.raises(FileExistsError):k.new_directory(p)
