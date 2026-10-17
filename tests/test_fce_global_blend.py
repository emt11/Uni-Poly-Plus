"""r10 small array fixtures only: no old tests, fitting or model inference."""
import copy
import numpy as np
import pytest
from src.training import fce_kernel as kernel
from src.training import fce_param as public
from scripts import run_fce_global_blend as runner


def validation():
    return [dict(task=t,fold=f,y_true=np.array([0.,1.,2.]),g05=np.array([0.,1.,2.]),krr=np.array([2.,1.,0.])) for t in kernel.TASKS for f in range(5)]


def test_one_global_validation_objective_and_tie():
    rows=validation();selected=public.select_global_finetune_weight(rows)
    assert selected['weight']==1. and len(selected['scores'])==5
    for r in rows:r['krr']=r['g05'].copy()
    assert public.select_global_finetune_weight(rows)['weight']==1.
    with pytest.raises(ValueError):public.select_global_finetune_weight(rows[:-1])
    with pytest.raises(ValueError):public.select_global_finetune_weight(rows[:-1]+rows[:1])


def test_equal_units_not_pooled_sample_count():
    rows=validation()
    for r in rows[:30]:r['y_true']=np.tile(r['y_true'],3);r['g05']=np.tile(r['g05'],3);r['krr']=np.tile(r['krr'],3)
    for r in rows[30:]:r['g05'],r['krr']=r['krr'],r['g05']
    selected=public.select_global_finetune_weight(rows)
    for score in selected['scores']:
        expected=np.mean([kernel.metrics(r['y_true'],kernel.blend_arrays(r['g05'],r['krr'],score['weight']))['r2'] for r in rows])
        assert score['mean_validation_r2']==expected


def old_freeze():
    trial=[dict(alpha=a,fused=dict(rmse=1.)) for a in kernel.ALPHAS]
    units=[]
    for t in kernel.TASKS:
        for f in range(5):units.append(dict(task=t,fold=f,sample_count=4,train_indices=[0],validation_indices=[1],test_indices=[2,3],alpha=10.,validation_trials=trial,chosen_validation=trial[-1],scaler={'same':True},reference=dict(scaler={'same':True})))
    return dict(policy=copy.deepcopy(kernel.POLICY),frozen=True,units=units)


def frozen(weight=None):
    selected=public.select_global_finetune_weight(validation())
    if weight is not None:selected['weight']=weight
    return dict(selected,frozen=True,r9_freeze=old_freeze())


def test_freeze_blocks_test_and_rejects_task_overrides(tmp_path):
    gate=kernel.GlobalBlendGate()
    with pytest.raises(ValueError):runner.load_npz(tmp_path/'missing.npz',gate,test=True)
    with pytest.raises(ValueError):public.blend_finetune_arrays([1.,2.],[3.,4.],gate)
    with pytest.raises(ValueError):gate.freeze(frozen(.5))
    bad=frozen();bad['r9_freeze']['units'].pop()
    with pytest.raises(ValueError):gate.freeze(bad)
    good=frozen();gate.freeze(good);good['weight']=0.
    np.testing.assert_equal(public.blend_finetune_arrays([1.,2.],[3.,4.],gate),[1.,2.])
    with pytest.raises(FileNotFoundError):runner.load_npz(tmp_path/'missing.npz',gate,test=True)
    with pytest.raises(TypeError):public.blend_finetune_arrays([1.,2.],[3.,4.],gate,task='xc')


def test_binding_dynamic_order_sample_scaler_alpha_and_finite():
    unit=dict(validation_indices=[5,3],test_indices=[3,5],alpha=1.,scaler=dict(mean=[2.],scale=[.5],fit='train',transform='standard'))
    identities={'3':dict(sample_key='a'),'5':dict(sample_key='b')}
    values=dict(indices=[3,5],sample_keys=['a','b'],y_true=[2.,4.],y_pred=[1.,3.],scaler_mean=[2.],scaler_scale=[.5],alpha=1.)
    result=kernel.bind_branch_arrays(values,unit,'validation',identities,('y_pred',));np.testing.assert_equal(result['y_pred'],[3.,1.])
    for altered in (dict(indices=[3,3]),dict(sample_keys=['wrong','b']),dict(scaler_scale=[1.]),dict(alpha=10.),dict(y_pred=[np.nan,1.])):
        with pytest.raises((ValueError,FloatingPointError)):kernel.bind_branch_arrays(dict(values,**altered),unit,'validation',identities,('y_pred',))
    with pytest.raises(ValueError):kernel.bind_branch_arrays(values,unit,'train',identities,('y_pred',))


def test_public_array_interface_does_not_fit_or_infer(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('fit/feature/model replay forbidden')
    for name in ('fit_branch','predict_branch','fit_once','predict_dual','generator','fingerprint'):monkeypatch.setattr(kernel,name,forbidden)
    selected=public.select_global_finetune_weight(validation());gate=kernel.GlobalBlendGate();gate.freeze(dict(selected,frozen=True,r9_freeze=old_freeze()))
    np.testing.assert_equal(public.blend_finetune_arrays([0.,1.],[2.,3.],gate),[0.,1.])
    assert public.GLOBAL_FINETUNE_BLEND_POLICY==kernel.GLOBAL_BLEND_POLICY


def test_budget_zero_fit_and_reuse_limits():
    b={k:0 for k in runner.LIMITS};b.update(validation_units=35,validation_scores=175,formal_units=35,reused_outputs=35,final_cpu_starts=1);runner.check_budget(b)
    for key in runner.LIMITS:
        bad=dict(b);bad[key]=runner.LIMITS[key]+1
        with pytest.raises(ValueError):runner.check_budget(bad)
    bad=dict(b,new_outputs=1)
    with pytest.raises(ValueError):runner.check_budget(bad)


def test_grid_endpoints_and_existing_output_rejection(tmp_path):
    a=np.array([1.,2.]);b=np.array([4.,6.])
    np.testing.assert_equal(kernel.blend_arrays(a,b,0.),b)
    np.testing.assert_equal(kernel.blend_arrays(a,b,.5),.5*a+.5*b)
    np.testing.assert_equal(kernel.blend_arrays(a,b,1.),a)
    with pytest.raises(ValueError):kernel.blend_arrays(a,b,.3)
    folder=kernel.new_directory(tmp_path/'existing')
    with pytest.raises(FileExistsError):kernel.new_directory(folder)
