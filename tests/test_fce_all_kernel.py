"""Small checks for the common FT stage and its complete seven-task contract."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.training import fce_kernel as kernel
from src.training import fce_param as ft
from scripts import run_fce_all_kernel as runner

SCALER=dict(mean=[3.],scale=[float(np.sqrt(5))],fit='train',transform='standard')


def entry(fold=0):
    test=[fold,(fold+1)%8];validation=[(fold+2)%8,(fold+3)%8]
    return dict(fold=fold,train_indices=[i for i in range(8) if i not in test+validation],validation_indices=validation,test_indices=test)


def reference(task='eea',fold=0):
    e=entry(fold);cfg=dict(task=task,arm='T3B_G05',seed='42+fold',parent='parent',parent_step=5000,
        recipe=dict(task=task,arm='R0'),training_config=dict(arm='T3B_G05',revision='r5',loss_weights=[1,.5,0],schedule_total_steps=20000))
    u=dict(e,task=task,arm='T3B_G05',seed=42+fold,config=cfg,parent='parent',best='best',head_initial='head',scaler=copy.deepcopy(SCALER),best_epoch=4)
    u['test_indices']=sorted(e['test_indices'])
    record=dict(config=cfg,status='PASS',outer_test='NOT_RUN',best_epoch=4)
    return u,record


def frozen():
    units=[]
    for task in kernel.TASKS:
        for fold in range(5):
            ref,_=reference(task,fold);trials=[dict(alpha=a,fused=dict(rmse=1.)) for a in kernel.ALPHAS]
            units.append(dict(entry(fold),task=task,sample_count=8,reference=ref,scaler=copy.deepcopy(SCALER),
                alpha=10.,validation_trials=trials,chosen_validation=trials[-1]))
    return dict(policy=copy.deepcopy(kernel.POLICY),units=units,frozen=True)


def example():
    train=[dict(original_row=i,sample_key=f'k{i}',normalized_smiles='*CC*',label=float(2*i)) for i in range(4)]
    val=[dict(original_row=i+4,sample_key=f'k{i+4}',normalized_smiles='*CO*',label=float(1+4*i)) for i in range(2)]
    ref=dict(indices=np.array([4,5]),sample_keys=np.array(['k4','k5']),y_true=np.array([1.,5.]),
        y_pred=np.array([1.2,4.8]),scaler_mean=np.array(SCALER['mean']),scaler_scale=np.array(SCALER['scale']))
    return train,val,ref


def test_public_finetune_stage_is_shared_by_all_seven_tasks(monkeypatch):
    assert ft.TASKS==kernel.TASKS==runner.TASKS
    assert ft.FINETUNE_PREDICTION_POLICY==kernel.POLICY
    assert ft.fit_finetune_predictions is runner.fit_finetune_predictions is kernel.fit_branch
    assert ft.predict_finetune_predictions is runner.predict_finetune_predictions is kernel.predict_branch
    monkeypatch.setattr(kernel,'fit_once',lambda k,z,a:np.full(len(z),a/10))
    for task in ft.TASKS:
        calls=[];trials=[];train,val,ref=example()
        result=ft.fit_finetune_predictions(train,val,ref,SCALER,on_fit=calls.append,
            on_trial=lambda a,v,m:trials.append((a,v,m)))
        assert calls==['attempted','completed']*3 and [t[0] for t in trials]==list(kernel.ALPHAS)
        assert result['chosen_validation']==kernel.choose_alpha(result['validation_trials'])
        assert all(np.array_equal(v['y_fused'],.5*v['y_g05']+.5*v['y_krr']) for _,v,_ in trials)


def test_dynamic_cohort_counts_and_reference_seed_schema():
    for task in ft.TASKS:
        for fold in range(5):
            e=entry(fold);kernel.validate_split(e,8);u,m=reference(task,fold)
            kernel.validate_reference(u,task,fold,e,SCALER,m)
            bad=copy.deepcopy(u);bad['config']['seed']=42+fold
            with pytest.raises(ValueError):kernel.validate_reference(bad,task,fold,e,SCALER,m)
    bad=entry();bad['test_indices'][0]=bad['train_indices'][0]
    with pytest.raises(ValueError):kernel.validate_split(bad,8)
    with pytest.raises(ValueError):kernel.validate_split(entry(),370)


def test_global_gate_requires_all_35_choices_and_same_policy():
    gate=kernel.PhaseGate()
    with pytest.raises(ValueError):gate.require_test()
    for change in (lambda p:p['units'].pop(),lambda p:p['policy']['feature_spec'].update(g05_weight=.7),
                   lambda p:p['units'][0].update(alpha=.1),lambda p:p['units'][0].update(task='unsupported')):
        p=frozen();change(p)
        with pytest.raises(ValueError):gate.freeze(p)
    p=frozen();gate.freeze(p);p['units'].clear();gate.require_test()


def test_source_and_reference_test_reads_obey_global_gate(tmp_path,monkeypatch):
    gate=kernel.PhaseGate();monkeypatch.setattr(runner,'SOURCE',tmp_path/'missing')
    with pytest.raises(ValueError,match='35'):
        runner.source_rows('eea',entry()['test_indices'],dict(sample_count=8,folds=[entry()]),0,'test',gate)


def test_reference_policy_scaler_best_and_fold_mismatch_are_rejected():
    for mutate in (lambda u:u.update(task='xc'),lambda u:u.update(seed=100),
                   lambda u:u['config']['recipe'].update(arm='G_LR'),
                   lambda u:u['config']['training_config'].update(loss_weights=[1,1,0])):
        u,m=reference();mutate(u)
        with pytest.raises(ValueError):kernel.validate_reference(u,'eea',0,entry(),SCALER,m)
    u,m=reference();m['best_epoch']=5
    with pytest.raises(ValueError):kernel.validate_reference(u,'eea',0,entry(),SCALER,m)
    u,m=reference()
    with pytest.raises(ValueError):kernel.validate_reference(u,'eea',0,entry(),dict(SCALER,mean=[9.]),m)


def test_equivalent_unit_reuse_never_fits_or_predicts_a_model(monkeypatch):
    train,val,ref=example();stored={}
    monkeypatch.setattr(kernel,'fit_once',lambda k,z,a:np.full(len(z),a/10))
    fitted=kernel.fit_branch(train,val,ref,SCALER,on_fit=lambda k:None,
        on_trial=lambda a,v,m:stored.update({a:(v,m)}))
    model=fitted.pop('model');binding=dict(task='ei')
    unit=dict(task='ei',fold=0,reference=binding,scaler=SCALER,train_indices=[0,1,2,3],validation_indices=[4,5],test_indices=[6,7])
    old=dict(unit,**fitted,model='/saved/selected_model.npz',fit_count=3)
    def read_npz(path):
        path=Path(path)
        return model if path.name=='selected_model.npz' else stored[float(path.parent.name.split('_')[1])][0]
    monkeypatch.setattr(runner,'load_npz',read_npz)
    monkeypatch.setattr(runner,'read',lambda path:stored[float(Path(path).parent.name.split('_')[1])][1])
    def forbidden(*args,**kwargs):raise AssertionError('reused units must not fit or run model prediction')
    monkeypatch.setattr(kernel,'fit_once',forbidden);monkeypatch.setattr(runner,'predict_finetune_predictions',forbidden)
    restored=runner.restore_unit(Path('/cached'),old,'OLD_FIXED_POLICY',unit,train,val,ref)
    assert restored['model']==old['model'] and restored['alpha']==old['alpha']
    bad=copy.deepcopy(old);bad['scaler']=dict(SCALER,mean=[0.])
    with pytest.raises(ValueError):runner.restore_unit(Path('/cached'),bad,'OLD_FIXED_POLICY',dict(unit,scaler=SCALER),train,val,ref)


def test_model_uses_own_training_identity_and_fixed_spec(monkeypatch):
    train,val,ref=example();monkeypatch.setattr(kernel,'fit_once',lambda k,z,a:np.full(len(z),a/10))
    r=kernel.fit_branch(train,val,ref,SCALER,on_fit=lambda k:None,on_trial=lambda *a:None)
    model=r.pop('model');u=dict(r,scaler=SCALER,train_indices=[0,1,2,3])
    kernel.validate_model(model,u)
    for key,value in [('train_keys',np.array(['wrong']*4)),('scale',np.float64(99)),('dual_coef',np.array([np.nan]*4)),('feature_spec',json.dumps(dict(kernel.FEATURE_SPEC,g05_weight=.8)))]:
        bad=dict(model,**{key:value})
        with pytest.raises(ValueError):kernel.validate_model(bad,u)


def test_budget_distinguishes_new_work_from_reuse_and_forbids_extra_tasks():
    b=dict(kernel.LIMITS);kernel.check_budget(b)
    for k in kernel.LIMITS:
        with pytest.raises(ValueError):kernel.check_budget(dict(b,**{k:kernel.LIMITS[k]+1}))
    with pytest.raises(ValueError):kernel.check_budget(dict(b,extra_task_fit=1))


def test_existing_outputs_are_preserved(tmp_path,monkeypatch):
    p=kernel.new_directory(tmp_path/'owned');(p/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):kernel.new_directory(p)
    monkeypatch.setattr(runner,'require_tmux',lambda:None);monkeypatch.setattr(runner,'require_environment',lambda:None)
    monkeypatch.setattr(runner,'OUT',p);monkeypatch.setattr(runner,'LOG',tmp_path/'no_logs')
    with pytest.raises(FileExistsError):runner.setup()
    assert (p/'sentinel').read_text()=='keep' and not (tmp_path/'no_logs').exists()
