"""r8-specific small fixtures: Ei bindings, global freeze and budget isolation."""
import copy
import json
import numpy as np
import pytest
from src.training.ei_kernel import (ALPHAS,CANDIDATE,FEATURE_SPEC,LIMITS,PhaseGate,
    align_reference,check_budget,choose_alpha,new_directory,read_predictions,
    validate_freeze,validate_reference_binding,validate_reference_protocol,validate_split,collect_sample_identity)
from scripts import run_fce_ei_kernel as runner

SCALER=dict(mean=[6.],scale=[.9],fit='train',transform='standard')


def entry(fold=0):
    return dict(fold=fold,train_indices=list(range(236)),validation_indices=list(range(236,296)),test_indices=list(range(296,370)))


def reference(fold=0):
    e=entry(fold)
    config=dict(arm='T3B_G05',task='ei',parent='g05_parent',parent_step=5000,seed='42+fold',
        recipe=dict(arm='R0',task='ei'),training_config=dict(arm='T3B_G05',revision='r5',loss_weights=[1,.5,0],schedule_total_steps=20000))
    return dict(e,arm='T3B_G05',task='ei',config=config,seed=42+fold,parent='g05_parent',head_initial='ei_head',best='ei_best',best_epoch=4,scaler=copy.deepcopy(SCALER))


def validation_metrics(ref):
    return dict(status='PASS',config=ref['config'],best_epoch=4,outer_test='NOT_RUN')


def binding(ref=None,e=None,scaler=None,metrics=None):
    ref=reference() if ref is None else ref
    validate_reference_binding(ref,0,entry() if e is None else e,SCALER if scaler is None else scaler,
        validation_metrics(ref) if metrics is None else metrics,parent='g05_parent',head='ei_head',best='ei_best')


def test_ei_reference_task_parent_head_best_seed_isolation():
    binding()
    for field,value in [('task','xc'),('arm','G05_GLR'),('parent','t2_parent'),('head_initial','xc_head'),('best','other_best'),('seed',43),('fold',1)]:
        r=reference();r[field]=value
        with pytest.raises(ValueError):binding(ref=r)
    for field,value in [('task','xc'),('parent','t2_parent'),('seed',43),('parent_step',2500)]:
        r=reference();r['config'][field]=value
        with pytest.raises(ValueError):binding(ref=r)
    r=reference();r['config']['recipe']['arm']='T2_G_LR'
    with pytest.raises(ValueError):validate_reference_protocol(r)
    r=reference();r['config']['training_config']['loss_weights']=[1,1,0]
    with pytest.raises(ValueError):validate_reference_protocol(r)


def test_seed_rule_and_resolved_fold_seeds_are_distinct():
    for fold in range(5):
        r=reference(fold)
        validate_reference_binding(r,fold,entry(fold),SCALER,validation_metrics(r),
            parent='g05_parent',head='ei_head',best='ei_best')
        r['config']['seed']=42+fold
        with pytest.raises(ValueError):
            validate_reference_binding(r,fold,entry(fold),SCALER,validation_metrics(r),
                parent='g05_parent',head='ei_head',best='ei_best')


def test_ei_236_60_74_split_and_train_scaler_metadata():
    validate_split(entry());binding()
    for role in ('train','validation','test'):
        e=entry();e[role+'_indices'].pop()
        with pytest.raises(ValueError):validate_split(e)
    e=entry();e['test_indices'][0]=0
    with pytest.raises(ValueError):validate_split(e)
    e=entry();e['train_indices'].reverse()
    with pytest.raises(ValueError):binding(e=e)
    with pytest.raises(ValueError):binding(scaler=dict(SCALER,scale=[1.]))
    r=reference();r['scaler']['fit']='train+validation'
    with pytest.raises(ValueError):binding(ref=r,scaler=r['scaler'])
    m=validation_metrics(reference());m['best_epoch']=5
    with pytest.raises(ValueError):binding(metrics=m)


def test_global_identity_includes_rows_never_in_any_training_fold():
    row=lambda i:dict(original_row=i,sample_key=f'key{i}',normalized_smiles='*C*')
    units=[dict(train_samples=[row(0)],validation_samples=[row(29)]) for _ in range(5)]
    assert set(collect_sample_identity(units))=={'0','29'}
    units[1]['validation_samples'][0]['sample_key']='conflicting'
    with pytest.raises(ValueError):collect_sample_identity(units)


def test_selection_is_fused_validation_and_exact_tie_not_branch_or_test():
    trials=[dict(alpha=a,fused=dict(rmse=1.),branch=dict(rmse=1/a),test=dict(rmse=a)) for a in ALPHAS]
    assert choose_alpha(trials)['alpha']==10.
    trials[0]['fused']['rmse']=np.nextafter(1.,0.)
    assert choose_alpha(trials)['alpha']==.1


def frozen():
    identity={str(i):dict(sample_key=f'key{i}',normalized_smiles='*C*') for i in range(370)}
    units=[]
    for f in range(5):
        r=reference(f);trials=[dict(alpha=a,fused=dict(rmse=1.)) for a in ALPHAS]
        u=dict(entry(f),task='ei',alpha=10.,fit_count=3,scaler=copy.deepcopy(SCALER),reference=r,
            validation_trials=trials,chosen_validation=trials[-1])
        for role in ('train','validation'):
            u[role+'_samples']=[dict(original_row=i,**identity[str(i)]) for i in u[role+'_indices']]
        units.append(u)
    return dict(candidate=CANDIDATE,task='ei',feature_spec=copy.deepcopy(FEATURE_SPEC),frozen=True,units=units,sample_identity=identity)


def test_all_five_frozen_before_test_read_or_source_access(tmp_path,monkeypatch):
    gate=PhaseGate()
    with pytest.raises(ValueError,match='before all five'):read_predictions(tmp_path/'never_opened.npz',gate=gate,test=True)
    monkeypatch.setattr(runner,'SOURCE',tmp_path/'no_source')
    with pytest.raises(ValueError,match='before all five'):runner.source_rows(entry()['test_indices'],dict(folds=[entry()]),0,'test',gate)
    bad=frozen();bad['units'].pop()
    with pytest.raises(ValueError):gate.freeze(bad)
    good=frozen();gate.freeze(good);good['units'].clear();gate.require_test()
    with pytest.raises(FileNotFoundError):read_predictions(tmp_path/'never_opened.npz',gate=gate,test=True)


def test_ei_freeze_rejects_xc_or_modified_selection_identity_scaler():
    validate_freeze(frozen())
    for mutate in (
        lambda p:p.update(candidate='XC_G05_MKRR50'),
        lambda p:p.update(task='xc'),
        lambda p:p['units'][0].update(alpha=.1),
        lambda p:p['units'][0]['train_samples'][0].update(sample_key='wrong'),
        lambda p:p['units'][0]['reference'].update(task='xc'),
        lambda p:p['units'][0]['scaler'].update(mean=[9.]),
        lambda p:p['sample_identity'].pop('369'),
    ):
        bad=frozen();mutate(bad)
        with pytest.raises(ValueError):validate_freeze(bad)


def test_ei_reference_prediction_identity_and_scaler():
    rows=[dict(original_row=1,sample_key='ei_1',label=6.)]
    pred=dict(indices=[1],sample_keys=['ei_1'],y_true=[6.],y_pred=[6.1],scaler_mean=[6.],scaler_scale=[.9])
    align_reference(pred,rows,SCALER)
    for altered in (dict(indices=[2]),dict(sample_keys=['xc_1']),dict(scaler_mean=[9.]),dict(y_true=[9.])):
        with pytest.raises(ValueError):align_reference(dict(pred,**altered),rows,SCALER)


def test_ei_budget_matrix_and_no_neural_or_other_task_work():
    b={k:0 for k in LIMITS};b.update(fit_attempts=15,fit_completed=15,formal_starts=5,formal_completed=5,prediction_groups=10,final_cpu_starts=1);check_budget(b)
    for field in LIMITS:
        bad=dict(b);bad[field]=LIMITS[field]+1
        with pytest.raises(ValueError):check_budget(bad)
    bad=dict(b,xc_fit=1)
    with pytest.raises(ValueError):check_budget(bad)


def test_saved_fit_recovery_keeps_consumed_budget_and_rejects_formal_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'SAVED_FITS',tmp_path)
    consumed={k:0 for k in LIMITS};consumed.update(fit_attempts=15,fit_completed=15)
    path=tmp_path/'consumption.json';path.write_text(json.dumps(consumed))
    assert runner.saved_fit_budget()==consumed
    for changed in (dict(consumed,fit_completed=14),dict(consumed,formal_starts=1)):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):runner.saved_fit_budget()
    path.write_text(json.dumps(consumed));(tmp_path/'selection_before_test.json').write_text('{}')
    with pytest.raises(ValueError):runner.saved_fit_budget()


def test_existing_outputs_and_roots_are_not_replayed(tmp_path,monkeypatch):
    p=new_directory(tmp_path/'owned');(p/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):new_directory(p)
    assert (p/'sentinel').read_text()=='keep'
    monkeypatch.setattr(runner,'require_tmux',lambda:None)
    monkeypatch.setattr(runner,'require_environment',lambda:None)
    monkeypatch.setattr(runner,'OUT',p);monkeypatch.setattr(runner,'LOG',tmp_path/'new_logs')
    with pytest.raises(FileExistsError):runner.setup()
    assert not (tmp_path/'new_logs').exists()
