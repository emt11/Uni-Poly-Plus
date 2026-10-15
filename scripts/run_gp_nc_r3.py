"""Single bounded Nc continuation: reuse GP/GU, validate five units, freeze before test."""
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.gp_sota_contract import OUT, LOG, PLAN, TASKS, read, ft_recipe, pretrain_recipe, validate_ft_scope
from src.training.glt_dual_runtime import require_tmux, save_checkpoint
from src.training.mcl_ph_direct_contract import assert_actual_equal
from scripts.run_mcl_ph_r4_fast import atomic_json, timestamp, write_shell, start_window
from scripts.run_gp_development_stage import audit
from scripts.check_gp_pretrain_package import finite

EVIDENCE = LOG / 'stop003_execution_r3_01'
STATE = OUT / 'nc_r3_status.json'
FROZEN = OUT / 'frozen_selection_r3.json'
CLAIM = OUT / 'nc_r3_writer.claim'


def unit(route, fold):
    return dict(route=route, task='nc', fold=fold, recipe='F0')


def key(u):
    return f"F0/{u['route']}/nc/fold{u['fold']}/seed{42+u['fold']}"


def package(route):
    return OUT / f'pretrain/{route}/attempt01/deploy_05000.pt'


def ledger(status, **extra):
    state = read(LOG / 'status.json')
    budgets = [read(p) for p in OUT.glob('**/budget.json')]
    dev = [b for b in budgets if b.get('kind') == 'finetune' and b.get('stage') == 'development']
    state['budgets'].update(complete_finetune_starts=len(dev),
        complete_finetune_epochs=sum(b['attempted_epochs'] for b in dev),
        complete_finetune_completed_epochs=sum(b['epochs'] for b in dev),
        complete_finetune_updates=sum(b['attempted_updates'] for b in dev),
        formal_test_only_units=len(list((OUT/'formal_r3').glob('*/*/*/fold*/test_predictions.npz'))))
    state.update(status=status, revision='r3', execution_status='NC_R3_'+status,
                 updated_at=timestamp(), release_for_training=status=='RUNNING',
                 release_scope='Nc r3 five fixed development units only', release_for_gp_a=False, **extra)
    atomic_json(LOG / 'status.json', state)
    atomic_json(STATE, dict(status=status, at=timestamp(), **extra))


def run(name, command, gpu=None, *, run_context=None, child_logs=()):
    log = LOG / f'{name}_01'
    write_shell(log, command, gpu=gpu)
    if run_context is not None:
        run_context.setdefault('child_logs', []).extend([log, *child_logs])
    start_window('gp_'+name, log)
    while not (log/'exit.code').exists():
        ledger('RUNNING', current_stage=name, current_log=str(log))
        time.sleep(10)
    code = int((log/'exit.code').read_text())
    if code:
        raise RuntimeError(f'{name} exit {code}; preserve {log}')
    return log


def scope(name, units):
    s = dict(plan=PLAN, revision='r3', configuration_revision='r2', continuation='NC_R3',
        status='LOCKED', name=name, units=units, allowed_units=[key(u) for u in units],
        packages={u['route']:str(package(u['route'])) for u in units})
    for u in units:
        validate_ft_scope('development', u['task'], u['fold'], u['route'], u['recipe'], s)
    return s


def stage(name, units, *, run_context=None):
    path = OUT/'scopes'/f'{name}.json'
    assert not path.exists() and not (OUT/'development'/name).exists()
    child_logs=[LOG/f"{name}_{u['route']}_{u['task']}_f{u['fold']}_01" for u in units]
    assert all(not p.exists() for p in child_logs), 'refuse existing child logs'
    atomic_json(path, scope(name, units))
    run(name, ['python','scripts/run_gp_development_stage.py','--scope',str(path)],
        run_context=run_context, child_logs=child_logs)
    rows = read(OUT/'development'/name/'report.json')['rows']
    assert len(rows) == len(units)
    return sorted(rows, key=lambda row:row['unit']['fold'])


def preflight():
    # Reject reentry before reading packages or acquiring any writing authority.
    assert not STATE.exists() and not FROZEN.exists() and not (OUT/'seven_task_results_r3.json').exists()
    assert not CLAIM.exists() and not (OUT/'nc_r3_locked_candidate.json').exists()
    assert read(EVIDENCE/'engineering_review.json')['status']=='PASS_SCOPED_ENGINEERING_REVIEW'
    for log in ['cpu_validation','engineering_review_run','scope_validation','scope_output_validation']:
        assert int((EVIDENCE/log/'exit.code').read_text())==0
    dev = [read(p) for p in OUT.glob('**/budget.json') if read(p).get('kind')=='finetune' and read(p).get('stage')=='development']
    assert len(dev)==8 and sum(b['attempted_updates'] for b in dev)==6160
    reused = []
    for route in ('GP','GU'):
        folder = package(route).parent
        a = read(folder/'acceptance.json'); r = read(folder/'runtime.json'); b = read(folder/'budget.json')
        assert a['status']=='PASS' and a['step']==5000 and a['evaluated_rows']==1024 and a['evaluation_calls_per_package']==1
        assert a['complete_encoder_exact'] and a['step0_205_exact'] and a['resume_finite']
        assert r['status']=='PASS' and r['executed_updates']==5000 and r['recipe']==pretrain_recipe(route)
        assert b['attempted_updates']==b['completed_updates']==5000
        saved = torch.load(folder/'resume_05000.pt', map_location='cpu', weights_only=False)
        deploy = torch.load(package(route), map_location='cpu', weights_only=False)
        finite(saved);finite(deploy)
        assert saved['step']==deploy['step']==5000 and deploy['route']==route
        assert saved['contract']['recipe']==pretrain_recipe(route)
        assert_actual_equal({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},deploy['state_dict'],'r3.preflight.'+route)
        reused.append(dict(route=route,acceptance=str(folder/'acceptance.json'),new_validation_calls=0))
        del saved, deploy
    old = [audit(OUT/'development/P3a/GP/nc'/f'fold{f}',unit('GP',f)) for f in (0,1)]
    assert all(row['glt_relative_gain']>0 and row['incumbent_relative_gain']>0 for row in old)
    for ref in ('glt','incumbent'):
        assert np.mean([row[ref+'_relative_gain'] for row in old])>=.01
    cap = sum(70*int(np.ceil(len(ft_recipe('nc',f,r,'F0')['split']['folds'][f]['train_indices'])/24)) for r,folds in [('GU',(0,1)),('GP',(2,3,4))] for f in folds)
    assert cap==3850
    return old, reused


def confirmation_gate(old, extra):
    checks = {}
    for ref in ('glt','incumbent'):
        q = np.array([r[ref+'_relative_gain'] for r in extra]);five = np.array([r[ref+'_relative_gain'] for r in old+extra])
        assert q.shape==(3,) and five.shape==(5,) and np.isfinite(q).all() and np.isfinite(five).all()
        checks[ref] = dict(passed=bool(q.mean()>=.01 and (q>0).sum()>=2 and q.min()>=-.02 and five.mean()>=.01),
            confirmation_gains=q.tolist(),five_fold_gains=five.tolist(),mean_confirmation_gain=float(q.mean()),
            mean_five_fold_gain=float(five.mean()),positive_confirmation_folds=int((q>0).sum()),worst_confirmation_gain=float(q.min()))
    return dict(status='PASS' if all(v['passed'] for v in checks.values()) else 'STOP_CONFIRMATION_GATE',gates=checks)


def freeze(rows):
    s = dict(plan=PLAN,revision='r3',configuration_revision='r2',continuation='NC_R3',status='FROZEN',
        frozen_at=timestamp(),allowed_units=[],unit_contracts={},seven_task_routes={},historical_fp_step0_exact='未核实')
    for t in TASKS:
        s['seven_task_routes'][t]=dict(kind='new_candidate',route='GP',recipe='F0',package=str(package('GP'))) if t=='nc' else dict(kind='reused_incumbent')
    oldrows=read(ROOT/'results/mcl_ph_r6_20261004/seven_task_results.json')['rows']
    for row in oldrows:
        if row['task']!='nc':s['seven_task_routes'][row['task']]['prediction_paths']=row['prediction_paths']
    for row in rows:
        u=row['unit'];folder=Path(row['folder']);m=read(folder/'metrics.json');saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        weights=OUT/'frozen_weights_r3/nc'/f"fold{u['fold']}.pt";weights.parent.mkdir(parents=True,exist_ok=True);assert not weights.exists()
        save_checkpoint(weights,dict(state_dict=saved['state_dict'],head_initial=saved['head_initial'],contract=saved['contract']))
        k=key(u);s['allowed_units'].append(k);s['unit_contracts'][k]=dict(contract=m['contract'],development_unit=str(folder),best_epoch=m['best_epoch'],
            scaler_mean=saved['scaler_mean'],scaler_scale=saved['scaler_scale'],scaler_train_raw=saved['scaler_train_raw'],frozen_weights=str(weights))
    for row in rows:
        u=row['unit'];validate_ft_scope('formal','nc',u['fold'],'GP','F0',s)
    assert not FROZEN.exists();atomic_json(FROZEN,s)


def summarize(paths, reason):
    old=read(ROOT/'results/mcl_ph_r6_20261004/seven_task_results.json');rows=[]
    for r in old['rows']:
        task=r['task'];new=task=='nc' and bool(paths);predictions=paths if new else r['prediction_paths'];values=[];errors=[]
        assert len(predictions)==5
        labels=pd.read_csv(ROOT/f'data/raw/smi_{task}.csv').iloc[:,1].to_numpy(dtype=float)
        for fold,p in enumerate(predictions):
            with np.load(p,allow_pickle=False) as z:
                y,zp=z['y_true'].astype(float).reshape(-1),z['y_pred'].astype(float).reshape(-1);ids=z['test_indices'].astype(int)
                assert np.isfinite(y).all() and np.isfinite(zp).all()
                np.testing.assert_allclose(y,labels[ids],rtol=2e-6,atol=2e-6)
                entry=ft_recipe(task,fold,'GP','F0')['split']['folds'][fold]
                assert ids.tolist()==sorted(entry['test_indices'])
                values.append(float(1-np.sum((y-zp)**2)/np.sum((y-y.mean())**2)));errors.append(float(np.sqrt(np.mean((y-zp)**2))))
                if new:assert int(z['seed'])==42+fold and int(z['fold'])==fold
            if new:
                m=read(Path(p).parent/'metrics.json');assert m['promotion_optimizer_updates']==0
                assert abs(m['test_r2']-values[-1])<1e-9 and abs(m['test_rmse']-errors[-1])<1e-9
                assert FROZEN.stat().st_mtime_ns<Path(p).stat().st_mtime_ns
            else:assert abs(values[-1]-r['selected_fold_r2'][fold])<1e-12
        mean=float(np.mean(values));rows.append(dict(task=task,result_kind='new_GP_formal' if new else 'reused_accepted_incumbent',
            new_gp_status='完成，待独立审查' if new else '未完成／不可比较：'+reason,prediction_paths=predictions,
            fold_r2=values,fold_rmse=errors,mean_r2=mean,std_r2=float(np.std(values)),paper_model=r['paper_model'],paper_best_r2=r['paper_best_r2'],project_minus_paper=mean-r['paper_best_r2']))
    report=dict(status='SELF_CHECK_PASS_PENDING_INDEPENDENT_REVIEW',plan=PLAN,revision='r3',rows=rows,
        fold_seed_protocol='paper5_outer; five-fold arithmetic mean; seed42+fold; std ddof0',paper_protocol_equivalence='未核实，仅数值对照',
        independent_blind_test=False,project_history_archived=False,original_r2_A_complete=False,stop_reason=reason)
    assert not (OUT/'seven_task_results_r3.json').exists();atomic_json(OUT/'seven_task_results_r3.json',report)
    return report


def main(*, run_context=None):
    require_tmux();torch.set_num_threads(1)
    old,reused=preflight()
    # Persist the claim; an interrupted attempt requires review, never auto-retry.
    CLAIM.mkdir(exist_ok=False)
    if run_context is not None:run_context['owned']=True
    candidate=OUT/'nc_r3_locked_candidate.json';assert not candidate.exists()
    atomic_json(candidate,dict(status='LOCKED',at=timestamp(),route='GP',recipe='F0',selection_uses='previous Nc fold0/1 validation only',
        GU_deploy_eligible=False,F1_enabled=False,reused_packages=reused,max_new_development_starts=5,max_new_epochs=350,max_new_updates=3850))
    ledger('RUNNING',current_stage='GU_NC_DIAGNOSTIC')
    gu=stage('R3_Nc_GU',[unit('GU',f) for f in (0,1)],run_context=run_context)
    diagnostic=[dict(fold=f,GP_rmse=g['best_validation_rmse'],GU_rmse=u['best_validation_rmse'],
        relative_GP_gain_vs_GU=(u['best_validation_rmse']-g['best_validation_rmse'])/u['best_validation_rmse']) for f,g,u in zip((0,1),old,gu)]
    atomic_json(OUT/'nc_r3_gu_diagnostic.json',dict(status='DIAGNOSTIC_ONLY',rows=diagnostic,deploy_eligible=False,
        mean_GP_relative_gain_vs_GU=float(np.mean([r['relative_GP_gain_vs_GU'] for r in diagnostic])),interpretation='two development folds; no stable mechanism or PH attribution'))
    extra=stage('R3_Nc_GP',[unit('GP',f) for f in (2,3,4)],run_context=run_context)
    gate=confirmation_gate(old,extra);atomic_json(OUT/'nc_r3_confirmation_gate.json',gate)
    paths=[]
    if gate['status']=='PASS':
        rows=old+extra;freeze(rows)
        for row in rows:
            f=row['unit']['fold'];folder=OUT/'formal_r3/GP/F0/nc'/f'fold{f}'
            run(f'R3_P6_nc_f{f}',['python','scripts/finetune_glt_mcl_ph.py','--task','nc','--fold',str(f),'--route','GP','--recipe','F0',
                '--stage','formal','--scope',str(FROZEN),'--checkpoint',str(package('GP')),'--reuse',row['folder'],'--output',str(folder)],gpu=f%4,run_context=run_context)
            paths.append(str(folder/'test_predictions.npz'))
    reason='原r2 A包停止；r3只确认Nc' if paths else 'Nc r3确认门槛未通过；其他任务原r2停止'
    report=summarize(paths,reason)
    new=[read(p) for name in ('R3_Nc_GU','R3_Nc_GP') for p in (OUT/'development'/name).glob('**/budget.json')]
    assert len(new)==5 and sum(b['attempted_epochs'] for b in new)==350 and sum(b['attempted_updates'] for b in new)==3850
    assert sum(b['completed_updates'] for b in new)==3850
    ledger('PENDING_REVIEW',current_stage='R3_NC_EXECUTION_FINISHED',plan_status='待审查（r3 Nc执行完成，原A包未完成）',
           r3_report=str(OUT/'seven_task_results_r3.json'),r3_confirmation_status=gate['status'],original_r2_A_complete=False)
    print(json.dumps(dict(status=report['status'],gate=gate,new_starts=5,new_epochs=350,new_updates=3850,new_test_units=len(paths)),ensure_ascii=False),flush=True)


def stop_r3_children(child_logs):
    # Exact commands from this owner's new logs; never scan by shared R3 paths.
    commands=set()
    for log in child_logs:
        p=Path(log)/'command.txt'
        if p.exists():commands.add(tuple(shlex.split(p.read_text())))
    for line in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()[1:]:
        pid,_,command=line.strip().partition(' ')
        try:matches=tuple(shlex.split(command)) in commands
        except ValueError:matches=False
        if matches:
            try:os.kill(int(pid),signal.SIGTERM)
            except ProcessLookupError:pass


def run_cli():
    run_context=dict(owned=False,child_logs=[])
    try:main(run_context=run_context)
    except BaseException as error:
        if run_context['owned']:
            try:stop_r3_children(run_context['child_logs'])
            except Exception as stop_error:print(f'owned child stop unavailable: {stop_error}',file=sys.stderr,flush=True)
            try:ledger('BLOCKED',plan_status='阻断（r3停止，现场保留）',error=f'{type(error).__name__}: {error}',automatic_retry=False)
            except Exception as ledger_error:print(f'owned failure ledger unavailable: {ledger_error}',file=sys.stderr,flush=True)
        raise


if __name__=='__main__':run_cli()
