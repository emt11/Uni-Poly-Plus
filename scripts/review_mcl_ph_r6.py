"""Independent CPU review using saved predictions and direct source labels."""
from pathlib import Path
import json,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from src.training.glt_dual_runtime import open_source
torch.set_num_threads(1)
OUT=ROOT/'results/mcl_ph_r6_20261004'
LOG=ROOT/'logs/mcl_ph_r6_20261004'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
R4=ROOT/'results/mcl_ph_r4_20261002/fast_campaign_04'
PLAN='MCL-PH-NEXT-20261004-02'
def read(p):return json.loads(Path(p).read_text())
def same(a,b):
    assert a.keys()==b.keys()
    for k in a:
        assert a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and torch.equal(a[k],b[k]),k

status=read(LOG/'campaign_01/status.json');assert status['status']=='COMPLETED_PENDING_REVIEW'
assert not status['running_units'] and status['pending_units']==0
frozen=read(OUT/'frozen_selection.json');assert frozen['status']=='FROZEN'
passed=frozen['gates']['xc']['passed']
units=read(LOG/'campaign_01/units.json')
assert len(units)==(7 if passed else 2) and sum(u['reused'] for u in units)==(2 if passed else 0)
assert sum(not u['reused'] for u in units)==(5 if passed else 2)
complete_epochs=sum(u['epochs'] for u in units if not u['reused'])
complete_updates=sum(u['optimizer_updates'] for u in units)
assert complete_epochs==(350 if passed else 140) and complete_updates==(4200 if passed else 1680)
assert complete_epochs<=350 and complete_updates<=4200
report=read(OUT/'seven_task_results.json');old_report=read(ROOT/'results/mcl_ph_r5_20261004/seven_task_results.json')
cohorts={}
for row in report['rows']:
    task=row['task'];manifest=read(ROOT/f'data/splits/mips_paper5_outer/{task}.json')
    raw=pd.read_csv(ROOT/f'data/raw/smi_{task}.csv').iloc[:,1].to_numpy(dtype=np.float64)
    source,frame=open_source(SOURCE,SOURCE,task=task,selected_indices=list(range(len(raw))),
        dual_static_root=SOURCE,expected_task_rows=len(raw),record_index_path=SOURCE/f'{task}_offsets.npy',split_manifest=manifest)
    try:
        assert frame['original_row'].tolist()==list(range(len(raw)))
        np.testing.assert_array_equal(raw,frame['label'].to_numpy(dtype=np.float64))
        cohorts[task]=(manifest,raw,[k.hex() for k,_ in source.samples])
    finally:source.close()
reviewed=[]
smoke=OUT/'smoke/xc_standard/glt_ref/xc/fold0'
for unit in units+[dict(folder=str(smoke),phase='smoke',task='xc',arm='glt_ref',fold=0,reused=False,
        log=str(LOG/'units/smoke/xc_standard/fold0'))]:
    folder=Path(unit['folder']);m=read(folder/'metrics.json');r=read(folder/'runtime.json')
    assert r['status']=='PASS' and r['exit_code']==0
    assert (Path(unit['log'])/'exit.code').read_text().strip()=='0'
    assert m['experiment_plan']==PLAN and m['target_transform']=='standard' and m['scaler_fit_split']=='train'
    assert m['source_root']==str(SOURCE) and m['pretrain_step']==5000
    assert m['fold']==unit['fold'] and m['task']==unit['task'] and m['stage']==unit['phase']
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    assert checkpoint['config']==read(m['config_path'])
    for k in ['target_transform','experiment_plan','best_epoch','best_validation_rmse','optimizer_updates','executed_epochs']:
        assert checkpoint[k]==m[k]
    recipe=read(OUT/'candidate.json')
    assert recipe['backbone_joint_lr']==1e-4 and recipe['head_initial_lr']==3e-4 and recipe['head_joint_lr']==1e-3
    assert m['learning_rate_table']==dict(head_stage1=3e-4,backbone_stage2=1e-4,head_stage2=1e-3)
    assert checkpoint['training_contract']==m['training_contract']
    contract=m['training_contract'];assert contract['recipe']==recipe and contract['seed']==42+m['fold']
    assert contract['config']==checkpoint['config']==read(ROOT/'results/mcl_ph_r4_20261001/resolved/glt_ref.json')
    assert contract['split']==read(ROOT/'data/splits/mips_paper5_outer/xc.json')
    assert contract['source_metadata']==read(SOURCE/'direct_source.json')
    assert contract['checkpoint']==m['pretrain_package_path']==str(ROOT/'results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/deploy_05000.pt')
    assert m['weight_decay']==1e-4
    assert recipe['precision']=='fp32' and recipe['batch_size']==24 and recipe['head_dropout']==.3
    assert recipe['head_epochs']==10 and recipe['joint_epochs']==60
    assert recipe['head_scheduler']==dict(T_max=10,eta_min=3e-5) and recipe['joint_scheduler']==dict(T_0=10,T_mult=1,eta_min=1e-6)
    cmd=read(folder/'runtime.json')['command']
    assert cmd[cmd.index('--checkpoint')+1]==contract['checkpoint']
    assert cmd[cmd.index('--candidate-config')+1]==str(OUT/'candidate.json')
    assert all(x['training_steps']==12 for x in m['history'])
    for item in m['history']:
        e=item['stage_epoch'];factor=(1+np.cos(np.pi*(e if item['stage']=='head' else e%10)/10))/2
        lrs=[3e-5+(3e-4-3e-5)*factor] if item['stage']=='head' else [1e-6+(v-1e-6)*factor for v in (1e-4,1e-3)]
        np.testing.assert_allclose(item['learning_rates'],lrs,rtol=1e-12,atol=1e-15)
    if unit['phase']=='paper5_outer':
        assert datetime.fromisoformat(frozen['frozen_at']) < datetime.fromisoformat(unit['started_at'])
        assert frozen['unit_contracts'][f"glt_ref/xc/fold{m['fold']}"]['training_contract']==contract
    manifest,labels,keys=cohorts[m['task']];f=manifest['folds'][m['fold']];train=f['train_indices']
    np.testing.assert_allclose(checkpoint['scaler_mean'],[labels[train].mean()],rtol=0,atol=1e-12)
    np.testing.assert_allclose(checkpoint['scaler_scale'],[labels[train].std()],rtol=0,atol=1e-12)
    history=m['history'];best=min(history,key=lambda x:x['validation_rmse'])
    assert best['epoch']==m['best_epoch'] and best['validation_rmse']==m['best_validation_rmse']
    assert sum(x['training_steps'] for x in history)==m['optimizer_updates']
    assert [x['stage'] for x in history]==(['head','joint'] if unit['phase']=='smoke' else ['head']*10+['joint']*60)
    initial=torch.load(folder/'head_initial.pt',map_location='cpu',weights_only=False)
    base=R4/f"paper5_outer/glt_ref/xc/fold{m['fold']}" if m['task']=='xc' else R4/'development/m_xattn/nc/fold0'
    same(initial,torch.load(base/'head_initial.pt',map_location='cpu',weights_only=False))
    for role in ['validation','test'] if unit['phase']=='paper5_outer' else ['validation']:
        indices=f['validation_indices'] if role=='validation' else sorted(f['test_indices'])
        with np.load(folder/f'{role}_predictions.npz',allow_pickle=False) as data:
            y,z=[data[k].reshape(-1) for k in ['y_true','y_pred']]
            assert data[f'{role}_indices'].tolist()==indices and data['sample_keys'].tolist()==[keys[i] for i in indices]
            assert np.isfinite(y).all() and np.isfinite(z).all()
            np.testing.assert_allclose(y,labels[indices],rtol=2e-6,atol=2e-6)
            rmse=float(np.sqrt(np.mean((y-z)**2)))
            np.testing.assert_allclose(rmse,m['best_validation_rmse' if role=='validation' else 'test_rmse'],rtol=1e-6,atol=1e-7)
            if role=='test':
                assert (folder/'test_predictions.npz').stat().st_mtime>=datetime.fromisoformat(frozen['frozen_at']).timestamp()
    if unit['phase']!='paper5_outer':
        assert m['outer_test']=='NOT_RUN' and not (folder/'test_predictions.npz').exists()
    if unit['reused']:
        assert m['promotion_optimizer_updates']==0
        previous=Path(m['reused_development_unit']);assert not (previous/'test_predictions.npz').exists()
        same(checkpoint['state_dict'],torch.load(previous/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
    reviewed.append(dict(folder=str(folder),status='PASS',best_epoch=m['best_epoch']))
    del checkpoint
assert read(smoke/'metrics.json')['optimizer_updates']==24
for task,folds,variant in [('xc',[0,1],'xc_standard')]:
    gains=[]
    for fold in folds:
        arm='glt_ref' if task=='xc' else 'm_xattn'
        m=read(OUT/f'development/{variant}/{arm}/{task}/fold{fold}/metrics.json')
        baseline=read(ROOT/f'results/mcl_ph_r5_20261004/development/xc_standard/glt_ref/xc/fold{fold}/metrics.json')
        gains.append((baseline['best_validation_rmse']-m['best_validation_rmse'])/baseline['best_validation_rmse'])
    np.testing.assert_allclose(gains,frozen['gates'][task]['gains'],rtol=0,atol=1e-12)
    passed=(np.mean(gains)>=.02 and sum(g>0 for g in gains)==2) if task=='xc' else (np.mean(gains)>=.01 and sum(g>0 for g in gains)>=2 and min(gains)>=-.02)
    assert bool(passed)==frozen['gates'][task]['passed']
assert not list(OUT.glob('development/**/test_predictions.npz'))
if not passed:
    assert not (OUT/'paper5_outer').exists() and frozen['allowed_units']==[]
assert len(list(OUT.glob('**/runtime.json')))==len(units)+1
paper={};paper_models={}
for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
    c=[x.strip() for x in line.split('|')]
    if len(c)==5 and c[3].startswith('**'):
        paper[c[1].lower()]=float(c[3].strip('*').split(' ± ')[0]);paper_models[c[1].lower()]=c[2]
for row,original in zip(report['rows'],old_report['rows']):
    values=[];task=row['task'];manifest,labels,keys=cohorts[task]
    for fold,path in enumerate(row['prediction_paths']):
        with np.load(path,allow_pickle=False) as data:
            y,z=[data[k].reshape(-1) for k in ['y_true','y_pred']]
            indices=sorted(manifest['folds'][fold]['test_indices'])
            assert data['test_indices'].tolist()==indices and data['sample_keys'].tolist()==[keys[i] for i in indices]
            np.testing.assert_allclose(y,labels[indices],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(float(np.sqrt(np.mean((y-z)**2))),row['selected_fold_rmse'][fold],rtol=0,atol=1e-12)
            values.append(float(1-np.sum((y-z)**2)/np.sum((y-y.mean())**2)))
    np.testing.assert_allclose(values,row['selected_fold_r2'],rtol=0,atol=1e-12)
    np.testing.assert_allclose([np.mean(values),np.std(values),np.mean(values)-original['selected_mean_r2'],np.mean(values)-paper[task]],
        [row['selected_mean_r2'],row['selected_std_r2'],row['project_minus_r5'],row['project_minus_paper']],rtol=0,atol=1e-12)
    if task!='xc' or not passed:assert row['prediction_paths']==original['prediction_paths']
    assert row['paper_best_r2']==paper[task]
    assert row['paper_model']==paper_models[task]
output=dict(status='PASS',reviewed_at=datetime.now(timezone.utc).isoformat(),units=reviewed,
    tasks=7,budgets=report['budgets'],actual_complete_updates=complete_updates,actual_updates_including_smoke=complete_updates+24,
    scaler_fit_train_only=True,initial_heads_exact=True,promotion_weights_exact=True,
    gate_recomputed=True,seven_task_metrics_recomputed=True,extra_model_calls=0,extra_updates=0,
    standalone_probe='not executed',candidate_promoted=bool(passed))
(LOG/'final_review_02/report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='PASS',units=len(reviewed),tasks=7,actual_complete_updates=complete_updates,smoke_updates=24)),flush=True)
