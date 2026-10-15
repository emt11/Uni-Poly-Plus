"""CPU-only r5 source and saved-artifact acceptance; no model inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mcl_ph_r4_fast import atomic_json, timestamp, finite
from src.training.glt_dual_runtime import open_source

OUT = ROOT / 'results/mcl_ph_r6_20261004'
LOG = ROOT / 'logs/mcl_ph_r6_20261004'
R4 = ROOT / 'results/mcl_ph_r4_20261002/fast_campaign_04'
SOURCE = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLITS = ROOT / 'data/splits/mips_paper5_outer'
PLAN = 'MCL-PH-NEXT-20261004-02'
CONFIGS = {'xc_standard': ROOT / 'results/mcl_ph_r4_20261001/resolved/glt_ref.json',
           'nc_fixed_m_xattn': R4 / 'm_xattn_fast.json'}
PACKS = {'xc_standard': ROOT / 'results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/deploy_05000.pt',
         'nc_fixed_m_xattn': ROOT / 'results/mcl_ph_r4_20261002/r3_m_xattn_fast_01/deploy_05000.pt'}

def read(path): return json.loads(Path(path).read_text())

def cohort(task):
    manifest = read(SPLITS / f'{task}.json')
    indices = list(range(manifest['sample_count']))
    source, frame = open_source(SOURCE, SOURCE, task=task, selected_indices=indices,
        dual_static_root=SOURCE, expected_task_rows=len(indices),
        record_index_path=SOURCE / f'{task}_offsets.npy', split_manifest=manifest)
    try:
        assert frame['original_row'].astype(int).tolist() == indices
        keys = [key.hex() for key, _ in source.samples]
        labels = frame['label'].to_numpy(dtype=np.float64)
        raw = pd.read_csv(ROOT / f'data/raw/smi_{task}.csv')
        np.testing.assert_array_equal(labels, raw.iloc[:, 1].to_numpy(dtype=np.float64))
        assert np.isfinite(labels).all() and len(set(keys)) == len(keys)
        return manifest, keys, labels
    finally: source.close()

def tensors_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
        assert torch.equal(left[key], right[key]), key

def check_predictions(folder, metrics, manifest, keys, labels, *, test=False):
    fold = next(x for x in manifest['folds'] if x['fold'] == metrics['fold'])
    train, validation, testing = [fold[k] for k in ('train_indices','validation_indices','test_indices')]
    assert not (set(train)&set(validation) or set(train)&set(testing) or set(validation)&set(testing))
    assert set(train+validation+testing) == set(range(len(labels)))
    roles = ('validation','test') if test else ('validation',)
    for role in roles:
        indices = validation if role == 'validation' else sorted(testing)
        with np.load(folder / f'{role}_predictions.npz', allow_pickle=False) as prediction:
            y, z = [prediction[k].reshape(-1) for k in ('y_true','y_pred')]
            assert prediction[f'{role}_indices'].tolist() == indices
            assert prediction['sample_keys'].tolist() == [keys[i] for i in indices]
            assert y.shape == z.shape == (len(indices),) and np.isfinite(y).all() and np.isfinite(z).all()
            np.testing.assert_allclose(y, labels[indices], rtol=2e-6, atol=2e-6)
            rmse = float(np.sqrt(np.mean((y-z)**2)))
            np.testing.assert_allclose(rmse, metrics['test_rmse' if role=='test' else 'best_validation_rmse'], rtol=1e-6, atol=1e-7)
            if test and role=='test':
                r2=float(1-np.sum((y-z)**2)/np.sum((y-y.mean())**2))
                np.testing.assert_allclose(r2,metrics['test_r2'],rtol=1e-6,atol=1e-7)
    return train

def audit_unit(folder, variant, stage, fold):
    folder=Path(folder)
    assert folder.resolve().is_relative_to(OUT)
    task,arm=('xc','glt_ref') if variant=='xc_standard' else ('nc','m_xattn')
    m=read(folder/'metrics.json'); runtime=read(folder/'runtime.json'); finite(m)
    assert runtime['status']=='PASS' and runtime['exit_code']==0
    for k,v in dict(task=task,arm=arm,stage=stage,fold=fold,target_transform='standard',
            scaler_fit_split='train',experiment_plan=PLAN,pretrain_step=5000,
            initialization_seed=42+fold,source_root=str(SOURCE),executed_epochs=2 if stage=='smoke' else 70).items():
        assert m[k]==v,(folder,k,m.get(k),v)
    assert Path(m['config_path'])==CONFIGS[variant] and Path(m['pretrain_package_path'])==PACKS[variant]
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(t).all() for t in state['state_dict'].values())
    assert state['config']==read(CONFIGS[variant])
    for k in ('arm','task','fold','stage','target_transform','experiment_plan','best_epoch','best_validation_rmse','optimizer_updates','executed_epochs'):
        assert state[k]==m[k],(folder,k)
    manifest,keys,labels=cohort(task)
    train=check_predictions(folder,m,manifest,keys,labels,test=stage=='paper5_outer')
    scaler=StandardScaler().fit(labels[train].reshape(-1,1))
    np.testing.assert_allclose(state['scaler_mean'],scaler.mean_,rtol=0,atol=1e-12)
    np.testing.assert_allclose(state['scaler_scale'],scaler.scale_,rtol=0,atol=1e-12)
    from scripts.finetune_mcl_ph import training_contract
    from types import SimpleNamespace
    args=SimpleNamespace(experiment_plan=PLAN, candidate_config=OUT/'candidate.json', fold=fold, config=CONFIGS[variant], split_root=SPLITS, cohort_root=SOURCE, checkpoint=PACKS[variant])
    assert m['training_contract']==state['training_contract']==training_contract(args)
    assert m['learning_rate_table']==dict(head_stage1=3e-4,backbone_stage2=1e-4,head_stage2=1e-3)
    for item in m['history']:
        e=item['stage_epoch']; cosine=(1+np.cos(np.pi*(e if item['stage']=='head' else e%10)/10))/2
        expected=[3e-5+(3e-4-3e-5)*cosine] if item['stage']=='head' else [1e-6+(lr-1e-6)*cosine for lr in (1e-4,1e-3)]
        np.testing.assert_allclose(item['learning_rates'],expected,rtol=1e-12,atol=1e-15)
    h=m['history']; assert len(h)==m['executed_epochs']
    assert [x['stage'] for x in h]==(['head','joint'] if stage=='smoke' else ['head']*10+['joint']*60)
    assert sum(x['training_steps'] for x in h)==m['optimizer_updates']
    assert all(x['training_steps']==int(np.ceil(len(train)/24)) for x in h)
    best=min(h,key=lambda x:x['validation_rmse']);assert best['epoch']==m['best_epoch'] and best['validation_rmse']==m['best_validation_rmse']
    initial=torch.load(folder/'head_initial.pt',map_location='cpu',weights_only=False)
    reference=(R4/f'paper5_outer/glt_ref/xc/fold{fold}' if task=='xc' else R4/'development/m_xattn/nc/fold0')
    tensors_equal(initial,torch.load(reference/'head_initial.pt',map_location='cpu',weights_only=False))
    if stage!='paper5_outer':
        assert m['outer_test']=='NOT_RUN' and not (folder/'test_predictions.npz').exists()
    else:
        frozen=read(OUT/'frozen_selection.json')
        from datetime import datetime
        assert (folder/'test_predictions.npz').stat().st_mtime>=datetime.fromisoformat(frozen['frozen_at']).timestamp()
        if 'reused_development_unit' in m:
            old=Path(m['reused_development_unit']); old_m=read(old/'metrics.json')
            assert old_m['outer_test']=='NOT_RUN' and not (old/'test_predictions.npz').exists()
            assert m['promotion_optimizer_updates']==0
            tensors_equal(state['state_dict'],torch.load(old/'best.pt',map_location='cpu',weights_only=False)['state_dict'])
    report=dict(status='PASS',checked_at=timestamp(),actual_labels_keys_scaler=True,
        finite_model=True,initial_head_exact=True,metric_recomputed=True,outer_test=m['outer_test'])
    atomic_json(folder/'acceptance.json',report)
    return m

def audit_sources():
    report=[]
    aggregate=read(R4/'seven_task_results.json')
    assert read(ROOT/'logs/mcl_ph_r4_20261002/final_review_02/report.json')['status']=='PASS'
    for row in aggregate['rows']:
        task=row['task'];manifest,keys,labels=cohort(task);folds=[]
        for fold in range(5):
            folder=R4/f"paper5_outer/{row['selected_arm']}/{task}/fold{fold}"
            m=read(folder/'metrics.json');train=check_predictions(folder,m,manifest,keys,labels)
            state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            scaler=StandardScaler().fit(labels[train].reshape(-1,1)) if task!='xc' else None
            np.testing.assert_allclose(state['scaler_mean'],scaler.mean_ if scaler else [0.],rtol=0,atol=1e-12)
            np.testing.assert_allclose(state['scaler_scale'],scaler.scale_ if scaler else [1.],rtol=0,atol=1e-12)
            f=manifest['folds'][fold];valid=f['validation_indices']
            folds.append(dict(fold=fold,train_mean=float(labels[train].mean()),train_std=float(labels[train].std()),
                validation_std=float(labels[valid].std()),best_epoch=m['best_epoch'],rmse=m['best_validation_rmse'],
                learning_rates=m['learning_rate_table'],target_transform=m['target_transform']))
            del state
        report.append(dict(task=task,rows=len(labels),raw_csv_labels_exact=True,folds=folds,
            unit='raw CSV task column; physical unit not stated in file, unchanged'))
    for variant,package in PACKS.items():
        assert package.exists()
        acceptance=ROOT/'logs/mcl_ph_r4_20261002'/package.parent.name/'acceptance.json'
        assert read(acceptance)['conclusion']=='PASS'
    p=OUT/'a0';p.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS',checked_at=timestamp(),rows=report,model_calls=0,updates=0,
        note='Validated all seven raw CSV/source labels, validation keys/truth/scalers; physical units unspecified, no conversion.')
    atomic_json(p/'report.json',result);print(json.dumps(dict(status='PASS',tasks=7,model_calls=0,updates=0)),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--a0',action='store_true');a=p.parse_args()
    if a.a0: audit_sources()
