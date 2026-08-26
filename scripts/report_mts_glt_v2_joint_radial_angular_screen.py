#!/usr/bin/env python3
"""Strict paired B/AC/RA reporter for the joint relation-basis screen."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/mts/glt_v2_joint_radial_angular_screen_v1.json'

def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    text=json.dumps(payload,indent=2,sort_keys=True)+'\n' if not isinstance(payload,str) else payload
    tmp.write_text(text,encoding='utf-8'); os.replace(tmp,path)

def unit(root,task,fold):
    shard=root/'shards/42'/task/f'fold_{fold}.csv'; prediction=root/'predictions/42'/task/f'fold_{fold}.npz'
    row=pd.read_csv(shard).iloc[0]
    with np.load(prediction,allow_pickle=False) as p: pred={'indices':np.asarray(p['sample_indices']),'targets':np.asarray(p['y_true']),'metadata':json.loads(str(np.asarray(p['metadata']).item()))}
    score=float(row['avg_test_r2'])
    if not np.isfinite(score): raise RuntimeError(f'nonfinite {shard}')
    return row,pred,score

def distribution(v):
    v=np.asarray(v,float); return {'fold_mean':float(v.mean()),'fold_median':float(np.median(v)),'fold_p25':float(np.quantile(v,.25)),'fold_p75':float(np.quantile(v,.75)),'fold_min':float(v.min()),'fold_max':float(v.max()),'positive_folds':int((v>0).sum())}

def main():
    c=json.loads(CONFIG.read_text()); out=ROOT/c['output_root']; roots={'B':ROOT/c['baseline_root'],'AC':out/'ac','RA':out/'ra'}; expected=str((ROOT/c['checkpoint']).resolve())
    rows=[]; diagnostics=[]
    for task in c['tasks']:
      for fold in c['folds']:
        u={k:unit(v,task,fold) for k,v in roots.items()}; ref=u['B'][1]
        for arm,(row,pred,_) in u.items():
          if str(row['checkpoint_path'])!=expected: raise RuntimeError(f'{arm} checkpoint mismatch {task}/{fold}')
          expected_mode={'B':'o8_glt_atom','AC':'o8_glt_atom_sbf_angle_control','RA':'o8_glt_atom_sbf_radial_angle'}[arm]
          if str(row['mts_glt_mode'])!=expected_mode: raise RuntimeError(f'{arm} mode mismatch {task}/{fold}')
          if str(row['mts_glt_fusion_strategy'])!='legacy_zero' or str(row['amp_dtype'])!='fp32': raise RuntimeError(f'{arm} protocol mismatch {task}/{fold}')
          if not np.array_equal(pred['indices'],ref['indices']) or not np.array_equal(pred['targets'],ref['targets']): raise RuntimeError(f'{arm} pairing mismatch {task}/{fold}')
          if int(pred['metadata']['fold_seed'])!=int(ref['metadata']['fold_seed']): raise RuntimeError(f'{arm} fold seed mismatch {task}/{fold}')
        b,ac,ra=(u[x][2] for x in ('B','AC','RA')); rows.append({'task':task,'fold':fold,'fold_seed':int(ref['metadata']['fold_seed']),'B':b,'AC':ac,'RA':ra,'SBFControlEffect':ac-b,'RadialContextEffect':ra-ac,'TotalJointEffect':ra-b})
        for arm in ('ac','ra'):
          p=out/'joint_basis_units'/arm/task/f'fold_{fold}.json'; d=json.loads(p.read_text());
          if d['new_parameter_count']!=672: raise RuntimeError(f'param mismatch {p}')
          diagnostics.append({'arm':arm.upper(),'task':task,'fold':fold,'mean_abs_sbf_bias':d['joint_bias_abs']['mean'],'median_abs_sbf_bias':d['joint_bias_abs']['median'],'p25_abs_sbf_bias':d['joint_bias_abs']['p25'],'p75_abs_sbf_bias':d['joint_bias_abs']['p75'],'p95_abs_sbf_bias':d['joint_bias_abs']['p95'],'mean_abs_angle_bias':d['existing_angle_bias_abs']['mean'],'sbf_angle_ratio':d['mean_joint_to_angle_ratio'],'distance_spearman':d['spearman_source_distance_bias_norm'],'angle_spearman':d['spearman_angle_bias_norm'],'distance_quartiles':json.dumps(d['source_distance_quartiles']),'internal':json.dumps(d['internal_relation_bias_norm']),'cross_ru':json.dumps(d['cross_ru_relation_bias_norm'])})
          cp=out/'checkpoints'/arm/task/f'fold_{fold}.pth'; meta=torch.load(cp,map_location='cpu',weights_only=False)
          if meta['mts_glt_mode']!={'ac':'o8_glt_atom_sbf_angle_control','ra':'o8_glt_atom_sbf_radial_angle'}[arm]: raise RuntimeError(f'checkpoint mode mismatch {cp}')
    folds=pd.DataFrame(rows); cols=('B','AC','RA','SBFControlEffect','RadialContextEffect','TotalJointEffect'); tasks=pd.DataFrame([{'task':t,**{k:float(folds[folds.task==t][k].mean()) for k in cols}} for t in c['tasks']])
    contrasts={}
    for key in cols[3:]: contrasts[key]={'macro3':float(tasks[key].mean()),'median_task':float(tasks[key].median()),'positive_tasks':int((tasks[key]>0).sum()),**distribution(folds[key])}
    passed=lambda x:x['macro3']>0 and x['median_task']>0 and x['positive_tasks']>=2 and x['positive_folds']>=5
    sanity=json.loads((out/'paired_occurrence_sanity.json').read_text()); grad=json.loads((out/'gradient_sanity.json').read_text()); basis=json.loads((out/'basis_sanity.json').read_text())
    summary={'schema':c['schema'],'baseline':c['baseline'],'checkpoint':expected,'checkpoint_step':5000,'tasks':c['tasks'],'folds':c['folds'],'seed':42,'protocol':c['evaluation_protocol'],'reused_b_runs':9,'new_ac_runs':9,'new_ra_runs':9,'failed_runs':0,'basis_contract':{'implementation':basis['implementation'],'num_spherical':7,'num_radial':6,'dimension':42,'cutoff':3.75,'reference_distance':1.40705,'directed_relation':'target A-B, source B-C','source_radial':'same-occurrence |B-C|'},'basis_sanity':basis,'step0_parity':sanity['step0_prediction_parity'],'first_gradient':grad['first_backward'],'new_parameter_count_ac':672,'new_parameter_count_ra':672,'contrasts':contrasts,'radial_context_decision':'POSITIVE_SCREEN' if passed(contrasts['RadialContextEffect']) else 'NOT_ESTABLISHED','joint_basis_total_decision':'GO_candidate' if passed(contrasts['TotalJointEffect']) else 'STOP_candidate','independent_blind_test':False,'anomalies':[]}
    folds.to_csv(out/'per_fold_results.csv',index=False); tasks.to_csv(out/'task_results.csv',index=False); pd.DataFrame(diagnostics).to_csv(out/'joint_basis_diagnostics.csv',index=False)
    atomic(out/'joint_ra_screen_summary.json',summary); atomic(out/'run_manifest.json',{'schema':'mts-glt-v2-joint-ra-run-manifest-v1','config':str(CONFIG.relative_to(ROOT)),'baseline_source':str(roots['B'].relative_to(ROOT)),'ac_source':str(roots['AC'].relative_to(ROOT)),'ra_source':str(roots['RA'].relative_to(ROOT)),'paired_units_verified':9})
    table_columns=['task','B','AC','RA','SBFControlEffect','RadialContextEffect','TotalJointEffect']
    table=['| '+' | '.join(table_columns)+' |','| '+' | '.join(['---']+['---:']*(len(table_columns)-1))+' |']
    for _, row in tasks.iterrows():
        table.append('| '+str(row['task'])+' | '+' | '.join(f"{float(row[key]):.6f}" for key in table_columns[1:])+' |')
    lines=['# MTS-GLT-v2 Joint Radial-Angular Screen','',f"- Radial-context decision: `{summary['radial_context_decision']}`",f"- Total candidate decision: `{summary['joint_basis_total_decision']}`",'- Protocol: historical_shared5 screening; not an independent blind test.','',*table]
    atomic(out/'joint_ra_screen_report.md','\n'.join(lines)+'\n'); print(json.dumps(summary,sort_keys=True))
if __name__=='__main__': main()
