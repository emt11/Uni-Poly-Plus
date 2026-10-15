#!/usr/bin/env python3
"""Final saved-evidence closeout. No model inference, tests or digests."""
import json,math,difflib,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_dual_opt import OUT,LOG,DOWN,SPLIT,ARMS,TASKS,read,write,reference,consumption,comparison,now
from src.training.glt_dual_runtime import require_tmux

def main():
 require_tmux()
 result=read(OUT/'final_review/seven_task_results.json');checks=[];budgets=consumption();group_rows={a:[] for a in ARMS}
 if budgets['formal_process_starts']!=210:raise ValueError('formal units incomplete')
 for arm in ARMS:
  for task in TASKS:
   raw=[json.loads(line) for line in (DOWN/f'{task}.jsonl').read_text().splitlines()]
   split=read(SPLIT/f'{task}.json')
   for fold in range(5):
    folder=OUT/'formal'/arm/task/f'fold{fold}';dev=OUT/'development'/arm/task/f'fold{fold}'
    m=read(folder/'metrics.json');d=read(dev/'metrics.json');entry=split['folds'][fold]
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as a:
     y=a['y_true'].reshape(-1);pred=a['y_pred'].reshape(-1);ids=a['indices'].tolist();keys=a['sample_keys'].tolist()
     if ids!=sorted(entry['test_indices']) or keys!=[raw[i]['sample_key'] for i in ids]:raise ValueError('prediction rows/keys differ')
     if not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('prediction nonfinite')
     np.testing.assert_allclose(y,[raw[i]['label'] for i in ids],rtol=1e-6,atol=1e-5)
     train=np.array([raw[i]['label'] for i in entry['train_indices']],dtype=np.float64)
     np.testing.assert_allclose(a['scaler_mean'],[train.mean()],rtol=1e-10,atol=1e-10)
     np.testing.assert_allclose(a['scaler_scale'],[train.std() if train.std()>0 else 1.],rtol=1e-10,atol=1e-10)
     if str(a['outer_test'])!='RUN':raise ValueError('formal marker differs')
     rmse=float(np.sqrt(np.mean((y-pred)**2)));r2=1-float(np.sum((y-pred)**2)/np.sum((y-y.mean())**2));mae=float(np.mean(np.abs(y-pred)))
     if not math.isclose(rmse,m['rmse'],rel_tol=1e-10,abs_tol=1e-12) or not math.isclose(r2,m['r2'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('saved metrics differ')
     best_epoch=int(a['best_epoch'])
     if best_epoch not in range(1,d['epochs']+1):raise ValueError('best epoch range')
    log=LOG/f'formal_{arm}_{task}_f{fold}'
    if (log/'exit.txt').read_text().strip()!='0':raise ValueError('formal exit failure')
    if m['epochs']!=0 or m['updates']!={'attempted':0,'completed':0}:raise ValueError('test-only performed updates')
    checks.append(dict(arm=arm,task=task,fold=fold,r2=r2,rmse=rmse,mae=mae,best=str(dev/'best.pt'),prediction=str(folder/'test_predictions.npz'),indices=ids,sample_count=len(ids),best_epoch=best_epoch,formal_updates=0))
   group_rows[arm].append(next(r for r in result['rows'] if r['arm']==arm and r['task']==task))
 comparisons={a:dict(H=comparison(a,'H',True),I=comparison(a,'I',True),primary=comparison(a,'H' if a=='R_INCIDENT' else 'B_FP',True)) for a in ARMS}
 for values in comparisons.values():
  for comp in values.values():
   comp['positive_fold_counts']=[sum(g>0 for g in row) for row in comp['fold_gains']]
   comp['worst_fold_gains']=[min(row) for row in comp['fold_gains']]
 for row in result['rows']:
  observed=[c for c in checks if c['arm']==row['arm'] and c['task']==row['task']]
  row['mae_mean']=float(np.mean([c['mae'] for c in observed]));row['mae_std']=float(np.std([c['mae'] for c in observed]))
 # Local actual diffs against retained pre-edit sources; pre-existing changes stay separate.
 patch=[];before=LOG/'before'
 for saved in sorted(before.rglob('*.py')):
  name=saved.relative_to(before);current=ROOT/name
  if current.exists():patch.extend(difflib.unified_diff(saved.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='before/'+str(name),tofile=str(name)))
 for name in ('src/modules/glt_dual_opt.py','scripts/run_glt_dual_opt.py','scripts/finish_glt_dual_opt.py','scripts/collect_glt_dual_opt.py','tests/test_glt_dual_opt.py'):
  file=ROOT/name;patch.extend(difflib.unified_diff([],file.read_text().splitlines(True),fromfile='/dev/null',tofile=name))
 (LOG/'actual_code.diff').write_text(''.join(patch))
 (LOG/'context_tracked.diff').write_bytes(subprocess.check_output(['git','diff']))
 text=['# GLT-DUAL-OPT-20261006-01/r1 实际正式结果','', '冻结选择：I；全部validation候选未满足所有门槛。新正式结果仅用于完整科学报告，未据test重选。','', '五折算术均值，seed42+fold，std ddof=0。论文协议等价性未核实，仅数值对照；不是独立盲测或多seed稳定优势。','']
 for arm in ARMS:
  text+=['## '+arm,'','|任务|R²均值|R² std|RMSE均值|RMSE std|论文最佳模型|论文R²|项目−论文|','|---|---:|---:|---:|---:|---|---:|---:|']
  for row in group_rows[arm]:
   text.append(f"|{row['task']}|{row['r2_mean']:.3f}|{row['r2_std']:.3f}|{row['rmse_mean']:.3f}|{row['rmse_std']:.3f}|{row['paper_model']}|{row['paper_r2']:.3f}|{row['difference']:+.3f}|")
  text+=['','相对H/I及主要参考的逐折收益、正收益折数和最差折见report.json；各五折R²/RMSE明细与实际预测路径见seven_task_results.json。','']
 report=dict(plan='GLT-DUAL-OPT-20261006-01',revision='r1',status='SELF_CLOSEOUT_COMPLETE_PENDING_INDEPENDENT_AGENT_REVIEW',time=now(),scientific_result='No eligible validation candidate; retain I; complete negative/comparative results',consumption=budgets,logical_training_units=215,pretraining_worker_rank_starts=20,ft_starts=210,formal_units=210,engineering_test_calls=2,first_test='5 passed / 2 failed: helper import path',repair_test='7 passed; same scenarios; user-authorized additional final call',repair_authorization='请你解决以上问题并继续执行！',training_updates_added_in_continuation=0,smoke=0,probes=0,benchmark=0,extra_pretrain_validation=0,checks=checks,comparisons=comparisons,M_minus_F=result['M_minus_F'],selection_before_test=str(OUT/'selection_before_test.json'),rows=result['rows'],architecture_notes=dict(R='incident readout only; historical GLT parent',B='complete reconstructed FP step0, not proven historical FP state',A='FP replaced by existing ALIGN',F='token fusion plus all-copy 3D identity mask; confounded mechanism combination',M='same F structure/mask/init; CE reads fused atoms',G='seam length weights and center-shared angle supervision only'),diff=str(LOG/'actual_code.diff'),context_diff_note='context_tracked.diff includes existing unrelated changes; finetune_mcl_ph.py local GLTReferenceArm constructor adds fusion_mode/readout arguments; before/related.diff retains prior changes',tests=[str(LOG/'final_pytest/stdout.log'),str(LOG/'final_pytest_repair_01/stdout.log')],independent_review='NOT_YET_RUN',history_archived=False)
 write(OUT/'final_review/report.json',report)
 (OUT/'final_review/results.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(dict(status=report['status'],formal_units=210,budget=budgets),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
