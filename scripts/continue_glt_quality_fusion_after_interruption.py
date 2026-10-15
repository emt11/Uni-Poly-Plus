#!/usr/bin/env python3
"""User-authorized reruns of three interrupted folds; retain all prior evidence."""
import datetime,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_quality_fusion import LOG,launch,wait,consumption
from scripts.run_glt_quality_fusion_unit import OUT,read,write,ft_folder

def main():
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 plan=ROOT/'CURRENT_PLAN.md'
 switched=sys.argv[1:]==['--gpu123']
 if sys.argv[1:] and not switched:raise ValueError('unexpected continuation arguments')
 gpus=(1,2,3) if switched else (0,1,2)
 with plan.open('a') as p:p.write('\n\n2026-10-11 用户追加明确授权“请重新跑所有未完成/中断的任务”：继续现有124个未启动单元，不中止或重跑接续中的正常单元；随后仅重跑Q/Egc fold2/3/4。旧13个完成FT、全部PT和validation只读复用；旧三个中断目录保留，新三折写 finetune_recovery/，使用相同PT parent、head初态、split、seed、scaler与训练条件。FT预算明确扩展为143次启动、9735 epochs、218085次完整记录updates，另计最多273个未落盘更新，硬上限218358；正式预测仍140次，CPU测试仍唯一一次300秒。恢复控制器窗口 glt_qf_r1_controller_recovery，日志 controller_recovery/。\n')
 try:
  if switched:
   policy=read(OUT/'resource_gpu123.json')
   if policy['allowed_gpus']!=[1,2,3]:raise ValueError('latest GPU policy mismatch')
   for name in ('controller_ft_continue','controller_recovery'):
    previous=LOG/name/'exit.txt'
    while not previous.exists():time.sleep(1)
    if int(previous.read_text())!=143:raise ValueError('unexpected previous controller exit; retain evidence')
   for name in policy['active_jobs']:wait(LOG/name/'exit.txt')
   from scripts.run_glt_quality_fusion import fixed_units
   fixed_units(False,gpus=gpus,only_unstarted=True)
  else:
   previous=LOG/'controller_ft_continue/exit.txt'
   while not previous.exists():time.sleep(10)
   stop=read(OUT/'ft_continuation_stop.json')
   if 'remaining unstarted FT finished; three interrupted folds' not in stop['error']:raise ValueError('unexpected failure in unstarted FT; no automatic retry')
  c=consumption()
  if (c['ft_starts'],c['ft_epochs'],c['ft_completed_updates'])!=(140,9555,201705):raise ValueError('completed unstarted FT budget mismatch')
  units=[dict(arm='Q',task='egc',fold=f) for f in (2,3,4)]
  if (OUT/'budget_extension.json').exists():raise FileExistsError('recovery already authorized and initialized')
  write(OUT/'budget_extension.json',dict(authorization='用户明确要求重新跑所有未完成/中断任务',rerun_units=units,ft_limits=dict(starts=143,epochs=9735,updates=218358),recorded_updates_expected=218085,unlogged_updates_upper_bound=273,old_stop=str(OUT/'ft_continuation_stop.json'),automatic_retry=False))
  s=plan.read_text().replace('状态：**阻断**。','状态：**执行中**。',1);plan.write_text(s)
  jobs=[]
  for gpu,u in zip(gpus,units):
   if ft_folder(u['arm'],u['task'],u['fold']).exists():raise FileExistsError('recovery output exists')
   jobs.append(launch('ft_recovery_Q_egc_f'+str(u['fold']),['python','scripts/run_glt_quality_fusion_unit.py','--arm','Q','--task','egc','--fold',str(u['fold'])],[gpu]))
  errors=[]
  for job in jobs:
   try:wait(job)
   except RuntimeError as error:errors.append(str(error))
  if errors:raise RuntimeError('\n'.join(errors))
  c=consumption()
  if (c['ft_epochs'],c['ft_completed_updates'],c['ft_starts'])!=(9735,218085,143):raise ValueError('extended FT budget incomplete')
  wait(launch('evaluation_freeze',['python','scripts/select_glt_quality_fusion.py']))
  from scripts.run_glt_quality_fusion import fixed_units
  fixed_units(True,gpus=gpus if switched else (0,1,2,3))
  wait(launch('summary',['python','scripts/select_glt_quality_fusion.py','--summarize']))
  wait(launch('final',['python','-m','pytest','-q','tests/test_glt_quality_fusion.py'],limit=300))
  write(OUT/'execution.json',dict(status='待审查',independent_review='NOT_RUN',consumption=consumption(),budget_extension=read(OUT/'budget_extension.json'),time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  wait(launch('handoff',['python','scripts/select_glt_quality_fusion.py','--handoff']))
 except BaseException as error:
  state='需返修' if (LOG/'final/exit.txt').exists() else '阻断'
  write(OUT/'recovery_failure.json',dict(status=state,error=repr(error),traceback=traceback.format_exc(),retry=False))
  s=plan.read_text().replace('状态：**执行中**。',f'状态：**{state}**。',1)
  plan.write_text(s+'\n\n恢复执行停止：'+repr(error)+'；现场 recovery_failure.json，无自动重试。\n')
  raise

if __name__=='__main__':main()
