#!/usr/bin/env python3
"""One controller, fixed budget, tmux logs, no retry or historical rerun."""
import datetime,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_quality_fusion_unit import OUT,ARMS,TASKS,read,write,ft_folder
LOG=ROOT/'logs/glt_quality_fusion_20261010/r1'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'

def launch(name,command,gpus=(),limit=None):
 policy=OUT/'resource_gpu123.json'
 if policy.exists() and not set(gpus).issubset(set(read(policy)['allowed_gpus'])):raise ValueError('GPU outside latest authorized resource policy')
 folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
 cmd=shlex.join(command)
 if limit:cmd=f'timeout {limit}s '+cmd
 (folder/'command.txt').write_text(cmd+'\n');(folder/'cwd.txt').write_text(str(ROOT)+'\n')
 script=folder/'command.sh'
 script.write_text(f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str,gpus)))}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Iseconds > {shlex.quote(str(folder/'start.txt'))}
{cmd} > {shlex.quote(str(folder/'stdout.log'))} 2> {shlex.quote(str(folder/'stderr.log'))}
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(folder/'exit.txt'))}
date -Iseconds > {shlex.quote(str(folder/'end.txt'))}
exit "$code"
''')
 subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_qf_r1_'+name,'bash '+shlex.quote(str(script))],check=True)
 return folder/'exit.txt'

def consumption():
 pt=[];ft=[];formal=[];val=[]
 extension=read(OUT/'budget_extension.json') if (OUT/'budget_extension.json').exists() else None
 if extension:
  for u in read(OUT/'ft_interruption.json')['units']:
   ft.append(dict(epochs=u['epochs'],updates=u['updates']))
 for a in ARMS:
  p=OUT/'pretrain'/a/'consumption.json'
  if p.exists():pt.append(read(p))
  for t in TASKS:
   for f in range(5):
    path=ft_folder(a,t,f)
    if path.exists():
     m=read(path/'metrics.json') if (path/'metrics.json').exists() else read(path/'progress.json') if (path/'progress.json').exists() else {'updates':{'attempted':0,'completed':0},'epochs':0}
     ft.append(dict(epochs=m.get('epochs',len(m.get('history',[]))),updates=m['updates']))
    path=OUT/'formal'/a/t/f'fold{f}'
    if path.exists():formal.append((path/'metrics.json').exists())
  for step in (1000,2500,5000):
   path=OUT/'pt_validation'/a/f'step{step:05d}'
   if path.exists():val.append(sum(read(p)['presented'] for p in path.glob('progress_rank*.json')))
 result=dict(pt_starts=len(list(LOG.glob('pt_*/command.txt'))),pt_attempted_updates=sum(p['attempted_updates'] for p in pt),pt_completed_updates=sum(p['completed_updates'] for p in pt),pt_train_presentations_completed=sum(p['completed_updates'] for p in pt)*1008,validation_starts=len(val),validation_presentations=sum(val),ft_starts=len(list(LOG.glob('ft_*/command.txt'))),ft_epochs=sum(m['epochs'] for m in ft),ft_attempted_updates=sum(m['updates']['attempted'] for m in ft),ft_completed_updates=sum(m['updates']['completed'] for m in ft),formal_starts=len(list(LOG.glob('test_*/command.txt'))),formal_completed=sum(formal),pytest_starts=len(list(LOG.glob('final/command.txt'))))
 interruption=OUT/'ft_interruption.json'
 if interruption.exists():result['ft_updates_unlogged_upper_bound']=read(interruption)['updates_unlogged_upper_bound']
 limits=dict(starts=140,epochs=9600,updates=205800)
 if extension:limits=extension['ft_limits']
 if result['pt_starts']>4 or result['pt_attempted_updates']>20000 or result['validation_starts']>12 or result['validation_presentations']>12288 or result['ft_starts']>limits['starts'] or result['ft_epochs']>limits['epochs'] or result['ft_attempted_updates']+result.get('ft_updates_unlogged_upper_bound',0)>limits['updates'] or result['formal_starts']>140 or result['pytest_starts']>1:raise RuntimeError('budget exceeded')
 return result

def wait(path):
 while not path.exists():
  write(OUT/'consumption.json',consumption());time.sleep(10)
 code=int(path.read_text())
 write(OUT/'consumption.json',consumption())
 if code:raise RuntimeError(f'{path} exit {code}; stopped, no retry')

def fixed_units(formal,gpus=(1,2,3),only_unstarted=False):
 units=[(a,t,f) for a in ARMS for t in TASKS for f in range(5)]
 if only_unstarted:units=[(a,t,f) for a,t,f in units if not ft_folder(a,t,f).exists() and not (LOG/f'ft_{a}_{t}_f{f}').exists()]
 for start in range(0,len(units),len(gpus)):
  jobs=[]
  for gpu,(a,t,f) in zip(gpus,units[start:start+len(gpus)]):
   cmd=['python','scripts/run_glt_quality_fusion_unit.py','--arm',a,'--task',t,'--fold',str(f)]
   if formal:cmd.append('--formal')
   jobs.append(launch(('test_' if formal else 'ft_')+f'{a}_{t}_f{f}',cmd,[gpu]))
  failures=[]
  for path in jobs:
   try:wait(path)
   except RuntimeError as error:failures.append(str(error))
  if failures:raise RuntimeError('\n'.join(failures))

def continue_unstarted_ft():
 resource=read(OUT/'resource.json')
 if resource['allowed_gpus']!=[0,1,2,3] or resource['world_size']!=4:raise ValueError('four authorized GPUs required')
 c=consumption()
 if (c['pt_completed_updates'],c['validation_starts'],c['validation_presentations'],c['formal_starts'],c['pytest_starts'])!=(20000,12,12288,0,0):raise ValueError('unexpected budget before FT continuation')
 interrupted=[];completed=0
 for a in ARMS:
  for t in TASKS:
   for f in range(5):
    folder=ft_folder(a,t,f)
    if (folder/'metrics.json').exists():
     if read(folder/'metrics.json')['status']!='PASS':raise ValueError('failed completed FT')
     completed+=1
    elif folder.exists():
     progress=read(folder/'progress.json')
     interrupted.append(dict(arm=a,task=t,fold=f,epochs=len(progress['history']),updates=progress['updates'],path=str(folder),resumable=False))
 if completed!=13 or [(u['arm'],u['task'],u['fold']) for u in interrupted]!=[('Q','egc',2),('Q','egc',3),('Q','egc',4)]:raise ValueError('interruption inventory changed')
 write(OUT/'ft_interruption.json',dict(status='partial states unavailable',units=interrupted,updates_unlogged_upper_bound=273,automatic_retry=False))
 plan=ROOT/'CURRENT_PLAN.md'
 with plan.open('a') as p:p.write('\n\n2026-10-11 用户授权继续：四臂PT与12次过程validation已完成并只读复用；13个完整FT保留。Q/Egc fold2/3/4仅有head_initial与progress，分别44/46/45 epochs，缺少模型、优化器、scheduler、RNG与best状态，无法精确恢复。系统当前启动时间与tmux缺失已核实，异常原因未确定。继续124个尚未启动FT，使用GPU0/1/2/3；不重跑三折、不扩大预算、不执行不完整正式矩阵或最终测试。中断未落盘更新以最多273保守计入；现场记录 ft_interruption.json。控制器窗口 glt_qf_r1_controller_ft_continue，日志 controller_ft_continue/。\n')
 try:
  fixed_units(False,only_unstarted=True)
  raise RuntimeError('remaining unstarted FT finished; three interrupted folds lack recoverable state; strict budget prevents rerun, full evaluation blocked')
 except BaseException as error:
  write(OUT/'ft_continuation_stop.json',dict(status='阻断',error=repr(error),consumption=consumption(),retry=False))
  s=plan.read_text().replace('状态：**执行中**。','状态：**阻断**。',1)
  plan.write_text(s+'\n\nFT接续停止：'+repr(error)+'；现场 ft_continuation_stop.json，未自动重试。\n')
  raise

def main():
 if sys.argv[1:]==['--continue-ft-unstarted']:return continue_unstarted_ft()
 if sys.argv[1:]!=['--continue-quality']:raise ValueError('explicit user-authorized quality continuation required')
 if set(p.name for p in OUT.iterdir())!={'quality','resource.json','failure.json','consumption.json'}:raise ValueError('unexpected execution artifacts before authorized continuation')
 if int((LOG/'quality/exit.txt').read_text())!=143:raise ValueError('original stop evidence mismatch')
 if any(consumption().values()):raise ValueError('training/test budget already consumed before continuation')
 try:
  resource=read(OUT/'resource.json')
  if resource['allowed_gpus']!=[0,1,2,3] or resource['world_size']!=4:raise ValueError('GPU0,1,2,3 and four ranks required by latest authorization')
  wait(launch('quality_continue',['python','scripts/derive_glt_geometry_quality.py']))
  wait(launch('initialization',['python','scripts/run_glt_quality_fusion_unit.py','--initialize']))
  for a in ARMS:
   wait(launch('pt_'+a,['torchrun','--standalone','--nproc_per_node='+str(resource['world_size']),'scripts/pretrain_glt_dual.py',
    '--config',str(OUT/'initialization'/f'{a}.json'),'--cohort-root',str(PI),'--cache-root',str(PI),'--dual-static-root',str(PI),
    '--output',str(OUT/'pretrain'/a),'--prep-workers','8','--lmdb-readahead'],[0,1,2,3]))
   if read(OUT/'pretrain'/a/'consumption.json')['completed_updates']!=5000:raise ValueError('PT budget incomplete')
  fixed_units(False)
  c=consumption()
  if (c['ft_epochs'],c['ft_completed_updates'])!=(9600,205800):raise ValueError('FT budget incomplete')
  wait(launch('evaluation_freeze',['python','scripts/select_glt_quality_fusion.py']))
  fixed_units(True)
  wait(launch('summary',['python','scripts/select_glt_quality_fusion.py','--summarize']))
  wait(launch('final',['python','-m','pytest','-q','tests/test_glt_quality_fusion.py'],limit=300))
  write(OUT/'execution.json',dict(status='待审查',independent_review='NOT_RUN',consumption=consumption(),time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  wait(launch('handoff',['python','scripts/select_glt_quality_fusion.py','--handoff']))
 except BaseException as error:
  state='需返修' if (LOG/'final/exit.txt').exists() else '阻断'
  failure=dict(status=state,error=repr(error),traceback=traceback.format_exc(),consumption=consumption(),retry=False)
  write(OUT/'continuation_failure.json',failure)
  plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text().replace('状态：**执行中**。',f'状态：**{state}**。',1)
  plan.write_text(text+'\n\n实际停止条件：'+repr(error)+'。现场见 '+str(OUT/'continuation_failure.json')+'；无自动重试。\n')
  raise
if __name__=='__main__':main()
