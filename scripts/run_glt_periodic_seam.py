#!/usr/bin/env python3
"""One controller, fixed budget, tmux logs, no retry or historical rerun."""
import datetime,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_periodic_seam_unit import OUT,ARMS,TASKS,read,write,ft_folder,restarted_unit
LOG=ROOT/'logs/glt_periodic_seam_20261009/r1'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'

def launch(name,command,gpus=(),limit=None):
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
 subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_ps_r1_'+name,'bash '+shlex.quote(str(script))],check=True)
 return folder/'exit.txt'

def consumption():
 pt=[];ft=[];formal=[];val=[]
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
 limits=dict(starts=140,epochs=9600,updates=205800)
 if (OUT/'resume_gpu3.json').exists():
  restart=read(OUT/'resume_gpu3.json');limits=restart['ft_budget_limits']
  result.update(ft_matrix_epochs=result['ft_epochs'],ft_matrix_completed_updates=result['ft_completed_updates'])
  result['ft_epochs']+=restart['interrupted_epochs_logged']
  result['ft_attempted_updates']+=restart['interrupted_updates_logged']
  result['ft_completed_updates']+=restart['interrupted_updates_logged']
  result['ft_updates_unlogged_upper_bound']=restart['interrupted_updates_unlogged_upper_bound']
  result['ft_updates_upper_bound']=result['ft_attempted_updates']+result['ft_updates_unlogged_upper_bound']
  result['allowed_gpus']=[3]
 if (OUT/'resource_handoff_03.json').exists() and read(OUT/'resource_handoff_03.json')['status']=='switched':result['allowed_gpus']=[0,3]
 if result['pt_starts']>4 or result['pt_attempted_updates']>20000 or result['validation_starts']>12 or result['validation_presentations']>12288 or result['ft_starts']>limits['starts'] or result['ft_epochs']>limits['epochs'] or result['ft_attempted_updates']+result.get('ft_updates_unlogged_upper_bound',0)>limits['updates'] or result['formal_starts']>140 or result['pytest_starts']>1:raise RuntimeError('budget exceeded')
 return result

def wait(path):
 while not path.exists():
  write(OUT/'consumption.json',consumption());time.sleep(10)
 code=int(path.read_text())
 write(OUT/'consumption.json',consumption())
 if code:raise RuntimeError(f'{path} exit {code}; stopped, no retry')

def fixed_units(formal,gpus=(0,1,2,3),resume_completed=False):
 units=[(a,t,f) for a in ARMS for t in TASKS for f in range(5)]
 if resume_completed:
  pending=[]
  for a,t,f in units:
   folder=LOG/(('test_' if formal else 'ft_gpu3_restart_' if restarted_unit(a,t,f) else 'ft_')+f'{a}_{t}_f{f}')
   output=OUT/'formal'/a/t/f'fold{f}' if formal else ft_folder(a,t,f)
   if (folder/'command.txt').exists():
    if not (folder/'exit.txt').exists() or int((folder/'exit.txt').read_text())!=0:raise RuntimeError(f'cannot resume incomplete/failed unit {folder}')
    m=read(output/'metrics.json')
    if m['status']!='PASS':raise ValueError(f'completed unit status mismatch {output}')
   else:
    if folder.exists() or output.exists():raise FileExistsError(f'unlaunched unit already has artifacts {output}')
    pending.append((a,t,f))
  units=pending
 for start in range(0,len(units),len(gpus)):
  jobs=[]
  for gpu,(a,t,f) in zip(gpus,units[start:start+len(gpus)]):
   cmd=['python','scripts/run_glt_periodic_seam_unit.py','--arm',a,'--task',t,'--fold',str(f)]
   if formal:cmd.append('--formal')
   jobs.append(launch(('test_' if formal else 'ft_gpu3_restart_' if restarted_unit(a,t,f) else 'ft_')+f'{a}_{t}_f{f}',cmd,[gpu]))
  # Complete other in-flight authorized independent units; never launch the
  # next batch after any failure. Each writes a distinct directory.
  failures=[]
  for path in jobs:
   try:wait(path)
   except RuntimeError as error:failures.append(str(error))
  if failures:raise RuntimeError('\n'.join(failures))

def main():
 handoff=sys.argv[1:]==['--resource-handoff-123']
 resume_gpu3=sys.argv[1:]==['--resume-gpu3']
 handoff03=sys.argv[1:]==['--resource-handoff-03']
 if sys.argv[1:] and not (handoff or resume_gpu3 or handoff03):raise ValueError('unknown controller arguments')
 if not (handoff or resume_gpu3 or handoff03):OUT.mkdir(parents=True,exist_ok=False)
 try:
  gpus=(0,3) if handoff03 else (3,) if resume_gpu3 else (1,2,3) if handoff else (0,1,2,3)
  if handoff03:
   change=read(OUT/'resource_handoff_03.json')
   if change['status']!='draining' or change['gpus']!=list(gpus) or (OUT/'failure.json').exists() or (OUT/'execution.json').exists() or (OUT/'selection_before_test.json').exists():raise ValueError('GPU03 handoff state mismatch')
   old_pid=change['previous_controller_pid']
   if 'State:\tT' not in Path(f'/proc/{old_pid}/status').read_text():raise RuntimeError('previous controller must be paused before handoff')
   for path in change['active_exit_paths']:wait(Path(path))
   os.kill(old_pid,signal.SIGTERM);os.kill(old_pid,signal.SIGCONT)
   previous_exit=Path(change['previous_controller_log'])/'exit.txt'
   while not previous_exit.exists():time.sleep(1)
   if int(previous_exit.read_text())!=143:raise RuntimeError('unexpected previous controller exit')
   change.update(status='switched',time=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_controller_exit=143)
   write(OUT/'resource_handoff_03.json',change)
   for a in ARMS:
    if read(OUT/'pretrain'/a/'consumption.json')['completed_updates']!=5000:raise ValueError('PT budget incomplete')
  elif resume_gpu3:
   restart=read(OUT/'resume_gpu3.json')
   if restart['allowed_gpus']!=[3] or restart['restarted_units']!=[['P_RAW','egc',f] for f in (2,3,4)]:raise ValueError('GPU3 resume authorization mismatch')
   if (OUT/'failure.json').exists() or (OUT/'execution.json').exists() or (OUT/'selection_before_test.json').exists():raise ValueError('unexpected resume phase')
   for a,t,f in restart['restarted_units']:
    if int((LOG/f'ft_{a}_{t}_f{f}'/'exit.txt').read_text())!=143:raise ValueError('interrupted unit identity mismatch')
   for a in ARMS:
    if read(OUT/'pretrain'/a/'consumption.json')['completed_updates']!=5000:raise ValueError('PT budget incomplete')
  elif handoff:
   change=read(OUT/'resource_handoff.json')
   if change['status']!='draining' or change['gpus']!=list(gpus) or (OUT/'failure.json').exists() or (OUT/'execution.json').exists():raise ValueError('resource handoff state mismatch')
   old_pid=change['previous_controller_pid']
   if 'State:\tT' not in Path(f'/proc/{old_pid}/status').read_text():raise RuntimeError('previous controller must be paused before handoff')
   for command in sorted(LOG.glob('ft_*/command.txt')):wait(command.with_name('exit.txt'))
   os.kill(old_pid,signal.SIGTERM);os.kill(old_pid,signal.SIGCONT)
   while not (LOG/'controller/exit.txt').exists():time.sleep(1)
   if int((LOG/'controller/exit.txt').read_text())!=143:raise RuntimeError('unexpected previous controller exit')
   for a in ARMS:
    if read(OUT/'pretrain'/a/'consumption.json')['completed_updates']!=5000:raise ValueError('PT budget incomplete')
   change.update(status='switched',time=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_ft_before_switch=len(list((OUT/'finetune').glob('*/*/*/metrics.json'))),previous_controller_exit=143)
   write(OUT/'resource_handoff.json',change)
  else:
   wait(launch('initialization',['python','scripts/run_glt_periodic_seam_unit.py','--initialize']))
   for a in ARMS:
    wait(launch('pt_'+a,['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_dual.py',
     '--config',str(OUT/'initialization'/f'{a}.json'),'--cohort-root',str(PI),'--cache-root',str(PI),'--dual-static-root',str(PI),
     '--output',str(OUT/'pretrain'/a),'--prep-workers','8','--lmdb-readahead'],range(4)))
    if read(OUT/'pretrain'/a/'consumption.json')['completed_updates']!=5000:raise ValueError('PT budget incomplete')
  fixed_units(False,gpus,resume_completed=handoff or resume_gpu3 or handoff03)
  c=consumption()
  if (c.get('ft_matrix_epochs',c['ft_epochs']),c.get('ft_matrix_completed_updates',c['ft_completed_updates']))!=(9600,205800):raise ValueError('FT budget incomplete')
  wait(launch('selection',['python','scripts/select_glt_periodic_seam.py']))
  fixed_units(True,gpus)
  wait(launch('summary',['python','scripts/select_glt_periodic_seam.py','--summarize']))
  wait(launch('final',['python','-m','pytest','-q','tests/test_glt_periodic_seam.py'],limit=300))
  write(OUT/'execution.json',dict(status='待审查',independent_review='NOT_RUN',consumption=consumption(),time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 except BaseException as error:
  write(OUT/'failure.json',dict(status='需返修' if (LOG/'final/exit.txt').exists() else '阻断',error=repr(error),traceback=traceback.format_exc(),consumption=consumption(),retry=False))
  raise
if __name__=='__main__':main()
