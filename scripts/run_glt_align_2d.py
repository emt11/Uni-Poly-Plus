#!/usr/bin/env python3
"""One four-rank ALIGN PT and uniform four-condition FT matrix, exact budgets, tmux logs, no retries."""
import datetime,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_align_2d_unit import OUT,RECOVERY,REVIEW,BASE,COMMON,parent,ARMS,TASKS,read,write,ft_folder,prediction_config,validate_config
LOG=ROOT/'logs/glt_optimization_20261011/r4/U1/recovery1'
GPUS=(0,1,2,3)
ACTIVE=[]

def launch(name,command,gpus=(),limit=None):
 if any(g not in GPUS for g in gpus) or (len(gpus)>1 and not (name.startswith('pt_') and tuple(gpus)==GPUS)):raise ValueError('only one physical GPU0/1/2/3 per unit')
 folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
 cmd=shlex.join(command)
 if limit:cmd=f'timeout {limit}s '+cmd
 visible=','.join(map(str,gpus))
 (folder/'command.txt').write_text('CUDA_VISIBLE_DEVICES='+visible+' '+cmd+'\n')
 (folder/'cwd.txt').write_text(str(ROOT)+'\n')
 (folder/'environment.json').write_text(json.dumps(dict(CUDA_VISIBLE_DEVICES=visible,logical_device=('cuda:LOCAL_RANK' if len(gpus)==4 else 'cuda:0') if gpus else 'CPU',OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1)))
 script=folder/'command.sh'
 script.write_text(f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Iseconds > {shlex.quote(str(folder/'start.txt'))}
{cmd} > {shlex.quote(str(folder/'stdout.log'))} 2> {shlex.quote(str(folder/'stderr.log'))} &
process_pid=$!
printf '%s\\n' "$process_pid" > {shlex.quote(str(folder/'pid.txt'))}
wait "$process_pid"
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(folder/'exit.txt'))}
date -Iseconds > {shlex.quote(str(folder/'end.txt'))}
exit "$code"
''')
 subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_opt_r4_U1_recovery1_'+name,'bash '+shlex.quote(str(script))],check=True)
 ACTIVE.append(folder)
 return folder/'exit.txt'

def consumption(enforce=True):
 epochs=attempted=completed=formal=ft_completed=head_epochs=joint_epochs=0
 for a in ARMS:
  for t in TASKS:
   for f in range(5):
    folder=ft_folder(a,t,f)
    if (folder/'progress.json').exists():
     history=read(folder/'progress.json')['history'];epochs+=len(history)
     head_epochs+=sum(h['stage']=='head' for h in history);joint_epochs+=sum(h['stage']=='joint' for h in history)
    ft_completed+=int((folder/'metrics.json').exists())
    if (folder/'updates.json').exists():
     u=read(folder/'updates.json');attempted+=u['attempted'];completed+=u['completed']
    formal+=int((OUT/'formal'/a/t/f'fold{f}'/'metrics.json').exists())
 result=dict(geometry_derivations=0,conformer_generation=0,
  ft_starts=len(list(LOG.glob('ft_*/command.txt'))),ft_completed=ft_completed,ft_epochs=epochs,ft_head_epochs=head_epochs,ft_joint_epochs=joint_epochs,ft_validation_completed=epochs,ft_attempted_updates=attempted,ft_completed_updates=completed,
  formal_starts=len(list(LOG.glob('test_*/command.txt'))),formal_completed=formal,pytest_starts=int((LOG/'final/command.txt').exists()))
 pt=[read(p) for p in (OUT/'pretrain').glob('*/consumption.json')]
 val=[read(p) for p in (OUT/'pt_validation').glob('*/*/metrics.json')]
 result.update(pt_starts=len(list(LOG.glob('pt_*/command.txt'))),pt_attempted_updates=sum(p['attempted_updates'] for p in pt),pt_completed_updates=sum(p['completed_updates'] for p in pt),pt_train_presentations=sum(p['completed_updates'] for p in pt)*1008,pt_attempted_train_presentations=sum(p['attempted_updates'] for p in pt)*1008,pt_validation_completed=len(val),pt_validation_presentations=sum(p['presented'] for p in val),pt_validation_attempted_presentations=sum(read(p)['presented'] for p in (OUT/'pt_validation').glob('*/*/progress_rank*.json')))
 if enforce and (result['pt_starts']>1 or result['pt_attempted_updates']>5000 or result['pt_validation_attempted_presentations']>3072 or result['pt_validation_completed']>3 or result['ft_starts']>140 or epochs>9800 or attempted>224000 or result['formal_starts']>140 or head_epochs>1400 or joint_epochs>8400 or result['pytest_starts']>1):raise RuntimeError('hard budget exceeded')
 return result

def wait(path):
 while not path.exists():
  write(RECOVERY/'consumption.json',consumption());time.sleep(5)
 write(RECOVERY/'consumption.json',consumption())
 if int(path.read_text())!=0:raise RuntimeError(str(path)+' nonzero exit; stopped, no retry')

def matrix(formal):
 units=iter((a,t,f) for a in ARMS for t in TASKS for f in range(5))
 running={};exhausted=False
 while running or not exhausted:
  for gpu,path in list(running.items()):
   if path.exists():
    wait(path);del running[gpu]
  for gpu in GPUS:
   if gpu in running or exhausted:continue
   try:a,t,f=next(units)
   except StopIteration:exhausted=True;break
   cmd=['python','scripts/run_glt_align_2d_unit.py','--arm',a,'--task',t,'--fold',str(f)]
   if formal:cmd.append('--formal')
   running[gpu]=launch(('test_' if formal else 'ft_')+f'{a}_{t}_{f}',cmd,[gpu])
  write(RECOVERY/'consumption.json',consumption())
  if running:time.sleep(5)

def main():
 if sys.argv[1:]!=['--continue-after-initialization-failure']:raise ValueError('only explicitly authorized initialization recovery is supported')
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 if not OUT.is_dir() or any(read(OUT/'consumption.json').values()) or read(OUT/'failure.json').get('status')!='阻断':raise ValueError('recovery limited to zero-budget initialization failure')
 if any((OUT/p).exists() for p in ('pretrain','pt_validation','finetune','formal','evaluation_before_test.json')):raise FileExistsError('recovery cannot overwrite prior scientific execution')
 RECOVERY.mkdir(parents=True,exist_ok=False)
 REVIEW.mkdir(parents=True,exist_ok=False)
 def interrupted(signum,frame):raise RuntimeError('controller interrupted by signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  write(RECOVERY/'resource_check.json',dict(gpus=gpu_status,compute_apps=apps))
  rows=[row.split(',') for row in gpu_status.strip().splitlines()]
  if apps.strip() or {int(r[0]) for r in rows}!=set(GPUS) or any(float(r[1])<12000 for r in rows):raise RuntimeError('four physical GPUs unavailable; no fallback or termination of other jobs')
  wait(launch('initialization',['python','scripts/run_glt_align_2d_unit.py','--complete-initialization']))
  pi=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
  arm='P_ALIGN'
  wait(launch('pt_'+arm,['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_dual.py',
   '--config',str(OUT/'initialization'/f'{arm}.json'),'--cohort-root',str(pi),'--cache-root',str(pi),
   '--output',str(OUT/'pretrain'/arm),'--prep-workers','8','--lmdb-readahead'],GPUS))
  c=consumption()
  if (c['pt_starts'],c['pt_completed_updates'],c['pt_validation_completed'],c['pt_validation_presentations'])!=(1,5000,3,3072) or not parent('UA_2D').exists():raise ValueError('PT matrix incomplete')
  matrix(False)
  c=consumption()
  if (c['ft_starts'],c['ft_epochs'],c['ft_completed_updates'])!=(140,9800,224000):raise ValueError('FT matrix/budget incomplete')
  wait(launch('evaluation_freeze',['python','scripts/select_glt_align_2d.py']))
  matrix(True)
  if consumption()['formal_completed']!=140:raise ValueError('formal matrix incomplete')
  wait(launch('summary',['python','scripts/select_glt_align_2d.py','--summarize']))
  wait(launch('final',['python','-m','pytest','-q','tests/test_glt_align_2d.py'],limit=300))
  write(RECOVERY/'execution.json',dict(status='待审查',engineering='PASS',independent_review='NOT_RUN',consumption=consumption(),time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  wait(launch('handoff',['python','scripts/select_glt_align_2d.py','--handoff']))
  wait(launch('actual_diff',['python','scripts/select_glt_align_2d.py','--capture-diff']))
 except BaseException as error:
  for folder in ACTIVE:
   if not (folder/'exit.txt').exists() and (folder/'pid.txt').exists():
    pid=int((folder/'pid.txt').read_text())
    try:
     command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
     if 'glt_align_2d' in command or 'pytest' in command or 'pretrain_glt_dual' in command:os.kill(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError):pass
  state='需返修' if (LOG/'final/command.txt').exists() else '阻断'
  write(RECOVERY/'failure.json',dict(status=state,error=repr(error),traceback=traceback.format_exc(),consumption=consumption(enforce=False),retry=False))
  plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('**U1，执行中**',f'**U1，{state}**').replace('（r4，执行中）',f'（r4，{state}）')
  from scripts.select_glt_align_2d import capture_diff
  plan.write_text(s+'\n\nU1 recovery1实际停止：'+repr(error)+'。现场 results/glt_optimization_20261011/r4/U1/recovery1/failure.json；无自动重试。实耗见recovery1/consumption.json／failure.json；未完成候选不可比较。原初始化中断现场保留。\n')
  capture_diff()
  raise

if __name__=='__main__':main()
