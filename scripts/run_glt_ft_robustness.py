#!/usr/bin/env python3
"""Fixed two-GPU FT-only matrix, exact budgets, tmux logs, no retries."""
import datetime,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_ft_robustness_unit import OUT,BASE,PARENT,ARMS,TASKS,read,write,ft_folder
LOG=ROOT/'logs/glt_ft_robustness_20261011/r1'
GPUS=(0,3)
ACTIVE=[]

def launch(name,command,gpus=(),limit=None):
 if any(g not in GPUS for g in gpus) or len(gpus)>1:raise ValueError('only one physical GPU0 or3 per unit')
 folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
 cmd=shlex.join(command)
 if limit:cmd=f'timeout {limit}s '+cmd
 visible=','.join(map(str,gpus))
 (folder/'command.txt').write_text('CUDA_VISIBLE_DEVICES='+visible+' '+cmd+'\n')
 (folder/'cwd.txt').write_text(str(ROOT)+'\n')
 (folder/'environment.json').write_text(json.dumps(dict(CUDA_VISIBLE_DEVICES=visible,logical_device='cuda:0' if gpus else 'CPU',OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1)))
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
 subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_ftr_r1_'+name,'bash '+shlex.quote(str(script))],check=True)
 ACTIVE.append(folder)
 return folder/'exit.txt'

def consumption(enforce=True):
 epochs=attempted=completed=formal=0
 for a in ARMS:
  for t in TASKS:
   for f in range(5):
    folder=ft_folder(a,t,f)
    if (folder/'progress.json').exists():epochs+=len(read(folder/'progress.json')['history'])
    if (folder/'updates.json').exists():
     u=read(folder/'updates.json');attempted+=u['attempted'];completed+=u['completed']
    formal+=int((OUT/'formal'/a/t/f'fold{f}'/'metrics.json').exists())
 result=dict(geometry_derivations=0,conformer_generation=0,pt_updates=0,pt_validation_presentations=0,
  ft_starts=len(list(LOG.glob('ft_*/command.txt'))),ft_epochs=epochs,ft_attempted_updates=attempted,ft_completed_updates=completed,
  formal_starts=len(list(LOG.glob('test_*/command.txt'))),formal_completed=formal,pytest_starts=int((LOG/'final/command.txt').exists()))
 if enforce and (result['ft_starts']>105 or epochs>7200 or attempted>154350 or result['formal_starts']>105 or result['pytest_starts']>1):raise RuntimeError('hard budget exceeded')
 return result

def wait(path):
 while not path.exists():
  write(OUT/'consumption.json',consumption());time.sleep(5)
 write(OUT/'consumption.json',consumption())
 if int(path.read_text())!=0:raise RuntimeError(str(path)+' nonzero exit; stopped, no retry')

def matrix(formal):
 units=[(a,t,f) for a in ARMS for t in TASKS for f in range(5)]
 for start in range(0,105,2):
  jobs=[]
  for gpu,(a,t,f) in zip(GPUS,units[start:start+2]):
   cmd=['python','scripts/run_glt_ft_robustness_unit.py','--arm',a,'--task',t,'--fold',str(f)]
   if formal:cmd.append('--formal')
   jobs.append(launch(('test_' if formal else 'ft_')+f'{a}_{t}_{f}',cmd,[gpu]))
  failures=[]
  for path in jobs:
   try:wait(path)
   except RuntimeError as error:failures.append(str(error))
  if failures:raise RuntimeError('\n'.join(failures))

def main():
 if sys.argv[1:]:raise ValueError('no resume/retry arguments authorized')
 from src.training.glt_dual_runtime import require_tmux
 require_tmux();OUT.mkdir(parents=True,exist_ok=False)
 def interrupted(signum,frame):raise RuntimeError('controller interrupted by signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  cfg=read(BASE/'initialization/F_CE.json')
  for a in ARMS:
   config={k:cfg[k] for k in ('fusion_mode','fusion_revision','geometry_revision','geometry_mode','angle_value_enabled','readout','use_md200','md200_root','atom_ce_input','align_enabled','local_enabled','third_task','torsion_modules')}
   config.update(plan='GLT-FT-ROBUSTNESS-20261011-01',revision='r1',arm=a,cycle_root=str(OUT),parent=str(PARENT),
    parent_fusion_gate='none',fusion_gate='incident' if a in ('G','DG') else 'none',joint_3d_drop_probability=.2 if a in ('D','DG') else 0.,
    mask_rng_seed_formula='10000+42+fold',quality_root=None,ft_batch_size=24,ft_amp_dtype='fp32',ft_workers=0,ft_seed_formula='42+fold',split='paper5_outer',target_transform='train_only_standard')
   write(OUT/'initialization'/f'{a}.json',config)
  matrix(False)
  c=consumption()
  if (c['ft_starts'],c['ft_epochs'],c['ft_completed_updates'])!=(105,7200,154350):raise ValueError('FT matrix/budget incomplete')
  wait(launch('evaluation_freeze',['python','scripts/select_glt_ft_robustness.py']))
  matrix(True)
  if consumption()['formal_completed']!=105:raise ValueError('formal matrix incomplete')
  wait(launch('summary',['python','scripts/select_glt_ft_robustness.py','--summarize']))
  wait(launch('final',['python','-m','pytest','-q','tests/test_glt_ft_robustness.py'],limit=300))
  write(OUT/'execution.json',dict(status='待审查',engineering='PASS',independent_review='NOT_RUN',consumption=consumption(),time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  wait(launch('handoff',['python','scripts/select_glt_ft_robustness.py','--handoff']))
 except BaseException as error:
  for folder in ACTIVE:
   if not (folder/'exit.txt').exists() and (folder/'pid.txt').exists():
    pid=int((folder/'pid.txt').read_text())
    try:
     command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
     if 'scripts/' in command and ('glt_ft_robustness' in command or 'pytest' in command):os.kill(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError):pass
  state='需返修' if (LOG/'final/command.txt').exists() else '阻断'
  write(OUT/'failure.json',dict(status=state,error=repr(error),traceback=traceback.format_exc(),consumption=consumption(enforce=False),retry=False))
  plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('状态：**执行中**。',f'状态：**{state}**。',1)
  plan.write_text(s+'\n\n本轮实际停止：'+repr(error)+'。现场 results/glt_ft_robustness_20261011/r1/failure.json；未自动重试。\n')
  raise

if __name__=='__main__':main()
