#!/usr/bin/env python3
import os, subprocess, time
from pathlib import Path
ROOT=Path('/root/workspace/Uni-Poly-Plus-master'); UNIT=ROOT/'scripts/run_fce_path_ablation_unit.py'; OUT=ROOT/'results/fce_path_ablation_20261015/p1/finetune'
arms=['R0_RAW2D','R0_FUSED2D','R0_FUSED']; tasks=['eea','egb','egc','ei','eps','nc','xc']
units=[(a,t,f) for a in arms for t in tasks for f in range(5)]
pending=[u for u in units if not (OUT/u[0]/u[1]/f'fold{u[2]}'/'metrics.json').exists()]
active=[]; slots=[2,2,3,3]
log=ROOT/'logs/fce_path_ablation_20261015/p1/controller.log'; log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as lf:
 lf.write(f'controller start pending={len(pending)} slots={slots}\n'); lf.flush()
 while pending or active:
  active=[x for x in active if x[0].poll() is None]
  used=[x[1] for x in active]
  for gpu in slots:
   if used.count(gpu)>=2: continue
   if not pending: break
   arm,task,fold=pending.pop(0)
   folder=OUT/arm/task/f'fold{fold}'
   if (folder/'metrics.json').exists(): continue
   env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
   cmd=['python',str(UNIT),'--arm',arm,'--task',task,'--fold',str(fold)]
   lf.write(f'START gpu={gpu} arm={arm} task={task} fold={fold}\n'); lf.flush()
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
   active.append((p,gpu,arm,task,fold)); used.append(gpu)
  time.sleep(5)
  for p,gpu,arm,task,fold in list(active):
   if p.poll() is not None:
    active.remove((p,gpu,arm,task,fold)); lf.write(f'END gpu={gpu} arm={arm} task={task} fold={fold} rc={p.returncode}\n'); lf.flush()
    if p.returncode!=0: raise SystemExit(f'unit failed {arm} {task} {fold} rc={p.returncode}')
 lf.write('controller complete\n')
