#!/usr/bin/env python3
"""One authorized continuation, GPU2 PT2/FT4/TEST4, inherited clock and budget."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import copy
import datetime
import difflib
import fcntl
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_opt_r6 as t
from scripts import run_ua3d_base_opt_r6 as serial
from scripts.run_ua2d_a020_fp_fusion import atomic_exit,poll_exit,owned_group,pid_aliases
RESOURCE_REVISION='gpu2-pt2-fttest4-c1'
HARD_SECONDS=96*3600;CPU_SECONDS=600;CLOSEOUT_SECONDS=300
CAPS=dict(pt=2,ft=4,test=4)
# Admission reservations, not batch or scientific changes. Historical PT peak ~9.51GiB.
MEMORY_MIB=dict(pt=12288,ft=8192,test=4096)
T0=1793023633.8442488
DEADLINE=1793369233.8442488
EXTRA_UNIT=('A_L_A8_B8','eea',0)
NEW_SOURCES=('scripts/run_ua3d_base_opt_r6_concurrency.py','configs/ua3d_base_opt_r6_concurrency.json')

class Deadline(serial.Deadline):pass

def verify_identity_dead(identity):
    if t.process_alive(identity):raise RuntimeError('protected parent process still alive: '+repr(identity))

def verify_parent():
    manifest=t.read(t.SOURCE/'execution_manifest.json');budget=t.read(t.SOURCE/'budget.json')
    stop=t.read(t.SOURCE/'authorization/user_stop.json');exits=t.read(t.SOURCE/'controller_exit.json')
    if manifest['t0_unix']!=T0 or manifest['deadline_unix']!=DEADLINE or exits['exit']!=1:raise ValueError('parent clock/exit differs')
    if stop['reason']!='USER_REQUESTED_STOP' or [job['exit'] for job in budget['jobs']]!=[0,1]:raise ValueError('parent stop/jobs mismatch')
    expected=dict.fromkeys(t.LIMITS,0);expected.update(initialize_starts=1,ft_starts=1,ft_epochs=70,ft_updates=700)
    if budget['charged_or_reserved']!=expected:raise ValueError('parent reservation differs from authorized continuation')
    actual=budget['known_actual_lower_bound'];completed=budget['known_completed']
    if any(actual[k]!=v for k,v in dict(initialize_starts=1,ft_starts=1,ft_epochs=69,ft_updates=684).items()):raise ValueError('parent actual differs')
    if completed['ft_epochs']!=68 or completed['ft_updates']!=684:raise ValueError('parent completed differs')
    verify_identity_dead(manifest['controller_identity'])
    identities=[manifest['controller_identity']]
    for job in budget['jobs']:
        path=Path(job['log']);value=t.read(path/'process.json')
        if int((path/'exit.txt').read_text())!=job['exit']:raise ValueError('parent true exit differs')
        identities.extend([value['wrapper'],value['timeout_identity']])
        if job.get('consumption') and Path(job['consumption']).exists():
            usage=t.read(job['consumption']);identities.extend([usage['process'],*usage.get('workers',[])])
    for identity in identities:verify_identity_dead(identity)
    record=t.read(t.SOURCE/'registry/recipes.json')
    if list(record['recipes'])!=[t.BASELINE] or len(record['slots'])!=1 or record['selections'] or record['pt'] or record['aliases']:
        raise ValueError('parent registry outside single interrupted FT boundary')
    recipe=record['recipes'][t.BASELINE]
    if recipe['ft']!=t.m.DEFAULT_FT or recipe['pt_id']!='HIST_R4_BASE':raise ValueError('parent baseline recipe differs')
    return dict(manifest=manifest,budget=budget,stop=stop,controller_exit=exits,registry=record,identities=identities)

def cumulative_ledger(parent_budget,jobs):
    local=serial.ledger(jobs);combined=copy.deepcopy(local)
    for field in ('charged_or_reserved','known_actual_lower_bound','known_completed'):
        combined[field]={key:parent_budget[field][key]+local[field][key] for key in t.LIMITS}
    combined['limits']=dict(t.LIMITS)
    combined['budget_violations']=[key for key,limit in t.LIMITS.items()
        if combined['charged_or_reserved'][key]>limit or combined['known_actual_lower_bound'][key]>combined['charged_or_reserved'][key]
        or combined['known_completed'][key]>combined['known_actual_lower_bound'][key]]
    new_limits=dict(t.LIMITS,initialize_starts=0,ft_starts=805,ft_epochs=56350,ft_updates=1288000)
    combined['budget_violations']+=['continuation_'+key for key,value in local['charged_or_reserved'].items() if value>new_limits[key]]
    replacement=[job for job in jobs if job['kind']=='ft' and (job['arm'],job['task'],job['fold'])==EXTRA_UNIT]
    if len(replacement)>1:combined['budget_violations'].append('replacement_allowance_reused')
    combined.update(parent_budget_source=str(t.SOURCE/'budget.json'),parent_jobs=parent_budget['jobs'],
        continuation_jobs=jobs,jobs=parent_budget['jobs']+jobs,continuation_only=local,
        extra_ft_unit=list(EXTRA_UNIT),extra_ft_used=len(replacement),parent_initialization_reused=True,
        accounting='parent failed reservation remains charged; new complete replacement only for A_L_A8_B8/eea/fold0; no refunds')
    return combined

def gpu_inventory():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[line.split(',') for line in lines.splitlines() if line.split(',')[0].strip()=='2']
    if len(rows)!=1:raise RuntimeError('physical GPU2 inventory mismatch')
    _,uuid,free,total=[x.strip() for x in rows[0]]
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    processes={}
    for line in lines.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if parts[0]==uuid:processes[int(parts[1])]=int(parts[2]) if parts[2].isdigit() else 0
    return dict(physical_gpu=2,uuid=uuid,free_mib=int(free),total_mib=int(total),processes=processes,time=t.now())

def known_gpu_aliases(job):
    path=Path(job['log'])/'process.json'
    if not path.exists():return set()
    process=t.read(path);identity=process['timeout_identity']
    usage=Path(job['consumption']) if job.get('consumption') else None
    if usage is not None and usage.exists():
        saved=t.read(usage)
        if saved.get('process')!=identity:raise ValueError('GPU child identity not equal to wrapper-owned process')
    # Only kernel-reported namespace identities; never match unknown NVML PIDs by count/time.
    return pid_aliases(identity)

def slot_available(active_kinds,kind):
    if kind not in CAPS:raise ValueError('GPU stage kind required')
    if any(value!=kind for value in active_kinds):raise ValueError('PT/FT/TEST phase mixing forbidden')
    return len(active_kinds)<CAPS[kind]

class Controller(serial.Controller):
    def __init__(self):
        t.require_tmux()
        if not t.CONTINUATION or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit continuation/CPU controller environment required')
        registration=t.read(t.OUT/'authorization/registration.json')
        if registration['resource_revision']!=RESOURCE_REVISION or (t.LOG/'registration.exit.txt').read_text().strip()!='0':raise ValueError('fresh registration missing')
        self.lock=(t.OUT/'controller.lock').open('x');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.parent_lock=(t.SOURCE/'controller.lock').open('r');fcntl.flock(self.parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.parent=verify_parent()
        if self.parent['budget']!=registration['parent_budget']:raise ValueError('parent budget changed after registration')
        t.exclusive(t.OUT/'authorization/controller_started.json',dict(pid=os.getpid(),identity=t.process_identity(os.getpid()),time=t.now(),once=True))
        self.started=T0;self.deadline=DEADLINE;self.jobs=[];self.active=None;self.pending_ft=[];self.accepted_baseline=False
        self.continued_at=time.time();self.stopping=False;self.dispatch_counter=0
        sys.stdout=(t.LOG/'controller.stdout.log').open('x',buffering=1);sys.stderr=(t.LOG/'controller.stderr.log').open('x',buffering=1)
        def cutoff(sig,frame):raise Deadline('inherited deadline closeout reserve reached')
        signal.signal(signal.SIGALRM,cutoff);signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-CLOSEOUT_SECONDS-time.time()))
        self.manifest=dict(plan=t.PLAN,revision=t.REVISION,resource_revision=RESOURCE_REVISION,status='执行中',
            started=self.parent['manifest']['started'],t0_unix=T0,deadline_unix=DEADLINE,deadline=self.parent['manifest']['deadline'],hard_seconds=HARD_SECONDS,
            continued_at=t.now(),continued_at_unix=self.continued_at,cwd=str(ROOT),command=sys.argv,
            outer_launch=t.read(t.OUT/'authorization/controller_launch.json'),
            env={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','UA3D_BASE_OPT_RESOURCE_REVISION','UA3D_BASE_OPT_EXECUTION_ROOT')},
            controller_identity=t.process_identity(os.getpid()),tmux=os.environ.get('TMUX_PANE'),
            tmux_window=subprocess.check_output(['tmux','display-message','-p','#{window_id} #{window_name}'],text=True).strip(),
            based_on=registration['based_on'],parent_run=str(t.SOURCE),parent_controller_identity=self.parent['manifest']['controller_identity'],
            parent_controller_exit=1,parent_initialization_exit=0,parent_interrupted_ft_exit=1,parent_user_stop_source=str(t.SOURCE/'authorization/user_stop.json'),
            physical_gpus=[2],physical_gpu=2,pt_max_concurrency=2,ft_max_concurrency=4,test_max_concurrency=4,world_size=1,phase_exclusive=True,
            caps=CAPS,memory_admission_mib=MEMORY_MIB,limits=t.LIMITS,initialization_source=str(t.INITIAL),
            implementation_started_unix=registration['implementation_started_unix'],implementation_seconds_before_controller=self.continued_at-registration['implementation_started_unix'],
            automatic_retry=False,automatic_resume=False,independent_review='PENDING')
        t.exclusive(t.OUT/'execution_manifest.json',self.manifest)
        t.exclusive(t.OUT/'config.json',t.read(ROOT/'configs/ua3d_base_opt_r6_concurrency.json'))
        t.exclusive(t.OUT/'authorization/CURRENT_PLAN.before.json',dict(text=(ROOT/'CURRENT_PLAN.md').read_text()))
        self.capture_diff();self.refresh()

    def capture_diff(self):
        folder=t.OUT/'authorization/source_before';diff=[]
        registration=t.read(t.OUT/'authorization/registration.json')
        for name in registration['source_files_before']:
            before=(folder/name).read_text();after=(ROOT/name).read_text()
            diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        for name in NEW_SOURCES:
            if (ROOT/name).exists():diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
        path=t.OUT/'final_review/actual_changes.diff';path.parent.mkdir(exist_ok=True);path.write_text(''.join(diff))

    def refresh(self):
        for job in self.jobs:
            if job['exit'] is None:
                rc=poll_exit(job)
                if rc is not None:job.update(exit=rc,ended=t.now())
        value=cumulative_ledger(self.parent['budget'],self.jobs)
        value.update(time=t.now(),elapsed_seconds=time.time()-T0,continued_elapsed_seconds=time.time()-self.continued_at,deadline_unix=DEADLINE)
        active=[job for job in self.jobs if job['exit'] is None]
        counts={kind:sum(job['kind']==kind for job in active) for kind in CAPS}
        if any(counts[k]>CAPS[k] for k in CAPS) or (counts['pt'] and (counts['ft'] or counts['test'])) or (counts['ft'] and counts['test']):
            value['budget_violations'].append('concurrency_or_phase_overlap')
        value['current_slots']=counts;value['current_pt_worker_limit']=4*counts['pt']
        t.write(t.OUT/'budget.json',value)
        t.write(t.OUT/'scheduler/slots.json',dict(counts=counts,active=[dict(name=j['name'],kind=j['kind'],window=j.get('window'),physical_gpu=j['physical_gpu']) for j in active],time=t.now()))
        if value['budget_violations'] and not self.stopping:raise RuntimeError('cumulative budget/resource violation '+repr(value['budget_violations']))
        return value

    def check_failures(self):
        bad=[job for job in self.jobs if job['exit'] not in (None,0)]
        if bad:raise RuntimeError(f"{bad[0]['name']} true exit={bad[0]['exit']}; no retry or automatic repair")

    def available(self,kind):
        active=[job for job in self.jobs if job['exit'] is None and job['kind'] in CAPS]
        if not slot_available([job['kind'] for job in active],kind):return None
        inventory=gpu_inventory();known=set();unmaterialized=0
        for job in active:
            aliases=known_gpu_aliases(job);known.update(aliases)
            observed=sum(memory for pid,memory in inventory['processes'].items() if pid in aliases)
            unmaterialized+=max(0,MEMORY_MIB[kind]-observed)
        unknown=set(inventory['processes'])-known
        enough=inventory['free_mib']>=MEMORY_MIB[kind]+unmaterialized
        wait=dict(inventory,processes={str(k):v for k,v in inventory['processes'].items()},kind=kind,active=len(active),cap=CAPS[kind],
            known_pids=sorted(known),unknown_pids=sorted(unknown),unmaterialized_reserved_mib=unmaterialized,
            required_new_mib=MEMORY_MIB[kind],dispatch_allowed=not unknown and enough,
            policy='unknown NVML PID waits; no identity inference by time/count; reserve unmaterialized active memory')
        t.write(t.OUT/'occupancy_wait.json',wait)
        return inventory if not unknown and enough else None

    def launch(self,kind,arm=None,task=None,fold=None):
        self.guard_clock(kind);self.refresh();self.check_failures()
        if kind not in (*CAPS,'cpu'):raise ValueError('initialize/prepare forbidden in continuation')
        if any(job['kind']=='cpu' for job in self.jobs):raise RuntimeError('no dispatch after sole CPU suite')
        frozen=t.OUT/'selection/final_frozen.json'
        if kind in ('pt','ft') and frozen.exists():raise RuntimeError('training after freeze')
        if kind in ('test','cpu') and not frozen.exists():raise RuntimeError('early TEST/CPU')
        if kind=='pt' and arm not in t.registry()['pt']:raise ValueError('unregistered PT')
        if kind=='ft':t.config(arm,task,fold)
        if kind=='test':t.frozen_unit(t.read(frozen),arm,task,fold)
        if kind=='cpu':
            if any(job['exit'] is None for job in self.jobs):raise RuntimeError('CPU cannot overlap any job')
            if len([j for j in self.jobs if j['kind']=='test' and j['exit']==0])!=len(t.read(frozen)['units']):raise RuntimeError('CPU before all TEST')
            inventory=None
        else:
            inventory=self.available(kind)
            if inventory is None:return None
        name='base_opt_r6_c1_'+kind+(f'_{arm}' if arm else '')+(f'_{task}_{fold}' if task is not None else '')
        if any(job['name']==name for job in self.jobs):raise ValueError('duplicate estimator dispatch')
        if kind=='ft' and not any(j['kind']=='ft' for j in self.jobs) and (arm,task,fold)!=EXTRA_UNIT:raise ValueError('authorized replacement must be first FT dispatch')
        reserve=serial.reservation(kind,task,fold);charged=self.refresh()['charged_or_reserved']
        if any(charged[key]+value>t.LIMITS[key] for key,value in reserve.items()):raise RuntimeError('cumulative reservation unavailable')
        if kind=='pt':output=t.OUT/'pretrain'/arm;consumption=output/'consumption.json'
        elif kind in ('ft','test'):output=t.unit_path(arm,task,fold,kind=='test');consumption=output/'consumption.json'
        else:output=None;consumption=None
        if output is not None and output.exists():raise FileExistsError('unit writer/overwrite conflict '+str(output))
        log=t.LOG/name;log.mkdir(exist_ok=False)
        if kind=='cpu':command=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_ua3d_base_opt_r6.py']
        elif kind=='pt':command=[sys.executable,'-u','scripts/pretrain_ua3d_base_opt_r6.py','--arm',arm,'--physical-gpu','2']
        else:command=[sys.executable,'-u','scripts/run_ua3d_base_opt_r6_concurrency.py',kind,'--arm',arm,'--task',task,'--fold',str(fold),'--physical-gpu','2']
        gpu=kind in CAPS;uuid=inventory['uuid'] if gpu else ''
        env=dict(CUDA_VISIBLE_DEVICES='2' if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONDONTWRITEBYTECODE='1',UA3M_PHYSICAL_GPU_UUID=uuid,UA3D_BASE_OPT_RESOURCE_REVISION=RESOURCE_REVISION,
            UA3D_BASE_OPT_EXECUTION_ROOT=str(t.OUT),UA3D_BASE_OPT_JOB_KIND=kind)
        command=['env','CUDA_VISIBLE_DEVICES=2' if gpu else 'CUDA_VISIBLE_DEVICES=',*command]
        job=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,physical_gpus=[2] if gpu else [],physical_gpu=2 if gpu else None,
            world_size=1,job_concurrency_limit=CAPS.get(kind,1),resource_revision=RESOURCE_REVISION,gpu_uuid=uuid,argv=command,env=env,cwd=str(ROOT),
            log=str(log),reservation=reserve,consumption=str(consumption) if consumption else None,exit=None,reserved_at=t.now(),
            attempt='concurrency_r1-once',dispatch_pending=True,dispatch_index=self.dispatch_counter,
            replaces_parent_job='base_opt_r6_ft_A_L_A8_B8_eea_0' if kind=='ft' and (arm,task,fold)==EXTRA_UNIT else None)
        self.jobs.append(job);self.dispatch_counter+=1;self.refresh();t.write(log/'command.json',job)
        wrapper_env=['env',*[key+'='+value for key,value in env.items() if key!='CUDA_VISIBLE_DEVICES'],'CUDA_VISIBLE_DEVICES=']
        cmd=[*wrapper_env,sys.executable,'-u','scripts/run_ua3d_base_opt_r6_concurrency.py','wrap','--spec',str(log/'command.json')]
        job['window']=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly:','-n',name,'-c',str(ROOT),shlex.join(cmd)],text=True).strip()
        subprocess.run(['tmux','set-option','-w','-t',job['window'],'remain-on-exit','on'],check=True)
        pane=int(subprocess.check_output(['tmux','display-message','-p','-t',job['window'],'#{pane_pid}'],text=True))
        job['wrapper_identity']=t.process_identity(pane);job['dispatch_pending']=False
        budget=self.refresh()
        with (t.OUT/'scheduler/dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(time=t.now(),name=name,index=job['dispatch_index'],slots=budget['current_slots'],inventory=inventory))+'\n')
        t.update_plan('执行中',kind,f'派发{name}；当前槽位{budget["current_slots"]}；继承硬截止{self.manifest["deadline"]}。')
        return job

    def execute_batch(self,requests):
        pending=deque(requests)
        while pending or any(job['exit'] is None for job in self.jobs):
            self.guard_clock();self.refresh();self.check_failures()
            while pending:
                job=self.launch(*pending[0])
                if job is None:break
                pending.popleft()
                self.refresh();self.check_failures()
            t.write(t.OUT/'scheduler/pending.json',dict(time=t.now(),requests=list(pending)))
            if pending or any(job['exit'] is None for job in self.jobs):time.sleep(3)
        self.refresh();self.check_failures()

    def ft_recipe(self,slot,stage,pt_id='HIST_R4_BASE',pt=None,ft=None,parent_id=None,control=None):
        if slot==t.BASELINE and not self.accepted_baseline:
            recipe=t.get_recipe(slot)
            if recipe['ft']!=ft or recipe['pt_id']!=pt_id or stage!='A':raise ValueError('accepted parent recipe differs')
            canonical,new=slot,True;self.accepted_baseline=True
        else:canonical,new=t.register_recipe(slot,stage,pt_id,pt,ft,parent_id,control)
        if new:self.pending_ft.append(canonical)
        else:t.aggregate(canonical)
        return canonical

    def execute_ft(self):
        recipes=self.pending_ft;self.pending_ft=[]
        self.execute_batch([('ft',key,task,fold) for key in recipes for task in t.TASKS for fold in range(5)])
        for key in recipes:t.aggregate(key)

    def accept_parent(self):
        self.guard_clock('ft');t.check_prepared()
        identity=t.read(t.identity_path())
        if identity['model']!=t.m.CONFIG:raise ValueError('parent initialization model contract differs')
        common=t.torch.load(t.INITIAL,map_location='cpu',weights_only=False)
        if (common.get('plan')!=t.PLAN or common.get('revision')!='r6' or common.get('step')!=0 or common.get('seed')!=42 or common.get('config')!=t.m.CONFIG):raise ValueError('parent common step0 identity differs')
        for name,tensor in common['state_dict'].items():
            if not t.torch.is_tensor(tensor) or tensor.dtype!=t.torch.float32 or not t.torch.isfinite(tensor).all():raise ValueError('parent tensor dtype/finite mismatch '+name)
        qfp=t.torch.load(t.INITIAL.parent/'QFP_step0.pt',map_location='cpu',weights_only=False)
        q2={key.removeprefix('Q2.'):value for key,value in common['state_dict'].items() if key.startswith('Q2.')}
        if qfp.get('source')!='common_step0.Q2' or set(qfp['state_dict'])!=set(q2):raise ValueError('parent QFP source/key mismatch')
        for key,value in qfp['state_dict'].items():
            if value.shape!=q2[key].shape or value.dtype!=q2[key].dtype or not t.torch.equal(value,q2[key]):raise ValueError('parent QFP value/shape/dtype mismatch')
        del common,qfp,q2
        t.exclusive(t.OUT/'registry/recipes.json',copy.deepcopy(self.parent['registry']))
        t.exclusive(t.OUT/'initialization/source.json',dict(identity=str(t.identity_path()),common_step0=str(t.INITIAL),QFP=str(t.INITIAL.parent/'QFP_step0.pt'),
            original_initialize_exit=0,reinitialized=False,model_forward=False,checks='schema/key/shape/dtype/finite and parent QFP equals loaded Q2',time=t.now()))
        t.exclusive(t.OUT/'authorization/parent_evidence.json',dict(**self.parent,source=str(t.SOURCE)))
        t.write(t.OUT/'continuation_registration.json',dict(status='COMPLETE',time=t.now(),parent=str(t.SOURCE),initialize_again=False))

    def develop(self):
        order=[]
        for name,a,b in (('L_A8_B8',.08,.08),('L_A0_B8',0.,.08),('L_A8_B0',.08,0.),('L_A0_B0',0.,0.)):
            ft=dict(t.m.DEFAULT_FT,align=a,entropy=b)
            order.append(self.ft_recipe('A_'+name,'A',ft=ft,control=t.BASELINE))
        self.execute_ft();a=t.select_stage('A',order,t.BASELINE);order=[]
        for name,lr,sched in (('U_HI_RESTART',.001,'restart'),('U_LO_RESTART',.0003,'restart'),('U_HI_MONO',.001,'mono'),('U_LO_MONO',.0003,'mono')):
            order.append(self.descendant('B_'+name,'B',a,a,joint_lr=lr,scheduler=sched))
        self.execute_ft();b=t.select_stage('B',order,a);order=[]
        for name,p2,p3 in (('D2_0_D3_0',0.,0.),('D2_1_D3_0',.1,0.),('D2_0_D3_1',0.,.1),('D2_1_D3_1',.1,.1)):
            order.append(self.descendant('C_'+name,'C',b,b,p2=p2,p3=p3))
        self.execute_ft();c=t.select_stage('C',order,b);order=[]
        for name,pfp,phead in (('DF_2_H5',.2,.5),('DF_0_H5',0.,.5),('DF_2_H1',.2,.1),('DF_0_H1',0.,.1)):
            order.append(self.descendant('D_'+name,'D',c,c,pfp=pfp,phead=phead))
        self.execute_ft();d=t.select_stage('D',order,c);order=[];ft=t.get_recipe(d)['ft'];pt_jobs=[]
        for name in t.m.PT_ARMS:
            pt_id='E_'+name;pt=t.m.pt_config(name);t.register_pt(pt_id,pt);pt_jobs.append(('pt',pt_id,None,None))
        self.execute_batch(pt_jobs)
        for name in t.m.PT_ARMS:
            pt_id='E_'+name;pt=t.registry()['pt'][pt_id]
            order.append(self.ft_recipe('E_'+name,'E',pt_id,pt,ft,d,'E_P_CURRENT_SHARED'))
        self.execute_ft();e=t.select_stage('E',order,'E_P_CURRENT_SHARED');order=[];selected=t.get_recipe(e);pt_jobs=[];slots=[]
        for name,pn,pfp in (('PD_N1_F2',.1,.2),('PD_N0_F2',0.,.2),('PD_N1_F0',.1,0.),('PD_N0_F0',0.,0.)):
            pt=copy.deepcopy(selected['pt']);pt.update(pnative=pn,pfp=pfp)
            if name=='PD_N1_F2':pt_id=selected['pt_id']
            else:
                pt_id='F_'+name;t.register_pt(pt_id,pt);pt_jobs.append(('pt',pt_id,None,None))
            slots.append((name,pt_id,pt))
        self.execute_batch(pt_jobs)
        for name,pt_id,pt in slots:order.append(self.ft_recipe('F_'+name,'F',pt_id,pt,ft,e,e))
        self.execute_ft();t.select_stage('F',order,e)
        g=t.select_stage('G_PARENT',list(t.registry()['recipes']),t.BASELINE);order=[]
        for name,fusion in (('G_GATE','gate'),('G_RESIDUAL','residual'),('G_MODALNORM','modalnorm')):order.append(self.descendant(name,'G',g,g,fusion=fusion))
        self.execute_ft();record=t.registry()
        record['selections']['G']=dict(validation_completed=True,diagnostic_order=order,validation={key:t.aggregate(key) for key in order},time=t.now(),
            selection='no separate G winner; FINAL uses global pool and R6_BASE_REFT guard exactly once')
        t.write(t.OUT/'registry/recipes.json',record);t.write(t.OUT/'selection/G.json',record['selections']['G'])

    def stop_owned(self):
        self.stopping=True;t.ignore_termination();signal.setitimer(signal.ITIMER_REAL,0)
        groups=[]
        for job in self.jobs:
            if job['exit'] is not None:continue
            group=owned_group(job)
            if group is not None:
                try:os.killpg(group,signal.SIGTERM);groups.append((job,group))
                except ProcessLookupError:pass
        until=time.monotonic()+30
        while time.monotonic()<until and any(j['exit'] is None for j,g in groups):
            self.refresh();time.sleep(1)
        for job,group in groups:
            if job['exit'] is None and owned_group(job) is not None:
                try:os.killpg(group,signal.SIGKILL)
                except ProcessLookupError:pass
        until=time.monotonic()+10
        while time.monotonic()<until and any(j['exit'] is None for j,g in groups):self.refresh();time.sleep(1)

    def report(self,status,error=None):
        result=super().report(status,error)
        result.update(resource_revision=RESOURCE_REVISION,parent_run=str(t.SOURCE),parent_budget_source=str(t.SOURCE/'budget.json'),
            parent_controller_exit=1,parent_first_ft_exit=1,parent_initialization_exit=0,parent_stop_source=str(t.SOURCE/'authorization/user_stop.json'),
            original_t0=T0,inherited_deadline=DEADLINE,resources=dict(physical_gpus=[2],caps=CAPS,phase_exclusive=True))
        t.write(t.OUT/'results.json',result);t.write(t.OUT/'final_review/results.json',result)
        for name in ('report.md','final_review/report.md'):
            with (t.OUT/name).open('a') as f:f.write('\n资源修订：'+RESOURCE_REVISION+'。父运行用户停止与原exit1保留；成功初始化exit0只读复用。累计预算包含父失败预留70 epochs／700 updates及本次补单元。父证据入口：'+str(t.SOURCE)+'。原t0与截止不刷新。\n')
        return result

    def run(self):
        error=None;rc=1;status='失败停止，待审查'
        try:
            inventory=gpu_inventory();self.manifest['gpu_uuid']=inventory['uuid'];t.write(t.OUT/'execution_manifest.json',self.manifest)
            t.exclusive(t.OUT/'startup_gpu_inventory.json',inventory)
            t.update_plan('执行中','接受父初始化与停止现场',f'继承t0={self.manifest["started"]}；截止={self.manifest["deadline"]}；原初始化不重做。')
            self.accept_parent();self.develop();self.guard_clock('test');frozen=t.freeze()
            t.update_plan('执行中','唯一冻结完成',f'FINAL={frozen["FINAL"]}；固定{len(frozen["units"])}个估计器开始正式TEST。')
            self.execute_batch([('test',key,task,fold) for key in t.registry()['recipes'] for task in t.TASKS for fold in range(5)])
            self.report('执行中');self.execute_batch([('cpu',None,None,None)])
            status='完成，待审查';rc=0
        except BaseException:
            error=traceback.format_exc()
            if isinstance(sys.exc_info()[1],serial.Deadline):status='截止停止，部分完成，待审查'
            try:self.stop_owned()
            except BaseException:error+='\nowned cleanup failure: '+traceback.format_exc()
            t.write(t.OUT/'failure.json',dict(error=error,time=t.now(),automatic_retry=False,automatic_resume=False))
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        try:self.report(status,error)
        except BaseException:
            error=(error or '')+'\ncloseout report failure: '+traceback.format_exc();rc=1;status='失败停止，待审查'
            t.write(t.OUT/'final_review/report_failure.json',dict(error=error,time=t.now()));t.update_plan(status,'交付写入阻断',error.splitlines()[-1])
        self.manifest.update(status=status,run_outcome='COMPLETE' if rc==0 else 'STOPPED',ended=t.now(),elapsed_seconds=time.time()-T0,error=error)
        t.write(t.OUT/'execution_manifest.json',self.manifest)
        (t.OUT/'final_review/closeout.md').write_text('状态：'+status+'；independent_review=PENDING。\ncontroller真实exit='+str(rc)+'。\n'+
            '原初始化exit0、原controller/FT exit1及用户停止记录保留；累计账本包含原失败预留，不退款。\n'+
            '未完成时为未完成／不可比较，不补CPU、不自动retry/resume/修复、不扩时。\n阻断：'+(error or '无')+'\n'+
            ('下一步：独立审查。\n' if rc==0 else '下一步：只读审查事故；任何修复或恢复需要新的明确授权。\n'))
        self.capture_diff();t.write(t.OUT/'controller_exit.json',dict(exit=rc,time=t.now(),process=t.process_identity(os.getpid())))
        atomic_exit(t.LOG/'controller.exit.txt',rc);return rc

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('controller','ft','test','wrap'))
    parser.add_argument('--accept-registration',action='store_true');parser.add_argument('--arm');parser.add_argument('--task',choices=t.TASKS)
    parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--physical-gpu',type=int,choices=(2,),default=2);parser.add_argument('--spec')
    args=parser.parse_args();t.torch.set_num_threads(2)
    if args.action=='wrap':return serial.wrap(args.spec)
    def stop(sig,frame):t.ignore_termination();raise RuntimeError('controlled termination '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.action=='controller':
        if not args.accept_registration:parser.error('one fresh authorized registration is required')
        return Controller().run()
    if args.arm is None or args.task is None or args.fold is None:parser.error('registered estimator identity required')
    t.unit(args.arm,args.task,args.fold,args.action=='test',args.physical_gpu);return 0

if __name__=='__main__':raise SystemExit(main())
