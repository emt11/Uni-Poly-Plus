#!/usr/bin/env python3
"""One exclusive four-GPU takeover dispatcher for the existing retest campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',required=True,choices=['r1_retest1']);parser.add_argument('--cycle-root',required=True);parser.add_argument('--log-root',required=True)
parser.add_argument('--takeover',action='store_true',required=True)
parser.add_argument('--retry-units',default='');parser.add_argument('--share-with-pt',action='store_true');parser.add_argument('--handoff-tag',default='v1')
parser.add_argument('--drain-01-at',help='Timezone-aware deadline: stop GPU0/1 dispatch; no new PT on that pair from startup')
args=parser.parse_args()
from src.training.ua2d_sota_context import configure
configure(args.campaign,args.cycle_root,args.log_root)
os.environ.update(UA2D_SOTA_ROLE='controller',UA2D_SOTA_RESOURCE_REVISION='four_gpu_v1')
from src.training.ua2d_sota import *
from src.training.ua2d_sota_scheduler import (GPUS,PT_PAIRS,RESOURCE_REVISION,classify_attempt,verified_process,occupied_gpus,available_pair)
PT_ARMS=('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH')
INITIAL_ARMS=('T0',*PARAMS,'A')
JOBS={};PORTS=set();LEASE=None
RETRY_UNITS=tuple(s for s in args.retry_units.split(',') if s)
HANDOFF=OUT/('handoff_four_gpu_'+args.handoff_tag)
DRAIN_AT=datetime.datetime.fromisoformat(args.drain_01_at) if args.drain_01_at else None
if DRAIN_AT is not None and DRAIN_AT.utcoffset() is None:
    raise ValueError('--drain-01-at requires an explicit timezone')

def dispatch_gpus(kind):
    if DRAIN_AT is not None and (kind=='pt' or datetime.datetime.now(datetime.timezone.utc)>=DRAIN_AT):
        return (2,3)
    return GPUS

def drain_status():
    if DRAIN_AT is None: return None
    remaining=[j['name'] for j in active_jobs() if any(g in (0,1) for g in j['gpus'])]
    due=datetime.datetime.now(datetime.timezone.utc)>=DRAIN_AT
    state='scheduled' if not due else ('draining' if remaining else 'campaign_released')
    return dict(deadline=DRAIN_AT.isoformat(),timezone='Asia/Shanghai',state=state,
        active_on_gpu01=remaining,new_pt_gpus=[2,3],dispatch_gpus=list(dispatch_gpus('ft')),
        forced_stop=False,time=now())

def other_dispatchers():
    result=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid(): continue
        try:
            argv=[s.decode() for s in (d/'cmdline').read_bytes().split(b'\0') if s]
            if not argv or not Path(argv[0]).name.startswith('python') or '--campaign' not in argv: continue
            if argv[argv.index('--campaign')+1]!=CAMPAIGN: continue
            if any(s.endswith(('scripts/run_ua2d_sota_campaign.py','scripts/run_ua2d_sota_four_gpu.py')) for s in argv):
                result.append(dict(pid=int(d.name),argv=argv,cwd=str((d/'cwd').resolve())))
        except (OSError,ValueError,UnicodeError): pass
    return result

def gpu_counts():
    result={}
    for gpu in GPUS:
        raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        lines=[s.strip() for s in raw.splitlines() if s.strip()]
        if any(not s.isdigit() for s in lines): raise RuntimeError('unrecognized GPU context query')
        result[gpu]=len(set(lines))
    return result

def worker_args(unit,arm='UA',task=None,fold=None,stage='development',round_name=None,strategy=None):
    command=[sys.executable,'scripts/run_ua2d_sota.py','--plan','UA2D-SOTA-20261018-01','--revision','r1',
        '--campaign',CAMPAIGN,'--cycle-root',str(OUT),'--log-root',str(LOG),'--stage',stage,'--unit',unit,'--arm',arm]
    if task is not None: command+=['--task',task,'--fold',str(fold)]
    if round_name is not None: command+=['--round',round_name,'--strategy',strategy]
    return command

def prepare_retry(name,output):
    """Archive the interrupted attempt, then let the same unit start a fresh attempt."""
    folder=LOG/name;output=Path(output)
    if not (folder/'exit.txt').exists(): return
    code=int((folder/'exit.txt').read_text())
    if code==0: return
    n=1
    while (LOG/f'{name}.attempt{n}_interrupted').exists(): n+=1
    archived_log=LOG/f'{name}.attempt{n}_interrupted'
    folder.rename(archived_log)
    archived_output=None
    if output.exists():
        archived_output=output.with_name(f'{output.name}.attempt{n}_interrupted')
        if archived_output.exists(): raise RuntimeError('retry output archive exists: '+str(archived_output))
        output.rename(archived_output)
    path=OUT/'scheduling/retry_attempts.json'
    records=read(path) if path.exists() else []
    records.append(dict(time=now(),unit=name,exit=code,archived_log=str(archived_log.relative_to(ROOT)),
        archived_output=str(archived_output.relative_to(ROOT)) if archived_output else None,
        authorization='user 2026-10-18: immediate rerun, no idle GPU wait'))
    write(path,records)

def job(kind,arm='UA',task=None,fold=None,round_name=None,strategy=None):
    if kind=='ft':
        name=f'ft_{arm}_{task}_{fold}';output=unit_path(arm,task,fold);success=output/'metrics.json'
        command=worker_args('ft',arm,task,fold)
    elif kind=='pt':
        name='pt_'+arm;output=OUT/'pretrain'/arm;success=output/'runtime.json';command=None
    elif kind=='neural_formal':
        name=f'test_{arm}_{task}_{fold}';output=OUT/'formal/neural'/arm/task/f'fold{fold}';success=output/'metrics.json'
        command=worker_args(kind,arm,task,fold,'formal')
    elif kind=='kernel':
        name='kernel_'+round_name;output=OUT/'kernel'/round_name;success=output/'summary.json';command=worker_args(round_name,arm)
    elif kind=='kernel_formal':
        name=f'kernel_test_{round_name}_{strategy}_{task}_{fold}';output=OUT/'formal/kernel'/round_name/strategy/task/f'fold{fold}'
        success=output/'metrics.json';command=worker_args(kind,arm,task,fold,'formal',round_name,strategy)
    elif kind=='cpu':
        name='final_cpu';output=OUT/'final_review';success=LOG/name/'stdout.log'
        command=['timeout','300s',sys.executable,'-m','pytest','-q','tests/test_ua2d_sota_plan.py']
    else: raise ValueError(kind)
    if name in JOBS: return JOBS[name]
    if name in RETRY_UNITS: prepare_retry(name,output)
    folder=LOG/name;status=classify_attempt(folder,success,output if kind!='cpu' else None)
    j=dict(name=name,kind=kind,arm=arm,task=task,fold=fold,round=round_name,strategy=strategy,
        output=str(output),success=str(success),log=str(folder),status=status,gpus=[],command=command,adopted=status!='pending')
    j['share_with_pt']=bool(args.share_with_pt and name in RETRY_UNITS)
    if (folder/'command.json').exists():
        record=read(folder/'command.json');j['gpus']=record.get('gpus',[]);j['command']=record['argv']
        if record.get('cwd')!=str(ROOT): raise RuntimeError('launched job cwd mismatch: '+name)
    if status=='success': validate_success(j)
    elif status=='active': attest(j,adopting=True)
    if status.startswith('failed') or status in ('interrupted','unowned_output'):
        j['no_retry']=True
        if (folder/'exit.txt').exists(): j['exit']=int((folder/'exit.txt').read_text())
    JOBS[name]=j
    if j['status']=='unverified_active': raise RuntimeError('unverified existing writer; preserve process and stop dispatch: '+name)
    return j

def validate_success(j):
    p=Path(j['success'])
    if j['kind']=='cpu': return
    record=read(p)
    if record.get('status')!='PASS': j.update(status='failed_success_metadata',no_retry=True);return
    if j['kind']=='pt':
        if record.get('completed_steps')!=5000 or not (Path(j['output'])/'deploy_05000.pt').exists(): j.update(status='failed_parent',no_retry=True)
    elif j['kind']=='kernel':
        if record.get('feature_units')!=35 or record.get('fits')!=315 or record.get('outer_test')!='NOT_RUN': j.update(status='failed_kernel_summary',no_retry=True)
    else:
        if (record.get('arm'),record.get('task'),record.get('fold'))!=(j['arm'],j['task'],j['fold']):
            j.update(status='failed_unit_identity',no_retry=True)
        if j['kind']=='ft' and (record.get('epochs')!=70 or record.get('outer_test')!='NOT_RUN'):
            j.update(status='failed_ft_completion',no_retry=True)

def attest(j,adopting=False):
    folder=Path(j['log'])
    if not (folder/'pid.txt').exists():
        if adopting: j.update(status='unverified_active',no_retry=True,reason='launched job has no pid/exit evidence')
        return
    pid=int((folder/'pid.txt').read_text())
    identity=verified_process(pid,j['command'],ROOT,','.join(map(str,j['gpus'])),birth=j.get('identity',{}).get('start_ticks'),
        stdout=folder/'stdout.log',stderr=folder/'stderr.log')
    if identity:
        j['identity']=identity;j.pop('missing_since',None)
    elif adopting: j.update(status='unverified_active',no_retry=True,reason='cannot attest live launch; do not restart')
    else:
        j.setdefault('missing_since',time.monotonic())
        if time.monotonic()-j['missing_since']>15: j.update(status='lost_without_exit',no_retry=True,reason='process gone or identity changed without true exit record')

def active_jobs(): return [j for j in JOBS.values() if j['status']=='active']
def terminal(j): return j['status'] not in ('pending','active')
def success(j): return j['status']=='success'

def poll_jobs():
    for j in active_jobs():
        folder=Path(j['log'])
        if (folder/'exit.txt').exists():
            code=int((folder/'exit.txt').read_text());j['exit']=code
            j['status']=classify_attempt(folder,j['success'])
            if j['status']=='success': validate_success(j)
            if j['status']!='success': j['no_retry']=True
        else: attest(j)

def rendezvous(arm):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    while port in PORTS:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    PORTS.add(port)
    return port,f'{CAMPAIGN}-{arm}-four-gpu'

def launch(j,gpus=()):
    if j['status']!='pending': raise RuntimeError('refuse repeat launch '+j['name'])
    # Recheck at launch as the deadline may pass during a queue iteration.
    if any(g not in dispatch_gpus(j['kind']) for g in gpus): return False
    dest=Path(j['log']);dest.mkdir(parents=True,exist_ok=False)
    command=j['command'];rdzv=None
    if j['kind']=='pt':
        port,run_id=rendezvous(j['arm']);rdzv=dict(port=port,run_id=run_id,world_size=2)
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
            '--rdzv_backend=c10d',f'--rdzv_endpoint=127.0.0.1:{port}',f'--rdzv_id={run_id}',*worker_args('pt',j['arm'])[1:]]
    j.update(command=command,gpus=list(gpus))
    window='fourgpu_'+j['name']
    write(dest/'command.json',dict(argv=command,cwd=str(ROOT),campaign=CAMPAIGN,cycle_root=str(OUT),log_root=str(LOG),
        gpus=list(gpus),threads=4,window=window,time=now(),resource_revision=RESOURCE_REVISION,rendezvous=rdzv))
    env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        UA2D_SOTA_CAMPAIGN=CAMPAIGN,UA2D_SOTA_CYCLE_ROOT=str(OUT),UA2D_SOTA_LOG_ROOT=str(LOG),UA2D_SOTA_ROLE='worker',UA2D_SOTA_RESOURCE_REVISION=RESOURCE_REVISION)
    script=dest/'run.sh'
    script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\n'+
        '\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'+
        'pwd > '+shlex.quote(str(dest/'cwd.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'start.txt'))+'\n'+
        'setsid '+shlex.join(command)+' > '+shlex.quote(str(dest/'stdout.log'))+' 2> '+shlex.quote(str(dest/'stderr.log'))+' &\n'+
        'scheduler_pid=$!\nprintf "%s\\n" "$scheduler_pid" > '+shlex.quote(str(dest/'pid.txt'))+'\nwait "$scheduler_pid"\n'+
        'scheduler_exit=$?\nprintf "%s\\n" "$scheduler_exit" > '+shlex.quote(str(dest/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'end.txt'))+'\nexit "$scheduler_exit"\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',window,'bash '+shlex.quote(str(script))],check=True)
    j.update(status='active',launched_at=now())
    return True

def update_plan(status,detail):
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text();a=text.index('- 状态：');b=text.index('\n',a)
    text=text[:a]+f'- 状态：**{status}（r1_retest1，四卡接管）**。用户已授权继续四卡接管、立即补跑 4 个 P_HLR/Egc 中断单元（与 PT 同卡并行），并授权将本批 FT 上限调至 424 starts／29600 epochs 以保证完整执行；科学矩阵、训练配方与其余上限不变。'+text[b:]
    marker='### r1_retest1 四卡接管状态'
    if marker in text:
        a=text.index(marker);b=text.index('\n## 13.',a);text=text[:a]+text[b:]
    at=text.index('\n## 13.');b=read(OUT/'budget.json')
    body=marker+'\n\n'+f'- {now()}：{detail}\n'
    body+='- 单一 dispatcher／预算 writer 为 `scripts/run_ua2d_sota_four_gpu.py`，运行于 `Uni-Poly` 的 controller window，持有本批 controller.lock；旧 r1/r2 及旧 retest controller 不恢复。\n'
    body+='- FT／独立预测：GPU0—3，每卡至多2路；PT 每作业两 rank，固定 microbatch84／accumulation6／global_batch1008，在(0,1)/(2,3)独立进程组和独立 rendezvous 端口运行。PT 卡不混跑 FT，唯一例外是本次授权的 4 个 P_HLR/Egc 补跑单元（每卡 1 PT rank + 1 FT）；就绪 FT 跨 arm 补位，按 parent 依赖释放。\n'
    body+='- 四个 P_HLR/Egc folds0—3 的中断尝试（真实 exit143）已归档为 *.attempt1_interrupted；用户 2026-10-18 授权立即补跑且不等待 GPU 空闲，补跑与 PT 同卡并行并按完整 70 epochs 完成（均 exit 0），中断尝试与补跑消耗都计入累计预算。\n'
    body+=f'- 累计实际值见 `{OUT.relative_to(ROOT)}/budget.json`：{json.dumps(b["actual"],ensure_ascii=False)}。接管前账本、单元归属、资源变更和源码差异见 `handoff_four_gpu_{args.handoff_tag}/`；不清零、不兑换或扩容。\n'
    body+='- 当前队列及每个单元的 success／interrupted／pending／active 状态见 `scheduling/four_gpu_jobs.json`；未知 PID 映射不推测、不重启。唯一 CPU suite 留至本次授权执行结束，不增加 smoke／benchmark／FP suite。\n'
    if DRAIN_AT is not None:
        body+=f'- 用户授权定时退卡：{DRAIN_AT.isoformat()}（北京时间）起 GPU0、1 停止全部新增派发；接管后不再向(0,1)派发新 PT，当前 PT 自然完成后仅用 FT 填充截止前空档。截止后等待在途单元真实退出，不杀 worker、不迁卡、不重试；GPU2、3继续，资源与排空状态见 `scheduling/gpu_drain_status.json`。08:00是释放目标，不是强杀期限；科学矩阵与累计上限不变。\n'
    review=OUT/'final_review/seven_task_comparison.md'
    if status!='执行中' and review.exists(): body+='\n'+review.read_text()+'\n'
    body+='\n- 已有执行记录与停止现场保留；执行自检不等于独立审查，不归档 PROJECT_HISTORY.md。\n'
    path.write_text(text[:at]+'\n'+body+text[at:])

def publish(phase):
    refresh_budget(publish=True)
    drain=drain_status()
    if drain is not None: write(OUT/'scheduling/gpu_drain_status.json',drain)
    write(OUT/'scheduling/four_gpu_jobs.json',dict(time=now(),phase=phase,jobs=list(JOBS.values()),
        resources=dict(gpus=dispatch_gpus('ft'),pt_pairs=[p for p in PT_PAIRS if all(g in dispatch_gpus('pt') for g in p)],max_per_gpu=2,world_size=2,revision=RESOURCE_REVISION),drain=drain))
    statuses={s:sum(j['status']==s for j in JOBS.values()) for s in {j['status'] for j in JOBS.values()}}
    write(OUT/'phase.json',dict(phase=phase,status='执行中',time=now(),states=statuses,
        active=[{k:j[k] for k in ('name','kind','gpus')} for j in active_jobs()],controller_pid=os.getpid(),resource_revision=RESOURCE_REVISION,drain=drain))

def block(j,reason):
    if j['status']=='pending': j.update(status='blocked_dependency',reason=reason,no_retry=True)

def all_ft(arm): return [job('ft',arm,t,f) for t in TASKS for f in range(5)]

def run_development():
    f0=job('kernel',round_name='F0')
    initial={a:all_ft(a) for a in INITIAL_ARMS}
    pts={a:job('pt',a) for a in PT_ARMS};downstream={a:all_ft(a) for a in PT_ARMS}
    write(HANDOFF/'adoption_start.json',dict(time=now(),jobs=list(JOBS.values()),
        counts={state:sum(j['status']==state for j in JOBS.values()) for state in {j['status'] for j in JOBS.values()}},
        no_completed_unit_relaunched=True,no_interrupted_unit_retried=False,retried_units=list(RETRY_UNITS)))
    while True:
        poll_jobs();contexts=gpu_counts();active=active_jobs()
        t0_ok=all(success(j) for j in initial['T0'])
        t0_done=all(terminal(j) for j in initial['T0'])
        if t0_done and not t0_ok:
            for arm in PT_ARMS:
                block(pts[arm],'T0 complete matched control unavailable')
                for j in downstream[arm]: block(j,'T0 complete matched control unavailable')
        for arm in PT_ARMS:
            if terminal(pts[arm]) and not success(pts[arm]):
                for j in downstream[arm]: block(j,'PT parent failed or incomplete; no retry')
        ready_pt=[j for j in pts.values() if j['status']=='pending' and t0_ok]
        # Reserve whole pairs for ready PT; drain existing FT naturally, then launch.
        reserved=set()
        pt_occupied,_=occupied_gpus(active)
        for pair in PT_PAIRS:
            if not ready_pt: break
            if any(g not in dispatch_gpus('pt') for g in pair): continue
            if any(g in pt_occupied for g in pair): continue
            reserved.update(pair)
            if available_pair(pair,active,contexts):
                j=ready_pt.pop(0)
                if refresh_budget(publish=True)['pt_starts']>=5: raise RuntimeError('PT starts budget exhausted')
                if launch(j,pair): active.append(j);pt_occupied.update(pair)
        # One kernel job at a time; GPU slot is reserved while it extracts/fits.
        cpu_busy=any(j['kind'] in ('kernel','kernel_formal','cpu') for j in active)
        pt_occupied,counts=occupied_gpus(active)
        if f0['status']=='pending' and not cpu_busy:
            gpu=next((g for g in dispatch_gpus('kernel') if g not in pt_occupied and g not in reserved and max(counts[g],contexts[g])<2),None)
            if gpu is not None and launch(f0,(gpu,)): active.append(f0)
        ready_ft=[j for arm in INITIAL_ARMS for j in initial[arm] if j['status']=='pending']
        ready_ft += [j for arm in PT_ARMS if success(pts[arm]) for j in downstream[arm] if j['status']=='pending']
        for gpu in dispatch_gpus('ft'):
            pt_occupied,counts=occupied_gpus(active)
            if gpu in reserved: continue
            on_pt=gpu in pt_occupied
            if on_pt and not args.share_with_pt: continue
            slots=2-max(counts[gpu],contexts[gpu])
            for _ in range(max(0,slots)):
                j=next((x for x in ready_ft if (not on_pt) or x.get('share_with_pt')),None)
                if j is None: break
                if refresh_budget(publish=True)['ft_starts']>=LIMITS['ft_starts']: raise RuntimeError('FT start budget exhausted')
                ready_ft.remove(j)
                if launch(j,(gpu,)): active.append(j)
        publish('four_gpu_development')
        pending=[j for j in JOBS.values() if j['status'] in ('pending','active')]
        if not pending: break
        time.sleep(5)
    passed={a:all(success(j) for j in initial[a]) for a in INITIAL_ARMS}
    passed.update({a:success(pts[a]) and all(success(j) for j in downstream[a]) for a in PT_ARMS});passed['F0']=success(f0)
    return passed


def run_gpu_queue(jobs,phase):
    """No PT is created here; use all free cards for conditional FT or formal."""
    while any(j['status'] in ('pending','active') for j in jobs):
        poll_jobs();contexts=gpu_counts();active=active_jobs();pt,counts=occupied_gpus(active)
        ready=[j for j in jobs if j['status']=='pending']
        for gpu in dispatch_gpus('ft'):
            if gpu in pt: continue
            for _ in range(max(0,2-max(counts[gpu],contexts[gpu]))):
                if not ready: break
                j=ready.pop(0)
                if launch(j,(gpu,)): active.append(j)
        publish(phase);time.sleep(5)
    return all(success(j) for j in jobs)


def run_cpu_job(j,phase,gpus=()):
    if j['status']=='pending':
        if j['kind']=='kernel':
            while True:
                poll_jobs();contexts=gpu_counts();pt,counts=occupied_gpus(active_jobs())
                gpu=next((g for g in dispatch_gpus('kernel') if g not in pt and max(contexts[g],counts[g])<2),None)
                if gpu is not None: gpus=(gpu,);break
                publish(phase);time.sleep(5)
        if not launch(j,gpus):
            return run_cpu_job(j,phase)
    while j['status']=='active':
        poll_jobs();publish(phase)
        if j['status']=='active': time.sleep(5)
    return success(j)


def snapshot_sources():
    import difflib
    files=['scripts/run_ua2d_sota_four_gpu.py','src/training/ua2d_sota_scheduler.py','src/training/ua2d_sota_pt.py','tests/test_ua2d_sota_plan.py']
    diff=''
    for name in files:
        path=ROOT/name;dest=HANDOFF/'source_used'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(path.read_text())
        previous=OUT/'source_used'/name
        before=previous.read_text() if previous.exists() else ''
        diff+=''.join(difflib.unified_diff(before.splitlines(True),path.read_text().splitlines(True),fromfile='before_four_gpu/'+name,tofile='after_four_gpu/'+name))
    (HANDOFF/'scheduler.diff').write_text(diff)


def finish(passed):
    required=(*PARAMS,'A','T0',*PT_ARMS)
    complete=False
    if all(passed.get(a) for a in required):
        proposal=propose_integration()
        required_i=len(TASKS)*5
        remaining=LIMITS['ft_starts']-refresh_budget(publish=False)['ft_starts']
        if proposal['run'] and remaining<required_i:
            # Authorized retries consume FT-start budget; never start a partial I block.
            passed['I']=False
            write(OUT/'development/I_blocked_budget.json',dict(status='阻断',reason='FT start budget cannot cover a complete I block after the authorized retries',
                remaining=remaining,required=required_i,retries=list(RETRY_UNITS),time=now()))
            write(OUT/'development/global_selection_blocked.json',dict(status='阻断',missing_arms=['I_budget'],
                reason='authorized P_HLR/Egc retries consumed FT start budget; I needs 35 starts; no partial family, no freeze or TEST',time=now()))
        elif proposal['run']:
            passed['I']=run_gpu_queue(all_ft('I'),'integration')
        else:
            passed['I']=True
        if not proposal['run']: write(OUT/'skipped/I.json',dict(reason='no eligible change or equivalent candidate',proposal=proposal,time=now()))
        if passed['I']:
            core=choose_core()['core']
            values,units=validation_scores()
            write(OUT/'development/validation_summary.json',dict(scores=values,units=units,integration=proposal,core=core,outer_test='NOT_RUN',time=now()))
            passed['F1']=run_cpu_job(job('kernel',core,round_name='F1'),'F1',(0,)) if core!='UA' else passed['F0']
            if core=='UA': write(OUT/'skipped/F1.json',dict(reason='reuse this campaign F0 for identical UA core',time=now()))
            if passed['F0'] and passed['F1']:
                selection=read(OUT/'selection_before_test.json') if (OUT/'selection_before_test.json').exists() else freeze_selection()
                neural=[job('neural_formal',a,t,f) for a in ['UA',*selection['neural_arms']] for t in TASKS for f in range(5)]
                formal_ok=run_gpu_queue(neural,'four_gpu_formal')
                for k in selection['kernel_strategies']:
                    for t in TASKS:
                        for f in range(5):
                            j=job('kernel_formal',k['arm'],t,f,k['round'],k['strategy'])
                            formal_ok=run_cpu_job(j,'kernel_formal') and formal_ok
                complete=formal_ok
    else:
        missing=[a for a in required if not passed.get(a)]
        write(OUT/'development/global_selection_blocked.json',dict(status='阻断',missing_arms=missing,
            reason='incomplete fixed development matrix; no partial-family ranking, no I/core selection, no freeze or TEST',time=now()))
        values,units=validation_scores()
        write(OUT/'development/validation_summary.json',dict(scores=values,units=units,missing_arms=missing,outer_test='NOT_RUN',time=now()))
    write(OUT/'development/stage_outcomes.json',dict(passed=passed,formal_complete=complete,resource_revision=RESOURCE_REVISION,time=now()))
    summarize_results()
    cpu=job('cpu');cpu_ok=run_cpu_job(cpu,'final_cpu')
    folder=Path(cpu['log']);review=OUT/'final_review'
    (review/'final_cpu.log').write_text((folder/'stdout.log').read_text()+(folder/'stderr.log').read_text())
    (review/'final_cpu_exit.txt').write_text((folder/'exit.txt').read_text())
    # Preserve original implementation evidence; append the scheduling patch separately.
    (review/'four_gpu_scheduler.diff').write_text((HANDOFF/'scheduler.diff').read_text())
    refresh_budget(publish=True)
    status='待审查' if complete and cpu_ok else ('需返修' if not cpu_ok else '阻断')
    m=read(OUT/'execution_manifest.json');m.update(status=status,end=now(),formal_complete=complete,cpu_pass=cpu_ok,
        controller_exit=0 if complete and cpu_ok else 1,outer_test='RUN' if (OUT/'selection_before_test.json').exists() else 'NOT_RUN')
    write(OUT/'execution_manifest.json',m);update_plan(status,'独立授权工作已结束；缺项、中断及正式阶段阻断按实际记录，等待独立审查。')
    publish('finished');write(OUT/'phase.json',dict(phase='finished',status=status,time=now(),controller_exit=m['controller_exit']))
    return m['controller_exit']


def authorize_plan_resources():
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text()
    old='- GPU 仅 2、3；PT 两 rank 各占一 GPU，PT 期间不与本计划 FT 同卡并跑；global batch 不因显存调整。FT 每 GPU 最多两个单元，总计最多四个，已有占用计入资源决策，不能结束其他工作来腾卡。'
    new='- 用户四卡接管授权覆盖原 GPU2/3 限制：FT／独立预测使用 GPU0、1、2、3，每卡最多2路、合计最多8路；PT 仍每项 world_size=2、microbatch84、accumulation6、global_batch1008，在(0,1)和(2,3)两组上至多同时运行两个不同候选。各 PT 独立 torchrun 进程组、rendezvous 端口及 ALIGN 负样本池，绝不合成4-rank作业。PT 卡不混跑 FT；其他卡可运行已就绪 FT。已有占用计入资源决策，不终止无关进程。'
    text=text.replace(old,new,1)
    old='执行顺序固定：**绑定已有 reference → F0 → P → A → T0/T → M → I → F1 → 冻结名单 → 正式 TEST → 最终一次 CPU 测试 → 交付审查**。'
    new='科学依赖保持：**绑定 reference → F0；P/A/T0/T/M 开发 → I → F1 → 冻结名单 → 正式 TEST → 最终一次 CPU 测试 → 交付审查**。按用户四卡接管指令，没有科学依赖的 P/A/T0 单元使用统一就绪队列；PT 保持共同 step0 与匹配控制，PT→对应 FT 的 parent 依赖、I／F1／全局冻结的完整性条件不变。完成时间顺序不再要求逐 arm 串行。'
    text=text.replace(old,new,1)
    text=text.replace('GPU仅2/3，PT两rank独占，FT每GPU最多两路；CPU KRR顺序、至多4threads、workers0。','GPU0—3，FT/独立预测每GPU最多两路；PT每项固定两rank，可在(0,1)/(2,3)独立组上同时运行两个候选，PT卡不混跑FT；累计科学预算不增加。CPU KRR顺序、至多4threads、workers0。',1)
    text=text.replace('当前 P_BLR，实时计数见本批 budget.json','四卡接管后执行统一就绪队列；四个 P_HLR/Egc 中断单元不重试，实时计数见本批 budget.json',1)
    path.write_text(text)


def execute():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    global LEASE
    if not OUT.is_dir() or read(OUT/'execution_manifest.json').get('campaign')!=CAMPAIGN: raise ValueError('existing campaign identity required')
    LEASE=(OUT/'controller.lock').open('a+')
    fcntl.flock(LEASE,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if other_dispatchers(): raise RuntimeError('another campaign dispatcher still exists; no dual writer allowed')
    HANDOFF.mkdir(parents=True,exist_ok=False)
    (HANDOFF/'CURRENT_PLAN.before.md').write_text((ROOT/'CURRENT_PLAN.md').read_text())
    write(HANDOFF/'budget_before.json',read(OUT/'budget.json'));write(HANDOFF/'manifest_before.json',read(OUT/'execution_manifest.json'))
    authorize_plan_resources()
    previous=read(OUT/'execution_manifest.json')
    manifest=dict(previous);manifest.pop('controller_exit',None)
    manifest.update(status='执行中',pid=os.getpid(),active_controller='four_gpu_v1',resumed_at=now(),
        controller_log=str(LOG/('controller_four_gpu_'+args.handoff_tag)),resources=dict(gpus=list(GPUS),pt_pairs=PT_PAIRS,pt_ranks=2,ft_slots_per_gpu=2,blas_threads=4,workers=0),
        resource_authorization='user four-GPU instruction; 2026-10-18 user reply authorizes immediate rerun of the 4 interrupted P_HLR/Egc units, sharing GPUs with running PT, no idle wait')
    if DRAIN_AT is not None:
        manifest['drain_01_at']=DRAIN_AT.isoformat()
        manifest['resources']['new_pt_pairs']=[[2,3]]
        manifest['resource_authorization']+='; user schedules GPU0/1 drain 2026-10-19 07:30 Asia/Shanghai, preserve in-flight jobs'
    write(OUT/'execution_manifest.json',manifest)
    if DRAIN_AT is not None:
        write(OUT/'scheduling/gpu_drain_plan.json',dict(deadline=DRAIN_AT.isoformat(),new_pt_gpus=[2,3],
            retiring_gpus=[0,1],remaining_gpus=[2,3],authorization='user: schedule GPU0/1 drain tomorrow 07:30',
            preserve_inflight=True,budget_unchanged=True,installed_at=now()))
    write(OUT/'controller_owner.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        argv=sys.argv,lock=str(OUT/'controller.lock'),resource_revision=RESOURCE_REVISION,time=now()))
    snapshot_sources();refresh_budget(publish=True)
    update_plan('执行中','定时排空接管生效；原 controller 已退出，在途 PT 身份核实后接续跟踪，成功单元复用，不新增重试；GPU0、1当前 PT 自然完成后仅接截止前 FT，后续 PT 只用GPU2、3。' if DRAIN_AT is not None else '已接管原批次；原 controller 已退出，既有成功单元保留；按用户 2026-10-18 授权，四个 P_HLR/Egc 中断单元归档后立即补跑并与 PT 同卡并行；统一就绪队列和独立双 rank PT 调度继续。')
    try:
        passed=run_development()
        write(HANDOFF/'adoption_end.json',dict(passed=passed,time=now()))
        return finish(passed)
    except BaseException as error:
        # Preserve verified independent workers; never relaunch them after a controller error.
        write(OUT/'four_gpu_controller_failure.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),retry=False,
            workers_preserved=[j['name'] for j in active_jobs()],time=now()))
        m=read(OUT/'execution_manifest.json');m.update(status='阻断',controller_exit=1,error=repr(error));write(OUT/'execution_manifest.json',m)
        refresh_budget(publish=True);update_plan('阻断','四卡 controller 异常，停止新派发；在途现场保留，无自动重试。'+repr(error))
        raise
if __name__=='__main__': sys.exit(execute())
