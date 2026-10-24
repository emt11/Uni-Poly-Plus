#!/usr/bin/env python3
"""Single bounded r1_retest1 controller; no resume, retries, or legacy candidate reuse."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser()
p.add_argument('--campaign',required=True,choices=['r1_retest1']);p.add_argument('--cycle-root',required=True);p.add_argument('--log-root',required=True)
args=p.parse_args()
from src.training.ua2d_sota_context import configure
configure(args.campaign,args.cycle_root,args.log_root)
os.environ['UA2D_SOTA_ROLE']='controller'
from src.training.ua2d_sota import *
ACTIVE=[]

def gpu_contexts(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return len([line for line in raw.splitlines() if line.strip()])

def idle(gpus):
    while any(gpu_contexts(g) for g in gpus):
        write(OUT/'resource_wait.json',dict(gpus=list(gpus),status='WAITING',time=now()))
        refresh_budget(publish=True);time.sleep(10)

def worker_args(unit,arm='UA',task=None,fold=None,stage='development',round_name=None,strategy=None):
    command=[sys.executable,'scripts/run_ua2d_sota.py','--plan','UA2D-SOTA-20261018-01','--revision','r1',
        '--campaign',CAMPAIGN,'--cycle-root',str(OUT),'--log-root',str(LOG),'--stage',stage,'--unit',unit,'--arm',arm]
    if task is not None: command+=['--task',task,'--fold',str(fold)]
    if round_name is not None: command+=['--round',round_name,'--strategy',strategy]
    return command

def launch(name,command,gpus=()):
    dest=LOG/name;dest.mkdir(parents=True,exist_ok=False)
    window='retest1_'+name
    write(dest/'command.json',dict(argv=command,cwd=str(ROOT),campaign=CAMPAIGN,cycle_root=str(OUT),log_root=str(LOG),
        gpus=list(gpus),threads=4,window=window,time=now()))
    env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        UA2D_SOTA_CAMPAIGN=CAMPAIGN,UA2D_SOTA_CYCLE_ROOT=str(OUT),UA2D_SOTA_LOG_ROOT=str(LOG),UA2D_SOTA_ROLE='worker')
    script=dest/'run.sh'
    script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\n'+
        '\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'+
        'pwd > '+shlex.quote(str(dest/'cwd.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'start.txt'))+'\n'+
        'setsid '+shlex.join(command)+' > '+shlex.quote(str(dest/'stdout.log'))+' 2> '+shlex.quote(str(dest/'stderr.log'))+' &\n'+
        'retest_pid=$!\nprintf "%s\\n" "$retest_pid" > '+shlex.quote(str(dest/'pid.txt'))+'\nwait "$retest_pid"\n'+
        'retest_exit=$?\nprintf "%s\\n" "$retest_exit" > '+shlex.quote(str(dest/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'end.txt'))+'\nexit "$retest_exit"\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',window,'bash '+shlex.quote(str(script))],check=True)
    ACTIVE.append(dest)
    return dest/'exit.txt'

def wait_exit(path):
    while not path.exists():
        refresh_budget(publish=True);time.sleep(5)
    refresh_budget(publish=True)
    return int(path.read_text())==0

def phase(name,detail):
    write(OUT/'phase.json',dict(phase=name,status='执行中',detail=detail,time=now()))
    refresh_budget(publish=True);update_execution_plan('执行中',detail)

def matrix(arm,formal=False):
    tag=('test_' if formal else 'ft_')+arm
    record=OUT/'dispatch'/f'{tag}.json'
    if record.exists(): raise FileExistsError('matrix already dispatched; no retry: '+tag)
    queue=iter((t,f) for t in TASKS for f in range(5));running={};done=[];end=False;failed=False
    phase(tag,('正式评估 ' if formal else '完整 FT ')+arm+'；其他范围按固定顺序排队。')
    while running or not end:
        for slot,(path,t,f) in list(running.items()):
            if path.exists():
                code=int(path.read_text());done.append(dict(task=t,fold=f,exit=code));del running[slot]
                if code: failed=True;end=True
        for gpu in (2,3):
            for number in (0,1):
                if end or (gpu,number) in running: continue
                count=sum(g==gpu for g,_ in running)
                if max(gpu_contexts(gpu),count)>=2: continue
                try: t,f=next(queue)
                except StopIteration: end=True;break
                actual=refresh_budget(publish=True)
                if not formal and actual['ft_starts']>=LIMITS['ft_starts']: raise RuntimeError('FT start budget exhausted')
                cmd=worker_args('neural_formal' if formal else 'ft',arm,t,f,'formal' if formal else 'development')
                running[gpu,number]=(launch(tag+'_'+t+'_'+str(f),cmd,(gpu,)),t,f)
        write(record,dict(arm=arm,formal=formal,status='执行中' if running or not end else ('阻断' if failed else 'PASS'),
            completed=done,active=[dict(gpu=k[0],slot=k[1],task=v[1],fold=v[2]) for k,v in running.items()],time=now(),retry=False))
        refresh_budget(publish=True)
        if running or not end: time.sleep(5)
    return not failed and len(done)==35

def kernel_round_job(round_name,arm):
    phase(round_name,f'{round_name}：{arm} 的 train/validation 图表示与固定三核／三 α；无 TEST。')
    idle((2,3))
    return wait_exit(launch('kernel_'+round_name,worker_args(round_name,arm),(2,)))

def pt_job(arm):
    phase('pt_'+arm,f'{arm}：两 rank、5000 updates；PT validation 和中间 FT 均为 0。')
    idle((2,3))
    if refresh_budget(publish=True)['pt_starts']>=5: raise RuntimeError('PT start budget exhausted')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',*worker_args('pt',arm)[1:]]
    return wait_exit(launch('pt_'+arm,command,(2,3)))

def capture_sources():
    import difflib
    files=['scripts/run_ua2d_sota_campaign.py','scripts/run_ua2d_sota.py','src/training/ua2d_sota_context.py',
        'src/training/ua2d_sota.py','src/training/ua2d_sota_pt.py','src/modules/glt_dual_pretrain.py','tests/test_ua2d_sota_plan.py']
    root=OUT/'source_used';root.mkdir(parents=True,exist_ok=False)
    diff=''
    for name in files:
        source=ROOT/name;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(source.read_text())
        tracked=subprocess.run(['git','show','HEAD:'+name],capture_output=True,text=True)
        before=tracked.stdout if tracked.returncode==0 else ''
        diff+=''.join(difflib.unified_diff(before.splitlines(True),source.read_text().splitlines(True),fromfile='HEAD/'+name,tofile='worktree/'+name))
    review=OUT/'final_review';review.mkdir(parents=True,exist_ok=True);(review/'implementation.diff').write_text(diff)
    write(review/'implementation_scope.json',dict(new_entry='scripts/run_ua2d_sota_campaign.py',new_context='src/training/ua2d_sota_context.py',
        modified=['scripts/run_ua2d_sota.py','src/training/ua2d_sota.py','src/training/ua2d_sota_pt.py','tests/test_ua2d_sota_plan.py'],
        preexisting='HEAD diff includes the preexisting untracked r1 implementation and glt_dual_pretrain modifications; not all lines originate in this retest turn',
        changes='campaign/root propagation, single budget writer, complete sequential controller and formal closeout, deployment packages, strict split freeze, direct CPU coverage',
        new_scientific_grid=False))

def final_cpu():
    phase('final_cpu','本批执行已结束；唯一一次 CPU suite，timeout 300s；不运行 FP 专项。')
    result=wait_exit(launch('final_cpu',['timeout','300s',sys.executable,'-m','pytest','-q','tests/test_ua2d_sota_plan.py']))
    dest=OUT/'final_review';folder=LOG/'final_cpu'
    (dest/'final_cpu.log').write_text((folder/'stdout.log').read_text()+(folder/'stderr.log').read_text())
    (dest/'final_cpu_exit.txt').write_text((folder/'exit.txt').read_text())
    return result

def stop_own_active():
    stopped=[]
    for dest in ACTIVE:
        if (dest/'exit.txt').exists() or not (dest/'pid.txt').exists(): continue
        pid=int((dest/'pid.txt').read_text())
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
            if '--campaign r1_retest1' in cmd or (dest.name=='final_cpu' and 'tests/test_ua2d_sota_plan.py' in cmd):
                os.killpg(pid,signal.SIGTERM);stopped.append(dict(pid=pid,command=cmd,log=str(dest)))
        except (FileNotFoundError,ProcessLookupError): pass
    write(OUT/'stopped_own_processes.json',dict(stopped=stopped,time=now(),unrelated_processes_touched=False))

def execute():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    if CAMPAIGN!='r1_retest1': raise ValueError('only explicitly authorized campaign')
    OUT.mkdir(parents=True,exist_ok=False)
    history={}
    for name,root in [('r1',ROOT/'results/ua2d_sota_20261018/r1'),('r2',ROOT/'results/ua2d_sota_20261018/r2/fp_fusion')]:
        path=root/'budget.json'
        history[name]=dict(path=str(path),snapshot=read(path),mtime=path.stat().st_mtime) if path.exists() else dict(status='未核实：历史账本缺失，不替代为零')
    write(OUT/'historical_consumption.json',dict(history=history,interpretation='separate historical snapshots; not retest consumption or permissions',time=now()))
    manifest=dict(plan='UA2D-SOTA-20261018-01',revision='r1',campaign=CAMPAIGN,branch='dev',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),status='执行中',start=now(),pid=os.getpid(),
        cwd=str(ROOT),cycle_root=str(OUT),log_root=str(LOG),authorization='user current message full fixed r1_retest1 matrix/formal/one CPU suite',
        resources=dict(gpus=[2,3],pt_ranks=2,ft_slots_per_gpu=2,blas_threads=4,workers=0),limits=LIMITS,
        references=dict(ua=str(UA),t0=str(T0),split=str(SPLIT),downstream=str(DOWN)),outer_test='NOT_RUN',independent_review='NOT_RUN')
    write(OUT/'execution_manifest.json',manifest);refresh_budget(publish=True);capture_sources()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('controller SIGTERM')))
    signal.signal(signal.SIGINT,lambda *_:(_ for _ in ()).throw(InterruptedError('controller SIGINT')))
    complete=False
    try:
        passed={};passed['F0']=kernel_round_job('F0','UA')
        for arm in PARAMS+('A','T0'): passed[arm]=matrix(arm)
        for arm in ('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH'):
            if not passed['T0']:
                passed[arm]=False;write(OUT/'skipped'/f'{arm}.json',dict(reason='T0 matched control incomplete',time=now()));continue
            passed[arm]=pt_job(arm)
            if passed[arm]: passed[arm]=matrix(arm)
        neural_required=(*PARAMS,'A','T0','T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH')
        if all(passed.get(a) for a in neural_required):
            phase('I','按完整 validation 门槛生成至多一项组合，等价配置直接复用。')
            proposal=propose_integration()
            if proposal['run']: passed['I']=matrix('I')
            else: passed['I']=True;write(OUT/'skipped/I.json',dict(reason='no eligible changes or equivalent existing candidate',proposal=proposal,time=now()))
            if passed['I']:
                core=choose_core()['core']
                passed['F1']=kernel_round_job('F1',core) if core!='UA' else passed['F0']
                if core=='UA': write(OUT/'skipped/F1.json',dict(reason='selected core is UA; reuse this campaign F0',time=now()))
                if passed['F0'] and passed['F1']:
                    phase('freeze','全部开发结束，冻结神经核心、核策略、best／alpha 与正式名单。')
                    selection=freeze_selection();manifest['outer_test']='FROZEN_BEFORE_TEST';write(OUT/'execution_manifest.json',manifest)
                    # All encoders run once per task/fold; F1 kernel queries reuse their neural features.
                    formal_ok=True
                    for arm in ['UA',*selection['neural_arms']]: formal_ok=matrix(arm,True) and formal_ok
                    for kernel in selection['kernel_strategies']:
                        phase('kernel_test',f'冻结 kernel TEST：{kernel["round"]}/{kernel["strategy"]}，CPU 顺序查询。')
                        for task in TASKS:
                            for fold in range(5):
                                command=worker_args('kernel_formal',kernel['arm'],task,fold,'formal',kernel['round'],kernel['strategy'])
                                if not wait_exit(launch(f'kernel_test_{kernel["round"]}_{kernel["strategy"]}_{task}_{fold}',command)):
                                    formal_ok=False
                    complete=formal_ok
        write(OUT/'development/stage_outcomes.json',dict(passed=passed,formal_complete=complete,time=now()))
        summarize_results()
        cpu_ok=final_cpu();refresh_budget(publish=True)
        status='待审查' if complete and cpu_ok else ('需返修' if not cpu_ok else '阻断')
        manifest.update(status=status,end=now(),formal_complete=complete,cpu_pass=cpu_ok,outer_test='RUN' if (OUT/'selection_before_test.json').exists() else 'NOT_RUN',controller_exit=0 if complete and cpu_ok else 1)
        write(OUT/'execution_manifest.json',manifest)
        update_execution_plan(status,'本批授权执行已结束；实际范围及失败／跳过记录见 stage_outcomes.json；仅执行自检，等待独立审查。')
        return manifest['controller_exit']
    except BaseException as error:
        stop_own_active();refresh_budget(publish=True)
        write(OUT/'failure.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),retry=False,time=now()))
        manifest.update(status='阻断',end=now(),error=repr(error),controller_exit=1);write(OUT/'execution_manifest.json',manifest)
        update_execution_plan('阻断','控制器异常停止；保留现场，无自动重试。'+repr(error))
        raise
if __name__=='__main__': sys.exit(execute())
