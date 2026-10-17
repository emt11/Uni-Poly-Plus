"""Bounded G05_GLR execution in tmux; no automatic retries or TEST-driven choices."""
import datetime
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fce_tune_g05_glr_unit import OUT,LOG,SPLIT,ARMS,TASKS,read,write,config_for,epochs,pt_config_for,parent,validate_parent
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python'
GPUS=(2,3)
LIMITS=dict(ft_starts=35,ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,
    ft_completed_updates=51450,formal_starts=35,formal_completed=35,cpu_test_starts=1,
    pt_starts=0,pt_attempted_updates=0,pt_validation_starts=0,pt_endpoint_starts=0,new_pt_initial_states=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def check_budget(value):
    if any(value.get(k,0)>v or value.get(k,0)<0 for k,v in LIMITS.items()):
        raise RuntimeError('hard budget exceeded '+str(value))


def launch(name,argv,gpu=None):
    if gpu is not None and gpu not in GPUS:raise ValueError('physical GPU2/3 only')
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    environment=dict(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='16' if gpu is None else '4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    write(folder/'command.json',dict(argv=argv,command=shlex.join(argv),cwd=str(ROOT),environment=environment,gpu=gpu))
    launcher='''import datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
f=Path(sys.argv[1])
if any((f/n).exists() for n in ('execution.json','exit.txt','stdout.log','stderr.log')):raise FileExistsError('launcher already executed; refusing replay')
j=json.loads((f/'command.json').read_text());start=time.monotonic()
j.update(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RUNNING')
def record():
 t=f/'execution.json.tmp';t.write_text(json.dumps(j,indent=2));os.replace(t,f/'execution.json')
record();rc=1
try:
 with (f/'stdout.log').open('xb') as out,(f/'stderr.log').open('xb') as err:
  p=subprocess.Popen(j['argv'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),stdout=out,stderr=err,start_new_session=True)
  (f/'pid.txt').write_text(str(p.pid));rc=p.wait()
except BaseException:
 with (f/'stderr.log').open('a') as err:err.write(traceback.format_exc())
finally:
 j.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,exit_code=rc,status='PASS' if rc==0 else 'FAILED');record();(f/'exit.txt').write_text(str(rc))
if f.name=='controller' and rc==0:
 subprocess.run([j['argv'][0], 'scripts/run_fce_tune_g05_glr.py','--start-finish'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),check=True)
sys.exit(rc)
'''
    script=folder/'command.sh';script.write_text('cd '+shlex.quote(str(ROOT))+'\n'+shlex.join([PYTHON,'-c',launcher,str(folder)])+'\n')
    window='fce_g05_glr_'+name
    # Set remain-on-exit before launching the actual command so even immediate failure is retained.
    wid=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{window_id}','-t','Uni-Poly','-n',window],text=True).strip()
    subprocess.run(['tmux','set-option','-w','-t',wid,'remain-on-exit','on'],check=True)
    subprocess.run(['tmux','send-keys','-t',wid,'bash '+shlex.quote(str(script))+'; exit','Enter'],check=True)
    return folder


def consumption():
    c={k:0 for k in LIMITS}
    c['ft_starts']=len(list(LOG.glob('ft_*/command.json')))
    c['formal_starts']=len(list(LOG.glob('test_*/command.json')))
    c['cpu_test_starts']=len(list(LOG.glob('final_cpu/command.json')))
    for f in (OUT/'finetune').glob('*/*/fold*'):
        if (f/'updates.json').exists():
            u=read(f/'updates.json');c['ft_attempted_updates']+=u['attempted'];c['ft_completed_updates']+=u['completed']
        if (f/'progress.json').exists():c['ft_epochs']+=len(read(f/'progress.json')['history'])
        c['ft_completed']+=int((f/'metrics.json').exists())
    c['formal_completed']=len(list((OUT/'formal').glob('*/*/fold*/metrics.json')))
    c['pt_starts']=len(list(LOG.glob('pt_*/command.json')))
    c['pt_validation_starts']=len(list((OUT/'pt_validation').glob('**/metrics.json')))
    c['pt_endpoint_starts']=len(list(LOG.glob('endpoint*/command.json')))
    c['new_pt_initial_states']=len(list((OUT/'initialization').glob('*step0.pt')))
    check_budget(c);write(OUT/'consumption.json',c);return c


def snapshot(stage,running,queue):
    c=consumption()
    value=dict(time=now(),stage=stage,consumption=c,status='执行中',
        running=[dict(gpu=v['gpu'],unit=v['unit'],log=str(v['folder'])) for v in running.values()],waiting_units=len(queue),independent_review='NOT_PERFORMED')
    write(OUT/'status.json',value)
    with (LOG/'controller/progress.jsonl').open('a') as f:f.write(__import__('json').dumps(value,ensure_ascii=False)+'\n')


def slots(gpu,running):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuid=next(s.split(',')[1].strip() for s in rows.splitlines() if int(s.split(',')[0])==gpu)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    actual=sum(s.split(',')[0].strip()==uuid for s in apps.splitlines() if s.strip())
    own=[v for v in running.values() if v['gpu']==gpu]
    # NVML exposes host PIDs, /proc exposes another PID namespace. Count actual
    # GPU contexts plus own processes not yet known to have reached GPU training.
    # Pending processes may be double counted briefly, which only reduces slots.
    pending=0
    for v in own:
        arm,task,fold=v['unit']
        ready=OUT/'finetune'/arm/task/f'fold{fold}'/'progress.json'
        if v['formal'] or not ready.exists():pending+=1
    return max(0,4-max(actual+pending,len(own)))


def matrix(queue,formal=False):
    requested=list(queue);queue=[];running={};failures=[]
    # Reattach original steps after a controller-only interruption. A unit with
    # uncertain state is blocked, never redispatched or silently retried.
    for arm,task,fold in requested:
        unit=[arm,task,fold];name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
        log=LOG/name;folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
        if not log.exists():
            if folder.exists():raise FileExistsError('artifact without launcher '+str(folder))
            queue.append(unit);continue
        command=read(log/'command.json')
        if command['gpu'] not in GPUS or command['argv']!=[PYTHON,'scripts/run_fce_tune_g05_glr_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]+(['--formal'] if formal else []):
            raise ValueError('existing launcher binding mismatch')
        if (log/'exit.txt').exists():
            rc=int((log/'exit.txt').read_text())
            if rc:failures.append(dict(unit=unit,exit_code=rc,log=str(log)))
            elif not (folder/'metrics.json').exists():raise ValueError('exit0 without metrics')
        else:
            pid=int((log/'pid.txt').read_text());os.kill(pid,0)
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
            if argv!=command['argv']:raise ValueError('live unit PID/argv mismatch')
            running[name]=dict(gpu=command['gpu'],folder=log,unit=unit,formal=formal)
    while queue or running:
        for name,v in list(running.items()):
            exit_path=v['folder']/'exit.txt'
            if exit_path.exists():
                rc=int(exit_path.read_text())
                if rc:failures.append(dict(unit=v['unit'],exit_code=rc,log=str(v['folder'])))
                del running[name]
        if failures and queue:
            write(OUT/('formal_stopped_queue.json' if formal else 'ft_stopped_queue.json'),dict(unstarted=queue,failures=failures,reason='stop dispatch after failure; retain active units and do not retry'))
            queue=[]
        # Retain failures and active units; never retry or enter formal with incomplete FT.
        for gpu in GPUS:
            for _ in range(slots(gpu,running)):
                if not queue:break
                arm,task,fold=queue.pop(0)
                name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
                folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
                if folder.exists() or (LOG/name).exists():raise FileExistsError('writer/overwrite risk '+str(folder))
                c=consumption()
                if c['formal_starts' if formal else 'ft_starts']>=LIMITS['formal_starts' if formal else 'ft_starts']:raise RuntimeError('start budget exhausted')
                argv=[PYTHON,'scripts/run_fce_tune_g05_glr_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
                if formal:argv.append('--formal')
                log=launch(name,argv,gpu);running[name]=dict(gpu=gpu,folder=log,unit=[arm,task,fold],formal=formal)
        snapshot('正式预测' if formal else 'FT',running,queue)
        if queue or running:time.sleep(15)
    return failures


def save_diff():
    names=['CURRENT_PLAN.md','scripts/run_fce_tune_g05_glr.py','scripts/run_fce_tune_g05_glr_unit.py','scripts/select_fce_tune_g05_glr.py','tests/test_fce_tune_g05_glr.py']
    chunks=[]
    for name in names:
        tracked=subprocess.run(['git','ls-files','--error-unmatch',name],capture_output=True).returncode==0
        args=['git','diff','--',name] if tracked else ['git','diff','--no-index','--','/dev/null',name]
        r=subprocess.run(args,capture_output=True,text=True)
        if r.returncode not in (0,1):raise RuntimeError(r.stderr)
        chunks.append(r.stdout)
    (OUT/'final_review/actual_changes.diff').write_text(''.join(chunks))
    import difflib
    incremental=[]
    for name in names:
        before=OUT/'source_before'/name
        old=before.read_text().splitlines(keepends=True) if before.exists() else []
        incremental.extend(difflib.unified_diff(old,(ROOT/name).read_text().splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
    (OUT/'final_review/g05_glr_only_changes.diff').write_text(''.join(incremental))


def wait_original(folder,stage,allow_failure=False):
    # Wait on completion writes, without periodic PT/CPU status monitoring.
    import ctypes
    path=folder/'exit.txt'
    if not path.exists():
        libc=ctypes.CDLL(None,use_errno=True)
        fd=libc.inotify_init1(os.O_CLOEXEC)
        if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        try:
            if libc.inotify_add_watch(fd,os.fsencode(folder),0x00000008|0x00000080)<0:
                raise OSError(ctypes.get_errno(),'inotify_add_watch')
            snapshot(stage,{},[])
            while not path.exists():os.read(fd,4096)
        finally:os.close(fd)
    rc=int(path.read_text())
    if rc and not allow_failure:raise RuntimeError(stage+' original execution failed exit '+str(rc))
    return rc


def require_stage(stage,budget):
    check_budget(budget)
    if stage in ('freeze','formal','CPU') and any(budget.get(k)!=v for k,v in dict(ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,ft_completed_updates=51450).items()):
        raise ValueError('complete candidate FT required')
    if stage=='CPU' and budget.get('formal_completed')!=35:raise ValueError('complete candidate formal matrix required')


def campaign(resume_controller=False):
    require_tmux();started=now()
    try:
        names=['scripts/run_fce_tune_g05_glr.py','scripts/run_fce_tune_g05_glr_unit.py','scripts/select_fce_tune_g05_glr.py',
            'tests/test_fce_tune_g05_glr.py','src/training/fce_param.py','scripts/finetune_mcl_ph.py',
            'scripts/run_fce_tune_t3b_unit.py','scripts/finalize_fce_tune_t3.py',
            'src/modules/glt_dual_pretrain.py','src/modules/glt_dual.py','src/modules/glt_dual_opt.py',
            'src/modules/glt_adaptation.py','src/dataset/glt_dual.py','src/training/glt_dual_runtime.py']
        for name in names:
            dest=OUT/'source_used'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if not resume_controller:raise FileExistsError('runtime snapshot exists')
            else:shutil.copyfile(ROOT/name,dest)
        if not (OUT/'setup_complete.json').exists():raise ValueError('missing original-source snapshot')
        planned_epochs=sum(epochs(t)*5 for t in TASKS)
        planned_updates=sum(epochs(t)*math.ceil(len(read(SPLIT/(t+'.json'))['folds'][f]['train_indices'])/24) for t in TASKS for f in range(5))
        if not resume_controller:write(OUT/'planned_budget.json',dict(pt=0,pt_updates=0,pt_validation_calls=0,pt_endpoint=0,ft=35,epochs=planned_epochs,updates=planned_updates,formal_max=35,reused_reference_units=35))
        if (planned_epochs,planned_updates)!=(2400,51450):raise ValueError('actual split budget differs from authorized plan')
        initialize=LOG/'initialize'
        if not initialize.exists():initialize=launch('initialize',[PYTHON,'scripts/run_fce_tune_g05_glr_unit.py','--initialize'])
        wait_original(initialize,'初始化导出')
        require_stage('FT',consumption())
        failures=matrix([(a,t,f) for a in ARMS for t in TASKS for f in range(5)])
        if failures:
            write(OUT/'blocked.json',dict(failed_units=failures,reason='incomplete FT matrix; no retry, selection and TEST prohibited'))
            raise RuntimeError('incomplete FT matrix')
        from scripts.select_fce_tune_g05_glr import freeze,summarize
        require_stage('freeze',consumption())
        unique=freeze();require_stage('formal',consumption());failures=matrix(unique,True)
        if failures:
            write(OUT/'blocked.json',dict(failed_units=failures,reason='incomplete formal predictions; no retry'))
            raise RuntimeError('incomplete formal predictions')
        require_stage('CPU',consumption())
        delivery=summarize()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_tune_g05_glr.py'])
        rc=wait_original(test,'最终CPU检查',allow_failure=True);c=consumption()
        state='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=state,start=started,end=now(),test_exit=rc,formal_unique=len(unique),consumption=c,independent_review='NOT_PERFORMED',reused_reference='T3B_G05 + R0; existing 35 FT and 35 formal predictions'))
        plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text().replace('状态：**执行中（r6）**。授权来源：','状态：**'+state+'**。授权来源：',1)
        plan.write_text(text+f'\n\n## 30. G05_GLR最终执行交接\n\n状态：**{state}**；真实CPU exit {rc}；独立审查未进行。预算：'+__import__('json').dumps(c,ensure_ascii=False)+'。\n\n'+delivery)
        snapshot('执行完成',{},[])
        final=read(OUT/'status.json');final['status']=state;write(OUT/'status.json',final)
        save_diff()
    except BaseException as error:
        c=consumption()
        write(OUT/'campaign_failure.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),time=now(),consumption=c))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r6）**。授权来源：','状态：**阻断**。授权来源：',1)
        p.write_text(text+'\n\nG05_GLR执行阻断：'+repr(error)+'。现场 results/fce_tune_20261016/g05_glr/campaign_failure.json。已用预算保留，不静默重跑；不完整矩阵不选参／读TEST。最终CPU检查未执行（除非已有 final_cpu 实际记录）。\n')
        raise


def finish_delivery():
    from scripts.finalize_fce_tune_t3 import exit_index
    review=OUT/'final_review';review.mkdir(exist_ok=True)
    index=exit_index(LOG,exclude=('finish_delivery',))
    index.update(scope='r6 new execution only; G05 R0 reference reused',
        finishing_process='own return code recorded separately in logs/finish_delivery/{execution.json,exit.txt} after this function exits')
    index['limitations']='记录来自原始执行JSON和exit.txt，不插补未知时间；本轮没有PT。区间和调度快照分别报告观测峰值。'
    write(review/'execution_exits.json',index)
    if (OUT/'completion.json').exists():
        completion=read(OUT/'completion.json');completion['controller_exit']=read(LOG/'controller/execution.json')['exit_code'];write(OUT/'completion.json',completion)
        summary=read(review/'seven_task_results.json')
        labels=dict(eea='Eea',egb='Egb',egc='Egc',ei='Ei',eps='EPS',nc='Nc',xc='Xc')
        plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text()
        anchor=text.index('### 27.6');start=text.index('|Eea|',anchor);end=text.index('\n\n协议限制',start)
        rows=[]
        for x in summary['rows']:
            r=x['reference'];c=x['candidate'];p=x['paper']
            rows.append(f"|{labels[x['task']]}|{r['r2_mean']:.3f}±{r['r2_std']:.3f}|{c['r2_mean']:.3f}±{c['r2_std']:.3f}|{x['delta_r2']:+.3f}|{p['model']}|{p['r2']:.3f}|{x['reference_minus_paper']:+.3f}|{x['candidate_minus_paper']:+.3f}|")
        text=text[:start]+'\n'.join(rows)+text[end:]
        text=text.replace('当前七任务候选均为**未完成／不可比较：r6尚未启动**。','本轮七任务完整正式结果已交付，状态待审查（若最终CPU失败则以具体需返修状态为准）。')
        text+='\n最终退出索引已生成；收尾自身的真实exit单独保存在logs/fce_tune_20261016/g05_glr/finish_delivery/。同一执行主体的自检不称独立审查，不重复运行实验或测试。\n';plan.write_text(text)
        extra='\n实际执行／预算见completion.json、consumption.json；controller、初始化、FT、正式预测与唯一CPU检查的真实exit见execution_exits.json。收尾自身真实exit单独保存于logs/fce_tune_20261016/g05_glr/finish_delivery/，未完成时不记成功。\n'
        extra+='只新增35FT／2400epochs／51450 attempted updates、35次候选预测；旧G05参考35FT及35预测只读复用。PT／PT validation／endpoint／新PT初始化包均为0；没有额外seed、smoke、benchmark或ensemble，未重跑r5或修改共享科学实现。\n'
        extra+='tmux Uni-Poly:fce_g05_glr_*；结果results/fce_tune_20261016/g05_glr/，日志logs/fce_tune_20261016/g05_glr/；增量diff见g05_glr_only_changes.diff。未修改模型历史／SOTA_RESULT.md，未部署、删除、pull、commit或push；独立审查未进行。\n'
        with (review/'delivery.md').open('a') as f:f.write(extra)
    save_diff()
    result=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md'],cwd=ROOT,capture_output=True,text=True)
    write(LOG/'finish_delivery/document_format.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume-controller',action='store_true')
    parser.add_argument('--controller',action='store_true');parser.add_argument('--finish-delivery',action='store_true');parser.add_argument('--start-finish',action='store_true')
    args=parser.parse_args();require_tmux()
    if args.start_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_tune_g05_glr.py','--finish-delivery'])
    elif args.finish_delivery:finish_delivery()
    elif args.controller:campaign(args.resume_controller)
    else:raise ValueError('explicit controller/finish action required')
