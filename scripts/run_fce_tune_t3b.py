"""Bounded T3b execution in tmux; no automatic retries or TEST-driven choices."""
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
from scripts.run_fce_tune_t3b_unit import OUT,LOG,SPLIT,ARMS,TASKS,read,write,config_for,epochs,pt_config_for,parent,PI,validate_parent
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python'
GPUS=(2,3)
LIMITS=dict(ft_starts=35,ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,
    ft_completed_updates=51450,formal_starts=35,formal_completed=35,cpu_test_starts=1,pt_starts=1,pt_completed=1,pt_attempted_updates=5000,pt_completed_updates=5000,
    pt_train_attempted_presentations=5040000,pt_validation_starts=3,pt_validation_presentations=3072)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def check_budget(value):
    if any(value.get(k,0)>v or value.get(k,0)<0 for k,v in LIMITS.items()):
        raise RuntimeError('hard budget exceeded '+str(value))


def launch(name,argv,gpu=None):
    if gpu is not None and gpu not in (*GPUS,'2,3'):raise ValueError('physical GPU2/3 only')
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
 subprocess.run([j['argv'][0], 'scripts/run_fce_tune_t3b.py','--start-finish'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),check=True)
sys.exit(rc)
'''
    script=folder/'command.sh';script.write_text('cd '+shlex.quote(str(ROOT))+'\n'+shlex.join([PYTHON,'-c',launcher,str(folder)])+'\n')
    window='fce_t3b_'+name
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
    for arm in ARMS:
        path=OUT/'pretrain'/arm
        if (path/'consumption.json').exists():
            ledger=read(path/'consumption.json')
            c['pt_attempted_updates']+=ledger['attempted_updates'];c['pt_completed_updates']+=ledger['completed_updates']
            c['pt_train_attempted_presentations']+=ledger['attempted_updates']*1008
            if ledger['status']=='COMPLETED':c['pt_completed']+=1
        for folder in (OUT/'pt_validation'/arm).glob('step*'):
            c['pt_validation_starts']+=1
            for progress in folder.glob('progress_rank*.json'):c['pt_validation_presentations']+=read(progress)['presented']
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
        if command['gpu'] not in GPUS or command['argv']!=[PYTHON,'scripts/run_fce_tune_t3b_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]+(['--formal'] if formal else []):
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
                argv=[PYTHON,'scripts/run_fce_tune_t3b_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
                if formal:argv.append('--formal')
                log=launch(name,argv,gpu);running[name]=dict(gpu=gpu,folder=log,unit=[arm,task,fold],formal=formal)
        snapshot('正式预测' if formal else 'FT',running,queue)
        if queue or running:time.sleep(15)
    return failures


def save_diff():
    names=['CURRENT_PLAN.md','scripts/finalize_fce_tune_t3.py','scripts/pretrain_glt_dual.py','scripts/evaluate_glt_pt_endpoint.py','scripts/run_fce_tune_t3b.py','scripts/run_fce_tune_t3b_unit.py','scripts/select_fce_tune_t3b.py','tests/test_fce_tune_t3b.py']
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
    (OUT/'final_review/t3_only_changes.diff').write_text(''.join(incremental))


def pt_gpus_empty():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids={s.split(',')[1].strip() for s in rows.splitlines() if int(s.split(',')[0]) in GPUS}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return not any(s.split(',')[0].strip() in uuids for s in apps.splitlines() if s.strip())


def pt_command(arm):
    return [PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
        '--log_dir',str(LOG/('pt_'+arm)/'ranks'),'--redirects=3','--tee=3',
        'scripts/pretrain_glt_dual.py','--config',str(OUT/'initialization'/f'{arm}.json'),
        '--cohort-root',str(PI),'--cache-root',str(PI),'--dual-static-root',str(PI),
        '--output',str(OUT/'pretrain'/arm),'--prep-workers','8','--lmdb-readahead']


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


def complete_pt():
    import torch
    metadata={}
    for arm in ARMS:
        cfg=pt_config_for(arm);folder=OUT/'pretrain'/arm
        ledger=read(folder/'consumption.json');run=read(folder/'run.json');runtime=read(folder/'runtime.json')
        if ledger!=dict(starts=1,attempted_updates=5000,completed_updates=5000,status='COMPLETED'):
            raise ValueError('incomplete PT ledger '+arm)
        if run['identity']['config']!=cfg or run['identity']['world_size']!=2 or run['identity']['sample_count']!=911391 or run['accumulation']!=6 or runtime['status']!='PASS' or runtime['prep_workers']!=8:
            raise ValueError('PT run metadata mismatch '+arm)
        if not (folder/'resume_05000.pt').is_file():raise FileNotFoundError('missing fixed final resume package')
        package=torch.load(parent(arm),map_location='cpu',weights_only=False);validate_parent(package,arm)
        metadata[arm]=dict(parent=str(parent(arm)),pt_config=cfg,run=run,consumption=ledger)
        del package
        for step in (1000,2500,5000):
            v=OUT/'pt_validation'/arm/f'step{step:05d}';m=read(v/'metrics.json')
            if m['status']!='PASS' or m['presented']!=1024 or m['step']!=step or m['config']!=cfg or m['checkpoint_selection'] is not False:
                raise ValueError('PT validation mismatch '+arm)
            if any(read(v/f'progress_rank{rank}.json')['presented']!=512 for rank in (0,1)):
                raise ValueError('PT validation per-rank presentation mismatch')
    require_stage('FT',consumption())
    write(OUT/'pt_completion.json',dict(status='PASS',parents=metadata,validation_calls=3,validation_presentations=3072,checkpoint_selection=False))


def require_stage(stage,budget):
    check_budget(budget)
    if stage in ('FT','freeze','formal','CPU') and any(budget.get(k)!=v for k,v in dict(pt_completed=1,pt_completed_updates=5000,pt_attempted_updates=5000,pt_validation_starts=3,pt_validation_presentations=3072).items()):
        raise ValueError('complete candidate PT and process validation required')
    if stage in ('freeze','formal','CPU') and any(budget.get(k)!=v for k,v in dict(ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,ft_completed_updates=51450).items()):
        raise ValueError('complete FT matrix required')
    if stage=='CPU' and budget.get('formal_completed')!=35:raise ValueError('complete fixed formal matrix required')


def campaign(resume_controller=False):
    require_tmux();started=now()
    try:
        names=['scripts/finalize_fce_tune_t3.py','scripts/run_fce_tune_t3b.py','scripts/run_fce_tune_t3b_unit.py','scripts/select_fce_tune_t3b.py',
            'scripts/pretrain_glt_dual.py','scripts/evaluate_glt_pt_endpoint.py','tests/test_fce_tune_t3b.py',
            'src/training/fce_param.py','scripts/finetune_mcl_ph.py','src/modules/glt_dual_pretrain.py',
            'src/modules/glt_dual.py','src/modules/glt_dual_opt.py','src/modules/glt_adaptation.py',
            'src/dataset/glt_dual.py','src/dataset/glt_dual_pretrain.py','src/training/glt_dual_runtime.py']
        for name in names:
            dest=OUT/'source_used'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if not resume_controller:raise FileExistsError('runtime snapshot exists')
            else:shutil.copyfile(ROOT/name,dest)
        if not (OUT/'setup_complete.json').exists():raise ValueError('missing original-source snapshot')
        planned_epochs=sum(epochs(t)*5 for t in TASKS)
        planned_updates=sum(epochs(t)*math.ceil(len(read(SPLIT/(t+'.json'))['folds'][f]['train_indices'])/24) for t in TASKS for f in range(5))
        if not resume_controller:write(OUT/'planned_budget.json',dict(pt=1,pt_updates=5000,pt_train_presentations=5040000,pt_validation_calls=3,pt_validation_presentations=3072,ft=35,epochs=planned_epochs,updates=planned_updates,formal_max=35,reused_reference_units=35))
        if (planned_epochs,planned_updates)!=(2400,51450):raise ValueError('actual split budget differs from authorized plan')
        repair=LOG/'repair_T3a_delivery_r5'
        if not repair.exists():repair=launch('repair_T3a_delivery_r5',[PYTHON,'scripts/finalize_fce_tune_t3.py'])
        repair_rc=wait_original(repair,'T3a交付补充',allow_failure=True)
        if repair_rc:write(OUT/'delivery_repair_issue.json',dict(exit_code=repair_rc,log=str(repair),science_not_affected=True,reason='preserve failed delivery and continue independent scientific candidate'))
        initialize=LOG/'initialize'
        if not initialize.exists():initialize=launch('initialize',[PYTHON,'scripts/run_fce_tune_t3b_unit.py','--initialize'])
        wait_original(initialize,'初始化导出')
        for arm in ARMS:
            if read(OUT/'initialization'/f'{arm}.json')!=pt_config_for(arm):raise ValueError('PT config mismatch')
            folder=LOG/('pt_'+arm)
            if not folder.exists():
                while not pt_gpus_empty():snapshot('PT等待GPU2/3空闲',{},[]);time.sleep(15)
                if (OUT/'pretrain'/arm).exists():raise FileExistsError('PT output without original launcher')
                folder=launch('pt_'+arm,pt_command(arm),'2,3')
            wait_original(folder,'PT:'+arm)
        complete_pt()
        failures=matrix([(a,t,f) for a in ARMS for t in TASKS for f in range(5)])
        if failures:
            write(OUT/'blocked.json',dict(failed_units=failures,reason='incomplete FT matrix; no retry, selection and TEST prohibited'))
            raise RuntimeError('incomplete FT matrix')
        from scripts.select_fce_tune_t3b import freeze,summarize
        require_stage('freeze',consumption())
        unique=freeze();require_stage('formal',consumption());failures=matrix(unique,True)
        if failures:
            write(OUT/'blocked.json',dict(failed_units=failures,reason='incomplete formal predictions; no retry'))
            raise RuntimeError('incomplete formal predictions')
        require_stage('CPU',consumption())
        delivery=summarize()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_tune_t3b.py'])
        rc=wait_original(test,'最终CPU检查',allow_failure=True);c=consumption()
        state='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=state,start=started,end=now(),test_exit=rc,formal_unique=len(unique),consumption=c,independent_review='NOT_PERFORMED',reused_reference='T3_REF; existing 35 FT and 35 formal predictions'))
        plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text().replace('状态：**执行中（r5）**。授权来源：','状态：**'+state+'**。授权来源：',1)
        plan.write_text(text+f'\n\n## 25. T3b最终执行交接\n\n状态：**{state}**；真实CPU exit {rc}；独立审查未进行。预算：'+__import__('json').dumps(c,ensure_ascii=False)+'。\n\n'+delivery)
        snapshot('执行完成',{},[])
        final=read(OUT/'status.json');final['status']=state;write(OUT/'status.json',final)
        save_diff()
    except BaseException as error:
        c=consumption()
        write(OUT/'campaign_failure.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),time=now(),consumption=c))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r5）**。授权来源：','状态：**阻断**。授权来源：',1)
        p.write_text(text+'\n\nT3b执行阻断：'+repr(error)+'。现场 results/fce_tune_20261015/t3b/campaign_failure.json。已用预算保留，不静默重跑；不完整矩阵不选参／读TEST。最终CPU检查未执行（除非已有 final_cpu 实际记录）。\n')
        raise


def setup():
    require_tmux()
    if OUT.exists() or LOG.exists():raise FileExistsError('new campaign root already exists')
    OUT.mkdir(parents=True);LOG.mkdir(parents=True)
    for name in ('CURRENT_PLAN.md','scripts/pretrain_glt_dual.py','scripts/evaluate_glt_pt_endpoint.py','scripts/finalize_fce_tune_t3.py'):
        dest=OUT/'source_before'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(OUT/'setup_complete.json',dict(time=now(),scientific_updates=0,authorization='user explicitly authorized autonomous optimization and execution'))
    print('Original source snapshots saved; no scientific execution.')


def finish_delivery():
    from scripts.finalize_fce_tune_t3 import exit_index
    review=OUT/'final_review'
    if not review.exists():return
    index=exit_index(LOG,exclude=('finish_delivery',))
    index['scope']='T3b new execution only; T3_REF existing evidence reused'
    write(review/'execution_exits.json',index)
    save_diff()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume-controller',action='store_true')
    parser.add_argument('--setup',action='store_true');parser.add_argument('--start',action='store_true')
    parser.add_argument('--controller',action='store_true');parser.add_argument('--finish-delivery',action='store_true');parser.add_argument('--start-finish',action='store_true')
    args=parser.parse_args();require_tmux()
    if args.setup:setup()
    elif args.start:
        launch('controller',[PYTHON,'scripts/run_fce_tune_t3b.py','--controller'])
    elif args.start_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_tune_t3b.py','--finish-delivery'])
    elif args.finish_delivery:finish_delivery()
    elif args.controller:
        campaign(args.resume_controller)
    else:raise ValueError('explicit setup/start/controller action required')
