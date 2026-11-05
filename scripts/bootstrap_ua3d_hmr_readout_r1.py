#!/usr/bin/env python3
"""Readout r1 isolated bootstrap and proven r3 lifecycle, no scientific imports."""
import argparse,datetime,errno,fcntl,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
P1=ROOT/'results/ua3d_hmr_20261103/r1'
P2=P1/'repairs/schema_surface_r2'
P3=P1/'repairs/scheduler_r3'
OUT=ROOT/'results/ua3d_hmr_readout_20261103/r1'
LOG=ROOT/'logs/ua3d_hmr_readout_20261103/r1'
PYTHON=P1/'env/bin/python'
SOURCES=('src/dataset/ua3d_hmr_readout_r1.py','src/modules/ua3d_hmr_readout_r1.py','src/training/ua3d_hmr_readout_r1.py','scripts/prepare_ua3d_hmr_readout_r1.py','scripts/run_ua3d_hmr_readout_r1.py','scripts/bootstrap_ua3d_hmr_readout_r1.py','configs/ua3d_hmr_readout_r1_gpu2.json','tests/test_ua3d_hmr_readout_r1.py')
THREADS={'CUDA_VISIBLE_DEVICES':'','WORLD_SIZE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT)}
TASKS=('eea','egb','egc','ei','eps','nc','xc')
ARMS=('C_ANCHOR','C_FIXED','N_BASE','N_ADD','N_BILIN')
TRAIN_ARMS=ARMS[2:]
def read(p):return json.loads(Path(p).read_text())

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)

def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def identity(pid=None):
    pid=pid or os.getpid();s=Path(f'/proc/{pid}/stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))

def size(root):
    total=0
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[x for x in dirs if not (Path(folder)/x).is_symlink()]
        for name in files:
            p=Path(folder)/name
            if p.is_symlink():continue
            try:total+=p.stat().st_size
            except FileNotFoundError:pass # Only atomic writer temporary files may disappear.
    return total

def parse_jobs(jobs):
    seen=set();counts={};intervals=[];wall=0
    for j in jobs:
        if j['id'] in seen:raise ValueError('duplicate identity')
        seen.add(j['id'])
        for name in ('reserved_unix','started_unix','ended_unix'):
            if name not in j:raise ValueError('missing '+name)
            if j[name] is not None and (not isinstance(j[name],(float,int)) or not math.isfinite(j[name])):raise ValueError('time type')
        if j['started_unix'] is not None and j['started_unix']<j['reserved_unix']:raise ValueError('start before reservation')
        if j['ended_unix'] is not None and (j['started_unix'] is None or j['ended_unix']<j['started_unix']):raise ValueError('end before start')
        c=counts.setdefault(j['kind'],dict(reserved=0,started=0,attempted=0,completed=0));c['reserved']+=1;c['started']+=int(j['started_unix'] is not None);c['attempted']+=int(j['started_unix'] is not None);c['completed']+=int(j['exit']==0 and j.get('qualified',False))
        if j['started_unix'] is not None:
            end=j['ended_unix'] if j['ended_unix'] is not None else time.time();wall+=end-j['started_unix']
            if j['gpu']:intervals.append((j['started_unix'],end))
    last=0;busy=0
    for a,b in sorted(intervals):busy+=max(0,b-max(a,last));last=max(last,b)
    return dict(counts=counts,process_wall_seconds=wall,physical_gpu_union_seconds=busy)

def state_for(jobs,expected):
    parse_jobs(jobs);active=any(j['exit'] is None for j in jobs);bad=any(j['exit'] is not None and (j['exit']!=0 or not j.get('qualified',False)) for j in jobs)
    return dict(status='NOT_RUN' if not jobs else 'FAILED' if bad else 'RUNNING' if active else 'PASSED' if len(jobs)==expected else 'PARTIAL',attempts=len(jobs),exit=None if not jobs or active else next((j['exit'] or 1 for j in jobs if j['exit']!=0 or not j.get('qualified',False)),0) if bad else 0 if len(jobs)==expected else None)

def missing_proc(error):return isinstance(error,OSError) and error.errno in (errno.ENOENT,errno.ESRCH)

class ProcReader:
    def stat(self,pid):return identity(pid)
    def cmdline(self,pid):return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    def environ(self,pid):return dict(x.split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
    def cwd(self,pid):return os.readlink(f'/proc/{pid}/cwd')
    def stdout(self,pid):return os.readlink(f'/proc/{pid}/fd/1')
    def namespace(self,pid):
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('NSpid:'):return [int(x) for x in line.split()[1:]]
        return [pid]
    def send(self,p,j,sig):
        # A pinned pidfd cannot signal an unrelated process after PID reuse.
        fd=os.pidfd_open(p.pid)
        try:
            if p.poll() is not None:return False
            if self.stat(p.pid)!=j['process']:raise ValueError('live identity changed before signal')
            signal.pidfd_send_signal(fd,sig);return True
        finally:os.close(fd)

class Lifecycle:
    """Injectable proc/clock/runner; all terminal paths call the same idempotent reap."""
    def __init__(self,validate,changed=lambda:None,proc=None,clock=time.time,runner=subprocess.Popen,pause=time.sleep):
        self.validate=validate;self.changed=changed;self.proc=proc or ProcReader();self.clock=clock;self.runner=runner;self.pause=pause;self.active={};self.events=[]
    def spawn(self,j,env):
        if self.pending():raise RuntimeError('cannot dispatch with EXIT_PENDING')
        stream=(Path(j['log'])/'stdout.log').open('x');p=self.runner(j['argv'],cwd=j['cwd'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        j.update(started_unix=self.clock(),local_pid=p.pid,process=None,identity_source=None,lifecycle='RUNNING',exit_pending_since=None)
        self.active[j['id']]=(p,j,stream);self.changed();self.observe(j['id']);return j
    def pending(self):return any(j.get('lifecycle')=='EXIT_PENDING' for p,j,s in self.active.values())
    def reap(self,key):
        if key not in self.active:return True
        p,j,stream=self.active[key];rc=p.poll()
        if rc is None:return False
        rc=p.wait();end=self.clock();problem=None
        try:
            process=read(Path(j['log'])/'process.json')
            if process['pid']!=p.pid or not isinstance(process['start_ticks'],int):raise ValueError('exited worker process identity')
            if j['process'] is not None and j['process']!={k:process[k] for k in ('pid','start_ticks')}:raise ValueError('worker live/saved identity differs')
            if j['process'] is None:j['process']={k:process[k] for k in ('pid','start_ticks')};j['identity_source']='exited_worker_record'
            if rc==0:self.validate(j)
        except BaseException as e:problem=e
        j.update(exit=rc,ended_unix=end,qualified=rc==0 and problem is None,lifecycle='REAPED',exit_pending_since=None);stream.close();del self.active[key]
        exclusive(Path(j['log'])/'wait_exit.json',dict(exit=rc,started_unix=j['started_unix'],ended_unix=end,qualified=j['qualified'],identity=j['process'],identity_source=j['identity_source'],validation_error=repr(problem) if problem else None));self.events.append(dict(event='REAP',id=key,exit=rc,time=end));self.changed()
        if problem:raise problem
        if rc:raise RuntimeError('worker nonzero exit '+key+'='+str(rc))
        return True
    def pending_exit(self,key):
        if key not in self.active:return None
        p,j,s=self.active[key]
        if p.poll() is not None:self.reap(key);return None
        now=self.clock()
        if j.get('exit_pending_since') is None:j['exit_pending_since']=now
        j['lifecycle']='EXIT_PENDING';self.events.append(dict(event='EXIT_PENDING',id=key,time=now));self.changed()
        if now-j['exit_pending_since']>=5:raise TimeoutError('EXIT_PENDING did not close within 5 seconds: '+key)
        return None
    def observe(self,key):
        if key not in self.active:return None
        p,j,stream=self.active[key]
        if p.poll() is not None:self.reap(key);return None
        try:
            snap=dict(identity=self.proc.stat(p.pid),argv=self.proc.cmdline(p.pid),env=self.proc.environ(p.pid),cwd=self.proc.cwd(p.pid),stdout=self.proc.stdout(p.pid),nspid=self.proc.namespace(p.pid));after=self.proc.stat(p.pid)
        except OSError as e:
            if not missing_proc(e):raise
            return self.pending_exit(key)
        if p.poll() is not None:self.reap(key);return None
        expected=j['process']
        good=(snap['identity']==after and (expected is None or after==expected) and after['pid']==p.pid and snap['cwd']==j['cwd'] and snap['stdout']==str(Path(j['log'])/'stdout.log') and snap['env'].get('CUDA_VISIBLE_DEVICES')==('2' if j['gpu'] else '') and all(part in snap['argv'] for part in j['ownership_tokens']))
        if not good:
            if p.poll() is not None:self.reap(key);return None
            raise ValueError('live worker identity/argv/cwd/CVD/stdout FD mismatch '+key)
        if j['process'] is None:j['process']=after;j['identity_source']='live_proc_snapshot'
        j['lifecycle']='RUNNING';j['exit_pending_since']=None;self.changed();return snap
    def poll_all(self):
        for key in list(self.active):
            if key in self.active:self.observe(key)
    def snapshots(self):
        self.poll_all();out=[]
        for key in list(self.active):
            snap=self.observe(key)
            if snap is not None:out.append(snap)
        return out
    def stop(self):
        errors=[]
        for key in list(self.active):
            try:
                snap=self.observe(key)
                if key not in self.active:continue
                if snap is None:continue
                p,j,s=self.active[key]
                try:self.proc.send(p,j,signal.SIGTERM)
                except OSError as e:
                    if not missing_proc(e):raise
                    self.pending_exit(key)
            except BaseException as e:errors.append(repr(e))
        deadline=self.clock()+5
        while self.active and self.clock()<deadline:
            for key in list(self.active):
                try:self.observe(key)
                except BaseException as e:errors.append(repr(e))
            if self.active:self.pause(.02)
        for key in list(self.active):
            try:
                snap=self.observe(key)
                if key not in self.active:continue
                if snap is None:raise TimeoutError('unresolved process during stop')
                p,j,s=self.active[key];self.proc.send(p,j,signal.SIGKILL)
                p.wait(timeout=5);self.reap(key)
            except BaseException as e:errors.append(repr(e))
        if errors:self.events.append(dict(event='STOP_ERRORS',errors=errors,time=self.clock()))
        return errors

def wait_wrapper(cmd,log,root,name,env):
    """Outer/controller waits also tolerate exit before initial /proc identity."""
    with Path(log).open('x') as stream:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);started=time.time();observed=None;pending_since=None
        while observed is None:
            if p.poll() is not None:
                record=read(Path(root)/f'{name}_self_process.json');observed={k:record[k] for k in ('pid','start_ticks')}
                if observed['pid']!=p.pid:raise ValueError('wrapper exited identity mismatch')
                break
            try:observed=identity(p.pid)
            except OSError as e:
                if not missing_proc(e):raise
                pending_since=pending_since or time.time()
                if time.time()-pending_since>5:raise TimeoutError('wrapper initial identity pending')
                time.sleep(.01)
        exclusive(Path(root)/f'{name}_process.json',dict(**observed,argv=cmd,cwd=str(ROOT),env=THREADS,started_unix=started));rc=p.wait();exclusive(Path(root)/f'{name}_exit.json',dict(exit=rc,started_unix=started,ended_unix=time.time(),wait_source='subprocess.Popen.wait'));return rc

def envelope(root,log,action):
    root=Path(root);log=Path(log);ctx={'phase':'initialize'};rc=1
    try:
        root.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True);rc=action(ctx)
        if rc:raise RuntimeError('outer nonzero exit '+str(rc))
        ctx['phase']='report';report(root,log,None,ctx['phase'],0);rc=0
    except BaseException:
        error=traceback.format_exc();rc=rc or 1
        if not (root/'first_failure.json').exists():exclusive(root/'first_failure.json',dict(phase=ctx['phase'],error=error,exit=rc,time=time.time()))
        try:report(root,log,error,ctx['phase'],rc)
        except BaseException:write(root/'report_failure.json',dict(error=traceback.format_exc(),primary_error=error))
    finally:write(root/'bootstrap_exit.json',dict(exit=rc,ended_unix=time.time(),phase=ctx['phase']))
    return rc

def check_clock(preparation=False,reserve=2400):
    c=read(OUT/'authorization/clock.json');now=time.time()
    if now>=c['deadline_unix']-reserve or preparation and now>=c['prepare_deadline_unix']:raise TimeoutError('READOUT deadline/reserve')
def fail(phase,error,code=1):
    if not (OUT/'first_failure.json').exists():exclusive(OUT/'first_failure.json',dict(phase=phase,error=error,exit=code,time=time.time()))
    write(OUT/'execution_manifest.json',dict(status='STOPPED',phase=phase,execution_complete=False,independent_review='PENDING',exit=code))
def cumulative(jobs,usage):
    hist=read(OUT/'authorization/historical_cumulative.json') if (OUT/'authorization/historical_cumulative.json').exists() else dict(cumulative_update_interval=[18090,18230],FT_attempts=107)
    count=sum(j['kind']=='ft' for j in jobs);updates=usage.get('ft',{}).get('attempted',{}).get('updates',0)+usage.get('diagnostic',{}).get('attempted',{}).get('updates',0)
    return dict(historical_hmr=dict(source=str(P3/'cumulative_budget.json'),geometry_reserved=7300,geometry_attempted=4231,FT_attempts=107,update_interval=hist['cumulative_update_interval'],preflight=3,final_cpu=1,formal_units=140,formal_rows=23500,unknown_old_updates=[0,140]),current_readout=dict(FT_attempts=count,recorded_updates=updates,preflight=sum(j['kind']=='preflight' for j in jobs),final_cpu=sum(j['kind']=='cpu' for j in jobs),formal_completed=sum(j['kind']=='test' and j['qualified'] for j in jobs),new_model_outputs=usage.get('test',{}).get('completed',{}).get('new_outputs',0),reused_outputs=usage.get('test',{}).get('completed',{}).get('alias_outputs',0),new_geometry=0,new_BASE_encoding=0,new_fit=0),family_total=dict(geometry_reserved=7300,geometry_attempted=4231,FT_attempts=107+count,FT_attempts_cap=212,update_interval=[hist['cumulative_update_interval'][0]+updates,hist['cumulative_update_interval'][1]+updates],update_upper_cap=36320,unknown_old_updates=[0,140]),accounting='history already nests P1/P2; old controls are output aliases, never new prediction/FT')

def report(root,log,error,phase,code):
    root=Path(root);b=read(root/'budget.json') if (root/'budget.json').exists() else dict(jobs=[],cumulative={});jobs=b['jobs'];cpu=state_for([j for j in jobs if j['kind']=='cpu'],1);pre=state_for([j for j in jobs if j['kind']=='preflight'],1);result=read(root/'final_review/pre_cpu_results.json') if (root/'final_review/pre_cpu_results.json').exists() else dict(FINAL=None,reference=.804675462276949,selected_macro=None,selected_delta=None,delta_C_FIXED=None,formal={},formal_complete=False,near_goal_met=False,target_met=False,performance_pass=False,comparisons={});first=read(root/'first_failure.json') if (root/'first_failure.json').exists() else None;complete=not error and cpu['status']=='PASSED';result.update(status='COMPLETE_PENDING_REVIEW' if complete else 'STOPPED',execution_complete=complete,final_cpu_pass=cpu['status']=='PASSED',preflight=pre,final_cpu=cpu,independent_review='PENDING',first_failure=first,error=error,phase=phase,cumulative=b.get('cumulative'),metrics_status=('正式数值保留、执行失败' if error else '正式评价完成') if result['formal_complete'] else '未完成/不可比较');result['exits']={n:read(root/f'{n}_exit.json') if (root/f'{n}_exit.json').exists() else dict(exit=code if n=='bootstrap' else None) for n in ('controller','outer','bootstrap')};write(root/'final_review/results.json',result)
    import re
    paper_rows=re.findall(r'\|\s*(Eea|Egb|Egc|Ei|EPS|Nc|Xc)\s*\|\s*([^|]+)\|\s*\*\*(\d+\.\d+)\s*±\s*(\d+\.\d+)\*\*\s*\|',(ROOT/'SOTA_RESULT.md').read_text())
    sota={name.lower():(float(mean),float(std)) for name,model,mean,std in paper_rows}
    if len(sota)!=7:raise ValueError('SOTA_RESULT seven-task reference parsing')
    sota={task:sota[task] for task in TASKS}
    lines=['# UA3D-HMR-READOUT-20261103-01/r1 交付','',f"状态：{result['status']}；{result['metrics_status']}；independent_review=PENDING。",f"FINAL={result['FINAL']}；macro={result['selected_macro']}；ΔBASE={result['selected_delta']}；ΔC_FIXED={result['delta_C_FIXED']}。",f"near={result['near_goal_met']}；target={result['target_met']}；performance_pass=target_met={result['performance_pass']}。",'性能目标达到。' if result['target_met'] else '已接近目标，尚未达到0.810。' if result['near_goal_met'] else '性能目标未达。',f"实际阶段：{b.get('counts',{})}；前置CPU={pre['status']}/{pre['exit']}；最终CPU={cpu['status']}/{cpu['exit']}。",f"累计账目：{b.get('cumulative')}。旧时钟、旧STOPPED与P3待审查状态不改写。",'']
    for arm in ARMS:
        lines+=[f'## {arm}','','|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|','|---|---:|---:|---:|---|']
        for task,(mean,std) in sota.items():
            v=result['formal'].get(arm,{}).get('tasks',{}).get(task);model='Periodic-TDL (+Chem, +RC)' if task in ('egb','egc') else 'Periodic-TDL';lines.append(f"|{task}|"+(f"{v['mean']:.3f} ± {v['std']:.3f}|{mean:.3f} ± {std:.3f}|{v['mean']-mean:+.3f}|{model}|" if v else f"未完成/不可比较|{mean:.3f} ± {std:.3f}|不可比较|{model}|"))
        if arm in result['formal']:
            value=result['formal'][arm]['macro'];lines+=['',f'macro={value:.3f}；ΔBASE={value-.804675462276949:+.3f}；原始值={value!r}。']
        lines+=['']
    if result.get('comparisons'):
        lines+=['## 预登记消融差','','|比较|Δmacro|','|---|---:|']
        for name,v in result['comparisons'].items():lines.append(f"|{name}|{v['macro']:+.3f}|")
        lines+=['','完整精度、逐任务/逐折配对差、35套BEST epoch见results.json。']
    lines+=['','七任务五折mean，std(ddof=1)，seed42+fold；论文参考SOTA_RESULT.md逐任务最佳配置包络，不是单一模型。Egc3380/论文4124、Xc432/论文430，协议不等价，仅数值对照。共享开发评价，非独立盲测或跨seed稳定。','N_BASE取消表面输入但仍包含原BASE的3D潜变量；N_ADD/N_BILIN为同参数量分支对照，普通MLP本身已有非线性交互。额外分支同时读取统计与谱，不能单独归因于谱；小差值不证明因果或显著性。有限Trimer含端部效应，不是周期链/真实体相。',f"首错：{first}",f"真实退出：{result['exits']}；详见各层subprocess wait文件。",'证据：authorization/、registry/sources.json/canonical.json/test_qualification、preflight/、diagnostics/、gates/、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/audit.json、budget.json/cumulative_budget.json。','105新预测与70旧输出复用独立计账，不将alias记为新训练/模型前向。没有重构面、求谱、BASE编码或统计重拟合。','不自动替换BASE、不部署、不追加搜索。']
    text='\n'.join(lines)+'\n';(root/'delivery_summary.md').write_text(text);(root/'final_review/report.md').write_text(text);write(root/'execution_manifest.json',dict(status=result['status'],execution_complete=complete,formal_complete=result['formal_complete'],final_cpu=cpu,preflight=pre,independent_review='PENDING'))
    if root==OUT:
        status=f"<!-- UA3D_HMR_READOUT_R1_STATUS_START -->\n- 状态：{result['status']}；{result['metrics_status']}；independent_review=PENDING。\n- 完成层级和新/旧消费分账见results/ua3d_hmr_readout_20261103/r1/budget.json及cumulative_budget.json；前置CPU={pre['status']}/{pre['exit']}，最终CPU={cpu['status']}/{cpu['exit']}。\n- FINAL={result['FINAL']}；正式macro={result['selected_macro']}；未舍入ΔBASE={result['selected_delta']}；near={result['near_goal_met']}，target/performance_pass={result['target_met']}。\n- 新时钟见authorization/clock.json；旧[18090,18230]、父现场与BASE不变；交付见delivery_summary.md。\n<!-- UA3D_HMR_READOUT_R1_STATUS_END -->"
        p=ROOT/'CURRENT_PLAN.md';s=p.read_text();a='<!-- UA3D_HMR_READOUT_R1_STATUS_START -->';z='<!-- UA3D_HMR_READOUT_R1_STATUS_END -->'
        if a in s:s=s[:s.index(a)]+status+s[s.index(z)+len(z):]
        else:s=s.replace('## 1. 依据、科学问题与目标',status+'\n\n## 1. 依据、科学问题与目标',1)
        s=s.replace('状态：**AUTHORIZED_RUNNING／已授权执行**。',f"状态：**{result['status']}**。")
        p.write_text(s)
        with (ROOT/'PROJECT_HISTORY.md').open('a') as f:f.write('\n\n## UA3D-HMR-READOUT-20261103-01/r1（待审查）\n\n'+status+'\n固定P2表面/谱与BASE表征；比较无表面、额外加性与低秩乘性分支。训练/TEST/独立审查分层，未完整正式评价不报告性能；旧HMR结论不追溯改判。\n')
        changes=subprocess.run(['git','diff','--','CURRENT_PLAN.md','PROJECT_HISTORY.md'],capture_output=True,text=True,check=True).stdout
        for name in SOURCES:changes+=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True).stdout
        (root/'actual_changes.diff').write_text(changes)
    return result

def bootstrap():
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    if (OUT/'bootstrap.lock').exists():raise RuntimeError('existing cycle root lock; no resume')
    def action(ctx):
        ctx['phase']='lock';lock=(OUT/'bootstrap.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_clock(True);ctx['phase']='config';spec=read(ROOT/'configs/ua3d_hmr_readout_r1_gpu2.json')
        if spec['resource_revision']!='gpu2-pt0-fttest8-hmr-readout-r1':raise ValueError('resource contract')
        ctx['phase']='source_lock'
        for name in SOURCES:
            p=OUT/'authorization/source_locked'/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('x') as f:f.write((ROOT/name).read_text())
        exclusive(OUT/'authorization/source_lock.json',dict(files=SOURCES,locked_unix=time.time(),hash_contract=False));ctx['phase']='outer';rc=wait_wrapper([str(PYTHON),str(ROOT/'scripts/run_ua3d_hmr_readout_r1.py'),'outer'],LOG/'outer.log',OUT,'outer',dict(os.environ,**THREADS));lock.close();return rc
    return envelope(OUT,LOG,action)
def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic-root');p.add_argument('--synthetic-outcome',choices=['success','failure']);a=p.parse_args()
    if a.synthetic_root:
        root=Path(a.synthetic_root).resolve()
        if ROOT==root or ROOT in root.parents or not (root/'SYNTHETIC_ONLY').exists():raise ValueError('synthetic fixture outside project required')
        def action(ctx):
            ctx['phase']='synthetic_import'
            import scripts.bootstrap_ua3d_hmr_readout_r1
            if a.synthetic_outcome=='failure':raise ValueError('injected synthetic first failure')
            write(root/'synthetic_success.json',dict(import_path=True,real_scientific_jobs=0));return 0
        return envelope(root,root/'logs',action)
    return bootstrap()
if __name__=='__main__':sys.exit(main())
