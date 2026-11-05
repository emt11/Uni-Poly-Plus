#!/usr/bin/env python3
"""Standard-library lifecycle, canonical resolver, bootstrap and durable closeout."""
import argparse
import datetime
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
P1=ROOT/'results/ua3d_hmr_20261103/r1'
P2=P1/'repairs/schema_surface_r2'
OUT=P1/'repairs/scheduler_r3'
LOG=ROOT/'logs/ua3d_hmr_20261103/r1/repairs/scheduler_r3'
PYTHON=P1/'env/bin/python'
SOURCES=('src/training/ua3d_hmr_r3.py','scripts/bootstrap_ua3d_hmr_r3.py','scripts/run_ua3d_hmr_r3.py','configs/ua3d_hmr_r3_gpu2.json','tests/test_ua3d_hmr_r3.py')
THREADS={'CUDA_VISIBLE_DEVICES':'','WORLD_SIZE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT)}
ALIASES={('S_HMR','egc',1):2,('S_STAT','eea',0):0}
TASKS=('eea','egb','egc','ei','eps','nc','xc')
ARMS=('S_STAT','S_FIXED','S_HMR')

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

def canonical_entry(root,arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('canonical matrix identity')
    e=read(Path(root)/'registry/canonical.json')['entries'][f'{arm}/{task}/{fold}'];expected='parent_complete_alias' if (arm,task,fold) in ALIASES else 'r3_new_ft'
    if e['source_kind']!=expected:raise ValueError('canonical source kind')
    parent=Path(read(Path(root)/'registry/canonical.json')['parent_root']);folder=(parent if expected=='parent_complete_alias' else Path(root))/'finetune'/arm/task/f'fold{fold}'
    if Path(e['folder'])!=folder or any(e[k]!=v for k,v in dict(arm=arm,task=task,fold=fold,seed=42+fold).items()):raise ValueError('canonical source path/identity')
    if expected=='parent_complete_alias' and e['expected_best_epoch']!=ALIASES[(arm,task,fold)]:raise ValueError('alias epoch')
    return e

def unwrap_science(saved,entry,config):
    if entry['source_kind']=='parent_complete_alias':payload=saved
    elif entry['source_kind']=='r3_new_ft':
        if saved.get('schema')!='ua3d-hmr-execution-r3' or saved.get('science_revision')!='r2' or saved.get('execution_revision')!='r3' or saved.get('source_kind')!='r3_new_ft':raise ValueError('r3 wrapper identity')
        payload=saved['science_payload']
    else:raise ValueError('unknown source kind')
    if payload.get('schema')!='ua3d-hmr-residual-r2' or payload.get('config')!=config:raise ValueError('scientific schema/config differs')
    if any(payload[k]!=entry[k] for k in ('arm','task','fold','seed')):raise ValueError('scientific payload task/fold/seed')
    if payload['epoch'] not in range(71):raise ValueError('scientific epoch')
    return payload

def require_frozen(root):
    p=Path(root)/'selection/final_frozen.json'
    if not p.exists() or not read(p)['frozen']:raise ValueError('TEST reader before r3 freeze')

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
def cumulative(new_jobs,known_updates):
    count=sum(j['kind']=='ft' for j in new_jobs)
    if count>103 or known_updates>17150:raise ValueError('r3 new FT budget')
    return dict(geometry_reserved=7300,geometry_attempted=4231,preflight_total=3,FT_attempts=4+count,FT_attempts_cap=107,parent_recorded_updates=940,new_recorded_updates=known_updates,cumulative_recorded_updates=940+known_updates,old_unknown_updates=dict(each=[0,70],sum=[0,140]),cumulative_update_interval=[940+known_updates,1080+known_updates],update_upper_cap=18230,accounting='r2 already nests r1; never add ancestor geometry again; aliases are not new FT')
def fail(phase,error,code=1):
    if not (OUT/'first_failure.json').exists():exclusive(OUT/'first_failure.json',dict(phase=phase,error=error,exit=code,time=time.time()))
    write(OUT/'execution_manifest.json',dict(status='STOPPED',phase=phase,execution_complete=False,independent_review='PENDING',exit=code))
def check_clock(preparation=False,reserve=2400):
    c=read(OUT/'authorization/clock.json');now=time.time()
    if now>=c['deadline_unix']-reserve or preparation and now>=c['prepare_deadline_unix']:raise TimeoutError('inherited deadline/reserve')

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

def report(root,log,error,phase,code):
    root=Path(root);b=read(root/'budget.json') if (root/'budget.json').exists() else dict(jobs=[],cumulative=cumulative([],0));jobs=b['jobs'];cpu=state_for([j for j in jobs if j['kind']=='cpu'],1);pre=state_for([j for j in jobs if j['kind']=='preflight'],1)
    result=read(root/'final_review/pre_cpu_results.json') if (root/'final_review/pre_cpu_results.json').exists() else dict(reference=.804675462276949,FINAL=None,formal={},formal_complete=False,selected_macro=None,selected_delta=None,near_goal_met=False,target_met=False,performance_pass=False)
    first=read(root/'first_failure.json') if (root/'first_failure.json').exists() else None;complete=error is None and cpu['status']=='PASSED';result.update(status='COMPLETE_PENDING_REVIEW' if complete else 'STOPPED',execution_complete=complete,science_revision='r2',execution_revision='r3',resource_revision='gpu2-pt0-fttest8-hmr-r3',independent_review='PENDING',error=error,first_failure=first,phase=phase,preflight=pre,final_cpu=cpu,final_cpu_pass=cpu['status']=='PASSED',cumulative=b.get('cumulative'),metrics_status=('正式数值保留、执行失败' if error else '正式输出完成') if result['formal_complete'] else '未完成/不可比较')
    result['exits']={name:read(root/f'{name}_exit.json') if (root/f'{name}_exit.json').exists() else dict(exit=code if name=='bootstrap' else None) for name in ('controller','outer','bootstrap')};write(root/'final_review/results.json',result)
    sota={'eea':(.934,.013),'egb':(.923,.009),'egc':(.911,.008),'ei':(.839,.018),'eps':(.787,.033),'nc':(.856,.032),'xc':(.423,.102)}
    lines=['# UA3D-HMR-20261103-01 science r2 / execution r3','','状态：'+result['status']+'；'+result['metrics_status']+'；independent_review=PENDING。',f"FINAL={result['FINAL']}；reference=0.804675462276949；macro={result['selected_macro']}；未舍入Δ={result['selected_delta']}。",f"near={result['near_goal_met']}；target={result['target_met']}；performance_pass={result['performance_pass']}；最后CPU={cpu['status']}/{cpu['exit']}。",f"实际阶段：{b.get('counts',{})}；2 parent aliases不计新训练。",f"累计账目：{b.get('cumulative')}。父r1/r2仍STOPPED/三层exit1，全部失败、中止、旧预约不退款。",'']
    for arm in ('C_ANCHOR',*ARMS):
        lines.extend([f'## {arm}','','|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|','|---|---:|---:|---:|---|'])
        for task,(mean,std) in sota.items():
            v=result['formal'].get(arm,{}).get('tasks',{}).get(task);model='Periodic-TDL (+Chem, +RC)' if task in ('egb','egc') else 'Periodic-TDL'
            lines.append(f"|{task}|"+(f"{v['mean']:.3f} ± {v['std']:.3f}|{mean:.3f} ± {std:.3f}|{v['mean']-mean:+.3f}|{model}|" if v else f"未完成/不可比较|{mean:.3f} ± {std:.3f}|不可比较|{model}|"))
        if arm in result['formal']:
            macro=result['formal'][arm]['macro'];lines+=['',f'macro={macro:.3f}；ΔBASE={macro-.804675462276949:+.3f}；完整精度见results.json。']
        lines+=['']
    lines+=['七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置逐任务包络；Egc3380/论文4124，Xc432/论文430，协议不等价，仅数值对照。共享开发TEST，非独立盲测或跨seed稳定。','有限Trimer软球表面/全部闭合分量/显式零模沿用r2；不是电子密度、周期无限链或体相自由体积。',f"首错：{first}",f"退出：{result['exits']}。真实exit以各层subprocess wait文件为准；最后CPU仅在全部140输出后一次。",'证据：authorization/、registry/canonical.json及parent_qualification.json、lifecycle/、budget.json、cumulative_budget.json、finetune/、selection/final_frozen.json、formal/*/test_predictions.npz、final_cpu/与对应logs。','不自动替换BASE、不部署、不追加搜索。']
    text='\n'.join(lines)+'\n';(root/'delivery_summary.md').write_text(text);(root/'final_review/report.md').write_text(text);write(root/'execution_manifest.json',dict(status=result['status'],execution_complete=complete,preflight=pre,final_cpu=cpu,formal_complete=result['formal_complete'],independent_review='PENDING'))
    if root==OUT:
        status=f"- 状态：{result['status']}；{result['metrics_status']}；independent_review=PENDING。\n- science_revision=r2，execution_revision=r3；父r1/r2保持STOPPED/三层exit1；原tN及截止不变。\n- 完成层级与消费见本根budget.json；canonical为2 parent alias+103新FT合同；前置CPU={pre['status']}/{pre['exit']}，最后CPU={cpu['status']}/{cpu['exit']}。\n- FINAL={result['FINAL']}；正式macro={result['selected_macro']}；未舍入Δ={result['selected_delta']}；reference不变。\n- 交付：results/ua3d_hmr_20261103/r1/repairs/scheduler_r3/delivery_summary.md；真实exit、首错、累计U区间和原始预测保留。\n"
        p=ROOT/'CURRENT_PLAN.md';s=p.read_text();a='<!-- UA3D_HMR_R3_STATUS_START -->';z='<!-- UA3D_HMR_R3_STATUS_END -->';p.write_text(s[:s.index(a)+len(a)]+'\n'+status+s[s.index(z):])
        with (ROOT/'PROJECT_HISTORY.md').open('a') as f:f.write('\n\n## UA3D-HMR-20261103-01/r3（science r2，待审查）\n\n'+status+'\n本修订仅生命周期调度与canonical解析，不改变模型/表面数学。父失败不覆盖，含旧两中止未知U∈[0,140]。完整正式输出前未完成/不可比较，不用validation填补。\n')
        changes=subprocess.run(['git','diff','--','CURRENT_PLAN.md','PROJECT_HISTORY.md'],capture_output=True,text=True,check=True).stdout
        for name in SOURCES:changes+=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True).stdout
        (root/'actual_changes.diff').write_text(changes)
    return result

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

def bootstrap():
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    if (OUT/'bootstrap.lock').exists():raise RuntimeError('existing root: no resume')
    def action(ctx):
        ctx['phase']='lock';lock=(OUT/'bootstrap.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_clock(True)
        ctx['phase']='config';config=read(ROOT/'configs/ua3d_hmr_r3_gpu2.json')
        if config['resource_revision']!='gpu2-pt0-fttest8-hmr-r3':raise ValueError('resource revision')
        ctx['phase']='source_lock'
        for name in SOURCES:
            p=OUT/'authorization/source_locked'/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('x') as f:f.write((ROOT/name).read_text())
        exclusive(OUT/'authorization/source_lock.json',dict(files=SOURCES,locked_unix=time.time(),hashes=False));ctx['phase']='outer';rc=wait_wrapper([str(PYTHON),str(ROOT/'scripts/run_ua3d_hmr_r3.py'),'outer'],LOG/'outer.log',OUT,'outer',dict(os.environ,**THREADS));lock.close();return rc
    return envelope(OUT,LOG,action)

def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic-root');p.add_argument('--synthetic-outcome',choices=['success','failure']);a=p.parse_args()
    if a.synthetic_root:
        root=Path(a.synthetic_root).resolve()
        if ROOT==root or ROOT in root.parents or not (root/'SYNTHETIC_ONLY').exists():raise ValueError('synthetic root forbidden')
        def action(ctx):
            ctx['phase']='synthetic_import_initialize'
            import scripts.bootstrap_ua3d_hmr_r3
            if a.synthetic_outcome=='failure':raise ValueError('injected synthetic first failure')
            write(root/'synthetic_success.json',dict(import_path=True,real_scientific_jobs=0));return 0
        return envelope(root,root/'logs',action)
    return bootstrap()
if __name__=='__main__':sys.exit(main())
