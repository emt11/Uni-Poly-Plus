#!/usr/bin/env python3
"""Standard-library launch, irreversible jobs and first-failure preservation."""
import argparse
import datetime
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
PARENT_RUN=ROOT/'results/ua3d_hmr_20261103/r1'
OUT=PARENT_RUN/'repairs/schema_surface_r2'
LOG=ROOT/'logs/ua3d_hmr_20261103/r1/repairs/schema_surface_r2'
SOURCES=('src/dataset/ua3d_hmr_r2.py','src/modules/ua3d_hmr_r2.py','src/training/ua3d_hmr_r2.py','scripts/prepare_ua3d_hmr_r2.py','scripts/run_ua3d_hmr_r2.py','scripts/bootstrap_ua3d_hmr_r2.py','configs/ua3d_hmr_r2_gpu2.json','tests/test_ua3d_hmr_r2.py')
THREADS={'CUDA_VISIBLE_DEVICES':'','WORLD_SIZE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT)}

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def identity(pid=None):
    pid=pid or os.getpid();s=Path(f'/proc/{pid}/stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))
def parse_jobs(jobs):
    ids=set();counts={};intervals=[];wall=0
    for j in jobs:
        key=j['id']
        if key in ids:raise ValueError('duplicate job identity')
        ids.add(key)
        for k in ('reserved_unix','started_unix','ended_unix'):
            if k not in j:raise ValueError('missing time field '+k)
            if j[k] is not None and (not isinstance(j[k],(float,int)) or not math.isfinite(j[k])):raise ValueError('time type')
        if j['started_unix'] is not None and j['started_unix']<j['reserved_unix']:raise ValueError('start before reservation')
        if j['ended_unix'] is not None and (j['started_unix'] is None or j['ended_unix']<j['started_unix']):raise ValueError('end without start')
        c=counts.setdefault(j['kind'],dict(reserved=0,started=0,attempted=0,completed=0));c['reserved']+=1;c['started']+=int(j['started_unix'] is not None);c['attempted']+=int(j['started_unix'] is not None);c['completed']+=int(j['exit']==0)
        if j['started_unix'] is not None:
            end=j['ended_unix'] or time.time();wall+=end-j['started_unix']
            if j['gpu']:intervals.append((j['started_unix'],end))
    busy=0;last=0
    for a,b in sorted(intervals):busy+=max(0,b-max(a,last));last=max(last,b)
    return dict(counts=counts,process_wall_seconds=wall,physical_gpu_union_seconds=busy)
def state_for(jobs,expected):
    parse_jobs(jobs)
    return dict(status='NOT_RUN' if not jobs else 'FAILED' if any(j['exit'] not in (None,0) for j in jobs) else 'RUNNING' if any(j['exit'] is None for j in jobs) else 'PASSED' if len(jobs)==expected else 'PARTIAL',attempts=len(jobs),exit=next((j['exit'] for j in jobs if j['exit'] not in (None,0)),None) if any(j['exit']!=0 for j in jobs) else 0 if len(jobs)==expected else None)
def size(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
def fail(phase,error,code=1):
    if not (OUT/'first_failure.json').exists():exclusive(OUT/'first_failure.json',dict(phase=phase,error=error,exit=code,time=time.time()))
    write(OUT/'execution_manifest.json',dict(status='STOPPED',phase=phase,execution_complete=False,independent_review='PENDING',error_source='first_failure.json',exit=code))
def check_clock(preparation=False,reserve=2400):
    c=read(OUT/'authorization/clock.json');now=time.time()
    if now>=c['deadline_unix']-reserve or (preparation and now>=c['prepare_deadline_unix']):raise TimeoutError('deadline/preparation reserve')

def report(root,log,error,phase,code):
    """Total standard-library closeout; reads only already saved JSON/files."""
    root=Path(root);log=Path(log)
    if error and not (root/'first_failure.json').exists():exclusive(root/'first_failure.json',dict(phase=phase,error=error,exit=code,time=time.time()))
    b=read(root/'budget.json') if (root/'budget.json').exists() else dict(jobs=[],counts={})
    jobs=b.get('jobs',[]);cpu=state_for([j for j in jobs if j['kind']=='cpu'],1);pre=state_for([j for j in jobs if j['kind']=='preflight'],1)
    value=read(root/'final_review/pre_cpu_results.json') if (root/'final_review/pre_cpu_results.json').exists() else dict(reference=.804675462276949,FINAL=None,formal={},formal_complete=False,selected_macro=None,selected_delta=None,near_goal_met=False,target_met=False,performance_pass=False)
    complete=not error and cpu['status']=='PASSED';value.update(plan='UA3D-HMR-20261103-01',revision='r2',resource_revision='gpu2-pt0-cpuall-fttest8-hmr-r2',execution_complete=complete,independent_review='PENDING',status='COMPLETE_PENDING_REVIEW' if complete else 'STOPPED',error=error,phase=phase,preflight=pre,final_cpu=cpu,final_cpu_pass=cpu['status']=='PASSED',metrics_status=('正式数值保留、执行失败' if error else '正式输出完成') if value['formal_complete'] else '未完成/不可比较',budget=b)
    for name in ('controller','outer','bootstrap'):
        path=root/f'{name}_exit.json';value[name+'_exit']=read(path) if path.exists() else dict(exit=code if name=='bootstrap' else None,status='NOT_RUN' if name!='bootstrap' else 'FINALIZING')
    parent=root/'authorization/parent_budget.json';pb=read(parent) if parent.exists() else {};surface=root/'surface/structures';ends=[read(p) for p in surface.glob('*/end.json')];starts=[read(p) for p in surface.glob('*/process.json')]
    states=[read(p) for p in surface.glob('*/status.json')];geom=dict(reserved=len(read(root/'surface/queue.json')) if (root/'surface/queue.json').exists() else 0,attempted=len(starts),completed=sum(e['exit']==0 for e in ends),failed=sum(e['exit']!=0 for e in ends),without_end=len(starts)-len(ends),usable=sum(s['available'] for s in states),unavailable={})
    for s in states:
        if not s['available']:geom['unavailable'][s['reason']]=geom['unavailable'].get(s['reason'],0)+1
    ps=pb.get('surface_accounting',{});cumulative=dict(parent_reserved=ps.get('reserved',3650),parent_attempted=ps.get('attempted',581),child=geom,cumulative_reserved=ps.get('reserved',3650)+geom['reserved'],cumulative_attempted=ps.get('attempted',581)+geom['attempted'],reserved_cap=7300,attempted_cap=4231,parent_budget_snapshot=pb,parent_surface_seconds=sum(read(p)['ended_unix']-read(p)['started_unix'] for p in (PARENT_RUN/'surface/structures').glob('*/end.json')) if root==OUT else None,parent_interrupted_seconds='unknown; 106 no-end starts retained, do not fill zero',child_surface_completed_process_seconds=sum(e['ended_unix']-e['started_unix'] for e in ends),child_without_end_seconds='unknown; start and pool-stop evidence retained')
    comparisons=[]
    if root==OUT:
        for p in surface.glob('*/status.json'):
            prior=PARENT_RUN/'surface/structures'/p.parent.name/'status.json';current=read(p);old=read(prior) if prior.exists() else None;comparisons.append(dict(id=p.parent.name,r1_reason=old.get('reason') if old else 'NOT_COMPLETED',r1_available=old.get('available') if old else None,r2_reason=current['reason'],r2_available=current['available']))
    write(root/'surface/r1_r2_partial_comparison.json',comparisons)
    value['cumulative']=cumulative;write(root/'cumulative_budget.json',cumulative);write(root/'final_review/results.json',value)
    sota={'eea':(.934,.013),'egb':(.923,.009),'egc':(.911,.008),'ei':(.839,.018),'eps':(.787,.033),'nc':(.856,.032),'xc':(.423,.102)}
    lines=['# UA3D-HMR-20261103-01/r2 交付','',f"状态：{value['status']}；{value['metrics_status']}；independent_review=PENDING。",f"reference=0.804675462276949；FINAL={value['FINAL']}；macro={value['selected_macro']}；Δ={value['selected_delta']}。",f"前置CPU={pre['status']}/{pre['exit']}；最终CPU={cpu['status']}/{cpu['exit']}。",f"表面：{geom}；累计预留={cumulative['cumulative_reserved']}/7300，累计尝试={cumulative['cumulative_attempted']}/4231。",'父r1仍STOPPED/三层exit1；旧预约、失败与106中止不退款，不重写。','']
    for arm in ('C_ANCHOR','S_STAT','S_FIXED','S_HMR'):
        lines.extend([f'## {arm} 七任务论文数值对照','','|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|','|---|---:|---:|---:|---|'])
        for task,(mean,std) in sota.items():
            v=value.get('formal',{}).get(arm,{}).get('tasks',{}).get(task);model='Periodic-TDL (+Chem, +RC)' if task in ('egb','egc') else 'Periodic-TDL'
            lines.append(f"|{task}|"+(f"{v['mean']:.3f} ± {v['std']:.3f}|{mean:.3f} ± {std:.3f}|{v['mean']-mean:+.3f}|{model}|" if v else f"未完成/不可比较|{mean:.3f} ± {std:.3f}|不可比较|{model}|"))
        lines+=['']
    lines+=['七任务五折mean，std(ddof=1)，seed=42+fold。论文macro是不同配置包络；Egc3380/论文4124，Xc432/论文430，协议不等价，只作数值对照。共享开发TEST，不称独立盲测或跨seed稳定。','表面为tau=0.15 Å、rho0=1.000001、尾界1e-10的软球隐式面；多分量均保留，统一64模与显式零模；不是电子密度或无限周期体相。',f'错误：{error}',f'真实退出见controller_exit.json、outer_exit.json、bootstrap_exit.json。证据：budget.json、cumulative_budget.json、surface/、diagnostics/、finetune/、selection/、formal/及对应logs。','本周期不自动替换BASE、不追加搜索；错误后不修复重试。']
    text='\n'.join(lines)+'\n';(root/'delivery_summary.md').write_text(text);(root/'final_review/report.md').write_text(text)
    write(root/'execution_manifest.json',dict(status=value['status'],execution_complete=complete,independent_review='PENDING',phase=phase,preflight=pre,final_cpu=cpu,formal_complete=value['formal_complete']))
    # Only real production reporting edits the single authorized STATUS block/history.
    if root==OUT:
        status=f"- 状态：{value['status']}；{value['metrics_status']}；independent_review=PENDING。\n- 仅r2子根 `{OUT.relative_to(ROOT)}`；父r1维持STOPPED/三层exit1；原tN/截止不变。\n- 前置CPU={pre['status']}/{pre['exit']}；最终CPU={cpu['status']}/{cpu['exit']}；几何子预留{geom['reserved']}/尝试{geom['attempted']}；父子累计预留{cumulative['cumulative_reserved']}/尝试{cumulative['cumulative_attempted']}。\n- FINAL={value['FINAL']}；macro={value['selected_macro']}；reference=0.804675462276949；证据见本子根delivery_summary.md、cumulative_budget.json、真实exit与first_failure.json（如有）。\n"
        p=ROOT/'CURRENT_PLAN.md';s=p.read_text();a='<!-- UA3D_HMR_R2_STATUS_START -->';z='<!-- UA3D_HMR_R2_STATUS_END -->';p.write_text(s[:s.index(a)+len(a)]+'\n'+status+s[s.index(z):])
        with (ROOT/'PROJECT_HISTORY.md').open('a') as f:f.write('\n\n## UA3D-HMR-20261103-01/r2（待审查）\n\n'+status+'\n受控修订：合法负哨兵、平滑隐式表面与全部闭合分量/显式零模、标准库收尾。父失败不改判，r2不是纯工程等价修复；未完整TEST时未完成/不可比较。不替换BASE。\n')
        changes=subprocess.run(['git','diff','--','CURRENT_PLAN.md','PROJECT_HISTORY.md'],capture_output=True,text=True,check=True).stdout
        for name in SOURCES:changes+=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True).stdout
        (root/'actual_changes.diff').write_text(changes)
    return value

def envelope(root,log,action):
    """Shared real/synthetic bootstrap path, including construction and report failures."""
    root=Path(root);log=Path(log);ctx=dict(phase='initialize');rc=1;error=None
    try:
        root.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True);rc=action(ctx)
        if rc:raise RuntimeError('outer/controller nonzero exit '+str(rc))
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
    if (OUT/'bootstrap.lock').exists():raise RuntimeError('Existing root/lock: no resume or overwrite')
    def action(ctx):
        ctx['phase']='lock';lock=(OUT/'bootstrap.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_clock(True);ctx['phase']='config';cfg=read(ROOT/'configs/ua3d_hmr_r2_gpu2.json')
        if cfg['resource_revision']!='gpu2-pt0-cpuall-fttest8-hmr-r2':raise ValueError('r2 resource identity')
        ctx['phase']='source_lock'
        for name in SOURCES:
            p=OUT/'authorization/source_locked'/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('x') as f:f.write((ROOT/name).read_text())
        exclusive(OUT/'authorization/source_lock.json',dict(files=SOURCES,locked_unix=time.time(),no_hashes=True))
        # Import path of THIS process is set above, not only child environment.
        ctx['phase']='parent_environment';python=PARENT_RUN/'env/bin/python'
        if not python.exists():raise ValueError('parent environment missing')
        for name in ('controller','outer','bootstrap'):
            if read(PARENT_RUN/f'{name}_exit.json')['exit']!=1:raise ValueError('parent failure identity')
        env=dict(os.environ,**THREADS);cmd=[str(python),str(ROOT/'scripts/run_ua3d_hmr_r2.py'),'outer'];ctx['phase']='outer'
        with (LOG/'outer.log').open('x') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);exclusive(OUT/'outer_process.json',dict(**identity(p.pid),argv=cmd,cwd=str(ROOT),env=THREADS,started_unix=time.time()));rc=p.wait()
        exclusive(OUT/'outer_exit.json',dict(exit=rc,ended_unix=time.time()));lock.close();return rc
    return envelope(OUT,LOG,action)

def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic-root');p.add_argument('--synthetic-outcome',choices=['success','failure']);a=p.parse_args()
    if a.synthetic_root:
        root=Path(a.synthetic_root).resolve()
        if root==OUT or ROOT in root.parents or not (root/'SYNTHETIC_ONLY').exists():raise ValueError('synthetic path forbidden')
        def action(ctx):
            ctx['phase']='synthetic_config_import_initialize_run'
            import scripts.bootstrap_ua3d_hmr_r2
            if a.synthetic_outcome=='failure':raise ValueError('synthetic injected first error')
            write(root/'synthetic_success.json',dict(import_path_ok=True,current_cwd=str(Path.cwd()),real_jobs=0));return 0
        return envelope(root,root/'logs',action)
    return bootstrap()
if __name__=='__main__':sys.exit(main())
