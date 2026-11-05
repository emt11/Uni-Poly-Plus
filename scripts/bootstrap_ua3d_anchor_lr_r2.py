#!/usr/bin/env python3
"""Standard-library metadata contract, immutable first failure, and three real waits."""
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
OUT=ROOT/'results/ua3d_anchor_lr_20261102/r1/repairs/init_r2'
LOG=ROOT/'logs/ua3d_anchor_lr_20261102/r1/repairs/init_r2'
CONFIG=ROOT/'configs/ua3d_anchor_lr_r2_gpu2.json'
SOURCES=('src/dataset/ua3d_anchor_lr_r2.py','src/modules/ua3d_anchor_lr_r2.py','src/training/ua3d_anchor_lr_r2.py',
 'scripts/prepare_ua3d_anchor_lr_r2.py','scripts/run_ua3d_anchor_lr_r2.py','scripts/bootstrap_ua3d_anchor_lr_r2.py',
 'configs/ua3d_anchor_lr_r2_gpu2.json','tests/test_ua3d_anchor_lr_r2.py')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def identity(pid=None):
    pid=os.getpid() if pid is None else pid;s=Path('/proc',str(pid),'stat').read_text()
    return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))


def required(data,key,source):
    if key not in data:raise ValueError(f'missing metadata field {source}.{key}')
    return data[key]


def number(value,name,integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (integer and int(value)!=value):
        raise ValueError('invalid nonnegative '+name)
    return value


def validate_contract(activation,spec,parent_activation):
    """Production and synthetic preflight use this same strict parser."""
    for key in ('schema','plan','revision','resource_revision','tN_unix','deadline_unix','preparation_deadline_unix','repair_started_unix',
                'parent_failure_root','scientific_cache_root','base_root','cache_reuse','estimated_new_artifact_bytes'):
        required(activation,key,'activation')
    if activation['schema']!='ua3d-anchor-lr-activation-r2' or activation['plan']!='UA3D-ANCHOR-LR-20261102-01' or activation['revision']!='r2' or activation['resource_revision']!='gpu2-pt0-fttest4-anchor-lr-r2':
        raise ValueError('activation r2 identity')
    for key in ('plan','revision','resource_revision','limits','artifact_estimate_limit_bytes','metadata_schema'):
        required(spec,key,'config')
    if any(spec[key]!=activation[key] for key in ('plan','revision','resource_revision')) or spec['metadata_schema']!=activation['schema']:
        raise ValueError('config/activation identity differs')
    for key in ('tN_unix','deadline_unix'):
        number(activation[key],key)
        if activation[key]!=required(parent_activation,key,'parent_activation'):raise ValueError('inherited clock mismatch: '+key)
    if activation['deadline_unix']!=activation['tN_unix']+86400 or activation['preparation_deadline_unix']!=activation['tN_unix']+14400:
        raise ValueError('24h/4h clock contract')
    number(activation['repair_started_unix'],'repair_started_unix')
    if not activation['tN_unix']<=activation['repair_started_unix']<activation['preparation_deadline_unix']:raise ValueError('repair outside original preparation window')
    expected={'parent_failure_root':'results/ua3d_anchor_lr_20261102/r1','scientific_cache_root':'results/ua3d_anchor_opt_20261102/r2','base_root':'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'}
    for key,suffix in expected.items():
        if Path(activation[key])!=ROOT/suffix:raise ValueError('source role/path mismatch: '+key)
    reuse=activation['cache_reuse']
    for key in ('expected_reused_bytes','expected_source','new_teacher_cache_bytes','new_teacher_cache_units','new_encoder_batches'):
        required(reuse,key,'activation.cache_reuse')
    number(reuse['expected_reused_bytes'],'cache_reuse.expected_reused_bytes',True)
    if reuse['expected_reused_bytes']<=0 or Path(reuse['expected_source'])!=Path(activation['scientific_cache_root'])/'budget.json':raise ValueError('reuse estimate source')
    for key in ('new_teacher_cache_bytes','new_teacher_cache_units','new_encoder_batches'):
        if number(reuse[key],key,True)!=0:raise ValueError('new cache/encoder forbidden: '+key)
    estimate=number(activation['estimated_new_artifact_bytes'],'estimated_new_artifact_bytes',True)
    if estimate>number(spec['artifact_estimate_limit_bytes'],'artifact_estimate_limit_bytes',True) or spec['artifact_estimate_limit_bytes']>8*1024**3:
        raise ValueError('new artifacts estimate exceeds8GiB')
    if 'estimated_cache_bytes' in activation:raise ValueError('obsolete generated-cache field forbidden in reuse activation')
    for key in ('cache_bytes','full_encoder_batches','encoder_U_batches','new_features','p_learn_reductions','pt_starts','pt_updates'):
        if required(spec['limits'],key,'config.limits')!=0:raise ValueError('nonzero forbidden generation/update budget: '+key)
    return dict(schema=activation['schema'],expected_reused_bytes=reuse['expected_reused_bytes'],estimated_new_artifact_bytes=estimate,
                tN_unix=activation['tN_unix'],deadline_unix=activation['deadline_unix'],preparation_deadline_unix=activation['preparation_deadline_unix'],repair_started_unix=activation['repair_started_unix'])


def state_for(jobs,expected):
    if not jobs:return dict(status='NOT_RUN',attempts=0,exit=None,started=None,ended=None)
    failure=next((j['exit'] for j in jobs if j['exit'] not in (None,0)),None);active=any(j['exit'] is None for j in jobs)
    return dict(status='FAILED' if failure is not None else 'RUNNING' if active else 'PASSED' if len(jobs)==expected else 'PARTIAL',
        attempts=len(jobs),expected=expected,exit=failure if failure is not None else None if active or len(jobs)<expected else 0,
        started=min(j['started_unix'] if 'started_unix' in j else j['reserved_unix'] for j in jobs),
        ended=None if active else max(j['ended_unix'] if 'ended_unix' in j else j['reserved_unix'] for j in jobs))


def scientific_ledger(jobs,spec,contract,activation,parent_accounting,root):
    limits=spec['limits'];reserved=dict.fromkeys(limits,0);attempted=dict(reserved);completed=dict(reserved);started=dict(reserved)
    starts=dict(preflight='preflight_cpu_suites',cache_trainval='cache_trainval_units',cache_test='cache_test_units',diagnostic='diagnostic_starts',ft='ft_starts',test='test_starts',cpu='cpu_suites')
    intervals=[];wall=0.;identities=set()
    for job in jobs:
        ident=tuple(job['identity'])
        if ident in identities:raise ValueError('duplicate scientific identity')
        identities.add(ident)
        for key,value in job['reservation'].items():reserved[key]+=number(value,'reservation.'+key,True)
        key=starts[job['kind']];p=Path(job['log'])/'process.json'
        if p.exists():
            started[key]+=1;attempted[key]+=1;process=read(p);end=job['ended_unix'] if 'ended_unix' in job else time.time();wall+=end-process['started_unix']
            if job['uses_gpu']:intervals.append((process['started_unix'],end))
        path=Path(job['consumption'])
        if path.exists():
            usage=read(path)
            if 'attempted' in usage:
                attempt=usage['attempted'];done=usage['completed']
            else:
                attempt=usage;done=usage if usage['status']=='COMPLETE' else {}
            for name in job['reservation']:
                if name==key:continue
                attempted[name]+=int(attempt[name]) if name in attempt else 0
                completed[name]+=int(done[name]) if name in done else 0
        if job['exit']==0:completed[key]+=1
    for target in (reserved,attempted,completed,started):target['total_updates']=target['ft_updates']+target['diagnostic_updates']
    reserved['freezes']=attempted['freezes']=int((root/'selection/freeze_reservation.json').exists());completed['freezes']=int((root/'selection/final_frozen.json').exists())
    busy=0.;last=0.
    for a,b in sorted(intervals):busy+=max(0.,b-max(a,last));last=max(last,b)
    violations=[key for key in limits if reserved[key]>limits[key] or attempted[key]>reserved[key] or completed[key]>attempted[key]]
    if busy>86400:violations.append('physical_GPU_busy_seconds')
    return dict(**contract,limits=limits,reserved=reserved,started=started,attempted=attempted,completed=completed,jobs=jobs,budget_violations=violations,
        physical_gpu_busy_hours=busy/3600,process_wall_seconds=wall,tN_unix=activation['tN_unix'],deadline_unix=activation['deadline_unix'],parent_accounting=parent_accounting,
        accounting='irreversible reservations; no refunds; nested ancestor totals are not added twice; reused teacher bytes independent of new artifacts/encoder0')


def artifact_size(*roots):
    total=0
    for root in roots:
        if not Path(root).exists():continue
        for folder,dirs,files in os.walk(root,followlinks=False):
            for name in files:
                path=Path(folder)/name
                try:total+=path.lstat().st_size
                except FileNotFoundError:pass  # Atomic replacements may remove only the old temporary name.
    return total


def initialize_metadata(root,log,activation,spec,parent_activation,manifest,jobs,parent_accounting,contract):
    parsed=validate_contract(activation,spec,parent_activation)
    runtime=MetadataRuntime(root,log,activation,spec,parent_activation,manifest,jobs,parent_accounting,contract)
    return runtime,runtime.refresh()


class MetadataRuntime:
    def __init__(self,root,log,activation,spec,parent_activation,manifest,jobs,parent_accounting,contract):
        self.root=Path(root);self.log=Path(log);self.activation=activation;self.spec=spec;self.parent_activation=parent_activation
        self.manifest=manifest;self.jobs=jobs;self.parent=parent_accounting;self.contract=contract
    def refresh(self):
        parsed=validate_contract(self.activation,self.spec,self.parent_activation)
        b=scientific_ledger(self.jobs,self.spec,self.contract,self.activation,self.parent,self.root)
        reused=0
        for j in self.jobs:
            if j['kind'] in ('cache_trainval','cache_test') and Path(j['consumption']).exists():
                usage=read(j['consumption'])
                if 'reused_bytes' in usage:reused+=number(usage['reused_bytes'],'consumption.reused_bytes',True)
        cache=dict(schema='ua3d-anchor-lr-reuse-budget-r2',new_teacher_cache_bytes=0,new_teacher_cache_units=0,new_encoder_batches=0,
            expected_reused_bytes=parsed['expected_reused_bytes'],reused_bytes=reused,reuse_source=self.activation['scientific_cache_root'],
            units_completed=dict(trainval=b['completed']['cache_trainval_units'],test=b['completed']['cache_test_units']),capacity_semantics='expected source saved cache footprint; actual qualified referenced row tensor bytes; no new generation')
        disk=dict(schema='ua3d-anchor-lr-artifact-budget-r2',estimated_new_artifact_bytes=parsed['estimated_new_artifact_bytes'],actual_new_artifact_bytes=artifact_size(self.root,self.log),observed_unix=time.time(),
            estimate_limit_bytes=self.spec['artifact_estimate_limit_bytes'],roots=[str(self.root),str(self.log)],exclude='all parent teacher files and parent LR files')
        write(self.root/'budget.json',b);write(self.root/'cache_budget.json',cache);write(self.root/'artifact_budget.json',disk);write(self.root/'metadata_contract.json',parsed)
        for stage,kind,count in [('preflight','preflight',1),('cache_trainval','cache_trainval',35),('diagnostic','diagnostic',16),('ft','ft',140),('cache_test','cache_test',35),('final_cpu','cpu',1)]:
            self.manifest['stages'][stage]=state_for([j for j in self.jobs if j['kind']==kind],count)
        self.manifest['stages']['pilot']=state_for([j for j in self.jobs if 'pilot' in j and j['pilot']],1)
        self.manifest['stages']['anchor_output']=state_for([j for j in self.jobs if j['kind']=='test' and j['arm']=='C_ANCHOR'],35)
        self.manifest['stages']['adapter_output']=state_for([j for j in self.jobs if j['kind']=='test' and j['arm']!='C_ANCHOR'],140)
        write(self.root/'execution_manifest.json',self.manifest)
        if b['budget_violations']:raise ValueError('scientific budget violation '+str(b['budget_violations']))
        return b


def stage(root,name):
    write(Path(root)/'launch_phase.json',dict(phase=name,time=now(),process=identity()))


def record_failure(root,phase,error,code=1,process=None):
    """No imports/refresh/report/encoder. Preserve first error even if cleanup also fails."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    first=root/'first_failure.json'
    if not first.exists():
        exclusive(first,dict(phase=phase,error=error,exit=code,time=now(),process=process,source_identity='authorization/source_at_launch; immutable byte snapshots, no fingerprints'))
        budget=root/'budget.json'
        if budget.exists():
            with (root/'failure_ledger_snapshot.txt').open('x') as f:f.write(budget.read_text())
    manifest_path=root/'execution_manifest.json'
    if manifest_path.exists():
        try:m=read(manifest_path)
        except (OSError,ValueError):m=dict(stages={})
    else:m=dict(stages={})
    m.update(status='STOPPED',execution_complete=False,controller_exit=code,independent_review='PENDING',failure_source=str(first),failure_phase=phase)
    write(manifest_path,m)
    jobs=[]
    if (root/'budget.json').exists():
        try:jobs=read(root/'budget.json')['jobs']
        except (OSError,ValueError,KeyError):pass
    for kind,name in [('preflight','preflight_cpu_exit.json'),('cpu','cpu_exit.json')]:
        group=[j for j in jobs if j['kind']==kind];rc=None
        if group:
            exit_file=Path(group[0]['log'])/'exit.txt'
            if exit_file.exists():rc=int(exit_file.read_text())
            elif 'exit' in group[0]:rc=group[0]['exit']
        write(root/name,dict(attempts=len(group),exit=rc,status='NOT_RUN' if not group else 'PASSED' if rc==0 else 'FAILED' if rc is not None else 'INTERRUPTED'))
    complete=sum(j['kind']=='test' and j['exit']==0 for j in jobs)==175
    failure=dict(status='STOPPED',execution_complete=False,metrics_status='正式输出完整；执行失败，数值保留在formal单元' if complete else '未完成/不可比较',formal_test_complete=complete,
        first_failure=read(first),controller_exit=code,independent_review='PENDING',parent_failure_root='results/ua3d_anchor_lr_20261102/r1',parent_exit=1,parent_scientific_jobs=0,
        new_jobs=len(jobs),preflight_cpu=read(root/'preflight_cpu_exit.json'),final_cpu=read(root/'cpu_exit.json'),no_retry=True)
    write(root/'final_review/failure_status.json',failure)
    if not (root/'final_review/results.json').exists():write(root/'final_review/results.json',failure)
    text='# ANCHOR-LR r2 失败交付，待审查\n\nSTOPPED；'+failure['metrics_status']+'；independent_review=PENDING。\n首次错误阶段：'+read(first)['phase']+'。\n'+read(first)['error']+'\n\n真实exit由controller/outer/bootstrap各层wait文件记录；父r1 STOPPED/exit1/科学0保留。\nCPU：'+str(failure['preflight_cpu'])+' / '+str(failure['final_cpu'])+'。\n已派发'+str(len(jobs))+'个作业；科学预留/尝试/完成见budget.json，失败不退款。\n不调用完整refresh/report，不自动修复重试或补单元；BASE不替换。\n'
    if (root/'delivery_summary.md').exists():write(root/'final_review/delivery_before_failure.json',dict(text=(root/'delivery_summary.md').read_text()))
    (root/'delivery_summary.md').write_text(text)
    return code


def run_guarded(root,phase,runner):
    try:
        result=runner()
        if not isinstance(result,int) or isinstance(result,bool):raise TypeError('runner must return real integer exit')
        return result
    except BaseException:
        error=traceback.format_exc();record_failure(root,phase,error,1,identity());return 1


def run_wait(root,phase,runner):
    """The real Popen wait and the synthetic fake runner share this boundary."""
    rc=run_guarded(root,phase,runner)
    if rc!=0 and not (Path(root)/'first_failure.json').exists():record_failure(root,phase,'subprocess true wait exit='+str(rc),rc,identity())
    return rc


def cleanup_registered(root):
    """Only exact recorded timeout PID/start_ticks/argv/cwd/env/log-FDs; never NVML guesses."""
    budget=Path(root)/'budget.json';records=[]
    if not budget.exists():return records
    for j in read(budget)['jobs']:
        process=Path(j['log'])/'process.json'
        if not process.exists() or (Path(j['log'])/'exit.txt').exists():continue
        saved=read(process)['child'];base=Path('/proc')/str(saved['pid'])
        try:
            if identity(saved['pid'])!=saved:continue
            command=(base/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            if command!=j['argv'] or str((base/'cwd').resolve())!=j['cwd']:continue
            env=dict(x.split('=',1) for x in (base/'environ').read_bytes().decode().split('\0') if '=' in x)
            if any(env[k]!=j['env'][k] for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE')):continue
            if (base/'fd/1').resolve()!=Path(j['log'])/'stdout.log' or (base/'fd/2').resolve()!=Path(j['log'])/'stderr.log':continue
            os.killpg(saved['pid'],signal.SIGTERM);records.append(dict(process=saved,action='TERM owned exact-recorded timeout group'))
        except (OSError,ValueError,KeyError):continue
    cutoff=time.time()+35
    while time.time()<cutoff:
        pending=[]
        for j in read(budget)['jobs']:
            if (Path(j['log'])/'process.json').exists() and not (Path(j['log'])/'exit.txt').exists():pending.append(j)
        if not pending:break
        time.sleep(1)
    evidence=[]
    for j in read(budget)['jobs']:
        consumption=Path(j['consumption']);exitfile=Path(j['log'])/'exit.txt'
        item=dict(identity=j['identity'],reservation=j['reservation'],log=j['log'],actual_exit=int(exitfile.read_text()) if exitfile.exists() else None)
        if consumption.exists():
            try:item['consumption']=read(consumption)
            except (OSError,ValueError):item['raw_consumption']=consumption.read_text()
        evidence.append(item)
    write(Path(root)/'failure_cleanup.json',dict(time=now(),owned_only=records,unknown_pids_killed=[]))
    write(Path(root)/'failure_consumption_after_stop.json',dict(time=now(),jobs=evidence,accounting='append latest per-job evidence; first ledger snapshot retained, no full refresh/retry/refund'))
    return records


def status_failure_block(root):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();a='<!-- UA3D_ANCHOR_LR_R2_STATUS_START -->';b='<!-- UA3D_ANCHOR_LR_R2_STATUS_END -->'
    left,rest=text.split(a,1);block,right=rest.split(b,1)
    block+='\n- 失败追加：'+now()+'；STOPPED；首次异常见'+str(Path(root)/'first_failure.json')+'；三层真实exit见修复根；不修复重试，independent_review=PENDING。\n'
    p.write_text(left+a+block+b+right)


def wait_subprocess(argv,label):
    folder=LOG/label;folder.mkdir(exist_ok=False)
    environment={key:os.environ[key] for key in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','PYTHONDONTWRITEBYTECODE')}
    exclusive(folder/'command.json',dict(argv=argv,cwd=str(ROOT),env=environment,time=now(),parent=identity()))
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        child=subprocess.Popen(argv,cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err)
        exclusive(folder/'process.json',dict(child=identity(child.pid),parent=identity(),started_unix=time.time()))
        rc=child.wait()
    exclusive(folder/'exit.json',dict(exit=rc,ended_unix=time.time()));return rc



def finalize_actual_wait(root,log,bootstrap_rc,observer_identity):
    """Only observing parent calls this after bootstrap subprocess.wait returns."""
    root=Path(root);log=Path(log)
    exclusive(root/'bootstrap_exit.json',dict(exit=bootstrap_rc,source='observing tmux launcher actual subprocess.wait',time=now(),observer=observer_identity))
    exits={}
    for label,name in [('controller','controller_exit.json'),('outer','outer_launcher_exit.json'),('bootstrap','bootstrap_exit.json')]:
        exits[label]=read(root/name)['exit'] if (root/name).exists() else None
    exclusive(root/'actual_exit_chain.json',dict(exits=exits,time=now(),observer=observer_identity))
    result=root/'final_review/results.json'
    if result.exists():
        value=read(result);value['actual_exits']=exits
        if (root/'execution_manifest.json').exists():value['stage_facts']=read(root/'execution_manifest.json')['stages']
        if bootstrap_rc!=0:value.update(status='STOPPED',execution_complete=False)
        write(result,value)
    summary=root/'delivery_summary.md'
    if summary.exists():
        with summary.open('a') as f:f.write('\n实际wait退出链：'+json.dumps(exits,ensure_ascii=False)+'；independent_review=PENDING。\n')
    disk=root/'artifact_budget.json'
    if disk.exists():
        value=read(disk);value.update(actual_new_artifact_bytes=artifact_size(root,log),observed_unix=time.time());write(disk,value)
    return bootstrap_rc


def controller():
    phase='configuration_read'
    try:
        stage(OUT,phase);activation=read(OUT/'authorization/activation.json');spec=read(CONFIG);parent=read(OUT/'authorization/parent_r1/authorization/activation.json')
        validate_contract(activation,spec,parent)
        def deadline_signal(sig,frame):raise TimeoutError('inherited hard cutoff during import/construction/run')
        signal.signal(signal.SIGALRM,deadline_signal);signal.setitimer(signal.ITIMER_REAL,max(.001,activation['deadline_unix']-900-time.time()))
        phase='module_import';stage(OUT,phase)
        sys.path.insert(0,str(ROOT));from scripts.run_ua3d_anchor_lr_r2 import Controller
        phase='controller_construct';stage(OUT,phase);c=Controller()
        phase='controller_run';stage(OUT,phase);return c.run()
    except BaseException:
        error=traceback.format_exc();signal.setitimer(signal.ITIMER_REAL,0)
        last=OUT/'launch_phase.json'
        if last.exists():phase=read(last)['phase']
        record_failure(OUT,phase,error,1,identity())
        try:cleanup_registered(OUT)
        except BaseException:write(OUT/'cleanup_exception.json',dict(error=traceback.format_exc()))
        status_failure_block(OUT);return 1


def outer():
    rc=run_wait(OUT,'outer_wait',lambda:wait_subprocess([sys.executable,'-u',str(Path(__file__).resolve()),'controller'],'controller_process'))
    exclusive(OUT/'controller_exit.json',dict(exit=rc,source='outer real subprocess.wait',time=now()))
    return rc


def bootstrap():
    if not os.environ.get('TMUX') or os.environ['CUDA_VISIBLE_DEVICES']!='':raise ValueError('bootstrap requires hiddenGPU tmux')
    lock=(OUT/'bootstrap.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    activation=read(OUT/'authorization/activation.json')
    if time.time()>=activation['preparation_deadline_unix'] or time.time()+900>=activation['deadline_unix']:raise TimeoutError('inherited preparation/closing cutoff at bootstrap')
    exclusive(OUT/'authorization/bootstrap_start.json',dict(process=identity(),cwd=str(ROOT),argv=sys.argv,time=now()))
    rc=run_wait(OUT,'bootstrap_outer_wait',lambda:wait_subprocess([sys.executable,'-u',str(Path(__file__).resolve()),'outer'],'outer_process'))
    exclusive(OUT/'outer_launcher_exit.json',dict(exit=rc,source='bootstrap real subprocess.wait',time=now()))
    if rc!=0:status_failure_block(OUT)
    return rc


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('bootstrap','outer','controller'));args=p.parse_args()
    if args.mode=='controller':sys.exit(controller())
    if args.mode=='outer':sys.exit(outer())
    # Includes failures before imports/config/Controller; independent of the scientific refresh/report.
    try:exit_code=bootstrap()
    except BaseException:
        error=traceback.format_exc();record_failure(OUT,'bootstrap_setup',error,1,identity());exit_code=1
        write(OUT/'bootstrap_return_failure.json',dict(requested_exit=1,error=error,time=now()))
    sys.exit(exit_code)
