"""Attest local CUDA workers without inventing NVML host-PID translations."""
import json
from pathlib import Path


def attest_worker(job, proc_root=Path('/proc')):
    """Fail closed on command, cwd, visibility, start-tick, lineage or log-FD mismatch.

    A positive result identifies the local worker only. It never asserts that an
    unmatched host PID belongs to that worker.
    """
    try:
        usage=json.loads(Path(job['consumption']).read_text())
        environment=json.loads((Path(job['consumption']).parent/'environment.json').read_text())
        saved=usage['process']; pid=saved['pid']; base=Path(proc_root)/str(pid)
        stat=(base/'stat').read_text(); fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z' or int(fields[19])!=saved['start_ticks']:
            return dict(attested=False,reason='dead/reused local PID',local_pid=pid)
        command=(base/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        argv=job['argv']; start=next(i for i,x in enumerate(argv) if Path(x).name in ('python','python3','python3.11','python3.12'))
        expected=argv[start:]
        if command!=expected:
            return dict(attested=False,reason='argv mismatch',local_pid=pid)
        if str((base/'cwd').resolve())!=job['cwd']:
            return dict(attested=False,reason='cwd mismatch',local_pid=pid)
        env=dict(line.split('=',1) for line in (base/'environ').read_bytes().decode().split('\0') if '=' in line)
        for key in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','UA3M_PHYSICAL_GPU_UUID'):
            if env.get(key)!=job['env'].get(key):
                return dict(attested=False,reason='environment mismatch: '+key,local_pid=pid)
        if env.get('CUDA_VISIBLE_DEVICES')!='2' or env.get('WORLD_SIZE')!='1':
            return dict(attested=False,reason='wrong physical device contract',local_pid=pid)
        if environment['physical']['actual_uuid']!=job['env']['UA3M_PHYSICAL_GPU_UUID']:
            return dict(attested=False,reason='runtime UUID mismatch',local_pid=pid)
        for fd,name in (('1','stdout.log'),('2','stderr.log')):
            if (base/'fd'/fd).resolve()!=Path(job['log'])/name:
                return dict(attested=False,reason='log FD mismatch',local_pid=pid)
        wrapper=json.loads((Path(job['log'])/'process.json').read_text())
        ancestor=wrapper['child']['pid']; pending=[ancestor]; descendants=set()
        while pending:
            current=pending.pop()
            if current in descendants:continue
            descendants.add(current)
            children=Path(proc_root)/str(current)/'task'/str(current)/'children'
            if children.exists():pending.extend(int(x) for x in children.read_text().split())
        if pid not in descendants:
            return dict(attested=False,reason='not a launched wrapper descendant',local_pid=pid)
        return dict(attested=True,local_pid=pid,start_ticks=saved['start_ticks'],argv_verified=True,cwd_verified=True,environment_verified=True,logs_verified=True,lineage_verified=True,host_pid_mapping='UNKNOWN')
    except (OSError,ValueError,KeyError,StopIteration,UnicodeError) as exc:
        return dict(attested=False,reason=type(exc).__name__+': '+str(exc))
