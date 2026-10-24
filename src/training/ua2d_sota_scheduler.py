"""Scheduling rules only; no scientific recipe changes or implicit retries."""
import os
import shutil
from pathlib import Path

GPUS=(0,1,2,3)
PT_PAIRS=((0,1),(2,3))
RESOURCE_REVISION='four_gpu_v1'

def validate_pt_mapping(world,visible,campaign,revision):
    allowed=('2,3',)
    if campaign=='r1_retest1' and revision==RESOURCE_REVISION: allowed=('0,1','2,3')
    if world!=2 or visible not in allowed: raise ValueError('PT requires two ranks on an authorized independent GPU pair')
    return tuple(int(g) for g in visible.split(','))

def classify_attempt(log,success_path,output_path=None):
    """A prior launch is never converted back into an unstarted task."""
    log=Path(log);success_path=Path(success_path)
    started=(log/'command.json').exists()
    if (log/'exit.txt').exists():
        code=int((log/'exit.txt').read_text())
        if code!=0: return 'interrupted' if code in (137,143,-9,-15) else 'failed'
        return 'success' if success_path.exists() else 'failed_missing_success'
    if started: return 'active'
    if success_path.exists() or (output_path is not None and Path(output_path).exists()): return 'unowned_output'
    return 'pending'

def verified_process(pid,argv,cwd,visible,*,birth=None,stdout=None,stderr=None):
    """Attest local identity without translating NVML host PIDs."""
    root=Path('/proc')/str(pid)
    try:
        actual=[p.decode() for p in (root/'cmdline').read_bytes().split(b'\0') if p]
        if len(actual)!=len(argv) or actual[1:]!=argv[1:] or Path(shutil.which(actual[0]) or actual[0]).resolve()!=Path(shutil.which(argv[0]) or argv[0]).resolve(): return None
        if (root/'cwd').resolve()!=Path(cwd).resolve(): return None
        stat=(root/'stat').read_text().rsplit(')',1)[1].split();ticks=int(stat[19])
        if birth is not None and ticks!=birth: return None
        env=dict(p.split(b'=',1) for p in (root/'environ').read_bytes().split(b'\0') if b'=' in p)
        if env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()!=visible: return None
        if env.get(b'UA2D_SOTA_CAMPAIGN',b'').decode()!='r1_retest1': return None
        for fd,target in ((1,stdout),(2,stderr)):
            if target is not None and (root/'fd'/str(fd)).resolve()!=Path(target).resolve(): return None
        return dict(pid=pid,start_ticks=ticks,argv=actual,cwd=str(cwd),visible=visible,state=stat[0])
    except (OSError,ValueError,UnicodeError): return None

def occupied_gpus(active):
    pt=set();ft={g:0 for g in GPUS};exclusive={g:0 for g in GPUS}
    for job in active:
        if job['kind']=='pt': pt.update(job['gpus'])
        elif job['gpus']:
            for gpu in job['gpus']:
                ft[gpu]+=1
                if not job.get('share_with_pt'): exclusive[gpu]+=1
    if any(ft[g]>2 for g in GPUS) or any(exclusive[g] for g in pt): raise RuntimeError('resource reservation conflict')
    return pt,ft

def available_pair(pair,active,contexts):
    pt,ft=occupied_gpus(active)
    return not any(g in pt or ft[g] or contexts[g] for g in pair)
