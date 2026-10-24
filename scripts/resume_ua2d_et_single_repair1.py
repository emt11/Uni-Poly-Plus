#!/usr/bin/env python3
"""Explicit user-authorized repair1; reuse complete preparation, retain attempt1 ledger."""
import os
os.environ['UA2D_ET_SINGLE_REPAIR']='repair1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import fcntl
import json
import shutil
import signal
import subprocess
import time
import traceback
from scripts.run_ua2d_et_single import Controller as OriginalController
from src.training.ua2d_et_single import *


def descendants(roots,parents):
    owned=set(roots)
    while True:
        extra={pid for pid,parent in parents.items() if parent in owned}-owned
        if not extra:return owned
        owned.update(extra)


class RepairController(OriginalController):
    def __init__(self):
        require_tmux()
        if read(OUT/'atomic_fields/metadata.json')['status']!='COMPLETE': raise RuntimeError('complete atomic preparation required')
        if (OUT/'repair1_execution_manifest.json').exists(): raise RuntimeError('repair1 already dispatched; no automatic retry')
        self.lock=(OUT/'controller.lock').open('r+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_budget=read(OUT/'budget.json')
        if any(j['exit'] is None for j in old_budget['jobs']): raise RuntimeError('prior launcher still active')
        if old_budget['used']['ft_starts'] or old_budget['used']['test_starts']: raise RuntimeError('unexpected prior downstream work')
        for rank in (0,1):
            c=read(OUT/f'pretrain/ET_SINGLE/rank{rank}_consumption.json')
            if c['completed_updates']!=0 or c['attempted_updates']!=1:raise RuntimeError('repair assumes zero completed PT updates')
        incident=OUT/'incidents/attempt1';incident.mkdir(parents=True,exist_ok=False)
        shutil.copytree(OUT/'final_review',incident/'final_review')
        for name in ('budget.json','status.json','execution_manifest.json','failure.json'):
            shutil.copy2(OUT/name,incident/name)
        folder=LOG/'controller_repair1';folder.mkdir(exist_ok=False)
        sys.stdout=(folder/'stdout.log').open('a',buffering=1);sys.stderr=(folder/'stderr.log').open('a',buffering=1)
        self.jobs=old_budget['jobs'];self.started=read(OUT/'wallclock_start.json')['monotonic']
        for j in self.jobs:
            if j['kind']=='pt':
                j['pt_output']=str(OUT/'pretrain/ET_SINGLE')
                j['presentation_reservation']=2352
                j['accounting_note']='747 persisted lower bound; rank1 counters stale on forced stop; reserve 2*(6 consumed-window+8 prefetched microbatches)*84, not an asserted actual count'
        self.window=subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],'#{window_id}'],text=True).strip()
        write(OUT/'repair1_execution_manifest.json',dict(plan=PLAN,revision=REVISION,repair='repair1',
            authorization=['请你执行解决并执行全部！','注意 3D 失败样本直接弃用！！！'],started=now(),
            reuse_atomic_rows=911391,policy='drop_failed_3d_entire_record',limits=LIMITS,
            root_cause=dict(source_row=346560,position=410,cached_positions=0,identity_count=206,geometry_valid=False,
                bug='unconditional coordinate/identity length comparison before geometry validity; legal no-geometry source representation'),
            original_wallclock_start=read(OUT/'wallclock_start.json'),additional_pt_starts=1,new_completed_pt_updates=5000,
            attempted_update_cumulative_limit=5001,prior_presentation_actual='unverified; persisted >=747, conservative reservation2352',
            additional_presentations_cap=5040000,additional_final_cpu_suites=1,cpu_seconds=300,
            automatic_retry=False,pt_output=str(PARENT.parent),cwd=str(ROOT),
            command=[sys.executable,str(Path(__file__).resolve())],tmux_window=self.window,tmux_session='Uni-Poly',pid=os.getpid(),
            ft_starts_limit=35,ft_updates_limit=56000,formal_test_gate='same V1 validation delta >=0.010',
            original_artifacts=str(incident),independent_review='PENDING'))
        update_plan('执行中','repair1实现／过滤',f'新用户指令授权修复恢复及弃用3D失败样本；controller {self.window}。复用911391条身份字段，原失败账本保留；仅新增一次过滤、一次PT尝试和一次修复最终suite。')
        self.refresh()

    def stop_owned(self):
        # Torchrun can leave ranks in different process groups: identify actual
        # rank PIDs and their descendants rather than trusting launcher exit.
        parents={int(a):int(b) for a,b in (line.split() for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())}
        roots=set()
        for job in self.jobs:
            if job['exit'] is not None and job.get('pt_output')!=str(PARENT.parent):continue
            candidates=[]
            pidfile=Path(job['log'])/'pid.txt'
            if pidfile.exists(): candidates.append(int(pidfile.read_text()))
            if job['kind']=='pt':
                for rank in (0,1):
                    p=Path(job.get('pt_output',str(PARENT.parent)))/f'rank{rank}_consumption.json'
                    if p.exists():candidates.append(read(p)['pid'])
            if job.get('consumption') and Path(job['consumption']).exists(): candidates.append(read(job['consumption'])['pid'])
            for pid in candidates:
                p=Path(f'/proc/{pid}/cmdline')
                if not p.exists():continue
                cmd=p.read_bytes().replace(b'\0',b' ').decode()
                if not cmd:continue
                if 'ua2d_et_single' not in cmd:raise RuntimeError('owned PID changed identity')
                roots.add(pid)
        owned=descendants(roots,parents)
        for pid in sorted(owned,reverse=True):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(3)
        for pid in sorted(owned,reverse=True):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        write(LOG/'controller_repair1/stopped_owned.json',dict(pids=sorted(owned),time=now(),method='verified rank PID and ancestry; TERM then KILL'))
        for _ in range(10):
            self.refresh(enforce=False)
            if all(j['exit'] is not None for j in self.jobs):break
            time.sleep(1)

    def run(self):
        error=None;cpu=None
        try:
            job=self.launch('et_filter_repair1','filter');self.wait([job])
            coverage=read(OUT/'sampling_repair1/metadata.json')
            update_plan('执行中','repair1 PT',f"有效采样池{coverage['valid_rows']}条，弃用{coverage['excluded_rows']}条3D失败记录；开启唯一修复PT尝试。")
            job=self.launch('et_pretrain_repair1','pt');self.wait([job])
            update_plan('执行中','FT','repair1 PT完成5000更新；开始原35FT矩阵，不重训历史参照。')
            self.matrix();decision=decide_and_freeze()
            if decision['passed']:
                update_plan('执行中','正式TEST',f"validation对V1差值{decision['delta']:+.3f}通过，35份best冻结；仅一次正式TEST。")
                self.matrix(formal=True)
            else:update_plan('执行中','负结果收尾',f"validation对V1差值{decision['delta']:+.3f}未达0.010；TEST0。")
        except BaseException:
            error=traceback.format_exc();write(OUT/'repair1_failure.json',dict(time=now(),error=error,automatic_retry=False))
            self.stop_owned()
        try:
            update_plan('执行中' if error is None else '需返修','repair1最终CPU','只运行一次修复针对性suite，不重放原6项suite；结束后交独立审查。')
            cpu=self.launch('et_cpu_repair1','cpu');self.wait([cpu])
        except BaseException:error=(error or '')+'\n'+traceback.format_exc()
        self.refresh(enforce=False);status='待审查' if error is None else '需返修'
        final_report(status,error)
        report_path=OUT/'final_review/results.json';result=read(report_path)
        result['repair1']=read(OUT/'repair1_execution_manifest.json')
        if (OUT/'sampling_repair1/metadata.json').exists():result['sampling']=read(OUT/'sampling_repair1/metadata.json')
        result['comparison_limit']='Failed-3D records excluded from entire ET PT sampling pool by latest user instruction; historical V1 retained them for 2D. Recipe/data differences cannot be attributed only to ET.'
        write(report_path,result)
        paths=['src/modules/ua2d_et_single.py','src/dataset/ua2d_et_single.py','src/training/ua2d_et_single.py',
            'scripts/prepare_ua2d_et_single.py','scripts/filter_ua2d_et_single.py','scripts/pretrain_ua2d_et_single.py',
            'scripts/run_ua2d_et_single.py','scripts/resume_ua2d_et_single_repair1.py','tests/test_ua2d_et_single.py','tests/test_ua2d_et_single_repair1.py']
        patches=[subprocess.run(['git','diff','--no-index','/dev/null',p],text=True,capture_output=True).stdout for p in paths]
        (OUT/'final_review/code_diff.patch').write_text('\n'.join(patches))
        update_plan(status,'repair1执行收尾',f'controller {self.window}；修复CPU exit={cpu["exit"] if cpu else "未启动"}。\n\n'+(OUT/'final_review/report.md').read_text())
        (OUT/'final_review/plan_diff.patch').write_text(subprocess.check_output(['git','diff','--','CURRENT_PLAN.md'],text=True))
        write(LOG/'controller_repair1/exit.json',dict(exit=0 if error is None else 1,time=now(),error=error))
        return 0 if error is None else 1

if __name__=='__main__':
    torch.set_num_threads(2)
    def terminate(signum,frame):raise RuntimeError(f'controlled termination {signum}; no retry')
    signal.signal(signal.SIGTERM,terminate)
    raise SystemExit(RepairController().run())
