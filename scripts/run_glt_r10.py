"""r10 execution on physical GPUs 2/3; preserve every attempt and real exit."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import signal
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/glt_optimization_20261011/r10'
LOG = ROOT / 'logs/glt_optimization_20261011/r10'
PYTHON = '/opt/conda/bin/python'
GPUS = (2, 3)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def launch(cycle, name, argv, gpus=(), timeout=None, repair=None, extra_env=None):
    if any(g not in GPUS for g in gpus):
        raise ValueError('r10 permits only physical GPU2/3')
    folder = LOG / cycle
    if repair:
        folder /= repair
    folder /= name
    folder.mkdir(parents=True, exist_ok=False)
    if timeout:
        argv = ['timeout', str(timeout) + 's', *argv]
    environment = dict(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                       OMP_NUM_THREADS='16' if not gpus else '4',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
    environment.update(extra_env or {})
    metadata = dict(argv=argv, command=shlex.join(argv), cwd=str(ROOT),
                    environment=environment, gpus=list(gpus))
    write(folder / 'command.json', metadata)
    # The launcher owns stdout/stderr and a new process session, so stopping a
    # failed unit can terminate its torchrun children without touching others.
    code = '''import datetime,json,os,subprocess,sys,time
from pathlib import Path
f=Path(sys.argv[1]); j=json.loads((f/'command.json').read_text())
j.update(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RUNNING')
(f/'execution.json').write_text(json.dumps(j,indent=2)); start=time.monotonic()
with (f/'stdout.log').open('xb') as out,(f/'stderr.log').open('xb') as err:
 p=subprocess.Popen(j['argv'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),stdout=out,stderr=err,start_new_session=True)
 (f/'pid.txt').write_text(str(p.pid)); rc=p.wait()
j.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,exit_code=rc,status='PASS' if rc==0 else 'FAILED')
(f/'execution.json').write_text(json.dumps(j,indent=2)); (f/'exit.txt').write_text(str(rc));sys.exit(rc)
'''
    command = [PYTHON, '-c', code, str(folder)]
    script = folder / 'command.sh'
    script.write_text('cd ' + shlex.quote(str(ROOT)) + '\n' + shlex.join(command) + '\n')
    label = 'glt_r10_' + cycle + '_' + (repair + '_' if repair else '') + name
    subprocess.run(['tmux', 'new-window', '-d', '-t', 'Uni-Poly', '-n', label,
                    'bash ' + shlex.quote(str(script))], check=True)
    return folder


def wait(folder):
    while not (folder / 'exit.txt').exists():
        time.sleep(3)
    rc = int((folder / 'exit.txt').read_text())
    if rc:
        raise RuntimeError(str(folder) + ': exit ' + str(rc))
    return json.loads((folder / 'execution.json').read_text())


def available(gpu, minimum=40000):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    row = next([v.strip() for v in line.split(',')] for line in rows.splitlines()
               if int(line.split(',')[0]) == gpu)
    return int(row[2]) >= minimum and not any(
        line.split(',')[0].strip() == row[1] for line in apps.splitlines() if line.strip())


def e1_repair():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    repair = OUT / 'E1/repair1'
    repair.mkdir(parents=True, exist_ok=False)
    write(repair / 'execution.json', dict(status='执行中', reason='PyG fixture attribute collision fixed'))
    try:
        cpu = wait(launch('E1', 'final_cpu', [PYTHON, '-m', 'pytest', '-q',
                         'tests/test_glt_e1.py'], timeout=300, repair='repair1'))
        while not available(3):
            time.sleep(10)
        gpu = wait(launch('E1', 'final_gpu', [PYTHON, 'scripts/validate_glt_e1_gpu.py'],
                         (3,), timeout=300, repair='repair1'))
        result = dict(status='待审查', cpu=cpu, gpu=gpu,
                      independent_review='NOT_RUN',
                      consumption=json.loads((OUT / 'E1/gpu_consumption.json').read_text()))
        write(repair / 'execution.json', result)
        plan = ROOT / 'CURRENT_PLAN.md'
        text = plan.read_text().replace('|E1|跨注意力批量计算与 head-only 特征缓存|本次最终验证结果见§14.1|需返修|',
            '|E1|跨注意力批量计算与 head-only 特征缓存|修复复验及GPU有界对照见§15|待审查|')
        text += '\n\nE1 repair1执行完成：CPU／GPU真实exit均0，结果与累计预算分别见E1/repair1/execution.json和E1/gpu_comparison.json；旧失败保留，独立审查未执行。\n'
        plan.write_text(text)
    except BaseException as error:
        write(repair / 'failure.json', dict(error=repr(error), status='需返修'))
        raise


SOURCE_FILES=('src/modules/glt_r10.py','src/dataset/glt_r10.py','src/modules/glt_dual_pretrain.py',
    'src/modules/glt_dual.py','src/modules/glt_dual_opt.py','src/modules/glt_adaptation.py',
    'src/dataset/glt_dual.py','src/dataset/glt_dual_pretrain.py','src/training/glt_dual_runtime.py',
    'scripts/pretrain_glt_dual.py','scripts/evaluate_glt_pt_endpoint.py','scripts/finetune_mcl_ph.py',
    'scripts/run_glt_r10.py','scripts/run_glt_r10_unit.py','scripts/select_glt_r10.py','tests/test_glt_r10.py')


def mark_cycle(cycle,state,note):
    plan=ROOT/'CURRENT_PLAN.md';lines=plan.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('|'+cycle+'|') and len(line.split('|'))==6:
            parts=line.split('|');parts[-2]=state;lines[i]='|'.join(parts)
    lines+=['',f'{cycle}：{state}。{note}。实际预算见results/glt_optimization_20261011/r10/{cycle}/consumption.json。']
    plan.write_text('\n'.join(lines)+'\n')


def campaign():
    from src.training.glt_dual_runtime import require_tmux
    from src.modules.glt_r10 import PT_ARMS,FT_ARMS,TASKS
    from scripts.select_glt_r10 import consumption
    require_tmux()
    if not (OUT/'E1/gpu_comparison.json').exists() or json.loads((OUT/'E1/repair1/execution.json').read_text())['status']!='待审查':
        raise ValueError('E1 final verification incomplete')
    active=[];current=None
    def stopped(signum,frame):raise RuntimeError('controller interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def run(cycle,name,argv,gpus=(),**kwargs):
        folder=launch(cycle,name,argv,gpus,**kwargs);active.append(folder);return wait(folder)
    def accounting(cycle):
        value=consumption(cycle);write(OUT/cycle/'consumption.json',value)
        return value
    def matrix(cycle,formal):
        queue=iter((a,t,f) for a in FT_ARMS[cycle] for t in TASKS for f in range(5))
        running={};exhausted=False
        while running or not exhausted:
            for gpu,folder in list(running.items()):
                if (folder/'exit.txt').exists():wait(folder);del running[gpu]
            for gpu in GPUS:
                if gpu in running or exhausted or not available(gpu,12000):continue
                try:arm,task,fold=next(queue)
                except StopIteration:exhausted=True;break
                argv=[PYTHON,'scripts/run_glt_r10_unit.py','--cycle',cycle,'--arm',arm,'--task',task,'--fold',str(fold)]
                if formal:argv.append('--formal')
                folder=launch(cycle,('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold),argv,(gpu,))
                active.append(folder);running[gpu]=folder
            accounting(cycle)
            if running or not exhausted:time.sleep(5)
    try:
        for cycle in FT_ARMS:
            current=cycle
            if (OUT/cycle).exists():raise FileExistsError('nonempty/new cycle directory already exists: '+cycle)
            (OUT/cycle).mkdir()
            mark_cycle(cycle,'执行中','完整固定周期已启动；不做阶段测试')
            # Source copies describe the actual version used; no fingerprints.
            import shutil
            for name in SOURCE_FILES:
                destination=OUT/cycle/'source_used'/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,destination)
            run(cycle,'initialization',[PYTHON,'scripts/run_glt_r10_unit.py','--cycle',cycle,'--initialize'])
            for arm in PT_ARMS[cycle]:
                while not all(available(gpu) for gpu in GPUS):time.sleep(10)
                run(cycle,'pt_'+arm,[str(Path(PYTHON).parent/'torchrun'),'--standalone','--nproc_per_node=2',
                    'scripts/pretrain_glt_dual.py','--config',str(OUT/cycle/'initialization'/(arm+'_PT.json')),
                    '--cohort-root',str(ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'),
                    '--cache-root',str(ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'),
                    '--output',str(OUT/cycle/'pretrain'/arm),'--prep-workers','8','--lmdb-readahead'],GPUS)
            c=accounting(cycle);npt=len(PT_ARMS[cycle]);nft=35*len(FT_ARMS[cycle])
            if (c['pt_completed_updates'],c['pt_validation_completed'],c['pt_validation_presentations'])!=(5000*npt,3*npt,3072*npt):raise ValueError('PT budget/matrix incomplete')
            matrix(cycle,False)
            c=accounting(cycle)
            if (c['ft_completed'],c['ft_epochs'],c['ft_completed_updates'])!=(nft,70*nft,1600*nft):raise ValueError('FT matrix incomplete')
            run(cycle,'freeze',[PYTHON,'scripts/select_glt_r10.py','--cycle',cycle,'--action','freeze'])
            matrix(cycle,True)
            if accounting(cycle)['formal_completed']!=nft:raise ValueError('formal matrix incomplete')
            run(cycle,'summary',[PYTHON,'scripts/select_glt_r10.py','--cycle',cycle,'--action','summarize'])
            run(cycle,'final_cpu',[PYTHON,'-m','pytest','-q','tests/test_glt_r10.py'],timeout=300,
                extra_env={'GLT_R10_TEST_CYCLE':cycle})
            run(cycle,'handoff',[PYTHON,'scripts/select_glt_r10.py','--cycle',cycle,'--action','handoff'])
            mark_cycle(cycle,'待审查','正式矩阵与最终测试完成；独立审查未执行')
        all_results={c:accounting(c) for c in FT_ARMS}
        write(OUT/'campaign_execution.json',dict(status='待审查',cycles=all_results,independent_review='NOT_RUN',p2='undefined/budget0',end=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text().replace('状态：**执行中**（用户授权自行修复并使用物理GPU2、3完成全部已定义周期；从E1修复复验恢复，旧失败证据保留）。',
            '状态：**待审查**（全部已定义周期执行完成；独立审查未执行；P2未定义、预算0）。')
        plan.write_text(text)
    except BaseException as error:
        for folder in active:
            if not (folder/'exit.txt').exists() and (folder/'pid.txt').exists():
                try:os.killpg(int((folder/'pid.txt').read_text()),signal.SIGTERM)
                except ProcessLookupError:pass
        if current:
            value=accounting(current);write(OUT/current/'failure.json',dict(status='需返修',error=repr(error),traceback=traceback.format_exc(),consumption=value))
            mark_cycle(current,'需返修',repr(error)+'；现场保留，后续依赖暂停')
        write(OUT/'campaign_failure.json',dict(cycle=current,error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        import difflib
        changes=[]
        for name in SOURCE_FILES:
            path=OUT/'implementation_before'/name;before=path.read_text() if path.exists() else ''
            changes.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        with (OUT/'campaign_actual_changes.diff').open('x') as stream:stream.write(''.join(changes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--e1-repair', action='store_true')
    parser.add_argument('--campaign',action='store_true')
    args = parser.parse_args()
    if args.e1_repair:
        e1_repair()
    elif args.campaign:
        campaign()
    else:
        parser.error('select an execution mode')
