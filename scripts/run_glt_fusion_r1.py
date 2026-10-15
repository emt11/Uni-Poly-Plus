#!/usr/bin/env python3
"""固定r1矩阵控制入口；复用已完成单元，不重试、不调参、不启动旧周期。"""
import datetime
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_fusion_20261007/r1';BASE_LOG=ROOT/'logs/glt_fusion_20261007/r1';LOG=BASE_LOG
CORE=('B0','T_RAW','T0','T0_A','K0','K_MD','K_MD_A','TK_MD','TK_MD_A','TK_MD_L')
TASKS=('eea','egb','egc','ei','eps','nc','xc')
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
MD_PI=OUT/'md200/pi1m';MD_DOWN=OUT/'md200/downstream'
MD_EXITS=[BASE_LOG/'md200_prepare.exit']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temporary,p)
def launch(name,command,gpus=(),timeout=None):
    folder=LOG/name;folder.mkdir(exist_ok=False)
    script=folder/'command.sh';cmd=shlex.join(command)
    (folder/'cwd.txt').write_text(str(ROOT)+'\n');(folder/'command.txt').write_text(cmd+'\n')
    if timeout:cmd=f'timeout --signal=TERM {timeout}s '+cmd
    script.write_text(f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str,gpus)))}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
date -Iseconds > {shlex.quote(str(folder/'start.txt'))}
{cmd} > {shlex.quote(str(folder/'stdout.log'))} 2> {shlex.quote(str(folder/'stderr.log'))}
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(folder/'exit.txt'))}
date -Iseconds > {shlex.quote(str(folder/'end.txt'))}
exit "$code"
''')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_fusion_r1_'+name,'bash '+shlex.quote(str(script))],check=True)
    return folder/'exit.txt'
def md_failure():
    for path in MD_EXITS:
        if path.exists() and int(path.read_text()):raise RuntimeError(f'MD200 preparation failed: {path}; preserve data and log, no automatic retry')
def completed_exit(name):
    paths=list(BASE_LOG.glob(f'**/{name}/exit.txt'))
    if len(paths)!=1:raise ValueError('completed unit requires exactly one original exit record: '+name)
    return paths[0]
def wait(exit_path):
    while not exit_path.exists():
        md_failure();write(OUT/'consumption.json',consumption());time.sleep(5)
    code=int(exit_path.read_text())
    if code:raise RuntimeError(f'{exit_path}: exit={code}; no retry')
def free_gpus():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={line.split(',')[0].strip() for line in result.splitlines() if line.strip()}
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    return [int(line.split(',')[0]) for line in devices.splitlines() if line.split(',')[1].strip() not in busy]
def pretrain(arm):
    cfg=read(OUT/'initialization'/f'{arm}.json')
    folder=OUT/'pretrain'/arm
    if folder.exists():
        exit_path=BASE_LOG/'pt_B0.exit' if arm=='B0' else completed_exit('pt_'+arm)
        if not exit_path.exists():raise ValueError('existing PT has no completed exit; no duplicate writer '+arm)
        wait(exit_path)
        if (read(folder/'run.json')['identity']['config'] != cfg
                or read(folder/'runtime.json')['status'] != 'PASS'
                or read(folder/'runtime.json')['completed_steps'] != 5000
                or read(folder/'consumption.json')['completed_updates'] != 5000
                or not (folder/'deploy_05000.pt').is_file()
                or not (folder/'resume_05000.pt').is_file()):
            raise ValueError('existing PT completion/config mismatch '+arm)
        print('reuse completed PT '+arm,flush=True)
    else:
        if cfg['use_md200']:
            for path in MD_EXITS:wait(path)
            if not MD_DOWN.exists():
                wait(launch('prepare_downstream',['python','scripts/prepare_glt_fusion_md200.py','--source',str(DOWN),'--output',str(MD_DOWN)]))
            if not read(MD_PI/'metadata.json')['complete'] or not read(MD_DOWN/'metadata.json')['complete']:raise ValueError('MD sidecar incomplete')
        while len(free_gpus())!=4:md_failure();time.sleep(5)
        exit_path=launch('pt_'+arm,['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_dual.py','--config',str(OUT/'initialization'/f'{arm}.json'),'--cohort-root',str(PI),'--cache-root',str(PI),'--dual-static-root',str(PI),'--output',str(OUT/'pretrain'/arm),'--prep-workers','8','--lmdb-readahead'],range(4))
    wait(exit_path)
    if read(OUT/'pretrain'/arm/'consumption.json')['completed_updates']!=5000:raise ValueError('PT final budget mismatch')
def finetune(arm,formal=False):
    pending=[];active={};failure=None
    cfg=read(OUT/'initialization'/f'{arm}.json')
    for task in TASKS:
        split=read(ROOT/f'data/splits/mips_paper5_outer/{task}.json')
        for fold in range(5):
            folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
            if folder.exists():
                name=('formal_' if formal else 'ft_')+arm+'_'+task+'_f'+str(fold)
                wait(completed_exit(name))
                metric=read(folder/'metrics.json')
                required=('test_predictions.npz',) if formal else ('best.pt','scaler.json','head_initial.pt','validation_predictions.npz')
                expected_epochs=0 if formal else (60 if task=='egc' else 70)
                expected_updates=expected_epochs*((len(split['folds'][fold]['train_indices'])+23)//24)
                if (metric['status']!='PASS' or metric['modes']!=cfg or metric['arm']!=arm
                        or metric['task']!=task or metric['fold']!=fold or metric['epochs']!=expected_epochs
                        or metric['updates']!={'attempted':expected_updates,'completed':expected_updates}
                        or any(not (folder/file).is_file() for file in required)):
                    raise ValueError('existing FT/formal completion/config mismatch '+name)
                print('reuse completed '+name,flush=True)
            else:pending.append((task,fold))
    while pending or active:
        md_failure()
        for gpu in free_gpus():
            if not pending or failure:break
            if gpu in active:continue
            task,fold=pending.pop(0);name=('formal_' if formal else 'ft_')+arm+'_'+task+'_f'+str(fold)
            command=['python','scripts/run_glt_fusion_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
            if formal:command.append('--formal')
            if formal and arm=='T0_A' and not (OUT/'selection_before_test.json').exists() and (OUT/'selection_before_T0_A_test.json').exists():
                command.append('--early-t0-a-test')
            active[gpu]=(name,launch(name,command,[gpu]))
        for gpu,(name,path) in list(active.items()):
            if path.exists():
                code=int(path.read_text());print(f'{name} exit={code}',flush=True);del active[gpu]
                if code:failure=f'{name} exit={code}'
        write(OUT/'consumption.json',consumption())
        if failure and not active:raise RuntimeError(failure+'; no retry')
        time.sleep(5)
def consumption():
    pt_starts=int((BASE_LOG/'pt_B0.log').exists());ft_starts=formal_starts=formal_completed=0
    for start in BASE_LOG.rglob('start.txt'):
        folder=start.parent
        if folder.is_dir():
            if folder.name.startswith('pt_'):pt_starts+=1
            elif folder.name.startswith('ft_'):ft_starts+=1
            elif folder.name.startswith('formal_'):
                formal_starts+=1
                if (folder/'exit.txt').exists() and int((folder/'exit.txt').read_text())==0:formal_completed+=1
    pt_attempted=pt_completed=ft_attempted=ft_completed=epochs=0
    for path in (OUT/'pretrain').glob('*/consumption.json'):
        data=read(path);pt_attempted+=data['attempted_updates'];pt_completed+=data['completed_updates']
    for folder in (OUT/'finetune').glob('*/*/fold*'):
        path=next((folder/name for name in ('metrics.json','failure.json','progress.json') if (folder/name).exists()),None)
        if path:
            data=read(path);ft_attempted+=data['updates']['attempted'];ft_completed+=data['updates']['completed'];epochs+=data.get('epochs',data.get('completed_epochs',len(data.get('history',[]))))
    data=dict(pt_starts=pt_starts,pt_attempted_updates=pt_attempted,pt_completed_updates=pt_completed,ft_starts=ft_starts,ft_attempted_updates=ft_attempted,ft_completed_updates=ft_completed,ft_completed_epochs=epochs,science_starts=pt_starts+ft_starts,total_completed_updates=pt_completed+ft_completed,formal_starts=formal_starts,formal_completed=formal_completed,engineering_test_calls=sum(path.parent.name=='final' for path in BASE_LOG.rglob('start.txt')),
              previous_data_preparation=dict(starts=1,exit=1,completed_rows=67313,wall_seconds=2177),
              data_preparation_runs=[dict(log=str(path.parent),exit=int((path.parent/'exit.txt').read_text()) if (path.parent/'exit.txt').exists() else None) for path in BASE_LOG.rglob('start.txt') if path.parent.name in ('prepare_pi1m','prepare_downstream')])
    if pt_starts>12 or pt_attempted>60000 or ft_starts>420 or ft_attempted>617400 or epochs>28800 or formal_starts>420:raise RuntimeError('fixed experiment budget exceeded')
    return data
def main():
    global LOG,MD_PI,MD_DOWN,MD_EXITS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-after-repair',action='store_true')
    parser.add_argument('--continue-after-mapping-fix',action='store_true')
    parser.add_argument('--continue-after-ledger-fix',action='store_true')
    args=parser.parse_args()
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    status=OUT/'execution_status.json'
    if sum((args.continue_after_repair,args.continue_after_mapping_fix,args.continue_after_ledger_fix))>1:raise ValueError('choose one authorized continuation')
    if args.continue_after_repair or args.continue_after_mapping_fix or args.continue_after_ledger_fix:
        repair='repair_03' if args.continue_after_ledger_fix else 'repair_02' if args.continue_after_mapping_fix else 'repair_01'
        LOG=BASE_LOG/repair
        MD_PI=OUT/'md200/pi1m_repair_01';MD_DOWN=OUT/('md200/downstream_repair_02' if args.continue_after_ledger_fix else 'md200/downstream_'+repair)
        status=OUT/repair/'execution_status.json'
        previous=OUT/'repair_02/execution_status.json' if args.continue_after_ledger_fix else OUT/'repair_01/execution_status.json' if args.continue_after_mapping_fix else OUT/'execution_status.json'
        if read(previous)['status'] not in ('阻断','BLOCKED'):raise ValueError('repair continuation requires stopped prior controller')
        MD_EXITS=[BASE_LOG/'repair_01/prepare_pi1m/exit.txt',(BASE_LOG/'repair_02' if args.continue_after_ledger_fix else LOG)/'prepare_downstream/exit.txt']
    if status.exists():raise FileExistsError('controller already recorded; do not restart')
    write(status,dict(status='执行中',plan='GLT-FUSION-20261007-01/r1',started=now()))
    try:
        if args.continue_after_ledger_fix:
            for path in MD_EXITS:wait(path)
            if not read(MD_PI/'metadata.json')['complete'] or not read(MD_DOWN/'metadata.json')['complete']:
                raise ValueError('existing completed MD sidecars missing')
        if args.continue_after_repair or args.continue_after_mapping_fix:
            if args.continue_after_repair:
                launch('prepare_pi1m',['python','scripts/prepare_glt_fusion_md200.py','--source',str(PI),'--output',str(MD_PI),'--reuse-partial',str(OUT/'md200/pi1m'),'--workers','8'])
            elif not (BASE_LOG/'repair_01/prepare_pi1m/start.txt').exists() or not MD_PI.exists():
                raise ValueError('existing PI1M writer/sidecar missing')
            launch('prepare_downstream',['python','scripts/prepare_glt_fusion_md200.py','--source',str(DOWN),'--output',str(MD_DOWN),'--workers','8'])
            for path in MD_EXITS:wait(path)
            metadata=read(MD_PI/'metadata.json')
            if not metadata['complete'] or not read(MD_DOWN/'metadata.json')['complete']:raise ValueError('repaired MD sidecars incomplete')
            metadata['downstream_sidecar']=str(MD_DOWN)
            write(MD_PI/'metadata.json',metadata)
        for arm in CORE:
            print('core '+arm,flush=True);pretrain(arm);finetune(arm)
        wait(launch('prepare_core_selection',['python','scripts/select_glt_fusion_r1.py','core']))
        wait(launch('prepare_statistics',['python','scripts/glt_fusion_geometry_statistics.py']))
        wait(launch('prepare_objectives',['python','scripts/select_glt_fusion_r1.py','objectives']))
        for arm in ('G_NORM','G_FGR'):pretrain(arm);finetune(arm)
        wait(launch('prepare_selection',['python','scripts/select_glt_fusion_r1.py','final']))
        wait(launch('final',['python','-m','pytest','-q','tests/test_dual_glt_pretrain.py','tests/test_glt_dual_diagnostics.py','tests/test_glt_dual_static.py','tests/test_glt_dual_opt.py','tests/test_glt_fusion_r1.py','--junitxml='+str(LOG/'final/junit.xml')],timeout=300))
        for arm in (*CORE,'G_NORM','G_FGR'):finetune(arm,formal=True)
        wait(launch('prepare_summary',['python','scripts/select_glt_fusion_r1.py','summarize']))
        write(status,dict(status='待审查',completed=now(),consumption=consumption(),independent_review='未开始'))
    except BaseException as error:
        write(status,dict(status='阻断',time=now(),error=repr(error),traceback=traceback.format_exc(),no_retry=True))
        write(OUT/'consumption.json',consumption());raise
if __name__=='__main__':main()
