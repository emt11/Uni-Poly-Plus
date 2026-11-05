#!/usr/bin/env python3
"""Exclusive r6 root; pre-registered two-hour repair window, original total clock."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/ua3d_cond_spectral_20261104/r2';P3=P/'repairs/memory_r3';P4=P/'repairs/lifecycle_r4';P5=P/'repairs/coverage_r5'
R=P/'repairs/mm_lifecycle_r6';L=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def identity(pid,role):
    parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(parts[19]),role=role)
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('dedicated Uni-Poly tmux window required')
    if '--outer' in sys.argv:
        roots=json.loads((R/'authorization/process_roots.json').read_text());roots.append(identity(os.getpid(),'outer'));write(R/'authorization/process_roots.json',roots)
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_cond_spectral_r6.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),started_unix=time.time(),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')}))
        with (L/'controller.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()));write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name r6 scene/writer conflict; no overwrite/automatic revision change')
    config=json.loads((ROOT/'configs/ua3d_cond_spectral_r6_gpu2.json').read_text());parent=json.loads((P5/'authorization/clock.json').read_text())
    started=config['repair_started_unix'];prepare_deadline=min(started+7200,parent['deadline_unix'])
    if config['repair_prepare_deadline_unix']!=prepare_deadline:raise ValueError('pre-registered repair window differs')
    clock=dict(plan=config['plan'],revision='r6',execution_revision='r6',science_recipe='r2',resource_revision=config['resource_revision'],t0_unix=parent['t0_unix'],deadline_unix=parent['deadline_unix'],engineering_target_unix=parent['engineering_target_unix'],parent_prepare_deadline_unix=parent['parent_prepare_deadline_unix'],parent_pt_deadline_unix=parent['parent_pt_deadline_unix'],parent_r5_clock=parent,repair_started_unix=started,repair_prepare_deadline_unix=prepare_deadline,bootstrap_started_unix=time.time(),new_PT=0)
    if parent['deadline_unix']!=1794048655.5459058:raise ValueError('original deadline mismatch')
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(identity(os.getpid(),'bootstrap'),f)
    write(R/'authorization/clock.json',clock);write(R/'authorization/contract.json',config);write(R/'authorization/process_roots.json',[identity(os.getpid(),'bootstrap')])
    for name in ('CURRENT_PLAN.md','AGENTS.md','docs/research/BASE_CONDITIONAL_SPECTRAL_R6_EXECUTION_PLAN.md','docs/research/BASE_CONDITIONAL_SPECTRAL_R6_EXECUTION_PROMPT.md'):
        dest=R/'authorization'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/name).read_text())
    files=['configs/ua3d_cond_spectral_r6_gpu2.json',*[f'src/{x}/ua3d_cond_spectral_r6.py' for x in ('dataset','modules','training')],*[f'scripts/{x}_ua3d_cond_spectral_r6.py' for x in ('prepare','run','bootstrap')],'tests/test_ua3d_cond_spectral_r6.py']
    for file in files:
        dest=R/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/file).read_text())
    write(R/'authorization/locked_sources.json',dict(files=files,locked_unix=time.time(),source_snapshot=str(R/'source_snapshot'),hash_contract=False,read_only_scientific_dependencies=['src/modules/ua3d_cond_spectral_r3.py','src/dataset/ua3d_cond_spectral_r3.py','src/training/ua3d_cond_spectral_r4.py','src/training/ua3d_cond_spectral_r5.py']))
    (R/'repair_diff.md').write_text('# r6 memory-context classification and controlled continuation\n\nOnly 8 isolated files; parent code/roots/constants remain read-only. Same r3 science, RNG/data/batch/FP32/head10+joint60, fork4/prefetch2. NO_USER_MM_OBSERVED requires two complete evidence rounds in the existing 3-read/250ms ESRCH branch; null PSS, separate immutable 30s exit wait, no admission or phase transition, actual exit only. Fixed-cohort/pending5s and160GiB/environment gates retained. 198 complete aliases, only12 fresh full FT/15190 updates; cumulative354471 attempts/[354470,354471]. PT/initialization/statistics/geometry0. Original clock and all parent unknown tails retained. One freeze then210 new formal TEST units then sole CPU suite.\n')
    import difflib
    with (R/'implementation.diff').open('x') as f:
        for file in files:
            old=ROOT/file.replace('_r6','_r5')
            f.writelines(difflib.unified_diff(old.read_text().splitlines(True),(ROOT/file).read_text().splitlines(True),fromfile=str(old),tofile=file))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    argv=[sys.executable,str(Path(__file__).resolve()),'--outer'];write(R/'authorization/bootstrap_command.json',dict(argv=argv,invocation_argv=[sys.executable,*sys.argv],cwd=str(ROOT),**identity(os.getpid(),'bootstrap'),started_unix=time.time(),repair_started_unix=started))
    with (L/'outer.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    write(R/'exits/bootstrap.json',dict(exit=code,pid=os.getpid(),outer_pid=p.pid,ended_unix=time.time()))
    if (R/'delivery.json').exists():
        value=json.loads((R/'delivery.json').read_text());value['real_exits']={k:json.loads((R/'exits'/f'{k}.json').read_text())['exit'] for k in ('controller','outer','bootstrap')};write(R/'delivery.json',value);write(R/'shutdown_audit.json',value)
        if (R/'failure_closeout.json').exists():write(R/'failure_closeout.json',value)
        with (R/'delivery_summary.md').open('a') as f:f.write('\n三层真实exit：'+str(value['real_exits'])+'；tmux实际exit见对应日志根tmux.exit。\n')
    return code
if __name__=='__main__':raise SystemExit(main())
