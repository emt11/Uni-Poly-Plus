#!/usr/bin/env python3
"""Exclusive stabilization root; independently registered48h clock,4h preparation."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/ua3d_cond_spectral_20261104/r2';P3=P/'repairs/memory_r3';P4=P/'repairs/lifecycle_r4';P5=P/'repairs/coverage_r5'
R=ROOT/'results/ua3d_base_stabilize_20261105/r1';L=ROOT/'logs/ua3d_base_stabilize_20261105/r1'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def identity(pid,role):
    parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(parts[19]),role=role)
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('dedicated Uni-Poly tmux window required')
    if '--outer' in sys.argv:
        roots=json.loads((R/'authorization/process_roots.json').read_text());roots.append(identity(os.getpid(),'outer'));write(R/'authorization/process_roots.json',roots)
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_base_stabilize_r1.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),started_unix=time.time(),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')}))
        with (L/'controller.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()));write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name stabilization scene/writer conflict; no overwrite/automatic revision change')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='dev' or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!='2992e3cd095ba636c0f120251c43ca435ae3ea89':raise ValueError('authorized branch/commit changed')
    current=(ROOT/'CURRENT_PLAN.md').read_text()
    if 'UA3D-BASE-STABILIZE-20261105-01' not in current or '<!-- BASE_STABILIZE_R1_STATUS_START -->' not in current:raise ValueError('current authorization version conflict')
    config=json.loads((ROOT/'configs/ua3d_base_stabilize_r1_gpu2.json').read_text());started=config['t0_unix']
    clock=dict(plan=config['plan'],revision='r1',resource_revision=config['resource_revision'],t0_unix=started,prepare_deadline_unix=config['prepare_deadline_unix'],deadline_unix=config['deadline_unix'],bootstrap_started_unix=time.time(),hard_hours=48,prepare_hours=4,pt_combined_hours=18,pt_single_hours=12,delivery_reserve_seconds=7200,independent_of_old_clocks=True)
    if clock['deadline_unix']!=started+48*3600 or clock['prepare_deadline_unix']!=started+4*3600:raise ValueError('new-cycle clock registration differs')
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(identity(os.getpid(),'bootstrap'),f)
    write(R/'authorization/clock.json',clock);write(R/'authorization/contract.json',config);write(R/'authorization/process_roots.json',[identity(os.getpid(),'bootstrap')])
    for name in ('CURRENT_PLAN.md','AGENTS.md','docs/research/BASE_STABILIZATION_EXECUTION_PLAN.md','docs/research/BASE_STABILIZATION_EXECUTION_PROMPT.md'):
        dest=R/'authorization'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/name).read_text())
    files=['configs/ua3d_base_stabilize_r1_gpu2.json',*[f'src/{x}/ua3d_base_stabilize_r1.py' for x in ('dataset','modules','training')],*[f'scripts/{x}_ua3d_base_stabilize_r1.py' for x in ('prepare','pretrain','run','bootstrap')],'tests/test_ua3d_base_stabilize_r1.py']
    for file in files:
        dest=R/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/file).read_text())
    dependencies=['src/modules/ua2d_galformer_pretrain.py','src/modules/ua3d_base_opt_r6.py','src/modules/ua3m_paperfuse.py','src/modules/ua3d_stability.py','src/dataset/ua3d_cond_spectral_r3.py','src/dataset/ua3m_paperfuse.py','src/training/ua3d_cond_spectral_r6.py']
    write(R/'authorization/locked_sources.json',dict(files=files,locked_unix=time.time(),source_snapshot=str(R/'source_snapshot'),hash_contract=False,read_only_dependencies=dependencies))
    for name in dependencies:
        dest=R/'source_text'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/name).read_text())
    write(R/'authorization/old_cycles.json',dict(conditional_spectral_source=str(P/'repairs/mm_lifecycle_r6/cumulative_budget.json'),conditional_spectral=json.loads((P/'repairs/mm_lifecycle_r6/cumulative_budget.json').read_text()),old_stability_budget_source='results/ua3d_stability_20261028/r1/budget.json',old_stability=json.loads((ROOT/'results/ua3d_stability_20261028/r1/budget.json').read_text()),independent_budget=True,old_model_tensors_loaded=0))
    (R/'recipe_implementation.md').write_text('New scientific cycle: restored source random factory, strict DET, same BASE operators/optimizer/loader RNG semantics, fixed four-point SWA on each normal trajectory. No old model tensors or graphics inputs. PT2/FT140/compare1+35/freeze1/TEST210/finalCPU1; no pre-suite/extra forwards/retries. Source-only dependencies immutable. Full contract copied in authorization.\n')
    import difflib
    with (R/'actual_diff.patch').open('x') as f:
        for file in files:f.writelines(difflib.unified_diff([], (ROOT/file).read_text().splitlines(True),fromfile='/dev/null',tofile=file))
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='20261028',CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    argv=[sys.executable,str(Path(__file__).resolve()),'--outer'];write(R/'authorization/bootstrap_command.json',dict(argv=argv,invocation_argv=[sys.executable,*sys.argv],cwd=str(ROOT),**identity(os.getpid(),'bootstrap'),started_unix=time.time(),t0_unix=started))
    with (L/'outer.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    write(R/'exits/bootstrap.json',dict(exit=code,pid=os.getpid(),outer_pid=p.pid,ended_unix=time.time()))
    if (R/'delivery.json').exists():
        value=json.loads((R/'delivery.json').read_text());value['real_exits']={k:json.loads((R/'exits'/f'{k}.json').read_text())['exit'] for k in ('controller','outer','bootstrap')};write(R/'delivery.json',value);write(R/'shutdown_audit.json',value)
        if (R/'failure_closeout.json').exists():write(R/'failure_closeout.json',value)
        with (R/'delivery_summary.md').open('a') as f:f.write('\n三层真实exit：'+str(value['real_exits'])+'；tmux实际exit见对应日志根tmux.exit。\n')
    return code
if __name__=='__main__':raise SystemExit(main())
