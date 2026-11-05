#!/usr/bin/env python3
"""r3 exclusive bootstrap: inherit clocks; actual wait exits; CPU-only controller."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/ua3d_cond_spectral_20261104/r2'
R=P/'repairs/memory_r3';L=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/memory_r3'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)
def identity(pid,role):
    parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(parts[19]),role=role)
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('dedicated Uni-Poly tmux window required')
    if '--outer' in sys.argv:
        roots=json.loads((R/'authorization/process_roots.json').read_text());roots.append(identity(os.getpid(),'outer'));write(R/'authorization/process_roots.json',roots)
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_cond_spectral_r3.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')},started_unix=time.time()))
        with (L/'controller.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()));write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name r3 scene/writer conflict; no overwrite')
    parent=json.loads((P/'authorization/clock.json').read_text());pt=json.loads((P/'pretrain/BASE_FRESH/clock.json').read_text())
    clock=dict(parent,revision='r3',science_recipe='r2',resource_revision='gpu2-pt1-fttest4-pss-boundedio-r3',parent_clock_source=str(P/'authorization/clock.json'),parent_pt_clock_source=str(P/'pretrain/BASE_FRESH/clock.json'),pt_deadline_unix=pt['deadline_unix'],parent_pt_started_unix=pt['started_unix'],repair_started_unix=time.time())
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(identity(os.getpid(),'bootstrap'),f)
    write(R/'authorization/clock.json',clock);write(R/'authorization/process_roots.json',[identity(os.getpid(),'bootstrap')])
    config=json.loads((ROOT/'configs/ua3d_cond_spectral_r3_gpu2.json').read_text());write(R/'authorization/contract.json',config)
    files=['configs/ua3d_cond_spectral_r3_gpu2.json',*[f'src/{folder}/ua3d_cond_spectral_r3.py' for folder in ('dataset','modules','training')],*[f'scripts/{name}_ua3d_cond_spectral_r3.py' for name in ('prepare','run','bootstrap')],'tests/test_ua3d_cond_spectral_r3.py']
    for file in files:
        target=R/'source_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x') as f:f.write((ROOT/file).read_text())
    write(R/'authorization/locked_sources.json',dict(files=files,source_snapshot=str(R/'source_snapshot'),locked_unix=time.time(),hash_contract=False))
    (R/'repair_diff.md').write_text('''# r3 repair and scientific invariance

仅新增登记的八个r3文件；r2文件、旧模型和父证据只读。src/modules复制科学r2定义，仅将schema/config provenance显式转为r3、science_recipe=r2，无模型算子或参数初态数学改变。
新随机seed42 step0、35套新随机头，完整10K新PT；不加载父/历史模型张量。资格和stat16仅元数据只读alias，不重做全数据资格或拟合。
PT从8 worker/prefetch2改为4 spawn worker/prefetch1；LRU128条且唯一CPU storage≤256MiB，readonly LMDB readahead=False，事务buffers=False返回bytes，事务外解码；每worker64源microbatch后close再open自身environment，采样器/映射/LRU/计数保留。
PSS+SwapPss主门禁160GiB，RSS仅诊断；登记PID+start_ticks，包括bootstrap/outer/controller/本轮后代；每5秒轻量及15秒/launch/exit PSS，决定样本先flush落盘后停止。共享cgroup只作环境观察，MemAvailable32GiB/cgroup有限余量8GiB为环境门禁。
继承父JSON完整浮点准备/PT/总截止，不重起时钟。parent858次尝试与[857,858]完成区间、未知预取尾部保留，累计尝试上限346858。
正常前5148源microbatch对照父完整身份前缀，无额外batch/前向。FT/validation/TEST数学与读取配置不变。所有科学与工程最终CPU检查仅正式TEST完成后一次。
逐进程样本：memory.jsonl/memory_latest.json；正常IO缓存与重开统计：PT history/sampling；分组/实际参数：FT history/parameters.json；任务运行命令、PID与真实exit：jobs.json及exits/。
''')
    if any(time.time()>=clock[key] for key in ('prepare_deadline_unix','pt_deadline_unix','deadline_unix')):
        value=dict(status='EXPIRED',reason='inherited applicable deadline elapsed before controller',clock=clock,PT=0,FT=0,TEST=0,CPU=0,independent_review='PENDING',formal_status='未完成/不可比较')
        write(R/'failure_closeout.json',value);(R/'delivery_summary.md').write_text(json.dumps(value,ensure_ascii=False,indent=2));return 1
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    argv=[sys.executable,str(Path(__file__).resolve()),'--outer'];write(R/'authorization/bootstrap_command.json',dict(argv=argv,cwd=str(ROOT),**identity(os.getpid(),'bootstrap'),started_unix=clock['repair_started_unix']))
    with (L/'outer.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    write(R/'exits/bootstrap.json',dict(exit=code,pid=os.getpid(),outer_pid=p.pid,ended_unix=time.time()))
    if (R/'delivery.json').exists():
        value=json.loads((R/'delivery.json').read_text());value['real_exits']={k:json.loads((R/'exits'/f'{k}.json').read_text())['exit'] for k in ('controller','outer','bootstrap')};write(R/'delivery.json',value)
        write(R/'shutdown_audit.json',value)
        if (R/'failure_closeout.json').exists():write(R/'failure_closeout.json',value)
        with (R/'delivery_summary.md').open('a') as f:f.write('\n三层真实exit：'+str(value['real_exits'])+'。\n')
    return code
if __name__=='__main__':raise SystemExit(main())
