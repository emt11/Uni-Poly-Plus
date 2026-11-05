#!/usr/bin/env python3
"""Exclusive r4 root; pre-registered two-hour repair window, original total clock."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/ua3d_cond_spectral_20261104/r2';P3=P/'repairs/memory_r3'
R=P/'repairs/lifecycle_r4';L=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def identity(pid,role):
    parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(parts[19]),role=role)
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('dedicated Uni-Poly tmux window required')
    if '--outer' in sys.argv:
        roots=json.loads((R/'authorization/process_roots.json').read_text());roots.append(identity(os.getpid(),'outer'));write(R/'authorization/process_roots.json',roots)
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_cond_spectral_r4.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),started_unix=time.time(),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')}))
        with (L/'controller.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()));write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name r4 scene/writer conflict; no overwrite/automatic revision change')
    config=json.loads((ROOT/'configs/ua3d_cond_spectral_r4_gpu2.json').read_text());parent=json.loads((P3/'authorization/clock.json').read_text())
    started=config['repair_started_unix'];prepare_deadline=min(started+7200,parent['deadline_unix'])
    if config['repair_prepare_deadline_unix']!=prepare_deadline:raise ValueError('pre-registered repair window differs')
    clock=dict(plan=config['plan'],revision='r4',execution_revision='r4',science_recipe='r2',resource_revision=config['resource_revision'],t0_unix=parent['t0_unix'],deadline_unix=parent['deadline_unix'],engineering_target_unix=parent['engineering_target_unix'],parent_prepare_deadline_unix=parent['prepare_deadline_unix'],parent_pt_deadline_unix=parent['pt_deadline_unix'],parent_r3_clock=parent,repair_started_unix=started,repair_prepare_deadline_unix=prepare_deadline,bootstrap_started_unix=time.time(),new_PT=0)
    if parent['deadline_unix']!=1794048655.5459058:raise ValueError('original deadline mismatch')
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(identity(os.getpid(),'bootstrap'),f)
    write(R/'authorization/clock.json',clock);write(R/'authorization/contract.json',config);write(R/'authorization/process_roots.json',[identity(os.getpid(),'bootstrap')])
    files=['configs/ua3d_cond_spectral_r4_gpu2.json',*[f'src/{x}/ua3d_cond_spectral_r4.py' for x in ('dataset','modules','training')],*[f'scripts/{x}_ua3d_cond_spectral_r4.py' for x in ('prepare','run','bootstrap')],'tests/test_ua3d_cond_spectral_r4.py']
    for file in files:
        dest=R/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/file).read_text())
    write(R/'authorization/locked_sources.json',dict(files=files,locked_unix=time.time(),source_snapshot=str(R/'source_snapshot'),hash_contract=False,read_only_scientific_dependencies=['src/modules/ua3d_cond_spectral_r3.py','src/dataset/ua3d_cond_spectral_r3.py']))
    (R/'repair_diff.md').write_text('''# r4 lifecycle repair and scientific invariance

八个隔离文件见authorization/locked_sources.json及source_snapshot。父r2/r3源码、快照、账本、PT、头和失败现场均只读。
modules/dataset显式导入原科学算子、Model、optimizer_ft、Data/loader；不改数学、初始化顺序、seed、batch24、FP32、head10/joint60、dropout、损失或shuffle；不修改旧模块全局变量。pretrain/head-r3保留原schema与CONFIG，通过显式白名单正常FT初始化严格加载；仅新BEST schema为best-r4。PT入口、FreshPT初始化和随机头生成均不存在。
按PID+start_ticks和可选pidfd登记；绑定pidfd前后核对身份、终态关闭FD。活动集合与追加终态历史分开。直接worker poll/wait先落盘，再采资源；非直接后代不调用waitpid、不伪造exit0。
PSS读取前后复核，ESRCH/ENOENT仅同次采样最多3读/250ms；明确Z/X/x、身份消失/复用或pidfd退出事件才归并终态。持续存活不可测保留错误停止。主合计仅含末次确认存活集合，退出历史值另存；样本非原子，不宣称连续精确峰值。
保持160GiB PSS+SwapPss、MemAvailable32GiB、有限cgroup余量8GiB门禁。逐次identity/errno/读取结果及决定先flush到memory.jsonl/latest，首份breach/error独占保存，再停止。
阶段时钟使用原总截止和实施开始时已登记的r4两小时准备窗口；父prepare/PT截止仅历史封存，不阻拦FT，不刷新72h/12h。
父11155尝试/[11154,11155]区间、中断297更新、预取未知尾部和PT配额保留；新增PT0、FT336000，满额累计347155/[347154,347155]。
先六臂EPS/fold0正常完整FT计入210，再按匹配task/fold历史峰值、本轮观察及每worker1GiB上下文余量准入。及时补空位，实测wall/并发估ETA，不默认4倍加速。无前置pytest/smoke/额外模型前向；唯一CPU suite只能完整正式TEST后运行。
''')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    argv=[sys.executable,str(Path(__file__).resolve()),'--outer'];write(R/'authorization/bootstrap_command.json',dict(argv=argv,cwd=str(ROOT),**identity(os.getpid(),'bootstrap'),started_unix=time.time(),repair_started_unix=started))
    with (L/'outer.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    write(R/'exits/bootstrap.json',dict(exit=code,pid=os.getpid(),outer_pid=p.pid,ended_unix=time.time()))
    if (R/'delivery.json').exists():
        value=json.loads((R/'delivery.json').read_text());value['real_exits']={k:json.loads((R/'exits'/f'{k}.json').read_text())['exit'] for k in ('controller','outer','bootstrap')};write(R/'delivery.json',value);write(R/'shutdown_audit.json',value)
        if (R/'failure_closeout.json').exists():write(R/'failure_closeout.json',value)
        with (R/'delivery_summary.md').open('a') as f:f.write('\n三层真实exit：'+str(value['real_exits'])+'；tmux实际exit见对应日志根tmux.exit。\n')
    return code
if __name__=='__main__':raise SystemExit(main())
