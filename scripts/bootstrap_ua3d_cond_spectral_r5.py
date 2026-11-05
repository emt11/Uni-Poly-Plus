#!/usr/bin/env python3
"""Exclusive r5 root; pre-registered two-hour repair window, original total clock."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
P=ROOT/'results/ua3d_cond_spectral_20261104/r2';P3=P/'repairs/memory_r3';P4=P/'repairs/lifecycle_r4'
R=P/'repairs/coverage_r5';L=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def identity(pid,role):
    parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(parts[19]),role=role)
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('dedicated Uni-Poly tmux window required')
    if '--outer' in sys.argv:
        roots=json.loads((R/'authorization/process_roots.json').read_text());roots.append(identity(os.getpid(),'outer'));write(R/'authorization/process_roots.json',roots)
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_cond_spectral_r5.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),started_unix=time.time(),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')}))
        with (L/'controller.log').open('x') as log:p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()));write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name r5 scene/writer conflict; no overwrite/automatic revision change')
    config=json.loads((ROOT/'configs/ua3d_cond_spectral_r5_gpu2.json').read_text());parent=json.loads((P4/'authorization/clock.json').read_text())
    started=config['repair_started_unix'];prepare_deadline=min(started+7200,parent['deadline_unix'])
    if config['repair_prepare_deadline_unix']!=prepare_deadline:raise ValueError('pre-registered repair window differs')
    clock=dict(plan=config['plan'],revision='r5',execution_revision='r5',science_recipe='r2',resource_revision=config['resource_revision'],t0_unix=parent['t0_unix'],deadline_unix=parent['deadline_unix'],engineering_target_unix=parent['engineering_target_unix'],parent_prepare_deadline_unix=parent['parent_prepare_deadline_unix'],parent_pt_deadline_unix=parent['parent_pt_deadline_unix'],parent_r4_clock=parent,repair_started_unix=started,repair_prepare_deadline_unix=prepare_deadline,bootstrap_started_unix=time.time(),new_PT=0)
    if parent['deadline_unix']!=1794048655.5459058:raise ValueError('original deadline mismatch')
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(identity(os.getpid(),'bootstrap'),f)
    write(R/'authorization/clock.json',clock);write(R/'authorization/contract.json',config);write(R/'authorization/process_roots.json',[identity(os.getpid(),'bootstrap')])
    files=['configs/ua3d_cond_spectral_r5_gpu2.json',*[f'src/{x}/ua3d_cond_spectral_r5.py' for x in ('dataset','modules','training')],*[f'scripts/{x}_ua3d_cond_spectral_r5.py' for x in ('prepare','run','bootstrap')],'tests/test_ua3d_cond_spectral_r5.py']
    for file in files:
        dest=R/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as f:f.write((ROOT/file).read_text())
    write(R/'authorization/locked_sources.json',dict(files=files,locked_unix=time.time(),source_snapshot=str(R/'source_snapshot'),hash_contract=False,read_only_scientific_dependencies=['src/modules/ua3d_cond_spectral_r3.py','src/dataset/ua3d_cond_spectral_r3.py','src/training/ua3d_cond_spectral_r4.py']))
    (R/'repair_diff.md').write_text('# r5 fixed-cohort coverage and canonical alias repair\n\n只新增8个r5文件，锁定文本见source_snapshot。r2/r3/r4源码、结果、账本和停止证据只读；不修改旧OUT/REVISION/PARENT。\n模型、数据、optimizer数学引用原科学模块；batch24、FP32、seed、shuffle、4 fork workers/prefetch2/persistent_workers=False均不改。PT/新初始化/统计拟合/几何/构面/求谱新增0。\n唯一完整C_FRESH/EPS/fold0按完成身份alias父r4，BEST epoch32固定且不重选、不复制模型；209套新FT完整从r3 PT/原随机头开始，禁止中断BEST接续。registry/canonical.json精确210条，estimator_config与runner_config分开；新best-r5、唯一父best-r4严格白名单；正式输出全部在r5/formal。\n每PSS pass固定有限C；尾部新发现身份进入first_seen+5s不可刷新截止的EDF待测队列。PENDING不等于通过，pss_total_all_active=null、不刷新last_full、不准入新GPU且不写首错。每≤250ms安排服务并记录实际延迟，持续实际读失败保留3读/250ms语义直接STOP，不能放回队列豁免。跨5s窗口重新实测固定C，不把旧值当新值。\n保留160GiB PSS+SwapPss和环境门禁、直接Popen先poll/wait、PID身份与终态归并；非直接后代不回收，Python pidfd API不可用只走严格stat，不引入依赖或硬编码syscall。采样/待测/服务/退出/超时/是否阻止派发均先flush落盘，首份STOP独占保存。\n父累计12288尝试/[12287,12288]完成区间、两次中断297+363和预取未知保留，父完整770不重收费；新增335230，满额347518尝试/[347517,347518]完成区间。\n采用实施开始已登记的r5两小时准备窗口，父r4窗口和原总deadline完整保留，不刷新72h。无前置测试/额外前向；完整210正式TEST后唯一CPU suite。\n')
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
