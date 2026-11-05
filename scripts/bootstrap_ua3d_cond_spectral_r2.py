#!/usr/bin/env python3
"""Exclusive-root bootstrap; preserve actual controller/outer/bootstrap exits."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
R=ROOT/'results/ua3d_cond_spectral_20261104/r2';L=ROOT/'logs/ua3d_cond_spectral_20261104/r2'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def main():
    if not os.environ.get('TMUX'):raise RuntimeError('must run in dedicated Uni-Poly tmux window')
    if '--outer' in sys.argv:
        argv=[sys.executable,str(ROOT/'scripts/run_ua3d_cond_spectral_r2.py')]
        write(R/'authorization/controller_command.json',dict(argv=argv,cwd=str(ROOT),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','OMP_NUM_THREADS')},started_unix=time.time()))
        with (L/'controller.log').open('x') as log:
            p=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
        write(R/'exits/controller.json',dict(exit=code,pid=p.pid,ended_unix=time.time()))
        write(R/'exits/outer.json',dict(exit=code,pid=os.getpid(),ended_unix=time.time()))
        return code
    if R.exists() or L.exists():raise RuntimeError('same-name r2 scene/writer conflict; no overwrite')
    R.mkdir(parents=True,exist_ok=False);L.mkdir(parents=True,exist_ok=False)
    with (R/'writer.lock').open('x') as f:json.dump(dict(pid=os.getpid(),started_unix=time.time(),automatic_resume=False),f)
    # t0 is immediately before the first and only controller launch chain.
    t0=time.time();write(R/'authorization/clock.json',dict(plan='UA3D-COND-SPECTRAL-20261104-01',revision='r2',resource_revision='gpu2-pt1-fttest4-fresh-condspec-r2',t0_unix=t0,prepare_deadline_unix=t0+14400,deadline_unix=t0+259200,engineering_target_unix=t0+86400))
    write(R/'authorization/contract.json',json.loads((ROOT/'configs/ua3d_cond_spectral_r2_gpu2.json').read_text()))
    (R/'implementation_diff.md').write_text('''# r2 implementation

新增8个隔离文件：configs/ua3d_cond_spectral_r2_gpu2.json、src/{dataset,modules,training}/ua3d_cond_spectral_r2.py、scripts/{prepare,run,bootstrap}_ua3d_cond_spectral_r2.py、tests/test_ua3d_cond_spectral_r2.py。
引用只读原BASE模型数学和原始数据接口；新随机step0及35头，无历史模型张量、旧z或教师缓存。
谱传播借鉴HMR/DiffusionNet；当前BASE上下文控制64通道热时间、中心/能量/Gram读出及六臂组合为本项目适配。具体公式和文献见docs/research/BASE_CONDITIONAL_SPECTRAL_PLAN.md r2。
FT优化器native joint1e-4、modal joint1e-3、surface3e-4、beta joint1e-4；head3e-4；weight_decay.02/clip5。PT AdamW2e-4/weight_decay0/clip1，两个连续5K周期，原三向ALIGN。
各单元parameters.json、history.json含实际参数量/优化器分组/时间/显存；jobs.json含真实argv/cwd/exit。
静态AST/接口检查允许；没有前置CPU suite、smoke、额外GPU诊断。仅完整正式TEST后一次最终pytest。
''')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    argv=[sys.executable,str(Path(__file__).resolve()),'--outer']
    write(R/'authorization/bootstrap_command.json',dict(argv=argv,cwd=str(ROOT),pid=os.getpid(),started_unix=t0))
    with (L/'outer.log').open('x') as log:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    write(R/'exits/bootstrap.json',dict(exit=code,pid=os.getpid(),outer_pid=p.pid,ended_unix=time.time()))
    # Fill only independently observed real waits; no fabricated worker success.
    if (R/'delivery.json').exists():
        value=json.loads((R/'delivery.json').read_text());value['real_exits']={k:json.loads((R/'exits'/f'{k}.json').read_text())['exit'] for k in ('controller','outer','bootstrap')};write(R/'delivery.json',value)
        if (R/'failure_closeout.json').exists():write(R/'failure_closeout.json',value)
        with (R/'delivery_summary.md').open('a') as f:f.write('\n三层真实exit：'+str(value['real_exits'])+'。\n')
    return code
if __name__=='__main__':raise SystemExit(main())
