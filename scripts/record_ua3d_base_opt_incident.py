#!/usr/bin/env python3
"""Record the pre-launch CPU-suite ordering deviation; never dispatch work."""
import datetime
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ua3d_base_opt_20261026/r5'
LOG=ROOT/'logs/ua3d_base_opt_20261026/r5'
PLAN='UA3D-BASE-OPT-20261026-01'
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    if not os.environ.get('TMUX') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('incident recording requires tmux and hidden GPUs')
    if OUT.exists() or LOG.exists():raise FileExistsError('r5 root already exists; stop without overwrite')
    OUT.mkdir(parents=True,exist_ok=False);LOG.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/ua3d_base_opt_r5.json').read_text())
    before=(ROOT/'CURRENT_PLAN.md').read_text()
    write(OUT/'config.json',config)
    (OUT/'authorization').mkdir()
    (OUT/'authorization/CURRENT_PLAN.before.md').write_text(before)
    write(OUT/'authorization/user_authorization.json',dict(plan=PLAN,revision='r5',stage='Stage0',
        source='user supplied fixed Stage0 execution prompt in this conversation',stages1_to5_authorized=False))
    command=['env','CUDA_VISIBLE_DEVICES=','python','-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_ua3d_base_opt.py','-k',
        'original_control or symmetric_direction or warmup or independent_projection or serial_budget']
    job=dict(name='pre_execution_cpu_contract_tests',kind='cpu',phase='before_execution',
        argv=command,cwd=str(ROOT),env={'CUDA_VISIBLE_DEVICES':''},pid=None,gpu_uuid=None,
        started=None,ended=None,exit=0,reported='5 passed, 1 deselected, 1 warning in 5.71s',
        wall_time_seconds=6.17715347,log=str(LOG/'pre_execution_cpu_contract_tests'),
        reservation={'cpu_suites':1},consumption=None,
        evidence='Observed tools.exec_command result; PID and exact wall-clock endpoints were not recorded. No fabricated values.')
    write(LOG/'pre_execution_cpu_contract_tests/command.json',job)
    (LOG/'pre_execution_cpu_contract_tests/stdout_observed.txt').write_text(
        '..... [100%]\n5 passed, 1 deselected, 1 warning in 5.71s\n'
        'Observed warning: torch_geometric.distributed has been deprecated since 2.7.0.\n'
        'This is an observed output summary, not a reconstructed complete terminal transcript.\n')
    (LOG/'pre_execution_cpu_contract_tests/exit.txt').write_text('0\n')
    actual={key:0 for key in config['limits']};actual['cpu_suites']=1
    incident=dict(code='EARLY_CPU_SUITE_ORDERING_DEVIATION',detected_at=NOW,
        requirement='最后仅一次 CPU suite / 唯一最终 CPU suite',
        actual='Before PT and before root creation, the assistant ran five CPU contract tests with pytest -k; exit 0.',
        impact='Required execution ordering was violated. Conservative accounting consumes the one CPU-suite allowance.',
        action='STOPPED before campaign launch; no additional test, retry, repair, resume, or GPU dispatch.',
        missing_process_evidence='Original CPU-test PID and exact start/end timestamps were not captured.',
        automatic_retry=False,automatic_resume=False)
    budget=dict(limits=config['limits'],charged_or_reserved=actual,known_actual_lower_bound=actual,
        jobs=[job],budget_violations=[],execution_order_violations=[incident['code']],
        final_cpu_suites=0,pre_execution_cpu_test_invocations=1,
        accounting='The early pytest invocation is conservatively charged as the sole CPU suite; no final suite may be added under this authorization.')
    write(OUT/'budget.json',budget);write(OUT/'failure.json',incident)
    write(OUT/'identity.json',dict(plan=PLAN,revision='r5',stage='Stage0',based_on=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch='dev',recorded_at=NOW,
        common_step0=None,pt_endpoints=None,controller_started=False,
        protected_dirty_at_start=['CURRENT_PLAN.md','TODO.md'],todo_read=False,remote_sync=False,
        hash_contract_added=False,shared_inputs_modified=False))
    write(OUT/'execution_manifest.json',dict(plan=PLAN,revision='r5',status='待审查',run_outcome='STOPPED_BEFORE_CAMPAIGN',
        independent_review='PENDING',controller_started=False,controller_exit=None,stage0_exit=None,
        cpu_test_true_exit=0,physical_gpu_authorized=2,gpu_tasks_started=0,
        cwd=str(ROOT),recording_pid=os.getpid(),recording_command=sys.argv,recorded_at=NOW,
        tmux_pane=os.environ.get('TMUX_PANE'),reason=incident['code']))
    write(OUT/'final_review/cpu_exit.json',dict(attempts=1,exit=0,scope='five pre-execution contract tests only',
        final_cpu_suite='NOT_RUN',ordering_compliant=False,log=job['log']))
    sota={task:dict(project=None,reference=ref,reference_model=model,delta=None,source='SOTA_RESULT.md',status='未完成／不可比较')
        for task,model,ref in [('Eea','Periodic-TDL',.934),('Egb','Periodic-TDL (+Chem, +RC)',.923),
            ('Egc','Periodic-TDL (+Chem, +RC)',.911),('Ei','Periodic-TDL',.839),('EPS','Periodic-TDL',.787),
            ('Nc','Periodic-TDL',.856),('Xc','Periodic-TDL',.423)]}
    result=dict(plan=PLAN,revision='r5',stage='Stage0',status='待审查',run_outcome='STOPPED_BEFORE_CAMPAIGN',
        independent_review='PENDING',reference=config['reference'],PT_DIRECTION_SELECTED=None,
        formal={},delta_macro=None,performance_pass=None,sota=sota,budget=budget,incident=incident,
        actual_tests=job,final_cpu_suite='NOT_RUN',
        missing=['common step0','PT histories/endpoints','FULL BEST/history/validation/scaler',
            'global selection/freeze','row+sample-key alignment','raw TEST predictions','final CPU suite'])
    write(OUT/'results.json',result);write(OUT/'final_review/results.json',result)
    report='''# UA3D BASE OPT r5 Stage0 停止记录

状态：待审查；run_outcome=STOPPED_BEFORE_CAMPAIGN；independent_review=PENDING。

执行偏差：助手在正式执行前运行了一次 pytest -k，5 passed、1 deselected，真实 exit 0。这违反“最后仅一次 CPU suite”的顺序要求；按保守口径计入唯一 CPU suite 额度。原测试命令未记录 PID 和精确起止时刻，证据字段保留 null，不补造。

已停止受影响链，未启动 controller、公共 step0 初始化、PT、FT、validation、冻结、TEST 或任何 GPU 任务；未再运行 CPU suite。预算：PT starts/updates/microbatches/presentations 全部 0；FULL starts/epochs/updates 全部 0；TEST units/rows 全部 0；提前 CPU 测试调用 1，最终 CPU suite 0。

已新增独立 base_opt 模型、PT 入口、controller、配置和测试。5 项测试仅覆盖损失与梯度一致性、双向正对计数、warmup/归一化、独立投影初态和预算/tie 规则；不能证明完整运行或 runtime 正确。实现未完成正式运行验证，不能视作验收通过。

当前 BASE reference 为用户指定的 0.795244745287872；本周期没有正式 TEST，Δmacro 未完成／不可比较。没有 PT_DIRECTION_SELECTED，不能用 validation 或历史结果填补。

| 任务 | 项目 R² | 论文参考 | 论文模型 | Δ |
|---|---:|---:|---|---:|
| Eea | 未完成／不可比较 | 0.934 | Periodic-TDL | — |
| Egb | 未完成／不可比较 | 0.923 | Periodic-TDL (+Chem, +RC) | — |
| Egc | 未完成／不可比较 | 0.911 | Periodic-TDL (+Chem, +RC) | — |
| Ei | 未完成／不可比较 | 0.839 | Periodic-TDL | — |
| EPS | 未完成／不可比较 | 0.787 | Periodic-TDL | — |
| Nc | 未完成／不可比较 | 0.856 | Periodic-TDL | — |
| Xc | 未完成／不可比较 | 0.423 | Periodic-TDL | — |

论文来源 SOTA_RESULT.md；逐任务最佳包络，不是一个可运行单模型。若完整执行，采用七任务五折、seed42+fold、mean±std(ddof=1)、七任务等权 macro；本次没有实验结果，不作公平性或超过论文声明。

证据：budget.json、identity.json、failure.json、execution_manifest.json、results.json、final_review/actual_changes.diff；日志根同周期 pre_execution_cpu_contract_tests 下保存完整命令、观察到的输出摘要和真实 exit。没有 campaign/controller exit，因为它们没有启动；不能用记录脚本的 exit 代替。

保留用户 CURRENT_PLAN.md/TODO.md dirty 状态、父 r4 代码、旧结果、失败现场、共享 cache/checkpoint 和历史账本。未读取 TODO.md，未远程同步，未新增 hash 合同。Stage1—5 未启动。

后续执行需要用户针对本次偏差、已消耗 CPU 测试额度和已存在 r5 根给出新的明确授权；本次不自动修复、覆盖、重试或恢复。
'''
    (OUT/'report.md').write_text(report);(OUT/'final_review/report.md').write_text(report)
    text=before
    for _ in range(2):text=text.replace('状态：**待授权**','状态：**待审查**',1)
    text=text.rstrip()+f'''

<!-- UA3D_BASE_OPT_R5_STATUS_START -->
- 更新时间：{NOW}；Stage0 状态：**待审查**；run_outcome=STOPPED_BEFORE_CAMPAIGN；independent_review=PENDING。
- 用户已明确授权 Stage0；Stage1—5 未授权。助手提前运行 5 项 CPU 合同测试，exit 0；违反“最后仅一次 CPU suite”的顺序，已停止，未启动 controller 或 GPU。
- 实际预算：PT/FULL/TEST 全部 0；提前 CPU 测试调用 1（保守计入唯一 suite），最终 CPU suite 0。原 CPU 测试 PID 和精确起止时刻未记录，保留缺项。
- 新增独立 base_opt 源码、配置、测试及本周期事故记录；正式 TEST 未完成／不可比较。没有候选选择或冻结，不替换 BASE。
- 证据：`results/ua3d_base_opt_20261026/r5/report.md`、`results.json`、`budget.json`、`failure.json`；日志：`logs/ua3d_base_opt_20261026/r5/`。
- 不自动 retry/resume/修复或再次运行 suite；既有 r5 根保留，后续执行需新的明确授权。
<!-- UA3D_BASE_OPT_R5_STATUS_END -->
'''
    (ROOT/'CURRENT_PLAN.md').write_text(text)
    write(OUT/'status.json',dict(status='待审查',run_outcome='STOPPED_BEFORE_CAMPAIGN',phase='执行偏差停止',
        independent_review='PENDING',time=NOW,reason=incident['code']))
    sources=['src/modules/ua3d_base_opt.py','src/training/ua3d_base_opt.py','scripts/pretrain_ua3d_base_opt.py',
        'scripts/run_ua3d_base_opt.py','scripts/record_ua3d_base_opt_incident.py',
        'configs/ua3d_base_opt_r5.json','tests/test_ua3d_base_opt.py']
    diff=[]
    for name in sources:diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    diff.extend(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
    (OUT/'final_review/actual_changes.diff').write_text(''.join(diff))
    write(LOG/'incident_recording.json',dict(cwd=str(ROOT),pid=os.getpid(),argv=sys.argv,
        command=os.environ.get('BASE_OPT_RECORDING_COMMAND'),tmux_pane=os.environ.get('TMUX_PANE'),
        gpu_uuid=None,started=NOW,ended=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        scope='record incident only; no training, evaluation, tests, or GPU task'))
    print('Recorded STOPPED_BEFORE_CAMPAIGN; CPU tests exit 0; all GPU/PT/FULL/TEST starts 0.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
