"""Additive r5 delivery repair; read original evidence without rerunning science."""
import json
import datetime
import difflib
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/fce_tune_20261015/t3'
LOG=ROOT/'logs/fce_tune_20261015/t3'

def read(path):return json.loads(Path(path).read_text())

def execution_record(folder):
    """Keep the raw JSON and separately report the exit.txt evidence and unknown times."""
    folder=Path(folder);raw=read(folder/'execution.json')
    exit_path=folder/'exit.txt';exit_code=int(exit_path.read_text()) if exit_path.exists() else None
    conflicts=[]
    if exit_code is not None and raw.get('exit_code')!=exit_code:conflicts.append('raw JSON exit_code missing/differs from exit.txt')
    if exit_code is not None and raw.get('status')=='RUNNING':conflicts.append('raw status RUNNING but exit.txt exists')
    if not raw.get('end'):conflicts.append('raw end missing')
    if raw.get('start') and raw.get('end') and raw['start']>raw['end']:conflicts.append('raw start after end')
    time_verified=not conflicts and bool(raw.get('start') and raw.get('end'))
    return dict(unit=folder.name,log=str(folder),raw_execution=raw,raw_status=raw.get('status'),
        exit_txt=exit_code,exit_code=exit_code,exit_evidence=str(exit_path) if exit_path.exists() else None,
        start=raw.get('start') if time_verified else None,end=raw.get('end') if time_verified else None,
        time_status='原始完整JSON记录，未额外恢复' if time_verified else '未核实',
        metadata_conflicts=conflicts,raw_execution_path=str(folder/'execution.json'))


def exit_index(log_root,exclude=()):
    records=[execution_record(p.parent) for p in sorted(Path(log_root).glob('*/execution.json')) if p.parent.name not in exclude]
    events={2:[],3:[]}
    for r in records:
        gpu=r['raw_execution'].get('gpu')
        if gpu in (2,3) and r['start'] and r['end']:events[gpu].extend([(r['start'],1),(r['end'],-1)])
    peaks={}
    for gpu,ev in events.items():
        active=peak=0
        for _,d in sorted(ev):active+=d;peak=max(peak,active)
        peaks[str(gpu)]=peak
    snapshot_peaks={2:0,3:0}
    path=Path(log_root)/'controller/progress.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);running=row.get('running',[])
            if isinstance(running,list):
                for gpu in snapshot_peaks:snapshot_peaks[gpu]=max(snapshot_peaks[gpu],sum(x.get('gpu')==gpu for x in running))
    return dict(records=records,count=len(records),exit_counts={str(k):sum(r['exit_code']==k for r in records) for k in (0,1,None)},
        complete_interval_ft_formal_observed_peaks=peaks,controller_snapshot_observed_peaks=snapshot_peaks,
        limitations='缺失或矛盾时间未插补；区间与快照仅为观测峰值，不声称全部精确时间线齐全。PT为独立两rank作业，每卡一rank。')


def repair():
    review=OUT/'final_review'
    paths=[review/n for n in ('execution_exits.json','delivery_addendum_r5.md','r5_delivery_changes.diff')]
    if any(p.exists() for p in paths):raise FileExistsError('addendum already exists; original records must not be overwritten')
    repair_names=tuple(p.name for p in LOG.iterdir() if p.is_dir() and p.name.startswith('delivery_repair_'))
    index=exit_index(LOG,exclude=repair_names)
    index['previous_repair_attempts']=[execution_record(LOG/name) for name in repair_names if (LOG/name/'execution.json').exists()]
    for r in index['records']:
        if not r['metadata_conflicts']:continue
        if r['unit']=='ft_T3_REF_eea_4':
            folder=OUT/'finetune/T3_REF/eea/fold4';m=read(folder/'metrics.json');progress=read(folder/'progress.json')
            result=subprocess.run(['tmux','display-message','-p','-t','Uni-Poly:fce_t3_ft_T3_REF_eea_4','#{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
            r['completion_evidence']=dict(metrics_path=str(folder/'metrics.json'),metrics_status=m.get('status'),epochs=m.get('epochs'),updates=m.get('updates'),
                history_epochs=len(progress['history']),best_exists=(folder/'best.pt').is_file(),stdout_path=str(LOG/r['unit']/'stdout.log'),
                tmux_raw=result.stdout.strip(),tmux_query_exit=result.returncode,tmux_stderr=result.stderr,
                explanation='exit.txt=0及完整产物／日志／tmux支持科学单元已完成；raw JSON仍为RUNNING且start晚于原退出证据，原因未核实。start/end保持null，不伪造时间。')
    index.update(status='待审查',scope='T3a交付返修；科学结果沿用，不重跑',original_expected_records=147,original_expected_exit_counts={'0':145,'1':2},
        repair_scientific_updates=0,independent_review='NOT_PERFORMED',created=datetime.datetime.now(datetime.timezone.utc).isoformat())
    paths[0].write_text(json.dumps(index,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    old=read(OUT/'completion.json');text='T3a交付返修状态：**待审查（补交完成，科学结果沿用）**；独立审查未进行。\n\n'
    text+=f"退出索引覆盖原{index['count']}份记录，exit.txt统计{index['exit_counts']}。两份finalizer exit1、stderr及原始JSON全部保留，没有将helper失败改记成功。\n\n"
    text+='ft_T3_REF_eea_4 的原execution.json为RUNNING并缺失end/exit_code；exit.txt=0、70轮日志／完整metrics／best和tmux提供完成证据。原JSON start晚于原退出证据，回退／覆盖原因未核实，补交start/end均为null。并发区间和controller快照分别列出，缺失时间使完整精确时序不可恢复；不重跑该单元。\n\n'
    text+='本轮只新增execution_exits.json、delivery_addendum_r5.md和r5_delivery_changes.diff；保留原delivery、completion、diff和全部原始日志。原科学预算、唯一旧CPU检查沿用completion.json及原测试日志，未重新运行训练／预测／pytest或模型评估。\n\n'
    text+='原科学交付（原文保留，以下引用既有汇总）：\n\n'+(review/'delivery.md').read_text()
    paths[1].write_text(text)
    p=ROOT/'CURRENT_PLAN.md';before=p.read_text();after=before+'\n\n### 24.1 T3a补交执行记录\n\n授权来源：用户本轮明确授权第22节完整执行。T3a补交状态：**待审查（仅交付返修完成，科学结果沿用）**；原147份退出记录及两次finalizer exit1保留。单条FT metadata冲突和未核实时间已在execution_exits.json中分别记录，不改写原记录、不重跑。T3b状态：**执行中（实现／准备）**，科学预算尚未消费。\n'
    p.write_text(after)
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/CURRENT_PLAN.md',tofile='after/CURRENT_PLAN.md'))
    for name in ('scripts/finalize_fce_tune_t3.py',):diff+=''.join(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    paths[2].write_text(diff)
    print(json.dumps(dict(status='待审查',count=index['count'],exit_counts=index['exit_counts'],scientific_updates=0),ensure_ascii=False))

if __name__=='__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux();repair()
