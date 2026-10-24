"""Publish current authorized execution identity and consumption in tmux."""
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_sota import *
from src.training.glt_dual_runtime import require_tmux
require_tmux()
if not (OUT/'failure_consumption.json').exists():
    write(OUT/'failure_consumption.json',dict(feature_units=1,feature_graphs=294,krr_fits=0,
        reason='F0 eea fold0 extraction completed; FP old table column mismatch before first KRR fit; no saved feature arrays',
        log=str(LOG/'F0/stderr.log'),exit=1,retry_authorized=False))
if not (OUT/'execution_manifest.json').exists():
    write(OUT/'execution_manifest.json',dict(plan='UA2D-SOTA-20261018-01',revision='r1',branch='dev',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),authorization='2026-10-18 user current message sections 3-12',
        preexisting=['M AGENTS.md','M CURRENT_PLAN.md','M TODO.md','D CHATGPT_HANDOFF.md','D CHATGPT_START_PROMPT.md','D FP_GRAPH_FUSION_DESIGN.md','D README.md','D src/result_contract.py'],
        tasks=TASKS,folds=list(range(5)),seed='42+fold',gpus=[2,3],max_ft_per_gpu=2,pt_ranks=2,
        input=dict(ua=str(UA),t0=str(T0),downstream=str(DOWN),split=str(SPLIT)),limits=LIMITS,
        status='执行中',start=now(),controller=str(LOG/'controller'),writer='single controller; isolated arm/task/fold units',
        outer_test='NOT_RUN',final_cpu='NOT_RUN',independent_review='NOT_RUN'))
refresh_budget()
