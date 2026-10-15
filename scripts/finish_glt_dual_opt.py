#!/usr/bin/env python3
"""Explicitly authorized final-stage repair continuation; never retrains/reselects."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_glt_dual_opt import OUT,LOG,ARMS,PLAN,read,write,launch,jobs,finish,consumption,now
from src.training.glt_dual_runtime import require_tmux

def main():
 require_tmux()
 units=read(OUT/'units.json'); selection=read(OUT/'selection_before_test.json')
 if len(units)!=210 or any(u['status']!='COMPLETED' for u in units):raise ValueError('development matrix incomplete')
 if selection['development_units']!=210:raise ValueError('selection incomplete')
 if (OUT/'formal').exists():raise FileExistsError('formal outputs already exist; no repeated inference')
 write(OUT/'execution_status.json',dict(plan=PLAN,status='RUNNING_FINAL_REPAIR',chosen=selection['chosen'],training='COMPLETED; no new updates',authorization='用户：请你解决以上问题并继续执行！',time=now()))
 try:
  log=launch('final_pytest_repair_01',['python','-m','pytest','-q','tests/test_glt_dual_opt.py'],[])
  while not (log/'exit.txt').exists():time.sleep(3)
  if int((log/'exit.txt').read_text()):raise RuntimeError('authorized repair final engineering test failed; retained; no automatic retry')
  for arm in ARMS:jobs(arm,True)
  finish()
  write(OUT/'consumption.json',consumption())
 except BaseException as error:
  write(OUT/'consumption.json',consumption())
  write(OUT/'execution_status.json',dict(plan=PLAN,status='BLOCKED',error=repr(error),traceback=traceback.format_exc(),time=now(),release_for_training=False,no_retry=True))
  raise
if __name__=='__main__':main()
