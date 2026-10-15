"""Bounded r5: one reserved smoke, five validation fits, task-wise freezing."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp,write_shell,start_window,TASKS
from scripts.audit_mcl_ph_r6 import OUT,LOG,R4,SOURCE,SPLITS,PLAN,CONFIGS,PACKS,read,audit_unit
from src.training.glt_dual_runtime import require_tmux
STATISTICS=ROOT/'data/processed/mcl_ph_r4_20261001/statistics_train_full.npz'

def task_gate(task,candidate,reference):
    c,r=np.asarray(candidate,dtype=float),np.asarray(reference,dtype=float)
    assert c.shape==r.shape==((2,) if task=='xc' else (3,))
    assert np.isfinite(c).all() and np.isfinite(r).all() and (r>0).all()
    gains=(r-c)/r; mean=float(gains.mean()); positive=int((gains>0).sum())
    passed=(mean>=.02 and positive==2) if task=='xc' else (mean>=.01 and positive>=2 and float(gains.min())>=-.02)
    return dict(passed=bool(passed),gains=gains.tolist(),mean_relative_rmse_gain=mean,positive_folds=positive,
        primary_folds=[0,1] if task=='xc' else [2,3,4],selection_metric='validation_raw_rmse')

def unit_command(phase,variant,fold,*,selection=None,reuse=None):
    arm,task=('glt_ref','xc') if variant=='xc_standard' else ('m_xattn','nc')
    output=OUT/phase/variant;folder=output/arm/task/f'fold{fold}'
    cmd=[sys.executable,'scripts/finetune_mcl_ph.py','--arm',arm,'--stage',phase,
        '--config',str(CONFIGS[variant]),'--checkpoint',str(PACKS[variant]),
        '--expected-pretrain-step','5000','--cohort-root',str(SOURCE),'--cache-root',str(SOURCE),
        '--dual-static-root',str(SOURCE),'--split-root',str(SPLITS),
        '--cohort-index',str(SOURCE/f'{task}_offsets.npy'),'--statistics',str(STATISTICS),
        '--task',task,'--fold',str(fold),'--epochs','2' if phase=='smoke' else '70',
        '--output',str(output),'--experiment-plan',PLAN,'--target-transform','standard','--candidate-config',str(OUT/'candidate.json')]
    if selection:cmd+=['--frozen-selection',str(selection)]
    if reuse:cmd+=['--reuse-development-unit',str(reuse)]
    return cmd,folder,arm,task

class Campaign:
    def __init__(self):
        self.log=LOG/'campaign_01';self.log.mkdir(parents=True,exist_ok=False)
        self.units=[];self.state=dict(status='RUNNING',stage='PRECHECKS',plan=PLAN,started_at=timestamp(),
            max_complete_trainers=5,max_complete_epochs=350,max_complete_updates=4200,max_smoke_epochs=2,max_smoke_updates=24,
            standalone_probe_forwards=0,standalone_probe_backwards=0,new_pretrain_updates=0,units=[],running_units=[])
        self.publish()

    def publish(self,**fields):
        self.state.update(fields,updated_at=timestamp());self.state['units']=self.units
        atomic_json(self.log/'status.json',self.state)

    def batch(self,phase,requested,selection=None):
        queue=deque(requested);active={}
        while queue or active:
            while queue and len(active)<4:
                variant,fold,reuse=queue.popleft();gpu=next(g for g in range(4) if g not in active)
                cmd,folder,arm,task=unit_command(phase,variant,fold,selection=selection,reuse=reuse)
                assert not folder.exists()
                if reuse is None:
                    count=sum(not u['reused'] for u in self.units)+sum(not u['reused'] for u in active.values())
                    assert count<5, 'complete training budget exhausted'
                log=LOG/'units'/phase/variant/f'fold{fold}'
                write_shell(log,cmd,gpu=gpu);window=f'mcl_r6_{gpu}_{phase}_{task}_{fold}'
                start_window(window,log)
                active[gpu]=dict(phase=phase,variant=variant,fold=fold,arm=arm,task=task,
                    folder=str(folder),log=str(log),window=window,reused=reuse is not None,started_at=timestamp())
            self.publish(running_units=list(active.values()),pending_units=len(queue))
            for gpu,unit in list(active.items()):
                log=Path(unit['log'])
                if not (log/'exit.code').exists():continue
                assert (log/'exit.code').read_text().strip()=='0', 'unit failed: '+str(unit)
                m=audit_unit(Path(unit['folder']),unit['variant'],phase,unit['fold'])
                self.units.append(dict(unit,status='PASS',finished_at=timestamp(),epochs=m['executed_epochs'],
                    optimizer_updates=0 if unit['reused'] else m['optimizer_updates'],
                    validation_rmse=m['best_validation_rmse'],test_r2=m.get('test_r2')))
                assert sum(u['optimizer_updates'] for u in self.units if u['phase']!='smoke')<=4200
                del active[gpu];atomic_json(self.log/'units.json',self.units)
            time.sleep(3 if active else .1)
        self.publish(running_units=[],pending_units=0)

    def development(self,variant,fold):
        return Path(next(u['folder'] for u in self.units if u['phase']=='development' and u['variant']==variant and u['fold']==fold))

    def run(self):
        assert (LOG/'local_tests_01/exit.code').read_text().strip()=='0'
        assert (LOG/'gate_tests_01/exit.code').read_text().strip()=='0'
        self.publish(stage='SMOKE')
        self.batch('smoke',[('xc_standard',0,None)])
        smoke=self.units.pop();assert smoke['optimizer_updates']==24
        atomic_json(self.log/'smoke.json',smoke);atomic_json(self.log/'units.json',self.units)
        self.publish(stage='DEVELOPMENT',smoke_epochs=2,smoke_updates=24)
        self.batch('development',[('xc_standard',f,None) for f in (0,1)])
        reference=ROOT/'results/mcl_ph_r5_20261004'
        candidates=[read(self.development('xc_standard',f)/'metrics.json')['best_validation_rmse'] for f in (0,1)]
        refs=[read(reference/f'development/xc_standard/glt_ref/xc/fold{f}/metrics.json')['best_validation_rmse'] for f in (0,1)]
        gates={'xc':task_gate('xc',candidates,refs)}
        formal=[];contracts={}
        if gates['xc']['passed']:
            from types import SimpleNamespace
            from scripts.finetune_mcl_ph import training_contract
            for fold in range(5):
                reuse=self.development('xc_standard',fold) if fold<2 else None
                formal.append(('xc_standard',fold,reuse))
                args=SimpleNamespace(experiment_plan=PLAN,candidate_config=OUT/'candidate.json',fold=fold,config=CONFIGS['xc_standard'],split_root=SPLITS,cohort_root=SOURCE,checkpoint=PACKS['xc_standard'])
                contracts[f'glt_ref/xc/fold{fold}']=dict(experiment_plan=PLAN,target_transform='standard',
                    checkpoint=str(PACKS['xc_standard']),source_root=str(SOURCE),config=str(CONFIGS['xc_standard']),split_root=str(SPLITS),training_contract=training_contract(args))
        frozen=OUT/'frozen_selection.json'
        atomic_json(frozen,dict(status='FROZEN',plan=PLAN,revision='r1',frozen_at=timestamp(),gates=gates,
            allowed_units=list(contracts),unit_contracts=contracts,validation_only=True,
            final_routes=dict(xc='r6_xc_standard_lr1e4' if gates['xc']['passed'] else 'r5_xc_standard')))
        self.publish(stage='FORMAL' if formal else 'REJECTED',gates=gates,frozen_selection=str(frozen))
        self.batch('paper5_outer',formal,selection=frozen)
        self.aggregate(gates)
        self.publish(status='COMPLETED_PENDING_REVIEW',stage='A4',finished_at=timestamp(),
            result=str(OUT/'seven_task_results.json'),complete_trainers=sum(not u['reused'] for u in self.units),
            complete_epochs=sum(u['epochs'] for u in self.units if not u['reused']),
            reused_test_units=sum(u['reused'] for u in self.units))

    def aggregate(self,gates):
        old=read(ROOT/'results/mcl_ph_r5_20261004/seven_task_results.json');rows=[]
        for original in old['rows']:
            task=original['task'];passed=task in gates and gates[task]['passed'];paths=[];folds=[];rmses=[]
            for fold in range(5):
                if passed:
                    variant='xc_standard' if task=='xc' else 'nc_fixed_m_xattn'
                    arm='glt_ref' if task=='xc' else 'm_xattn';folder=OUT/f'paper5_outer/{variant}/{arm}/{task}/fold{fold}'
                else:folder=Path(original['prediction_paths'][fold]).parent
                m=read(folder/'metrics.json');folds.append(m['test_r2']);rmses.append(m['test_rmse']);paths.append(str(folder/'test_predictions.npz'))
            mean=float(np.mean(folds)); rows.append(dict(task=task,selected_arm=('xc_standard' if task=='xc' else 'nc_fixed_m_xattn') if passed else original['selected_arm'],
                selected_fold_r2=folds,selected_mean_r2=mean,selected_std_r2=float(np.std(folds)),
                selected_fold_rmse=rmses,r5_mean_r2=original['selected_mean_r2'],project_minus_r5=mean-original['selected_mean_r2'],
                paper_best_r2=original['paper_best_r2'],paper_model=original['paper_model'],
                project_minus_paper=mean-original['paper_best_r2'],prediction_paths=paths,
                result_cycle='r6_new' if passed else 'r5_accepted_reused'))
        report=dict(status='PASS',plan=PLAN,completed_at=timestamp(),rows=rows,gates=gates,
            mean7_r2=float(np.mean([r['selected_mean_r2'] for r in rows])),
            protocol='paper5_outer; five-fold arithmetic mean R2, seed42+fold, std ddof0',
            caveat='r4 test results were already used in development; not an independent blind test. Paper protocol equivalence unverified.',
            budgets=dict(new_pretrain_updates=0,new_complete_trainers=sum(not u['reused'] for u in self.units),
                new_complete_epochs=sum(u['epochs'] for u in self.units if not u['reused']),smoke_epochs=2,smoke_updates=24,
                probe_forwards=0,probe_backwards=0,probe_updates=0))
        atomic_json(OUT/'seven_task_results.json',report)

def main():
    require_tmux();campaign=Campaign()
    try:campaign.run()
    except BaseException as error:
        campaign.publish(status='STOPPED',error=f'{type(error).__name__}: {error}',automatic_retry=False,stopped_at=timestamp())
        for u in campaign.state.get('running_units',[]):
            subprocess.run(['tmux','send-keys','-t','Uni-Poly:'+u['window'],'C-c'],check=False)
        raise

if __name__=='__main__':main()
