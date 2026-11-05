"""Sole final CPU suite, after all 140 formal TEST units. Never invoked as preflight."""
import copy
import math
import os
from pathlib import Path
import pytest
import torch
from torch import nn
from src.training import ua3d_ft_balance as t
from scripts import run_ua3d_ft_balance as c


def test_epoch_schedule_boundaries_and_original_policy():
    for epoch in range(1,71):
        assert t.m.align_coefficient('constant',epoch)==.08
        x=t.m.align_coefficient('joint_late_decay',epoch)
        assert math.isclose(x,.08 if epoch<=40 else .08-.06*(epoch-40)/30,abs_tol=1e-12)
    assert t.m.align_coefficient('joint_late_decay',40)==.08
    assert math.isclose(t.m.align_coefficient('joint_late_decay',41),.078,abs_tol=1e-12)
    assert math.isclose(t.m.align_coefficient('joint_late_decay',70),.02,abs_tol=1e-12)
    assert t.m.align_coefficient('joint_late_decay',51)<t.m.align_coefficient('joint_late_decay',50)
    with pytest.raises(ValueError):t.m.align_coefficient('constant',0)
    assert t.m.POLICIES==t.SPEC['recipes']


class SyntheticModel(nn.Module):
    """Tiny parameter containers for optimizer grouping; no scientific model forward."""
    def __init__(self,policy):
        super().__init__();self.lr_policy=policy;self.ft=copy.deepcopy(t.m.DEFAULT_FT);self.stage='head'
        for name in t.m.NATIVE:
            layer=nn.Linear(2,2)
            if name=='encoder3d':layer.register_parameter('mask_token',nn.Parameter(torch.zeros(2)))
            setattr(self,name,layer)
        for name in (*t.m.MODAL,*t.m.TASK_MODULES):setattr(self,name,nn.Linear(2,2))


def test_optimizer_coverage_frozen_mask_and_scheduler():
    for policy in ('original','modal_low'):
        model=SyntheticModel(policy)
        for stage,total in (('head',10),('joint',60)):
            opt,groups=t.m.optimizer_ft(model,stage)
            assert len(groups)==(1 if stage=='head' else 3 if policy=='modal_low' else 2)
            assert model.encoder3d.mask_token.requires_grad is False
            assert all(p.requires_grad==(stage=='joint') for n,p in model.named_parameters() if n.split('.')[0] in t.m.NATIVE and n!='encoder3d.mask_token')
            sched=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
            for epoch in range(total):
                expect=t.m.expected_learning_rates(stage,epoch,policy)
                assert len(expect)==len(opt.param_groups)
                assert all(math.isclose(g['lr'],x,rel_tol=1e-10,abs_tol=1e-12) for g,x in zip(opt.param_groups,expect))
                t.m.optimizer_snapshot(model,opt)
                opt.step();sched.step()  # no gradients, inputs, forward, or actual model training


def test_fixed_selection_guard_ties_and_base_keep():
    frozen=t.read(t.OUT/'selection/final_frozen.json');chosen,decision=t.select(frozen['validation'])
    assert (chosen,decision)==(frozen['FINAL'],frozen['selection'])
    v={a:dict(macro=.5,tasks={task:dict(mean=.5) for task in t.TASKS}) for a in t.ORDER}
    assert t.select(v)[0]=='BASE_KEEP'
    v['BASE_REFT_HISTORY']['macro']=1.0
    v['A_ALIGN_DECAY']['macro']=.505;v['B_MODAL_LR']['macro']=.505
    assert t.select(v)[0]=='A_ALIGN_DECAY'
    v['A_ALIGN_DECAY']['tasks']['eps']['mean']=.49
    assert t.select(v)[0]=='B_MODAL_LR'
    v['C0_BASE']['macro']=.51
    assert t.select(v)[0]=='BASE_KEEP'
    v['B_MODAL_LR']['macro']=.515;v['C0_BASE']['tasks']['nc']['mean']=.51
    assert t.select(v)[0]=='BASE_KEEP'
    assert t.read(t.OUT/'selection/validation_selection.json')['FINAL']==chosen


def test_140_units_aliases_single_test_order_and_exits():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    b=t.read(t.OUT/'budget.json');f=t.read(t.OUT/'selection/final_frozen.json');jobs=b['jobs']
    assert f['frozen'] and len(f['units'])==210 and f['new_estimators']==140 and f['alias_estimators']==70
    assert len(t.registry()['aliases'])==70
    assert sum(j['kind']=='register' for j in jobs)==1 and sum(j['kind']=='cpu' for j in jobs)==1
    assert not any(j['arm'] in t.ALIASES for j in jobs)
    for kind in ('ft','test'):
        selected=[j for j in jobs if j['kind']==kind]
        assert len(selected)==140 and all(j['exit']==0 for j in selected)
        assert {(j['arm'],j['task'],j['fold']) for j in selected}=={(a,task,fold) for a in t.ARMS for task in t.TASKS for fold in range(5)}
    assert len({(j['kind'],j['arm'],j['task'],j['fold']) for j in jobs})==len(jobs)
    assert max(j['ended'] for j in jobs if j['kind']=='ft')<f['time']<min(j['reserved_at'] for j in jobs if j['kind']=='test')
    assert max(j['ended'] for j in jobs if j['kind']=='test')<next(j['reserved_at'] for j in jobs if j['kind']=='cpu')
    for j in jobs:
        if j['kind']=='cpu':continue
        assert int((Path(j['log'])/'exit.txt').read_text())==j['exit']==t.read(Path(j['log'])/'end.json')['exit']==0
    acceptance=t.read(t.OUT/'registry/source_acceptance.json')
    assert acceptance['model_constructed'] is False and acceptance['checkpoint_loaded'] is False


def test_trajectories_best_config_checkpoint_and_eval():
    frozen=t.read(t.OUT/'selection/final_frozen.json')
    for arm in t.ARMS:
        for task in t.TASKS:
            for fold in range(5):
                path=t.unit_path(arm,task,fold);cfg=t.read(path/'config.json');hist=t.read(path/'history.json');met=t.read(path/'metrics.json')
                assert cfg==t.config(arm,task,fold) and cfg['recipe']['ft']==t.m.DEFAULT_FT
                assert len(hist)==70 and [h['stage'] for h in hist]==['head']*10+['joint']*60
                best=min(range(70),key=lambda i:hist[i]['validation']['rmse'])+1
                assert met['best_epoch']==best and met['best_stage']==('head' if best<=10 else 'joint')
                u=frozen['units'][f'{arm}/{task}/{fold}'];assert u['config']==cfg and u['p']==.5
                for h in hist:
                    epoch=h['epoch'];runtime=h['runtime_contract']
                    assert math.isclose(h['align_coefficient'],t.m.align_coefficient(cfg['loss_schedule'],epoch),abs_tol=1e-12)
                    assert h['entropy_coefficient']==.08 and runtime['dropout']['head.2']==dict(p=.5,training=True)
                    assert all(runtime['native_modes'][n]==(epoch>10) for n in t.m.NATIVE)
                    assert all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(h['learning_rates'],t.m.expected_learning_rates(h['stage'],h['stage_epoch']-1,cfg['lr_policy'])))
                    names=[n for g in h['optimizer_groups'] for n in g['parameter_names']]
                    assert len(names)==len(set(names)) and 'encoder3d.mask_token' not in names
                    assert all(g['weight_decay']==.02 for g in h['optimizer_groups'])
                # Metadata/state reads only: no actual model construction or forward.
                saved=torch.load(path/'best.pt',map_location='cpu',weights_only=False,mmap=True)
                assert saved['schema']==t.FT_SCHEMA and saved['config']==cfg and saved['best_epoch']==best
                assert saved['strategy_metadata']==u['strategy_metadata'] and saved['scaler']==u['scaler']
                head=t.m.native.property_head(cfg['recipe']['ft']['phead'])
                head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['state_dict'].items() if k.startswith('head.')},strict=True);head.eval()
                assert isinstance(head[2],nn.Dropout) and head[2].p==.5 and not any(x.training for x in head.modules())
                del saved,head
                runtime=t.read(t.unit_path(arm,task,fold,True)/'consumption.json')['evaluation_runtime']
                assert runtime['training'] is False and all(not x['training'] for x in runtime['dropout'].values())
                assert runtime['loss_schedule']==cfg['loss_schedule'] and runtime['lr_policy']==cfg['lr_policy']
                assert runtime['global_epoch']==best and math.isclose(runtime['align_coefficient'],u['strategy_metadata']['align_coefficient'],abs_tol=1e-12)


def test_original_row_key_target_scaler_and_zero_cost_aliases():
    f=t.read(t.OUT/'selection/final_frozen.json');assert f['test_targets_read'] is False
    for task in t.TASKS:
        assert f['registry']['splits'][task]==t.read(t.SPLIT/f'{task}.json')
        for fold in range(5):
            with t.np.load(t.unit_path('BASE_ORIGINAL',task,fold,True)/'test_predictions.npz',allow_pickle=False) as ref:
                for arm in t.ORDER:
                    with t.np.load(t.unit_path(arm,task,fold,True)/'test_predictions.npz',allow_pickle=False) as p:
                        for k in ('indices','sample_keys','y_true','availability'):assert t.np.array_equal(ref[k],p[k])
                        assert t.np.isfinite(p['y_pred']).all()
                    assert t.read(t.unit_path(arm,task,fold)/'scaler.json')==t.read(t.base_path(task,fold)/'scaler.json')
                    if arm in t.ALIASES:
                        assert f['units'][f'{arm}/{task}/{fold}']['alias'] is True
                        assert not (t.OUT/'finetune'/arm).exists() and not (t.OUT/'formal'/arm).exists()


def test_budget_gpu_phase_writer_clock_and_fail_closed():
    b=t.read(t.OUT/'budget.json');assert b['budget_violations']==[] and b['charged_or_reserved']==t.LIMITS
    assert c.ledger(b['jobs'])['budget_violations']==[]
    counts={'ft':0,'test':0};events=[]
    for j in b['jobs']:
        assert j['env']['WORLD_SIZE']=='1' and j['env']['CUDA_VISIBLE_DEVICES']==('2' if j['kind'] in c.CAPS else '')
        if j['kind'] in c.CAPS:
            assert j['physical_gpus']==[2]
            p=t.read(Path(j['log'])/'process.json');u=t.read(j['consumption'])
            assert u['process']==p['timeout_identity'] and u['actual_device']['logical_device']=='cuda:0'
            assert u['actual_device']['gpu_uuid']==j['gpu_uuid']
            events.extend([(j['reserved_at'],1,j['kind']),(j['ended'],-1,j['kind'])])
    for _,change,kind in sorted(events):
        counts[kind]+=change;assert 0<=counts[kind]<=4 and not(counts['ft'] and counts['test'])
    assert c.slot_available(['ft']*3,'ft') and not c.slot_available(['ft']*4,'ft')
    with pytest.raises(ValueError):c.slot_available(['ft'],'test')
    with pytest.raises(ValueError):t.frozen_unit(dict(plan=t.PLAN,frozen=False),'C0_BASE','eea',0)
    with pytest.raises(ValueError):t.frozen_unit(t.read(t.OUT/'selection/final_frozen.json'),'BASE_ORIGINAL','eea',0)
    m=t.read(t.OUT/'execution_manifest.json');assert m['t0_unix']==t.SPEC['t0_unix'] and m['deadline_unix']==m['t0_unix']+86400
    for name in c.SOURCES:assert (t.OUT/'authorization/source_at_launch'/name).read_text()==(t.ROOT/name).read_text()
