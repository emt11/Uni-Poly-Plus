"""Final-only CPU suite. Invoked once after all 175 formal TEST exits are 0."""
import copy
import os
from pathlib import Path
import pytest
import torch
from src.training import ua3d_head_dropout as t
from scripts import run_ua3d_head_dropout as c


def test_final_order_completeness_and_true_exits():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    budget=t.read(t.OUT/'budget.json');jobs=budget['jobs'];f=t.read(t.OUT/'selection/final_frozen.json')
    assert len(f['units'])==210 and f['frozen'] is True
    assert [j['kind'] for j in jobs].count('register')==1
    assert len([j for j in jobs if j['kind']=='ft' and j['exit']==0])==175
    assert len([j for j in jobs if j['kind']=='test' and j['exit']==0])==175
    assert len([j for j in jobs if j['kind']=='cpu'])==1
    assert all(j['exit']==0 for j in jobs if j['kind']!='cpu')
    for j in jobs:
        if j['kind']=='cpu':continue
        assert int((Path(j['log'])/'exit.txt').read_text())==0
        assert t.read(Path(j['log'])/'end.json')['exit']==0
    ft=[j for j in jobs if j['kind']=='ft'];tests=[j for j in jobs if j['kind']=='test']
    assert max(j['ended'] for j in ft)<f['time']<min(j['reserved_at'] for j in tests)
    assert max(j['ended'] for j in tests)<next(j['reserved_at'] for j in jobs if j['kind']=='cpu')
    assert len({(j['kind'],j['arm'],j['task'],j['fold']) for j in jobs})==len(jobs)
    for kind in ('ft','test'):
        assert {(j['arm'],j['task'],j['fold']) for j in jobs if j['kind']==kind}=={(a,task,fold) for a in t.ORDER[1:] for task in t.TASKS for fold in range(5)}
    assert set(t.registry()['recipes'])==set(t.ORDER)
    assert len(t.registry()['aliases'])==35


def test_single_factor_and_dropout_instance_restoration():
    record=t.registry();frozen=t.read(t.OUT/'selection/final_frozen.json')
    for arm,p in t.SPEC['recipes'].items():
        ft=record['recipes'][arm]['ft']
        assert ft==dict(t.m.DEFAULT_FT,phead=p)
        # Dropout.p is not a state_dict member; restore the actual FT config explicitly.
        head=t.m.native.property_head(p)
        assert isinstance(head[2],torch.nn.Dropout) and head[2].p==p
        saved=copy.deepcopy(head.state_dict());restored=t.m.native.property_head(ft['phead']);restored.load_state_dict(saved,strict=True);restored.eval()
        assert restored[2].p==p and all(not x.training for x in restored.modules())
        assert not any('phead' in k for k in saved)
    for arm in t.ORDER[1:]:
        for task in t.TASKS:
            for fold in range(5):
                cfg=t.read(t.unit_path(arm,task,fold)/'config.json');assert cfg==t.config(arm,task,fold)
                hist=t.read(t.unit_path(arm,task,fold)/'history.json');assert len(hist)==70
                assert [h['stage'] for h in hist]==['head']*10+['joint']*60
                assert all(h['runtime_contract']['dropout']['head.2']['p']==t.SPEC['recipes'][arm] for h in hist)
                assert all(h['runtime_contract']['dropout']['head.2']['training'] for h in hist)
                for h in hist:
                    assert h['runtime_contract']['training'] is True
                u=frozen['units'][f'{arm}/{task}/{fold}']
                assert u['config']==cfg and u['p']==cfg['recipe']['ft']['phead']
                assert u['pt_source']==str(t.parent_path(arm)) and u['head_source']==str(t.head_path(task,fold))


def test_matched_data_scaler_alias_and_predictions():
    frozen=t.read(t.OUT/'selection/final_frozen.json')
    assert frozen['test_targets_read'] is False
    for task in t.TASKS:
        assert frozen['registry']['splits'][task]==t.read(t.SPLIT/f'{task}.json')
        for fold in range(5):
            base=t.unit_path('H50_BASE',task,fold,True)
            alias=frozen['units'][f'H50_BASE/{task}/{fold}']
            assert alias['alias'] and alias['formal_alias']==str(base)
            assert str(base).startswith(str(t.REF))
            with t.np.load(base/'test_predictions.npz',allow_pickle=False) as b:
                for arm in t.ORDER[1:]:
                    folder=t.unit_path(arm,task,fold,True)
                    with t.np.load(folder/'test_predictions.npz',allow_pickle=False) as a:
                        for key in ('indices','sample_keys','y_true','availability'):assert t.np.array_equal(a[key],b[key])
                        assert t.np.isfinite(a['y_pred']).all()
                    assert t.read(folder/'scaler.json')==t.read(t.base_path(task,fold)/'scaler.json')
                    assert t.read(folder/'identity_alignment.json')['matched'] is True
                    runtime=t.read(folder/'consumption.json')['evaluation_runtime']
                    assert runtime['training'] is False and all(not x['training'] for x in runtime['dropout'].values())
                    assert runtime['dropout']['head.2']['p']==t.SPEC['recipes'][arm]
    assert not (t.OUT/'finetune/H50_BASE').exists() and not (t.OUT/'formal/H50_BASE').exists()
    jobs=t.read(t.OUT/'budget.json')['jobs'];assert not any(j['arm']=='H50_BASE' for j in jobs)


def test_selection_guard_and_freeze_validation_only():
    frozen=t.read(t.OUT/'selection/final_frozen.json');chosen,rule=t.select(frozen['validation'])
    assert chosen==frozen['FINAL'] and rule==frozen['selection']
    assert frozen['FINAL_p']==t.SPEC['recipes'][chosen]
    # Boundary and exact-tie priorities using synthetic scalar summaries, no forward.
    v={a:dict(macro=.5,tasks={task:dict(mean=.5) for task in t.TASKS}) for a in t.ORDER}
    assert t.select(v)[0]=='H50_BASE'
    v['H00']['macro']=.9;v['H00']['tasks']['eps']['mean']=.49
    assert t.select(v)[0]=='H50_BASE'
    v['H10']['macro']=.6;v['H20']['macro']=.6
    assert t.select(v)[0]=='H10'


def test_resource_phase_unique_writer_and_budget():
    b=t.read(t.OUT/'budget.json');assert b['budget_violations']==[]
    assert b['charged_or_reserved']==t.LIMITS
    events=[]
    for j in b['jobs']:
        assert j['env']['WORLD_SIZE']=='1'
        assert j['env']['CUDA_VISIBLE_DEVICES']==('2' if j['kind'] in c.CAPS else '')
        if j['kind'] in c.CAPS:
            assert j['physical_gpus']==[2]
            p=t.read(Path(j['log'])/'process.json');usage=t.read(j['consumption'])
            assert usage['process']==p['timeout_identity']
            assert usage['actual_device']['logical_device']=='cuda:0'
            assert usage['actual_device']['gpu_uuid']==j['gpu_uuid']
            events.extend([(j['reserved_at'],1,j['kind']),(j['ended'],-1,j['kind'])])
    count={'ft':0,'test':0}
    for stamp,delta,kind in sorted(events):
        count[kind]+=delta;assert 0<=count[kind]<=4
        assert not (count['ft'] and count['test'])
    assert c.slot_available(['ft']*3,'ft') and not c.slot_available(['ft']*4,'ft')
    with pytest.raises(ValueError):c.slot_available(['ft'],'test')
    with pytest.raises(ValueError):t.frozen_unit(dict(plan=t.PLAN,frozen=False),'H00','eea',0)
    with pytest.raises(ValueError):t.frozen_unit(t.read(t.OUT/'selection/final_frozen.json'),'H50_BASE','eea',0)
    assert t.read(t.OUT/'execution_manifest.json')['t0_unix']==t.SPEC['t0_unix']
    assert c.ledger(b['jobs'])['budget_violations']==[]
    source_files=t.OUT/'authorization/source_at_launch'
    for name in c.SOURCES:assert (source_files/name).read_text()==(t.ROOT/name).read_text()
