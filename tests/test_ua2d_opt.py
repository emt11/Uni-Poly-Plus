"""One final CPU suite for the isolated UA2D-OPT implementation; synthetic fixtures."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_glt_2d_sample
from src.dataset.ua2d_opt import (extend,context_signatures,prepare_opt_sample,
    collate_opt as glt_2d_collate,numeric_context_record,complete_cache,CACHE_CONTRACT,HOP_FIELDS)
from src.modules.ua2d_opt import (OptModel,Hop3O8,optimizer,restore,SCHEMA,ARCHITECTURE)
from src.modules.glt_dual import BondPathO8
from src.modules.ua2d_galformer_pretrain import PretrainConfig,objective
from src.training.ua2d_opt import (choose,a_decision,c_decision,final_decision,factorial,test_outcome as outcome,TIE_ORDER,TASKS,LIMITS)

torch.set_num_threads(2)


def fixture(smiles='*CC*'):
    top=build_canonical_periodic_topology(smiles,2)
    return top,extend(top,smiles)


def test_lifted_shortest_paths_and_image_multiplicity():
    top,data=fixture('*C*')
    assert sorted(data.lga_source_image_shift.tolist())==[-3,-2,-1,0,1,2,3]
    assert torch.equal(data.lga_source_image_shift.abs(),data.lga_spd)
    assert torch.equal(data.lga_path_mask.sum(1),data.lga_spd+1)
    assert torch.equal(data.bond_path_mask.sum(1),data.lga_spd)
    assert (data.lga_path_index[~data.lga_path_mask]==-1).all()
    signature=context_signatures(data)
    assert signature==['5|5:0:2']


def test_old_hop2_arithmetic_and_batch_offsets():
    top,data=fixture();old=build_glt_2d_sample(top,'*CC*')
    batch=glt_2d_collate([data,data]);n=len(data.mips_x);r=len(data.lga_spd)
    assert torch.equal(batch.lga_edge_index[:,r:],data.lga_edge_index+n)
    assert torch.equal(batch.lga_path_index[r:][data.lga_path_mask],data.lga_path_index[data.lga_path_mask]+n)
    assert (batch.lga_path_index[r:][~data.lga_path_mask]==-1).all()
    oldmodel=BondPathO8(0.);new=Hop3O8([2]*6,0.)
    state=new.state_dict()
    for k,v in oldmodel.state_dict().items():
        if state[k].shape==v.shape:state[k]=v.clone()
        elif k=='spd_embedding.weight':state[k][:3].copy_(v)
        elif k=='bond_bias.position':state[k][:2].copy_(v)
        else:raise AssertionError(k)
    new.load_state_dict(state,strict=True)
    with torch.no_grad():
        a=oldmodel(glt_2d_collate([old]))[0];b=new(glt_2d_collate([data]))[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('schedule',[[1,1,2,2,3,3],[2]*6,[3]*6])
def test_candidate_edges_filtered_before_softmax(schedule):
    _,data=fixture('*C*');batch=glt_2d_collate([data]);model=Hop3O8(schedule,0.)
    observed=[];handles=[]
    for layer in model.layers:
        handles.append(layer.register_forward_pre_hook(lambda m,args:observed.append((args[1].numel(),args[2].numel(),len(args[3])))))
    with torch.no_grad():model(batch)
    for h in handles:h.remove()
    assert observed==[(int((data.lga_spd<=h).sum()),)*3 for h in schedule]


def test_numeric_context_and_cached_hop_rows():
    _,data=fixture('*C*');signatures=context_signatures(data);vocab={}
    record=numeric_context_record(dict(clean=data,context=signatures),signatures,vocab)
    assert 'context' not in record and record['context_ids'].dtype==torch.long
    assert record['context_ids'].tolist()==[0] and record['cache_contract']==CACHE_CONTRACT
    assert numeric_context_record(dict(clean=data),signatures,vocab)['context_ids'].tolist()==[0]
    batch=glt_2d_collate([data,data]);offset=len(data.lga_spd)
    for name in HOP_FIELDS:
        local=getattr(data,name);merged=getattr(batch,name)
        assert torch.equal(merged,torch.cat([local,local+offset]))
    del batch.opt_hop2_rows
    with pytest.raises(ValueError,match='cached hop candidate'):Hop3O8([2]*6,0.)(batch)


def test_numeric_cache_requires_complete_publication(tmp_path):
    import json
    from src.dataset.ua2d_opt import SCHEMA as INPUT_SCHEMA
    manifest=dict(schema=INPUT_SCHEMA,contract=CACHE_CONTRACT,status='COMPLETE',
        prepared_pt_rows=910756,prepared_downstream_records=5875,vocab_size=1)
    (tmp_path/'coverage.json').write_text(json.dumps(dict(status='RUNNING')))
    (tmp_path/'cache_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):complete_cache(tmp_path)
    (tmp_path/'coverage.json').write_text(json.dumps(dict(status='COMPLETE')))
    assert complete_cache(tmp_path)['vocab_size']==1
    manifest['contract']='legacy_string_context'
    (tmp_path/'cache_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):complete_cache(tmp_path)


def test_top2_trainability_eval_optimizer_and_head_phase():
    model=OptModel('TOP2');opt,_=optimizer(model,'joint');model.train()
    train={n for n,p in model.named_parameters() if p.requires_grad}
    assert train and all(n.startswith(('o8.layers.4.','o8.layers.5.','norm2.','head.')) for n in train)
    assert not model.o8.atom_embedding.training and not model.o8.spd_embedding.training and not model.o8.path_bias.training
    assert not model.o8.bond_bias.training and all(not l.training for l in model.o8.layers[:4])
    assert all(l.training for l in model.o8.layers[4:])
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    optimizer(model,'head');model.train()
    assert not model.o8.training and not model.norm2.training
    assert all(n.startswith('head.') for n,p in model.named_parameters() if p.requires_grad)


def test_mix_zero_initial_readout_and_gradient():
    top,_=fixture();batch=glt_2d_collate([build_glt_2d_sample(top,'*CC*')])
    model=OptModel('MIX246');optimizer(model,'joint');model.eval()
    atoms,_=model.o8(batch)
    original=model.norm2(atoms.mean(0,keepdim=True))
    torch.testing.assert_close(model.features(batch),original,rtol=0,atol=0)
    model(batch).square().sum().backward()
    assert model.a2.grad is not None and model.a4.grad is not None
    assert model.a2.grad.abs()>0 and model.a4.grad.abs()>0
    optimizer(model,'head');assert not model.a2.requires_grad and not model.a4.requires_grad


@pytest.mark.parametrize('policy,schedule',[('TOP2',None),('MIX246',None),('NONE',[1,1,2,2,3,3]),('MIX246',[3]*6),('TOP2',[2]*6)])
def test_full_strict_restore_and_pure2d(policy,schedule):
    top,extended=fixture();data=extended if schedule else build_glt_2d_sample(top,'*CC*')
    batch=glt_2d_collate([data]);model=OptModel(policy,schedule=schedule).eval()
    cfg=dict(policy=policy,schedule=schedule)
    package=dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,best_stage='joint',state_dict=model.state_dict())
    restored=restore(package,cfg).eval()
    with torch.no_grad():torch.testing.assert_close(model(batch),restored(batch),rtol=0,atol=0)
    bad=copy.deepcopy(package);bad['state_dict'].pop('norm2.weight')
    with pytest.raises((ValueError,RuntimeError)):restore(bad,cfg)
    with pytest.raises(ValueError):restore(package,dict(policy=policy,schedule=[1]*6))
    assert not hasattr(batch,'bond_distance')


def small_template():
    _,clean=fixture()
    clean.geometry_valid=True;clean.bond_distance=torch.tensor([1.]);clean.line_angle=torch.zeros(1,2);clean.line_mask=torch.zeros(1,2,dtype=torch.bool)
    return dict(clean=clean,positions=torch.tensor([[0.,0.,0.],[1.,0.,0.]]),endpoint_a=torch.tensor([0]),endpoint_b=torch.tensor([1]),triplets=torch.zeros(1,2,3,dtype=torch.long),physical_center=torch.tensor([True,True]),group_index=torch.tensor([0]),group_first=torch.tensor([0]),atom_label=clean.mips_x[:,:101].argmax(-1),token_label=torch.tensor([0]),identity='synthetic',context=torch.tensor([7]*len(clean.mips_x)))


def test_clean_context_separated_and_matching_random_streams():
    t=small_template();cfg=PretrainConfig()
    a,la=prepare_opt_sample(t,cfg,17,'E');b,lb=prepare_opt_sample(t,cfg,17,'C')
    assert torch.equal(a.mips_x,b.mips_x) and torch.equal(a.bond_distance,b.bond_distance)
    for k in ('atom_selected','token_selected','token_action','token_donor','physical_pos'):assert torch.equal(la[k],lb[k])
    assert torch.equal(lb['atom_label'],t['context']) and torch.equal(la['atom_label'],t['atom_label'])
    assert torch.equal(t['clean'].mips_x[:,:101].argmax(-1),t['atom_label'])
    t['clean'].geometry_valid=False
    with pytest.raises(ValueError):prepare_opt_sample(t,cfg,17,'C')


def v(m):return dict(macro=m,tasks={t:dict(folds=[m]*5) for t in TASKS})

def test_global_triggers_ties_test_rejection_and_factorial():
    values={'A_TOP2':v(.81),'A_MIX246':v(.81)}
    assert a_decision(values,.8)['FT_POLICY']=='TOP2'
    assert a_decision(values,.81)['FT_POLICY']=='NONE'
    values.update(B2_E=v(.8),B3_E=v(.81),B2_C=v(.81),B3_C=v(.82))
    assert c_decision(values)['U']=='C' and c_decision(values)['trigger']
    assert choose({'B3_E':.8,'B3_C':.8},TIE_ORDER)=='B3_E'
    f=factorial(values);assert abs(f['macro']['interaction'])<1e-10
    values['C_SCHED_C']=v(.815);values['D_INTEGRATE']=v(.814)
    d=final_decision(values,.8,'B3_C');assert d['test_eligible'] and d['models']==['B2_E','B3_E','B2_C','B3_C','C_SCHED_C','D_INTEGRATE']
    assert final_decision(values,.99,'B3_C')['test_eligible']
    assert final_decision(values,.82,'B3_C')['FINAL']=='B3_C'
    assert 'B_FP' not in d['validation_scores'] and 'passed' not in d
    values['D_INTEGRATE']=v(.84)
    values['C_SCHED_C']=v(.83)
    d=final_decision(values,.8,'C_SCHED_C');assert d['models']==['B2_E','B3_E','B2_C','B3_C','C_SCHED_C','D_INTEGRATE']
    assert LIMITS['pt_starts']==5 and LIMITS['ft_starts']==280 and LIMITS['cpu_suites']==1


def test_formal_test_is_the_only_performance_evidence():
    assert outcome(None)['performance_pass'] is None
    assert outcome(None)['positive_test_delta'] is None
    assert outcome(-.01)['positive_test_delta'] is False
    assert outcome(0.)['performance_pass'] is False
    assert outcome(.00001)['positive_test_delta'] is True
    assert outcome(.00001)['performance_pass'] is False
    assert outcome(.010)['performance_pass'] is True
    with pytest.raises(ValueError):outcome(float('nan'))


def test_controller_handoff_preserves_completed_and_inflight_units(monkeypatch):
    from scripts import run_ua2d_opt as script
    controller=script.Controller.__new__(script.Controller)
    controller.handoff=True
    original=[dict(name='opt_ft_B3_C_eea_0',gpu=2,exit=0),
              dict(name='opt_ft_B3_C_eea_1',gpu=2,exit=None)]
    controller.jobs=list(original);controller.deadline=lambda:None;calls=[0];launched=[]
    def refresh():
        calls[0]+=1
        if calls[0]>2:
            for job in controller.jobs:job['exit']=0
    def launch(name,kind,args,gpu,**kwargs):
        assert name not in {j['name'] for j in controller.jobs}
        assert sum(j['gpu']==gpu and j['exit'] is None for j in controller.jobs)<4
        job=dict(name=name,gpu=gpu,exit=None);controller.jobs.append(job);launched.append(name);return job
    controller.refresh=refresh;controller.launch=launch
    monkeypatch.setattr(script.time,'sleep',lambda _:None)
    controller.matrix(['B3_C'])
    assert controller.jobs[:2]==original and len(launched)==33
    assert len(controller.jobs)==35 and len({j['name'] for j in controller.jobs})==35


def test_optimizer_window_global_denominator():
    x=torch.tensor(6.,requires_grad=True)
    result=dict(sums=dict(mask_2d=x,mask_3d=x*2,contrast=x*0),counts={})
    loss=objective(result,dict(mask_2d=1.,mask_3d=1.,contrast=.1),dict(mask_2d=torch.tensor(3.),mask_3d=torch.tensor(4.),contrast=torch.tensor(1.)))
    loss.backward();assert x.grad.item()==pytest.approx(1/3+2/4)


def test_required_C_runs_with_negative_delta_and_ties():
    values=dict(B2_E=v(.9),B3_E=v(.7),B2_C=v(.9),B3_C=v(.7))
    c=c_decision(values)
    assert c['required'] and c['trigger'] and c['U']=='E'
    assert c['delta']<0 and not c['historical_threshold_passed']


def test_required_D_reads_effective_MIX_policy_not_old_NONE(tmp_path,monkeypatch):
    import json
    from src.training import ua2d_opt as training
    monkeypatch.setattr(training,'OUT',tmp_path)
    folder=tmp_path/'selection';folder.mkdir()
    (folder/'a_policy.json').write_text(json.dumps(dict(FT_POLICY='NONE')))
    (folder/'cd_required.json').write_text(json.dumps(dict(execution_policy_revision='r1-CD-REQUIRED',C=dict(required=True),D=dict(required=True,FT_POLICY='MIX246'))))
    assert training.policy('D_INTEGRATE')=='MIX246'
    assert training.policy('B3_C')=='NONE'


def test_required_C_D_pre_frozen_TEST_branches_and_incomplete_rejection():
    values={a:v(.7) for a in ('A_TOP2','A_MIX246','B2_E','B3_E','B2_C','B3_C','C_SCHED_E','D_INTEGRATE')}
    values['A_MIX246']=v(.8)
    a=final_decision(values,.99,'B2_C')
    assert a['FINAL']=='A_MIX246'
    assert a['models']==['A_MIX246','C_SCHED_E','D_INTEGRATE','B3_E','B2_C']
    assert len(a['models'])*35==175
    values['B3_C']=v(.9)
    b=final_decision(values,.99,'C_SCHED_E')
    assert b['models']==['B2_E','B3_E','B2_C','B3_C','C_SCHED_E','D_INTEGRATE']
    assert len(b['models'])*35==210 and len(b['models'])*5875==35250
    values.pop('D_INTEGRATE')
    with pytest.raises(ValueError):final_decision(values,.99,'B3_E')


def test_dedicated_handoff_phase_budget_identity_and_one_claim(tmp_path):
    from src.training.ua2d_opt import (validate_cd_handoff,claim_cd_handoff,job_identity,B_ARMS)
    jobs=[]
    for arm in ('A_TOP2','A_MIX246','B2_E','B3_E','B2_C'):
        for task in TASKS:
            for fold in range(5):jobs.append(dict(name=f'opt_ft_{arm}_{task}_{fold}',kind='ft',arm=arm,exit=0))
    for kind in ('prepare','initialize'):jobs.append(dict(name='opt_'+kind,kind=kind,arm=None,exit=0))
    for arm in B_ARMS:
        jobs.append(dict(name='opt_pt_'+arm,kind='pt',arm=arm,exit=0,rank_progress=[dict(status='COMPLETE',completed_updates=5000,consumed_presentations=2520000)]*2))
    jobs.extend([dict(name='opt_ft_B3_C_eea_0',kind='ft',arm='B3_C',exit=0),dict(name='opt_ft_B3_C_eea_1',kind='ft',arm='B3_C',exit=None)])
    used=dict(prepare_starts=1,initialization_starts=1,pt_starts=4,pt_updates=20000,pt_presentations=20160000,test_starts=0,cpu_suites=0,ft_starts=177)
    budget=dict(limits=LIMITS,jobs=jobs,used=used)
    request=dict(plan='UA2D-OPT-20261021-01',execution_policy_revision='r1-CD-REQUIRED',state='controller_stopped_workers_untouched',expected_limits=LIMITS,wallclock_started_unix=1792541198.175118,c_io1_deployed_at='fixture',jobs=len(jobs),job_identities=[job_identity(j) for j in jobs])
    manifest=dict(wallclock_started_unix=1792541198.175118)
    adopted=validate_cd_handoff(request,budget,manifest)
    assert adopted==jobs and adopted is not jobs
    broken=copy.deepcopy(budget);broken['used']['prepare_starts']=2
    with pytest.raises(ValueError):validate_cd_handoff(request,broken,manifest)
    broken=copy.deepcopy(budget);broken['jobs'][0]['exit']=1
    with pytest.raises(ValueError):validate_cd_handoff(request,broken,manifest)
    broken=copy.deepcopy(budget);broken['jobs'].append(copy.deepcopy(jobs[-1]))
    with pytest.raises(ValueError):validate_cd_handoff(request,broken,manifest)
    broken=copy.deepcopy(budget);broken['jobs'][-1]['log']='other writer'
    with pytest.raises(ValueError):validate_cd_handoff(request,broken,manifest)
    claim=tmp_path/'claim.json';claim_cd_handoff(claim,dict(fixture=True))
    with pytest.raises(FileExistsError):claim_cd_handoff(claim,dict(fixture=True))


def test_ft_repair_epoch_interruption_charges_reservation_once():
    from src.training.ua2d_opt import begin_epoch_reservation,close_epoch_reservation,conservative_ft_counts
    state=dict(epochs_started=1,validation_started=1,attempted_updates=10,
        completed_updates=10,completed_epochs=1)
    begin_epoch_reservation(state,10)
    persisted=copy.deepcopy(state)
    charged=conservative_ft_counts(persisted)
    assert charged['known_actual_lower_bound']['attempted_updates']==10
    assert charged['charged']['attempted_updates']==20
    assert charged['charged']['validation_started']==2
    assert charged['uncertain']['attempted_updates']==10
    state.update(attempted_updates=14,completed_updates=14)
    assert conservative_ft_counts(state)['charged']['attempted_updates']==20
    with pytest.raises(ValueError):close_epoch_reservation(state)
    state.update(attempted_updates=20,completed_updates=20,completed_epochs=2,validation_started=2)
    close_epoch_reservation(state)
    final=conservative_ft_counts(state)
    assert not final['open_epoch'] and final['charged']['attempted_updates']==20
    assert all(v==0 for v in final['uncertain'].values())


def test_ft_repair_attempt_isolation_and_frozen_checkpoint(tmp_path,monkeypatch):
    import json
    from src.training import ua2d_opt as t
    monkeypatch.setattr(t,'OUT',tmp_path)
    root=tmp_path/'repair_ft1';root.mkdir();key='B3_C/egc/0'
    old=tmp_path/'finetune/B3_C/egc/fold0';old.mkdir(parents=True)
    (old/'best.pt').write_text('old interrupted model; must never be read')
    attempt=root/'attempts/B3_C/egc/fold0/attempt1'
    entry=dict(path=str(attempt),replacement=True,job_name='opt_repair_ft1_B3_C_egc_0',status='PLANNED')
    def save(): (root/'attempt_mapping.json').write_text(json.dumps(dict(revision='r1-FT-REPAIR1',units={key:entry})))
    save()
    assert t.unit_path('B3_C','egc',0)==attempt
    with pytest.raises(ValueError):t.repair_attempt('B3_C','egc',0,True)
    entry['status']='COMPLETE';save()
    selected=dict(best=str(attempt/'best.pt'),attempt_job=entry['job_name'])
    assert t.frozen_attempt(selected,'B3_C','egc',0)==attempt/'best.pt'
    with pytest.raises(ValueError):t.frozen_attempt(dict(selected,best=str(old/'best.pt')),'B3_C','egc',0)
    with pytest.raises(ValueError):t.frozen_attempt(dict(selected,attempt_job='wrong attempt'),'B3_C','egc',0)
    entry['path']=str(old);save()
    with pytest.raises(ValueError):t.unit_path('B3_C','egc',0)
    assert (old/'best.pt').read_text()=='old interrupted model; must never be read'


def test_ft_repair_matrix_reuses_success_and_starts_only_21(monkeypatch):
    from scripts import repair_ua2d_opt_ft1 as r
    controller=r.RepairController.__new__(r.RepairController)
    controller.jobs=[];entries={};success=0
    for task in TASKS:
        for fold in range(5):
            key=f'B3_C/{task}/{fold}';old=f'opt_ft_B3_C_{task}_{fold}'
            replacement=key in r.REPAIR_FAILED_UNITS
            done=not replacement and success<14
            if done:success+=1
            if done or replacement:controller.jobs.append(dict(name=old,exit=0 if done else 124))
            entries[key]=dict(status='COMPLETE' if done else 'PLANNED',job_name=f'opt_repair_ft1_B3_C_{task}_{fold}' if replacement else old)
    controller.old_count=len(controller.jobs);original=copy.deepcopy(controller.jobs)
    controller.refresh=lambda:None;controller.deadline=lambda:None
    launches=[]
    def launch(name,kind,args,**kwargs):
        assert name not in {j['name'] for j in controller.jobs}
        key=f"{kwargs['arm']}/{args[args.index('--task')+1]}/{args[args.index('--fold')+1]}"
        entries[key]['status']='COMPLETE';j=dict(name=name,exit=0,gpu=kwargs['gpu']);controller.jobs.append(j);launches.append(name);return j
    controller.launch=launch
    monkeypatch.setattr(r,'repair_attempt',lambda arm,task,fold,require_complete=False:entries[f'{arm}/{task}/{fold}'])
    monkeypatch.setattr(r,'unit_path',lambda *args:Path('/synthetic'))
    monkeypatch.setattr(r.time,'sleep',lambda _:None)
    controller.matrix(['B3_C'])
    assert len(launches)==21 and sum(n.startswith('opt_repair_ft1_') for n in launches)==8
    assert controller.jobs[:controller.old_count]==original
    controller.matrix(['B3_C'])
    assert len(launches)==21  # successful logical units are reused, never replayed


def test_ft_repair_aggregate_and_freeze_keep_successful_attempt(tmp_path,monkeypatch):
    import json
    from src.training import ua2d_opt as t
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'SPLIT',tmp_path/'splits');monkeypatch.setattr(t,'TASKS',('egc',))
    (tmp_path/'splits').mkdir();(tmp_path/'splits/egc.json').write_text(json.dumps(dict(folds=[dict(test_indices=[f]) for f in range(5)])))
    mapping=dict(revision='r1-FT-REPAIR1',units={});root=tmp_path/'repair_ft1';root.mkdir();(tmp_path/'selection').mkdir()
    monkeypatch.setattr(t,'config',lambda a,task,f:dict(arm=a,task=task,fold=f))
    for f in range(5):
        path=root/'attempts/B3_C/egc'/f'fold{f}'/'attempt1';path.mkdir(parents=True)
        cfg=t.config('B3_C','egc',f)
        (path/'metrics.json').write_text(json.dumps(dict(status='COMPLETE',config=cfg,r2=.7,best_epoch=7,best_stage='head',scaler=dict(mean=[0],scale=[1]))))
        (path/'consumption.json').write_text(json.dumps(dict(status='COMPLETE',completed_epochs=70,attempted_updates=70,completed_updates=70)))
        mapping['units'][f'B3_C/egc/{f}']=dict(path=str(path),replacement=True,status='COMPLETE',job_name=f'repair_{f}')
    (root/'attempt_mapping.json').write_text(json.dumps(mapping))
    values=t.aggregate('B3_C')
    assert values['units']['B3_C/egc/0']['best']==str(root/'attempts/B3_C/egc/fold0/attempt1/best.pt')
    t.freeze_final(dict(B3_C=values),dict(test_eligible=True,models=['B3_C']),{})
    frozen=json.loads((tmp_path/'selection/final_frozen.json').read_text())
    selected=frozen['units']['B3_C/egc/0']
    assert selected['attempt_job']=='repair_0'
    assert t.frozen_attempt(selected,'B3_C','egc',0)==Path(selected['best'])


def test_ft_repair_ledger_replaces_old_reservation_and_skips_unchanged_writes(tmp_path,monkeypatch):
    import json,time
    from scripts import repair_ua2d_opt_ft1 as r
    monkeypatch.setattr(r,'OUT',tmp_path);monkeypatch.setattr(r,'REPAIR',tmp_path/'repair_ft1')
    c=r.RepairController.__new__(r.RepairController)
    c.old_count=8;c.jobs=[dict(name=f'old{i}',exit=124) for i in range(8)]
    c.old_used=dict.fromkeys(r.LIMITS,0)
    c.old_used.update(ft_starts=197,epochs_started=13624,validation_started=13621,attempted_updates=313985,cpu_suites=1)
    c.old_failed_counts=dict(epochs_started=394,validation_started=391,attempted_updates=22435,test_graphs=0)
    c.failed_reserved=dict(epochs_started=560,validation_started=560,attempted_updates=34090,test_graphs=0)
    c.limits=dict(r.LIMITS);c.cached_state=None;c.cached_budget=None;c.start=time.monotonic();c.mapping=dict(units={})
    writes=[];original=r.write
    def capture(p,value):writes.append(str(p));original(p,value)
    monkeypatch.setattr(r,'write',capture)
    budget=c.refresh()
    assert budget['used']['attempted_updates']==325640
    assert budget['used']['epochs_started']==13790
    assert budget['used']['validation_started']==13790
    count=len(writes);c.refresh();assert len(writes)==count
    root=tmp_path/'new_attempt';root.mkdir();log=tmp_path/'log';log.mkdir()
    state=dict(status='RUNNING',epochs_started=1,validation_started=0,attempted_updates=0,
        epoch_reservation=dict(open=True,before_updates=0,before_validation=0,updates=91,validations=1))
    (root/'consumption.json').write_text(json.dumps(state))
    key='B3_C/egc/0';c.mapping['units'][key]=dict(status='DISPATCHED')
    c.jobs.append(dict(name='repair_egc0',kind='ft',exit=None,log=str(log),arm='B3_C',logical_unit=key,
        consumption=str(root/'consumption.json'),full_unit_reservation=dict(epochs_started=70,validation_started=70,attempted_updates=6370)))
    b=c.refresh()
    assert b['used']['attempted_updates']==325640+91
    assert b['new_known_lower_bound']['attempted_updates']==0
    assert b['new_conservatively_charged']['attempted_updates']==91
    assert b['active_unit_remaining_reserved']['attempted_updates']==6279
    assert c.jobs[:8]==[dict(name=f'old{i}',exit=124) for i in range(8)]
