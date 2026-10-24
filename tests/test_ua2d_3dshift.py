"""Prospective final CPU correctness suite. Written only; NOT run in fix1."""
import copy
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.modules.ua2d_3dshift import (SHIFT_SCHEMA,NEW_PREFIXES,LENGTH_WIDTH,ANGLE_WIDTH,RelationGeometry,
    ShiftedMaskedGalformer3D,UA2D3DShiftPretrainer,canonical_teacher,local_atom_mask,local_graph_count,
    strict_load_shift_initialization,shift_deployment_package,check_shift_package,_weighted_mean)
from src.modules.ua2d_galformer_pretrain import (PretrainConfig, UA2DGalformerPretrainer,
                                                LEGACY_OBJECTIVE_POLICY, UNIFORM_OBJECTIVE_POLICY)
from src.dataset.ua2d_3dshift import attach,collate_shift,decode_sidecar,SIDECAR_SCHEMA
from src.training import ua2d_3dshift as runtime


def uniform_config(**overrides):
    config = replace(PretrainConfig(), objective_policy=UNIFORM_OBJECTIVE_POLICY,
                     contrast_weight=1., fp_weight=1., weight_warmup=0, fp_enabled=True)
    return replace(config, **overrides)


def test_uniform_objective_has_no_hidden_warmup_and_old_configs_remain_explicit():
    config = uniform_config()
    for step in (0, 1, 499, 999, 1000, 5000, 9999):
        holder = SimpleNamespace(config=config, completed_updates=step, local_alignment=False)
        expected = dict(mask_2d=1., mask_3d=1., contrast=1., fp=1.)
        assert UA2DGalformerPretrainer.weights.fget(holder) == expected
        assert UA2D3DShiftPretrainer.weights.fget(holder) == expected
    assert config.alignment_multiplier == 1.
    historical = asdict(PretrainConfig(fp_enabled=True))
    historical.pop('objective_policy')
    legacy = PretrainConfig(**historical)
    assert legacy.objective_policy == LEGACY_OBJECTIVE_POLICY
    assert legacy.alignment_multiplier == 2. and legacy.weight_warmup == 1000
    with pytest.raises(ValueError, match='warmup conflict'):
        replace(config, weight_warmup=1000)
    with pytest.raises(ValueError, match='every task weight'):
        replace(config, fp_weight=.1)


@pytest.mark.parametrize('model_class', [UA2DGalformerPretrainer, UA2D3DShiftPretrainer])
@pytest.mark.parametrize('uniform', [True, False])
def test_alignment_forward_and_gradients_use_the_declared_direction_reduction(model_class, uniform):
    # Exercise both production forward methods with tiny encoder substitutes.
    # The independent oracle is the two cross-entropies, not alignment_loss().
    class AtomEncoder(torch.nn.Module):
        def forward(self, data):
            return torch.zeros(2, 512), None

    class BondEncoder(torch.nn.Module):
        def forward(self, data, labels):
            states = torch.zeros(2, 512)
            return states, states, data.geometry_valid.bool()

    class Projection(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.embeddings = torch.nn.Parameter(torch.nn.functional.pad(torch.tensor(values), (0, 126)))

        def forward(self, values):
            return self.embeddings

    config = uniform_config(fp_enabled=False) if uniform else PretrainConfig()
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.config = config
    model.variant = 'P_G'
    model.local_alignment = False
    model.o8, model.encoder3d = AtomEncoder(), BondEncoder()
    model.norm2, model.norm3 = torch.nn.Identity(), torch.nn.Identity()
    model.atom_head, model.line_head = torch.nn.Identity(), torch.nn.Identity()
    model.proj2 = Projection([[1., 0.], [0., 1.]])
    model.proj3 = Projection([[.8, .6], [.3, .9]])
    data = SimpleNamespace(geometry_valid=torch.tensor([True, True]),
                           canonical_graph_index=torch.arange(2), bond_batch=torch.arange(2),
                           incidence_index=torch.tensor([[0, 1], [0, 1]]),
                           bond_length_weight=torch.ones(2))
    labels = dict(strategy='mask_align', identity=['molecule-a', 'molecule-b'],
                  atom_selected=torch.ones(2, dtype=torch.bool), atom_label=torch.tensor([0, 1]),
                  token_selected=torch.ones(2, dtype=torch.bool), token_label=torch.tensor([0, 1]))
    actual = model(data, labels)
    z2, z3 = model.proj2.embeddings, model.proj3.embeddings
    logits = z2 @ z3.t() / config.temperature
    targets = torch.arange(2)
    mean_sum = .5 * (torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                     + torch.nn.functional.cross_entropy(logits.t(), targets, reduction='sum'))
    expected = mean_sum if uniform else 2 * mean_sum
    torch.testing.assert_close(actual['sums']['contrast'], expected)
    assert actual['counts']['contrast'].item() == 2
    actual_grad = torch.autograd.grad(actual['sums']['contrast'], (z2, z3), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (z2, z3))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)


def test_new_pt_recipe_does_not_mutate_the_historical_base_config(monkeypatch):
    historical = asdict(PretrainConfig(fp_enabled=True))
    monkeypatch.setattr(runtime, 'read', lambda path: dict(config=historical))
    config = PretrainConfig(**runtime.pt_values())
    assert config == uniform_config()
    assert historical['contrast_weight'] == .1 and historical['weight_warmup'] == 1000


def prepare_module():
    p=Path(__file__).resolve().parents[1]/'scripts/prepare_ua2d_3dshift.py'
    spec=importlib.util.spec_from_file_location('shift_prepare_test',p);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def carrier(translation=0):
    tri=SimpleNamespace(trimer_ru_offset=torch.tensor([-1,-1,0,0,1,1])+translation,
        trimer_base_ru_atom_id=torch.tensor([10,20,10,20,10,20]),
        mips_to_trimer_central_index=torch.tensor([2,3]),trimer_atomic_number=torch.full((6,),6))
    template=dict(endpoint_a=torch.tensor([0,1,2,3,4]),endpoint_b=torch.tensor([1,2,3,4,5]),
        group_index=torch.tensor([0,1,0,1,0]),group_first=torch.tensor([0,1]),
        atom_label=torch.zeros(2,dtype=torch.long),identity='normalized',sample_key=b'a'*32)
    return template,tri


def record(translation=0):
    template,tri=carrier(translation)
    return prepare_module().build_record(template,tri,b'a'*32,3,77)


def inputs():
    r=record();template,_=carrier();pos=torch.tensor([[0.,0.,0.],[1.,.2,0.],[2.,0.,.2],[3.,.2,0.],[4.,0.,.2],[5.,.2,0.]])
    n=r['token_count']
    d=SimpleNamespace(bond_z_a=torch.full((n,),6),bond_z_b=torch.full((n,),6),bond_type=torch.zeros(n,dtype=torch.long),
        bond_distance=(pos[template['endpoint_a']]-pos[template['endpoint_b']]).norm(dim=-1),
        bond_batch=torch.zeros(n,dtype=torch.long),geometry_valid=torch.tensor([True]),
        incidence_index=torch.tensor([[1,2,2,3],[0,0,1,1]]),canonical_graph_index=torch.zeros(2,dtype=torch.long),
        line_path=torch.tensor([[1,2,-1]]),line_angle=torch.tensor([[1.,0.]]),line_mask=torch.tensor([[True,False]]),
        line_is_self=torch.tensor([False]),line_path_group=torch.tensor([0]),line_target=torch.tensor([2]))
    attach(d,r);d.ds_group_action=torch.zeros(r['group_count'],dtype=torch.long);d.ds_group_masked=d.ds_group_action!=0
    d.ds_atom_graph=torch.zeros(r['atom_count'],dtype=torch.long)
    y=dict(token_action=torch.zeros(n,dtype=torch.long),token_donor=torch.arange(n),physical_pos=pos,
        atom_selected=torch.ones(2,dtype=torch.bool),strategy='mask_align')
    return d,y


def test_infinite_adjacency_full_image_shortest_paths_and_nonidentity_atom_ids():
    r=record();edges=set(zip(r['rel1_source'],r['rel1_target'],r['rel1_shift']))
    expected=prepare_module().covering_relations([(0,1,0),(0,1,-1)])
    assert edges==expected|{(0,0,0),(1,1,0)}
    assert (0,1,0) in edges and (0,1,1) in edges
    assert np.array_equal(r['group_atom_a'],[0,0]) and np.array_equal(r['group_atom_b'],[1,1])
    second=set(zip(r['rel2_source'],r['rel2_target'],r['rel2_shift']))
    assert second and not edges.intersection(second)
    assert (0,0,1) in second and (0,0,-1) in second
    assert all(a==b and q==0 for (a,b,q),zero in zip(sorted(edges),r['rel1_self']) if zero)
    other=record(9)
    for name in ('group_atom_a','group_atom_b','group_shift','rel1_shift','rel2_shift','token_weight','obs_pos'):
        assert np.array_equal(r[name],other[name])


def test_nonzero_self_image_has_two_canonical_incidences():
    tri=SimpleNamespace(trimer_ru_offset=torch.tensor([-1,0,1]),trimer_base_ru_atom_id=torch.tensor([99,99,99]),
        mips_to_trimer_central_index=torch.tensor([1]),trimer_atomic_number=torch.full((3,),6))
    template=dict(endpoint_a=torch.tensor([0,1]),endpoint_b=torch.tensor([1,2]),group_index=torch.tensor([0,0]),
        group_first=torch.tensor([0]),atom_label=torch.zeros(1,dtype=torch.long),identity='normalized')
    r=prepare_module().build_record(template,tri,b'b'*32,0,7)
    assert set(r['rel1_shift'])=={-1,0,1} and set(r['rel2_shift'])=={-2,2}
    assert r['rel1_self'].sum()==1
    assert r['inc_bond'].tolist()==[0,0] and r['inc_atom'].tolist()==[0,0]
    assert r['token_weight'].tolist()==[.5,.5]


def test_physical_center_weights_and_duplicate_copy_pooling():
    r=record();assert r['token_weight'].tolist()==[0.,.5,1.,.5,0.]
    x=torch.tensor([[1.],[3.],[5.]])
    assert torch.allclose(_weighted_mean(x,torch.tensor([0,0,0]),torch.tensor([0.,1.,0.]),1),x[1:2])
    repeated=x.repeat_interleave(2,0)
    assert torch.equal(_weighted_mean(repeated,torch.zeros(6,dtype=torch.long),torch.tensor([0.,0.,1.,1.,0.,0.]),1),x[1:2])


def test_identity_resolution_schema_and_bad_correspondence():
    assert prepare_module().validated_identity({'normalized_smiles':'normalized'})=='normalized'
    with pytest.raises(ValueError):prepare_module().validated_identity('normalized')
    with pytest.raises(ValueError):prepare_module().validated_identity({'wrong':'normalized'})
    import pickle
    assert decode_sidecar(pickle.dumps(record()))['schema']==SIDECAR_SCHEMA
    with pytest.raises(ValueError):decode_sidecar(pickle.dumps(dict(record(),schema='wrong')))
    template,tri=carrier();tri.mips_to_trimer_central_index=torch.tensor([2,4])
    with pytest.raises(ValueError):prepare_module().build_record(template,tri,b'a'*32,0,7)


def test_multigraph_collation_counts_indices_and_forward(monkeypatch):
    from src.dataset import ua2d_3dshift as dataset
    first,y1=inputs();second,y2=inputs();n=first.ds_token_count;a=first.ds_atom_count
    def base_collate(_):
        d=SimpleNamespace()
        for key in ('bond_z_a','bond_z_b','bond_type','bond_distance','line_angle','line_mask','line_is_self'):
            setattr(d,key,torch.cat([getattr(first,key),getattr(second,key)]))
        d.line_path=torch.cat([first.line_path,torch.where(second.line_path>=0,second.line_path+n,second.line_path)])
        d.line_path_group=torch.tensor([0,1]);d.line_target=torch.cat([first.line_target,second.line_target+n])
        d.geometry_valid=torch.tensor([True,True]);d.bond_batch=torch.cat([first.bond_batch,second.bond_batch+1])
        d.canonical_graph_index=torch.cat([first.canonical_graph_index,second.canonical_graph_index+1])
        d.incidence_index=torch.cat([first.incidence_index,second.incidence_index+torch.tensor([[n],[a]])],dim=1)
        y=dict(physical_pos=torch.cat([y1['physical_pos'],y2['physical_pos']]),token_action=torch.cat([y1['token_action'],y2['token_action']]),
            token_donor=torch.cat([y1['token_donor'],y2['token_donor']+n]),atom_selected=torch.ones(4,dtype=torch.bool))
        return d,y
    monkeypatch.setattr(dataset,'collate_samples',base_collate)
    data,labels=collate_shift([(first,y1),(second,y2)])
    assert data.ds_group_count==4 and data.ds_atom_count==4 and data.ds_token_count==10
    assert data.ds_group_graph.tolist()==[0,0,1,1]
    assert data.ds_atom_graph.tolist()==[0,0,1,1]
    assert data.ds_group_first.tolist()==[0,1,5,6]
    assert data.ds_inc_atom.max()==3 and data.ds_inc_bond.max()==3
    model=ShiftedMaskedGalformer3D(0.).eval()
    with torch.no_grad():out=model(data,labels)
    assert out['canonical_states'].shape==(4,512) and out['token_states'].shape==(10,512)
    assert out['graph_3d'].shape==(2,512) and out['valid'].tolist()==[True,True]


def test_value_heads_zero_initialization_and_gradients():
    model=ShiftedMaskedGalformer3D(0.).eval();assert not hasattr(model,'shift_embedding')
    assert LENGTH_WIDTH==pytest.approx(3/15) and ANGLE_WIDTH==pytest.approx(np.pi/15)
    states=torch.randn(2,512);source=torch.tensor([0,1]);target=torch.tensor([1,0]);bias=torch.zeros(2,8)
    geometry=RelationGeometry();context=geometry(torch.ones(2),torch.ones(2),torch.ones(2),torch.ones(2,dtype=torch.bool))
    out=model._block(model.layers[0],states,source,target,bias,model.value_layers[0](context))
    out.square().sum().backward()
    assert model.value_layers[0].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in geometry.parameters())
    model.zero_grad();geometry.zero_grad()
    with torch.no_grad():model.value_layers[0].weight.normal_(0.,.02)
    context=geometry(torch.ones(2),torch.ones(2),torch.ones(2),torch.ones(2,dtype=torch.bool))
    model._block(model.layers[0],states,source,target,bias,model.value_layers[0](context)).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in geometry.parameters())


def test_context_donor_lengths_mask_and_both_angle_positions():
    model=ShiftedMaskedGalformer3D(0.).eval();data,labels=inputs();captured=[];positions=[]
    labels['token_action'][[0,2,4]]=2;labels['token_donor'][[0,2,4]]=1
    hook=model.geometry.register_forward_pre_hook(lambda _,args:captured.append(tuple(x.detach().clone() for x in args)))
    hooks=[module.register_forward_hook(lambda _,args,out,k=k:positions.append(k)) for k,module in enumerate(model.angle_bias.position)]
    try:
        with torch.no_grad():model(data,labels)
        distance=torch.where(labels['token_action']==2,data.bond_distance[labels['token_donor']],data.bond_distance)
        assert torch.equal(captured[-1][0],distance[data.ds_obs_source])
        assert torch.equal(captured[-1][1],distance[data.ds_obs_target])
        assert {0,1}.issubset(positions)
        labels['token_action'][[0,2,4]]=1
        with torch.no_grad():model(data,labels)
        valid=(labels['token_action'][data.ds_obs_source]!=1)&(labels['token_action'][data.ds_obs_target]!=1)
        assert torch.equal(captured[-1][3],valid)
    finally:
        hook.remove()
        for h in hooks:h.remove()


def test_local_incidence_action_zero_and_empty_count():
    data,labels=inputs();labels['token_action'][1]=2
    assert local_atom_mask(data,labels,2,'F_L').tolist()==[False,True]
    data.ds_group_action=torch.tensor([0,2])
    assert local_atom_mask(data,labels,2,'P_L').tolist()==[False,False]
    labels['atom_selected'].zero_()
    assert local_graph_count(data,labels,2,'F_L').item()==0
    assert local_graph_count(data,labels,2,'P_L').item()==0
    values=torch.tensor([[1.],[3.]])
    teacher=canonical_teacher(values,data,'P_L',2)
    assert torch.equal(teacher,torch.full((2,1),2.))


@pytest.mark.parametrize('variant', ['F_G10K', 'P_G', 'P_L'])
def test_fp_initialization_strict_common_and_checkpoint_schema(variant):
    config=PretrainConfig(fp_enabled=True);torch.manual_seed(42)
    model=UA2D3DShiftPretrainer(config,variant=variant);fp=copy.deepcopy(model.fp_head.state_dict())
    evidence=strict_load_shift_initialization(model,runtime.INITIALIZATION)
    assert evidence['fp_initialization']=='independent seed20261019'
    for k,v in fp.items():assert torch.equal(v,model.fp_head.state_dict()[k])
    assert not any('shift_embedding' in k for k in model.state_dict())
    if variant == 'F_G10K':
        assert not any(key.startswith(NEW_PREFIXES) for key in model.state_dict())
        assert model.local_alignment is False
    package=shift_deployment_package(model,10000)
    assert package['schema']==runtime.PT_SCHEMA==SHIFT_SCHEMA
    check_shift_package(package,'test')
    with pytest.raises(ValueError):check_shift_package(dict(package,step=4000),'bad step')
    common=torch.load(runtime.INITIALIZATION/'common_step0.pt',map_location='cpu',weights_only=False)
    for k,v in common['state_dict'].items():assert torch.equal(v,model.state_dict()[k])


def test_original_base_lr_all_boundaries_and_empty_objective():
    from src.training.ua2d_fpdecay_sg import pt_learning_rate as base
    for u in (0,1,498,499,500,501,4998,4999,5000,5001,5499,9999):assert runtime.pt_learning_rate(u)==base(u%5000)
    assert runtime.pt_learning_rate(4999)==pytest.approx(1e-6)
    values=[runtime.pt_learning_rate(u) for u in range(499,5000)]
    assert all(x>=y for x,y in zip(values,values[1:]))
    assert runtime.pt_learning_rate(9999)==pytest.approx(1e-6)
    with pytest.raises(ValueError):runtime.pt_learning_rate(10000)
    with pytest.raises(ValueError):runtime.shift_objective(dict(sums={}),{}, {})
    assert runtime.shift_objective(dict(sums={'local':torch.tensor(0.)}),{'local':.1},{'local':torch.tensor(0.)}).item()==0


def test_budget_serialization_conservative_reservation_and_all_pair_differences(tmp_path,monkeypatch):
    from scripts import run_ua2d_3dshift as controller
    monkeypatch.setattr(controller,'OUT',tmp_path)
    c=controller.Controller.__new__(controller.Controller);c.jobs=[];c.start=None;c.started_unix=None;c.cached_state=None;c.cached_budget=None;c.evidence_only=False
    log=tmp_path/'job';log.mkdir();consumption=tmp_path/'consumption.json'
    runtime.write(consumption,dict(status='RUNNING',epochs_started=1,validation_started=0,attempted_updates=2,epoch_reservation=dict(open=True,before_updates=0,updates=7,before_validation=0,validations=1)))
    c.jobs=[dict(name='ft',kind='ft',exit=None,log=str(log),consumption=str(consumption),reservation=dict(epochs_started=70,validation_started=70,attempted_updates=490))]
    budget=c.refresh();assert budget['used']['attempted_updates']==7 and budget['used']['validation_started']==1
    assert budget['active_remaining_reservation']['epochs_started']==69
    assert 'Path' not in (tmp_path/'budget.json').read_text()
    def block(value):return dict(macro=value,tasks={t:dict(folds=[value]*5,mean=value,std=0.) for t in runtime.TASKS})
    reference=dict(validation_blocks={'F_G':block(.5)},formal_blocks={'F_G':block(.5),'P10K':block(.6),'B_FP':block(.55)})
    monkeypatch.setattr(runtime,'reference_scores',lambda:reference)
    models={'F_G10K':block(.7),'P_G':block(.9)}
    compared,interaction=runtime.comparisons(models,models)
    assert compared['P_G_minus_F_G10K']['test']['delta']==pytest.approx(.2)
    assert interaction is None and 'interaction' not in compared
    assert compared['P_G_minus_F_G10K']['interpretation']=='controlled 10k contrast'
    assert 'historical numerical reference' in compared['P_G_minus_F_G']['interpretation']
    assert len(compared['P_G_minus_F_G10K']['test']['tasks'])==7
    assert compared['P_G_minus_B_FP']['test']['delta']==pytest.approx(.35)


def test_sidecar_dependency_phase_and_path_identity(tmp_path,monkeypatch):
    import json
    from src.dataset import ua2d_3dshift as dataset
    meta=dict(plan='UA2D-3DSHIFT-20261022-01',schema=SIDECAR_SCHEMA,status='COMPLETE',records=910756,excluded_rows=635,
        parent_cache=str(dataset.PARENT_CACHE.resolve()),mapping=str(dataset.MAPPING.resolve()))
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    assert dataset.sidecar_metadata(tmp_path)['status']=='COMPLETE'
    meta['parent_cache']='/wrong/cache';(tmp_path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):dataset.sidecar_metadata(tmp_path)
    # A pre-prepare dependency check must not try to open a sidecar that does not exist.
    import ast
    source=Path(runtime.__file__).read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cache_references')
    guards=[n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='require_sidecar']
    assert len(guards)==1


def test_one_time_freeze_keeps_all70_and_fixed_final(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'OUT',tmp_path);monkeypatch.setattr(runtime,'SPLIT',tmp_path/'splits')
    monkeypatch.setattr(runtime,'reference_scores',lambda:dict(paths={'F_G':'frozen/reference'}))
    validation={}
    for task in runtime.TASKS:runtime.write(tmp_path/'splits'/f'{task}.json',dict(folds=[dict(test_indices=[i]) for i in range(5)]))
    for arm in runtime.ARMS:
        runtime.parent_path(arm).parent.mkdir(parents=True,exist_ok=True);runtime.parent_path(arm).touch()
        validation[arm]=dict(macro=-1.,tasks={t:dict(mean=-1.,std=0.) for t in runtime.TASKS})
        for task in runtime.TASKS:
            for fold in range(5):
                cfg=runtime.unit_config(arm,task,fold)
                folder=runtime.unit_path(arm,task,fold)
                runtime.write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=2,best_stage='head',scaler={'fit':'train'},attention=cfg['attention'],r2=-1.))
                runtime.write(folder/'consumption.json',dict(status='COMPLETE',completed_epochs=70))
                runtime.write(folder/'config.json',cfg);runtime.write(folder/'scaler.json',{'fit':'train'})
                (folder/'best.pt').touch()
    early=runtime.freeze_final(validation,arms=('F_G10K',))
    assert len(early['units'])==35 and set(early['models'])=={'F_G10K'}
    with pytest.raises(ValueError):runtime.freeze_final(validation,arms=('F_G10K',))
    frozen=runtime.freeze_final(validation)
    assert all(frozen['units'][k]==v for k,v in early['units'].items())
    assert len(frozen['units'])==70 and frozen['FINAL']=='P_G'
    assert set(frozen['models'])==set(runtime.ARMS)
    assert frozen['references']=={'F_G':'frozen/reference'}
    with pytest.raises(ValueError):runtime.freeze_final(validation)


def test_recovery_checkpoint_retention_and_consumed_cursor(tmp_path):
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    loader=SimpleNamespace(generator=torch.Generator().manual_seed(42))
    metadata=dict(arm='P_G',config={'seed':42})
    for step in (1000,2000,3000):
        state=dict(completed_updates=step,attempted_updates=step,consumed_presentations=step*1008,prefetched_not_consumed=1344)
        event=runtime.save_recovery_checkpoint(tmp_path,model,opt,loader,state,metadata)
    assert len(event['retained'])==2 and len(event['removed'])==1
    assert not (tmp_path/'checkpoints/intermediate_01000.pt').exists()
    package=torch.load(tmp_path/'checkpoints/intermediate_03000.pt',map_location='cpu',weights_only=False)
    assert runtime.validate_recovery_checkpoint(package)==3000
    assert package['next_global_position']==3024000
    assert package['loader_state']['prefetched_unconsumed']==1344
    assert package['automatic_resume'] is False
    assert set(package['rng'])=={'python','numpy','torch','cuda'}
    assert set(package['model'])==set(model.state_dict())
    broken=copy.deepcopy(package);broken['next_global_position']+=1344
    with pytest.raises(ValueError):runtime.validate_recovery_checkpoint(broken)
    with pytest.raises(FileExistsError):runtime.atomic_recovery_checkpoint(tmp_path/'checkpoints/intermediate_03000.pt',package)


def test_recovery_carry_is_historical_not_active_and_deadline_persists(tmp_path,monkeypatch):
    from scripts import run_ua2d_3dshift as controller
    monkeypatch.setattr(controller,'OUT',tmp_path)
    c=controller.Controller.__new__(controller.Controller)
    carry=dict.fromkeys(runtime.LIMITS,0);carry.update(prepare_starts=1,pt_starts=2,pt_updates=10630,pt_presentations=10715040)
    observed=dict(carry,pt_updates=6009,pt_presentations=6058080)
    c.carry=dict(conservative_charge=carry,observed_lower_bound=observed,active=False,exit=None)
    c.jobs=[];c.start=None;c.started_unix=runtime.ORIGINAL_WALLCLOCK_STARTED;c.cached_state=None;c.cached_budget=None;c.evidence_only=False
    b=c.refresh()
    assert b['used']['pt_updates']==10630 and b['known_actual_lower_bound']['pt_updates']==6009
    assert b['jobs']==[] and b['new_consumption']['pt_updates']==0
    assert b['active_remaining_reservation']['pt_updates']==0
    monkeypatch.setattr(controller.time,'time',lambda:runtime.ORIGINAL_DEADLINE-1)
    c.begin_clock();assert c.started_unix==runtime.ORIGINAL_WALLCLOCK_STARTED
    monkeypatch.setattr(controller.time,'time',lambda:runtime.ORIGINAL_DEADLINE)
    with pytest.raises(TimeoutError):c.deadline()
    assert runtime.SIDECAR==runtime.ORIGINAL_OUT/'inputs/sidecar'
    assert runtime.OUT==runtime.ORIGINAL_OUT/'all1_pcontrol_gpu2'
    assert runtime.EXECUTION_ARMS==('F_G10K','P_G') and runtime.LIMITS['test_starts']==105


def test_all1_stage_scope_budget_and_no_l():
    assert runtime.ARMS==('F_G10K','P_G') and runtime.FINAL=='P_G'
    assert runtime.LIMITS['pt_starts']==4 and runtime.LIMITS['pt_updates']==30630
    assert runtime.LIMITS['pt_presentations']==30875040
    assert runtime.LIMITS['ft_starts']==runtime.LIMITS['test_starts']==105
    assert runtime.LIMITS['test_graphs']==17625 and runtime.LIMITS['cpu_suites']==1
    from src.modules.ua2d_3dshift import VARIANTS
    assert all(VARIANTS[arm]['local'] is False for arm in runtime.ARMS)


def test_read_only_historical_base_remains_valid_under_new_policy():
    references=runtime.cache_references(require_sidecar=True)
    assert references['base']['step']==5000
    assert references['sidecar']['records']==910756
    assert references['mapping']['records']==910756
    assert references['transfer']['records']==5875


def test_pinned_pg5k_is_true_intermediate_and_strict_parent():
    assert runtime.parent_path('P_G5K')==runtime.OUT/'pretrain/P_G/preserved/step05000.pt'
    cfg=runtime.unit_config('P_G5K','eea',0)
    assert cfg['parent_step']==5000 and cfg['parent_arm']=='P_G' and cfg['variant']=='P_G'
    assert cfg['parent_schema']==runtime.RECOVERY_CHECKPOINT_SCHEMA and cfg['physical_gpu']==3
    package=torch.load(runtime.parent_path('P_G5K'),map_location='cpu',weights_only=False)
    assert runtime.validate_parent(package,'P_G5K').objective_policy==UNIFORM_OBJECTIVE_POLICY
    altered=dict(package,completed_updates=10000)
    with pytest.raises(ValueError):runtime.validate_parent(altered,'P_G5K')
