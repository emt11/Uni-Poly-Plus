"""The sole final CPU suite: synthetic mathematics/metadata, no real inference."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.modules import ua3m_paperfuse as m
from src.dataset import ua3m_paperfuse as d
from src.training import ua3m_paperfuse as t
from scripts import run_ua3m_paperfuse as c
from scripts import run_ua2d_a020_fp_fusion as runtime_helper

torch.set_num_threads(2)


def test_native_architecture_mask_tasks_and_no_unused_pt_fusion():
    model=m.PaperFuse('UNI3M',True)
    assert len(model.o8.layers)==len(model.encoder3d.layers)==6
    for layer in (*model.o8.layers,*model.encoder3d.layers):assert layer.attention.scale_denominator==512
    assert model.atom_head[-1].out_features==101 and model.line_head[-1].out_features==25755
    assert model.atom_head[0].in_features==model.line_head[0].in_features==512
    assert not hasattr(model,'fusion') and not hasattr(model,'head')
    assert not any(name.startswith('fp_head.') for name in model.state_dict())
    assert m.CONFIG['fp_as_input'] and m.CONFIG['fp_prediction'] is False
    model.completed_updates=500
    assert model.weights()['align_0_1']==pytest.approx(.20*.5/6)
    model.completed_updates=5000;assert model.weights()['align_0_1']==pytest.approx(.20/6)


def test_explicit_native_and_head_initialization_strictness():
    model=torch.nn.Module();model.arm='UNI3M'
    source={}
    for name in (*m.NATIVE,*m.MASK_HEADS):
        setattr(model,name,torch.nn.Linear(1,1))
        for key,value in getattr(model,name).state_dict().items():source[name+'.'+key]=torch.ones_like(value)*.5
    for name,module in m.modal_modules('UNI3M').items():setattr(model,name,module)
    common=dict(schema='ua2d_galformer_objectives_v3',step=0,seed=42,plan='UA2D-PRETRAIN-FP-20261019-01',state_dict=dict(source,**{'proj2.old':torch.zeros(1)}))
    heads=dict(schema=common['schema'],step=0,strategy='mask_align',state_dict={})
    modal=m.modal_modules('UNI3M').state_dict();evidence=m.load_initial(model,common,heads,modal)
    assert 'proj2.old' in evidence['excluded']
    for name in (*m.NATIVE,*m.MASK_HEADS):
        for value in getattr(model,name).state_dict().values():assert torch.equal(value,torch.ones_like(value)*.5)
    bad=copy.deepcopy(common);bad['state_dict'].pop('o8.weight')
    with pytest.raises(ValueError):m.load_initial(model,bad,heads,modal)
    before=torch.random.get_rng_state().clone();m.modal_modules('TMOL3M');assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('arm',m.ARMS)
def test_alignment_math_reduction_identity_and_missing_intersections(arm):
    features=[torch.eye(2,requires_grad=True) for _ in range(3)];available=torch.ones((2,3),dtype=torch.bool)
    sums,counts=m.alignment_terms(features,available,['a','b'],arm)
    assert len(sums)==(6 if arm=='UNI3M' else 3)
    x=features[0];sim=(x@x.T)/((x.norm(dim=-1)[:,None]*x.norm(dim=-1)[None,:])+ (1e-8 if arm=='TMOL3M' else 0)) /(.07 if arm=='UNI3M' else .1)
    expected=(torch.logsumexp(sim,-1)-sim.diag()).sum() if arm=='UNI3M' else (torch.logaddexp(torch.logsumexp(sim.masked_fill(torch.eye(2,dtype=torch.bool),-torch.inf),-1),torch.tensor(np.log(1e-8),dtype=torch.float32))-sim.diag()).sum()
    for name,value in sums.items():assert value.item()==pytest.approx(expected.item(),abs=1e-6) and counts[name]==2
    if arm=='TMOL3M':assert sum(sums.values())<0
    else:assert sum(sums.values())>=0
    sum(sums.values()).backward();assert all(f.grad is not None for f in features)
    available[1,1]=False
    sums,counts=m.alignment_terms(features,available,['a','b'],arm)
    for a,b in m.directions(arm):
        if 1 in (a,b):assert sums[f'align_{a}_{b}']==0 and counts[f'align_{a}_{b}']==0
    _,counts=m.alignment_terms(features,torch.ones((2,3),dtype=torch.bool),['a','a'],arm)
    assert all(x==0 for x in counts.values())


@pytest.mark.parametrize('fusion',(m.Uni3MFusion,m.Tmol3MGating))
def test_three_modal_weights_missing_mask_and_gradients(fusion):
    module=fusion().eval();z=[torch.randn(3,256,requires_grad=True) for _ in range(3)]
    available=torch.tensor([[True,True,True],[True,False,True],[True,True,True]])
    z[1]=z[1]*available[:,1:2];output,weights=module(z,available)
    assert output.shape==(3,256) and weights.shape==(3,3)
    assert weights[1,1].item()==0. and torch.allclose(weights.sum(-1),torch.ones(3))
    output.square().mean().backward()
    assert all(p.grad is not None for p in module.parameters())
    h=m.entropy(weights);assert torch.isfinite(h) and h>0
    assert (-.08*h)<0  # author's negative entropy term rewards higher entropy


def test_task_metadata_and_shared_contrastive_projection():
    model=m.PaperFuse('TMOL3M',False)
    assert not hasattr(model,'QFP') and not hasattr(model,'atom_head') and not hasattr(model,'line_head')
    z=[torch.randn(2,256) for _ in range(3)];q=model.contrastive_features(z)
    assert torch.equal(q[2],model.Q2(z[2])) and torch.equal(q[0],model.Q2(z[0]))
    assert model.fusion.task_number.num_embeddings==1 and model.fusion.task_type.num_embeddings==2
    assert model.head[2].p==.5
    assert model.fp_encoder[2].p==model.fp_encoder[5].p==.2


def test_fp_little_bitorder_only_input_and_fixed_missing_inventory():
    packed=np.zeros(256,dtype=np.uint8);packed[0]=129
    fp=d.unpack_fp(packed);assert fp.shape==(2048,) and fp[0]==fp[7]==1 and fp.sum()==2
    with pytest.raises(ValueError):d.unpack_fp(np.zeros(256,dtype=np.uint8))
    with pytest.raises(ValueError):d.unpack_fp(np.ones(2048,dtype=np.uint8))
    missing=[dict(task='egb',row=i,key='b'+str(i),smiles='s'+str(i),reason='TIMEOUT') for i in range(7)]
    missing +=[dict(task='egc',row=i,key='c'+str(i),smiles='s'+str(i),reason='TIMEOUT') for i in range(8)]
    assert d.validate_missing_inventory(missing)['egb']==7
    with pytest.raises(ValueError):d.validate_missing_inventory(missing[:-1])
    bad=copy.deepcopy(missing);bad[0]['reason']='PASS'
    with pytest.raises(ValueError):d.validate_missing_inventory(bad)


def test_valid_3d_subbatch_never_contains_missing_rows(monkeypatch):
    seen=[]
    monkeypatch.setattr(d,'glt_2d_collate',lambda values:('2d',len(values)))
    def valid_only(values):seen.extend(values);return ('3d',len(values))
    monkeypatch.setattr(d,'dual_glt_collate',valid_only)
    samples=[dict(two_d=i,three_d='valid'+str(i) if i!=1 else None,fp=torch.ones(2048),available=i!=1,identity=str(i),y=torch.tensor(float(i))) for i in range(3)]
    batch=d.collate(samples)
    assert seen==['valid0','valid2'] and batch.three_d_rows.tolist()==[0,2]
    assert batch.availability.tolist()==[[True,True,True],[True,False,True],[True,True,True]]
    for x in samples:x['available']=False;x['three_d']=None
    batch=d.collate(samples);assert batch.three_d is None and batch.three_d_rows.numel()==0


@pytest.mark.parametrize('arm',m.ARMS)
def test_ft_optimizer_stages_effective_coverage_and_pt_rng_finite_domain(arm):
    model=m.PaperFuse(arm,False)
    for stage in ('head','joint'):
        opt,groups=m.optimizer_ft(model,stage)
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        assert not model.encoder3d.mask_token.requires_grad
        model.train();assert model.encoder3d.training==(stage=='joint')
        assert groups[-1]['lr']==(3e-4 if stage=='head' else 1e-3)
    model.counters=[60000,60000,60000];state=model.stream_state();model.restore_stream(state)
    assert model.stream_state()==state
    with pytest.raises(ValueError):model.restore_stream(dict(state,counters=[60001,0,0]))
    for branch in range(3):
        assert m.modal_seed(branch,30000)-m.modal_seed(branch,29999)==(2 if branch<2 else 1)
        with pytest.raises(ValueError):m.modal_seed(branch,60000)
    sets=[{m.modal_seed(branch,k) for k in range(60000)} for branch in range(3)]
    assert all(len(s)==60000 for s in sets) and not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_scaler_identity_join_global_ties_self_difference_and_freeze(tmp_path):
    scaler=dict(mean=[2.],scale=[1.],fit='train',transform='standard')
    t.scaler_validate(np.array([1.,3.,999.]),[0,1],scaler)
    with pytest.raises(ValueError):t.scaler_validate(np.array([1.,3.,999.]),[0,1],dict(scaler,mean=[9.]))
    assert t.nominate({a:dict(macro=.8) for a in ('B_A020',*m.ARMS)})=='B_A020'
    scores={a:t.summary({task:[.8]*5 for task in t.TASKS}) for a in ('B_A020',*m.ARMS)}
    comparison=t.comparisons(scores,'B_A020')['FINAL_minus_B_A020'];assert comparison['delta']==0. and comparison['improved_folds']==0
    for v in comparison['tasks'].values():assert v['folds']==[0.]*5 and v['mean']==v['std']==0.
    with pytest.raises(ValueError):t.frozen_unit(dict(frozen=False),'UNI3M','eea',0)
    path=tmp_path/'freeze.json';t.exclusive(path,dict(frozen=True))
    with pytest.raises(FileExistsError):t.exclusive(path,dict(frozen=False))
    graph=dict(row=1,key='k');fp=dict(row=1,key='k',smiles='*CO*');d.validate_join(graph,fp,1,'k','*CO*')
    with pytest.raises(ValueError):d.validate_join(graph,fp,2,'k','*CO*')


def test_irreversible_budget_and_zero_forbidden_work():
    jobs=[]
    for kind in ('prepare','cpu'):
        jobs.append(dict(kind=kind,reservation=c.reservation(kind),consumption=None))
    for arm in m.ARMS:
        jobs.append(dict(kind='pt',reservation=c.reservation('pt'),consumption=None))
        for task in t.TASKS:
            for fold in range(5):jobs.append(dict(kind='ft',reservation=c.reservation('ft',task,fold),consumption=None))
    b=c.ledger(jobs)
    for key in ('prepare_starts','pt_starts','pt_updates','pt_microbatches','pt_presentations','ft_starts','epochs_started','attempted_updates','train_graphs','validation_started','validation_graphs','cpu_suites'):
        assert b['charged_or_reserved'][key]==t.LIMITS[key]
    assert not b['budget_violations'] and all(v==0 for v in b['zero_cost'].values())
    jobs.append(dict(kind='cpu',reservation=c.reservation('cpu'),consumption=None));assert 'cpu_suites' in c.ledger(jobs)['budget_violations']


def test_atomic_exit_empty_bad_dead_boundary_and_repeat_signal(tmp_path,monkeypatch):
    job=dict(name='synthetic',attempt='r1',log=str(tmp_path))
    monkeypatch.setattr(runtime_helper,'wrapper_status',lambda job:(True,False))
    assert c.poll_exit(job,clock=lambda:0.) is None
    (tmp_path/'exit.txt').write_text('')
    with pytest.raises(c.ExitProtocolError):c.poll_exit(job,clock=lambda:10.)
    (tmp_path/'exit.txt').unlink();c.atomic_exit(tmp_path/'exit.txt',1);assert c.poll_exit(job)==1
    with pytest.raises(FileExistsError):c.atomic_exit(tmp_path/'exit.txt',0)
    (tmp_path/'exit.txt').write_text('bad')
    with pytest.raises(c.ExitProtocolError):c.poll_exit(job)
    (tmp_path/'exit.txt').unlink();monkeypatch.setattr(runtime_helper,'wrapper_status',lambda job:(False,False))
    with pytest.raises(c.ExitProtocolError):c.poll_exit(job)
    handlers={};monkeypatch.setattr(c.signal,'signal',lambda sig,handler:handlers.__setitem__(sig,handler))
    first=c.install_termination()
    with pytest.raises(RuntimeError):first(c.signal.SIGTERM,None)
    assert handlers[c.signal.SIGTERM]==handlers[c.signal.SIGINT]==c.signal.SIG_IGN


@pytest.mark.parametrize('kind',('pt','ft','test'))
def test_mock_external_unknown_failed_query_gate_and_no_start(kind,tmp_path,monkeypatch):
    controller=c.Controller.__new__(c.Controller);controller.jobs=[]
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(controller,'deadline',lambda:None)
    def forbidden(*a,**kw):raise AssertionError('busy must not create output/consume startup')
    monkeypatch.setattr(controller,'refresh',forbidden);monkeypatch.setattr(c,'gpu_snapshot',lambda:{999})
    assert controller.launch(kind,'UNI3M','eea',0) is None
    monkeypatch.setattr(c,'gpu_snapshot',lambda:None);assert not controller.gpu_ready(kind)
    controller.jobs=[dict(kind=kind,exit=None,consumption=str(tmp_path/'main.json'))]
    t.write(tmp_path/'main.json',dict(process=dict(pid=11,start_ticks=20),workers=[dict(pid=12,start_ticks=21)]))
    monkeypatch.setattr(c,'pid_aliases',lambda identity:{11,111});monkeypatch.setattr(c,'gpu_snapshot',lambda:{111})
    assert controller.gpu_ready(kind)==(kind!='pt')
    monkeypatch.setattr(c,'gpu_snapshot',lambda:{111,12});assert not controller.gpu_ready(kind)


def test_window_sum_counts_no_extra_accumulation_and_pt_invalid_geometry():
    parameter=torch.tensor(2.,requires_grad=True);counts=[1,3,2,5,4,6]
    total=torch.tensor(float(sum(counts)));loss=sum(t.window_objective(dict(sums={'mask':parameter.square()*n}),{'mask':.3},{'mask':total}) for n in counts)
    assert loss.item()==pytest.approx(.3*4.)
    loss.backward();assert parameter.grad.item()==pytest.approx(1.2)
    assert t.pt_learning_rate(0)==t.pt_learning_rate(5000)
    assert t.pt_learning_rate(4999)==t.pt_learning_rate(9999)
    model=m.PaperFuse('UNI3M',True)
    with pytest.raises(ValueError,match='PT cannot include invalid geometry'):
        model.encode_modalities(SimpleNamespace(availability=torch.tensor([[True,False,True]])))


def test_repair_nested_heads_and_exclusive_initialization(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path/'empty')
    for task in t.TASKS:
        for fold in range(5):
            path=t.head_path(task,fold)
            t.save_initialization(path,{'weight':torch.tensor([fold])})
            assert torch.load(path,weights_only=True)['weight'].item()==fold
            with pytest.raises(FileExistsError):t.save_initialization(path,{'weight':torch.tensor([-1])})
            assert torch.load(path,weights_only=True)['weight'].item()==fold
    assert len(list((t.OUT/'initialization/heads').glob('*/*.pt')))==35
    for arm in t.ARMS:
        path=t.modal_path(arm);t.save_initialization(path,{'x':torch.ones(1)})
        with pytest.raises(FileExistsError):t.save_initialization(path,{})


def test_repair_codec_and_actual_collate_valid_and_missing():
    from torch_geometric.data import Data
    graph=Data(mips_x=torch.zeros(1,2),mips_backbone_mask=torch.ones(1,dtype=torch.bool),
        lga_spd=torch.zeros(1,dtype=torch.long),lga_path_mask=torch.ones(1,1,dtype=torch.bool),
        lga_path_shift=torch.zeros(1,1,dtype=torch.long),lga_source_image_shift=torch.zeros(1,dtype=torch.long),
        lga_relation_mask=torch.zeros(1,dtype=torch.bool),bond_path_features=torch.zeros(1,2,14),
        bond_path_mask=torch.ones(1,2,dtype=torch.bool),lga_edge_index=torch.zeros(2,1,dtype=torch.long),
        lga_path_index=torch.zeros(1,1,dtype=torch.long),graph_available=True)
    geometry=graph.clone()
    for name in ('bond_z_a','bond_z_b','bond_type','bond_canonical_a','bond_canonical_b','line_source','line_target','line_path_group'):
        setattr(geometry,name,torch.zeros(1,dtype=torch.long))
    for name in ('bond_distance','bond_length_weight'):setattr(geometry,name,torch.ones(1))
    for name in ('bond_center','line_is_self','line_center_angle'):setattr(geometry,name,torch.ones(1,dtype=torch.bool))
    geometry.line_angle=torch.zeros(1,1);geometry.line_mask=torch.ones(1,1,dtype=torch.bool)
    geometry.line_path=torch.zeros(1,1,dtype=torch.long);geometry.incidence_index=torch.zeros(2,1,dtype=torch.long)
    geometry.geometry_valid=True;geometry.geometry_invalid_reason='';geometry.geometry_mode='finite_incident'
    samples=[]
    for row,available in enumerate((True,False)):
        record=dict(schema=d.CACHE_SCHEMA,clean=graph,geometry=geometry if available else None,
            task='eea',row=row,sample_key=str(row),smiles='s'+str(row),available=available,fp=torch.ones(2048))
        decoded=d.decode_template(d.encode(record))
        assert decoded['schema']==d.CACHE_SCHEMA and decoded['row']==row
        assert torch.equal(decoded['clean'].mips_x,graph.mips_x)
        restored=Data.from_dict(decoded['geometry']) if decoded['geometry'] is not None else None
        samples.append(dict(two_d=decoded['clean'],three_d=restored,fp=decoded['fp'],available=decoded['available'],identity=decoded['smiles'],y=torch.tensor(float(row))))
    batch=d.collate(samples)
    assert batch.three_d_rows.tolist()==[0] and batch.identity==['s0','s1']
    assert batch.availability.tolist()==[[True,True,True],[True,False,True]]
    assert batch.two_d.canonical_graph_index.tolist()==[0,1]
    assert batch.three_d.canonical_graph_index.tolist()==[0] and batch.three_d.geometry_valid.tolist()==[True]
    assert batch.y.tolist()==[0.,1.] and batch.morgan_fp.shape==(2,2048)


def test_repair_parent_history_separate_and_cumulative_budget(tmp_path,monkeypatch):
    parent=c.ledger([dict(kind='prepare',reservation=c.reservation('prepare'),exit=1,consumption=None)])
    assert parent['limits']['prepare_starts']==1 and parent['limits']['cache_rows']==5875
    monkeypatch.setattr(t,'LIMITS',dict(t.DEFAULT_LIMITS,prepare_starts=2,cache_rows=11750))
    coverage=tmp_path/'coverage.json';t.write(coverage,dict(records=5875))
    jobs=[dict(kind='prepare',reservation=c.reservation('prepare'),exit=0,consumption=str(coverage))]
    b=c.ledger(jobs,parent)
    assert b['charged_or_reserved']['prepare_starts']==2 and b['charged_or_reserved']['cache_rows']==11750
    assert b['known_actual_lower_bound']['cache_rows']==5875 and b['new_execution']['charged_or_reserved']['cache_rows']==5875
    assert not b['budget_violations'] and b['parent_history']['jobs'][0]['exit']==1
    controller=c.Controller.__new__(c.Controller);controller.jobs=jobs;controller.history=parent
    controller.check_failures()
    assert controller.budget()['jobs']==jobs and len(controller.budget()['jobs'])==1
    jobs.append(dict(kind='prepare',reservation=c.reservation('prepare'),exit=None,consumption=None))
    assert set(c.ledger(jobs,parent)['budget_violations'])=={'prepare_starts','cache_rows'}


def test_repair_explicit_root_subprocess_and_state_protection(tmp_path,monkeypatch):
    import json,subprocess,sys
    # Import both real entrypoints in a fresh CPU process; do not execute their mains.
    code="""import json
from scripts import run_ua3m_paperfuse as c,pretrain_ua3m_paperfuse as pt
from src.training import ua3m_paperfuse as t
t.configure_execution(t.REPAIR_OUT)
print(json.dumps(dict(root=str(t.OUT),worker=str(pt.t.OUT),head=str(t.head_path('eea',0)),modal=str(t.modal_path('UNI3M')),pt=str(t.parent_path('UNI3M')),ft=str(t.unit_path('UNI3M','eea',0)),test=str(t.unit_path('UNI3M','eea',0,True)),cache=t.config('UNI3M','eea',0)['input_cache'])))
"""
    value=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
    assert value['root']==value['worker']==str(t.REPAIR_OUT)
    assert all(Path(value[key]).is_relative_to(t.REPAIR_OUT) for key in ('head','modal','pt','ft','test','cache'))
    assert t.OUT==t.PARENT_OUT and t.LIMITS==t.DEFAULT_LIMITS
    with pytest.raises(ValueError):t.configure_execution(tmp_path/'unauthorized')
    monkeypatch.setattr(t,'ROOT',tmp_path);monkeypatch.setattr(t,'OUT',tmp_path/'repair')
    monkeypatch.setattr(t,'EXECUTION_REVISION','prepare-r1')
    parent='\n<!-- UA3M_PAPERFUSE_STATUS_START -->\n父需返修 FINAL=D_A005\n<!-- UA3M_PAPERFUSE_STATUS_END -->'
    plan=tmp_path/'CURRENT_PLAN.md';plan.write_text('# Plan\n\n> index\n'+parent+'\n')
    t.update_plan('执行中','prepare','new');t.update_plan('待审查','交付','new report')
    assert parent in plan.read_text() and plan.read_text().count('<!-- UA3M_PAPERFUSE_REPAIR1_STATUS_START -->')==1
    assert '状态**待审查**' in plan.read_text().splitlines()[2]


def test_four_gpu_inventory_busy_card_and_slots(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'ALLOWED_GPUS',(0,1,2,3))
    snapshot={gpu:dict(uuid='GPU-'+str(gpu),pids=set()) for gpu in range(4)}
    monkeypatch.setattr(c,'gpu_snapshots',lambda allowed:snapshot)
    controller=c.Controller.__new__(c.Controller);controller.jobs=[]
    snapshot[0]['pids']={999}
    assert controller.available_gpu('pt')==1
    controller.jobs=[dict(kind='pt',exit=None,physical_gpu=1,consumption=str(tmp_path/'not_yet.json'))]
    assert controller.available_gpu('pt')==2
    controller.jobs.append(dict(kind='pt',exit=None,physical_gpu=2,consumption=str(tmp_path/'not_yet2.json')))
    assert controller.available_gpu('pt') is None
    with pytest.raises(ValueError):controller.available_gpu('ft')
    controller.jobs=[dict(kind='ft',exit=None,physical_gpu=gpu,consumption=str(tmp_path/f'{gpu}_{i}.json')) for gpu in range(4) for i in range(4)]
    assert controller.available_gpu('ft') is None
    controller.jobs.pop();assert controller.available_gpu('ft')==3
    monkeypatch.setattr(c,'gpu_snapshots',lambda allowed:None)
    assert controller.available_gpu('ft') is None


def test_four_gpu_uuid_mapping_fail_closed(monkeypatch):
    def query(argv,**kwargs):
        if '--query-gpu=index,uuid' in argv:return '\n'.join(f'{gpu}, GPU-{gpu}' for gpu in range(4))
        return 'GPU-1, 123\nGPU-3, 456\n'
    monkeypatch.setattr(c.subprocess,'check_output',query)
    value=c.gpu_snapshots((0,1,2,3))
    assert value[0]['pids']==set() and value[1]['pids']=={123} and value[3]['pids']=={456}
    monkeypatch.setattr(c.subprocess,'check_output',lambda argv,**kwargs:'0, GPU-0\n')
    assert c.gpu_snapshots((0,1,2,3)) is None


def test_four_gpu_new_inputs_and_device_not_scientific_config():
    try:
        t.configure_execution(t.RESTART_OUT)
        assert t.ALLOWED_GPUS==(0,1,2,3) and t.prepared_root()==t.RESTART_OUT
        assert t.head_path('eea',0).is_relative_to(t.RESTART_OUT)
        assert t.modal_path('UNI3M').is_relative_to(t.RESTART_OUT)
        cfg=t.config('UNI3M','eea',0)
        assert 'physical_gpu' not in cfg and cfg['world_size']==1 and cfg['batch']==24
        assert Path(cfg['parent']).is_relative_to(t.RESTART_OUT)
        assert Path(cfg['input_cache'])==t.RESTART_OUT/'inputs'
        assert t.unit_path('UNI3M','eea',0,True).is_relative_to(t.RESTART_OUT)
        assert t.LIMITS['pt_starts']==3 and t.LIMITS['pt_updates']==30000
        assert t.LIMITS['prepare_starts']==3 and t.LIMITS['cache_rows']==17625
    finally:t.configure_execution()
