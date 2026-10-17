"""Final-only r15 CPU fixtures; no production data/forward or old suite replay."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from src.training import fce_graph_distill as graph
from src.modules import glt_dual_pretrain as pretraining
from src.training import fce_param as ft
from scripts import run_fce_graph_distill_unit as unit
from scripts import run_fce_graph_distill as controller


class FakeEncoder(nn.Module):
    geometry_revision=None;fusion_revision=None
    def __init__(self):
        super().__init__();self.atoms=nn.Parameter(torch.randn(3,512));self.bonds=nn.Parameter(torch.randn(2,512))
        self.mix=nn.Linear(512,512);self.norm2=nn.LayerNorm(512);self.norm3=nn.LayerNorm(512);self.encodes=0;self.fuses=0
    def encode(self,data,atom_mask=None):
        self.encodes+=1
        return dict(atom_states=self.atoms,bond_states=self.bonds,geometry_valid=torch.tensor([True]))
    def fuse(self,encoded):
        self.fuses+=1
        return torch.cat((self.norm2(self.mix(encoded['atom_states'].mean(0,keepdim=True))),self.norm3(encoded['bond_states'].mean(0,keepdim=True))),dim=1)


def test_default_state_forward_and_single_pass_graph_gradients(monkeypatch):
    monkeypatch.setattr(pretraining,'build_dual_glt_model',lambda *a,**kw:FakeEncoder())
    model=pretraining.DualPretrainer();model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
    data=SimpleNamespace(graph_available=torch.tensor([True]),lga_edge_index=torch.tensor([[0,1,2],[0,1,2]]),canonical_graph_index=torch.zeros(3,dtype=torch.long),bond_center=torch.ones(2,dtype=torch.bool),bond_batch=torch.zeros(2,dtype=torch.long))
    labels=dict(atom_mask=torch.tensor([True,False,False]),atom_label=torch.tensor([1,1,1]),distance=torch.ones(2),angle_pairs=torch.tensor([[0,1]]),angle_cos=torch.ones(1),angle_graph=torch.zeros(1,dtype=torch.long))
    old=model(data,labels);assert 'graph_feature' not in old
    model.encoder.encodes=model.encoder.fuses=0
    new=model(data,labels,return_graph_feature=True)
    assert model.encoder.encodes==model.encoder.fuses==1 and new['graph_feature'].shape==(1,1024)
    for k in old:torch.testing.assert_close(old[k],new[k])
    assert before.keys()==model.state_dict().keys() and all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    target=torch.randn(1,1024,requires_grad=True);errors,*_=graph.graph_errors(new['graph_feature'],target);errors.mean().backward()
    assert target.grad is None
    for p in (model.encoder.mix.weight,model.encoder.norm2.weight,model.encoder.norm3.weight):assert p.grad is not None and p.grad.abs().sum()>0
    teacher=FakeEncoder().eval().requires_grad_(False)
    z=graph.teacher_target(teacher,data);assert not z.requires_grad and not teacher.training and all(p.grad is None for p in teacher.parameters())
    teacher.train()
    with pytest.raises(ValueError):graph.teacher_target(teacher,data)


def test_fp32_feature_sum_global_ddp_accumulation_and_weight():
    # 2 ranks x 6 micros x 2 graphs. DDP averages ranks, never micros again.
    errors=torch.arange(1.,25.,requires_grad=True);local=[]
    for rank in range(2):
        local.append(sum(graph.graph_objective(errors.reshape(2,6,2)[rank,m],2,24) for m in range(6)))
    objective=sum(local)/2;torch.testing.assert_close(objective,errors.mean())
    torch.testing.assert_close(torch.autograd.grad(objective,errors)[0],torch.ones(24)/24)
    s=torch.ones(2,1024,dtype=torch.bfloat16,requires_grad=True);t=-s.detach()
    values,*_=graph.graph_errors(s,t);assert values.dtype==torch.float32;torch.testing.assert_close(values,torch.full((2,),4.))
    for k,expected in [(1,.0001),(999,.0999),(1000,.1),(5000,.1)]:
        assert graph.graph_weight('GRAPH_DISTILL',k)==pytest.approx(expected)
        assert graph.graph_weight('CONT_REF',k)==0.
    for value in (torch.zeros(2,1024),torch.full((2,1024),float('nan')),torch.full((2,1024),float('inf'))):
        with pytest.raises(FloatingPointError):graph.graph_errors(value,torch.ones_like(value))
    with pytest.raises(ValueError):graph.graph_errors(torch.ones(2,512),torch.ones(2,512))


@pytest.mark.parametrize('geometry_valid',[False,True])
def test_paired_clean_independent_of_mask_alias_and_positions(monkeypatch,geometry_valid):
    from torch_geometric.data import Data
    shared=torch.tensor([6,8]);source=SimpleNamespace(samples=[(b'key',None)],source_indices=[27])
    def build(*a,**kw):
        return Data(mips_x=torch.ones(2,1),lga_edge_index=torch.tensor([[0],[1]]),incidence_index=torch.tensor([[0],[0]]),
            bond_canonical_a=torch.tensor([0,1]),bond_canonical_b=torch.tensor([1,0]),bond_z_a=shared,
            geometry_valid=geometry_valid,graph_available=True)
    def prepare(*a,**kw):
        noisy=build();noisy.bond_z_a.fill_(0);noisy.md200=torch.zeros(1,200);noisy.md200_valid=torch.tensor([False])
        assert kw['opt_mode']=='GLT_FUSION_R1' and kw['position']==5040000
        return noisy,dict(atom_mask=torch.tensor([True,True]))
    monkeypatch.setattr(graph,'pretrain_source_fields',lambda *a,**kw:(None,None,'smiles',{},None,{}))
    monkeypatch.setattr(graph,'build_dual_sample',build);monkeypatch.setattr(graph,'prepare_pretrain_sample',prepare)
    noisy,labels,clean,meta=graph.paired_sample(source,0,position=5040000,cfg=graph.pt_config_for('CONT_REF'))
    assert clean.bond_z_a.tolist()==[6,8] and noisy.bond_z_a.tolist()==[0,0] and meta==dict(key=b'key'.hex(),position=5040000,source_index=27)
    stream=graph.PairedStream([None]*3,graph.pt_config_for('CONT_REF'),0)
    other=graph.PairedStream([None]*3,graph.pt_config_for('GRAPH_DISTILL'),1)
    assert list(stream.positions(0))==list(range(5040000,5040084))
    assert list(other.positions(29999))==list(range(10079916,10080000))
    a=SimpleNamespace(pair_keys=['x'],pair_positions=[1],geometry_valid=torch.tensor([False]),canonical_graph_index=torch.zeros(2,dtype=torch.long),graph_available=torch.tensor([True]),md200_valid=torch.tensor([False]))
    graph.check_pair(a,copy.deepcopy(a));b=copy.deepcopy(a);b.pair_keys=['y']
    with pytest.raises(ValueError):graph.check_pair(a,b)


def resume_fixture():
    cfg=graph.pt_config_for('GRAPH_DISTILL');m=nn.Linear(2,2);opt=torch.optim.AdamW(m.parameters(),lr=graph.lr_at(4999,cfg),weight_decay=0.)
    for p in m.parameters():opt.state[p]=dict(step=torch.tensor(5000.),exp_avg=torch.ones_like(p),exp_avg_sq=torch.ones_like(p)*2)
    identity=dict(config=graph.g05_config('T3B_G05'),world_size=2,sample_count=3)
    rng=graph.rng_state()
    state=dict(identity=identity,ordered_keys=['a','b','c'],direct_binding=None,step=5000,next_position=5040000,
        model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=[copy.deepcopy(rng),copy.deepcopy(rng)],scheduler=dict(step=5000,lr=graph.lr_at(4999,cfg)))
    return cfg,m,opt,state


def test_branch_restores_model_adam_order_rank_rng_and_original_lr():
    cfg,m,opt,state=resume_fixture()
    with torch.no_grad():
        for p in m.parameters():p.zero_()
    for rank in (0,1):
        selected,ancestry=graph.restore_branch(m,opt,state,cfg,rank=rank,world=2,ordered_keys=['a','b','c'],source_identity=dict(sample_count=3))
        assert graph.actual_equal(m.state_dict(),state['model']) and graph.actual_equal(opt.state_dict(),state['optimizer'])
        saved=graph.rng_state()
        try:graph.restore_rng(selected);assert graph.actual_equal(graph.rng_state(),state['rng'][rank])
        finally:graph.restore_rng(saved)
        assert ancestry['next_position']==5040000 and ancestry['step']==5000
    assert graph.lr_at(5000,cfg)!=graph.lr_at(0,cfg)
    assert graph.lr_at(9999,cfg)>1e-9 and cfg['schedule_total_steps']==20000
    for mutation in (lambda s:s.update(next_position=0),lambda s:s['scheduler'].update(lr=2e-4),lambda s:s['identity'].update(world_size=1),lambda s:s['ordered_keys'].reverse(),lambda s:s['optimizer']['param_groups'][0]['params'].reverse(),lambda s:s.update(rng=[s['rng'][0]])):
        bad=copy.deepcopy(state);mutation(bad)
        with pytest.raises(ValueError):graph.restore_branch(m,opt,bad,cfg,rank=0,world=2,ordered_keys=['a','b','c'],source_identity=dict(sample_count=3))


def test_validation_nodes_subset_mode_rng_and_no_updates(monkeypatch,tmp_path):
    from torch_geometric.data import Data
    class Student(nn.Module):
        def forward(self,data,labels,return_graph_feature=False):
            assert not self.training and return_graph_feature;torch.rand(())
            n=data.n
            return dict(graph_feature=torch.ones(n,1024),sums=torch.tensor([float(n),float(n),0.]),raw_geometry_sums=torch.tensor([float(n),float(n)]),counts=torch.tensor([n,n,0]),angle_graphs=torch.tensor(n))
    class Teacher(nn.Module):
        def encode(self,data,atom_mask=None):assert atom_mask is None and not self.training;return data
        def fuse(self,data):return torch.ones(data.n,1024)
    class Source:
        entries=[dict(pretrain_role='validation')]*1024
        def __len__(self):return 1024
        def close(self):self.closed=True
    source=Source();student=Student();teacher=Teacher().eval().requires_grad_(False)
    monkeypatch.setattr(graph,'OUT',tmp_path);monkeypatch.setattr(graph.dist,'barrier',lambda:None);monkeypatch.setattr(graph.dist,'all_reduce',lambda v:v.mul_(2))
    monkeypatch.setattr(graph,'load_sample_index_artifact',lambda *a:dict(direct_contract=True,indices=np.arange(1024)))
    monkeypatch.setattr(graph,'open_source',lambda *a,**kw:(source,None))
    monkeypatch.setattr(graph,'paired_sample',lambda source,i,**kw:i)
    monkeypatch.setattr(graph,'paired_collate',lambda rows:(Data(n=len(rows)),{},Data(n=len(rows)),rows))
    before=graph.rng_state();cfg=graph.pt_config_for('CONT_REF');device=torch.device('cpu')
    graph.evaluate_validation(student,teacher,cfg,6000,device,0,2)
    assert student.training and not teacher.training and source.closed and graph.actual_equal(before,graph.rng_state())
    record=graph.read(tmp_path/'pt_validation/CONT_REF/step06000/metrics.json')
    assert record['student_presented']==record['teacher_presented']==1024 and record['updates']==0 and record['components']['graph_unweighted']['mean']==0.
    assert [i for start in range(0,1024,168) for rank in (0,1) for i in graph.validation_positions(start,rank,2)]==list(range(1024))
    for step in (5000,7000,9999):
        with pytest.raises(ValueError):graph.evaluate_validation(student,teacher,cfg,step,device,0,2)


@pytest.mark.parametrize('arm',graph.ARMS)
def test_ft_public_mse_full_default_policy_r0_seed_and_best(arm,monkeypatch):
    for task in ft.TASKS:
        cfg=unit.config_for(arm,task);seen={}
        def capture(model,train,validation,criterion,device,**kw):
            seen.update(kw);assert isinstance(criterion,nn.MSELoss);return 'public'
        monkeypatch.setattr(unit,'run_periodic_tdl_epochs',capture)
        assert unit.train_candidate(None,None,None,'cpu',cfg=cfg,task=task,scaler=None,progress=None,step_observer=None)=='public'
        assert seen['adaptation_policy'] is None and seen['recipe']==ft.recipe_for('R0',task) and cfg['seed']=='42+fold' and cfg['parent_step']==10000
    toy=nn.Module();toy.model=nn.Module();toy.model.predictor=nn.Linear(2,1);toy.model.encoder=nn.Linear(2,2)
    ft.periodic_tdl_trainability(toy,head_only=False,adaptation_policy=None);assert all(p.requires_grad for p in toy.parameters())
    assert unit.selected_history([dict(epoch=1,validation_rmse=1.,fused_rmse=100.),dict(epoch=2,validation_rmse=1.,fused_rmse=0.)])['epoch']==1


def test_complete_70_freeze_policy_and_budget_no_test_before_both_arms():
    units=[]
    for arm in graph.ARMS:
        for task in ft.TASKS:
            cfg=unit.config_for(arm,task)
            for fold in range(5):
                units.append(dict(arm=arm,task=task,fold=fold,seed=42+fold,config=cfg,parent=str(graph.parent(arm)),head_initial=str(unit.head_path(task,fold)),loss_policy='MSE',adaptation_policy=None,
                    scaler={'mean':[0.],'scale':[1.]},reference=dict(config=dict(seed='42+fold',recipe=cfg['recipe']),seed=42+fold),fixed_krr=dict(scaler={'mean':[0.],'scale':[1.]})))
    frozen=dict(frozen=True,policy=copy.deepcopy(graph.POLICY),units=units);unit.validate_frozen(frozen)
    for mutate in (lambda x:x['units'].pop(),lambda x:x.update(frozen=False),lambda x:x['units'][0].update(parent=str(graph.TEACHER)),lambda x:x['units'][0].update(adaptation_policy=ft.NORM_ADAPTATION_POLICY),lambda x:x['units'][0].update(seed='42+fold')):
        bad=copy.deepcopy(frozen);mutate(bad)
        with pytest.raises(ValueError):unit.validate_frozen(bad)
    controller.require_stage('CPU',dict(controller.LIMITS))
    for key in controller.LIMITS:
        bad=dict(controller.LIMITS);bad[key]+=1
        with pytest.raises(ValueError):controller.check_budget(bad)
    for key in ('pt_completed_updates','ft_completed','formal_completed','formal_outputs'):
        bad=dict(controller.LIMITS);bad[key]-=1
        with pytest.raises(ValueError):controller.require_stage('CPU',bad)


def test_existing_output_not_replayed(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'require_tmux',lambda:None);existing=tmp_path/'owned';existing.mkdir()
    monkeypatch.setattr(controller,'OUT',existing);monkeypatch.setattr(controller,'LOG',tmp_path/'newlogs')
    with pytest.raises(FileExistsError):controller.setup()
    assert not (tmp_path/'newlogs').exists()
