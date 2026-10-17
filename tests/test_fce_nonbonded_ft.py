"""r16 final-only artificial graph/array fixtures; no real source/model replay."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from src.training import fce_nonbonded_ft as nb
from src.training import fce_param as ft
from src.dataset.glt_r10 import nonbonded_fields
from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_dual import strict_module_state
from scripts import run_fce_nonbonded_ft_unit as unit
from scripts import run_fce_nonbonded_ft as controller


def physical(n=4,edges=None,pos=None,base=None,offset=None,valid=True):
    edges=edges or [(i,i+1) for i in range(n-1)]
    xyz=torch.tensor(pos,dtype=torch.float32) if pos is not None else torch.stack((torch.arange(n).float(),torch.zeros(n),torch.zeros(n)),1)
    tri=SimpleNamespace(trimer_atomic_number=torch.full((n,),6),trimer_pos=xyz,trimer_ru_offset=torch.tensor(offset if offset is not None else [0]*n),trimer_base_ru_atom_id=torch.tensor(base if base is not None else list(range(n))),trimer_geometry_valid=valid)
    static=dict(geometry_valid=valid,token_pos_index_a=np.array([a for a,b in edges]),token_pos_index_b=np.array([b for a,b in edges]),token_z_a=np.full(len(edges),6),token_z_b=np.full(len(edges),6))
    return static,tri


def dual(static,tri):
    n=len(tri.trimer_atomic_number);m=len(static['token_pos_index_a']);a=torch.tensor(static['token_pos_index_a']);b=torch.tensor(static['token_pos_index_b'])
    return Data(geometry_mode='finite_incident',geometry_valid=tri.trimer_geometry_valid,geometry_invalid_reason='',graph_available=True,
        mips_x=torch.zeros(n,101),mips_backbone_mask=torch.ones(n,dtype=torch.bool),lga_spd=torch.ones(m,dtype=torch.long),
        lga_path_mask=torch.ones(m,1,dtype=torch.bool),lga_path_shift=torch.zeros(m,1),lga_source_image_shift=torch.zeros(m,dtype=torch.long),
        lga_relation_mask=torch.zeros(m,dtype=torch.bool),lga_edge_index=torch.stack((a,b)),lga_path_index=a[:,None],
        bond_path_features=torch.zeros(m,1),bond_path_mask=torch.ones(m,1,dtype=torch.bool),bond_z_a=torch.full((m,),6),bond_z_b=torch.full((m,),6),
        bond_distance=torch.ones(m),bond_type=torch.ones(m,dtype=torch.long),bond_center=torch.ones(m,dtype=torch.bool),bond_length_weight=torch.ones(m),
        bond_canonical_a=a,bond_canonical_b=b,incidence_index=torch.stack((torch.arange(m).repeat_interleave(2),torch.stack((a,b),1).flatten())),
        line_angle=torch.zeros(1,2),line_mask=torch.ones(1,2,dtype=torch.bool),line_is_self=torch.zeros(1,dtype=torch.bool),line_center_angle=torch.ones(1,dtype=torch.bool),
        line_source=torch.tensor([0]),line_target=torch.tensor([1]),line_path=torch.tensor([[0,1]]),line_path_group=torch.tensor([0]),
        md200=torch.zeros(1,200),md200_valid=torch.tensor([False]))


def test_shortest_three_bond_relations_dedup_and_physical_images():
    s,t=physical();rows,d,mask,endpoints=nonbonded_fields(s,t,t.trimer_pos)
    assert rows.tolist()==[1] and d.tolist()==[3.] and mask.tolist()==[True]
    for edges in ([(0,1),(1,2),(2,3),(0,3)],[(0,1),(1,2),(2,3),(0,2)]):
        s,t=physical(edges=edges);assert nonbonded_fields(s,t,t.trimer_pos)[0].numel()==0
        data,stats=nb.apply_input(dual(s,t),s,t,'NB_REAL');assert stats['no_path'] and bool(data.geometry_valid)
    s,t=physical(n=8,edges=[(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)],base=[0,1,2,3]*2,offset=[0]*4+[1]*4)
    rows,_,_,endpoints=nonbonded_fields(s,t,t.trimer_pos)
    assert rows.tolist()==[1,4] and len(set(endpoints))==2
    broken=copy.deepcopy(s);broken['token_pos_index_a']=np.append(s['token_pos_index_a'],0);broken['token_pos_index_b']=np.append(s['token_pos_index_b'],1)
    broken['token_z_a']=np.append(s['token_z_a'],6);broken['token_z_b']=np.append(s['token_z_b'],6)
    with pytest.raises(ValueError):nonbonded_fields(broken,t,t.trimer_pos)


def test_real_rigid_invariance_constant_same_mask_and_degenerates():
    s,t=physical(pos=[[0,0,0],[1,0,0],[2,0,0],[3,1,0]])
    real,rs=nb.apply_input(dual(s,t),s,t,'NB_REAL');const,cs=nb.apply_input(dual(s,t),s,t,'NB_CONST')
    assert torch.equal(real.nonbonded_bond_index,const.nonbonded_bond_index) and torch.equal(real.nonbonded_mask,const.nonbonded_mask)
    assert const.nonbonded_distance.tolist()==[3.] and real.nonbonded_distance.item()==pytest.approx(10**.5)
    tr=copy.deepcopy(t);tr.trimer_pos=t.trimer_pos@torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])+torch.tensor([8.,-4.,5.])
    moved,_=nb.apply_input(dual(s,tr),s,tr,'NB_REAL');torch.testing.assert_close(moved.nonbonded_distance,real.nonbonded_distance)
    t.trimer_pos[3]=t.trimer_pos[0]
    for arm in nb.ARMS:
        data,stats=nb.apply_input(dual(s,t),s,t,arm)
        assert data.nonbonded_bond_index.tolist()==[1] and data.nonbonded_mask.tolist()==[False] and data.nonbonded_distance.tolist()==[0.] and stats['degenerate_masked']==1
    s,t=physical(valid=False);t.trimer_pos.fill_(float('nan'))
    data,stats=nb.apply_input(dual(s,t),s,t,'NB_REAL');assert data.nonbonded_mask.numel()==0 and stats['invalid_geometry']
    s,t=physical();t.trimer_pos[0,0]=float('nan')
    with pytest.raises(ValueError):nb.apply_input(dual(s,t),s,t,'NB_REAL')
    s,t=physical();s['token_z_a'][0]=8
    with pytest.raises(ValueError):nb.apply_input(dual(s,t),s,t,'NB_REAL')
    s,t=physical();t.trimer_base_ru_atom_id[1]=0
    with pytest.raises(ValueError):nb.apply_input(dual(s,t),s,t,'NB_REAL')
    s,t=physical();t.trimer_geometry_valid=False
    with pytest.raises(ValueError):nb.apply_input(dual(s,physical()[1]),s,t,'NB_REAL')


def test_collator_bond_offsets_and_zero_residual_empty():
    s,t=physical();first,_=nb.apply_input(dual(s,t),s,t,'NB_REAL');second,_=nb.apply_input(dual(s,t),s,t,'NB_CONST')
    batch=dual_glt_collate([first,second]);assert batch.nonbonded_bond_index.tolist()==[1,4]
    assert batch.bond_batch[batch.nonbonded_bond_index].tolist()==[0,1]
    m=nb.NonbondedInput();assert m(batch,6).shape==(6,512) and torch.count_nonzero(m(batch,6))==0
    empty=first.clone();empty.nonbonded_bond_index=torch.empty(0,dtype=torch.long);empty.nonbonded_distance=torch.empty(0);empty.nonbonded_mask=torch.empty(0,dtype=torch.bool)
    assert torch.count_nonzero(m(empty,3))==0


def initial():
    m=nb.NonbondedInput()
    return dict(revision=nb.REVISION,seed=314173,parameters=34368,state_dict=copy.deepcopy(m.state_dict()))


def test_common_module_rng_zero_initial_and_later_gradient_path():
    before=torch.get_rng_state().clone();a=nb.NonbondedInput();b=nb.NonbondedInput();assert torch.equal(before,torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters())==34368 and len(a.state_dict())==4
    assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items());nb.validate_initial(initial())
    s,t=physical();data,_=nb.apply_input(dual(s,t),s,t,'NB_REAL')
    opt=torch.optim.SGD(a.parameters(),lr=.01);a(data,3).sum().backward()
    assert torch.count_nonzero(a.net[0].weight.grad)==0 and a.net[-1].weight.grad.abs().sum()>0
    opt.step();opt.zero_grad();a(data,3).sum().backward();assert a.net[0].weight.grad.abs().sum()>0
    with torch.no_grad():
        a.net[0].weight.zero_();a.net[0].weight[0,8]=1.;a.net[0].bias.zero_();a.net[-1].weight.zero_();a.net[-1].weight[:,0]=1.;a.net[-1].bias.zero_()
    s,t=physical();c1,_=nb.apply_input(dual(s,t),s,t,'NB_CONST');r1,_=nb.apply_input(dual(s,t),s,t,'NB_REAL')
    t.trimer_pos[3]=torch.tensor([4.,0.,0.]);c2,_=nb.apply_input(dual(s,t),s,t,'NB_CONST');r2,_=nb.apply_input(dual(s,t),s,t,'NB_REAL')
    torch.testing.assert_close(a(c1,3),a(c2,3));assert not torch.equal(a(r1,3),a(r2,3))


def test_private_clean_cache_one_materialization_fresh_clone_y(monkeypatch):
    s,t=physical();calls=[]
    class Source:
        samples=[(b'one',None)]
        def __len__(self):return 1
    def fields(source,i):calls.append(i);return None,t,'text',s,None
    monkeypatch.setattr(nb,'pretrain_source_fields',fields);monkeypatch.setattr(nb,'build_dual_sample',lambda *a,**kw:dual(s,t))
    cfg=unit.config_for('NB_CONST','ei');ds=nb.NonbondedDataset(Source(),[5.],cfg,indices=[8],roles=['train'])
    first=ds[0];first.nonbonded_distance.fill_(99);first.y.fill_(99);ds.set_target_override([7.]);second=ds[0]
    assert calls==[0] and second.nonbonded_distance.tolist()==[3.] and second.y.item()==7. and second.nonbonded_raw_distance.tolist()==[3.]
    assert ds.statistics()['attempted_rows']==ds.statistics()['materialized_graphs']==1


def test_fake_parent_strict_extension_and_stage_optimizer_members(monkeypatch):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.shared=nn.Linear(3,3);self.predictor=nn.Linear(3,1);self.glt=nn.Module()
            self.glt.nonbonded_input=None;self.glt.nonbonded_revision=None;self.glt.torsion_mlp=None;self.glt.geometry_mode='finite_incident'
            self.token_fusion=SimpleNamespace(set_implementation=lambda x:None)
    original=Encoder();package=dict(state_dict=copy.deepcopy(original.state_dict()));head=nn.Linear(3,1).state_dict();called=[]
    def restore(p,expected_step):
        assert expected_step==5000;called.append('restore');e=Encoder();strict_module_state(e,p['state_dict']);return e
    monkeypatch.setattr(nb,'validate_g05',lambda p,a:None);monkeypatch.setattr(nb,'restore_deployment',restore)
    # Production counts are checked during the actual FT builds; this fixture uses small shared layers.
    monkeypatch.setattr(nb,'validate_model_counts',lambda m:None)
    cfg=unit.config_for('NB_REAL','ei');m=nb.build_model(package,cfg,head,initial());assert called==['restore']
    for n,v in package['state_dict'].items():
        expected=head[n[len('predictor.'):]] if n.startswith('predictor.') else v
        assert torch.equal(m.model.state_dict()[n],expected)
    saved=copy.deepcopy(m.state_dict());other=nb.build_model(package,cfg,head,initial());strict_module_state(other,saved)
    broken=copy.deepcopy(saved);broken.pop('model.glt.nonbonded_input.net.2.weight')
    with pytest.raises(ValueError):strict_module_state(other,broken)
    bad=copy.deepcopy(cfg);bad['input_policy']=nb.input_policy('NB_CONST')
    with pytest.raises(ValueError):nb.build_model(package,bad,head,initial())
    for stage in ('head','joint'):
        ft.periodic_tdl_trainability(m,head_only=stage=='head',adaptation_policy=None)
        opt,evidence=ft.periodic_tdl_optimizer(m,'ei',head_only=stage=='head',recipe=ft.recipe_for('R0','ei'))
        groups=nb.stage_groups(m,stage,evidence)
        actual=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in m.parameters() if p.requires_grad}
        assert all(p.requires_grad==(stage=='joint') for p in m.model.glt.nonbonded_input.parameters())
        assert any('nonbonded_input' in n for g in groups for n in g['parameter_names'])==(stage=='joint')


@pytest.mark.parametrize('arm',nb.ARMS)
def test_public_r0_mse_none_policy_seed_and_rmse_best(arm,monkeypatch):
    for task in ft.TASKS:
        cfg=unit.config_for(arm,task);captured={}
        def capture(model,train,validation,criterion,device,**kw):
            assert isinstance(criterion,nn.MSELoss);captured.update(kw);return 'public'
        monkeypatch.setattr(unit,'run_periodic_tdl_epochs',capture)
        assert unit.train_candidate(None,None,None,'cpu',cfg=cfg,task=task,scaler=None,progress=None,step_observer=None)=='public'
        assert captured['adaptation_policy'] is None and captured['recipe']==ft.recipe_for('R0',task) and cfg['seed']=='42+fold'
    assert unit.selected_history([dict(epoch=1,validation_rmse=1.,fused_rmse=99.),dict(epoch=2,validation_rmse=1.,fused_rmse=0.)])['epoch']==1


def test_joint_invalidates_head_cache_using_public_loop(monkeypatch):
    import src.modules.glt_adaptation as adaptation
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.model=nn.Module();self.model.encoder=nn.Linear(2,2);self.model.predictor=nn.Linear(2,1)
    model=Toy();model.feature_cache_context={};online=SimpleNamespace(dataset=[],batch_size=2,collate_fn=None);cached=object();events=[]
    class Cache:
        def __init__(self,*a):pass
        def build(self,*a):events.append('build')
        def loader(self,*a):return cached
        def invalidate(self):events.append('invalidate')
    monkeypatch.setattr(adaptation,'HeadFeatureCache',Cache)
    def train(wrapped,loader,criterion,opt,scheduler,device,**kw):
        if opt.param_groups[0]['name']=='backbone':assert events[-1]=='invalidate' and loader is online
        else:assert loader is cached
        return (0.,None,dict(training_steps=1))
    monkeypatch.setattr(ft,'train_epoch',train);monkeypatch.setattr(ft,'evaluate',lambda *a,**kw:(1.,.5,np.array([0.,1.]),np.array([1.,2.])))
    result=ft.run_periodic_tdl_epochs(model,online,online,ft.finetune_criterion(),'cpu',task='egc',scaler=None,recipe=ft.recipe_for('R0','egc'))
    assert events==['build','invalidate'] and len(result['history'])==60 and result['best_epoch']==1


def test_complete_70_policy_freeze_and_fixed_fusion_budget():
    units=[]
    for arm in nb.ARMS:
        for task in ft.TASKS:
            cfg=unit.config_for(arm,task)
            for fold in range(5):
                scaler=dict(mean=[0.],scale=[1.])
                units.append(dict(arm=arm,task=task,fold=fold,seed=42+fold,config=cfg,parent=str(nb.PARENT),head_initial=str(unit.head_path(task,fold)),loss_policy='MSE',adaptation_policy=None,
                    input_policy=nb.input_policy(arm),module_initial=str(nb.MODULE_INITIAL),nonbonded_revision=nb.NB_REVISION,scaler=scaler,reference=dict(seed=42+fold,config=dict(seed='42+fold',recipe=cfg['recipe'])),fixed_krr=dict(scaler=scaler)))
    frozen=dict(frozen=True,policy=nb.POLICY,units=units);unit.validate_frozen(frozen)
    for mutate in (lambda x:x['units'].pop(),lambda x:x.update(frozen=False),lambda x:x['units'][0].update(input_policy=nb.input_policy('NB_REAL')),lambda x:x['units'][0].update(seed='42+fold')):
        bad=copy.deepcopy(frozen);mutate(bad)
        with pytest.raises(ValueError):unit.validate_frozen(bad)
    controller.require_stage('CPU',dict(controller.LIMITS))
    for key in controller.LIMITS:
        bad=dict(controller.LIMITS);bad[key]+=1
        with pytest.raises(ValueError):controller.check_budget(bad)
    bad=dict(controller.LIMITS);bad['ft_completed']-=1
    with pytest.raises(ValueError):controller.require_stage('formal',bad)
    from src.training.fce_kernel import blend_arrays
    np.testing.assert_array_equal(blend_arrays(np.array([0.,4.]),np.array([4.,0.]),.75),[1.,3.])
    assert unit.POLICY['weight']==.75


def test_existing_outputs_not_replayed(tmp_path,monkeypatch):
    existing=tmp_path/'owned';existing.mkdir();monkeypatch.setattr(controller,'require_tmux',lambda:None);monkeypatch.setattr(controller,'OUT',existing);monkeypatch.setattr(controller,'LOG',tmp_path/'newlogs')
    with pytest.raises(FileExistsError):controller.setup()
    assert not (tmp_path/'newlogs').exists()
