"""One final CPU suite: synthetic features/encoders only, no experiment replay."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from src.modules.fp_graph_fusion import FPGraphFusion
from src.training import fp_graph_fusion as fp
from src.training import fce_param as ft
from scripts import run_fp_graph_fusion_unit as unit
from scripts import run_fp_graph_fusion as controller


def initial():
    m=FPGraphFusion()
    return dict(revision=fp.REVISION,seed=20261017,parameters=sum(p.numel() for p in m.parameters()),state_dict=copy.deepcopy(m.state_dict()))


def inputs():
    h=torch.randn(2,1024);bits=torch.zeros(2,2048);bits[0,:17]=1;bits[1,100:120]=1
    return h,bits,torch.tensor([True,False])


def test_seed_parameter_counts_zero_V_and_shared_initial():
    before=torch.random.get_rng_state();m=FPGraphFusion();assert torch.equal(before,torch.random.get_rng_state())
    assert sum(p.numel() for p in m.parameters())==2033152
    other=FPGraphFusion();assert all(torch.equal(v,other.state_dict()[n]) for n,v in m.state_dict().items())
    assert all(torch.count_nonzero(p)==0 for p in m.v_proj.parameters())
    h,bits,valid=inputs();h[1,512:]=0
    for constant,uniform in ((True,False),(False,True),(False,False)):
        torch.testing.assert_close(m(h,bits,valid,constant=constant,uniform=uniform),h,rtol=0,atol=0)
    bad=initial();bad['state_dict']['v_proj.fp.bias'][0]=1
    with pytest.raises(ValueError):fp.validate_initial(bad)


def reference_formula(m,h,bits,valid,constant=False,uniform=False):
    a=h[:,:512];b=h[:,512:].clone();b[~valid]=0
    x=torch.full_like(bits,.5) if constant else bits;p=x*(512**.5)/x.norm(dim=1,keepdim=True)
    c=(a,b,p);active=torch.stack((torch.ones_like(valid),valid,torch.ones_like(valid)),1)
    logits=torch.stack([((m.Wq(a)/(512**.5))*m.k_proj[n](v)).sum(1) for n,v in zip(m.names,c)],1)
    logits[~active]=-torch.inf;w=active.float()/active.sum(1,keepdim=True) if uniform else logits.softmax(1)
    vs=torch.stack([m.v_proj[n](v) for n,v in zip(m.names,c)],1);vs[~active]=0
    return torch.cat((a+.5*(w[:,:,None]*vs).sum(1),b),1)


@pytest.mark.parametrize('constant,uniform',[(True,False),(False,True),(False,False)])
def test_formula_normalization_masks_including_bias_and_FP_effect(constant,uniform):
    m=FPGraphFusion();h,bits,valid=inputs()
    with torch.no_grad():
        for v in m.v_proj.values():v.weight.normal_(0,.01);v.bias.fill_(.3)
        m.v_proj.geometry.bias.fill_(1000)
    out=m(h,bits,valid,constant=constant,uniform=uniform)
    torch.testing.assert_close(out,reference_formula(m,h,bits,valid,constant,uniform))
    assert torch.count_nonzero(out[1,512:])==0 and not torch.equal(out[:,:512],h[:,:512])
    new=bits.roll(87,1)
    if constant:torch.testing.assert_close(out,m(h,new,valid,constant=True,uniform=uniform),rtol=0,atol=0)
    else:assert not torch.allclose(out,m(h,new,valid,uniform=uniform))
    # Invalid geometry can never leak its values or projected bias.
    changed=h.clone();changed[1,512:]+=12345
    torch.testing.assert_close(out[1],m(changed,bits,valid,constant=constant,uniform=uniform)[1])
    for bad in (torch.zeros_like(bits),bits.fill_(float('nan')),torch.full_like(bits,.5)):
        with pytest.raises(ValueError):m(h,bad,valid)


def test_morgan_definition_dummy_preservation_and_shared_literal_cache(tmp_path):
    assert fp.FP_SPEC['radius']==2 and fp.FP_SPEC['fpSize']==2048
    for k in ('includeChirality','useBondTypes','includeRingMembership'):assert fp.FP_SPEC[k] is True
    assert fp.FP_SPEC['countSimulation'] is False
    c=fp.BitCache(tmp_path/'fp.sqlite');bits,meta=c.get('*C[C@H](O)C*');again,hit=c.get('*C[C@H](O)C*')
    assert bits.shape==(1,2048) and meta['dummy_atoms']==2 and not meta['cache_hit'] and hit['cache_hit']
    assert torch.equal(bits,again) and c.db.execute('SELECT generated,hits FROM counts').fetchone()==(1,1)
    assert c.db.execute('SELECT smiles FROM bits').fetchone()[0]=='*C[C@H](O)C*'
    with pytest.raises(ValueError):c.get('not_a_smiles')
    assert c.db.execute('SELECT generated,hits FROM counts').fetchone()==(1,1)
    c.close()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1024));self.predictor=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1));self.encodes=0;self.fuses=0
    def encode(self,data):
        self.encodes+=1;return dict(h=data.h*self.weight,geometry_valid=data.valid)
    def fuse(self,encoded):self.fuses+=1;return encoded['h']


def model(arm='FP_REAL_KFUSE'):
    e=Encoder();head=copy.deepcopy(e.predictor.state_dict());m=fp.FPGraphModel(e,arm,head,initial());return m,head


def test_wrapper_regressor_registration_roundtrip_online_and_gradient_paths():
    m,head=model();h,bits,valid=inputs();h[1,512:]=0;data=SimpleNamespace(h=h,fp_bits=bits,valid=valid)
    assert isinstance(m.model.predictor,nn.Identity) and not hasattr(m,'feature_cache_context')
    assert all(torch.equal(v,m.head.regressor.state_dict()[n]) for n,v in head.items())
    names=list(m.named_parameters(remove_duplicate=False));assert len({id(p) for n,p in names})==len(names)
    assert sum(p.numel() for p in m.head.parameters())==2558465
    ft.periodic_tdl_trainability(m,head_only=True);m.model.eval();m.head.train();opt,evidence=ft.periodic_tdl_optimizer(m,'ei',head_only=True,recipe=ft.recipe_for('R0','ei'))
    groups=fp.stage_groups(m,'head',evidence);assert groups[0]['num_parameters']==2558465
    out=m(data);assert m.model.encodes==1 and m.model.fuses==1
    out.square().sum().backward();assert m.model.weight.grad is None
    assert any(torch.count_nonzero(p.grad) for p in m.head.fusion.v_proj.parameters())
    assert torch.count_nonzero(m.head.fusion.Wq.weight.grad)==0
    opt.step();opt.zero_grad(set_to_none=True)
    m(data).sum().backward();assert torch.count_nonzero(m.head.fusion.Wq.weight.grad)>0
    assert all(torch.count_nonzero(p.grad)>0 for p in m.head.fusion.k_proj.parameters())
    m.zero_grad(set_to_none=True);ft.periodic_tdl_trainability(m,head_only=False);opt,evidence=ft.periodic_tdl_optimizer(m,'ei',head_only=False,recipe=ft.recipe_for('R0','ei'))
    fp.stage_groups(m,'joint',evidence);m(data).sum().backward();assert m.model.weight.grad is not None and torch.count_nonzero(m.model.weight.grad)>0
    m.eval();new,_=model();new.load_state_dict(m.state_dict(),strict=True);new.eval();torch.testing.assert_close(m(data),new(data),rtol=0,atol=0)
    state=copy.deepcopy(m.state_dict());state.pop('head.fusion.Wq.weight')
    with pytest.raises(RuntimeError):new.load_state_dict(state,strict=True)
    with pytest.raises(ValueError):fp.FPGraphModel(Encoder(),'FP_REAL_KFUSE',{'bad':torch.zeros(1)},initial())


def test_mean_QK_exist_compute_but_have_no_gradient():
    m,_=model('FP_REAL_MEAN');h,bits,valid=inputs();calls=[]
    handles=[layer.register_forward_hook(lambda *args:calls.append(1)) for layer in [m.head.fusion.Wq,*m.head.fusion.k_proj.values()]]
    m(SimpleNamespace(h=h,fp_bits=bits,valid=valid)).sum().backward()
    assert len(calls)==4 and all(p.grad is None for p in [m.head.fusion.Wq.weight,*m.head.fusion.k_proj.parameters()])
    assert all(p.requires_grad for p in m.head.fusion.parameters())
    for handle in handles:handle.remove()


def tiny_graph():
    # Existing collator consumes only these finite two-atom/one-bond fields.
    return Data(geometry_mode='finite_incident',geometry_valid=True,geometry_invalid_reason='',graph_available=True,
        mips_x=torch.zeros(2,101),mips_backbone_mask=torch.ones(2,dtype=torch.bool),lga_spd=torch.ones(1,dtype=torch.long),lga_path_mask=torch.ones(1,1,dtype=torch.bool),lga_path_shift=torch.zeros(1,1),lga_source_image_shift=torch.zeros(1,dtype=torch.long),lga_relation_mask=torch.zeros(1,dtype=torch.bool),lga_edge_index=torch.tensor([[0],[1]]),lga_path_index=torch.zeros(1,1,dtype=torch.long),bond_path_features=torch.zeros(1,1),bond_path_mask=torch.ones(1,1,dtype=torch.bool),bond_z_a=torch.tensor([6]),bond_z_b=torch.tensor([6]),bond_distance=torch.ones(1),bond_type=torch.ones(1,dtype=torch.long),bond_center=torch.ones(1,dtype=torch.bool),bond_length_weight=torch.ones(1),bond_canonical_a=torch.tensor([0]),bond_canonical_b=torch.tensor([1]),incidence_index=torch.tensor([[0,0],[0,1]]),line_angle=torch.zeros(1,2),line_mask=torch.ones(1,2,dtype=torch.bool),line_is_self=torch.ones(1,dtype=torch.bool),line_center_angle=torch.ones(1,dtype=torch.bool),line_source=torch.tensor([0]),line_target=torch.tensor([0]),line_path=torch.tensor([[0,0]]),line_path_group=torch.tensor([0]))


def test_dataset_row_roles_clone_fresh_labels_and_real_collator(tmp_path,monkeypatch):
    class Source:
        samples=[(b'a',),(b'b',)]
        def __len__(self):return 2
    src=Source();calls=[];texts=['*CC*','*CO*']
    monkeypatch.setattr(fp,'pretrain_source_fields',lambda source,index:(None,None,texts[index],None,None))
    def build(*args,**kwargs):calls.append(1);return tiny_graph()
    monkeypatch.setattr(fp,'build_dual_sample',build)
    cache=fp.BitCache(tmp_path/'fp.sqlite');cfg=unit.config_for('FP_REAL_KFUSE','ei')
    ds=fp.FPGraphDataset(src,[1.,2.],cfg,indices=[12,4],roles=['train','validation'],smiles=texts,cache=cache)
    x=ds[0];x.fp_bits.fill_(0);x.y.fill_(999)
    ds.set_target_override([3.,4.]);a,b=ds[0],ds[1]
    assert len(calls)==2 and len(ds.attempted)==2 and len(ds.rows)==2
    assert a.fp_bits.any() and a.y.item()==3 and a.fp_original_row==12 and a.fp_sample_key=='61'
    batch=fp.fp_collate([b,a]);assert batch.fp_bits.shape==(2,2048)
    assert batch.fp_original_row.tolist()==[4,12] and batch.fp_sample_keys==['62','61']
    assert torch.equal(batch.fp_bits[0],b.fp_bits[0]) and torch.equal(batch.fp_bits[1],a.fp_bits[0])
    assert batch.y.flatten().tolist()==[4.,3.] and batch.canonical_graph_index.tolist()==[0,0,1,1]
    with pytest.raises(ValueError):fp.FPGraphDataset(src,[1,2],cfg,indices=[12,4],roles=['train','test'],smiles=texts,cache=cache)
    wrong=fp.FPGraphDataset(src,[1,2],cfg,indices=[12,4],roles=['train','validation'],smiles=['WRONG',texts[1]],cache=cache)
    with pytest.raises(ValueError):wrong[0]
    with pytest.raises(RuntimeError):wrong[0]
    cache.close()


def frozen_fixture():
    units=[]
    for a in fp.ARMS:
        for t in unit.TASKS:
            for f in range(5):
                cfg=unit.config_for(a,t);scaler=dict(mean=[1.],scale=[2.],fit='train',transform='standard')
                units.append(dict(arm=a,task=t,fold=f,seed=42+f,config=cfg,parent=str(fp.PARENT),head_initial=str(unit.head_path(t,f)),loss_policy='MSE',adaptation_policy=None,input_policy=fp.input_policy(a),module_initial=str(fp.MODULE_INITIAL),fp_graph_revision=fp.REVISION,scaler=scaler,reference=dict(seed=42+f,config=dict(seed='42+fold',recipe=cfg['recipe']),scaler=scaler)))
    return dict(frozen=True,policy=fp.POLICY,units=units)


def test_full_freeze_before_TEST_and_unchanged_R0_policy(monkeypatch):
    frozen=frozen_fixture();assert len(unit.validate_frozen(frozen)['units'])==105
    for bad in (dict(frozen,frozen=False),dict(frozen,units=frozen['units'][:-1])):
        with pytest.raises(ValueError):unit.validate_frozen(bad)
    bad=copy.deepcopy(frozen);bad['units'][0]['seed']='42+fold'
    with pytest.raises(ValueError):unit.validate_frozen(bad)
    called=[]
    monkeypatch.setattr(unit,'run_periodic_tdl_epochs',lambda *args,**kw:called.append((args,kw)))
    for a in fp.ARMS:
        for t in unit.TASKS:
            cfg=unit.config_for(a,t);unit.train_candidate(None,None,None,torch.device('cpu'),cfg=cfg,task=t,scaler=None,progress=None,step_observer=None)
            args,kw=called[-1];assert isinstance(args[3],nn.MSELoss) and kw['recipe']==ft.recipe_for('R0',t) and kw['adaptation_policy'] is None
            assert cfg['feature_cache']=='disabled_online_frozen_encoder' and cfg['parent']==str(fp.PARENT)
    assert unit.selected_history([dict(epoch=1,validation_rmse=1.),dict(epoch=2,validation_rmse=1.)])['epoch']==1
    monkeypatch.setattr(unit,'require_tmux',lambda:None);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    cfg=unit.config_for(fp.ARMS[0],'ei');meta=dict(config=cfg,reference={},manifest={'folds':[{}]})
    def read(path):return dict(frozen=False) if str(path).endswith('selection_before_test.json') else cfg
    monkeypatch.setattr(unit,'read',read);monkeypatch.setattr(unit,'source_binding',lambda *a:meta)
    monkeypatch.setattr(unit,'open_source',lambda *a,**k:pytest.fail('TEST source read before freeze'))
    with pytest.raises(ValueError):unit.run_ft(fp.ARMS[0],'ei',0,formal=True)


def test_budget_once_prediction_guard_and_raw_precision_aggregate(tmp_path,monkeypatch):
    b={k:0 for k in controller.LIMITS};controller.check_budget(b)
    for k,v in controller.LIMITS.items():
        bad=dict(b);bad[k]=v+1
        with pytest.raises(ValueError):controller.check_budget(bad)
    with pytest.raises(ValueError):controller.require_stage('freeze',b)
    with pytest.raises(ValueError):controller.require_stage('CPU',b)
    full=dict(controller.LIMITS);controller.require_stage('CPU',full)
    assert (full['ft_starts'],full['ft_epochs'],full['ft_attempted_updates'],full['formal_starts'],full['formal_outputs'],full['cpu_test_starts'])==(105,7200,154350,105,105,1)
    assert full['pt']==full['krr_fit']==full['krr_forward']==full['ft_workers']==0
    papers={t:dict(model='fixture',r2=.8) for t in unit.TASKS};rows=[]
    for t in unit.TASKS:
        for f in range(5):
            vals={p:dict(r2=.7+f*.001+(0.01000001 if p=='FP_REAL_KFUSE' else 0),rmse=2.+f) for p in (*fp.ARMS,'g05','r10')}
            rows.append(dict(task=t,fold=f,metrics=vals,validation={a:dict(r2=.6,rmse=1.) for a in fp.ARMS}))
    result=controller.aggregate(rows,papers)
    assert result['milestones']==dict(mechanism=True,project=True)
    assert result['macro_differences']['real_minus_null']>0.01
    assert result['tasks']['ei']['stats']['FP_REAL_KFUSE']['r2']['std']==pytest.approx(np.std([.7+f*.001+.01000001 for f in range(5)],ddof=0))
    # Existing output roots are a hard stop, not an invitation to replay.
    monkeypatch.setattr(controller,'require_tmux',lambda:None);monkeypatch.setattr(controller,'OUT',tmp_path)
    with pytest.raises(FileExistsError):controller.setup()
