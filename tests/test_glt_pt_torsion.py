"""S4's sole final CPU suite; synthetic correctness and completed artifact evidence."""
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from tests.test_dual_glt import sample
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
from src.dataset.glt_dual_pretrain import physical_torsion_fields,attach_physical_torsion,prepare_pretrain_sample,pretrain_collate
from src.modules.glt_dual import build_dual_glt_model,GalformerTrimer3D,PHYSICAL_TORSION,S4_RESOURCE,torsion_options,s4_resource_options
from src.modules.glt_dual_pretrain import DualPretrainer,deployment_package,load_deployment
from scripts import run_glt_pt_torsion_unit as unit
from scripts import select_glt_pt_torsion as select
GEOMETRY=dict(geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')

def chain(phi=0.,degenerate=False):
    # Same canonical IDs in separate images remain separate physical positions.
    c,s=np.cos(phi),np.sin(phi)
    coords=torch.tensor([[0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,c,s]],dtype=torch.float32)
    if degenerate:coords[0]=torch.tensor([-1.,0.,0.])
    tri=SimpleNamespace(trimer_pos=coords,trimer_atomic_number=torch.tensor([6,6,6,6]),
        trimer_ru_offset=torch.tensor([-1,0,0,1]),trimer_base_ru_atom_id=torch.tensor([0,0,1,1]))
    static=dict(token_pos_index_a=np.array([0,1,2]),token_pos_index_b=np.array([1,2,3]),
                token_z_a=np.array([6,6,6]),token_z_b=np.array([6,6,6]))
    return static,tri

def physical(static,tri):return physical_torsion_fields(static,tri,tri.trimer_pos)

def test_physical_paths_phi_period_rigid_reflection_and_fixed_angles():
    static,tri=chain(0);index,values,mask,count=physical(static,tri)
    assert index.tolist()==[1] and mask.tolist()==[True] and count.tolist()==[[1,1,1,0,0,1]]
    torch.testing.assert_close(values,torch.tensor([[1.,1.]]))
    _,rot=chain(np.pi/2);_,other,valid,_=physical(static,rot)
    torch.testing.assert_close(other,torch.tensor([[0.,-1.]]),atol=1e-6,rtol=0)
    assert valid.all() and not torch.equal(values,other)
    # All three lengths and both adjacent angle cosines stay fixed.
    def invariants(x):
        d=x[1:]-x[:-1]
        return torch.cat((d.norm(dim=1),torch.nn.functional.cosine_similarity(-d[:-1],d[1:],dim=1)))
    torch.testing.assert_close(invariants(tri.trimer_pos),invariants(rot.trimer_pos))
    for phi in (.7,-.7,.7+2*np.pi):
        _,t=chain(phi);a=physical(static,t)[1]
        torch.testing.assert_close(a,torch.tensor([[np.cos(.7),np.cos(1.4)]],dtype=torch.float32))
        t.trimer_pos=t.trimer_pos@torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])+torch.tensor([4.,3.,-2.])
        torch.testing.assert_close(a,physical(static,t)[1],atol=1e-6,rtol=1e-6)
        t.trimer_pos[:,2]*=-1
        torch.testing.assert_close(a,physical(static,t)[1],atol=1e-6,rtol=1e-6)
    reversed_static=dict(static,token_pos_index_a=static['token_pos_index_b'],token_pos_index_b=static['token_pos_index_a'])
    assert physical(reversed_static,tri)[0].tolist()==[1]
    torch.testing.assert_close(physical(reversed_static,tri)[1],values)

def test_duplicate_bounds_identity_nonfinite_and_degenerate_are_distinct():
    static,tri=chain(degenerate=True);_,v,m,c=physical(static,tri)
    assert not m.any() and v.count_nonzero()==0 and c[0,4]==1
    _,tri=chain();tri.trimer_pos[2]=tri.trimer_pos[1]
    assert not physical(static,tri)[2].any()
    for key,change in [('token_pos_index_a',np.array([0,1,4])),('token_z_a',np.array([8,6,6]))]:
        with pytest.raises(ValueError):physical(dict(static,**{key:change}),chain()[1])
    duplicate={k:np.append(v,v[1]) for k,v in static.items()}
    with pytest.raises(ValueError,match='duplicate'):physical(duplicate,chain()[1])
    _,tri=chain();tri.trimer_pos[0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):physical(static,tri)
    _,tri=chain();tri.trimer_ru_offset[0]=2
    with pytest.raises(ValueError,match='RU'):physical(static,tri)

def test_valid_empty_invalid_bias_zero_and_per_bond_mean_and_gradients():
    static,tri=chain();empty={k:v[:1] for k,v in static.items()}
    data=Data(geometry_valid=True);attach_physical_torsion(data,empty,tri,tri.trimer_pos)
    assert data.geometry_valid and not data.torsion_available.any() and data.torsion_counts[0,3]==1
    invalid=Data(geometry_valid=False);attach_physical_torsion(invalid,static,tri,torch.full((4,3),float('nan')))
    assert not invalid.geometry_valid and invalid.torsion_values.shape==(0,2)
    m=GalformerTrimer3D(dropout=0,torsion=True,torsion_revision='glt_physical_torsion_r1',geometry_mode='finite_incident')
    with torch.no_grad():m.torsion_mlp[-1].bias.fill_(9)
    assert m._torsion_increment(data,3).count_nonzero()==0
    assert m._torsion_increment(invalid,3).count_nonzero()==0
    d=Data(torsion_revision='glt_physical_torsion_r1',torsion_values=torch.tensor([[1.,1.],[0.,-1.],[1.,1.]]),
        torsion_mask=torch.tensor([True,True,False]),torsion_bond_index=torch.tensor([1,1,2]))
    expected=m.torsion_mlp(d.torsion_values[:2]).mean(0)
    update=m._torsion_increment(d,4)
    torch.testing.assert_close(update[1],expected)
    assert update[[0,2,3]].count_nonzero()==0
    (m.torsion_gate*update).sum().backward()
    assert m.torsion_gate.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.count_nonzero()==0 for p in m.torsion_mlp.parameters())
    m.zero_grad();m.torsion_gate.data.fill_(.3)
    (m.torsion_gate*m._torsion_increment(d,4)).square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.torsion_mlp.parameters())

def synthetic_record(text='*CCCC*',position=0,sigma=.03,torsion=True):
    top,tri,_=sample(text);static=build_dual_static(top,tri,text)
    kwargs=dict(seed=42,key='synthetic',position=position,sigma=sigma,ratio=.3,static=static,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')
    if torsion:kwargs.update(torsion_mode='on',torsion_revision='glt_physical_torsion_r1')
    return top,tri,static,prepare_pretrain_sample(top,tri,text,**kwargs)

def test_noisy_pt_clean_ft_mask_targets_rng_and_source_unchanged(monkeypatch):
    import src.dataset.glt_dual_pretrain as prep
    top,tri,_=sample('*CCCC*');static=build_dual_static(top,tri,'*CCCC*')
    original=tri.trimer_pos.clone();captured=[];real=prep.attach_physical_torsion
    def capture(data,rows,carrier,coords):captured.append(coords.clone());return real(data,rows,carrier,coords)
    monkeypatch.setattr(prep,'attach_physical_torsion',capture)
    kwargs=dict(seed=42,key='same',position=15,sigma=.03,ratio=.3,static=static,opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')
    rng=torch.get_rng_state().clone()
    baseline,labels=prepare_pretrain_sample(top,tri,'*CCCC*',**kwargs)
    candidate,newlabels=prepare_pretrain_sample(top,tri,'*CCCC*',torsion_mode='on',torsion_revision='glt_physical_torsion_r1',**kwargs)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(tri.trimer_pos,original)
    assert len(captured)==1 and not torch.equal(captured[0],original)
    for key,value in labels.items():
        if torch.is_tensor(value):torch.testing.assert_close(value,newlabels[key],rtol=0,atol=0)
        else:assert value==newlabels[key]
    for key,value in baseline.items():
        if torch.is_tensor(value):torch.testing.assert_close(value,candidate[key],rtol=0,atol=0)
    clean=build_dual_sample(top,tri,'*CCCC*',static=static,geometry_mode='finite_incident',torsion_mode='on',torsion_revision='glt_physical_torsion_r1')
    expected=physical_torsion_fields(static,tri,original)
    torch.testing.assert_close(clean.torsion_values,expected[1]);assert clean.geometry_valid
    assert (candidate.bond_z_a[newlabels['atom_mask'][candidate.bond_canonical_a]]==0).all()
    assert (candidate.bond_z_b[newlabels['atom_mask'][candidate.bond_canonical_b]]==0).all()
    with pytest.raises(ValueError):prepare_pretrain_sample(top,tri,'*CCCC*',torsion_mode='off',torsion_revision='glt_physical_torsion_r1',**kwargs)

def test_collate_physical_offsets_isolation_and_zero_gate_equivalence():
    first=synthetic_record(position=0)[3];second=synthetic_record(position=1)[3]
    data,labels=pretrain_collate([first,second]);n=first[0].torsion_bond_index.numel();b=first[0].bond_distance.numel()
    torch.testing.assert_close(data.torsion_bond_index[n:],second[0].torsion_bond_index+b)
    assert (data.bond_batch[data.torsion_bond_index[:n]]==0).all() and (data.bond_batch[data.torsion_bond_index[n:]]==1).all()
    torch.manual_seed(42);legacy=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',dropout=0,**GEOMETRY).eval();rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',dropout=0,torsion=True,torsion_revision='glt_physical_torsion_r1',**GEOMETRY).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in legacy.state_dict().items():assert torch.equal(v,new.state_dict()[k])
    with torch.no_grad():torch.testing.assert_close(legacy(data),new(data),rtol=0,atol=0)
    new.glt.torsion_gate.data.fill_(1)
    changed=data.clone();changed.torsion_values[n:]+=1
    with torch.no_grad():torch.testing.assert_close(new.glt(data)['token_states'][:b],new.glt(changed)['token_states'][:b],rtol=0,atol=0)

def test_public_pt_heads_rng_saved_initialization_and_trainability():
    common=torch.load(unit.COMMON,map_location='cpu',weights_only=False)['model']
    initial=torch.load(unit.OUT/'initialization/T_FCE_step0.pt',map_location='cpu',weights_only=False)
    for k,v in common.items():assert torch.equal(v,initial['model'][k])
    extra=set(initial['model'])-set(common)
    assert len(extra)==5 and sum(initial['model'][k].numel() for k in extra)==33473
    assert initial['model']['encoder.glt.torsion_gate'].count_nonzero()==0
    args=dict(fusion_mode='token',fusion_revision='glt_fusion_r1',atom_ce_input='fused',geometry_head_norm=True,opt_mode='GLT_FUSION_R1',**GEOMETRY)
    torch.manual_seed(42);old=DualPretrainer(**args);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=DualPretrainer(**args,torsion=True,torsion_revision='glt_physical_torsion_r1')
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k])
    cfg=unit.read(unit.OUT/'initialization/T_FCE.json');package=torch.load(unit.parent('T_FCE'),map_location='cpu',weights_only=False)
    m=unit.TorsionArm(package,'T_FCE',0,cfg)
    from scripts.finetune_mcl_ph import periodic_tdl_trainability,periodic_tdl_optimizer
    periodic_tdl_trainability(m,head_only=True)
    assert all(not p.requires_grad for p in m.model.glt.parameters())
    periodic_tdl_trainability(m,head_only=False)
    opt,_=periodic_tdl_optimizer(m,'xc',head_only=False,profile='uniform_u1')
    torsion_ids={id(p) for p in m.model.glt.torsion_mlp.parameters()}|{id(m.model.glt.torsion_gate)}
    assert all(p.requires_grad for p in m.model.glt.parameters())
    group=next(g for g in opt.param_groups if torsion_ids&{id(p) for p in g['params']})
    assert group['lr']==1e-4 and group['weight_decay']==.02

def test_strict_metadata_legacy_restore_and_complete_frozen_resources():
    cfg=unit.read(unit.OUT/'initialization/T_FCE.json');package=torch.load(unit.parent('T_FCE'),map_location='cpu',weights_only=False)
    m=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',torsion=True,torsion_revision='glt_physical_torsion_r1',**GEOMETRY)
    load_deployment(m,package)
    for key in (*PHYSICAL_TORSION,*S4_RESOURCE,'common_initialization'):
        broken=dict(package);broken.pop(key)
        with pytest.raises(ValueError):load_deployment(m,broken)
    for key,value in [('expected_world_size',4),('accumulation',3),('torsion_revision','unknown'),('torsion_mode','off')]:
        with pytest.raises(ValueError):load_deployment(m,dict(package,**{key:value}))
    broken=dict(package,state_dict=dict(package['state_dict']));broken['state_dict'].pop('glt.torsion_gate')
    with pytest.raises(ValueError):load_deployment(m,broken)
    old=DualPretrainer('token',fusion_revision='glt_fusion_r1',**GEOMETRY)
    load_deployment(build_dual_glt_model('token',fusion_revision='glt_fusion_r1',**GEOMETRY),deployment_package(old,5000))
    frozen=unit.read(unit.OUT/'evaluation_before_test.json')
    assert len(frozen['units'])==frozen['validation_units']==35
    for entry in frozen['units']:
        assert all(entry[k]==v for k,v in {**PHYSICAL_TORSION,**S4_RESOURCE}.items())
        best=torch.load(entry['best'],map_location='cpu',weights_only=False)
        unit.validate_best(best,'T_FCE',entry['task'],entry['fold'],cfg)
        assert best['best_epoch']==entry['best_epoch'] and best['scaler_mean']==entry['scaler']['mean']

def test_two_rank_worker_inline_parameters_and_validation_positions(monkeypatch):
    import src.training.glt_dual_runtime as runtime
    from scripts.evaluate_glt_pt_endpoint import process_validation_positions
    source=SimpleNamespace(samples=[(b'x',None)]*2000)
    source.__class__ # no process, worker, source traversal or training is started
    class Source:
        samples=source.samples
        def __len__(self):return len(self.samples)
    monkeypatch.setattr(runtime,'pretrain_source_fields',lambda *a,**kw:('top','tri','text','static',None))
    seen=[]
    def prepare(*args,**kwargs):seen.append(kwargs);return kwargs['position']
    d=runtime.RankMicrobatchStream(Source(),seed=42,world=2,rank=1,microbatch=84,accumulation=6,start_step=0,max_steps=1,sigma=.03,ratio=.3,
        torsion_mode='on',torsion_revision='glt_physical_torsion_r1',opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')
    d._prepare=prepare;d._collate=lambda rows:rows
    positions=[p for i in range(6) for p in d[i]]
    assert len(positions)==504 and all(x['torsion_revision']=='glt_physical_torsion_r1' and x['torsion_mode']=='on' for x in seen)
    per_rank=[[p for start in range(0,1024,168) for p in process_validation_positions(start,rank,2)] for rank in (0,1)]
    assert [len(p) for p in per_rank]==[512,512] and sorted(per_rank[0]+per_rank[1])==list(range(1024))
    assert not set(per_rank[0])&set(per_rank[1])
    with pytest.raises(ValueError):process_validation_positions(0,2,2)

def test_process_validation_forwards_torsion_and_restores_rng_mode(monkeypatch,tmp_path):
    import scripts.evaluate_glt_pt_endpoint as endpoint
    import src.training.glt_dual_runtime as runtime
    import src.dataset.glt_dual_pretrain as prep
    import torch.distributed as dist
    class Source:
        entries=[dict(pretrain_role='validation')]*1024
        samples=[(b'x',None)]*1024
        closed=False
        def __len__(self):return 1024
        def close(self):self.closed=True
    source=Source();seen=[]
    monkeypatch.setattr(endpoint,'load_endpoint_subset',lambda *a:dict(indices=np.arange(1024)))
    monkeypatch.setattr(runtime,'open_source',lambda *a,**kw:(source,None))
    monkeypatch.setattr(runtime,'pretrain_source_fields',lambda *a,**kw:(None,None,'text',None,None))
    monkeypatch.setattr(runtime,'apply_geometry_quality',lambda s,*a:s)
    monkeypatch.setattr(runtime,'move_labels',lambda x,*a:x)
    def prepare(*a,**kw):seen.append(kw);return kw
    monkeypatch.setattr(prep,'prepare_pretrain_sample',prepare)
    def collate(records):
        n=len(records)
        return Data(bond_length_weight=torch.ones(n),geometry_valid=torch.ones(n,dtype=torch.bool),bond_distance=torch.ones(n),torsion_counts=torch.zeros((n,6),dtype=torch.long)),n
    monkeypatch.setattr(prep,'pretrain_collate',collate)
    monkeypatch.setattr(dist,'barrier',lambda:None);monkeypatch.setattr(dist,'all_reduce',lambda t:t.mul_(2))
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    class Model:
        training=True
        def eval(self):self.training=False
        def train(self,mode):self.training=mode
        def __call__(self,data,n):
            return dict(sums=torch.zeros(3),raw_geometry_sums=torch.zeros(2),extra_sums=torch.zeros(2),counts=torch.tensor([n,n,n]),
                angle_graphs=torch.tensor(n),extra_counts=torch.tensor([n,n]),targets=torch.tensor([n,n,n]))
    cfg=dict(unit.make_config(),cycle_root=str(tmp_path));m=Model();rng=torch.get_rng_state().clone()
    endpoint.evaluate_process_validation(m,cfg,1000,'cpu',0,2)
    assert m.training and source.closed and torch.equal(rng,torch.get_rng_state())
    assert len(seen)==512 and all(r['torsion_mode']=='on' and r['torsion_revision']=='glt_physical_torsion_r1' for r in seen)
    assert unit.read(tmp_path/'pt_validation/T_FCE/step01000/metrics.json')['presented']==1024
    with pytest.raises(ValueError):endpoint.evaluate_process_validation(m,cfg,1000,'cpu',0,4)

def test_scheduler_gpu_isolation_no_extra_gpu_and_cpu_environment(monkeypatch,tmp_path):
    from scripts import run_glt_pt_torsion as controller
    assert controller.GPUS==(2,3) and __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
    monkeypatch.setattr(controller,'LOG',tmp_path);monkeypatch.setattr(controller,'ACTIVE',[])
    calls=[];monkeypatch.setattr(controller.subprocess,'run',lambda args,**kw:calls.append(args))
    for gpu in (2,3):
        controller.launch('ft_'+str(gpu),['python','scripts/run_glt_pt_torsion_unit.py'],[gpu])
        env=unit.read(tmp_path/('ft_'+str(gpu))/'environment.json')
        assert env['CUDA_VISIBLE_DEVICES']==str(gpu) and env['logical_device']=='cuda:0'
    controller.launch('pt_T_FCE',['torchrun','--nproc_per_node=2'],[2,3])
    assert unit.read(tmp_path/'pt_T_FCE/environment.json')['logical_device']=='cuda:LOCAL_RANK'
    controller.launch('final',['python','-m','pytest'],limit=300)
    assert unit.read(tmp_path/'final/environment.json')['CUDA_VISIBLE_DEVICES']==''
    for devices in ([0],[1],[0,1],[2,3,0],[2,3]):
        with pytest.raises(ValueError):controller.launch('ft_bad',['python'],devices)
    assert len(calls)==4

def test_freeze_refuses_partial_matrix_and_threshold_uses_full_precision(monkeypatch,tmp_path):
    monkeypatch.setattr(select,'OUT',tmp_path)
    unit.write(tmp_path/'initialization/T_FCE.json',unit.make_config())
    monkeypatch.setattr(select,'ft_folder',lambda arm,task,fold:tmp_path/'finetune'/task/f'fold{fold}')
    with pytest.raises(FileNotFoundError):select.freeze()
    assert not (tmp_path/'evaluation_before_test.json').exists()
    assert not select.qualifies(.7955913245367517)
    assert select.qualifies(.7955913245367518)
