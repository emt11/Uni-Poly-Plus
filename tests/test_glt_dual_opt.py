"""Single final CPU invocation for GLT optimization correctness."""
import torch
from src.modules.glt_dual_opt import BondTokenFusion,incident_pool,weighted_graph_loss
from src.modules.glt_dual import build_dual_glt_model,element_index,triplet_type
from src.modules.glt_dual_pretrain import load_deployment
import pytest


def test_incident_weights_and_empty():
    s=torch.tensor([[2.],[6.],[100.]])
    # First central atom sees internal and seam; second sees only internal.
    graph,valid=incident_pool(s,torch.tensor([[0,1,0],[0,0,1]]),torch.tensor([0,0,1]),2)
    assert torch.equal(graph,torch.tensor([[3.],[0.]]));assert valid.tolist()==[True,False]
    with pytest.raises(ValueError):incident_pool(s,torch.tensor([[9],[0]]),torch.tensor([0]),1)


def test_weighted_length_denominator():
    loss,valid=weighted_graph_loss(torch.tensor([4.,16.,999.]),torch.tensor([1.,.5,0.]),torch.tensor([0,0,1]),2)
    assert torch.equal(loss,torch.tensor([8.,0.]));assert valid.tolist()==[True,False]


def test_fusion_isolation_invalid_and_gradients():
    torch.manual_seed(42);fusion=BondTokenFusion()
    atoms=torch.randn(5,512,requires_grad=True);bonds=torch.randn(6,512,requires_grad=True)
    ai=torch.tensor([0,0,1,1,2]);bi=torch.tensor([0,0,0,1,1,2]);valid=torch.tensor([True,True,False])
    original=fusion(atoms,bonds,ai,bi,valid)
    changed=bonds.detach().clone();changed[bi==1]+=7
    alternate=fusion(atoms,changed,ai,bi,valid)
    torch.testing.assert_close(original[ai==0],alternate[ai==0]);torch.testing.assert_close(original[ai==2],atoms[ai==2])
    original.square().mean().backward()
    assert fusion.q.weight.grad.norm()>0 and fusion.k.weight.grad.norm()>0
    assert bonds.grad[bi!=2].norm()>0 and bonds.grad[bi==2].norm()==0


def test_unknown_derived_element_categories():
    assert element_index(torch.tensor([0,6])).tolist()==[100,5]
    assert triplet_type(torch.tensor([0]),torch.tensor([6]),torch.tensor([1]))!=triplet_type(torch.tensor([8]),torch.tensor([6]),torch.tensor([1]))


def test_deployment_consumes_modes_and_rejects_missing():
    model=build_dual_glt_model('token',dropout=0)
    state={k:v for k,v in model.state_dict().items() if not k.startswith('predictor.')}
    package=dict(architecture=model.architecture_name,fusion_mode='token',step=5000,use_md200=False,readout='internal',state_dict=state)
    load_deployment(model,package)
    bad=dict(package,state_dict=dict(state));bad['state_dict'].pop('token_fusion.q.weight')
    with pytest.raises(ValueError):load_deployment(model,bad)
    with pytest.raises(ValueError):load_deployment(model,dict(package,readout='incident'))
    assert build_dual_glt_model().fusion_mode=='concat'


def test_physical_batch_seam_targets_and_copy_mask():
    from tests.test_dual_glt import sample
    from src.dataset.glt_dual import dual_glt_collate
    from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
    top,tri,clean=sample('*COC*')
    batch=dual_glt_collate([clean,clean])
    b,a=batch.incidence_index
    assert torch.equal(batch.bond_batch[b],batch.canonical_graph_index[a])
    assert (clean.bond_length_weight==.5).any()
    f,fl=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='toy',position=0,opt_mode='F_TOKEN')
    m,ml=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='toy',position=0,opt_mode='M_TOKEN_CE')
    for field in ('bond_z_a','bond_z_b','bond_distance','line_angle'):
        assert torch.equal(getattr(f,field),getattr(m,field))
    mask=fl['atom_mask']
    assert (f.bond_z_a[mask[f.bond_canonical_a]]==0).all()
    assert (f.bond_z_b[mask[f.bond_canonical_b]]==0).all()
    # Count original physical copies, not just masked token projections.
    for atom in torch.where(mask)[0]:
        endpoints=(f.bond_canonical_a==atom)|(f.bond_canonical_b==atom)
        assert int(endpoints.sum())>=3
    g,gl=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='toy',position=0,opt_mode='G_SEAM_GEO')
    assert gl['distance'].numel()>int(clean.bond_center.sum())
    assert gl['angle_pairs'].size(0)>0
    for left,right in gl['angle_pairs'].tolist():
        rows=(clean.line_path[:,0]==left)&(clean.line_path[:,1]==right)&~clean.line_is_self
        assert clean.line_center_angle[rows].all()
    assert not (gl['angle_pairs'][:,0]==gl['angle_pairs'][:,1]).any()


def test_fused_atom_supervision_gradient():
    from tests.test_dual_glt import sample
    from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
    from src.modules.glt_dual_pretrain import DualPretrainer
    top,tri,_=sample('*COC*')
    batch,labels=pretrain_collate([prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='toy',position=0,opt_mode='M_TOKEN_CE')])
    torch.manual_seed(42)
    model=DualPretrainer('token',opt_mode='M_TOKEN_CE')
    result=model(batch,labels)
    result['sums'][0].backward()
    assert model.encoder.token_fusion.q.weight.grad.norm()>0
    assert model.encoder.token_fusion.k.weight.grad.norm()>0
    assert model.encoder.glt.endpoint.weight.grad.norm()>0
