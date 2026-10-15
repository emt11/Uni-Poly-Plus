from types import SimpleNamespace
import torch
from scripts.check_mcl_ph_r2_batch import copy_baseline


def test_copy_baseline_exact_lengths_angles_nonbonds_and_empty_targets():
    data=SimpleNamespace(graph_available=torch.tensor([True]),mcl_pos=torch.tensor([[0.,0,0],[1.,0,0],[0.,1,0]]),mcl_bond_index=torch.tensor([[0,1],[0,2]]))
    statistics={'length':{'mu':0.,'sigma':1.},'nonbond':{'mu':0.,'sigma':1.}}
    labels=dict(mcl_length_pair=torch.tensor([[0,1],[0,2]]),mcl_angle_pair=torch.tensor([[0,1]]),mcl_nonbond_pair=torch.tensor([[1,2]]),mcl_length_target=torch.log1p(torch.ones(2)),mcl_angle_target=torch.zeros(1),mcl_nonbond_target=torch.log1p(torch.tensor([2.]).sqrt()),mcl_length_graph=torch.zeros(2,dtype=torch.long),mcl_angle_graph=torch.zeros(1,dtype=torch.long),mcl_nonbond_graph=torch.zeros(1,dtype=torch.long),mcl_nonbond_slot=torch.ones(1,dtype=torch.long))
    result=copy_baseline(data,labels,statistics)
    assert result['geometry']==0 and result['geometry_graphs']==1
    for key in labels: labels[key]=labels[key][:0]
    empty=copy_baseline(data,labels,statistics)
    assert empty['geometry']==0 and empty['geometry_graphs']==0


def test_attention_timer_uses_same_states_with_bf16_gradient_gate():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from src.modules.mcl_ph import FusionCrossAttention
    from scripts.check_mcl_ph_r2_batch import attention_timing
    torch.manual_seed(42)
    graph=torch.repeat_interleave(torch.arange(3,device='cuda'),torch.tensor([2,3,5],device='cuda'))
    data=SimpleNamespace(canonical_graph_index=graph,mcl_readout_valid=torch.tensor([True,False,True],device='cuda'),mcl_central_index=torch.arange(10,device='cuda'))
    model=SimpleNamespace(encoder=SimpleNamespace(fusion=FusionCrossAttention().cuda()))
    report=dict(atom_states=torch.randn(10,512,device='cuda'),mixed=torch.randn(10,128,device='cuda'))
    result=attention_timing(model,data,report)
    assert len(result['results'])==2 and result['repeats']==3
    assert max(result['bf16_relative_L2_errors'])<=2*torch.finfo(torch.bfloat16).eps
