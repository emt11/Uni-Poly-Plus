"""Only new endpoint reduction/reading and FT trainability/optimizer correctness."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.evaluate_glt_pt_endpoint import component_means, endpoint_positions, load_endpoint_subset, restore_endpoint
from scripts.finetune_mcl_ph import periodic_tdl_trainability, periodic_tdl_optimizer, periodic_tdl_head


class SmallArm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.o8=nn.Module();self.model.o8.layers=nn.ModuleList([nn.Linear(2,2) for _ in range(6)])
        self.model.glt=nn.Module();self.model.glt.layers=nn.ModuleList([nn.Linear(2,2) for _ in range(6)])
        self.model.norm2=nn.LayerNorm(2);self.model.norm3=nn.LayerNorm(2)
        self.model.embedding=nn.Embedding(3,2)
        self.model.predictor=nn.Sequential(nn.Linear(4,2),nn.GELU(),nn.Dropout(.1),nn.Linear(2,1))


@pytest.mark.parametrize('candidate',['FT_PARTIAL','FT_LOW_LR','FT_REG'])
@pytest.mark.parametrize('task',['eea','egc','xc'])
def test_actual_optimizer_members_and_single_factor_values(candidate,task):
    model=SmallArm()
    decay={'eea':.02,'egc':0.,'xc':.0001}[task]
    for head_only in (True,False,True,False):
        periodic_tdl_trainability(model,head_only=head_only,ft_candidate=candidate)
        optimizer,evidence=periodic_tdl_optimizer(model,task,head_only=head_only,ft_candidate=candidate)
        optimized=[p for group in optimizer.param_groups for p in group['params']]
        assert len({id(p) for p in optimized})==len(optimized)
        assert {id(p) for p in optimized}=={id(p) for p in model.parameters() if p.requires_grad}
        head={id(p) for p in periodic_tdl_head(model).parameters()}
        if head_only:assert {id(p) for p in optimized}==head
        elif candidate=='FT_PARTIAL':
            expected=head|{id(p) for m in [*model.model.o8.layers[4:],*model.model.glt.layers[4:],model.model.norm2,model.model.norm3] for p in m.parameters()}
            assert {id(p) for p in optimized}==expected
        else:assert len(optimized)==len(list(model.parameters()))
        for group in optimizer.param_groups:
            if group['name']=='head':
                assert group['lr']==(3e-4 if head_only else 1e-3)
                assert group['weight_decay']==decay
            else:
                assert group['lr']==pytest.approx((2e-4 if task=='xc' else 1e-4)*(.3 if candidate=='FT_LOW_LR' else 1))
                assert group['weight_decay']==pytest.approx(decay+(.01 if candidate=='FT_REG' else 0))
        assert [g['weight_decay'] for g in evidence]==[g['weight_decay'] for g in optimizer.param_groups]


def test_partial_frozen_weights_stay_unchanged_after_adamw_step():
    model=SmallArm();periodic_tdl_trainability(model,head_only=False,ft_candidate='FT_PARTIAL')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer,_=periodic_tdl_optimizer(model,'eea',head_only=False,ft_candidate='FT_PARTIAL')
    sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward();optimizer.step()
    for n,p in model.named_parameters():
        if not p.requires_grad:assert torch.equal(before[n],p)
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad)


def test_endpoint_sums_use_counts_and_empty_components_not_batch_means():
    first_s=torch.tensor([6.,0.]);first_c=torch.tensor([3.,0.])
    second_s=torch.tensor([20.,0.]);second_c=torch.tensor([2.,0.])
    assert component_means(first_s+second_s,first_c+second_c)==[5.2,None]
    with pytest.raises(ValueError):component_means(torch.tensor([1.]),torch.tensor([0.]))
    with pytest.raises(ValueError):component_means(torch.tensor([float('nan')]),torch.tensor([1.]))


def test_endpoint_1024_presentations_once_no_tail_padding():
    pools=[[i for r in range(4) for i in endpoint_positions(start,r)] for start in (0,336,672,1008)]
    assert [len(p) for p in pools]==[336,336,336,16]
    assert sorted(i for p in pools for i in p)==list(range(1024))


def test_endpoint_reads_fixed_pval_and_rejects_training_overlap(tmp_path):
    summary=tmp_path/'summary.json';summary.write_text(json.dumps(dict(schema='mcl-ph-next-direct-index-v1',pi1m_rows=1027,pretrain_train_count=3,pretrain_validation_count=1024,pretrain_fixed_validation_count=1024,excluded_overlap=0)))
    values=np.arange(3,1027,dtype=np.int64)
    np.savez(tmp_path/'pretrain_indices.npz',train_source_indices=np.arange(3,dtype=np.int64),validation_source_indices=values,fixed_validation_source_indices=values)
    assert np.array_equal(load_endpoint_subset(summary)['indices'],values)
    np.savez(tmp_path/'pretrain_indices.npz',train_source_indices=np.array([0,1,3],dtype=np.int64),validation_source_indices=values,fixed_validation_source_indices=values)
    with pytest.raises(ValueError,match='overlap'):load_endpoint_subset(summary)


def test_endpoint_strict_full_heads_and_readonly_model_mode():
    model=nn.Sequential(nn.Linear(2,2),nn.Dropout(.1),nn.Linear(2,1))
    cfg=dict(global_batch=1008)
    state=dict(step=5000,next_position=5040000,identity=dict(config=cfg),model=model.state_dict())
    restore_endpoint(model,state,cfg)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    with torch.no_grad():assert not model(torch.ones(1,2)).requires_grad
    missing=dict(state,model={k:v for k,v in state['model'].items() if k!='2.weight'})
    with pytest.raises(RuntimeError):restore_endpoint(model,missing,cfg)
    with pytest.raises(ValueError):restore_endpoint(model,dict(state,step=4999),cfg)
