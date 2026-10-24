"""One final synthetic CPU suite for v3/r2 correctness; no campaign starts."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest
import torch
from src.modules.ua2d_galformer_pretrain import (PretrainConfig,UA2DGalformerPretrainer,
    balanced_fp_loss,objective,alignment_valid,deployment_package,restore_downstream)
from src.dataset.ua2d_galformer_pretrain import prepare_sample,collate_samples
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from tests.test_dual_glt import _toy_pair
from src.training import ua2d_pretrain_fp as campaign


def example(config,text='*COC*',position=0,fp=None):
    top=build_canonical_periodic_topology(text);_,tri=_toy_pair(text)
    return prepare_sample(top,tri,text,config=config,seed=42,position=position,fp_target=fp)


def test_schedule_units_and_fp_balance():
    from scripts.pretrain_ua2d_galformer import learning_rate
    assert learning_rate(499)==2e-4
    assert learning_rate(4999)==1e-6
    assert learning_rate(0)==pytest.approx(2e-4/500)
    for cfg in [dict(strategy='mask_align_geometry',noise_sigma=0),dict(coordinate_unit='raw'),dict(fusion='concat')]:
        with pytest.raises(ValueError):PretrainConfig(**cfg)
    target=torch.zeros(2,2048);target[0,:2]=1;target[1,:100]=1
    z=torch.zeros_like(target,requires_grad=True)
    torch.testing.assert_close(balanced_fp_loss(z,target),torch.full((2,),np.log(2),dtype=torch.float32))
    bad=torch.full_like(target,-20)
    assert (balanced_fp_loss(bad,target)>9).all()
    balanced_fp_loss(z,target).mean().backward();assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError):balanced_fp_loss(z,torch.zeros_like(target))


def test_fp_initialization_rng_common_dropout_and_clean_views():
    for strategy in ('mask_align','mask_align_geometry'):
        cfg=PretrainConfig(strategy=strategy)
        torch.manual_seed(42);a=UA2DGalformerPretrainer(cfg);rng=torch.get_rng_state().clone()
        torch.manual_seed(42);b=UA2DGalformerPretrainer(replace(cfg,fp_enabled=True))
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        for step,weight in [(0,0.),(500,.05),(1000,.1),(4999,.1)]:
            b.completed_updates=step;assert b.weights['contrast']==weight and b.weights['fp']==weight
        fp=np.zeros(2048,dtype=np.uint8);fp[[1,5]]=1
        x,y=example(cfg);xx,yy=example(replace(cfg,fp_enabled=True),fp=fp)
        for field in x.keys():
            if torch.is_tensor(x[field]):assert torch.equal(x[field],xx[field])
        for k,v in y.items():
            if torch.is_tensor(v):assert torch.equal(v,yy[k])
        assert torch.equal(yy['fp_target'],torch.tensor(fp,dtype=torch.float32))
        assert 'fp_target' not in xx
        del a,b


def test_coordinate_normalization_zero_start_and_nontrivial_equivariance():
    cfg=PretrainConfig(strategy='mask_align_geometry',dropout=0.)
    data,labels=collate_samples([example(cfg),example(cfg,'*CCN*',1)])
    model=UA2DGalformerPretrainer(cfg).eval();model.completed_updates=1000
    assert torch.count_nonzero(model.coordinate_head[-1].weight)==0
    bonds=torch.randn(len(data.bond_distance),512)
    torch.testing.assert_close(model.coordinate_prediction(bonds,labels,2),torch.zeros_like(labels['physical_pos']))
    result=model(data,labels);mask=labels['physical_center']
    per=labels['coordinate_target'][mask].square().mean(-1)/(.03**2)
    expected=sum(per[labels['physical_graph'][mask]==i].mean() for i in range(2))
    torch.testing.assert_close(result['sums']['coordinate'],expected)
    with torch.no_grad():model.coordinate_head[-1].weight.normal_(0,.01)
    pred=model.coordinate_prediction(bonds,labels,2)
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    actual=model.coordinate_prediction(bonds,dict(labels,physical_pos=labels['physical_pos']@rotation+7),2)
    torch.testing.assert_close(actual,pred@rotation,atol=1e-5,rtol=2e-4)


def test_fp_gradients_and_forward_loss_invariance():
    cfg=PretrainConfig(strategy='mask_align_geometry',fp_enabled=True,dropout=0.)
    fp=np.zeros(2048,dtype=np.uint8);fp[[0,30,1984]]=1
    data,labels=collate_samples([example(cfg,fp=fp),example(cfg,'*CCN*',1,fp)])
    model=UA2DGalformerPretrainer(cfg).eval();model.completed_updates=1000
    result=model(data,labels)
    count=model.target_counts(data,labels)
    for k in count:torch.testing.assert_close(count[k],result['counts'][k])
    result['sums']['fp'].backward()
    for module in (model.o8,model.norm2,model.fp_head):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    assert all(p.grad is None for p in model.encoder3d.parameters())
    off=UA2DGalformerPretrainer(replace(cfg,fp_enabled=False)).eval();off.completed_updates=1000
    off.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('fp_head.')},strict=True)
    with torch.no_grad():other=off(data,{k:v for k,v in labels.items() if k!='fp_target'})
    for k in other['sums']:torch.testing.assert_close(result['sums'][k],other['sums'][k])
    assert not hasattr(off,'fp_head')


def test_morgan_dummy_bit_order_and_cache_binding(tmp_path):
    from src.training.xc_kernel import generator,fingerprint
    cfg=PretrainConfig();top=build_canonical_periodic_topology('*COC*');text=str(top.normalized_canonical_smiles)
    fp,diagnostics=fingerprint(text,generator());bits=np.asarray(list(fp),dtype=np.uint8)
    assert diagnostics['dummy_atoms']==2
    assert np.array_equal(np.unpackbits(np.packbits(bits,bitorder='little'),bitorder='little'),bits)
    db=sqlite3.connect(tmp_path/'labels.sqlite')
    db.execute('CREATE TABLE fp(normalized TEXT PRIMARY KEY,bits BLOB)');db.execute('CREATE TABLE rows(source_row INTEGER PRIMARY KEY,normalized TEXT,source_smiles TEXT)')
    db.execute('INSERT INTO fp VALUES (?,?)',(text,np.packbits(bits,bitorder='little').tobytes()))
    db.execute('INSERT INTO rows VALUES (?,?,?)',(17,text,'*COC*'));db.commit();db.close()
    campaign.write(tmp_path/'metadata.json',dict(status='COMPLETE',source='source',index='index',spec=campaign.FP_SPEC,rows=911391))
    reader=campaign.FPReader(tmp_path,'source','index')
    assert np.array_equal(reader.get(17,'*COC*',top),bits)
    with pytest.raises(ValueError):reader.get(18,'*COC*',top)
    with pytest.raises(ValueError):reader.get(17,'*CCN*',top)
    reader.close()


def test_strict_pure2d_and_auxiliary_removal():
    from src.modules.glt_dual import Pure2DGLTModel
    cfg=PretrainConfig(fp_enabled=True)
    model=UA2DGalformerPretrainer(cfg);package=deployment_package(model,0)
    pure=restore_downstream(package,Pure2DGLTModel().head.state_dict()).eval()
    assert not hasattr(pure,'fp_head') and not hasattr(pure,'encoder3d')
    with pytest.raises(ValueError):restore_downstream(dict(package,schema='ua2d_galformer_objectives_v2'),pure.head.state_dict())
    data,_=collate_samples([example(replace(cfg,fp_enabled=False))])
    # A downstream graph needs no auxiliary labels or 3D fields in its forward.
    from src.dataset.glt_dual import glt_2d_collate,build_glt_2d_sample
    top=build_canonical_periodic_topology('*COC*');graph=glt_2d_collate([build_glt_2d_sample(top,'*COC*')])
    with torch.no_grad():assert pure(graph).shape==(1,1)


def distributed_worker(rank,rendezvous):
    import torch.distributed as dist
    from src.modules.glt_dual_pretrain import alignment_loss,alignment_local_anchor_count
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        # Unequal lengths, duplicate identities, rank with no valid geometry.
        for no_local in (False,True):
            n=2 if rank==0 else 1
            ids=['repeat','other'] if rank==0 else ['repeat']
            valid=torch.ones(n,dtype=torch.bool)
            if no_local and rank==1:valid[:]=False
            torch.manual_seed(11+rank)
            z2=torch.nn.functional.normalize(torch.randn(n,128),dim=-1).detach().requires_grad_()
            z3=torch.nn.functional.normalize(torch.randn(n,128),dim=-1).detach().requires_grad_()
            total,count=alignment_loss(z2,z3,ids,valid)
            expected=alignment_local_anchor_count(ids,valid,device='cpu')
            torch.testing.assert_close(count,expected)
            denom=count.clone();dist.all_reduce(denom)
            loss=objective(dict(sums={'contrast':2*total},counts={'contrast':count}),{'contrast':.1},{'contrast':2*denom})
            loss.backward()
            assert z2.grad is not None and z3.grad is not None
            assert torch.isfinite(z2.grad).all() and torch.isfinite(z3.grad).all()
            if not (no_local and rank==1):assert z2.grad.abs().sum()>0 and z3.grad.abs().sum()>0
        # Rank1 only supplies candidates: verify remote candidate gradient path.
        z2=torch.nn.functional.normalize(torch.randn(1,128),dim=-1).detach().requires_grad_()
        z3=torch.nn.functional.normalize(torch.randn(1,128),dim=-1).detach().requires_grad_()
        value,_=alignment_loss(z2,z3,['a' if rank==0 else 'b'],torch.tensor([True]))
        (value*(1 if rank==0 else 0)).backward()
        if rank==1:assert z2.grad.abs().sum()>0 and z3.grad.abs().sum()>0
    finally:dist.destroy_process_group()


def test_cpu_two_rank_collectives_and_remote_gradients(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(distributed_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_budget_freezes_and_development_separation(tmp_path,monkeypatch):
    counts={k:0 for k in campaign.LIMITS};campaign.check_limits(counts)
    counts['ft_starts']=106
    with pytest.raises(RuntimeError):campaign.check_limits(counts)
    path=tmp_path/'frozen.json';campaign.freeze(path,{'frozen':True})
    with pytest.raises(FileExistsError):campaign.freeze(path,{'frozen':False})
    # No source can be opened for TEST before a final freeze exists.
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    with pytest.raises(FileNotFoundError):campaign.Data('eea',0,formal=True)


def test_recovery_preserves_failed_budget_and_uses_a_distinct_v1_parent(tmp_path,monkeypatch):
    budget=dict(plan=campaign.PLAN,revision='r2',limits=dict(campaign.R2_LIMITS),used=dict(campaign.R2_USED),jobs=[
        dict(name='pretrain_fp_initialization',kind='initialize',exit=0),
        dict(name='pretrain_fp_pt_V1',kind='pt',exit=1),
        dict(name='pretrain_fp_final_cpu',kind='cpu',exit=1)])
    inherited=campaign.inherited_recovery_jobs(budget)
    inherited[1]['exit']=0
    assert budget['jobs'][1]['exit']==1
    assert campaign.LIMITS['pt_starts']==budget['used']['pt_starts']+3
    assert campaign.LIMITS['pt_updates']==budget['used']['pt_updates']+15000
    assert campaign.LIMITS['pt_presentations']==budget['used']['pt_presentations']+15120000
    assert campaign.LIMITS['cpu_suites']==budget['used']['cpu_suites']+1
    with pytest.raises(ValueError):
        campaign.inherited_recovery_jobs(dict(budget,used=dict(budget['used'],pt_starts=0)))
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    parents={a:str(tmp_path/'pretrain'/('V1_attempt2' if a=='V1' else a)/'deploy.pt') for a in campaign.ARMS}
    campaign.freeze(tmp_path/'parent_bindings.json',dict(plan=campaign.PLAN,revision=campaign.REVISION,parents=parents))
    assert campaign.unit_config('V1','eea',0)['parent']==parents['V1']
    assert campaign.parent_path('V1')!=tmp_path/'pretrain/V1/deploy.pt'


def test_formal_aggregate_reads_training_scaler_without_a_formal_copy(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    monkeypatch.setattr(campaign,'TASKS',('eea',))
    monkeypatch.setattr(campaign,'unit_config',lambda arm,task,fold:dict(arm=arm,task=task,fold=fold))
    scaler=dict(mean=[3.],scale=[2.],fit='train',transform='standard')
    for fold in range(5):
        formal=campaign.unit_path('V1','eea',fold,formal=True)
        campaign.write(formal/'metrics.json',dict(status='COMPLETE',config=campaign.unit_config('V1','eea',fold),r2=.42,best_epoch=12))
        campaign.write(formal/'consumption.json',dict(status='COMPLETE'))
        campaign.write(campaign.unit_path('V1','eea',fold)/'scaler.json',scaler)
        assert not (formal/'scaler.json').exists()
    actual=campaign.aggregate('V1',formal=True)
    assert actual['macro']==pytest.approx(.42)
    assert all(unit['scaler']==scaler for unit in actual['units'].values())
