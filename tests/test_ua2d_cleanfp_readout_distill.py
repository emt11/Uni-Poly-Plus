"""One terminal synthetic CPU suite; no corpus, GPU or campaign execution."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.modules.ua2d_cleanfp_pretrain import (CleanConfig,CleanPretrainer,CleanPure2D,QueryReadout,
    local_targets,local_loss,deployment_package,restore_downstream,objective)
from src.modules.glt_dual import mean_pool
from src.dataset.ua2d_cleanfp_pretrain import paired_sample,collate_pairs,decode_clean_batch
from src.dataset.ua2d_fast_pretrain import build_template,encode
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static
from src.training.ua2d_readout_distill import LossBalancer
from tests.test_dual_glt import _toy_pair


def batch(config):
    records=[]
    for i,text in enumerate(('*COC*','*CCN*')):
        top=build_canonical_periodic_topology(text);_,tri=_toy_pair(text)
        static=build_dual_static(top,tri,text)
        template=build_template(top,tri,text,static)
        before=copy.deepcopy(template['clean'].to_dict())
        fp=np.zeros(2048,dtype=np.uint8);fp[[1,3,7]]=1
        record=paired_sample(template,config,i,fp)
        for k,v in before.items():
            if torch.is_tensor(v):torch.testing.assert_close(template['clean'][k],v,rtol=0,atol=0)
        assert torch.equal(record[2].mips_x,before['mips_x'])
        assert torch.equal(record[2].bond_distance,before['bond_distance'])
        records.append(record)
    return collate_pairs(records)


def test_query_initial_equivalence_permutation_and_cross_graph_isolation():
    torch.manual_seed(1);q=QueryReadout()
    x=torch.randn(7,512);ids=torch.tensor([0,0,0,1,1,1,1])
    torch.testing.assert_close(q(x,ids,2),mean_pool(x,ids,2),rtol=0,atol=0)
    q.alpha.data.fill_(.3)
    base=q(x,ids,2);order=torch.tensor([2,0,1,5,6,3,4])
    torch.testing.assert_close(q(x[order],ids[order],2),base)
    other=x.clone();other[ids==1]+=10
    torch.testing.assert_close(q(other,ids,2)[0],base[0])


def test_balancer_timing_limits_missing_tasks_and_restoration():
    names=['mask_2d','mask_3d','contrast','fp']
    one=LossBalancer(names,'ALL1');dynamic=LossBalancer(names,'DYN')
    counts=dict.fromkeys(names,1)
    for i in range(199):
        losses=dict(zip(names,[5/(1+.01*i),10/(1+.002*i),4.,.7]))
        one.observe(losses,counts);dynamic.observe(losses,counts)
        assert list(dynamic.weights.values())==[1.]*4
    dynamic.observe(losses,dict(counts,fp=0));assert dynamic.observations==199
    dynamic.observe(losses,counts);assert dynamic.adjustments==1
    assert dynamic.weights['fp']>dynamic.weights['mask_3d']
    for i in range(300):dynamic.observe(dict(zip(names,[.01,20.,.001,.8])),counts)
    weights=np.array(list(dynamic.weights.values()))
    assert weights.min()>=.25 and weights.max()<=4 and abs(weights.sum()-4)<1e-8
    assert list(one.weights.values())==[1.]*4
    restored=LossBalancer(names,'DYN');restored.load_state_dict(dynamic.state_dict())
    restored.observe(losses,counts);dynamic.observe(losses,counts)
    assert restored.state_dict()==dynamic.state_dict()


def test_clean_pair_roundtrip_and_common_initialization_rng():
    config=CleanConfig(dropout=0.)
    masked,labels,clean=batch(config)
    a,b,c,trace=decode_clean_batch(encode((masked.to_dict(),labels,clean.to_dict(),dict(micro_index=3))))
    assert trace['micro_index']==3
    for x,y in [(a,masked),(c,clean)]:
        for key in y.keys():
            if torch.is_tensor(y[key]):torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
    torch.manual_seed(42);g=CleanPretrainer(config);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);q=CleanPretrainer(replace(config,readout='query',local=True))
    assert torch.equal(torch.get_rng_state(),rng)
    for key,v in g.state_dict().items():torch.testing.assert_close(q.state_dict()[key],v,rtol=0,atol=0)


def test_clean_fp_does_not_read_masked_states_or_3d_and_counts_match():
    config=CleanConfig(dropout=0.);masked,labels,clean=batch(config)
    model=CleanPretrainer(config).eval()
    result=model(masked,labels,clean)
    poisoned=masked.clone();poisoned.mips_x.zero_()
    altered=model(poisoned,labels,clean)
    for name in ('fp','contrast'):torch.testing.assert_close(result['sums'][name],altered['sums'][name])
    expected=model.target_counts(masked,labels)
    for name in expected:torch.testing.assert_close(result['counts'][name],expected[name])
    result['sums']['fp'].backward()
    assert model.o8.atom_embedding.projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.encoder3d.parameters())


def test_local_mapping_molecule_reduction_and_stop_gradient():
    config=CleanConfig(readout='query',local=True,dropout=0.)
    masked,labels,clean=batch(config);model=CleanPretrainer(config).eval()
    result=model(masked,labels,clean)
    expected=model.target_counts(masked,labels)
    for name in expected:torch.testing.assert_close(result['counts'][name],expected[name])
    bonds=torch.randn(len(clean.bond_distance),512,requires_grad=True)
    atoms=torch.randn(len(clean.mips_x),512,requires_grad=True)
    target,valid=local_targets(bonds,clean)
    value,count,_=local_loss(atoms,bonds,clean,model.local_predictor)
    similarity=(F.normalize(model.local_predictor(atoms[valid]),dim=-1)*F.normalize(target[valid].detach(),dim=-1)).sum(-1)
    ids=clean.canonical_graph_index[valid]
    manual=sum((1-similarity[ids==i]).clamp_min(0).mean() for i in range(2))
    torch.testing.assert_close(value,manual);assert count==2
    value.backward();assert atoms.grad.abs().sum()>0 and bonds.grad is None
    result['sums']['local'].backward(retain_graph=True)
    assert all(p.grad is None for p in model.encoder3d.parameters())
    result['sums']['contrast'].backward()
    assert model.encoder3d.endpoint.weight.grad.abs().sum()>0
    wrong=clean.clone();wrong.incidence_index=clean.incidence_index.clone()
    wrong.incidence_index[1,0]=int((clean.canonical_graph_index==1).nonzero()[0])
    with pytest.raises(ValueError,match='cross-graph'):local_targets(bonds.detach(),wrong)
    empty=clean.clone();empty.incidence_index=torch.empty((2,0),dtype=torch.long)
    zero,n,_=local_loss(atoms,bonds,empty,model.local_predictor)
    assert n==0 and zero==0


def test_restore_retains_query_and_uses_only_2d():
    config=CleanConfig(readout='query',dropout=0.)
    model=CleanPretrainer(config).eval();model.readout.alpha.data.fill_(.4)
    head=CleanPure2D('query').head.state_dict()
    package=deployment_package(model,5000)
    restored=restore_downstream(package,head).eval()
    from src.dataset.glt_dual import build_glt_2d_sample,glt_2d_collate
    data=glt_2d_collate([build_glt_2d_sample(build_canonical_periodic_topology('*COC*'),'*COC*')])
    assert not hasattr(data,'bond_distance') and not hasattr(restored,'encoder3d')
    prediction=restored(data);assert prediction.shape==(1,1) and torch.isfinite(prediction).all()
    torch.testing.assert_close(restored.readout.alpha,model.readout.alpha)
    with pytest.raises(ValueError):restore_downstream(dict(package,schema='wrong'),head)
    from scripts.finetune_mcl_ph import periodic_tdl_trainability,periodic_tdl_optimizer
    periodic_tdl_trainability(restored,head_only=True)
    assert all(not p.requires_grad for p in restored.readout.parameters())
    periodic_tdl_trainability(restored,head_only=False)
    opt,_=periodic_tdl_optimizer(restored,'eea',head_only=False,profile='uniform_u1')
    assert {id(p) for p in restored.readout.parameters()} <= {id(p) for p in opt.param_groups[0]['params']}


def _distributed_window(rank,rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.modules.glt_dual_pretrain import alignment_loss
    from contextlib import nullcontext
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(8);module=nn_pair= torch.nn.Linear(3,128,bias=False)
        wrapped=DDP(module)
        xs=[torch.arange(12,dtype=torch.float32).reshape(4,3)/10+i for i in range(2)]
        ys=[x.flip(0)+.3 for x in xs]
        for i in range(2):
            with wrapped.no_sync() if i==0 else nullcontext():
                z=F.normalize(wrapped(torch.cat([xs[i][rank*2:rank*2+2],ys[i][rank*2:rank*2+2]])),dim=-1)
                value,count=alignment_loss(z[:2],z[2:],[str(rank*2),str(rank*2+1)],torch.ones(2,dtype=torch.bool))
                loss=objective(dict(sums=dict(contrast=2*value),counts=dict(contrast=count)),dict(contrast=1.),dict(contrast=torch.tensor(8.)))
                loss.backward()
        grad=module.weight.grad.clone()
    finally:dist.destroy_process_group()
    reference=torch.nn.Linear(3,128,bias=False);reference.load_state_dict(module.state_dict())
    total=0
    for x,y in zip(xs,ys):
        z=F.normalize(reference(torch.cat([x,y])),dim=-1)
        value,count=alignment_loss(z[:4],z[4:],list(map(str,range(4))),torch.ones(4,dtype=torch.bool))
        total+=2*value/8
    total.backward()
    torch.testing.assert_close(grad,reference.weight.grad,atol=2e-5,rtol=2e-4)


def test_two_rank_candidates_and_accumulated_window_gradient(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_window,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)
