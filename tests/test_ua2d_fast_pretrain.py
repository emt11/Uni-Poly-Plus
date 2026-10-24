"""One focused synthetic check of acceleration equivalence, IPC and ordering."""
import copy
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import pytest
import torch
from rdkit import Chem
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.ua2d_galformer_pretrain import prepare_sample,collate_samples
from src.dataset.ua2d_fast_pretrain import (build_template,prepare_fast_sample,encode,
    decode_template,decode_batch,PreparedMicrobatches,worker_init)
from src.modules.ua2d_galformer_pretrain import PretrainConfig
from tests.test_dual_glt import _toy_pair


def inputs(text,valid=True):
    text=Chem.MolToSmiles(Chem.MolFromSmiles(text),canonical=True)
    top=build_canonical_periodic_topology(text);_,tri=_toy_pair(text)
    if not valid:
        tri.trimer_geometry_valid=False
        tri.trimer_pos=torch.empty((0,3))
    return top,tri,text,build_dual_static(top,tri,text)


def same(a,b):
    if torch.is_tensor(a):
        assert a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('strategy',['mask_align','mask_align_geometry'])
@pytest.mark.parametrize('fp_enabled',[False,True])
def test_exact_cached_views_labels_and_collated_indices(strategy,fp_enabled):
    cfg=PretrainConfig(strategy=strategy,fp_enabled=fp_enabled)
    fp=np.zeros(2048,dtype=np.uint8);fp[[0,30,2047]]=1
    native=[];fast=[]
    for i,(text,valid) in enumerate([('*COC*',True),('*CCN*',True),('*C*',True),('*c1ccc(*)cc1',True),('*COC*',False)]):
        top,tri,text,static=inputs(text,valid)
        template=decode_template(encode(build_template(top,tri,text,static)))
        snapshot=copy.deepcopy(template)
        for position in (i,100803+i):
            original=prepare_sample(top,tri,text,static=static,config=cfg,seed=42,position=position,
                                    fp_target=fp if fp_enabled else None)
            actual=prepare_fast_sample(template,config=cfg,seed=42,position=position,
                                       fp_target=fp if fp_enabled else None)
            same(original[0].to_dict(),actual[0].to_dict());same(original[1],actual[1])
            native.append(original);fast.append(actual)
        same(snapshot['clean'].to_dict(),template['clean'].to_dict())
        same({k:v for k,v in snapshot.items() if k!='clean'},
             {k:v for k,v in template.items() if k!='clean'})
    a,al=collate_samples(native);b,bl=collate_samples(fast)
    same(a.to_dict(),b.to_dict());same(al,bl)
    decoded,labels,trace=decode_batch(encode((b,bl,dict(micro_index=7))))
    same(a.to_dict(),decoded.to_dict());same(al,labels);assert trace==dict(micro_index=7)


class TinyPrepared(PreparedMicrobatches):
    def __len__(self):return 4
    def _open(self):
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream
        self.environment=lmdb.open(str(self.root/'templates.lmdb'),readonly=True,lock=False)
        self.keys=np.load(self.root/'keys.npy',allow_pickle=False)
        self.rows=np.array([17,23])
        self.stream=OrderedSampleStream(2,42)


def test_spawn_prefetch_keeps_positions_source_order_and_actual_counts(tmp_path):
    import lmdb
    from src.training.glt_dual_runtime import OrderedSampleStream
    keys=np.array([b'a'*32,b'b'*32],dtype='V32')
    np.save(tmp_path/'keys.npy',keys,allow_pickle=False)
    env=lmdb.open(str(tmp_path/'templates.lmdb'),map_size=10**8)
    with env.begin(write=True) as txn:
        for key,text in zip(keys,('*COC*','*CCN*')):
            top,tri,text,static=inputs(text)
            template=build_template(top,tri,text,static);template['sample_key']=key.tobytes()
            txn.put(key.tobytes(),encode(template))
    env.close()
    context=mp.get_context('spawn');counts=context.Array('q',2)
    ds=TinyPrepared(tmp_path,PretrainConfig(),rank=1,prepared_counts=counts)
    loader=torch.utils.data.DataLoader(ds,batch_size=None,num_workers=2,multiprocessing_context=context,
        prefetch_factor=2,worker_init_fn=worker_init,generator=torch.Generator().manual_seed(42))
    stream=OrderedSampleStream(2,42)
    for i,payload in enumerate(loader):
        data,labels,trace=decode_batch(payload)
        start=(i*2+1)*84
        assert trace['micro_index']==i and trace['start_position']==start
        assert trace['source_rows']==[[17,23][stream.index_at(start+j)] for j in range(84)]
        assert len(labels['identity'])==84 and data.geometry_valid.numel()==84
    assert sum(counts[:])==4*84


def test_restart_context_has_distinct_parents_and_preserves_spent_work():
    from src.training import ua2d_pretrain_fp as c
    assert c.REVISION=='r4' and c.OUT.name=='fast_r4'
    assert c.parent_path('V1')==c.OUT/'pretrain/V1/deploy.pt'
    assert c.LIMITS['pt_starts']==2+3
    assert c.LIMITS['pt_updates']==1392+3*5000
    assert c.LIMITS['pt_presentations']==1402296+3*5040000
    assert c.LIMITS['cpu_suites']==2+1
