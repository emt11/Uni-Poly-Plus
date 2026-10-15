"""P1: one fixed PI1M batch, input parity, ten separately counted objective probes."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_mcl_ph_r4_fast import atomic_json,timestamp
from src.training.glt_dual_runtime import open_source,OrderedSampleStream,pretrain_source_fields,move_labels
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.dataset.glt_mcl_ph import prepare_gp_sample,gp_collate
from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache
from src.modules.glt_mcl_ph_pretrain import GPPretrainer
OUT=ROOT/'results/gp_sota_20261004';LOG=ROOT/'logs/gp_sota_20261004'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
COMMON=OUT/'initialization/common_step0_205_v2.pt'
def read(p):return json.loads(p.read_text())
def norm(module):
    grads=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    return float(sum(g.square().sum() for g in grads).sqrt()) if grads else 0.
def main():
    torch.set_num_threads(1)
    folder=OUT/'p1_joint_probe_01';folder.mkdir(exist_ok=False)
    assert read(OUT/'initialization/acceptance_01.json')['status']=='PASS'
    assert (LOG/'p1_cpu_02/exit.code').read_text().strip()=='0'
    with np.load(ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/pretrain_indices.npz') as z:train=z['train_source_indices']
    stream=OrderedSampleStream(len(train),42);positions=list(range(24));selected=[int(train[stream.index_at(p)]) for p in positions]
    assert len(set(selected))==24
    source,_=open_source(PI,PI,dual_static_root=PI,selected_indices=sorted(selected))
    cache=MCLPHTrajectoryCache(ROOT/'data/processed/mcl_ph_r4_20261001/trajectory_train_5040000',verify_checksums=False)
    local={source.source_index_at(i):i for i in range(len(source))};records=[];identities=[]
    try:
        for p,index in zip(positions,selected):
            i=local[index];top,trimer,smiles,static,target,identity=pretrain_source_fields(source,i,include_target=True,include_identity=True)
            key=source.samples[i][0].hex()
            old,truth=prepare_pretrain_sample(top,trimer,smiles,seed=42,key=key,position=p,static=static,target=target)
            online,labels=prepare_gp_sample(top,trimer,smiles,seed=42,key=key,position=p,static=static,target=target,identity=identity)
            # Every old model field and each original target must be exact, not approximate.
            for k in old.keys():
                a,b=getattr(old,k),getattr(online,k)
                assert torch.equal(a,b) if torch.is_tensor(a) else a==b,(index,k)
            for k in ('atom_mask','atom_label','distance','angle_pairs','angle_cos'):
                assert torch.equal(truth[k],labels['glt'][k]),(index,k)
            np.testing.assert_array_equal(online.mcl_trajectory.numpy(),cache[p])
            records.append((online,labels));identities.append(dict(position=p,source_index=index,key=key,normalized_smiles=identity['normalized_smiles']))
        atomic_json(folder/'input_acceptance.json',dict(status='PASS',samples=identities,old_glt_fields_targets_exact=True,online_cached_ph_exact=True,single_noisy_view=True))
    finally:source.close();cache.close()
    data,labels=gp_collate(records)
    assert torch.equal(data.mcl_atom_batch[data.mcl_bond_index[:,0]],data.bond_batch)
    assert torch.equal(data.mcl_atom_batch[data.mcl_bond_index[:,1]],data.bond_batch)
    device=torch.device('cuda');torch.manual_seed(42)
    model=GPPretrainer(route='GP_A',common_path=COMMON).to(device);model.train()
    modules=dict(o8=model.encoder.o8,galformer=model.encoder.glt,experts=model.encoder.experts,router=model.encoder.router,adaptor=model.encoder.adaptor)
    report=[]
    for mode in ('dense','top2'):
        model.encoder.router.configure(mode)
        for name in ('atom','geometry','fp','alignment','balance'):
            ledger=read(LOG/'status.json');n=ledger['budgets']['probe_forward_backward_calls'];assert n<10
            ledger['budgets']['probe_forward_backward_calls']=n+1;ledger['execution_status']='P1_PROBE_RUNNING';atomic_json(LOG/'status.json',ledger)
            atomic_json(folder/f'call_{n+1:02d}.json',dict(status='STARTED',objective=name,mode=mode,updates=0,started_at=timestamp()))
            model.zero_grad(set_to_none=True)
            output=model(data.clone().to(device),move_labels(labels,device))
            loss=(output['sums'][{'atom':0,'geometry':1,'fp':2}[name]]/output['counts'][{'atom':0,'geometry':1,'fp':2}[name]].clamp_min(1)
                  if name in ('atom','geometry','fp') else output['align_sum']/output['align_count'].clamp_min(1) if name=='alignment' else output['balance'])
            assert torch.isfinite(loss);loss.backward()
            norms={k:norm(m) for k,m in modules.items()}
            if name=='atom':assert norms['o8']>0 and norms['router']==norms['adaptor']==norms['experts']==0
            elif name in ('geometry','fp','alignment'):
                assert all(norms[k]>0 for k in ('galformer','experts','router','adaptor')),norms
                if name!='geometry':assert norms['o8']>0
            else:assert norms['router']>0 and norms['adaptor']==0
            row=dict(status='PASS',objective=name,mode=mode,loss=float(loss.detach()),gradient_norms=norms,updates=0,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(),finished_at=timestamp())
            report.append(row);atomic_json(folder/f'call_{n+1:02d}.json',row)
            print(json.dumps(row),flush=True)
    atomic_json(folder/'report.json',dict(status='PASS',calls=10,sample_count=24,updates=0,rows=report,completed_at=timestamp()))
if __name__=='__main__':main()
