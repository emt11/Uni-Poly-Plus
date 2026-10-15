"""One final CPU suite per r10 scientific cycle; no training, workers or GPU."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from src.modules.glt_r10 import (OUT,COMMON,PT_ARMS,FT_ARMS,pt_config,validate_pt_config,
    validate_package,GraphFiLM,GraphAttention,NonbondedInput,GEOMETRY)
from src.modules.glt_dual_pretrain import (alignment_loss,alignment_local_anchor_count,
    u1_alignment_options,global_objective)
from scripts.run_glt_r10_unit import trainer_for,ft_config,build_model,head_path,parent,read
from src.training.glt_dual_runtime import scheduled_lr
CYCLE=os.environ['GLT_R10_TEST_CYCLE']


@pytest.fixture(scope='module',autouse=True)
def cpu():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(16)


def test_strict_cycle_metadata_and_saved_parent_recovery():
    arm=FT_ARMS[CYCLE][0];cfg=ft_config(CYCLE,arm)
    package=torch.load(parent(CYCLE,arm),map_location='cpu',weights_only=False)
    if cfg['training_config']:
        validate_package(package)
        for key in ('r10_revision','runtime_revision','training_config','seed','align_direction','nonbonded_revision'):
            wrong=dict(package);wrong.pop(key)
            with pytest.raises(ValueError):validate_package(wrong)
        wrong=copy.deepcopy(cfg['training_config']);wrong['global_batch']=1009
        with pytest.raises(ValueError):validate_pt_config(wrong)
    head=torch.load(head_path(CYCLE,'eea',0),map_location='cpu',weights_only=False)
    model=build_model(package,cfg,head)
    from src.modules.glt_dual import strict_module_state
    best=torch.load(OUT/CYCLE/'finetune'/arm/'eea/fold0/best.pt',map_location='cpu',weights_only=False)
    strict_module_state(model,best['state_dict'])
    with pytest.raises(ValueError):strict_module_state(model,{})
    if CYCLE in ('A1','A2'):
        assert not hasattr(model,'glt') and not hasattr(model,'model')
        assert sum(p.numel() for p in model.parameters())==19262233
    elif CYCLE in ('F2','F3'):
        assert model.model.token_fusion.implementation=='loop' and not model.model.collect_fusion_attention


if CYCLE in ('A1','A2','P1','G0','G1'):
    def test_saved_public_initialization_and_independent_new_modules():
        public=torch.load(COMMON,map_location='cpu',weights_only=False)['model']
        for arm in PT_ARMS[CYCLE]:
            saved=torch.load(OUT/CYCLE/'initialization'/(arm+'_step0.pt'),map_location='cpu',weights_only=False)
            assert saved['config']==pt_config(CYCLE,arm)
            for name,value in public.items():assert torch.equal(value,saved['model'][name])
        if CYCLE=='A1':
            a,b=[torch.load(OUT/CYCLE/'initialization'/(arm+'_step0.pt'),map_location='cpu',weights_only=False)['model'] for arm in PT_ARMS[CYCLE]]
            assert set(a)==set(b)
            assert all(torch.equal(a[k],b[k]) for k in a)
            cfg=pt_config('A1','A1_FUSEDCE_2D');torch.manual_seed(7);m=trainer_for(cfg)
            rng=torch.get_rng_state().clone();cfg['arm']='A1_RAWCE_2D'
            torch.manual_seed(7);trainer_for(pt_config('A1','A1_RAWCE_2D'))
            assert torch.equal(rng,torch.get_rng_state())


if CYCLE in ('A1','A2'):
    def test_ce_entrance_and_gradient_path():
        from tests.test_dual_glt import sample
        from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
        records=[]
        for i,text in enumerate(('*COC*','*CCN*')):
            top,tri,_=sample(text)
            records.append(prepare_pretrain_sample(top,tri,text,seed=42,key='synthetic',position=i,
                opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident',sigma=0))
        data,labels=pretrain_collate(records)
        for arm in PT_ARMS[CYCLE]:
            m=trainer_for(pt_config(CYCLE,arm)).eval();seen={};original=m.encoder.encode
            def encode(*args,**kwargs):
                value=original(*args,**kwargs);seen['encoded']=value;return value
            m.encoder.encode=encode
            h=m.atom_head.register_forward_pre_hook(lambda mod,args:seen.update(ce=args[0]))
            result=m(data,labels);h.remove()
            expected='atom_states' if arm=='A1_RAWCE_2D' else 'fused_atom_states'
            assert seen['ce'] is seen['encoded'][expected]
            result['sums'][0].backward()
            fusion=sum(float(p.grad.abs().sum()) for p in m.encoder.token_fusion.parameters() if p.grad is not None)
            assert fusion==0 if arm=='A1_RAWCE_2D' else fusion>0
            assert sum(float(p.grad.abs().sum()) for p in m.encoder.o8.parameters() if p.grad is not None)>0

    def test_direction_multi_positive_invalid_mask_and_global_normalization():
        gen=torch.Generator().manual_seed(19)
        a=torch.nn.functional.normalize(torch.randn(4,128,generator=gen),dim=-1).requires_grad_()
        b=torch.nn.functional.normalize(torch.randn(4,128,generator=gen),dim=-1).requires_grad_()
        direction='2d_anchor' if CYCLE=='A2' else 'symmetric'
        loss,count=alignment_loss(a,b,['same','same','other','invalid'],torch.tensor([True,True,True,False]),direction=direction)
        score=a[:3]@b[:3].T/.1;positive=torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
        def formula(s):return (torch.logsumexp(s,1)-(s*positive).sum(1)/positive.sum(1)).sum()
        expected=formula(score) if direction=='2d_anchor' else .5*(formula(score)+formula(score.T))
        torch.testing.assert_close(loss,expected);assert count==3
        loss.backward();assert a.grad[:3].abs().sum()>0 and b.grad[:3].abs().sum()>0
        assert a.grad[3].abs().sum()==0 and b.grad[3].abs().sum()==0
        assert alignment_local_anchor_count(['same']*4,torch.ones(4,dtype=torch.bool))==0
        sums=torch.tensor([2.,6.],requires_grad=True);counts=torch.tensor([4.,8.])
        torch.testing.assert_close(global_objective(sums,counts,2,(1.,.1))/2,(sums/counts*torch.tensor([1.,.1])).sum())
        old=torch.load(Path('results/glt_optimization_20261011/r4/U1/pretrain/P_ALIGN/deploy_05000.pt'),map_location='cpu',weights_only=False)
        changed=dict(old,atom_ce_input='raw')
        with pytest.raises(ValueError):u1_alignment_options(changed)


if CYCLE=='P1':
    def test_scheduler_boundaries_and_only_declared_schedule_change():
        base=pt_config('P1','P1_BASE20K');candidate=pt_config('P1','P1_SCHED5K')
        differences={k for k in base if base[k]!=candidate[k]}
        assert differences=={'arm','initial_state','schedule_total_steps'}
        for cfg in (base,candidate):
            args={k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')}
            assert scheduled_lr(0,**args)==pytest.approx(1e-7)
            assert scheduled_lr(1999,**args)==pytest.approx(2e-4)
            assert scheduled_lr(2000,**args)==pytest.approx(2e-4)
            assert scheduled_lr(cfg['schedule_total_steps'],**args)==pytest.approx(1e-9)
        assert scheduled_lr(4999,**{k:candidate[k] for k in args}) < scheduled_lr(4999,**{k:base[k] for k in args})


if CYCLE in ('F2','F3'):
    def test_zero_initialization_mask_after_bias_rng_and_backbone_group():
        layer_type=GraphFiLM if CYCLE=='F2' else GraphAttention
        state=torch.get_rng_state().clone();module=layer_type();assert torch.equal(state,torch.get_rng_state())
        assert sum(p.numel() for p in module.parameters())==(99392 if CYCLE=='F2' else 263040)
        features=torch.randn(2,1024,requires_grad=True);valid=torch.tensor([True,False])
        assert torch.equal(module(features,valid),features)
        final=module.net[-1] if CYCLE=='F2' else module.output
        final.bias.data.fill_(1);result=module(features,valid)
        assert torch.equal(result[1],features[1]) and not torch.equal(result[0],features[0])
        result.sum().backward();assert features.grad is not None and features.grad.abs().sum()>0 and final.bias.grad.abs().sum()>0
        from scripts.finetune_mcl_ph import periodic_tdl_trainability,periodic_tdl_optimizer
        fake=nn.Module();fake.head=nn.Linear(1024,1);fake.graph_fusion=module
        periodic_tdl_trainability(fake,head_only=True);assert all(not p.requires_grad for p in module.parameters())
        periodic_tdl_trainability(fake,head_only=False);opt,_=periodic_tdl_optimizer(fake,'xc',head_only=False,profile='uniform_u1')
        group=next(g for g in opt.param_groups if g['name']=='backbone')
        assert {id(p) for p in module.parameters()} <= {id(p) for p in group['params']}

    if CYCLE=='F3':
        def test_two_keys_per_query_have_nonconstant_selection():
            module=GraphAttention();tokens=torch.randn(3,2,512)
            q=module.q(tokens).reshape(3,2,4,32).transpose(1,2)
            k=module.k(tokens).reshape(3,2,4,32).transpose(1,2)
            weights=torch.softmax(q@k.transpose(-2,-1)/(32**.5),-1)
            assert weights.shape==(3,4,2,2)
            assert (weights>0).all() and (weights<1).all() and weights.std()>0


if CYCLE=='G0':
    def test_second_random_stream_and_preserved_torsion_contract():
        from src.modules.glt_dual import PHYSICAL_TORSION
        a,b=[pt_config('G0',arm) for arm in PT_ARMS['G0']]
        assert a['seed']==b['seed']==43
        assert all(b[k]==v for k,v in PHYSICAL_TORSION.items())
        ignored=set(PHYSICAL_TORSION)|{'arm','initial_state'}
        assert all(a[k]==b[k] for k in a if k not in ignored)
        saved=torch.load(OUT/'G0/initialization/G0_TORSION_step0.pt',map_location='cpu',weights_only=False)['model']
        assert saved['encoder.glt.torsion_gate'].count_nonzero()==0


if CYCLE=='G1':
    def test_pure_dihedral_change_rigid_invariance_shortest_paths_and_empty_mask():
        from src.dataset.glt_r10 import nonbonded_fields,attach_nonbonded
        tri=SimpleNamespace(trimer_atomic_number=torch.tensor([6]*4),trimer_ru_offset=torch.tensor([-1,0,0,1]),trimer_base_ru_atom_id=torch.tensor([0,1,2,0]))
        static=dict(token_pos_index_a=np.asarray([0,1,2]),token_pos_index_b=np.asarray([1,2,3]),token_z_a=np.asarray([6]*3),token_z_b=np.asarray([6]*3))
        p=torch.tensor([[0.,1,0],[0,0,0],[1,0,0],[1,1,0]])
        p2=p.clone();p2[3]=torch.tensor([1.,-1,0])
        a=nonbonded_fields(static,tri,p);b=nonbonded_fields(static,tri,p2)
        assert a[0].tolist()==b[0].tolist()==[1] and a[1].item()!=b[1].item()
        rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
        moved=nonbonded_fields(static,tri,p@rotation+7)
        torch.testing.assert_close(a[1],moved[1]);assert a[3]==moved[3]
        # Closing a ring introduces a shorter endpoint path; all observations disappear.
        closed={k:np.append(v,0 if k=='token_pos_index_a' else 3 if k=='token_pos_index_b' else 6) for k,v in static.items()}
        assert not nonbonded_fields(closed,tri,p)[0].numel()
        data=SimpleNamespace(geometry_valid=True);attach_nonbonded(data,static,tri,p)
        layer=NonbondedInput();result=layer(data,3);assert result.count_nonzero()==0
        layer.net[-1].weight.data.fill_(.01);layer(data,3).sum().backward()
        assert all(v.grad is not None and v.grad.abs().sum()>0 for v in layer.parameters())
        data.geometry_valid=False;attach_nonbonded(data,static,tri,p)
        assert not data.geometry_valid and layer(data,3).count_nonzero()==0


if CYCLE=='Q':
    def test_quality_global_denominator_and_unchanged_coverage(tmp_path):
        from src.dataset.glt_r10 import build_qc_sidecar,attach_qc_source
        records=[];ledger=[]
        for i in range(3):
            key=f'{i:064x}';records.append(dict(source_index=i,sample_key=key,normalized_smiles=f'identity{i}',pretrain_role='train',geometry_status='PASS' if i<2 else 'FAIL'))
            ledger.append(dict(source_index=i,sample_key=key,normalized_smiles=f'identity{i}',geometry_valid=i<2,final_status='PASS' if i<2 else 'FAIL',selected_converged=i==0 if i<2 else None,mmff_status=0 if i==0 else 1 if i==1 else None))
        lp=tmp_path/'qc.jsonl';rp=tmp_path/'records.jsonl'
        lp.write_text(''.join(json.dumps(r)+'\n' for r in ledger));rp.write_text(''.join(json.dumps(r)+'\n' for r in records))
        build_qc_sidecar(tmp_path/'sidecar',lp,rp,count=3)
        source=SimpleNamespace(kind='pi1m',entries=records);before=copy.deepcopy(records)
        attach_qc_source(source,tmp_path/'sidecar');assert source.entries==before and source.r10_qc_weights==[1.,.5,0.]
        values=torch.tensor([3.,7.,9.],requires_grad=True);q=torch.tensor(source.r10_qc_weights);loss=global_objective((values*q).sum()[None],q.sum()[None],fractional_counts=True)
        torch.testing.assert_close(loss,torch.tensor(6.5/1.5));loss.backward();torch.testing.assert_close(values.grad,q/q.sum())
        torch.testing.assert_close(global_objective(torch.tensor([2.]),torch.tensor([.5]),fractional_counts=True),torch.tensor(4.))
        source.entries=copy.deepcopy(records);source.entries[0]['sample_key']='ff'*32
        with pytest.raises(ValueError):attach_qc_source(source,tmp_path/'sidecar')
        ledger[0]['selected_converged']=None;lp.write_text(''.join(json.dumps(r)+'\n' for r in ledger))
        with pytest.raises(ValueError):build_qc_sidecar(tmp_path/'bad',lp,rp,count=3)
