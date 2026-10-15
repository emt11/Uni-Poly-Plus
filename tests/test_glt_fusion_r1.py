"""最终集中一次运行的合成正确性测试；不消费正式数据或训练预算。"""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tests.test_dual_glt import sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.dataset.glt_fusion_md200 import MD200Sidecar
from src.modules.glt_dual_pretrain import DualPretrainer, deployment_package, load_deployment, local_attention_loss, alignment_loss
from src.modules.glt_dual import build_dual_glt_model
from src.modules.original_mips_knowledge_fusion import OriginalMIPSAttentiveFusion
from scripts.initialize_glt_fusion_r1 import modes, CORE


def batch_pair():
    records=[]
    for i,smiles in enumerate(('*COC*','*CCN*')):
        top,tri,_=sample(smiles)
        top.md200=torch.linspace(.1,.9,200).reshape(1,200)
        top.md200_valid=torch.tensor([True])
        records.append(prepare_pretrain_sample(top,tri,smiles,seed=42,key=str(i),position=i,opt_mode='GLT_FUSION_R1',third_task='fgr'))
    return pretrain_collate(records)


def test_sidecar_rows_validity_and_nonfinite(tmp_path):
    source=tmp_path/'source';source.mkdir()
    side=tmp_path/'md';side.mkdir()
    (side/'metadata.json').write_text(json.dumps(dict(schema='glt-fusion-md200-v1',complete=True,semantics='RDKit2DNormalized200/star_sub',source=str(source),columns=[str(i) for i in range(200)])))
    values=np.zeros((2,200),np.float32);values[0]=.5
    np.save(side/'records_values.npy',values);np.save(side/'records_valid.npy',np.array([True,False]))
    rows=[dict(source_row=i,normalized_smiles=s,valid=(i==0),reason='' if i==0 else 'failed') for i,s in enumerate(('*COC*','*CCN*'))]
    (side/'records_rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    entries=[dict(source_index=i,normalized_smiles=s) for i,s in enumerate(('*COC*','*CCN*'))]
    reader=MD200Sidecar(side,source,None,entries)
    assert reader.get(0)[1].item() and not reader.get(1)[1].item()
    for name in ('values.npy', 'valid.npy', 'rows.jsonl'):
        (side/f'eea_{name}').write_bytes((side/f'records_{name}').read_bytes())
    downstream_entries=[dict(original_row=i,source_index=390+i,normalized_smiles=s)
                        for i,s in enumerate(('*COC*','*CCN*'))]
    downstream_reader=MD200Sidecar(side,source,'eea',downstream_entries)
    assert downstream_reader.get(0)[1].item() and not downstream_reader.get(1)[1].item()
    with pytest.raises(ValueError,match='identity'):
        MD200Sidecar(side,source,None,[dict(source_index=0,normalized_smiles='wrong')])
    values[0,5]=np.nan;np.save(side/'records_values.npy',values)
    reader=MD200Sidecar(side,source,None,entries)
    with pytest.raises(ValueError,match='nonfinite'):reader.get(0)


def test_descriptor_partial_charge_failure_is_unavailable_not_valid_zero():
    from scripts.prepare_glt_fusion_md200 import descriptor
    class Generator:
        columns = [('MaxPartialCharge', float)] + [(str(i), float) for i in range(199)]
        def process(self, smiles):
            return [True, float('nan')] + [.5] * 199
    values, valid, reason = descriptor(Generator(), '*CC*')
    assert not valid and np.array_equal(values, np.zeros(200, np.float32))
    assert 'MaxPartialCharge' in reason and 'nan' in reason
    class Unexpected(Generator):
        columns = [('other_column', float)] + Generator.columns[1:]
    with pytest.raises(ValueError, match='unexpected nonfinite'):
        descriptor(Unexpected(), '*CC*')


def test_consumption_reader_never_observes_truncated_writer(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from scripts.pretrain_glt_dual import write_consumption
    path = tmp_path / 'consumption.json'
    write_consumption(path, attempted=0, completed=0, status='RUNNING')
    done = Event()
    def writer():
        try:
            for step in range(1, 101):
                write_consumption(path, attempted=step, completed=step, status='RUNNING')
        finally:
            done.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer)
        while not done.is_set():
            record = json.loads(path.read_text())
            assert record['attempted_updates'] == record['completed_updates']
        future.result()
    assert json.loads(path.read_text())['completed_updates'] == 100


def test_knowledge_availability_masks_projected_bias_and_isolates_graphs():
    module=OriginalMIPSAttentiveFusion(knodes=('md','glt3d'),knowledge_dims={'md':200,'glt3d':512})
    atoms=torch.randn(4,512);index=torch.arange(4)
    knowledge=dict(md=torch.randn(4,200),glt3d=torch.randn(4,512))
    availability=dict(md=torch.tensor([True,True,False,False]),glt3d=torch.tensor([True,False,True,False]))
    fused=module(atoms,knowledge,index,availability)
    torch.testing.assert_close(fused[3],atoms[3],rtol=0,atol=0)
    expected=atoms[1]+.5*module.v_proj['md'](knowledge['md'][1])
    torch.testing.assert_close(fused[1],expected)
    changed={k:v.clone() for k,v in knowledge.items()}
    changed['md'][2:]=100;changed['glt3d'][1]=100
    again=module(atoms,changed,index,availability)
    torch.testing.assert_close(again,fused)
    fused[0].square().mean().backward()
    for projection in (module.Wq,module.k_proj['md'],module.k_proj['glt3d'],module.v_proj['md'],module.v_proj['glt3d']):
        assert projection.weight.grad.norm()>0


@pytest.mark.parametrize('arm',CORE)
def test_core_ce_supervision_and_declared_controls(arm):
    data,labels=batch_pair();data.md200_valid.fill_(True)
    model=DualPretrainer(**modes(arm));result=model(data,labels)
    result['sums'][0].backward()
    if hasattr(model.encoder,'token_fusion'):
        grad=model.encoder.token_fusion.q.weight.grad
        if arm=='T_RAW':assert grad is None or grad.norm()==0
        else:assert grad is not None and grad.norm()>0
    if hasattr(model.encoder,'kfuse'):
        fusion=model.encoder.kfuse
        assert fusion.v_proj['glt3d'].weight.grad.norm()>0
        if arm=='K0':assert fusion.Wq.weight.grad is None or fusion.Wq.weight.grad.norm()==0
        else:
            assert fusion.Wq.weight.grad.norm()>0
            assert fusion.v_proj['md'].weight.grad.norm()>0
    encoded=model.encoder.encode(data)
    assert model.encoder.fuse(encoded).shape==(2,1024)
    assert not any('fp_head' in name for name in model.state_dict())


def test_mask_identity_md_rng_and_batch_incidence():
    top,tri,_=sample('*COC*');top.md200=torch.ones((1,200));top.md200_valid=torch.tensor([True])
    first=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='different-key',position=8,opt_mode='GLT_FUSION_R1')
    second=prepare_pretrain_sample(top,tri,'*COC*',seed=42,key='other-key',position=8,opt_mode='GLT_FUSION_R1')
    data,labels=first;mask=labels['atom_mask']
    assert labels['align_identity']==top.normalized_canonical_smiles
    assert (data.bond_z_a[mask[data.bond_canonical_a]]==0).all()
    assert (data.bond_z_b[mask[data.bond_canonical_b]]==0).all()
    assert torch.equal(data.bond_distance,second[0].bond_distance)
    assert torch.equal(mask,second[1]['atom_mask'])
    collated,_=pretrain_collate([first,second]);bond,atom=collated.incidence_index
    assert collated.md200.shape==(2,200)
    assert torch.equal(collated.bond_batch[bond],collated.canonical_graph_index[atom])


def test_local_kl_graph_reduction_and_remote_visibility():
    w=torch.tensor([[.2,.3,.5],[.1,.2,.7]],requires_grad=True)
    attention=[(0,torch.tensor([0,1]),torch.tensor([0,1,2]),w)]
    incidence=torch.tensor([[0,1,1],[0,0,1]])
    value,count=local_attention_loss(attention,incidence,w.sum()*0)
    expected=((-torch.log(w[0,:2]).mean()-np.log(2))-torch.log(w[1,1]))/2
    torch.testing.assert_close(value,expected);assert count==1
    value.backward();assert w.grad[0,2]==0 and w.grad[0,0]!=0


def test_combined_alignment_fgr_local_and_geometry_scales():
    data,labels=batch_pair();data.md200_valid.fill_(True)
    torch.manual_seed(42)
    normal=DualPretrainer('tk',fusion_revision='glt_fusion_r1',use_md200=True,atom_ce_input='fused',third_task='fgr',align_enabled=True,geometry_scales={'length':2.,'angle':3.})
    result=normal(data,labels)
    raw=result['raw_geometry_sums']
    torch.testing.assert_close(result['sums'][1],raw[0]/4+raw[1]/9)
    loss=result['sums'].sum()+result['extra_sums'].sum();loss.backward()
    assert normal.align_proj2.net[0].weight.grad.norm()>0
    assert normal.fgr_head.mlp[0].weight.grad.norm()>0
    local=DualPretrainer('tk',fusion_revision='glt_fusion_r1',use_md200=True,atom_ce_input='fused',third_task='fgr',local_enabled=True)
    result=local(data,labels);assert result['extra_counts'][1]>0 and result['extra_counts'][0]==0
    with pytest.raises(ValueError):DualPretrainer('tk',align_enabled=True,local_enabled=True)


def test_checkpoint_strict_roundtrip_and_old_package_rejection():
    pretrainer=DualPretrainer(**modes('TK_MD_A'));package=deployment_package(pretrainer,0)
    target=build_dual_glt_model('tk',fusion_revision='glt_fusion_r1',use_md200=True)
    load_deployment(target,package,0)
    data,_=batch_pair();target.eval();pretrainer.encoder.eval()
    torch.testing.assert_close(target.fuse(target.encode(data)),pretrainer.encoder.fuse(pretrainer.encoder.encode(data)))
    missing=dict(package,state_dict=dict(package['state_dict']));missing['state_dict'].pop(next(iter(missing['state_dict'])))
    with pytest.raises(ValueError):load_deployment(target,missing,0)
    with pytest.raises(ValueError):load_deployment(target,dict(package,fusion_revision=None,use_md200=False),0)
    from scripts.finetune_mcl_ph import GLTReferenceArm,periodic_tdl_trainability,periodic_tdl_optimizer
    model=GLTReferenceArm(fusion_mode='tk',fusion_revision='glt_fusion_r1',use_md200=True)
    periodic_tdl_trainability(model,head_only=True)
    assert not any(p.requires_grad for p in model.model.kfuse.parameters())
    periodic_tdl_trainability(model,head_only=False)
    optimizer,_=periodic_tdl_optimizer(model,'xc',head_only=False)
    backbone={id(p) for p in optimizer.param_groups[0]['params']}
    assert all(id(p) in backbone and p.requires_grad for p in model.model.kfuse.parameters())
    assert optimizer.param_groups[0]['lr']==2e-4


def test_alignment_duplicates_invalid_and_padding_excluded():
    z2=torch.nn.functional.normalize(torch.randn(4,128),dim=-1).requires_grad_()
    z3=torch.nn.functional.normalize(torch.randn(4,128),dim=-1).requires_grad_()
    value,count=alignment_loss(z2,z3,['same','same','other','padding'],torch.tensor([True,True,True,False]))
    assert count==3 and torch.isfinite(value)
    value.backward();assert z2.grad[-1].norm()==0 and z3.grad[-1].norm()==0
