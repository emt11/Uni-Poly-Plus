"""E0 only: one CPU suite, four existing record decodes, no workers/updates."""
import ast
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch

from src.dataset import glt_dual_pretrain as prep
from src.dataset.mcl_ph_direct import DirectMCLSource
from src.dataset.glt_dual import build_glt_2d_sample, glt_2d_collate
from src.modules import glt_dual as models
from src.modules.glt_dual_pretrain import (
    deployment_model_options, restore_deployment, deployment_package, DualPretrainer)
from src.training import glt_dual_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
PI = ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
UA = ROOT/'results/glt_optimization_20261011/r4/U1/pretrain/P_ALIGN/deploy_05000.pt'
FCE = ROOT/'results/glt_periodic_seam_20261009/r1/pretrain/F_CE/deploy_05000.pt'
S4 = ROOT/'results/glt_optimization_20261011/r6/S4/pretrain/T_FCE/deploy_05000.pt'
S5 = ROOT/'results/glt_optimization_20261011/r9/S5/pretrain/F_CE_2R/deploy_05000.pt'
HEAD = ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d/eea/fold0.pt'


@pytest.fixture(scope='module', autouse=True)
def cpu_only():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    previous = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get('E0_CPU_THREADS', '1')))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def records():
    # Explicit bounded selection; no source scans, no key checksum verification.
    source = DirectMCLSource(PI, task=None, selected_indices=[0,1,2,3],
        split_manifest=None, verify_key_digest=False)
    count = 0
    original = source.record
    def counted(index):
        nonlocal count
        count += 1
        assert count <= 4
        return original(index)
    source.record = counted
    try:
        fields = [runtime.pretrain_source_fields(source,i,include_target=True,
                    include_identity=True) for i in range(4)]
        assert count == 4
        assert any(bool(f[3]['geometry_valid']) for f in fields), 'bounded selection has no valid geometry'
        yield fields, source.samples, source.entries
    finally:
        source.close()
        evidence = os.environ.get('E0_TEST_EVIDENCE_DIR')
        if evidence:
            (Path(evidence)/'record_budget.json').write_text(json.dumps(dict(
                existing_record_decodes=count,selected_indices=[0,1,2,3],
                worker_starts=0,optimizer_updates=0,gpu_computation=0,
                cpu_threads=torch.get_num_threads()),indent=2)+'\n')
        print(f'E0 existing_record_decodes={count}; worker_starts=0; optimizer_updates=0')


@pytest.fixture(scope='module')
def baseline_prepare():
    # Execute only the original module definitions from this cycle's snapshot.
    # This is an old/new comparison, not another suite or a training entry.
    path=ROOT/'results/glt_optimization_20261011/r10/E0/baseline/src/dataset/glt_dual_pretrain.py'
    module=ModuleType('src.dataset._glt_e0_original')
    module.__package__='src.dataset'
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left: assert_same(left[key], right[key])
    elif isinstance(left, (tuple,list)):
        assert len(left) == len(right)
        for a,b in zip(left,right): assert_same(a,b)
    else:
        assert left == right


def kwargs(i, key):
    return dict(seed=42,key=key.hex(),position=i,sigma=.03,ratio=.3,
                opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')


def test_existing_cached_groups_mask_noise_labels_and_no_reparse(records, baseline_prepare, monkeypatch):
    fields, samples, _ = records
    assert any('fingerprint_packed' in field[4] for field in fields)
    baseline=[]
    for i,(top,tri,text,static,target,identity) in enumerate(fields):
        assert tuple(tuple(g) for g in target['brics_groups']) == baseline_prepare.chemical_groups(identity['normalized_smiles'])
        baseline.append(baseline_prepare.prepare_pretrain_sample(top,tri,text,static=static,
            **kwargs(i,samples[i][0])))
    def forbidden(*args,**kw): raise AssertionError('cached preparation re-parsed/ran BRICS')
    monkeypatch.setattr(prep,'resolve_normalized_identity',forbidden)
    monkeypatch.setattr(prep,'chemical_groups',forbidden)
    from rdkit.Chem import BRICS
    monkeypatch.setattr(BRICS,'FindBRICSBonds',forbidden)
    for i,(top,tri,text,static,target,identity) in enumerate(fields):
        old,old_labels=baseline[i]
        keys=tuple(target)
        fingerprint=target.get('fingerprint_packed')
        fingerprint_before=None if fingerprint is None else fingerprint.copy()
        # Both published cache generations must retain exactly the original
        # mask/noise stream; the obsolete fingerprint is never an input.
        for cached in (target, {'brics_groups':target['brics_groups']}):
            new,labels=prep.prepare_pretrain_sample(top,tri,text,static=static,target=cached,
                identity=identity,source_type='direct_pi1m',**kwargs(i,samples[i][0]))
            assert_same(dict(old.items()),dict(new.items()))
            assert labels['brics_source']=='cached'
            for name in old_labels:
                if name not in ('brics_source','preparation_source'):assert_same(old_labels[name],labels[name])
            assert old.geometry_valid == new.geometry_valid
        assert tuple(target)==keys
        if fingerprint is not None:assert_same(target['fingerprint_packed'],fingerprint_before)
        assert_same(tri.trimer_pos,fields[i][1].trimer_pos)


class MemorySource:
    kind='pi1m'
    def __init__(self, records):
        self.fields,self.samples,self.entries=records
        self.calls=[]
    def __len__(self):return len(self.fields)
    def validated_pretrain_fields_for(self,index,*,include_target):
        assert include_target is True
        self.calls.append(index)
        return self.fields[index]
    def __getitem__(self,index):raise AssertionError('extra record read')
    def static_for(self,index):raise AssertionError('extra static read')
    def target_for(self,index):raise AssertionError('extra target read')


def test_inline_worker_method_and_process_validation_transfer(records,monkeypatch):
    source=MemorySource(records)
    actual=prep.prepare_pretrain_sample
    captured=[]
    def observe(top,tri,text,**kw):
        field=next(f for f in source.fields if f[0] is top)
        assert kw['target'] is field[4] and kw['identity'] is field[5]
        assert kw['static'] is field[3] and kw['source_type']=='direct_pi1m'
        captured.append(kw)
        return actual(top,tri,text,**kw)
    monkeypatch.setattr(prep,'prepare_pretrain_sample',observe)
    worker=runtime.RankMicrobatchStream(source,seed=42,world=1,rank=0,microbatch=2,
        accumulation=1,start_step=0,max_steps=1,sigma=.03,ratio=.3,
        opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident')
    # Direct sample method only: never construct a DataLoader or process.
    data,labels=worker[0]
    assert len(source.calls)==2 and len(captured)==2
    assert labels['align_identity']==[source.fields[i][0].normalized_canonical_smiles for i in source.calls]
    # Evaluate the actual call expressions in the existing inline and process
    # validation entries without running their GPU/training/DDP outer functions.
    config=dict(seed=42,noise_sigma=.03,atom_mask_ratio=.3,opt_mode='GLT_FUSION_R1',
                geometry_mode='finite_incident',torsion_modules=False)
    for filename,function in [('scripts/pretrain_glt_dual.py',None),
            ('scripts/evaluate_glt_pt_endpoint.py','evaluate_process_validation')]:
        tree=ast.parse((ROOT/filename).read_text())
        region=tree if function is None else next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        calls=[n for n in ast.walk(region) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='prepare_source_sample']
        assert len(calls)==1
        scope=dict(prepare_source_sample=runtime.prepare_source_sample,source=source,
            index=0,position=0,config=config,third_task='none',fgr_mu=0.,fgr_sigma=1.,
            fgr_max_pairs=32,atom_target='element',env_vocab=None,torsion_mode=None)
        eval(compile(ast.Expression(calls[0]),filename,'eval'),scope)
    assert len(source.calls)==4 and len(captured)==4
    # Indexed source forwards the six fields as well.
    indexed=runtime.IndexedFrozenDualSource(source,np.array([2],dtype=np.int64))
    runtime.prepare_source_sample(indexed,0,**kwargs(0,source.samples[2][0]))
    assert captured[-1]['identity'] is source.fields[2][5]


def test_reader_target_missing_and_same_record_identity(records):
    fields,samples,entries=records
    top,tri,text,static,target,identity=fields[0]
    calls=[]
    record=SimpleNamespace(topology=top,trimer=tri,static=static,target=target,
                           validated_normalized_identity=identity)
    source=SimpleNamespace(kind='pi1m',entries=entries,
        manifest={'schema':'mcl-ph-direct-source-v1'},
        record=lambda i:(calls.append(i) or record))
    result=DirectMCLSource.validated_pretrain_fields_for(source,0,include_target=True)
    assert calls==[0] and result[4] is target and result[5] is identity
    record.target=None
    with pytest.raises(ValueError,match='target cache'):
        DirectMCLSource.validated_pretrain_fields_for(source,0,include_target=True)


def test_bad_cache_identity_schema_are_rejected(records):
    fields,samples,_=records
    top,tri,text,static,target,identity=fields[0]
    base=dict(static=static,target=target,identity=identity,source_type='direct_pi1m',**kwargs(0,samples[0][0]))
    for bad in ({},{'brics_groups':None},{'brics_groups':((0,0),)},
                {'brics_groups':(('0',),)},{'brics_groups':((True,),)},
                dict(target,unknown_schema='v2')):
        with pytest.raises(ValueError):prep.prepare_pretrain_sample(top,tri,text,**dict(base,target=bad))
    with pytest.raises(ValueError,match='missing'):
        prep.prepare_pretrain_sample(top,tri,text,**dict(base,target=None))
    for bad in (dict(identity,normalized_smiles='*N*'),
                dict(identity,validated_source_smiles='*N*'),
                dict(identity,validated_topology_schema='unknown'),
                dict(identity,canonical_to_normalized_base=torch.tensor([-1]))):
        with pytest.raises(ValueError,match='identity'):
            prep.prepare_pretrain_sample(top,tri,text,**dict(base,identity=bad))
    corrupt=top.clone();corrupt.canonical_periodic_topology_schema='unknown'
    with pytest.raises(ValueError,match='schema'):
        prep.prepare_pretrain_sample(corrupt,tri,text,**base)


@pytest.mark.parametrize('path',[FCE,UA,S4,S5])
def test_saved_metadata_constructs_then_strictly_restores(path):
    package=torch.load(path,map_location='cpu',weights_only=False)
    options=deployment_model_options(package)
    model=restore_deployment(package,dropout=0)
    assert model.fusion_mode==options['fusion_mode']
    assert model.geometry_mode==options['geometry_mode']
    assert model.torsion_revision==options['torsion_revision']
    for k,v in package['state_dict'].items():assert torch.equal(model.state_dict()[k],v)
    for key in ('fusion_revision','geometry_revision','torsion_revision','pretrain_revision','reference_revision'):
        with pytest.raises(ValueError):deployment_model_options(dict(package,**{key:'unknown'}))
    with pytest.raises(ValueError):deployment_model_options(dict(package,fusion_attention_mode='unknown'))
    for key in ('geometry_mode','angle_value_enabled','readout','torsion_modules','align_enabled'):
        bad=dict(package);bad.pop(key)
        with pytest.raises(ValueError):deployment_model_options(bad)
    state=package['state_dict'];name='o8.atom_embedding.projection.weight'
    for bad in (dict(state,unexpected=torch.zeros(1)),dict(state,**{name:state[name].double()}),
                dict(state,**{name:state[name][:1]})):
        with pytest.raises(ValueError):
            from src.modules.glt_dual_pretrain import load_deployment
            load_deployment(model,dict(package,state_dict=bad))
    if path==S4:
        with pytest.raises(ValueError):deployment_model_options(dict(package,global_batch=999))
    if path==S5:
        with pytest.raises(ValueError):deployment_model_options(dict(package,global_batch=999))
    if path==UA:
        missing=dict(package);missing.pop('fusion_gate')
        with pytest.raises(ValueError):deployment_model_options(missing)
        with pytest.raises(ValueError):deployment_model_options(dict(package,atom_ce_input='raw'))
    del model,package
    gc.collect()


def test_explicit_legacy_schema_defaults_and_unknown_package_rejection():
    trainer=DualPretrainer('concat')
    package=deployment_package(trainer,0)
    # These fields were absent in the old legacy schema, whose input_schema is explicit.
    for key in ('geometry_revision','geometry_mode','angle_value_enabled','readout','fusion_gate'):
        package.pop(key)
    restored=restore_deployment(package,expected_step=0)
    assert restored.geometry_mode=='finite_internal' and restored.readout=='internal'
    for bad in (dict(package,input_schema='unknown'),dict(package,architecture='unknown'),
                dict(package,geometry_revision='glt_periodic_seam_r1')):
        with pytest.raises(ValueError):deployment_model_options(bad,expected_step=0)


def test_pure_2d_cpu_matches_old_ua_path_no_galformer(records,monkeypatch,tmp_path):
    from scripts import run_glt_align_2d_unit as u1
    package=torch.load(UA,map_location='cpu',weights_only=False)
    head=torch.load(HEAD,map_location='cpu',weights_only=False)
    full=restore_deployment(package,dropout=0)
    old=u1.Pure2D(full).eval();models.strict_module_state(old.head,head)
    cfg_fused=json.loads((u1.OUT/'initialization/UA_FUSED.json').read_text())
    fused=u1.build_arm(package,'UA_FUSED',0,cfg_fused)
    assert fused.model.geometry_mode == 'finite_incident'
    del fused
    fields,_,_=records
    items=[build_glt_2d_sample(top,text,static=static) for top,tri,text,static,_,_ in fields[:2]]
    data=glt_2d_collate(items)
    with torch.no_grad():expected=old(data)
    def forbidden(*a,**kw):raise AssertionError('pure 2D constructed Galformer')
    monkeypatch.setattr(models,'GalformerTrimer3D',forbidden)
    new=models.restore_pure_2d(package,head,dropout=0).eval()
    assert set(dict(new.named_children()))=={'o8','norm2','head'}
    assert all(k not in data for k in ('geometry_valid','bond_distance','trimer_pos','line_angle','md200'))
    with torch.no_grad():
        assert torch.equal(new(data),expected)
        top,_,text,_,_,_=fields[0]
        assert torch.equal(new.forward_topology(top,text),new(glt_2d_collate(items[:1])))
    corrupt=top.clone();corrupt.feature_schema='unknown'
    with pytest.raises(ValueError,match='schema'):new.forward_topology(corrupt,text)
    # Only the final test's temporary directory receives a restoration example.
    destination=tmp_path/'ua_2d_example.pt'
    scaler=json.loads((u1.OUT/'finetune/UA_2D/eea/fold0/scaler.json').read_text())
    torch.save(dict(schema='glt-e0-2d-test-example-v1',state_dict=new.state_dict(),
        scaler=scaler),destination)
    example=torch.load(destination,map_location='cpu',weights_only=False)
    restored=models.Pure2DGLTModel(dropout=0).eval()
    models.strict_module_state(restored,example['state_dict'])
    with torch.no_grad():assert torch.equal(restored(data),expected)
    assert example['scaler']==scaler
    assert scaler['fit']=='train' and scaler['transform']=='standard'
    assert torch.equal(restored(data)*scaler['scale'][0]+scaler['mean'][0],
                       expected*scaler['scale'][0]+scaler['mean'][0])
    cfg=json.loads((u1.OUT/'initialization/UA_2D.json').read_text())
    hooked=u1.build_arm(package,'UA_2D',0,cfg).eval()
    models.strict_module_state(hooked.head,head)
    with torch.no_grad():assert torch.equal(hooked(data),expected)
    # Full schema is checked even for tensors outside the selected 2D modules.
    state=package['state_dict'];name='glt.endpoint.weight'
    for bad in (dict(state,unknown=torch.zeros(1)),dict(state,**{name:state[name].double()}),
                dict(state,**{name:state[name][:1]})):
        with pytest.raises(ValueError):models.restore_pure_2d(dict(package,state_dict=bad),head)
    with pytest.raises(ValueError):models.restore_pure_2d(package,{})
