"""S5's sole final CPU suite: new reference identity, binding, and two-rank wiring."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from scripts import run_glt_pt_matched_reference_unit as unit
from scripts import select_glt_pt_matched_reference as select
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import DualPretrainer,deployment_package

def test_saved_public_state_has_no_new_tensors_and_matches_torsion_public_initialization():
    common=torch.load(unit.COMMON,map_location='cpu',weights_only=False)['model']
    new=torch.load(unit.OUT/'initialization/F_CE_2R_step0.pt',map_location='cpu',weights_only=False)
    torsion=torch.load(unit.REFERENCE/'initialization/T_FCE_step0.pt',map_location='cpu',weights_only=False)
    assert new['step']==0 and set(new['model'])==set(common)
    assert new['config']==unit.make_config()
    for key,value in common.items():
        assert torch.equal(value,new['model'][key]) and torch.equal(value,torsion['model'][key])
    assert len(set(torsion['model'])-set(new['model']))==5

def test_reference_config_keeps_matched_training_and_rejects_changes():
    cfg=unit.make_config();old=unit.read(unit.REFERENCE/'initialization/T_FCE.json')
    assert all(cfg[key]==old[key] for key in unit.MATCHED_FIELDS)
    assert not cfg['torsion_modules'] and cfg['torsion_revision'] is None and cfg['torsion_mode']=='off'
    assert cfg['global_batch']==cfg['microbatch']*cfg['accumulation']*cfg['expected_world_size']==1008
    assert cfg['total_parameters']==39876344 and cfg['new_parameters']==0
    for key,value in [('reference_revision','unknown'),('expected_world_size',4),('accumulation',3),
                      ('torsion_modules',True),('atom_ce_input','raw'),('noise_sigma',0),('ft_batch_size',12)]:
        with pytest.raises(ValueError):unit.validate_config(dict(cfg,**{key:value}))

def test_completed_parent_strict_reference_contract_and_no_torsion_module():
    cfg=unit.make_config();package=torch.load(unit.parent('F_CE_2R'),map_location='cpu',weights_only=False)
    model=unit.ReferenceArm(package,'F_CE_2R',0,cfg)
    assert model.model.glt.torsion_mlp is None and model.model.glt.torsion_gate is None
    assert sum(p.numel() for p in model.parameters())==unit.PARAMETERS
    for key in (*unit.reference_metadata(cfg),'training_config'):
        broken=dict(package);broken.pop(key)
        with pytest.raises(ValueError):unit.ReferenceArm(broken,'F_CE_2R',0,cfg)
    broken=dict(package,state_dict=dict(package['state_dict']))
    broken['state_dict'].pop(next(iter(broken['state_dict'])))
    with pytest.raises(ValueError):unit.ReferenceArm(broken,'F_CE_2R',0,cfg)
    torsion=torch.load(unit.REFERENCE/'pretrain/T_FCE/deploy_05000.pt',map_location='cpu',weights_only=False)
    with pytest.raises(ValueError):unit.ReferenceArm(torsion,'F_CE_2R',0,cfg)

def test_deployment_binding_preserves_legacy_state_and_forbids_extra_branch():
    cfg=unit.make_config()
    p=dict(torsion_modules=False,torsion_revision=None,endpoint_revision=None,align_enabled=False,state_dict={'example':torch.ones(2)})
    bound=unit.bind_deployment(p,cfg)
    assert bound['state_dict'] is p['state_dict'] and 'reference_revision' not in p
    assert bound['training_config']==cfg
    for name in ('torsion_modules','align_enabled','endpoint_revision'):
        with pytest.raises(ValueError):unit.bind_deployment(dict(p,**{name:True}),cfg)

def test_process_validation_supports_reference_two_rank_and_forwards_no_torsion(monkeypatch,tmp_path):
    import scripts.evaluate_glt_pt_endpoint as endpoint
    import src.training.glt_dual_runtime as runtime
    import src.dataset.glt_dual_pretrain as prep
    import torch.distributed as dist
    class Source:
        entries=[dict(pretrain_role='validation')]*1024
        samples=[(b'x',None)]*1024
        closed=False
        def __len__(self):return 1024
        def close(self):self.closed=True
    source=Source();seen=[]
    monkeypatch.setattr(endpoint,'load_endpoint_subset',lambda *a:dict(indices=np.arange(1024)))
    monkeypatch.setattr(runtime,'open_source',lambda *a,**kw:(source,None))
    monkeypatch.setattr(runtime,'pretrain_source_fields',lambda *a,**kw:(None,None,'text',None,None))
    monkeypatch.setattr(runtime,'apply_geometry_quality',lambda s,*a:s)
    monkeypatch.setattr(runtime,'move_labels',lambda x,*a:x)
    monkeypatch.setattr(unit,'validate_config',lambda c,*a:c)
    def prepare(*a,**kw):seen.append(kw);return kw
    monkeypatch.setattr(prep,'prepare_pretrain_sample',prepare)
    def collate(records):
        n=len(records)
        return Data(bond_length_weight=torch.ones(n),geometry_valid=torch.ones(n,dtype=torch.bool),bond_distance=torch.ones(n)),n
    monkeypatch.setattr(prep,'pretrain_collate',collate)
    monkeypatch.setattr(dist,'barrier',lambda:None);monkeypatch.setattr(dist,'all_reduce',lambda t:t.mul_(2))
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    class Model:
        training=True
        def eval(self):self.training=False
        def train(self,mode):self.training=mode
        def __call__(self,data,n):
            return dict(sums=torch.zeros(3),raw_geometry_sums=torch.zeros(2),extra_sums=torch.zeros(2),
                counts=torch.tensor([n,n,n]),angle_graphs=torch.tensor(n),extra_counts=torch.tensor([n,n]),targets=torch.tensor([n,n,n]))
    cfg=dict(unit.make_config(),cycle_root=str(tmp_path));model=Model();rng=torch.get_rng_state().clone()
    endpoint.evaluate_process_validation(model,cfg,1000,'cpu',0,2)
    assert model.training and source.closed and torch.equal(rng,torch.get_rng_state())
    assert len(seen)==512 and all(r['torsion_mode'] is None and r['torsion_revision'] is None for r in seen)
    assert unit.read(tmp_path/'pt_validation/F_CE_2R/step01000/metrics.json')['presented']==1024
    with pytest.raises(ValueError):endpoint.evaluate_process_validation(model,cfg,1000,'cpu',0,4)

def test_final_freeze_is_complete_and_preserves_original_heads():
    cfg=unit.make_config();frozen=unit.read(unit.OUT/'evaluation_before_test.json')
    assert len(frozen['units'])==frozen['validation_units']==35
    assert frozen['reference_configuration']==unit.read(unit.REFERENCE/'initialization/T_FCE.json')
    for entry in frozen['units']:
        task,fold=entry['task'],entry['fold']
        best=torch.load(entry['best'],map_location='cpu',weights_only=False)
        unit.validate_best(best,'F_CE_2R',task,fold,cfg)
        assert best['best_epoch']==entry['best_epoch']==min(best['history'],key=lambda r:r['validation_rmse'])['epoch']
        assert best['scaler_mean']==entry['scaler']['mean'] and best['scaler_scale']==entry['scaler']['scale']
        assert entry['test_indices']==entry['reference_binding']['test_indices']
        original=torch.load(unit.head_path('F_CE_2R',task,fold),map_location='cpu',weights_only=False)
        saved=torch.load(unit.ft_folder('F_CE_2R',task,fold)/'head_initial.pt',map_location='cpu',weights_only=False)
        assert set(saved)==set(original) and all(torch.equal(v,saved[k]) for k,v in original.items())

def test_freeze_refuses_partial_matrix_and_identity_comparison_rejects_mismatch(monkeypatch,tmp_path):
    monkeypatch.setattr(select,'OUT',tmp_path)
    unit.write(tmp_path/'initialization/F_CE_2R.json',unit.make_config())
    monkeypatch.setattr(select,'ft_folder',lambda arm,task,fold:tmp_path/'finetune'/task/f'fold{fold}')
    with pytest.raises(FileNotFoundError):select.freeze()
    assert not (tmp_path/'evaluation_before_test.json').exists()
    fields=dict(indices=np.array([1]),sample_keys=np.array(['same']),y_true=np.array([2.]),scaler_mean=np.array([1.]),scaler_scale=np.array([3.]),geometry_valid=np.array([True]))
    select.check_prediction_identity(fields,fields,'same')
    for key,value in [('indices',np.array([2])),('y_true',np.array([3.])),('geometry_valid',np.array([False]))]:
        with pytest.raises(ValueError):select.check_prediction_identity(fields,dict(fields,**{key:value}),'changed')

def test_saved_comparison_direction_statistics_and_original_reference_rows():
    result=unit.read(unit.OUT/'final_review/seven_task_results.json')
    baseline=result['arms']['F_CE_2R'];torsion=result['arms']['T_FCE_reused']
    assert set(baseline['tasks'])==set(unit.TASKS)
    assert result['matched_macro_T_minus_F_CE_2R']==pytest.approx(torsion['macro_test_r2']-baseline['macro_test_r2'],abs=1e-14)
    for task in unit.TASKS:
        x=baseline['tasks'][task]
        values=np.array([r['T_FCE_r2']-r['r2'] for r in x['folds']])
        assert x['T_minus_F_CE_2R_r2']['mean']==pytest.approx(values.mean(),abs=1e-14)
        assert x['T_minus_F_CE_2R_r2']['std']==pytest.approx(values.std(ddof=0),abs=1e-14)
        for fold in range(5):
            with np.load(unit.OUT/f'formal/F_CE_2R/{task}/fold{fold}/test_predictions.npz') as p,np.load(unit.REFERENCE/f'formal/T_FCE/{task}/fold{fold}/test_predictions.npz') as ref:
                select.check_prediction_identity(p,ref,'saved')
                assert not p['torsion_available'].any()
    assert not select.qualifies(.7955913245367517) and select.qualifies(.7955913245367518)

def test_scheduler_isolates_gpu2_3_and_final_cpu_environment(monkeypatch,tmp_path):
    import os
    from scripts import run_glt_pt_matched_reference as controller
    assert controller.GPUS==(2,3) and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    monkeypatch.setattr(controller,'LOG',tmp_path);monkeypatch.setattr(controller,'ACTIVE',[])
    calls=[];monkeypatch.setattr(controller.subprocess,'run',lambda args,**kw:calls.append(args))
    for gpu in (2,3):
        controller.launch('ft_'+str(gpu),['python','scripts/run_glt_pt_matched_reference_unit.py'],[gpu])
        env=unit.read(tmp_path/('ft_'+str(gpu))/'environment.json')
        assert env['CUDA_VISIBLE_DEVICES']==str(gpu) and env['logical_device']=='cuda:0'
    controller.launch('pt_F_CE_2R',['torchrun','--nproc_per_node=2'],[2,3])
    assert unit.read(tmp_path/'pt_F_CE_2R/environment.json')['logical_device']=='cuda:LOCAL_RANK'
    controller.launch('final',['python','-m','pytest'],limit=300)
    assert unit.read(tmp_path/'final/environment.json')['CUDA_VISIBLE_DEVICES']==''
    for devices in ([0],[1],[0,1],[2,3,0],[2,3]):
        with pytest.raises(ValueError):controller.launch('ft_bad',['python'],devices)
    assert len(calls)==4

