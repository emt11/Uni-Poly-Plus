"""r16 private matched finite-Trimer distance inputs; public scientific code read-only."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from src.dataset.glt_r10 import attach_nonbonded
from src.modules.glt_r10 import NonbondedInput
from src.modules.glt_dual_pretrain import restore_deployment
from src.modules.glt_dual import strict_module_state
from src.dataset.glt_dual import build_dual_sample
from src.training.glt_dual_runtime import pretrain_source_fields
from scripts.finetune_mcl_ph import CleanDualDataset,clone_clean_record,periodic_tdl_head
from scripts.run_fce_tune_t3b_unit import Fused,validate_parent as validate_g05,pt_config_for
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/fce_tune_20261017/nonbonded_ft';LOG=ROOT/'logs/fce_tune_20261017/nonbonded_ft'
ARMS=('NB_CONST','NB_REAL');REVISION='fce_nonbonded_ft_r16_v1';NB_REVISION='glt_nonbonded_r10_g1'
PARENT=ROOT/'results/fce_tune_20261015/t3b/pretrain/T3B_G05/deploy_05000.pt'
MODULE_INITIAL=OUT/'initialization/nonbonded_initial.pt'
INPUT_SPEC=dict(revision=REVISION,nonbonded_revision=NB_REVISION,relations='existing finite Trimer shortest three-bond heavy-atom endpoints',
    constant_angstrom=3.0,rbf_centers=[0.,6.,16],rbf_width=.4,module='NonbondedInput 16-64-512 GELU',
    initial_seed=314173,initial_parameters=34368,total_parameters=39910712,masked_degenerate='keep relation, zero masked increment')
POLICY=dict(revision=REVISION,arms=list(ARMS),input_spec=INPUT_SPEC,weight=.75,ft_loss='MSE',adaptation_policy=None,
    recipe='R0',krr='r10 radius2 frozen alpha and arrays',task_switching=False)


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def parent(arm):
    if arm not in ARMS:raise ValueError('undefined nonbonded arm')
    return PARENT
def input_policy(arm):
    if arm not in ARMS:raise ValueError('undefined nonbonded arm')
    return dict(INPUT_SPEC,arm=arm,distance_mode='constant_3A' if arm=='NB_CONST' else 'real_original_trimer')
def validate_initial(package):
    if package.get('revision')!=REVISION or package.get('seed')!=314173 or package.get('parameters')!=34368:
        raise ValueError('common nonbonded initialization metadata mismatch')
    state=package['state_dict'];shapes={'net.0.weight':(64,16),'net.0.bias':(64,),'net.2.weight':(512,64),'net.2.bias':(512,)}
    if set(state)!=set(shapes):raise ValueError('common nonbonded initialization keys mismatch')
    for n,shape in shapes.items():
        if state[n].shape!=shape or state[n].dtype!=torch.float32 or not torch.isfinite(state[n]).all():raise ValueError('invalid common nonbonded tensor '+n)
    if torch.count_nonzero(state['net.2.weight']) or torch.count_nonzero(state['net.2.bias']):raise ValueError('nonzero common output initialization')
    return package

def validate_model_counts(model):
    if sum(p.numel() for p in model.model.glt.nonbonded_input.parameters())!=34368 or sum(p.numel() for p in model.parameters())!=39910712:raise ValueError('extended FT parameter count mismatch')

def build_model(package,cfg,head,initial):
    if cfg.get('parent')!=str(PARENT) or cfg.get('parent_step')!=5000 or cfg.get('input_policy')!=input_policy(cfg.get('arm')) or cfg.get('module_initial')!=str(MODULE_INITIAL):raise ValueError('nonbonded builder parent/policy mismatch')
    validate_g05(package,'T3B_G05');validate_initial(initial)
    # Restore the unmodified old deployment FIRST, then attach the opt-in module.
    encoder=restore_deployment(package,expected_step=5000)
    if encoder.glt.nonbonded_input is not None or encoder.glt.nonbonded_revision is not None or encoder.glt.torsion_mlp is not None or encoder.glt.geometry_mode!='finite_incident':raise ValueError('unexpected G05 extension')
    extra=NonbondedInput();strict_module_state(extra,initial['state_dict'])
    encoder.glt.nonbonded_input=extra;encoder.glt.nonbonded_revision=NB_REVISION
    encoder.token_fusion.set_implementation('loop');encoder.collect_fusion_attention=False
    model=Fused(encoder,'none');strict_module_state(periodic_tdl_head(model),head)
    validate_model_counts(model)
    return model

def stage_groups(model,stage,evidence):
    head_ids={id(p) for p in periodic_tdl_head(model).parameters()};named=dict(model.named_parameters());joint=stage=='joint'
    expected={id(p) for p in model.parameters() if joint or id(p) in head_ids}
    if {id(p) for p in model.parameters() if p.requires_grad}!=expected:raise ValueError('head/joint adaptation mismatch')
    if any(p.requires_grad!=joint for p in model.model.glt.nonbonded_input.parameters()):raise ValueError('new module freeze/joint mismatch')
    result=copy.deepcopy(evidence);all_ids=[]
    if [g['name'] for g in result]!=(['backbone','head'] if joint else ['head']):raise ValueError('unexpected optimizer groups')
    for g in result:
        members=[(n,p) for n,p in named.items() if p.requires_grad and ((id(p) in head_ids)==(g['name']=='head'))]
        if (g['num_tensors'],g['num_parameters'])!=(len(members),sum(p.numel() for n,p in members)):raise ValueError('optimizer group counts mismatch')
        g['parameter_names']=[n for n,p in members];all_ids.extend(id(p) for n,p in members)
    if len(all_ids)!=len(set(all_ids)) or set(all_ids)!=expected:raise ValueError('optimizer identity duplication/omission')
    return result


def apply_input(data,static,trimer,arm):
    input_policy(arm)
    available=bool(data.geometry_valid)
    if static is None or bool(static['geometry_valid'])!=available or bool(trimer.trimer_geometry_valid)!=available:raise ValueError('original geometry availability mismatch')
    if available:
        z=torch.as_tensor(trimer.trimer_atomic_number).reshape(-1)
        base=torch.as_tensor(trimer.trimer_base_ru_atom_id).reshape(-1);offset=torch.as_tensor(trimer.trimer_ru_offset).reshape(-1)
        if len(base)!=len(z) or len(offset)!=len(z):raise ValueError('physical atom/image size mismatch')
        identities=[(int(b),int(o)) for b,o,zz in zip(base,offset,z) if zz>1]
        if len(identities)!=len(set(identities)) or any(b<0 for b,o in identities):raise ValueError('duplicate/invalid heavy atom image identity')
    attach_nonbonded(data,static,trimer,getattr(trimer,'trimer_pos',torch.empty((0,3))))
    raw=data.nonbonded_distance.clone();index=data.nonbonded_bond_index;mask=data.nonbonded_mask
    if index.numel() and (int(index.min())<0 or int(index.max())>=data.bond_distance.numel()):raise ValueError('central physical bond index mismatch')
    data.nonbonded_raw_distance=raw.clone();data.nonbonded_distance=raw.clone()
    if arm=='NB_CONST':data.nonbonded_distance[mask]=3.0
    if not torch.isfinite(data.nonbonded_distance).all():raise FloatingPointError('nonfinite nonbonded input')
    valid=raw[mask];used=data.nonbonded_distance[mask]
    stats=dict(geometry_valid=available,relations=int(index.numel()),valid_relations=int(mask.sum()),degenerate_masked=int((~mask).sum()),
        no_path=bool(available and not index.numel()),invalid_geometry=not available,
        raw_min=float(valid.min()) if valid.numel() else None,raw_max=float(valid.max()) if valid.numel() else None,
        input_min=float(used.min()) if used.numel() else None,input_max=float(used.max()) if used.numel() else None)
    return data,stats


class NonbondedDataset(CleanDualDataset):
    """One construction per allowed row; immutable CPU cache, fresh clones/y."""
    def __init__(self,source,targets,cfg,*,indices,roles,on_stats=None):
        super().__init__(source,targets);self.cfg=cfg;self.indices=list(indices);self.roles=list(roles);self.on_stats=on_stats
        if len(indices)!=len(source) or len(set(indices))!=len(indices) or len(roles)!=len(indices) or not set(roles)<={'train','validation','test'}:raise ValueError('materialization row/role mismatch')
        if ('test' in roles) and set(roles)!={'test'}:raise ValueError('TEST and FT cache must remain separate')
        self.attempted=set();self.rows={};self.publish()
    def statistics(self):
        result=dict(attempted_rows=len(self.attempted),materialized_graphs=len(self.rows),input_policy=self.cfg['input_policy'],by_role={},rows=list(self.rows.values()))
        for role in sorted(set(self.roles)):
            rows=[r for r in self.rows.values() if r['role']==role]
            totals={k:sum(int(r[k]) for r in rows) for k in ('relations','valid_relations','degenerate_masked','no_path','invalid_geometry','geometry_valid')}
            for name in ('raw_min','input_min'):totals[name]=min((r[name] for r in rows if r[name] is not None),default=None)
            for name in ('raw_max','input_max'):totals[name]=max((r[name] for r in rows if r[name] is not None),default=None)
            result['by_role'][role]=dict(graphs=len(rows),**totals)
        return result
    def publish(self):
        if self.on_stats:self.on_stats(self.statistics())
    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<len(self):raise IndexError(index)
        if index not in self._clean_cache:
            if index in self.attempted:raise RuntimeError('failed materialization cannot be retried')
            self.attempted.add(index);self.publish()
            top,tri,text,static,_=pretrain_source_fields(self.source,index)
            data=build_dual_sample(top,tri,text,static=static,geometry_mode='finite_incident')
            data,stats=apply_input(data,static,tri,self.cfg['arm'])
            data.md200=torch.zeros((1,200));data.md200_valid=torch.zeros(1,dtype=torch.bool)
            self._clean_cache[index]=data
            self.rows[index]=dict(original_row=self.indices[index],sample_key=self.source.samples[index][0].hex(),role=self.roles[index],**stats);self.publish()
        data=clone_clean_record(self._clean_cache[index]);data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32)
        return data
