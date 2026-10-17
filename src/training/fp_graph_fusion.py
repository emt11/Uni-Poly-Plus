"""Private FP graph-fusion FT adapter and allowed-role materialization only."""
import copy
import json
from pathlib import Path
import sqlite3
import numpy as np
import torch
from torch import nn
from src.modules.fp_graph_fusion import FPGraphFusion,FPGraphHead
from src.modules.glt_dual import strict_module_state
from src.modules.glt_dual_pretrain import restore_deployment
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
from src.training.glt_dual_runtime import pretrain_source_fields
from src.training.xc_kernel import generator,fingerprint,FEATURE_SPEC
from scripts.finetune_mcl_ph import CleanDualDataset,clone_clean_record,periodic_tdl_head
from scripts.run_fce_tune_t3b_unit import validate_parent as validate_g05
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/fp_graph_fusion_20261017/f1';LOG=ROOT/'logs/fp_graph_fusion_20261017/f1'
ARMS=('FP_NULL_KFUSE','FP_REAL_MEAN','FP_REAL_KFUSE');REVISION='fp_graph_f1_v1'
PARENT=ROOT/'results/fce_tune_20261015/t3b/pretrain/T3B_G05/deploy_05000.pt'
MODULE_INITIAL=OUT/'initialization/fusion_initial.pt'
FP_SPEC={k:v for k,v in FEATURE_SPEC.items() if k not in ('kernel','returnDistance','kernel_dtype','g05_weight','krr_weight')}
INPUT_SPEC=dict(revision=REVISION,feature_spec=FP_SPEC,normalization='sqrt(512)*actual_input/L2(actual_input)',initial_seed=20261017,new_parameters=2033152,total_parameters=41909496,head_parameters=2558465,
    query='512->128 no bias; divide sqrt512',keys='independent self512/geometry512/fp2048 ->128 no bias',values='independent ->512 with bias; all initialized zero',geometry='encode true incident availability; postnorm zero and mask whole projected candidate',residual=.5)
POLICY=dict(revision=REVISION,arms=list(ARMS),main='FP_REAL_KFUSE',input_spec=INPUT_SPEC,ft_loss='MSE',adaptation_policy=None,recipe='R0',krr='reference only; no new fusion output',task_switching=False)
def parent(arm):
    if arm not in ARMS:raise ValueError('undefined FP graph arm')
    return PARENT
def input_policy(arm):
    parent(arm)
    return dict(INPUT_SPEC,arm=arm,fp_input='constant_0.5' if arm=='FP_NULL_KFUSE' else 'real',routing='uniform_valid' if arm=='FP_REAL_MEAN' else 'softmax_valid',fp_available=True)
def validate_initial(p):
    if p.get('revision')!=REVISION or p.get('seed')!=20261017 or p.get('parameters')!=2033152:raise ValueError('common FP fusion initialization mismatch')
    expected=FPGraphFusion().state_dict();s=p['state_dict']
    if set(s)!=set(expected):raise ValueError('initial fusion schema mismatch')
    for n,v in s.items():
        if v.shape!=expected[n].shape or v.dtype!=torch.float32 or not torch.isfinite(v).all():raise ValueError('initial fusion tensor mismatch '+n)
        if n.startswith('v_proj.') and torch.count_nonzero(v):raise ValueError('nonzero initial V')
    return p


class FPGraphModel(nn.Module):
    def __init__(self,encoder,arm,head_initial,fusion_initial):
        super().__init__();parent(arm)
        strict_module_state(encoder.predictor,head_initial)
        regressor=encoder.predictor;encoder.predictor=nn.Identity()
        self.model=encoder;self.head=FPGraphHead(regressor,constant=arm=='FP_NULL_KFUSE',uniform=arm=='FP_REAL_MEAN')
        strict_module_state(self.head.fusion,validate_initial(fusion_initial)['state_dict'])
        # Intentionally no feature_cache_context: online frozen-encoder features.
    def forward(self,data):
        with torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in self.model.parameters())):
            encoded=self.model.encode(data);h=self.model.fuse(encoded)
        return self.head(h,data.fp_bits,encoded['geometry_valid'])

def validate_model_counts(model):
    named=list(model.named_parameters(remove_duplicate=False));ids=[id(p) for n,p in named]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate parameter registration')
    if (sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.head.parameters()),sum(p.numel() for p in model.head.fusion.parameters()))!=(41909496,2558465,2033152):raise ValueError('FP graph actual parameter counts differ from design')

def build_model(package,cfg,head,initial):
    if cfg.get('parent')!=str(PARENT) or cfg.get('parent_step')!=5000 or cfg.get('input_policy')!=input_policy(cfg.get('arm')) or cfg.get('module_initial')!=str(MODULE_INITIAL):raise ValueError('FP builder parent/policy mismatch')
    validate_g05(package,'T3B_G05');encoder=restore_deployment(package,expected_step=5000)
    if encoder.glt.nonbonded_input is not None or encoder.glt.torsion_mlp is not None:raise ValueError('unexpected G05 extensions')
    encoder.token_fusion.set_implementation('loop');encoder.collect_fusion_attention=False
    model=FPGraphModel(encoder,cfg['arm'],head,initial);validate_model_counts(model);return model

def stage_groups(model,stage,evidence):
    head_ids={id(p) for p in model.head.parameters()};named=dict(model.named_parameters());joint=stage=='joint'
    expected={id(p) for p in model.parameters() if joint or id(p) in head_ids}
    if {id(p) for p in model.parameters() if p.requires_grad}!=expected:raise ValueError('head/joint trainability mismatch')
    if not all(p.requires_grad for p in model.head.fusion.parameters()):raise ValueError('new fusion must train during both stages')
    result=copy.deepcopy(evidence);ids=[]
    if [g['name'] for g in result]!=(['backbone','head'] if joint else ['head']):raise ValueError('optimizer groups mismatch')
    for g in result:
        members=[(n,p) for n,p in named.items() if p.requires_grad and ((id(p) in head_ids)==(g['name']=='head'))]
        if (g['num_tensors'],g['num_parameters'])!=(len(members),sum(p.numel() for n,p in members)):raise ValueError('optimizer group counts mismatch')
        g['parameter_names']=[n for n,p in members];ids.extend(id(p) for n,p in members)
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('optimizer identity omission/duplication')
    return result


class BitCache:
    """SQLite TEXT keys are literal normalized SMILES, never file fingerprints."""
    def __init__(self,path):
        self.db=sqlite3.connect(str(path),timeout=120,isolation_level=None)
        self.db.execute('CREATE TABLE IF NOT EXISTS bits (smiles TEXT PRIMARY KEY, bits BLOB NOT NULL, info TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS counts (id INTEGER PRIMARY KEY CHECK(id=1), generated INTEGER NOT NULL, hits INTEGER NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO counts VALUES (1,0,0)');self.gen=generator()
    def get(self,smiles):
        if not isinstance(smiles,str) or not smiles:raise ValueError('invalid FP source string')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT bits,info FROM bits WHERE smiles=?',(smiles,)).fetchone()
            hit=row is not None
            if row is None:
                count=self.db.execute('SELECT generated FROM counts WHERE id=1').fetchone()[0]
                if count>=5875:raise ValueError('unique FP generation budget exhausted')
                fp,info=fingerprint(smiles,self.gen);bits=np.asarray(list(fp),dtype=np.uint8)
                self.db.execute('INSERT INTO bits VALUES (?,?,?)',(smiles,bits.tobytes(),json.dumps(info)))
                self.db.execute('UPDATE counts SET generated=generated+1 WHERE id=1')
            else:
                bits=np.frombuffer(row[0],dtype=np.uint8).copy();info=json.loads(row[1]);self.db.execute('UPDATE counts SET hits=hits+1 WHERE id=1')
            if bits.shape!=(2048,) or not np.isin(bits,[0,1]).all() or not bits.any():raise ValueError('bad cached FP bits')
            self.db.execute('COMMIT');return torch.from_numpy(bits.copy()).float().reshape(1,2048),dict(info,cache_hit=hit)
        except BaseException:self.db.execute('ROLLBACK');raise
    def close(self):self.db.close()


def fp_collate(samples):
    if not samples:raise ValueError('empty batch')
    for x in samples:
        if x.fp_bits.shape!=(1,2048) or not torch.isfinite(x.fp_bits).all() or not ((x.fp_bits==0)|(x.fp_bits==1)).all() or not x.fp_bits.any():raise ValueError('invalid sample FP')
    batch=dual_glt_collate(samples);batch.fp_bits=torch.cat([x.fp_bits for x in samples],0)
    batch.fp_original_row=torch.tensor([x.fp_original_row for x in samples],dtype=torch.long)
    batch.fp_sample_keys=[x.fp_sample_key for x in samples];return batch


class FPGraphDataset(CleanDualDataset):
    def __init__(self,source,targets,cfg,*,indices,roles,smiles,cache,on_stats=None):
        super().__init__(source,targets);self.cfg=cfg;self.indices=list(indices);self.roles=list(roles);self.smiles=list(smiles);self.cache=cache;self.on_stats=on_stats
        if len(indices)!=len(source) or len(set(indices))!=len(indices) or len(roles)!=len(indices) or len(smiles)!=len(indices) or not set(roles)<={'train','validation','test'}:raise ValueError('FP row/role mismatch')
        if 'test' in roles and set(roles)!={'test'}:raise ValueError('FT and TEST cache must remain separate')
        self.attempted=set();self.rows={};self.publish()
    def statistics(self):
        rows=list(self.rows.values())
        return dict(attempted_rows=len(self.attempted),materialized_graphs=len(rows),input_policy=self.cfg['input_policy'],fp_generated=sum(not r['cache_hit'] for r in rows),fp_hits=sum(r['cache_hit'] for r in rows),rows=rows)
    def publish(self):
        if self.on_stats:self.on_stats(self.statistics())
    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<len(self):raise IndexError(index)
        if index not in self._clean_cache:
            if index in self.attempted:raise RuntimeError('failed materialization cannot retry')
            self.attempted.add(index);self.publish()
            top,tri,text,static,_=pretrain_source_fields(self.source,index)
            if text!=self.smiles[index]:raise ValueError('source FP normalized_smiles/row mismatch')
            data=build_dual_sample(top,tri,text,static=static,geometry_mode='finite_incident')
            data.md200=torch.zeros((1,200));data.md200_valid=torch.zeros(1,dtype=torch.bool)
            bits,info=self.cache.get(text);data.fp_bits=bits;data.fp_original_row=self.indices[index];data.fp_sample_key=self.source.samples[index][0].hex()
            self._clean_cache[index]=data
            self.rows[index]=dict(original_row=self.indices[index],sample_key=data.fp_sample_key,normalized_smiles=text,role=self.roles[index],**info);self.publish()
        data=clone_clean_record(self._clean_cache[index]);data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32);return data
