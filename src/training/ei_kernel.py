"""r8 Ei-only identity, reference, selection and freeze bindings.

Task-independent mathematics are reused read-only from the fixed r7 implementation.
"""
import copy
from pathlib import Path
import numpy as np
from src.training.xc_kernel import (ALPHAS,FEATURE_SPEC,REFERENCE_ARM,LIMITS,
    generator,fingerprint,bit_array,restore_bits,bound_kernel,validate_kernel,
    scaler_values,standardize,fuse,metrics,choose_alpha,fit_once,predict_dual,
    align_reference,read_predictions,check_budget,finite)

CANDIDATE='EI_G05_MKRR50'
TASK='ei'
COUNTS=dict(train=236,validation=60,test=74)


def new_directory(path):
    """An existing writer/output is an error, never a resume or overwrite."""
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path


def validate_split(entry):
    if entry.get('fold') not in range(5):raise ValueError('Ei fold outside fixed budget')
    ids=[]
    for role,count in COUNTS.items():
        values=entry[role+'_indices']
        if len(values)!=count or any(type(v) is not int for v in values):raise ValueError('Ei role count/index schema mismatch')
        ids.extend(values)
    if len(ids)!=370 or set(ids)!=set(range(370)):raise ValueError('Ei split overlap or missing identities')


def validate_reference_protocol(unit):
    cfg=unit['config'];pt=cfg['training_config']
    if unit.get('task')!=TASK or unit.get('arm')!=REFERENCE_ARM or cfg.get('arm')!=REFERENCE_ARM or cfg.get('task')!=TASK or pt.get('arm')!=REFERENCE_ARM or pt.get('revision')!='r5' or pt.get('loss_weights')!=[1,.5,0] or pt.get('schedule_total_steps')!=20000 or cfg['recipe'].get('arm')!='R0' or cfg['recipe'].get('task')!=TASK:
        raise ValueError('Ei reference must be r5 G05 plus R0, task ei')


def validate_reference_binding(unit,fold,entry,scaler,validation_metrics,*,parent,head,best):
    validate_reference_protocol(unit);validate_split(entry);cfg=unit['config'];m=validation_metrics
    if unit['fold']!=fold or entry['fold']!=fold or unit['seed']!=42+fold or cfg['seed']!='42+fold' or unit['parent']!=parent or cfg['parent']!=parent or cfg['parent_step']!=5000 or unit['head_initial']!=head or unit['best']!=best:
        raise ValueError('Ei parent/head/best/seed/fold mismatch')
    for role in COUNTS:
        k=role+'_indices';expected=sorted(entry[k]) if role=='test' else entry[k]
        if unit[k]!=expected:raise ValueError('Ei reference split/order mismatch')
    if scaler!=unit['scaler'] or m['status']!='PASS' or m['config']!=cfg or m['best_epoch']!=unit['best_epoch'] or m['outer_test']!='NOT_RUN':
        raise ValueError('Ei reference validation/scaler/best metadata mismatch')
    scaler_values(scaler)


def collect_sample_identity(units):
    identity={}
    for unit in units:
        for role in ('train_samples','validation_samples'):
            for row in unit[role]:
                key=str(row['original_row'])
                value=dict(sample_key=row['sample_key'],normalized_smiles=row['normalized_smiles'])
                if key in identity and identity[key]!=value:
                    raise ValueError('Ei conflicting source identity across folds')
                identity[key]=value
    return identity


def validate_freeze(payload):
    if not payload.get('frozen') or payload.get('candidate')!=CANDIDATE or payload.get('task')!=TASK or payload.get('feature_spec')!=FEATURE_SPEC:
        raise ValueError('Ei complete freeze required before TEST')
    identity=payload.get('sample_identity',{})
    if set(identity)!=set(map(str,range(370))):raise ValueError('Ei all-sample frozen identity coverage mismatch')
    units=payload.get('units',[])
    if len(units)!=5 or {u['fold'] for u in units}!=set(range(5)):raise ValueError('all five Ei folds must freeze before TEST')
    for u in units:
        validate_split(u);validate_reference_protocol(u['reference'])
        if u['alpha'] not in ALPHAS or u.get('fit_count')!=3:raise ValueError('Ei frozen alpha/fit count mismatch')
        if choose_alpha(u['validation_trials'])!=u['chosen_validation'] or u['chosen_validation']['alpha']!=u['alpha']:
            raise ValueError('Ei selected alpha must follow fused validation')
        if u['scaler']!=u['reference']['scaler']:raise ValueError('Ei frozen scaler/reference mismatch')
        scaler_values(u['scaler'])
        for role in COUNTS:
            values=u[role+'_indices'];expected=sorted(values) if role=='test' else values
            if u['reference'][role+'_indices']!=expected:raise ValueError('Ei frozen split/reference mismatch')
        for role in ('train','validation'):
            samples=u[role+'_samples']
            if [r['original_row'] for r in samples]!=u[role+'_indices'] or any(identity[str(r['original_row'])]!=dict(sample_key=r['sample_key'],normalized_smiles=r['normalized_smiles']) for r in samples):
                raise ValueError('Ei frozen sample identity/order mismatch')
        if u['reference']['fold']!=u['fold'] or u['reference']['seed']!=42+u['fold']:raise ValueError('Ei frozen reference fold/seed mismatch')
    return payload


class PhaseGate:
    def __init__(self):self.payload=None
    def freeze(self,payload):self.payload=copy.deepcopy(validate_freeze(payload))
    def require_test(self):
        if self.payload is None:raise ValueError('Ei TEST access before all five folds freeze')
        validate_freeze(self.payload)
