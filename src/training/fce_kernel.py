"""One post-finetuning prediction policy, shared by every F_CE task.

Neural checkpoints retain their original validation selection. Mathematics from
the completed kernel experiments are reused without task-specific overrides.
"""
import copy
import json
from pathlib import Path
import numpy as np
from src.training.xc_kernel import (ALPHAS, FEATURE_SPEC, generator, fingerprint,
    bit_array, restore_bits, bound_kernel, scaler_values, standardize, fuse,
    metrics, choose_alpha, fit_once, predict_dual, align_reference, finite)

TASKS=('eea','egb','egc','ei','eps','nc','xc')
POLICY=dict(name='G05_MKRR50',alphas=list(ALPHAS),feature_spec=copy.deepcopy(FEATURE_SPEC),
    selection='own fold fused validation raw RMSE minimum; exact ties larger alpha',
    neural_selection='existing own-fold validation RMSE best; no retraining',
    task_switching=False)
LIMITS=dict(fit_attempts=75,fit_completed=75,formal_starts=25,formal_completed=25,
    prediction_groups=50,reused_units=10,reused_prediction_groups=20,final_cpu_starts=1,
    pt_starts=0,neural_ft=0,epochs=0,optimizer_updates=0,gpu_inference=0,workers=0,refits=0)


def check_budget(budget):
    if set(budget)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in budget.items()):
        raise ValueError('unified FT budget mismatch')


def validate_split(entry,n):
    ids=[]
    if entry['fold'] not in range(5):raise ValueError('invalid fold')
    for role in ('train','validation','test'):
        values=entry[role+'_indices']
        if not values or any(type(i) is not int for i in values):raise ValueError('invalid split schema')
        ids.extend(values)
    if len(ids)!=n or set(ids)!=set(range(n)):raise ValueError('split overlap/coverage mismatch')


def validate_reference(unit,task,fold,entry,scaler,record):
    cfg=unit['config'];pt=cfg['training_config']
    if (unit['task'],unit['arm'],unit['fold'],unit['seed'])!=(task,'T3B_G05',fold,42+fold):
        raise ValueError('reference task/arm/fold/resolved seed mismatch')
    if cfg['task']!=task or cfg['arm']!='T3B_G05' or cfg['seed']!='42+fold' or cfg['parent']!=unit['parent'] or cfg['parent_step']!=5000:
        raise ValueError('reference config mismatch')
    if pt['arm']!='T3B_G05' or pt['revision']!='r5' or pt['loss_weights']!=[1,.5,0] or pt['schedule_total_steps']!=20000 or cfg['recipe']['arm']!='R0' or cfg['recipe']['task']!=task:
        raise ValueError('reference G05/R0 policy mismatch')
    for role in ('train','validation','test'):
        values=entry[role+'_indices'];expected=sorted(values) if role=='test' else values
        if unit[role+'_indices']!=expected:raise ValueError('reference split/order mismatch')
    if scaler!=unit['scaler'] or record['config']!=cfg or record['status']!='PASS' or record['outer_test']!='NOT_RUN' or record['best_epoch']!=unit['best_epoch']:
        raise ValueError('reference scaler/best mismatch')
    scaler_values(scaler)


def metadata(rows,features=None):
    result=[]
    for row in rows:
        item={k:row[k] for k in ('original_row','sample_key','normalized_smiles')}
        if features is not None:item['on_bits']=features[row['original_row']].GetNumOnBits()
        result.append(item)
    return result


def fit_branch(train,validation,reference,scaler,*,on_fit,on_trial,features=None):
    """Public common FT completion stage; no task argument or policy override."""
    features={} if features is None else features
    gen=generator();mu,sigma=scaler_values(scaler)
    target=finite([r['label'] for r in train])
    if not np.isclose(target.mean(),mu,rtol=1e-12,atol=1e-12) or not np.isclose(target.std(),sigma,rtol=1e-12,atol=1e-12):
        raise ValueError('training labels/train-only scaler mismatch')
    for row in train+validation:
        i=row['original_row']
        if i not in features:features[i]=fingerprint(row['normalized_smiles'],gen)[0]
    ti=[r['original_row'] for r in train];vi=[r['original_row'] for r in validation]
    ktrain=bound_kernel(features,ti,ti);kval=bound_kernel(features,vi,ti)
    truth,g05=align_reference(reference,validation,scaler)
    trials=[];coefficients={}
    for alpha in ALPHAS:
        on_fit('attempted');coef=fit_once(ktrain,standardize(target,scaler),alpha);on_fit('completed')
        branch=predict_dual(kval,coef,scaler);fusion=fuse(g05,branch)
        trial=dict(alpha=alpha,branch=metrics(truth,branch),fused=metrics(truth,fusion),
            reference=metrics(truth,g05),fit_rows=len(train),fit_completed=True)
        values=dict(indices=np.asarray(vi),sample_keys=np.asarray([r['sample_key'] for r in validation]),
            y_true=truth,y_g05=g05,y_krr=branch,y_fused=fusion,
            scaler_mean=np.asarray(scaler['mean']),scaler_scale=np.asarray(scaler['scale']))
        on_trial(alpha,values,trial);trials.append(trial);coefficients[alpha]=coef
    chosen=choose_alpha(trials);alpha=chosen['alpha']
    model=dict(alpha=np.float64(alpha),dual_coef=coefficients[alpha],
        train_bits=bit_array([features[i] for i in ti]),train_indices=np.asarray(ti),
        train_keys=np.asarray([r['sample_key'] for r in train]),
        train_normalized_smiles=np.asarray([r['normalized_smiles'] for r in train]),train_y=target,
        mean=np.float64(mu),scale=np.float64(sigma),feature_spec=json.dumps(FEATURE_SPEC,sort_keys=True))
    return dict(model=model,alpha=alpha,validation_trials=trials,chosen_validation=chosen,
        train_samples=metadata(train,features),validation_samples=metadata(validation,features))


def validate_model(model,unit):
    n=len(unit['train_indices']);mu,sigma=scaler_values(unit['scaler'])
    if float(model['alpha'])!=unit['alpha'] or model['train_indices'].tolist()!=unit['train_indices'] or model['train_keys'].tolist()!=[r['sample_key'] for r in unit['train_samples']] or model['train_normalized_smiles'].tolist()!=[r['normalized_smiles'] for r in unit['train_samples']]:
        raise ValueError('model/selected training identity mismatch')
    if json.loads(str(model['feature_spec']))!=FEATURE_SPEC or float(model['mean'])!=mu or float(model['scale'])!=sigma:
        raise ValueError('model feature policy/scaler mismatch')
    if model['train_bits'].shape!=(n,2048) or model['dual_coef'].shape!=(n,) or not np.isfinite(model['dual_coef']).all():
        raise ValueError('model dimensions/nonfinite coefficients')
    return model


def predict_branch(model,rows,reference,scaler):
    """Identical frozen raw prediction rule for every task/fold."""
    fps=restore_bits(model['train_bits']);ti=model['train_indices'].tolist()
    features=dict(zip(ti,fps));gen=generator();query=[]
    for row in rows:
        i=row['original_row']
        if i in features:raise ValueError('formal query overlaps training set')
        query.append(i);features[i]=fingerprint(row['normalized_smiles'],gen)[0]
    kernel=bound_kernel(features,query,ti)
    truth,g05=align_reference(reference,rows,scaler)
    branch=predict_dual(kernel,model['dual_coef'],scaler)
    return truth,g05,branch,fuse(g05,branch)


class PhaseGate:
    def __init__(self):self.payload=None
    def freeze(self,payload):
        if payload.get('policy')!=POLICY or not payload.get('frozen'):raise ValueError('unified policy/freeze mismatch')
        units=payload['units'];expected={(t,f) for t in TASKS for f in range(5)}
        if len(units)!=35 or {(u['task'],u['fold']) for u in units}!=expected:raise ValueError('all 35 choices required before TEST')
        for u in units:
            validate_split(u,u['sample_count'])
            if u['scaler']!=u['reference']['scaler'] or choose_alpha(u['validation_trials'])!=u['chosen_validation'] or u['alpha']!=u['chosen_validation']['alpha']:
                raise ValueError('frozen choice/scaler mismatch')
        self.payload=copy.deepcopy(payload)
    def require_test(self):
        if self.payload is None:raise ValueError('TEST before all 35 choices freeze')


def new_directory(path):
    p=Path(path);p.mkdir(parents=True,exist_ok=False);return p


# r10 adds an array-only policy. The r9 fitting and fixed-0.5 interfaces above
# remain unchanged so that historical alpha selection is never repeated.
GLOBAL_WEIGHTS=(0.,.25,.5,.75,1.)
GLOBAL_BLEND_POLICY=dict(name='G05_KRR_GLOBAL_VALIDATION_R2',grid=list(GLOBAL_WEIGHTS),
    objective='35 validation R2 arithmetic mean',tie='larger G05 weight',
    alpha_selection='unchanged r9 fixed-0.5 choices',task_switching=False)


def blend_arrays(g05,krr,weight):
    if type(weight) not in (int,float) or weight not in GLOBAL_WEIGHTS:raise ValueError('weight outside fixed global grid')
    g05=finite(g05).reshape(-1);krr=finite(krr).reshape(-1)
    if g05.shape!=krr.shape or len(g05)<2:raise ValueError('branch array shape mismatch')
    return finite(weight*g05+(1-weight)*krr)


def select_global_weight(validation):
    expected={(t,f) for t in TASKS for f in range(5)}
    if len(validation)!=35 or {(r['task'],r['fold']) for r in validation}!=expected:raise ValueError('exactly 35 validation units required')
    scores=[]
    for weight in GLOBAL_WEIGHTS:
        unit_scores=[dict(task=r['task'],fold=r['fold'],r2=metrics(r['y_true'],blend_arrays(r['g05'],r['krr'],weight))['r2']) for r in validation]
        value=float(np.mean([r['r2'] for r in unit_scores]))
        if not np.isfinite(value):raise FloatingPointError('nonfinite global validation score')
        scores.append(dict(weight=weight,mean_validation_r2=value,units=unit_scores))
    best=max(scores,key=lambda s:(s['mean_validation_r2'],s['weight']))
    return dict(policy=copy.deepcopy(GLOBAL_BLEND_POLICY),weight=best['weight'],scores=scores)


def bind_branch_arrays(values,unit,role,identities,value_keys):
    """Reorder stored raw-unit arrays by both original row and frozen sample key."""
    if role not in ('validation','test'):raise ValueError('no training labels used for global blend')
    indices=unit[role+'_indices'];saved=np.asarray(values['indices']).reshape(-1).tolist()
    keys=np.asarray(values['sample_keys']).astype(str).reshape(-1).tolist()
    if len(saved)!=len(indices) or len(set(saved))!=len(saved) or set(saved)!=set(indices) or len(keys)!=len(saved):raise ValueError('stored array index/shape mismatch')
    lookup={idx:n for n,idx in enumerate(saved)};order=[lookup[i] for i in indices]
    if any(keys[lookup[i]]!=identities[str(i)]['sample_key'] for i in indices):raise ValueError('stored array sample identity mismatch')
    mean,scale=scaler_values(unit['scaler'])
    if not np.array_equal(values['scaler_mean'],[mean]) or not np.array_equal(values['scaler_scale'],[scale]):raise ValueError('stored array scaler mismatch')
    if 'alpha' in values and float(values['alpha'])!=unit['alpha']:raise ValueError('stored array alpha mismatch')
    result={}
    for k in ('y_true',*value_keys):
        v=finite(values[k]).reshape(-1)
        if len(v)!=len(indices):raise ValueError('stored prediction length mismatch')
        result[k]=v[order]
    return result


class GlobalBlendGate:
    def __init__(self):self.payload=None
    def freeze(self,payload):
        if not payload.get('frozen') or payload.get('policy')!=GLOBAL_BLEND_POLICY:raise ValueError('global blend policy must freeze before TEST')
        old=PhaseGate();old.freeze(payload['r9_freeze'])
        scores=payload['scores']
        if len(scores)!=5 or {s['weight'] for s in scores}!=set(GLOBAL_WEIGHTS):raise ValueError('global weight score grid mismatch')
        expected={(t,f) for t in TASKS for f in range(5)}
        for s in scores:
            if len(s['units'])!=35 or {(r['task'],r['fold']) for r in s['units']}!=expected:raise ValueError('global validation unit coverage mismatch')
            actual=float(np.mean(finite([r['r2'] for r in s['units']])))
            if actual!=s['mean_validation_r2']:raise ValueError('global objective must equally weight 35 validation R2')
        winner=max(scores,key=lambda s:(s['mean_validation_r2'],s['weight']))
        if payload['weight']!=winner['weight']:raise ValueError('global weight must follow validation ranking/tie')
        self.payload=copy.deepcopy(payload)
    def require_test(self):
        if self.payload is None:raise ValueError('TEST before unique global weight freeze')


def blend_frozen_predictions(g05,krr,gate):
    """Public post-FT array interface: one frozen weight, no task argument."""
    gate.require_test()
    return blend_arrays(g05,krr,gate.payload['weight'])


ALIGNED_ALPHA_POLICY=dict(name='G05_KRR_ALIGNED_ALPHA',alpha_selection_weight=.75,
    prediction_weight=.75,alphas=list(ALPHAS),objective='own fold raw fused validation RMSE',
    tie='larger alpha',task_switching=False,
    feature_metadata='unchanged historical FEATURE_SPEC; its .5 fields are not prediction weights')


def select_aligned_alpha(unit,caches,identities):
    """All three cached alphas are bound to their own alpha, not the old choice."""
    if set(caches)!=set(ALPHAS):raise ValueError('exactly three alpha caches required')
    trials=[];truth=None;g05=None
    for alpha in ALPHAS:
        own=dict(unit,alpha=alpha)
        p=bind_branch_arrays(caches[alpha],own,'validation',identities,('y_g05','y_krr'))
        if truth is not None and (not np.array_equal(truth,p['y_true']) or not np.array_equal(g05,p['y_g05'])):
            raise ValueError('cached alpha label/G05 arrays mismatch')
        truth=p['y_true'];g05=p['y_g05'];branch=p['y_krr']
        trials.append(dict(alpha=alpha,fused=metrics(truth,blend_arrays(g05,branch,.75)),
            reference=metrics(truth,g05),branch=metrics(truth,branch)))
    chosen=choose_alpha(trials)
    return dict(old_alpha=unit['alpha'],alpha=chosen['alpha'],changed=chosen['alpha']!=unit['alpha'],
        validation_trials=trials,chosen_validation=chosen,
        alpha_selection_weight=.75,prediction_weight=.75)


def restore_aligned_model(old_model,unit,alpha,*,on_fit,cached_models=()):
    """No fit when unchanged; otherwise reuse exact coefficients or fit once."""
    validate_model(old_model,unit)
    if alpha==unit['alpha']:return old_model,'unchanged'
    if alpha not in ALPHAS:raise ValueError('alpha outside fixed grid')
    expected=dict(unit,alpha=alpha)
    for model in cached_models:
        validate_model(model,expected)
        if not np.array_equal(model['train_bits'],old_model['train_bits']) or not np.array_equal(model['train_y'],old_model['train_y']):raise ValueError('coefficient cache training problem mismatch')
        return model,'reused_coefficients'
    y=finite(old_model['train_y']).reshape(-1);mu,sigma=scaler_values(unit['scaler'])
    if len(y)!=len(unit['train_indices']) or not np.isclose(y.mean(),mu,rtol=1e-12,atol=1e-12) or not np.isclose(y.std(),sigma,rtol=1e-12,atol=1e-12):raise ValueError('saved train labels/scaler mismatch')
    indices=unit['train_indices'];features=dict(zip(indices,restore_bits(old_model['train_bits'])))
    kernel=bound_kernel(features,indices,indices)
    on_fit('attempted');coef=fit_once(kernel,standardize(y,unit['scaler']),alpha);on_fit('completed')
    result={k:np.asarray(v).copy() for k,v in old_model.items()};result.update(alpha=np.float64(alpha),dual_coef=coef)
    validate_model(result,expected)
    return result,'fitted'


class AlignedAlphaGate:
    def __init__(self):self.payload=None
    def freeze(self,payload):
        if payload.get('policy')!=ALIGNED_ALPHA_POLICY or not payload.get('frozen'):raise ValueError('aligned policy freeze required')
        base=GlobalBlendGate();base.freeze(payload['r10_freeze'])
        if base.payload['weight']!=.75:raise ValueError('r11 requires fixed r10 weight .75')
        old={(u['task'],u['fold']):u for u in base.payload['r9_freeze']['units']}
        units=payload['units']
        if len(units)!=35 or {(u['task'],u['fold']) for u in units}!=set(old):raise ValueError('all 35 aligned choices/models required')
        for u in units:
            prior=old[(u['task'],u['fold'])]
            for field in ('reference','scaler','train_indices','validation_indices','test_indices','train_samples','validation_samples'):
                if u[field]!=prior[field]:raise ValueError('aligned source binding changed')
            if u['old_alpha']!=prior['alpha'] or u['changed']!=(u['alpha']!=u['old_alpha']) or u['chosen_validation']!=choose_alpha(u['validation_trials']) or u['alpha']!=u['chosen_validation']['alpha']:
                raise ValueError('aligned choice/old alpha mismatch')
            if u['alpha_selection_weight']!=.75 or u['prediction_weight']!=.75 or not u.get('model_ready'):
                raise ValueError('aligned weights/model readiness mismatch')
        if payload['K']!=sum(u['changed'] for u in units):raise ValueError('changed alpha count mismatch')
        self.payload=copy.deepcopy(payload)
    def require_test(self):
        if self.payload is None:raise ValueError('TEST before all aligned models freeze')


def predict_aligned_branch(model,unit,reference,identities,gate,*,on_query):
    """One needed CPU KRR forward, with train bits restored and only TEST features."""
    gate.require_test()
    if not unit['changed']:raise ValueError('unchanged alpha must reuse r10 without a model forward')
    validate_model(model,unit);indices=unit['test_indices'];rows=[]
    # Reference supplies raw truth; samples and original SMILES remain frozen.
    bound=bind_branch_arrays(reference,unit,'test',identities,('y_pred',))
    for i,y in zip(indices,bound['y_true']):rows.append(dict(original_row=i,label=float(y),**identities[str(i)]))
    fps=restore_bits(model['train_bits']);features=dict(zip(unit['train_indices'],fps));gen=generator()
    for row in rows:
        if row['original_row'] in features:raise ValueError('TEST/train overlap')
        on_query();features[row['original_row']]=fingerprint(row['normalized_smiles'],gen)[0]
    kernel=bound_kernel(features,indices,unit['train_indices'])
    branch=predict_dual(kernel,model['dual_coef'],unit['scaler'])
    return bound['y_true'],bound['y_pred'],branch,blend_arrays(bound['y_pred'],branch,.75)


# r12 is an explicit new representation, not a mutation of historical schemas.
RADIUS3_FEATURE_SPEC={k:copy.deepcopy(v) for k,v in FEATURE_SPEC.items() if k not in ('g05_weight','krr_weight')}
RADIUS3_FEATURE_SPEC['radius']=3
RADIUS3_POLICY=dict(name='G05_KRR_RADIUS3',feature_spec=RADIUS3_FEATURE_SPEC,
    prediction_weight=.75,alpha_source='r11 frozen choices',validation='diagnostic only',task_switching=False)


def radius3_generator():
    from rdkit.Chem import rdFingerprintGenerator
    keys=('radius','fpSize','includeChirality','useBondTypes','countSimulation','includeRingMembership',
        'onlyNonzeroInvariants','includeRedundantEnvironments','countBounds','atomInvariantsGenerator','bondInvariantsGenerator')
    return rdFingerprintGenerator.GetMorganGenerator(**{key:RADIUS3_FEATURE_SPEC[key] for key in keys})


class Radius3Cache:
    """Single-process pure structure cache; no labels, coefficients or predictions."""
    def __init__(self,on_query,feature_spec=None):
        self.spec=copy.deepcopy(RADIUS3_FEATURE_SPEC if feature_spec is None else feature_spec)
        if self.spec!=RADIUS3_FEATURE_SPEC:raise ValueError('radius3 cache feature policy mismatch')
        self.on_query=on_query;self.entries={};self.gen=radius3_generator()
    def get(self,task,row,role,gate=None):
        if task not in TASKS or role not in ('train','validation','test'):raise ValueError('invalid feature task/role')
        if role=='test':
            if gate is None:raise ValueError('TEST feature requires global freeze')
            gate.require_test()
        if self.spec!=RADIUS3_FEATURE_SPEC:raise ValueError('cache policy changed')
        key=(task,row['original_row']);identity=(row['sample_key'],row['normalized_smiles'])
        if key in self.entries:
            old,fp=self.entries[key]
            if old!=identity:raise ValueError('structure cache row identity conflict')
            self.on_query(role,True);return fp
        self.on_query(role,False);fp=fingerprint(row['normalized_smiles'],self.gen)[0]
        self.entries[key]=(identity,fp);return fp


def fit_radius3_branch(old_model,unit,cache,*,on_fit):
    validate_model(old_model,unit)
    y=finite(old_model['train_y']).reshape(-1);mu,sigma=scaler_values(unit['scaler'])
    if len(y)!=len(unit['train_indices']) or not np.isclose(y.mean(),mu,rtol=1e-12,atol=1e-12) or not np.isclose(y.std(),sigma,rtol=1e-12,atol=1e-12):raise ValueError('radius3 original train target/scaler mismatch')
    features={r['original_row']:cache.get(unit['task'],r,'train') for r in unit['train_samples']}
    indices=unit['train_indices'];kernel=bound_kernel(features,indices,indices)
    on_fit('attempted');coef=fit_once(kernel,standardize(y,unit['scaler']),unit['alpha']);on_fit('completed')
    model=dict(alpha=np.float64(unit['alpha']),dual_coef=coef,train_bits=bit_array([features[i] for i in indices]),
        train_indices=np.asarray(indices),train_keys=np.asarray([r['sample_key'] for r in unit['train_samples']]),
        train_normalized_smiles=np.asarray([r['normalized_smiles'] for r in unit['train_samples']]),train_y=y,
        mean=np.float64(mu),scale=np.float64(sigma),feature_spec=json.dumps(RADIUS3_FEATURE_SPEC,sort_keys=True),
        prediction_weight=np.float64(.75))
    return validate_radius3_model(model,unit)


def validate_radius3_model(model,unit):
    n=len(unit['train_indices']);mu,sigma=scaler_values(unit['scaler'])
    if json.loads(str(model['feature_spec']))!=RADIUS3_FEATURE_SPEC or float(model['prediction_weight'])!=.75:raise ValueError('radius3 model feature/weight schema mismatch')
    if float(model['alpha'])!=unit['alpha'] or model['train_indices'].tolist()!=unit['train_indices'] or model['train_keys'].tolist()!=[r['sample_key'] for r in unit['train_samples']] or model['train_normalized_smiles'].tolist()!=[r['normalized_smiles'] for r in unit['train_samples']]:raise ValueError('radius3 model alpha/train column binding mismatch')
    if float(model['mean'])!=mu or float(model['scale'])!=sigma or model['train_bits'].shape!=(n,2048) or model['dual_coef'].shape!=(n,) or finite(model['train_y']).shape!=(n,):raise ValueError('radius3 model shape/scaler mismatch')
    finite(model['dual_coef']);restore_bits(model['train_bits'])
    return model


def predict_radius3_branch(model,unit,reference,identities,cache,role,gate=None):
    if role=='test':
        if gate is None:raise ValueError('radius3 TEST requires freeze')
        gate.require_test()
    if role not in ('validation','test'):raise ValueError('invalid prediction role')
    validate_radius3_model(model,unit)
    key='y_g05' if role=='validation' else 'y_pred'
    bound=bind_branch_arrays(reference,unit,role,identities,(key,));indices=unit[role+'_indices']
    features=dict(zip(unit['train_indices'],restore_bits(model['train_bits'])))
    for i in indices:
        if i in features:raise ValueError('radius3 query overlaps train columns')
        features[i]=cache.get(unit['task'],dict(original_row=i,**identities[str(i)]),role,gate)
    kernel=bound_kernel(features,indices,unit['train_indices'])
    branch=predict_dual(kernel,model['dual_coef'],unit['scaler'])
    return bound['y_true'],bound[key],branch,blend_arrays(bound[key],branch,.75)


class Radius3Gate:
    def __init__(self):self.payload=None
    def freeze(self,payload):
        if not payload.get('frozen') or payload.get('policy')!=RADIUS3_POLICY:raise ValueError('radius3 frozen policy required')
        base=AlignedAlphaGate();base.freeze(payload['r11_freeze'])
        old={(u['task'],u['fold']):u for u in base.payload['units']};units=payload['units']
        if len(units)!=35 or {(u['task'],u['fold']) for u in units}!=set(old):raise ValueError('35 radius3 models must freeze before TEST')
        for u in units:
            prior=old[(u['task'],u['fold'])]
            for key in ('alpha','reference','scaler','train_indices','validation_indices','test_indices','train_samples','validation_samples'):
                if u[key]!=prior[key]:raise ValueError('radius3 inherited binding mismatch')
            if not u.get('radius3_model_ready') or u.get('model_spec')!=RADIUS3_FEATURE_SPEC or u['model']==prior['model'] or u.get('prediction_weight')!=.75:raise ValueError('radius3 requires new model/schema for every unit')
        self.payload=copy.deepcopy(payload)
    def require_test(self):
        if self.payload is None:raise ValueError('TEST before all 35 radius3 models freeze')
