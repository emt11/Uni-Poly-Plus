"""Fixed Xc Morgan/Tanimoto KRR protocol; no neural model or artifact hashing."""
import json
import warnings
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

ALPHAS=(.1,1.,10.)
CANDIDATE='XC_G05_MKRR50'
REFERENCE_ARM='T3B_G05'
FEATURE_SPEC=dict(kind='MorganBinary',radius=2,fpSize=2048,includeChirality=True,
    useBondTypes=True,countSimulation=False,includeRingMembership=True,
    onlyNonzeroInvariants=False,includeRedundantEnvironments=False,countBounds=None,
    atomInvariantsGenerator=None,bondInvariantsGenerator=None,
    input='unchanged normalized_smiles with dummy connection atoms',kernel='Tanimoto similarity',
    returnDistance=False,kernel_dtype='float64',g05_weight=.5,krr_weight=.5)
LIMITS=dict(fit_attempts=15,fit_completed=15,formal_starts=5,formal_completed=5,prediction_groups=10,
    final_cpu_starts=1,pt_starts=0,neural_ft=0,epochs=0,optimizer_updates=0,gpu_inference=0,workers=0,refits=0)


def finite(value):
    value=np.asarray(value,dtype=np.float64)
    if not np.isfinite(value).all():raise FloatingPointError('nonfinite value')
    return value


def check_budget(budget):
    if set(budget)!=set(LIMITS) or any(not isinstance(budget[k],int) or budget[k]<0 or budget[k]>v for k,v in LIMITS.items()):
        raise ValueError('Xc protocol budget exceeded or schema mismatch')


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(**{k:FEATURE_SPEC[k] for k in (
        'radius','fpSize','includeChirality','useBondTypes','countSimulation','includeRingMembership',
        'onlyNonzeroInvariants','includeRedundantEnvironments','countBounds','atomInvariantsGenerator','bondInvariantsGenerator')})


def fingerprint(smiles,gen):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()==0:raise ValueError('invalid normalized_smiles')
    fp=gen.GetFingerprint(mol)
    if fp.GetNumBits()!=2048 or fp.GetNumOnBits()==0:raise ValueError('empty/invalid Morgan feature')
    return fp,dict(dummy_atoms=sum(a.GetAtomicNum()==0 for a in mol.GetAtoms()),atoms=mol.GetNumAtoms(),on_bits=fp.GetNumOnBits())


def bit_array(fps):
    return np.asarray([list(fp) for fp in fps],dtype=np.uint8)


def restore_bits(bits):
    bits=np.asarray(bits)
    if bits.ndim!=2 or bits.shape[1]!=2048 or not np.isin(bits,[0,1]).all() or not np.all(bits.sum(1)>0):
        raise ValueError('invalid frozen train bit matrix')
    result=[]
    for row in bits:
        fp=DataStructs.ExplicitBitVect(2048);fp.SetBitsFromList(np.flatnonzero(row).tolist());result.append(fp)
    return result


def validate_kernel(kernel,rows,cols,*,training=False):
    k=finite(kernel)
    if k.dtype!=np.float64 or k.shape!=(rows,cols) or rows<=0 or cols<=0 or (k<0).any() or (k>1).any():raise ValueError('illegal similarity kernel')
    if training and (rows!=cols or not np.allclose(k,k.T,rtol=0,atol=1e-14) or not np.allclose(np.diag(k),1,rtol=0,atol=1e-14)):
        raise ValueError('invalid symmetric training kernel')
    return k


def bound_kernel(features,query_indices,train_indices):
    if len(set(train_indices))!=len(train_indices) or len(set(query_indices))!=len(query_indices):raise ValueError('duplicate kernel index')
    query=[features[i] for i in query_indices];train=[features[i] for i in train_indices]
    k=np.asarray([DataStructs.BulkTanimotoSimilarity(fp,train,returnDistance=False) for fp in query],dtype=np.float64)
    return validate_kernel(k,len(query_indices),len(train_indices),training=query_indices==train_indices)


def scaler_values(scaler):
    mean=finite(scaler['mean']);scale=finite(scaler['scale'])
    if mean.shape!=(1,) or scale.shape!=(1,) or scale[0]<=0 or scaler.get('fit')!='train' or scaler.get('transform')!='standard':raise ValueError('invalid train-only scaler')
    return float(mean[0]),float(scale[0])


def standardize(y,scaler):
    mean,scale=scaler_values(scaler);return finite((finite(y)-mean)/scale)


def inverse(z,scaler):
    mean,scale=scaler_values(scaler);return finite(mean+scale*finite(z))


def fuse(g05,krr):
    g05=finite(g05);krr=finite(krr)
    if g05.shape!=krr.shape:raise ValueError('fusion shape mismatch')
    return finite(.5*g05+.5*krr)


def metrics(y,p):
    y=finite(y).reshape(-1);p=finite(p).reshape(-1)
    if y.shape!=p.shape or len(y)<2:raise ValueError('prediction shape mismatch')
    return dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(np.mean((y-p)**2))))


def choose_alpha(rows):
    if len(rows)!=3 or {r['alpha'] for r in rows}!=set(ALPHAS) or any(not np.isfinite(r['fused']['rmse']) for r in rows):raise ValueError('invalid alpha validation set')
    return min(rows,key=lambda r:(r['fused']['rmse'],-r['alpha']))


def fit_once(kernel,z,alpha):
    if alpha not in ALPHAS:raise ValueError('alpha outside authorized set')
    k=validate_kernel(kernel,len(z),len(z),training=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        model=KernelRidge(kernel='precomputed',alpha=alpha).fit(k,finite(z))
    coef=finite(model.dual_coef_).reshape(-1)
    if coef.shape!=(len(z),):raise ValueError('dual coefficients shape mismatch')
    return coef


def predict_dual(kernel,coef,scaler):
    k=finite(kernel);coef=finite(coef).reshape(-1)
    if k.ndim!=2 or k.shape[1]!=len(coef):raise ValueError('frozen train column mismatch')
    return inverse(k@coef,scaler)


def align_reference(pred,rows,scaler):
    """Align by original row AND sample identity, never by assumed file order."""
    indices=np.asarray(pred['indices']).reshape(-1).tolist();keys=np.asarray(pred['sample_keys']).astype(str).reshape(-1).tolist()
    if len(indices)!=len(set(indices)) or len(indices)!=len(rows) or len(keys)!=len(indices):raise ValueError('reference index/identity shape mismatch')
    lookup={v:i for i,v in enumerate(indices)}
    if set(lookup)!={r['original_row'] for r in rows}:raise ValueError('reference sample rows mismatch')
    order=[]
    for row in rows:
        i=lookup[row['original_row']]
        if keys[i]!=row['sample_key']:raise ValueError('reference sample identity mismatch')
        order.append(i)
    mean,scale=scaler_values(scaler)
    if not np.array_equal(np.asarray(pred['scaler_mean']),[mean]) or not np.array_equal(np.asarray(pred['scaler_scale']),[scale]):raise ValueError('reference scaler mismatch')
    truth=finite(pred['y_true']).reshape(-1)[order];value=finite(pred['y_pred']).reshape(-1)[order]
    labels=finite([r['label'] for r in rows])
    tolerance=8*np.finfo(np.float32).eps*(np.abs(labels)+abs(mean)+scale)
    if np.any(np.abs(truth-labels)>tolerance):raise ValueError('reference truth/source label mismatch')
    return truth,value


def validate_reference_protocol(unit):
    cfg=unit['config'];pt=cfg['training_config']
    if unit.get('arm')!=REFERENCE_ARM or cfg.get('arm')!=REFERENCE_ARM or cfg.get('task')!='xc' or pt.get('arm')!=REFERENCE_ARM or pt.get('revision')!='r5' or pt.get('loss_weights')!=[1,.5,0] or pt.get('schedule_total_steps')!=20000 or cfg['recipe'].get('arm')!='R0':
        raise ValueError('reference must be r5 G05 plus R0')


def validate_freeze(payload):
    if not payload.get('frozen') or payload.get('feature_spec')!=FEATURE_SPEC or payload.get('candidate')!=CANDIDATE:
        raise ValueError('all five selections must be frozen before TEST')
    units=payload.get('units',[])
    if len(units)!=5 or {u['fold'] for u in units}!=set(range(5)):raise ValueError('incomplete fixed five-fold freeze')
    for u in units:
        if u['alpha'] not in ALPHAS or len(u['train_indices'])!=276 or len(set(u['train_indices']))!=276 or u.get('fit_count')!=3:
            raise ValueError('invalid frozen train/alpha binding')
        scaler_values(u['scaler'])
    return payload


class PhaseGate:
    def __init__(self):self.payload=None
    def freeze(self,payload):self.payload=validate_freeze(payload)
    def require_test(self):
        if self.payload is None:raise ValueError('TEST access before all five folds freeze')
        validate_freeze(self.payload)


def read_predictions(path,*,gate=None,test=False):
    if test:
        if gate is None:raise ValueError('TEST access requires freeze gate')
        gate.require_test()
    with np.load(Path(path),allow_pickle=False) as p:
        return {k:p[k].copy() for k in ('indices','sample_keys','y_true','y_pred','scaler_mean','scaler_scale')}
