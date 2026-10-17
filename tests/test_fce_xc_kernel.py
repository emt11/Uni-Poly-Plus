"""Small CPU checks for the fixed r7 protocol; no real-data experiments."""
import copy
import numpy as np
import pytest
from rdkit import DataStructs
from src.training.xc_kernel import (ALPHAS,CANDIDATE,FEATURE_SPEC,LIMITS,PhaseGate,
    align_reference,bit_array,bound_kernel,check_budget,choose_alpha,fingerprint,
    fuse,generator,inverse,predict_dual,read_predictions,restore_bits,scaler_values,
    standardize,validate_freeze,validate_kernel,validate_reference_protocol)

SCALER=dict(mean=[10.],scale=[2.],fit='train',transform='standard')

def vector(bits):
    v=DataStructs.ExplicitBitVect(2048);v.SetBitsFromList(bits);return v


def test_tanimoto_direction_and_order():
    features={8:vector([0,1]),2:vector([1,2]),5:vector([0,1,2])}
    k=bound_kernel(features,[5],[2,8])
    np.testing.assert_allclose(k,[[2/3,2/3]])
    np.testing.assert_allclose(bound_kernel(features,[8],[8,2]),[[1,1/3]])
    assert k.dtype==np.float64
    np.testing.assert_allclose(bound_kernel(features,[8],[2,8]),[[1/3,1]])
    with pytest.raises(ValueError):bound_kernel(features,[8],[8,8])
    with pytest.raises(ValueError):validate_kernel([[0.,1.],[1.,0.]],2,2,training=True)
    with pytest.raises(FloatingPointError):validate_kernel([[np.nan]],1,1)


def test_scaling_fixed_raw_fusion_and_train_columns():
    y=np.array([6.,14.]);np.testing.assert_allclose(inverse(standardize(y,SCALER),SCALER),y)
    np.testing.assert_allclose(fuse([4.,8.],[10.,12.]),[7.,10.])
    np.testing.assert_allclose(predict_dual([[1.,0.]],[2.,9.],SCALER),[14.])
    with pytest.raises(ValueError):predict_dual([[1.]],[2.,9.],SCALER)
    with pytest.raises(ValueError):scaler_values(dict(SCALER,fit='train+validation'))


def test_alpha_uses_fused_validation_and_exact_tie():
    rows=[dict(alpha=a,fused=dict(rmse=2.),branch=dict(rmse=100-a)) for a in ALPHAS]
    assert choose_alpha(rows)['alpha']==10.
    rows[0]['fused']['rmse']=np.nextafter(2.,0.)
    assert choose_alpha(rows)['alpha']==.1
    with pytest.raises(ValueError):choose_alpha(rows[:2])
    with pytest.raises(ValueError):choose_alpha([dict(alpha=99,fused=dict(rmse=1))]+rows[1:])


def frozen():
    return dict(frozen=True,candidate=CANDIDATE,feature_spec=copy.deepcopy(FEATURE_SPEC),
        units=[dict(fold=f,alpha=1.,fit_count=3,train_indices=list(range(276)),scaler=SCALER) for f in range(5)])


def test_global_freeze_before_test_open(tmp_path):
    gate=PhaseGate()
    with pytest.raises(ValueError,match='before all five'):read_predictions(tmp_path/'absent.npz',gate=gate,test=True)
    bad=frozen();bad['units'].pop()
    with pytest.raises(ValueError):gate.freeze(bad)
    bad=frozen();bad['feature_spec']['g05_weight']=.7
    with pytest.raises(ValueError):validate_freeze(bad)
    gate.freeze(frozen());gate.require_test()
    with pytest.raises(FileNotFoundError):read_predictions(tmp_path/'absent.npz',gate=gate,test=True)


def test_budget_no_extra_science():
    budget={k:0 for k in LIMITS};check_budget(budget)
    budget.update(fit_attempts=15,fit_completed=15,formal_starts=5,formal_completed=5,prediction_groups=10)
    check_budget(budget)
    for key in ('fit_attempts','formal_starts','prediction_groups','pt_starts','neural_ft','epochs','gpu_inference','workers','refits'):
        bad=dict(budget);bad[key]=LIMITS[key]+1
        with pytest.raises(ValueError):check_budget(bad)


def test_reference_sample_scaler_binding():
    rows=[dict(original_row=3,sample_key='b',label=12.),dict(original_row=1,sample_key='a',label=8.)]
    pred=dict(indices=[1,3],sample_keys=['a','b'],y_true=[8.,12.],y_pred=[9.,11.],scaler_mean=[10.],scaler_scale=[2.])
    truth,p=align_reference(pred,rows,SCALER);np.testing.assert_equal(truth,[12.,8.]);np.testing.assert_equal(p,[11.,9.])
    for changed in (dict(sample_keys=['a','wrong']),dict(scaler_scale=[3.]),dict(indices=[1,4]),dict(y_true=[80.,12.])):
        with pytest.raises(ValueError):align_reference(dict(pred,**changed),rows,SCALER)


def test_feature_spec_and_dummy_connection_preservation():
    fp,details=fingerprint('*CC*',generator());assert details['dummy_atoms']==2
    assert FEATURE_SPEC['includeChirality'] and FEATURE_SPEC['includeRingMembership'] and FEATURE_SPEC['useBondTypes']
    assert not FEATURE_SPEC['countSimulation']
    restored=restore_bits(bit_array([fp]));assert DataStructs.TanimotoSimilarity(fp,restored[0])==1.


def test_g05_r0_reference_protocol_only():
    unit=dict(arm='T3B_G05',config=dict(arm='T3B_G05',task='xc',recipe=dict(arm='R0'),
        training_config=dict(arm='T3B_G05',revision='r5',loss_weights=[1,.5,0],schedule_total_steps=20000)))
    validate_reference_protocol(unit)
    for field,value in [('arm','G05_GLR'),('task','eps')]:
        bad=copy.deepcopy(unit);bad['config'][field]=value
        with pytest.raises(ValueError):validate_reference_protocol(bad)
    bad=copy.deepcopy(unit);bad['config']['recipe']['arm']='T2_G_LR'
    with pytest.raises(ValueError):validate_reference_protocol(bad)
