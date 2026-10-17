"""r12 small artificial structures, arrays and estimator substitutes only."""
import copy
import json
import numpy as np
import pytest
from src.training import fce_kernel as k
from src.training import fce_param as public
from scripts import run_fce_radius3 as runner


def unit(task='eea',fold=0):
    train=[dict(original_row=i,sample_key=f'key{i}',normalized_smiles=s) for i,s in enumerate(('*CCCCCCCCCC*','*CCCCCOCCCC*'))]
    return dict(task=task,fold=fold,alpha=1.,model='old_radius2_model',train_indices=[0,1],validation_indices=[2,3],test_indices=[4,5],
        train_samples=train,validation_samples=[dict(original_row=i,sample_key=f'key{i}',normalized_smiles='*CCCCCCCCCC*') for i in (2,3)],
        scaler=dict(mean=[2.],scale=[1.],fit='train',transform='standard'),reference=dict(seed=42+fold))


def old_model(u):
    bits=np.zeros((2,2048),dtype=np.uint8);bits[0,0]=bits[1,1]=1
    return dict(alpha=np.float64(u['alpha']),train_indices=np.array([0,1]),train_keys=np.array(['key0','key1']),
        train_normalized_smiles=np.array([r['normalized_smiles'] for r in u['train_samples']]),
        train_bits=bits,dual_coef=np.ones(2),train_y=np.array([1.,3.]),mean=np.float64(2.),scale=np.float64(1.),feature_spec=json.dumps(k.FEATURE_SPEC))


def test_actual_radius3_generator_binary_dimensions_and_old_policy_preserved():
    smiles='*c1ccc(CCOC(=O)NCCCCCC*)cc1'
    fp3,_=k.fingerprint(smiles,k.radius3_generator());fp2,_=k.fingerprint(smiles,k.generator())
    assert fp3.GetNumBits()==2048 and set(fp3).issubset({0,1}) and fp3!=fp2
    assert 'radius=3' in k.radius3_generator().GetInfoString()
    assert k.FEATURE_SPEC['radius']==2 and k.RADIUS3_FEATURE_SPEC['radius']==3
    assert 'g05_weight' not in k.RADIUS3_FEATURE_SPEC
    assert public.RADIUS3_FINETUNE_POLICY['prediction_weight']==.75


def test_pure_cache_policy_task_row_identity_and_test_gate():
    events=[];cache=k.Radius3Cache(lambda role,hit:events.append((role,hit)));row=unit()['train_samples'][0]
    first=cache.get('eea',row,'train');assert cache.get('eea',row,'validation') is first
    assert events==[('train',False),('validation',True)]
    cache.get('ei',dict(row,normalized_smiles='*CCOCC*'),'train')
    with pytest.raises(ValueError):cache.get('eea',dict(row,normalized_smiles='*CCOCC*'),'train')
    with pytest.raises(ValueError):k.Radius3Cache(lambda *a:None,k.FEATURE_SPEC)
    with pytest.raises(ValueError):cache.get('eea',row,'test',k.Radius3Gate())
    assert all(set(v[0])!=set(('label','prediction')) for v in cache.entries.values())


def test_fixed_inherited_alpha_original_train_once_and_schema(monkeypatch):
    u=unit();source=old_model(u);calls=[];events=[]
    def fake_fit(kernel,z,alpha):
        assert kernel.shape==(2,2);np.testing.assert_equal(z,[-1.,1.]);calls.append(alpha)
        return np.array([.2,-.1])
    monkeypatch.setattr(k,'fit_once',fake_fit)
    result=public.fit_radius3_finetune_predictions(source,u,k.Radius3Cache(lambda *a:None),on_fit=events.append)
    assert calls==[1.] and events==['attempted','completed']
    assert json.loads(str(result['feature_spec']))==k.RADIUS3_FEATURE_SPEC
    assert not np.array_equal(result['train_bits'],source['train_bits'])
    assert not np.array_equal(result['dual_coef'],source['dual_coef'])
    np.testing.assert_equal(result['train_y'],[1.,3.]);k.validate_radius3_model(result,u)
    with pytest.raises(ValueError):k.validate_model(result,u)
    with pytest.raises((ValueError,KeyError)):k.validate_radius3_model(source,u)
    for field,value in [('alpha',10.),('train_indices',np.array([1,0])),('scale',np.float64(9.))]:
        bad=dict(result);bad[field]=value
        with pytest.raises(ValueError):k.validate_radius3_model(bad,u)


def test_validation_test_common_features_and_fixed_public_prediction(monkeypatch):
    u=unit();cache=k.Radius3Cache(lambda *a:None)
    monkeypatch.setattr(k,'fit_once',lambda K,z,a:np.ones(2))
    result=public.fit_radius3_finetune_predictions(old_model(u),u,cache,on_fit=lambda *a:None)
    identities={str(i):dict(sample_key=f'key{i}',normalized_smiles='*CCCCCCCCCC*') for i in range(6)}
    reference=dict(indices=[2,3],sample_keys=['key2','key3'],y_true=[0.,2.],y_g05=[.1,1.9],scaler_mean=[2.],scaler_scale=[1.])
    calls=[]
    def predict(kernel,coef,scaler):calls.append(kernel.shape);return np.array([.5,1.5])
    monkeypatch.setattr(k,'predict_dual',predict)
    truth,g05,branch,fusion=public.predict_radius3_finetune_predictions(result,u,reference,identities,cache,'validation')
    np.testing.assert_allclose(fusion,.75*g05+.25*branch)
    with pytest.raises(ValueError):public.predict_radius3_finetune_predictions(result,u,reference,identities,cache,'test',k.Radius3Gate())
    assert calls==[(2,2)]
    gate=k.Radius3Gate();gate.payload={}
    test=dict(reference,indices=[4,5],sample_keys=['key4','key5'],y_pred=reference['y_g05'])
    public.predict_radius3_finetune_predictions(result,u,test,identities,cache,'test',gate)
    assert calls==[(2,2),(2,2)]


def frozen_fixture():
    old=[unit(t,f) for t in k.TASKS for f in range(5)];new=[]
    for prior in old:
        u=copy.deepcopy(prior);u.update(model='new_'+prior['task']+str(prior['fold']),model_spec=copy.deepcopy(k.RADIUS3_FEATURE_SPEC),radius3_model_ready=True,prediction_weight=.75);new.append(u)
    return dict(policy=copy.deepcopy(k.RADIUS3_POLICY),frozen=True,r11_freeze=dict(units=old),units=new)


def test_all_35_new_models_freeze_before_test_and_binding(monkeypatch,tmp_path):
    class PriorGate:
        def freeze(self,p):self.payload=copy.deepcopy(p)
    monkeypatch.setattr(k,'AlignedAlphaGate',PriorGate)
    gate=k.Radius3Gate()
    with pytest.raises(ValueError):runner.load_npz(tmp_path/'absent.npz',gate,True)
    gate.freeze(frozen_fixture())
    with pytest.raises(FileNotFoundError):runner.load_npz(tmp_path/'absent.npz',gate,True)
    for mutate in (lambda p:p['units'].pop(),lambda p:p['units'][0].update(alpha=10.),lambda p:p['units'][0].update(model='old_radius2_model'),
        lambda p:p['units'][0].update(prediction_weight=.5),lambda p:p['units'][0].update(radius3_model_ready=False),lambda p:p['units'][0].update(model_spec=k.FEATURE_SPEC)):
        bad=frozen_fixture();mutate(bad)
        with pytest.raises(ValueError):k.Radius3Gate().freeze(bad)


def test_budget_queries_generations_and_overwrite(tmp_path):
    b={key:0 for key in runner.LIMITS};b.update(fit_attempts=35,fit_completed=35,new_models=35,validation_reads=35,validation_predictions=35,validation_outputs=35,
        formal_starts=35,formal_completed=35,formal_outputs=70,feature_generations=5875,cache_hits=23500,train_queries=18786,validation_queries=4714,test_queries=5875,final_cpu_starts=1)
    runner.check_budget(b)
    for key in runner.LIMITS:
        bad=dict(b);bad[key]=runner.LIMITS[key]+1
        with pytest.raises(ValueError):runner.check_budget(bad)
    p=k.new_directory(tmp_path/'owned')
    with pytest.raises(FileExistsError):k.new_directory(p)
