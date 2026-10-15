"""Authorization boundaries only; no model, data loader, or accelerator."""
import pytest
from src.training.gp_sota_contract import PLAN, OUT, validate_ft_scope, validate_r3_output


def scope(name='R3_Nc_GP', stage='development'):
    route='GU' if name=='R3_Nc_GU' else 'GP'
    folds=range(5) if stage=='formal' else (0,1) if route=='GU' else (2,3,4)
    return dict(plan=PLAN, revision='r3', continuation='NC_R3', configuration_revision='r2',
                name=name, status='FROZEN' if stage=='formal' else 'LOCKED',
                allowed_units=[f'F0/{route}/nc/fold{f}/seed{42+f}' for f in folds])


def test_exact_allowed_units():
    for route,folds in [('GU',(0,1)),('GP',(2,3,4))]:
        for f in folds:
            assert validate_ft_scope('development','nc',f,route,'F0',scope('R3_Nc_'+route))
    for f in range(5):
        assert validate_ft_scope('formal','nc',f,'GP','F0',scope(stage='formal'))


@pytest.mark.parametrize('task,fold,route,recipe', [
    ('nc',0,'GP','F0'), ('nc',2,'GU','F0'), ('nc',2,'GP_A','F0'),
    ('nc',2,'GP','F1'), ('ei',2,'GP','F0'), ('xc',2,'GP','F0')])
def test_development_rejects_scope_expansion(task,fold,route,recipe):
    with pytest.raises(ValueError):validate_ft_scope('development',task,fold,route,recipe,scope())


@pytest.mark.parametrize('field,value', [('continuation','other'),('configuration_revision','r3'),
    ('plan','other'),('name','P5'),('allowed_units',['F0/GP/nc/fold2/seed44'])])
def test_rejects_invalid_contract(field,value):
    s=scope();s[field]=value
    with pytest.raises(ValueError):validate_ft_scope('development','nc',2,'GP','F0',s)


def test_formal_rejects_unfrozen_gu_and_f1():
    s=scope(stage='formal');s['status']='LOCKED'
    with pytest.raises(ValueError):validate_ft_scope('formal','nc',0,'GP','F0',s)
    for route,recipe in [('GU','F0'),('GP_A','F0'),('GP','F1')]:
        with pytest.raises(ValueError):validate_ft_scope('formal','nc',0,route,recipe,scope(stage='formal'))


def test_output_binding_prevents_alternate_retry_folders():
    expected=OUT/'development/R3_Nc_GP/GP/nc/fold2'
    validate_r3_output('development','nc',2,'GP','F0',scope(),expected)
    with pytest.raises(ValueError):validate_r3_output('development','nc',2,'GP','F0',scope(),str(expected)+'_retry')
    formal=OUT/'formal_r3/GP/F0/nc/fold0'
    validate_r3_output('formal','nc',0,'GP','F0',scope(stage='formal'),formal)
    with pytest.raises(ValueError):validate_r3_output('formal','nc',0,'GP','F0',scope(stage='formal'),OUT/'formal/GP/F0/nc/fold0')
