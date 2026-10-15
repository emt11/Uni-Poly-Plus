from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.training.mcl_ph_direct_contract import assert_actual_equal, direct_binding
from src.training.glt_dual_runtime import apply_common_initialization


def test_actual_binding_rejects_changed_dtype_value_and_rng():
    expected = {'rows': np.array([0,2], dtype=np.int64), 'model': {'w': torch.ones(2)}, 'rng': (1, np.array([2,3]))}
    assert_actual_equal(expected, expected)
    for changed in [dict(expected, rows=np.array([0,1])), dict(expected, model={'w':torch.zeros(2)}), dict(expected, model={'w':torch.ones(2,dtype=torch.float64)}), dict(expected, rng=(2,np.array([2,3])))]:
        with pytest.raises(ValueError): assert_actual_equal(changed, expected)


def test_direct_initialization_does_not_hash_and_requires_tensor_coverage(tmp_path, monkeypatch):
    import src.training.glt_dual_runtime as runtime
    monkeypatch.setattr(runtime, 'sha256_file', lambda path: pytest.fail('direct path attempted file digest'))
    model = torch.nn.Linear(2,1)
    state = {key:value.detach().clone() for key,value in model.state_dict().items()}
    path=tmp_path/'init.pt'; torch.save({'common_state_dict':state},path)
    with torch.no_grad(): model.weight.zero_()
    result=apply_common_initialization(model,path,direct=True)
    assert result['sha256'] is None
    assert_actual_equal(model.state_dict(),state)
    torch.save({'common_state_dict':{'weight':state['weight']}},path)
    with pytest.raises(ValueError,match='coverage'): apply_common_initialization(model,path,direct=True)


def test_checkpoint_binding_detects_same_count_new_source_row_and_init(tmp_path):
    rows=[dict(sample_key=(b'a'*32).hex(),pretrain_role='train',normalized_smiles='*CC*',source_smiles='*CC*')]
    base=SimpleNamespace(kind='pi1m',root=tmp_path,entries=rows)
    class Source:
        def __init__(self): self.base=base
        def __len__(self): return 1
        def source_index_at(self,index): return 0
    path=tmp_path/'init.pt'; torch.save({'common_state_dict':{'w':torch.ones(2)}},path)
    original=direct_binding(Source(),common_init=path)
    rows[0]['source_smiles']='*CCC*'
    with pytest.raises(ValueError): assert_actual_equal(direct_binding(Source(),common_init=path), original)
    rows[0]['source_smiles']='*CC*'
    torch.save({'common_state_dict':{'w':torch.zeros(2)}},path)
    with pytest.raises(ValueError): assert_actual_equal(direct_binding(Source(),common_init=path), original)


def test_restore_checks_optimizer_and_position_without_an_update():
    from src.training.mcl_ph_direct_contract import verify_restored_checkpoint
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters(),lr=.001)
    state=dict(step=1,next_position=1008,scheduler={'step':1},
               model={key:value.clone() for key,value in model.state_dict().items()},
               optimizer=opt.state_dict())
    verify_restored_checkpoint(model,opt,state,batch_size=1008)
    state['next_position']=0
    with pytest.raises(ValueError,match='position'): verify_restored_checkpoint(model,opt,state,batch_size=1008)
    state['next_position']=1008
    opt.param_groups[0]['lr']=.1
    with pytest.raises(ValueError,match='optimizer'): verify_restored_checkpoint(model,opt,state,batch_size=1008)
