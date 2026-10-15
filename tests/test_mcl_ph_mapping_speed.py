"""Mapping optimization retains the exact bounded lexicographic identity."""
import pytest
import torch
from rdkit import Chem
from src.dataset.canonical_periodic import find_atom_graph_mapping,_validate_full_atom_mapping


@pytest.mark.parametrize('smiles', ['*CC*','*c1ccc(*)cc1','*C(C)(C)c1ccc(C(C)(C)*)cc1','*C[C@H](F)C*','*CC(C)(C)CC(C)(C)*'])
def test_mapping_matches_previous_complete_filter(smiles):
    source=Chem.MolFromSmiles(smiles)
    target=Chem.MolFromSmiles(Chem.MolToSmiles(source,canonical=True,isomericSmiles=True))
    count=100000 if source.GetNumAtoms()<=24 else 512
    candidates=target.GetSubstructMatches(source,uniquify=False,useChirality=True,maxMatches=count)
    valid=[tuple(int(x) for x in match) for match in candidates if _validate_full_atom_mapping(source,target,match)]
    assert valid
    assert torch.equal(find_atom_graph_mapping(source,target),torch.tensor(min(valid)))


def test_readonly_registry_rejects_mixed_readahead_without_losing_first_handle(tmp_path):
    import lmdb
    from src.dataset.frozen_store import _acquire_readonly_env,_release_readonly_env
    root=tmp_path/'store';root.mkdir()
    writer=lmdb.open(str(root),map_size=1024**2)
    with writer.begin(write=True) as txn:txn.put(b'key',b'value')
    writer.close()
    env,key=_acquire_readonly_env(root,readahead=True)
    try:
        assert env.flags()['readonly'] and env.flags()['readahead']
        with pytest.raises(ValueError,match='different readahead'):
            _acquire_readonly_env(root,readahead=False)
        with env.begin() as txn:assert txn.get(b'key')==b'value'
    finally:_release_readonly_env(key,env)
    env,key=_acquire_readonly_env(root)
    try:assert not env.flags()['readahead']
    finally:_release_readonly_env(key,env)
