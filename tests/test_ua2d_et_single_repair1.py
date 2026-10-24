"""Single final repair suite; existing six original tests are not rerun."""
import copy
import json
from pathlib import Path
import pickle
import lmdb
import numpy as np
import pytest
import torch
from src.dataset.ua2d_fast_pretrain import decode_template,_tensors
from src.dataset.ua2d_et_single import prepare_et_sample
from src.modules.ua2d_galformer_pretrain import PretrainConfig
from src.training.ua2d_et_single import CACHE,OUT

FAILED_KEY=bytes.fromhex('0c24852e76211448f7aab24193d5a26d3d98f480a5deca7e69f59533eafbf430')

def cached(key):
    a=lmdb.open(str(CACHE/'templates.lmdb'),readonly=True,lock=False,max_readers=128)
    b=lmdb.open(str(OUT/'atomic_fields/fields.lmdb'),readonly=True,lock=False,max_readers=128)
    try:
        with a.begin() as txn:t=decode_template(txn.get(key))
        with b.begin() as txn:f=_tensors(pickle.loads(txn.get(key)))
        return t,f
    finally:a.close();b.close()


def test_actual_failed_record_is_excluded_before_geometry_length_check():
    template,fields=cached(FAILED_KEY)
    assert not template['clean'].geometry_valid
    assert len(template['positions'])==0 and fields['physical_count']==206
    with pytest.raises(ValueError,match='excluded from the entire ET training pool'):
        prepare_et_sample(template,fields,config=PretrainConfig(),position=410)


def test_filter_exact_coverage_and_actual_failed_record_absent():
    folder=OUT/'sampling_repair1';meta=json.loads((folder/'metadata.json').read_text())
    valid=np.load(folder/'eligible_indices.npy');excluded=np.load(folder/'excluded_indices.npy')
    assert meta['status']=='COMPLETE' and meta['policy']=='drop_failed_3d_entire_record'
    assert len(valid)==meta['valid_rows'] and len(excluded)==meta['excluded_rows']>0
    assert len(valid)+len(excluded)==911391
    np.testing.assert_array_equal(np.sort(np.concatenate([valid,excluded])),np.arange(911391))
    keys=np.load(CACHE/'keys.npy',mmap_mode='r')
    index=np.flatnonzero(keys==np.void(FAILED_KEY))
    assert len(index)==1 and index[0] in excluded and index[0] not in valid


def test_valid_geometry_mismatch_and_identity_error_still_stop():
    key=np.load(CACHE/'keys.npy',mmap_mode='r')[0].tobytes();template,fields=cached(key)
    assert template['clean'].geometry_valid
    altered=copy.deepcopy(template);altered['positions']=altered['positions'][:-1]
    with pytest.raises(ValueError,match='physical count mismatch'):
        prepare_et_sample(altered,fields,config=PretrainConfig(),position=0)
    fields=dict(fields,sample_key=b'incorrect')
    with pytest.raises(ValueError,match='identity mismatch'):
        prepare_et_sample(template,fields,config=PretrainConfig(),position=0)


def test_rank_failure_stops_before_launcher_exit_and_descendants_are_included(tmp_path):
    from scripts.run_ua2d_et_single import Controller
    from scripts.resume_ua2d_et_single_repair1 import descendants
    (tmp_path/'rank0_failure.json').write_text('{}')
    controller=Controller.__new__(Controller);controller.refresh=lambda:None
    job=dict(kind='pt',pt_output=str(tmp_path),name='fake_pt',exit=None)
    with pytest.raises(RuntimeError,match='rank failure'):controller.wait([job])
    assert descendants({10,30},{10:1,11:10,12:11,30:1,31:30,90:1})=={10,11,12,30,31}


def test_checkpointed_et_preserves_forward_and_gradient_paths():
    from src.modules.ua2d_et_single import AtomicET
    from src.dataset.ua2d_et_single import neighborhood
    model=AtomicET()
    pos=torch.tensor([[0.,0.,0.],[1.2,.1,0.],[1.8,.4,.3]])
    z=torch.tensor([5,101,7]);edge,chemical=neighborhood(pos,torch.tensor([[0,1,1,2],[1,0,2,1]]))
    model.eval()
    with torch.no_grad():expected=model(z,pos,edge,chemical)
    model.train();scalar,vector=model(z,pos,edge,chemical)
    torch.testing.assert_close(scalar,expected[0],rtol=0,atol=0)
    torch.testing.assert_close(vector,expected[1],rtol=0,atol=0)
    (scalar.square().sum()+vector.square().mean()).backward()
    assert model.embedding.weight.grad.abs().sum()>0
    assert model.layers[0].q_proj.weight.grad.abs().sum()>0
