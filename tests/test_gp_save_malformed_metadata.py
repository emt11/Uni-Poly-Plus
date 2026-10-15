"""Malformed old metadata fails closed through the coordinated conflict exit."""
import json

import pytest

import src.training.gp_checkpoint_save as saving


def test_list_files_metadata_is_refused_and_conflict_is_broadcast(tmp_path,monkeypatch):
    metadata=tmp_path/'save_result_00007.json'
    metadata.write_text(json.dumps(dict(step=7,status='PASS',complete=True,stop=None,
        metadata_protocol='gp-save-gated-v1',files=['resume_00007.pt','deploy_00007.pt'])))
    before=metadata.read_bytes()
    disk=saving.read_save_result(tmp_path,7)
    assert disk['status']=='UNCOMMITTED_UNVERIFIABLE' and disk['success_consumable'] is False
    broadcasts=[]
    monkeypatch.setattr(saving.dist,'get_rank',lambda:0)
    monkeypatch.setattr(saving.dist,'broadcast_object_list',lambda result,src:broadcasts.append(result[0]))
    def forbidden(path):pytest.fail('conflicted writer must not write checkpoint files')
    with pytest.raises(saving.CoordinatedCheckpointExit) as error:
        saving.synchronized_save(tmp_path,7,forbidden,forbidden)
    result=json.loads(str(error.value))
    assert len(broadcasts)==1 and broadcasts[0]==result and result['status']=='FAILED'
    assert result['disk_result']['success_consumable'] is False
    assert metadata.read_bytes()==before and not (tmp_path/'failure.json').exists()
