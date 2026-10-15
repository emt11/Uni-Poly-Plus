"""Exercise the actual CLI failure path with synthetic outputs and zero model calls."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

import scripts.finetune_glt_mcl_ph as ft
import src.training.gp_sota_contract as contract


@pytest.fixture
def cli(tmp_path, monkeypatch):
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES'] == ''
    output_root=tmp_path/'output';output_root.mkdir()
    monkeypatch.setattr(contract,'OUT',output_root)
    monkeypatch.setattr(ft,'OUT',output_root)
    monkeypatch.setattr(ft,'require_tmux',lambda:None)
    (output_root/'p1_acceptance.json').write_text(json.dumps(dict(status='PASS')))
    scope=tmp_path/'scope.json'
    scope.write_text(json.dumps(dict(plan=contract.PLAN,revision='r3',configuration_revision='r2',continuation='NC_R3',
        name='R3_Nc_GU',status='LOCKED',allowed_units=['F0/GU/nc/fold0/seed42','F0/GU/nc/fold1/seed43'])))
    folder=output_root/'development/R3_Nc_GU/GU/nc/fold0'
    argv=['scripts/finetune_glt_mcl_ph.py','--task','nc','--fold','0','--route','GU','--recipe','F0',
          '--stage','development','--scope',str(scope),'--checkpoint',str(tmp_path/'unused.pt'),'--output',str(folder)]
    monkeypatch.setattr(sys,'argv',argv)
    def no_model(*args,**kwargs):pytest.fail('model construction is outside this CPU test')
    monkeypatch.setattr(ft,'GPArm',no_model)
    return tmp_path,folder,argv,scope


def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}


def test_existing_output_is_unchanged(cli):
    root,folder,_,_=cli;folder.mkdir(parents=True)
    for name in ['failure.json','metrics.json','best.pt']:(folder/name).write_bytes(b'protected prior artifact')
    before=hashes(root)
    with pytest.raises(FileExistsError):ft.run_cli()
    assert hashes(root)==before


@pytest.mark.parametrize('outside',[False,True])
def test_rejected_output_never_created(cli,outside):
    root,folder,argv,_=cli;rejected=root/'outside_authorized_output' if outside else folder.with_name('fold0_retry')
    argv[-1]=str(rejected);before=hashes(root)
    with pytest.raises((AssertionError,ValueError)):ft.run_cli()
    assert not rejected.exists() and hashes(root)==before


def test_invalid_scope_never_creates_output(cli):
    root,folder,_,scope=cli;s=json.loads(scope.read_text());s['plan']='invalid';scope.write_text(json.dumps(s));before=hashes(root)
    with pytest.raises(ValueError):ft.run_cli()
    assert not folder.exists() and hashes(root)==before


def test_missing_output_argument_keeps_argparse_exit(cli):
    root,folder,argv,_=cli;del argv[-2:];before=hashes(root)
    with pytest.raises(SystemExit) as error:ft.run_cli()
    assert error.value.code==2 and hashes(root)==before and not folder.exists()


def test_help_keeps_success_exit(cli):
    root,folder,argv,_=cli;argv[1:]=['--help'];before=hashes(root)
    with pytest.raises(SystemExit) as error:ft.run_cli()
    assert error.value.code==0 and hashes(root)==before and not folder.exists()


def fail_before_model(*args,**kwargs):
    raise RuntimeError('synthetic failure after exclusive output allocation')


def test_owned_failure_writes_only_its_metadata(cli,monkeypatch):
    _,folder,_,_=cli;monkeypatch.setattr(ft,'set_global_seed',fail_before_model)
    with pytest.raises(RuntimeError,match='synthetic failure'):ft.run_cli()
    result=json.loads((folder/'failure.json').read_text());budget=json.loads((folder/'budget.json').read_text())
    assert result['status']=='FAILED' and result['automatic_retry'] is False and 'synthetic failure' in result['error']
    assert budget['attempted_updates']==budget['completed_updates']==budget['epochs']==0
    assert not (folder/'head_initial.pt').exists() and not (folder/'best.pt').exists()


def test_failure_metadata_error_preserves_original_exception(cli,monkeypatch,capsys):
    _,folder,_,_=cli;monkeypatch.setattr(ft,'set_global_seed',fail_before_model)
    original=ft.atomic_json
    def fail_metadata(path,value):
        if Path(path).name=='failure.json':raise OSError('synthetic metadata failure')
        return original(path,value)
    monkeypatch.setattr(ft,'atomic_json',fail_metadata)
    with pytest.raises(RuntimeError,match='synthetic failure after exclusive'):ft.run_cli()
    assert not (folder/'failure.json').exists() and 'synthetic metadata failure' in capsys.readouterr().err
    budget=json.loads((folder/'budget.json').read_text());assert budget['attempted_updates']==budget['completed_updates']==0
