"""Actual controller CLI rejection/ownership paths, with zero scientific execution."""
import hashlib
import shlex
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.run_gp_nc_r3 as campaign


@pytest.fixture
def context(tmp_path, monkeypatch):
    out=tmp_path/'out';out.mkdir();log=tmp_path/'log';log.mkdir()
    monkeypatch.setattr(campaign,'OUT',out);monkeypatch.setattr(campaign,'LOG',log)
    monkeypatch.setattr(campaign,'STATE',out/'nc_r3_status.json')
    monkeypatch.setattr(campaign,'FROZEN',out/'frozen_selection_r3.json')
    monkeypatch.setattr(campaign,'CLAIM',out/'nc_r3_writer.claim')
    monkeypatch.setattr(campaign,'require_tmux',lambda:None)
    (log/'status.json').write_text('protected previous root ledger')
    calls=[]
    monkeypatch.setattr(campaign,'ledger',lambda *a,**kw:calls.append(('ledger',a,kw)))
    monkeypatch.setattr(campaign,'stop_r3_children',lambda logs:calls.append(('stop',logs)))
    return out,log,calls


def hashes(root):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('existing',['nc_r3_status.json','frozen_selection_r3.json',
    'seven_task_results_r3.json','nc_r3_writer.claim','nc_r3_locked_candidate.json'])
def test_rejected_reentry_does_not_mutate_or_stop(context,existing):
    out,log,calls=context
    if existing.endswith('.claim'):(out/existing).mkdir()
    else:(out/existing).write_text('previous evidence')
    before=hashes(out.parent)
    with pytest.raises(AssertionError):campaign.run_cli()
    assert not calls and hashes(out.parent)==before


def test_claim_race_rejection_has_no_writing_authority(context,monkeypatch):
    out,_,calls=context
    # Another controller wins the atomic claim after our preflight returns.
    (out/'nc_r3_writer.claim').mkdir()
    monkeypatch.setattr(campaign,'preflight',lambda:([],[]))
    before=hashes(out.parent)
    with pytest.raises(FileExistsError):campaign.run_cli()
    assert not calls and hashes(out.parent)==before


def test_owned_failure_stops_only_owned_logs_and_records_zero_update_exit(context,monkeypatch):
    out,_,calls=context;monkeypatch.setattr(campaign,'preflight',lambda:([],[]))
    def stage(*args,**kwargs):
        raise RuntimeError('synthetic owned failure before any model call')
    monkeypatch.setattr(campaign,'stage',stage)
    with pytest.raises(RuntimeError,match='synthetic owned failure'):campaign.run_cli()
    assert (out/'nc_r3_writer.claim').is_dir()
    assert [c[0] for c in calls]==['ledger','stop','ledger']
    assert calls[0][1]==('RUNNING',) and calls[1]==('stop',[]) and calls[2][1]==('BLOCKED',)
    assert calls[2][2]['automatic_retry'] is False


def test_stop_filter_uses_owned_full_commands(tmp_path,monkeypatch):
    log=tmp_path/'owned_log';log.mkdir()
    owned=['python','scripts/finetune_glt_mcl_ph.py','--output','/example/development/R3_Nc_GP/GP/nc/fold2','--fold','2']
    unrelated=owned[:-1]+['3']
    (log/'command.txt').write_text(shlex.join(owned))
    ps='PID COMMAND\n101 '+shlex.join(owned)+'\n102 '+shlex.join(unrelated)+'\n103 python scripts/run_gp_nc_r3.py\n'
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=ps))
    killed=[];monkeypatch.setattr(campaign.os,'kill',lambda pid,signal:killed.append(pid))
    campaign.stop_r3_children([log,tmp_path/'never_started_log'])
    assert killed==[101]
