"""Rank-zero checkpoint writes with a collective result before any rank exits."""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist


class CoordinatedCheckpointExit(RuntimeError):
    """Every rank received the same persisted save failure or guard stop."""


def write_metadata(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    with _directory(path.parent) as descriptor:
        os.fsync(descriptor)


@contextmanager
def _directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _verified_payload(folder, step):
    """Validate all fallible disk reads before the writer publishes success."""
    result = json.loads((folder / f'save_result_{step:05d}.json').read_text())
    if not isinstance(result, dict) or result.get('metadata_protocol') != 'gp-save-gated-v1' or result.get('step') != step:
        raise ValueError('unrecognized save metadata')
    if result['status'] not in ('PASS', 'STOPPED_ROUTER_GUARD') or result['complete'] is not True:
        raise ValueError('save not complete')
    if not isinstance(result['files'], dict) or set(result['files']) != {f'resume_{step:05d}.pt', f'deploy_{step:05d}.pt'}:
        raise ValueError('checkpoint pair mismatch')
    for name, details in result['files'].items():
        path = folder / name
        if details['exists'] is not True or path.stat().st_size != details['bytes'] or details['bytes'] <= 0:
            raise ValueError('checkpoint file mismatch')
    if (result['status'] == 'STOPPED_ROUTER_GUARD') != (result['stop'] is not None):
        raise ValueError('stop status mismatch')
    if result['stop'] is not None:
        failure = json.loads((folder / 'failure.json').read_text())
        stop = json.loads((folder / 'router_stop.json').read_text())
        if failure != dict(result, automatic_retry=False) or stop != result['stop']:
            raise ValueError('stop metadata mismatch')
    return result


def read_save_result(folder, step):
    """A raw PASS is provisional until its persistent uncommitted gate is gone.

    Legacy/missing/malformed metadata is not evidence of a committed save.
    This is the consumption contract for this writer's new metadata only.
    """
    folder = Path(folder)
    try:
        result = json.loads((folder / f'save_result_{step:05d}.json').read_text())
        pending = folder / f'save_uncommitted_{step:05d}.json'
        if pending.exists():
            return dict(status='UNCOMMITTED_UNVERIFIABLE', success_consumable=False,
                        observed_result=result, reason='persistent uncommitted gate')
        result = _verified_payload(folder, step)
        # Recheck the gate after reads, failing closed if another writer appeared.
        if pending.exists():
            raise ValueError('uncommitted gate appeared during verification')
        return dict(status=result['status'], success_consumable=result['status'] == 'PASS',
                    metadata_commit='COMMITTED', observed_result=result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(status='UNCOMMITTED_UNVERIFIABLE', success_consumable=False,
                    reason=f'{type(error).__name__}: {error}')


def synchronized_save(folder, step, write_resume, write_deploy, *, stop=None):
    """Callbacks run only on rank0; followers wait through writes and metadata.

    Collective waits are bounded by the process group's configured timeout.
    A caught write error is propagated even when failure metadata also fails;
    stderr remains the fallback evidence. No failed pair receives PASS.
    """
    folder = Path(folder)
    resume = folder / f'resume_{step:05d}.pt'
    deploy = folder / f'deploy_{step:05d}.pt'
    result = [None]
    if dist.get_rank() == 0:
        pending = folder / f'save_uncommitted_{step:05d}.json'
        metadata = folder / f'save_result_{step:05d}.json'
        # Unknown collision state must fail closed without modifying an old save.
        conflict = True
        status = dict(step=step, status='SAVING', complete=False, stop=stop,
                      metadata_protocol='gp-save-gated-v1')
        try:
            conflict = pending.exists() or metadata.exists()
            # Do not replace a previous attempt or lift its gate.
            if conflict:
                raise FileExistsError('save metadata already exists for this step')
            write_metadata(pending, dict(step=step, status='UNCOMMITTED'))
            if stop is not None:
                write_metadata(folder / 'router_stop.json', stop)
            write_resume(resume)
            write_deploy(deploy)
            for path in (resume, deploy):
                if not path.is_file() or path.stat().st_size == 0:
                    raise OSError(f'checkpoint missing or empty: {path}')
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            with _directory(folder) as descriptor:
                os.fsync(descriptor)
            status.update(status='STOPPED_ROUTER_GUARD' if stop else 'PASS', complete=True)
        except Exception as error:
            status.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        try:
            status['files'] = {str(path.name): dict(exists=path.exists(), bytes=path.stat().st_size if path.exists() else None) for path in (resume, deploy)}
        except OSError as error:
            status.update(status='FAILED', error=f'{type(error).__name__}: {error}', files={})
        try:
            if conflict:
                raise FileExistsError('refuse to replace prior save metadata')
            write_metadata(metadata, status)
            if status['status'] != 'PASS':
                write_metadata(folder / 'failure.json', dict(status, automatic_retry=False))
            if status['status'] in ('PASS', 'STOPPED_ROUTER_GUARD'):
                verified = _verified_payload(folder, step)
                if verified != status:
                    raise ValueError('persisted metadata differs from writer result')
                committed = dict(status=verified['status'], success_consumable=verified['status'] == 'PASS',
                                 metadata_commit='COMMITTED', observed_result=verified)
                # All package and metadata fsync calls have succeeded. Last operation:
                # unlink the gate, with NO fallible persistence work after it.
                # We deliberately do not claim durable gate absence: after a crash,
                # reappearance can only reject an otherwise completed save.
                pending.unlink()
                # Publish the already verified result. No filesystem operation here
                # may turn the committed PASS into an in-memory FAILED.
                status['disk_result'] = committed
                status['metadata_commit'] = 'COMMITTED'
        except Exception as error:
            status.update(status='FAILED', metadata_error=f'{type(error).__name__}: {error}')
            # One best-effort correction; the gate stays even if both writes fail.
            # Never assume this correction committed: report its outcome explicitly.
            try:
                if not conflict:
                    write_metadata(metadata, status)
                    write_metadata(folder / 'failure.json', dict(status, automatic_retry=False))
            except Exception as correction:
                status['correction_error'] = f'{type(correction).__name__}: {correction}'
        if 'disk_result' not in status:
            status['disk_result'] = read_save_result(folder, step)
            status['metadata_commit'] = 'UNCOMMITTED_UNVERIFIABLE'
        if status['status'] != 'PASS':
            print('coordinated checkpoint exit: ' + json.dumps(status), file=sys.stderr, flush=True)
        result[0] = status
    # rank0 enters this collective only after both writes and metadata finish.
    dist.broadcast_object_list(result, src=0)
    if result[0]['status'] != 'PASS':
        raise CoordinatedCheckpointExit(json.dumps(result[0], sort_keys=True))
    return result[0]
