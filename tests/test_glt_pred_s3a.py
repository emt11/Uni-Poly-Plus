from pathlib import Path
from types import SimpleNamespace

from scripts.run_glt_pred_s3a import _command, _jobs


def _args():
    return SimpleNamespace(
        config='config.json', checkpoint='deploy.pt', raw_root='raw',
        split_root='split', cohort_root='cohort', cache_root='cache',
        dual_static_root='static', clean_cache_gib=4.0, lora_rank=8,
        lora_alpha=8.0, lora_dropout=0.0,
    )


def test_s3a_matrix_has_18_neural_and_24_ridge_unique_outputs(tmp_path):
    jobs = _jobs(tmp_path, ['xc', 'eps', 'eat'], [0, 1],
                 ['full', 'head', 'lora', 'ridge'], [0.1, 1.0, 10.0, 100.0])
    assert len(jobs) == 42
    paths = [job['unit'] for job in jobs]
    assert len(paths) == len(set(paths))
    assert sum(job['policy'] != 'ridge' for job in jobs) == 18
    assert sum(job['policy'] == 'ridge' for job in jobs) == 24


def test_s3a_child_command_is_development_only_and_forwards_cache(tmp_path):
    args = _args()
    job = _jobs(tmp_path, ['xc'], [0], ['full'], [0.1])[0]
    command = _command(args, job)
    assert '--development' in command
    assert '--formal-shard' not in command
    assert command[command.index('--clean-cache-gib') + 1] == '4'
    assert '--task' in command and command[command.index('--task') + 1] == 'xc'
    assert '--fold' in command and command[command.index('--fold') + 1] == '0'


def test_s3a_ridge_commands_keep_each_alpha_in_a_distinct_output(tmp_path):
    args = _args()
    jobs = _jobs(tmp_path, ['xc'], [0], ['ridge'], [0.1, 1.0, 10.0, 100.0])
    command_paths = []
    for job in jobs:
        command = _command(args, job)
        command_paths.append(Path(command[command.index('--output') + 1]))
        assert command[command.index('--ridge-alpha') + 1] in {'0.1', '1', '10', '100'}
    assert len(command_paths) == len(set(command_paths))
