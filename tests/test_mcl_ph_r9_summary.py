"""r9 regression: the downstream unit summary carries ``optimizer_groups`` once.

``run_unit`` records its evidence in ``common`` - which already includes
``optimizer_groups`` - and then builds the run summary from that record.  A
version of that construction also passed ``optimizer_groups`` as an explicit
keyword, so every unit raised ``TypeError: dict() got multiple values for
keyword argument 'optimizer_groups'`` after training and evaluating an epoch
successfully.

These tests use plain dicts and the module source: no model, no dataset, no GPU,
no training process, no checkpoint on disk.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_mcl_ph import build_summary  # noqa: E402

RUNNER = ROOT / 'scripts' / 'finetune_mcl_ph.py'
COMMAND = ['scripts/finetune_mcl_ph.py', '--arm', 'glt_ref']
CONFIG = {'seed': 42, 'finetune_batch': 32}
COMMON = {
    'arm': 'glt_ref',
    'task': 'xc',
    'fold': 0,
    'stage': 'smoke',
    'protocol': 'mcl_ph_smoke',
    'executed_epochs': 1,
    'optimizer_updates': 9,
    'optimizer_groups': [{'name': 'backbone', 'lr': 1e-05}],
    'history': [{'epoch': 1, 'train_loss': 1.0, 'validation_loss': 0.5,
                 'validation_r2': 0.01}],
    'load_state_dict_result': {'missing_keys': [], 'unexpected_keys': []},
}


def _summary():
    return build_summary(COMMON, config=CONFIG, device='cpu', command=COMMAND)


def test_a_common_already_carries_optimizer_groups_and_the_summary_builds():
    assert 'optimizer_groups' in COMMON
    summary = _summary()
    assert summary['status'] == 'PASS'
    assert summary['command'] == COMMAND
    assert summary['config'] == CONFIG
    assert summary['device'] == 'cpu'


def test_b_optimizer_groups_appears_once_with_the_common_value():
    summary = _summary()
    assert summary['optimizer_groups'] == COMMON['optimizer_groups']
    assert summary['optimizer_groups'] is COMMON['optimizer_groups']
    assert summary['arm'] == COMMON['arm']
    assert summary['optimizer_updates'] == COMMON['optimizer_updates']


def test_c_the_run_payload_keeps_history_but_summary_keeps_it_out():
    summary = _summary()
    assert 'history' not in summary
    assert 'load_state_dict_result' not in summary
    run_payload = dict(summary, history=COMMON['history'])
    assert run_payload['history'] == COMMON['history']
    assert 'load_state_dict_result' not in run_payload


def test_d_the_summary_never_passes_optimizer_groups_explicitly():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    keywords = [keyword.arg for call in ast.walk(functions['build_summary'])
                if isinstance(call, ast.Call)
                for keyword in call.keywords if keyword.arg]
    assert 'optimizer_groups' not in keywords
    # The duplicate keyword is exactly the signature of the r8 failure, so it
    # must stay impossible: this is the shape that used to raise.
    with pytest.raises(TypeError, match='multiple values'):
        dict(optimizer_groups=COMMON['optimizer_groups'],
             **{'optimizer_groups': COMMON['optimizer_groups']})
    called = {node.func.id for node in ast.walk(functions['run_unit'])
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert 'build_summary' in called
