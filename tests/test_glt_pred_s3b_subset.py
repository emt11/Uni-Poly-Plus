"""Focused S3b subset/source contract tests (no training, no GPU, no frozen cache).

Covers the properties the pretraining runner relies on to distinguish the full
cohort from P_train and P_val, the frozen split artifact counts, the formal
config contract and the trimmed FGR statistics schema.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.training.glt_dual_runtime import (IndexedFrozenDualSource, OrderedSampleStream,
                                            load_sample_index_artifact)

REPO = Path(__file__).resolve().parents[1]
SPLIT_ARTIFACT = REPO / 'results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json'
SPLIT_NPZ = REPO / 'results/glt_pred_20260918/s3b_prep/pretrain_split_v1.npz'
FIXED_PVAL = REPO / 'results/glt_pred_20260918/s3b_prep/fixed_pval_v1.json'
FGR_STATS = REPO / 'results/glt_pred_20260918/s3b_prep/fgr_train_stats.json'
CONFIG_DIR = REPO / 'configs/mts'


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        'validate_glt_pred_s3b_prep', REPO / 'scripts/validate_glt_pred_s3b_prep.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubBase:
    """Minimal stand-in for the frozen source inside IndexedFrozenDualSource."""

    def __init__(self, keys):
        self.samples = [(key, 'C') for key in keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return ('record', index)

    def static_for(self, index):
        return None

    def target_for(self, index):
        return None


@pytest.mark.skipif(not SPLIT_NPZ.is_file(), reason='S3b split artifact is not present locally')
def test_split_artifact_counts_and_disjointness():
    artifact = load_sample_index_artifact(str(SPLIT_ARTIFACT), 'train')
    payload = artifact['payload']
    archive = np.load(SPLIT_NPZ)
    train = set(archive['train_source_indices'].tolist())
    validation = set(archive['validation_source_indices'].tolist())
    fixed = set(archive['fixed_validation_source_indices'].tolist())

    assert len(train) == 911391 and len(validation) == 47968 and len(fixed) == 1024
    assert payload['pi1m_record_count'] == 959588
    assert payload['excluded_overlap_identity_count'] == 229
    assert train & validation == set()
    assert fixed <= validation
    assert len(train | validation) == payload['eligible_identity_count'] == 959359
    assert len(artifact['indices']) == 911391
    assert artifact['split'] == 'train'


@pytest.mark.skipif(not SPLIT_ARTIFACT.is_file(), reason='S3b split artifact is not present locally')
def test_subset_source_distinguishes_p_train_p_val_and_full_cohort():
    train = load_sample_index_artifact(str(SPLIT_ARTIFACT), 'train')
    validation = load_sample_index_artifact(str(SPLIT_ARTIFACT), 'validation')
    keys = [index.to_bytes(8, 'little') * 4 for index in range(959588)]
    base = _StubBase(keys)

    full = IndexedFrozenDualSource(base, np.arange(len(base), dtype=np.int64))
    p_train = IndexedFrozenDualSource(base, train['indices'])
    p_val = IndexedFrozenDualSource(base, validation['indices'])

    assert (len(full), len(p_train), len(p_val)) == (959588, 911391, 47968)

    # What a checkpoint records: (split artifact sha, split name, sample count).
    def identity(view, split):
        return (train['sha256'], split, len(view))

    assert identity(full, None) != identity(p_train, 'train')
    assert identity(p_train, 'train') != identity(p_val, 'validation')

    # The resume check also compares the ordered sample keys of the stream that
    # would actually be trained on, so one split cannot be resumed as another.
    def ordered_prefix(view, size=64):
        stream = OrderedSampleStream(len(view), 42)
        return [view.samples[stream.index_at(position)][0].hex() for position in range(size)]

    ordered_train = ordered_prefix(p_train)
    ordered_val = ordered_prefix(p_val)
    ordered_full = ordered_prefix(full)
    assert ordered_train != ordered_val
    assert ordered_train != ordered_full
    assert len(ordered_train) == len(ordered_val) == 64

    # Every logical index of the subset view maps inside the frozen base source.
    assert 0 <= train['indices'].min() and train['indices'].max() < len(base)
    assert p_train[0][1] == int(train['indices'][0])
    assert p_val[0][1] == int(validation['indices'][0])


def test_trimmed_fgr_stats_schema_is_accepted_by_the_validator():
    validator = _load_validator()
    assert FGR_STATS.is_file(), 'the full P_train FGR statistics must exist for the formal configs'
    stats = validator._load_fgr_stats(str(FGR_STATS))
    assert stats['full_p_train_scan'] is True
    assert stats['graphs_scanned'] == 911391
    assert stats['candidate_count'] > 0
    assert stats['spd2_count'] + stats['spd3_count'] == stats['candidate_count']
    assert math.isfinite(stats['mu']) and math.isfinite(stats['sigma']) and stats['sigma'] > 0

    trimmed = {'graphs_scanned': 10, 'candidate_count': 3, 'mu': 1.0, 'sigma': 2.0,
               'full_p_train_scan': True}
    path = FGR_STATS.parent / 'fgr_stats_schema_probe.json'
    try:
        path.write_text(json.dumps(trimmed), encoding='utf-8')
        assert validator._load_fgr_stats(str(path))['mu'] == 1.0
        trimmed['full_p_train_scan'] = False
        path.write_text(json.dumps(trimmed), encoding='utf-8')
        with pytest.raises(ValueError):
            validator._load_fgr_stats(str(path))
    finally:
        path.unlink(missing_ok=True)


def test_formal_configs_share_one_data_and_init_contract():
    names = {
        'fp': 'glt_pred_s3b_b_fp.json',
        'none': 'glt_pred_s3b_b_none.json',
        'fgr': 'glt_pred_s3b_t_fgr.json',
        'align': 'glt_pred_s3b_t_align.json',
    }
    configs = {task: json.loads((CONFIG_DIR / name).read_text(encoding='utf-8'))
               for task, name in names.items()}
    first = configs['fp']
    for task, config in configs.items():
        assert config['third_task'] == task
        assert config['sample_index_split'] == 'train'
        assert config['sample_index_artifact'] == first['sample_index_artifact']
        assert config['common_init_artifact'] == first['common_init_artifact']
        assert int(config['expected_world_size']) == 3
        assert int(config['microbatch']) == 84 and int(config['global_batch']) == 1008
        assert int(config['max_optimizer_steps']) == 5000 and int(config['save_every']) == 1000
        assert config['fusion_mode'] == 'concat' and config['geometry_head_norm'] is True
        assert int(config['seed']) == 42 and config['amp_dtype'] == 'bf16'
        assert config['loss_weights'][:2] == [1, 1]
    assert configs['fp']['loss_weights'][2] == 0.1
    assert configs['none']['loss_weights'][2] == 0.0
    assert configs['fgr']['loss_weights'][2] == 0.1
    assert configs['fgr']['fgr_max_pairs'] == 32
    assert float(configs['fgr']['fgr_sigma']) > 0
    assert float(configs['fgr']['fgr_mu']) != 0.0  # not the degenerate default
    assert float(configs['align']['align_temperature']) == 0.1
