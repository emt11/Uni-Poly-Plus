"""The single identity record for this round's C1 deployment.

The retention runner and the aggregator read the same committed record, so a
downstream unit cannot silently claim a different (or older) checkpoint: the
``--checkpoint`` file must hash to the recorded sha256, and the package's own
metadata must agree field by field.  Nothing here writes the record, and a
checkpoint that is not the recorded one is refused rather than accepted with a
warning.

The record itself lives in ``configs/`` because ``results/`` is not tracked; the
artifact it describes is referenced by relative path.
"""
import hashlib
import json
from pathlib import Path

DEFAULT_IDENTITY = 'configs/mts/glt_galph_c1_repair_5k_identity.json'
REQUIRED_FIELDS = ('record', 'checkpoint', 'sha256', 'step', 'architecture',
                   'summary_mode', 'ph_mode', 'ph_encoder_version')
PACKAGE_FIELDS = ('step', 'architecture', 'summary_mode', 'ph_mode',
                  'ph_encoder_version')
SHA256_LENGTH = 64


def file_sha256(path, chunk_bytes=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_identity(path=DEFAULT_IDENTITY):
    """Read and structurally validate the identity record."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint identity record not found: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, '')]
    if missing:
        raise ValueError(f'identity record is missing {missing}: {path}')
    sha = str(record['sha256'])
    if len(sha) != SHA256_LENGTH or any(character not in '0123456789abcdef' for character in sha):
        raise ValueError(f'identity record sha256 is not a hex digest: {path}')
    if int(record['step']) <= 0:
        raise ValueError(f'identity record step must be positive: {path}')
    record['record_path'] = str(path)
    return record


def verify_checkpoint(identity, checkpoint_path, package, sha256=None):
    """Refuse any checkpoint the record does not describe exactly.

    ``package`` is the loaded deployment payload; ``sha256`` is passed in by a
    caller that already hashed the file, so a long run hashes it once.
    """
    actual = file_sha256(checkpoint_path) if sha256 is None else sha256
    if actual != identity['sha256']:
        raise ValueError(
            f'checkpoint {checkpoint_path} has sha256 {actual}, but the identity record '
            f"{identity['record_path']} describes {identity['sha256']}; the previous cycle's "
            'C1 deployment is not this round\'s control')
    for field in PACKAGE_FIELDS:
        expected = identity[field]
        found = package.get(field)
        if field in ('step', 'ph_bins', 'ph_channels'):
            found = int(found) if found is not None else found
            expected = int(expected)
        if found != expected:
            raise ValueError(f'checkpoint {checkpoint_path} reports {field}={found!r}, '
                             f'but the identity record says {expected!r}')
    return {'record': identity['record'], 'record_path': identity['record_path'],
            'checkpoint': str(checkpoint_path), 'recorded_checkpoint': identity['checkpoint'],
            'sha256': actual, 'step': int(identity['step']),
            'architecture': identity['architecture'],
            'summary_mode': identity['summary_mode'], 'ph_mode': identity['ph_mode'],
            'ph_encoder_version': identity['ph_encoder_version']}


def frozen_checkpoint_sha256(path=DEFAULT_IDENTITY):
    """The recorded sha256 alone, for callers that only need to compare."""
    return load_identity(path)['sha256']
