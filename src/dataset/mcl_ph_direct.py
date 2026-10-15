"""Read explicitly published direct-identity MCL records without online repair."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache_lifecycle import deserialize_record
from .canonical_periodic import resolve_normalized_identity
from .frozen_store import _acquire_readonly_env, _release_readonly_env
from .lmdb_cache import sample_key_from_normalized


class DirectMCLSource(Dataset):
    def __init__(self, root, *, task, selected_indices, split_manifest, readahead=False, verify_key_digest=True, md200_root=None):
        self.root = Path(root).resolve()
        self.md200_root = md200_root
        self.md200_sidecar = None
        self.environment = self.registry_key = None
        self.readahead = bool(readahead)
        self.verify_key_digest = bool(verify_key_digest)
        self.manifest = json.loads((self.root / 'direct_source.json').read_text())
        if (not (self.root / '.frozen').is_file()
                or self.manifest.get('schema') != 'mcl-ph-direct-source-v1'
                or self.manifest.get('conclusion') != 'PASS'
                or self.manifest.get('coordinate_generation_calls') != 0):
            raise ValueError('direct source is not verified and frozen')
        self.kind = str(self.manifest.get('kind'))
        if self.kind == 'pi1m':
            self._init_pi1m(task, selected_indices, split_manifest)
            return
        if self.kind != 'downstream' or self.manifest.get('protocol') != 'paper5_outer':
            raise ValueError('unsupported direct source kind or protocol')
        self._init_downstream(task, selected_indices, split_manifest)

    def _init_pi1m(self, task, selected_indices, split_manifest):
        if task is not None or split_manifest is not None:
            raise ValueError('PI1M direct source does not accept task fold manifests')
        count = int(self.manifest.get('source_rows', -1))
        if count <= 0:
            raise ValueError('PI1M direct source has no source row count')
        indices = list(range(count)) if selected_indices is None else list(selected_indices)
        if (any(type(index) is not int or not 0 <= index < count for index in indices)
                or indices != sorted(set(indices)) or not indices):
            raise ValueError('PI1M selected indices must be sorted, unique and nonempty')
        offsets = np.load(self.root / 'record_offsets.npy', allow_pickle=False)
        if (offsets.dtype != np.int64 or offsets.shape != (count,)
                or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)):
            raise ValueError('PI1M source row offsets are invalid')
        self.entries = []
        with (self.root / 'records.jsonl').open('rb') as stream:
            for index in indices:
                stream.seek(int(offsets[index]))
                row = json.loads(stream.readline())
                if (int(row.get('source_index', -1)) != index
                        or row.get('pretrain_role') not in ('train', 'validation', 'excluded')
                        or not row.get('normalized_smiles') or not row.get('source_smiles')):
                    raise ValueError('PI1M direct source row identity or role is invalid')
                if self.verify_key_digest and row.get('sample_key') != sample_key_from_normalized(
                        row['normalized_smiles']).hex():
                    raise ValueError('PI1M direct source key differs from normalized identity')
                self.entries.append(row)
        self.samples = [(bytes.fromhex(row['sample_key']), row['source_smiles'])
                        for row in self.entries]
        if any(len(key) != 32 for key, _ in self.samples):
            raise ValueError('invalid PI1M direct LMDB key')
        self._open_environment()

    def _init_downstream(self, task, selected_indices, split_manifest):
        if split_manifest is None or (split_manifest.get('protocol') != 'paper5_outer'
                                     or split_manifest.get('task') != task):
            raise ValueError('direct reader requires the active task split contract')
        count = self.manifest['task_rows'].get(task)
        if count is None or split_manifest.get('sample_count') != count:
            raise ValueError('direct task source count differs from active split')
        # Inspect indices and role sets without reading any unselected labels.
        roles = [[None] * count for _ in range(5)]
        if len(split_manifest['folds']) != 5:
            raise ValueError('direct source requires five folds')
        for fold, entry in enumerate(split_manifest['folds']):
            if entry['fold'] != fold:
                raise ValueError('direct source fold ordering differs')
            for role in ('train', 'validation', 'test'):
                for index in entry[role + '_indices']:
                    if type(index) is not int or not 0 <= index < count or roles[fold][index] is not None:
                        raise ValueError('direct split indices are invalid or overlap')
                    roles[fold][index] = role
            if any(role is None for role in roles[fold]):
                raise ValueError('direct split leaves rows uncovered')
        indices = list(range(count)) if selected_indices is None else list(selected_indices)
        if (any(type(index) is not int or not 0 <= index < count for index in indices)
                or indices != sorted(set(indices)) or not indices):
            raise ValueError('direct selected indices must be sorted, unique and nonempty')
        offsets = np.load(self.root / f'{task}_offsets.npy', allow_pickle=False)
        if (offsets.dtype != np.int64 or offsets.shape != (count,)
                or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)):
            raise ValueError('direct row offsets are invalid')
        self.entries = []
        with (self.root / f'{task}.jsonl').open('rb') as stream:
            for index in indices:
                stream.seek(int(offsets[index]))
                row = json.loads(stream.readline())
                if row['task'] != task or row['original_row'] != index:
                    raise ValueError('direct selected row identity differs')
                expected_roles = [{'fold': fold, 'role': roles[fold][index]} for fold in range(5)]
                if row['fold_membership'] != expected_roles:
                    raise ValueError('direct selected row roles differ from activity split')
                self.entries.append(row)
        self.samples = [(bytes.fromhex(row['sample_key']), row['source_smiles'])
                        for row in self.entries]
        if any(len(key) != 32 for key, _ in self.samples):
            raise ValueError('invalid direct LMDB key')
        self._open_environment()

    def _open_environment(self):
        if self.md200_root is not None:
            from .glt_fusion_md200 import MD200Sidecar
            self.md200_sidecar = MD200Sidecar(self.md200_root, self.root,
                self.entries[0].get('task'), self.entries)
        self.environment, self.registry_key = _acquire_readonly_env(
            self.root / 'data.lmdb', readahead=self.readahead)
        if self.environment.stat()['entries'] != int(self.manifest['identity_records']):
            self.close()
            raise ValueError('direct LMDB identity count differs')
        # Compatibility metadata consumed by the existing pretrain/cache
        # identity code.  It is an explicit direct-source contract identifier,
        # not a digest substituted for row-level validation.
        contract = (f"mcl-ph-direct:{self.kind}:"
                    f"{self.manifest.get('source_rows', self.manifest.get('task_rows'))}:"
                    f"{self.manifest['identity_records']}")
        self.cohort = {'manifest_hash': contract}
        self.bundle = SimpleNamespace(bundle_hash=contract)
        self.static_cache = SimpleNamespace(manifest_hash=contract)
        self.target_cache = None

    def __len__(self):
        return len(self.entries)

    def source_index_at(self, index):
        """Return immutable source JSONL index for a reader-local row."""
        row = self.entries[int(index)]
        return int(row.get('source_index', row.get('original_row', -1)))

    def record(self, index):
        row = self.entries[int(index)]
        key, smiles = self.samples[int(index)]
        with self.environment.begin() as txn:
            payload = txn.get(key)
        if payload is None:
            raise ValueError('direct record absent; online repair is prohibited')
        record = deserialize_record(bytes(payload), key)
        if (record.normalized_smiles != row['normalized_smiles']
                or record.topology.normalized_canonical_smiles != row['normalized_smiles']
                or (self.kind == 'downstream' and record.final_geometry_status != row['geometry_status'])):
            raise ValueError('direct structure or geometry ledger identity differs')
        valid = (record.final_geometry_status == 'PASS')
        if self.kind == 'downstream':
            valid = row['geometry_status'] == 'PASS'
        if bool(record.trimer.trimer_geometry_valid) != valid or bool(record.static['geometry_valid']) != valid:
            raise ValueError('direct geometry validity differs from final status')
        if hasattr(self,'geometry_quality'):
            record.topology.geometry_quality_eligible=bool(self.geometry_quality[int(index)])
        record.validated_normalized_identity = resolve_normalized_identity(
            record.topology, smiles, require_fields=True)
        mapping = torch.as_tensor(record.trimer.mips_to_trimer_central_index).long()
        atoms = record.trimer.trimer_atomic_number
        if (mapping.shape != record.topology.z.shape or mapping.numel() == 0
                or mapping.min() < 0 or mapping.max() >= atoms.numel()
                or not torch.equal(record.topology.z, atoms[mapping])
                or not torch.equal(record.trimer.trimer_base_ru_atom_id[mapping],
                                   record.topology.canonical_to_trimer_base_atom_id)
                or not bool(record.trimer.trimer_ru_offset[mapping].eq(0).all())):
            raise ValueError('direct center mapping atomic identity differs')
        positions = torch.as_tensor(record.trimer.trimer_pos)
        if valid and (positions.shape != (atoms.numel(), 3)
                      or not bool(torch.isfinite(positions).all())):
            raise ValueError('direct successful coordinate payload is invalid')
        if not valid and positions.shape != (0, 3):
            raise ValueError('direct failed identity must have no coordinate payload')
        if self.md200_sidecar is not None:
            source_row = int(row['source_index'] if self.kind == 'pi1m' else row['original_row'])
            values, available, reason = self.md200_sidecar.get(source_row)
            record.topology.md200 = values
            record.topology.md200_valid = available
            record.topology.md200_failure_reason = reason
        return record

    def __getitem__(self, index):
        record = self.record(index)
        return record.topology, record.trimer, self.entries[int(index)]['source_smiles']

    def pretrain_fields_for(self, index, *, include_target):
        """Decode and validate a presentation once, retaining every check."""
        return self.validated_pretrain_fields_for(index, include_target=include_target)[:5]

    def validated_pretrain_fields_for(self, index, *, include_target):
        """Carry this record's resolved identity to its view, without re-solving."""
        record = self.record(index)
        target = record.target if include_target and self.kind == 'pi1m' else None
        if include_target and self.kind == 'pi1m' and not isinstance(target, dict):
            raise ValueError('direct PI1M record requires its existing target cache')
        identity = record.validated_normalized_identity
        identity['validated_source_smiles'] = self.entries[int(index)]['source_smiles']
        identity['validated_source_schema'] = self.manifest['schema']
        identity['validated_topology_schema'] = getattr(record.topology,'canonical_periodic_topology_schema',None)
        return (record.topology, record.trimer,
                self.entries[int(index)]['source_smiles'], record.static, target,
                record.validated_normalized_identity)

    def static_for(self, index):
        return self.record(index).static

    def target_for(self, index):
        if self.kind != 'pi1m':
            return None
        return self.record(index).target

    def close(self):
        if self.registry_key is not None:
            _release_readonly_env(self.registry_key, self.environment)
            self.registry_key = self.environment = None
