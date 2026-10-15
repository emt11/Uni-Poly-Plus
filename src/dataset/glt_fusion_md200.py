"""本周期MD200 sidecar读取；对应source行和规范结构，失败值不冒充有效。"""
import json
from pathlib import Path
import numpy as np
import torch


class MD200Sidecar:
    def __init__(self, root, source_root, task, entries):
        root = Path(root)
        metadata = json.loads((root / 'metadata.json').read_text())
        if (metadata.get('schema') != 'glt-fusion-md200-v1' or not metadata.get('complete')
                or metadata.get('semantics') != 'RDKit2DNormalized200/star_sub'
                or Path(metadata['source']).resolve() != Path(source_root).resolve()
                or len(metadata.get('columns', [])) != 200
                or len(set(metadata['columns'])) != 200):
            raise ValueError('MD200 sidecar source/schema/columns mismatch')
        task = task or 'records'
        self.values = np.load(root / f'{task}_values.npy', mmap_mode='r')
        self.valid = np.load(root / f'{task}_valid.npy', mmap_mode='r')
        if (self.values.dtype != np.float32 or self.values.ndim != 2
                or self.values.shape[1] != 200 or self.valid.dtype != np.bool_
                or self.valid.shape != self.values.shape[:1]):
            raise ValueError('MD200 sidecar array schema mismatch')
        row_field = 'original_row' if task != 'records' else 'source_index'
        wanted = {int(row[row_field]): row for row in entries}
        self.rows = {}
        with (root / f'{task}_rows.jsonl').open() as handle:
            count = 0
            for index, line in enumerate(handle):
                row = json.loads(line)
                if row['source_row'] != index:
                    raise ValueError('MD200 sidecar row mapping mismatch')
                if index in wanted:
                    source = wanted[index]
                    if (row['normalized_smiles'] != source['normalized_smiles']
                            or bool(row['valid']) != bool(self.valid[index])):
                        raise ValueError('MD200 sidecar source identity/validity mismatch')
                    self.rows[index] = row
                count += 1
        if count != len(self.valid) or set(self.rows) != set(wanted):
            raise ValueError('MD200 sidecar missing source rows')

    def get(self, row):
        record = self.rows[row]
        values = np.asarray(self.values[row]).copy()
        valid = bool(self.valid[row])
        if not np.isfinite(values).all() or (not valid and np.any(values != 0)):
            raise ValueError('MD200 sidecar payload nonfinite or invalid payload nonzero')
        return torch.from_numpy(values).reshape(1,200), torch.tensor([valid]), record['reason']
