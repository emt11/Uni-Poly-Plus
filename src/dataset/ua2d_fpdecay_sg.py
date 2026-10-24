"""Read-only original cache, eligible values are LOGICAL slots, never source rows."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from src.dataset.ua2d_fast_pretrain import CACHE_SCHEMA,decode_template,prepare_fast_sample,encode,decode_batch,worker_init
from src.dataset.ua2d_galformer_pretrain import collate_samples


def resolve_slot(eligible,keys,source_rows,selected):
    logical=int(eligible[selected])
    if logical<0 or logical>=len(keys) or len(keys)!=len(source_rows):raise ValueError('logical slot out of bounds')
    return logical,keys[logical].tobytes(),int(source_rows[logical])


def micro_start(micro_index):
    if not 0<=micro_index<5000*6:raise ValueError('single-GPU micro index outside fixed trajectory')
    return micro_index*168


class PreparedSingleMicrobatches(torch.utils.data.Dataset):
    """One ordered item is one CPU microbatch, independent of worker scheduling."""
    def __init__(self, root, mapping_root, config, *, prepared_counts, cache_entries=2048):
        self.root = Path(root)
        self.mapping_root = Path(mapping_root)
        self.config = config
        self.prepared_counts = prepared_counts
        self.cache_entries = cache_entries
        self.environment = self.keys = self.rows = self.stream = None
        self.templates = OrderedDict()

    def __len__(self):return 5000 * 6

    def _open(self):
        import json
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream
        meta = json.loads((self.root / 'metadata.json').read_text())
        if meta.get('schema') != CACHE_SCHEMA or meta.get('status') != 'COMPLETE' or meta.get('train_rows') != 911391:
            raise ValueError('prepared cache incomplete or incompatible')
        self.environment = lmdb.open(str(self.root / 'templates.lmdb'), readonly=True,
                                     lock=False, readahead=True, max_readers=128)
        self.keys = np.load(self.root / 'keys.npy', mmap_mode='r', allow_pickle=False)
        self.rows = np.load(self.root / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        if len(self.keys) != 911391 or self.rows.shape != (911391,):raise ValueError('cache row mapping changed')
        mapping=json.loads((self.mapping_root/'metadata.json').read_text())
        if (mapping.get('plan')!='UA2D-PTSCALE-20261022-01' or mapping.get('status')!='COMPLETE'
            or mapping.get('schema')!='ua2d-ptscale-logical-map-r1' or mapping.get('records')!=910756
            or mapping.get('parent_cache')!=str(self.root)):raise ValueError('mapping identity/COMPLETE mismatch')
        self.eligible=np.load(self.mapping_root/'eligible_logical.npy',mmap_mode='r',allow_pickle=False)
        self.selected_keys=np.load(self.mapping_root/'keys.npy',mmap_mode='r',allow_pickle=False)
        self.selected_rows=np.load(self.mapping_root/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        if self.eligible.shape!=(910756,) or self.selected_keys.shape!=(910756,) or self.selected_rows.shape!=(910756,):raise ValueError('logical mapping shape mismatch')
        self.stream = OrderedSampleStream(910756, 42)

    def __getitem__(self, micro_index):
        if self.environment is None:self._open()
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        start = micro_start(int(micro_index))
        records, rows, logical_slots = [], [], []
        for offset in range(168):
            with self.prepared_counts.get_lock():self.prepared_counts[worker] += 1
            position = start + offset
            selected = self.stream.index_at(position)
            index, key, source_row = resolve_slot(self.eligible,self.keys,self.rows,selected)
            if key!=self.selected_keys[selected].tobytes() or source_row!=int(self.selected_rows[selected]):raise ValueError('persisted logical/source mapping mismatch')
            if key in self.templates:
                template = self.templates.pop(key)
            else:
                with self.environment.begin() as txn:payload = txn.get(key)
                if payload is None:raise ValueError('prepared template missing')
                template = decode_template(payload)
                if template['sample_key'] != key:raise ValueError('prepared identity mismatch')
            if not bool(template['clean'].geometry_valid):raise ValueError('excluded invalid geometry entered PT; no fallback')
            self.templates[key] = template
            if len(self.templates) > self.cache_entries:self.templates.popitem(last=False)
            fp = np.unpackbits(template['fp_bits'].numpy(), bitorder='little') if self.config.fp_enabled else None
            records.append(prepare_fast_sample(template, config=self.config, seed=42, position=position, fp_target=fp))
            rows.append(source_row);logical_slots.append(index)
        data, labels = collate_samples(records)
        return encode((data, labels, dict(micro_index=int(micro_index), start_position=start, source_rows=rows,logical_slots=logical_slots)))
