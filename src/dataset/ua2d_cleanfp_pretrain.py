"""Read-only r4 templates, paired clean/corrupted views and fixed sample order."""
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from .glt_dual import dual_glt_collate
from .ua2d_galformer_pretrain import collate_samples
from .ua2d_fast_pretrain import (PreparedMicrobatches, prepare_fast_sample, decode_template,
                                encode, _tensors)


def paired_sample(template, config, position, fp):
    masked, labels = prepare_fast_sample(template, config=config, seed=42, position=position, fp_target=fp)
    return masked, labels, template['clean'].clone()


def collate_pairs(records):
    masked, labels = collate_samples([(a,b) for a,b,_ in records])
    clean = dual_glt_collate([c for _,_,c in records])
    if (not torch.equal(masked.canonical_graph_index,clean.canonical_graph_index)
            or not torch.equal(masked.incidence_index,clean.incidence_index)):
        raise ValueError('clean/masked index alignment changed')
    return masked, labels, clean


def decode_clean_batch(payload):
    masked, labels, clean, trace = _tensors(pickle.loads(payload))
    return Data.from_dict(masked), labels, Data.from_dict(clean), trace


class CleanMicrobatches(PreparedMicrobatches):
    def __getitem__(self, micro_index):
        if self.environment is None:self._open()
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        start = (int(micro_index)*2+self.rank)*84
        records, rows = [], []
        for offset in range(84):
            with self.prepared_counts.get_lock(): self.prepared_counts[worker] += 1
            position = start + offset
            index = self.stream.index_at(position)
            key = self.keys[index].tobytes()
            if key in self.templates: template = self.templates.pop(key)
            else:
                with self.environment.begin() as txn: payload = txn.get(key)
                if payload is None: raise ValueError('prepared template missing')
                template = decode_template(payload)
                if template['sample_key'] != key: raise ValueError('prepared identity mismatch')
            self.templates[key] = template
            if len(self.templates)>self.cache_entries:self.templates.popitem(last=False)
            fp = np.unpackbits(template['fp_bits'].numpy(), bitorder='little')
            records.append(paired_sample(template,self.config,position,fp))
            rows.append(int(self.rows[index]))
        masked,labels,clean = collate_pairs(records)
        return encode((masked.to_dict(),labels,clean.to_dict(),
                       dict(micro_index=int(micro_index),start_position=start,source_rows=rows)))
