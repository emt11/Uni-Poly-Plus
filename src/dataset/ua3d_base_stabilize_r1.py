"""Unchanged raw input semantics; PT bounded spawn reader, FT historical zero worker."""
import torch
from src.dataset import ua3m_paperfuse as raw
from src.dataset.ua3d_cond_spectral_r3 import DecodedPT as BoundedPT,pt_collate,pt_batch,worker_init

class DecodedPT(BoundedPT):
    def __getitem__(self,index):
        data,labels,trace=super().__getitem__(index)
        trace['identities']=[x.hex() if isinstance(x,bytes) else x for x in labels['identity']]
        trace['mask']=dict(atom_selected=labels['atom_selected'].nonzero().view(-1).tolist(),token_selected=labels['token_selected'].nonzero().view(-1).tolist(),token_action=labels['token_action'].tolist(),token_donor=labels['token_donor'].tolist(),seed=42,definition='position-dependent original mask/corruption streams')
        return data,labels,trace

class Data(raw.Data):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.training_rows=[];self.record_training=False
    def __getitem__(self,i):
        value=super().__getitem__(i)
        if self.record_training:self.training_rows.append(dict(row=self.indices[i],key=str(self.keys[i])))
        return value

def loader(ds,indices,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=24,shuffle=shuffle,num_workers=0,drop_last=False,collate_fn=raw.collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
