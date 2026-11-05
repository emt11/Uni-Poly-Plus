"""Stage0 changes PT only; every endpoint uses the unchanged BASE FT recipe."""
import math
import torch
from torch import nn
from src.modules import ua3m_paperfuse as native
from src.modules.ua3m_paperfuse import strict_module_state, optimizer_ft, NATIVE, TASKS

ARMS=('P_BASE','P_REVERSE','P_SYM','P_SYM_IND')
BASE_DIRECTIONS=((0,2),(1,0),(2,1))
REVERSE_DIRECTIONS=((0,1),(2,0),(1,2))
PT_SCHEMA='ua3d-base-opt-pretrain-r5'
FT_SCHEMA='ua3d-base-opt-best-r5'
QFP_SEED=20261026
CONFIG=dict(native=native.CONFIG,stage='Stage0',directions={
    'P_BASE':list(map(list,BASE_DIRECTIONS)),
    'P_REVERSE':list(map(list,REVERSE_DIRECTIONS)),
    'P_SYM':list(map(list,BASE_DIRECTIONS+REVERSE_DIRECTIONS)),
    'P_SYM_IND':list(map(list,BASE_DIRECTIONS+REVERSE_DIRECTIONS))},
    qfp_seed=QFP_SEED,ft_policy='unchanged BASE TMOL3M; omit PT-only QFP',
    ft_direction='BASE sum3, coefficient0.08, shared Q2/Q3')


def directions(arm):
    if arm not in ARMS:raise ValueError('outside Stage0 arms')
    return tuple(map(tuple,CONFIG['directions'][arm]))


def alignment_terms(features,availability,identities,arm):
    """Same native decoupled loss, separately normalized for every direction."""
    sums={};counts={}
    with torch.autocast(features[0].device.type,enabled=False):
        for a,b in directions(arm):
            key=f'align_{a}_{b}'
            ids=native.unique_indices(identities,availability[:,a]&availability[:,b]);n=len(ids)
            x,y=features[a][ids].float(),features[b][ids].float()
            if n<2:
                sums[key]=(x.sum()+y.sum())*0.;counts[key]=x.new_tensor(0.);continue
            similarity=(x@y.T)/(x.norm(dim=-1)[:,None]*y.norm(dim=-1)[None,:]+1e-8)/.1
            negative=similarity.masked_fill(torch.eye(n,device=x.device,dtype=torch.bool),-torch.inf)
            denominator=torch.logaddexp(torch.logsumexp(negative,dim=-1),x.new_tensor(math.log(1e-8)))
            sums[key]=(denominator-similarity.diag()).sum();counts[key]=x.new_tensor(float(n))
    return sums,counts


class BaseOptFuse(native.PaperFuse):
    def __init__(self,arm,pretraining=True):
        if arm not in ARMS:raise ValueError('outside Stage0 arms')
        super().__init__('TMOL3M',pretraining)
        self.pt_arm=arm
        if pretraining and arm=='P_SYM_IND':
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(QFP_SEED)
                self.QFP=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256))

    def contrastive_features(self,z):
        if self.pretraining and self.pt_arm=='P_SYM_IND':
            return [self.Q2(z[0].float()),self.Q3(z[1].float()),self.QFP(z[2].float())]
        return super().contrastive_features(z)

    def target_counts(self,batch,labels):
        counts=super().target_counts(batch,labels)
        counts={k:v for k,v in counts.items() if not k.startswith('align_')}
        for a,b in directions(self.pt_arm):
            n=len(native.unique_indices(batch.identity,batch.availability[:,a]&batch.availability[:,b]))
            counts[f'align_{a}_{b}']=torch.tensor(float(n if n>=2 else 0))
        return counts

    def pretrain_result(self,batch,labels):
        # Native CE objectives reproduced exactly, with one shared encoder forward.
        z,atoms,bonds=self.encode_modalities(batch,labels);data=batch.two_d;graphs=len(batch.availability)
        selected=labels['atom_selected'];selected3=labels['token_selected']&(data.bond_length_weight>0)
        logits=self.atom_head(atoms[selected]);logits3=self.line_head(bonds[selected3])
        with torch.autocast(atoms.device.type,enabled=False):
            loss=nn.functional.cross_entropy(logits.float(),labels['atom_label'][selected],reduction='none')
            loss3=nn.functional.cross_entropy(logits3.float(),labels['token_label'][selected3],reduction='none')
            sums={};counts={}
            sums['mask_2d'],counts['mask_2d']=native.task_sum(loss,data.canonical_graph_index[selected],graphs)
            sums['mask_3d'],counts['mask_3d']=native.task_sum(loss3,data.bond_batch[selected3],graphs,data.bond_length_weight[selected3])
            align,n=alignment_terms(self.contrastive_features(z),batch.availability,batch.identity,self.pt_arm)
            sums.update(align);counts.update(n)
        return dict(sums=sums,counts=counts)

    def weights(self):
        alpha=.08 if self.pt_arm in ARMS[:2] else .04
        return dict(mask_2d=1.,mask_3d=1.,**{
            f'align_{a}_{b}':alpha*min(1.,self.completed_updates/1000) for a,b in directions(self.pt_arm)})


def validate_package(package):
    if (package.get('schema')!=PT_SCHEMA or package.get('plan')!='UA3D-BASE-OPT-20261026-01'
        or package.get('revision')!='r5' or package.get('step')!=10000 or package.get('config')!=CONFIG
        or package.get('arm') not in ARMS or package.get('fp_prediction') is not False
        or package.get('stream',{}).get('counters')!=[60000]*3):
        raise ValueError('Stage0 PT checkpoint identity mismatch')


def initialize_ft(package,head):
    validate_package(package);model=BaseOptFuse(package['arm'],False)
    # PT QFP is auxiliary: discard it to keep downstream architecture/objective fixed.
    wanted={k:v for k,v in package['state_dict'].items() if not k.startswith(('atom_head.','line_head.','QFP.'))}
    expected={k for k in model.state_dict() if not k.startswith(('fusion.','head.'))}
    if set(wanted)!=expected:raise ValueError('Stage0 PT/FT explicit state set mismatch')
    for name in (*NATIVE,*native.modal_modules('TMOL3M').keys()):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in wanted.items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    return model
