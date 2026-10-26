"""UA-native three-modal paper adapters; fingerprint inputs have no decoder."""
import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from src.modules.glt_dual import BondPathO8,mean_pool,strict_module_state
from src.modules.ua2d_galformer_pretrain import MaskedGalformer3D,task_sum
from src.modules.ua2d_p10k_nofp_align import branch_deterministic_rng

ARMS=('UNI3M','TMOL3M');TASKS=('eea','egb','egc','ei','eps','nc','xc')
PT_SCHEMA='ua3m-paperfuse-pretrain-r1';FT_SCHEMA='ua3m-paperfuse-best-r1'
SEED=20261025;RNG_BASE=86400010;RNG_LIMIT=60000
NATIVE=('o8','norm2','encoder3d','norm3');MASK_HEADS=('atom_head','line_head')
CONFIG=dict(native_dimension=512,native_layers=6,native_attention_denominator=512,modal_dimension=256,
    fp_size=2048,fp_as_input=True,fp_prediction=False,geometry='finite_incident',rng_policy='disjoint_modal_v1',
    uni=dict(heads=8,ffn=1024,dropout=.1,temperature=.07,pt_alpha=.20,directed_reduction='mean6'),
    tmol=dict(fp_hidden=[512,256],fp_dropout=.2,task_embedding=8,gate_hidden=512,head_dropout=.5,
              temperature=.1,alpha=.08,beta=.08,directed_reduction='sum3',positive_in_denominator=False))


def modal_seed(branch,k):
    if branch not in (0,1,2) or type(k) is not int or not 0<=k<RNG_LIMIT:raise ValueError('modal RNG outside finite forward domain')
    return RNG_BASE+(2*k+branch if branch<2 else 120000+k)


def unique_indices(identities,valid):
    seen=set();keep=[]
    for i,ok in enumerate(valid.detach().cpu().tolist()):
        if ok and identities[i] not in seen:seen.add(identities[i]);keep.append(i)
    return torch.tensor(keep,dtype=torch.long,device=valid.device)


def directions(arm):
    if arm=='UNI3M':return ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1))
    if arm=='TMOL3M':return ((0,2),(1,0),(2,1))
    raise ValueError('unknown three-modal arm')


def alignment_terms(features,availability,identities,arm):
    sums={};counts={}
    with torch.autocast(features[0].device.type,enabled=False):
        for a,b in directions(arm):
            name=f'align_{a}_{b}';ids=unique_indices(identities,availability[:,a]&availability[:,b]);n=len(ids)
            if n<2:
                sums[name]=(features[a].float().sum()+features[b].float().sum())*0.;counts[name]=features[a].new_tensor(0.,dtype=torch.float32);continue
            x,y=features[a][ids].float(),features[b][ids].float()
            if arm=='UNI3M':
                similarity=F.normalize(x,dim=-1)@F.normalize(y,dim=-1).T/.07
                losses=torch.logsumexp(similarity,dim=-1)-similarity.diag()
            else:
                similarity=(x@y.T)/(torch.linalg.vector_norm(x,dim=-1)[:,None]*torch.linalg.vector_norm(y,dim=-1)[None,:]+1e-8)/.1
                negative=similarity.masked_fill(torch.eye(n,device=x.device,dtype=torch.bool),-torch.inf)
                denominator=torch.logaddexp(torch.logsumexp(negative,dim=-1),x.new_tensor(math.log(1e-8)))
                losses=denominator-similarity.diag()
            sums[name]=losses.sum();counts[name]=x.new_tensor(float(n))
    return sums,counts


def entropy(weights):return -(weights.float()*torch.log(weights.float()+1e-8)).sum(-1).mean()


class Uni3MFusion(nn.Module):
    def __init__(self):
        super().__init__();self.attention=nn.MultiheadAttention(256,8,dropout=.1,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,256))
        self.norm1=nn.LayerNorm(256);self.norm2=nn.LayerNorm(256);self.dropout=nn.Dropout(.1)
        self.scores=nn.ModuleList([nn.Linear(256,1) for _ in range(3)])
    def forward(self,z,available):
        token=torch.stack(z,dim=1)
        attended,_=self.attention(token,token,token,key_padding_mask=~available,need_weights=False)
        x=self.norm1(token+self.dropout(attended));x=self.norm2(x+self.dropout(self.ffn(x)))
        logits=torch.cat([score(x[:,i]) for i,score in enumerate(self.scores)],dim=-1)
        weights=logits.masked_fill(~available,-torch.inf).softmax(-1)
        return (x*weights[:,:,None]).sum(1),weights


class Tmol3MGating(nn.Module):
    def __init__(self):
        super().__init__();self.task_number=nn.Embedding(1,8);self.task_type=nn.Embedding(2,8)
        self.gate=nn.Sequential(nn.Linear(776,512),nn.ReLU(),nn.Linear(512,3))
    def forward(self,z,available):
        count=len(z[0]);number=torch.zeros(count,device=z[0].device,dtype=torch.long)
        task=self.task_number(number)+self.task_type(torch.ones_like(number))
        logits=self.gate(torch.cat((*z,task),dim=-1));weights=logits.masked_fill(~available,-torch.inf).softmax(-1)
        return (torch.stack(z,dim=1)*weights[:,:,None]).sum(1),weights


def modal_modules(arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(SEED)
        modules=nn.ModuleDict(dict(A2=nn.Linear(512,256),A3=nn.Linear(512,256)))
        if arm=='UNI3M':modules['fp_encoder']=nn.Linear(2048,256)
        else:
            modules['fp_encoder']=nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(.2),nn.Linear(256,256))
            modules['Q2']=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256))
            modules['Q3']=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256))
    return modules


def property_head(dropout):return nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(dropout),nn.Linear(256,1))


class PaperFuse(nn.Module):
    def __init__(self,arm,pretraining=True):
        super().__init__()
        if arm not in ARMS:raise ValueError('unregistered arm')
        self.arm=arm;self.pretraining=pretraining;self.stage='head';self.completed_updates=0;self.counters=[0,0,0]
        dropout=.1 if pretraining else 0.
        self.o8=BondPathO8(dropout,attention_scale_denominator=512);self.norm2=nn.LayerNorm(512)
        self.encoder3d=MaskedGalformer3D(dropout,attention_scale_denominator=512);self.norm3=nn.LayerNorm(512)
        for name,module in modal_modules(arm).items():setattr(self,name,module)
        if pretraining:
            self.atom_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,101))
            self.line_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,25755))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(SEED)
                self.fusion=Uni3MFusion() if arm=='UNI3M' else Tmol3MGating()
                self.head=property_head(.1 if arm=='UNI3M' else .5)
            # Native mask token remains in its native module state, but has no FT objective.
            self.encoder3d.mask_token.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode)
        if mode and not self.pretraining and self.stage=='head':
            for name in NATIVE:getattr(self,name).eval()
        return self
    def rng_context(self,branch,device):
        if not self.pretraining:return nullcontext()
        return branch_deterministic_rng(modal_seed(branch,self.counters[branch]),device)
    def stream_state(self):
        if any(type(k) is not int or not 0<=k<=60000 for k in self.counters):raise ValueError('invalid stream counter')
        return dict(policy='disjoint_modal_v1',B=RNG_BASE,counters=list(self.counters),forward_domain=[0,60000],terminal_counter=60000)
    def restore_stream(self,state):
        if state.get('policy')!='disjoint_modal_v1' or state.get('B')!=RNG_BASE or state.get('forward_domain')!=[0,60000] or state.get('terminal_counter')!=60000:
            raise ValueError('stream metadata mismatch')
        values=state.get('counters',[])
        if len(values)!=3 or any(type(k) is not int or not 0<=k<=60000 for k in values):raise ValueError('stream counter mismatch')
        self.counters=list(values)
    def encode_modalities(self,batch,labels=None):
        if self.pretraining:
            for branch in range(3):modal_seed(branch,self.counters[branch])
            if not batch.availability.all():raise ValueError('PT cannot include invalid geometry')
        fp=batch.morgan_fp
        if fp.shape!=(len(batch.availability),2048) or not ((fp==0)|(fp==1)).all() or not ((fp.sum(-1)>0)&(fp.sum(-1)<2048)).all():raise ValueError('invalid binary FP input')
        if labels is not None and 'fp_target' in labels:raise ValueError('FP prediction target forbidden')
        device=fp.device
        with self.rng_context(0,device):atoms,_=self.o8(batch.two_d)
        h2=self.norm2(mean_pool(atoms,batch.two_d.canonical_graph_index,len(fp)))
        bonds=None;h3=h2.new_zeros((len(fp),512))
        if batch.three_d is not None:
            with self.rng_context(1,device):bonds,graphs,valid=self.encoder3d(batch.three_d,labels)
            if not valid.all():raise ValueError('PASS 3D subbatch unexpectedly unavailable')
            h3=h3.index_copy(0,batch.three_d_rows,self.norm3(graphs))
        with self.rng_context(2,device):zfp=self.fp_encoder(fp.float())
        if self.pretraining:self.counters=[k+1 for k in self.counters]
        return [self.A2(h2),self.A3(h3)*batch.availability[:,1:2].to(h3),zfp],atoms,bonds
    def contrastive_features(self,z):return z if self.arm=='UNI3M' else [self.Q2(z[0].float()),self.Q3(z[1].float()),self.Q2(z[2].float())]
    def predict(self,batch):
        z,_,_=self.encode_modalities(batch);f,weights=self.fusion(z,batch.availability)
        return self.head(f).view(-1),weights,z
    def forward(self,batch):return self.predict(batch)[0]
    def training_loss(self,batch):
        pred,weights,z=self.predict(batch);mse=F.mse_loss(pred,batch.y.view(-1))
        if self.arm=='UNI3M':return mse,dict(mse=mse.detach())
        sums,counts=alignment_terms(self.contrastive_features(z),batch.availability,batch.identity,self.arm)
        align=sum(v/counts[k].clamp_min(1) for k,v in sums.items());h=entropy(weights)
        return mse+.08*align-.08*h,dict(mse=mse.detach(),align=align.detach(),entropy=h.detach())
    def target_counts(self,batch,labels):
        data=batch.two_d;graphs=len(batch.availability)
        def graphs_count(index):return torch.bincount(index,minlength=graphs).gt(0).sum().float()
        selected=labels['atom_selected'];line=labels['token_selected']&(data.bond_length_weight>0)
        counts=dict(mask_2d=graphs_count(data.canonical_graph_index[selected]),mask_3d=graphs_count(data.bond_batch[line]))
        for a,b in directions(self.arm):
            n=len(unique_indices(batch.identity,batch.availability[:,a]&batch.availability[:,b]));counts[f'align_{a}_{b}']=torch.tensor(float(n if n>=2 else 0))
        return counts
    def pretrain_result(self,batch,labels):
        z,atoms,bonds=self.encode_modalities(batch,labels);data=batch.two_d;graphs=len(batch.availability)
        selected=labels['atom_selected'];selected3=labels['token_selected']&(data.bond_length_weight>0)
        logits=self.atom_head(atoms[selected]);logits3=self.line_head(bonds[selected3])
        with torch.autocast(atoms.device.type,enabled=False):
            loss=F.cross_entropy(logits.float(),labels['atom_label'][selected],reduction='none')
            loss3=F.cross_entropy(logits3.float(),labels['token_label'][selected3],reduction='none')
            sums={};counts={};sums['mask_2d'],counts['mask_2d']=task_sum(loss,data.canonical_graph_index[selected],graphs)
            sums['mask_3d'],counts['mask_3d']=task_sum(loss3,data.bond_batch[selected3],graphs,data.bond_length_weight[selected3])
            align,n=alignment_terms(self.contrastive_features(z),batch.availability,batch.identity,self.arm);sums.update(align);counts.update(n)
        return dict(sums=sums,counts=counts)
    def weights(self):
        alpha=.20/6 if self.arm=='UNI3M' else .08
        return dict(mask_2d=1.,mask_3d=1.,**{f'align_{a}_{b}':alpha*min(1.,self.completed_updates/1000) for a,b in directions(self.arm)})


def load_initial(model,common,heads,modal_state):
    if common.get('step')!=0 or common.get('seed')!=42 or common.get('plan')!='UA2D-PRETRAIN-FP-20261019-01' or common.get('schema')!='ua2d_galformer_objectives_v3':raise ValueError('original step0 identity mismatch')
    if heads.get('step')!=0 or heads.get('strategy')!='mask_align' or heads.get('schema')!=common['schema']:raise ValueError('V1 mask heads mismatch')
    source=dict(common['state_dict'],**heads['state_dict'])
    if any(k.startswith('fp_head.') for k in source):raise ValueError('FP supervision in initialization')
    for name in (*NATIVE,*MASK_HEADS):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)})
    modules=modal_modules(model.arm);strict_module_state(modules,modal_state)
    for name,module in modules.items():strict_module_state(getattr(model,name),module.state_dict())
    return dict(loaded=list((*NATIVE,*MASK_HEADS)),excluded=[k for k in source if not k.startswith(tuple(name+'.' for name in (*NATIVE,*MASK_HEADS)))])


def initialize_ft(package,head):
    if package.get('plan')!='UA3M-PAPERFUSE-20261025-01' or package.get('revision')!='r1' or package.get('schema')!=PT_SCHEMA or package.get('step')!=10000 or package.get('config')!=CONFIG or package.get('fp_prediction') is not False:raise ValueError('PT checkpoint identity mismatch')
    model=PaperFuse(package['arm'],False)
    wanted={k:v for k,v in package['state_dict'].items() if not k.startswith(('atom_head.','line_head.'))}
    expected={k for k in model.state_dict() if not k.startswith(('fusion.','head.'))}
    if set(wanted)!=expected:raise ValueError('explicit PT to FT module set mismatch')
    for name in (*NATIVE,*modal_modules(model.arm).keys()):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in wanted.items() if k.startswith(prefix)})
    strict_module_state(model.head,head);return model


def optimizer_ft(model,stage):
    if stage not in ('head','joint'):raise ValueError('unknown FT stage')
    model.stage=stage
    def native(n):return n.startswith(tuple(k+'.' for k in NATIVE))
    for name,p in model.named_parameters():p.requires_grad_(name!='encoder3d.mask_token' and (stage=='joint' or not native(name)))
    groups=[]
    if stage=='joint':groups.append(dict(name='native',params=[p for n,p in model.named_parameters() if native(n) and p.requires_grad],lr=1e-4))
    groups.append(dict(name='modal_fusion_head',params=[p for n,p in model.named_parameters() if not native(n) and p.requires_grad],lr=3e-4 if stage=='head' else 1e-3))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage mismatch')
    return torch.optim.AdamW(groups,weight_decay=.02),[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups]
