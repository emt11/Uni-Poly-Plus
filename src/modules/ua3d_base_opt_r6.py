"""Independent r6 adapters; existing scientific modules are read-only dependencies."""
import copy
import math
import torch
from torch import nn
from src.modules import ua3m_paperfuse as native
from src.modules import ua3d_spatial as spatial
from src.modules.ua3m_paperfuse import strict_module_state, NATIVE, TASKS

PT_SCHEMA='ua3d-base-opt-pretrain-r6'
FT_SCHEMA='ua3d-base-opt-best-r6'
CURRENT=((0,2),(1,0),(2,1))
PAPER=((0,1),(2,0),(1,2))
PT_ARMS=('P_CURRENT_SHARED','P_CURRENT_IND','P_PAPER_SHARED','P_PAPER_IND','P_SYM_SHARED')
CONFIG=dict(native=native.CONFIG,revision='r6',ft_direction='current',ft_projection='shared Q2/Q3',
    qfp_initial='deepcopy common Q2; independent parameters; discard before FT',attention_denominator=512)
DEFAULT_FT=dict(align=.08,entropy=.08,joint_lr=.001,scheduler='restart',p2=0.,p3=0.,pfp=.2,phead=.5,fusion='gate')

def pt_config(arm='P_CURRENT_SHARED',native_p=.1,fp_p=.2):
    if arm not in PT_ARMS or native_p not in (0.,.1) or fp_p not in (0.,.2):
        raise ValueError('outside registered PT factors')
    return dict(arm=arm,directions=[list(p) for p in (CURRENT+PAPER if arm=='P_SYM_SHARED' else PAPER if 'PAPER' in arm else CURRENT)],
        independent=arm.endswith('_IND'),align_each=.04 if arm=='P_SYM_SHARED' else .08,pnative=native_p,pfp=fp_p)

def alignment_terms(features,availability,identities,directions=CURRENT):
    sums={};counts={}
    with torch.autocast(features[0].device.type,enabled=False):
        for a,b in directions:
            key=f'align_{a}_{b}'
            ids=native.unique_indices(identities,availability[:,a]&availability[:,b]);n=len(ids)
            if n<2:
                sums[key]=(features[a].float().sum()+features[b].float().sum())*0.
                counts[key]=features[a].new_tensor(0.,dtype=torch.float32);continue
            x,y=features[a][ids].float(),features[b][ids].float()
            sim=(x@y.T)/(x.norm(dim=-1)[:,None]*y.norm(dim=-1)[None,:]+1e-8)/.1
            neg=sim.masked_fill(torch.eye(n,device=x.device,dtype=torch.bool),-torch.inf)
            denominator=torch.logaddexp(torch.logsumexp(neg,dim=-1),x.new_tensor(math.log(1e-8)))
            sums[key]=(denominator-sim.diag()).sum();counts[key]=x.new_tensor(float(n))
    return sums,counts

def set_dropout(module,p):
    # Identity positions stay Identity, including the 3D FFN intermediate position.
    for child in module.modules():
        if isinstance(child,nn.Dropout):child.p=p

class BaseOptFuse(native.PaperFuse):
    def __init__(self,pretraining=True,pt=None,ft=None):
        super().__init__('TMOL3M',pretraining)
        self.pt=copy.deepcopy(pt if pt is not None else pt_config())
        self.ft=copy.deepcopy(ft if ft is not None else DEFAULT_FT)
        if set(self.ft)!=set(DEFAULT_FT):raise ValueError('FT configuration keys mismatch')
        if pretraining:
            set_dropout(self.o8,self.pt['pnative']);set_dropout(self.encoder3d,self.pt['pnative'])
            set_dropout(self.fp_encoder,self.pt['pfp'])
            if self.pt['independent']:self.QFP=copy.deepcopy(self.Q2)
        else:
            set_dropout(self.o8,self.ft['p2']);set_dropout(self.encoder3d,self.ft['p3'])
            set_dropout(self.fp_encoder,self.ft['pfp']);set_dropout(self.head,self.ft['phead'])
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(20261026)
                if self.ft['fusion']=='residual':
                    self.residual=nn.Linear(768,256)
                    nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
                elif self.ft['fusion']=='modalnorm':self.modalnorm=nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
                elif self.ft['fusion']!='gate':raise ValueError('unknown fusion')
        self.last_diagnostics={}

    def contrastive_features(self,z):
        return [self.Q2(z[0].float()),self.Q3(z[1].float()),
            self.QFP(z[2].float()) if self.pretraining and self.pt['independent'] else self.Q2(z[2].float())]

    def fuse(self,z,available):
        fused_z=z
        if hasattr(self,'modalnorm'):
            fused_z=[norm(value) for norm,value in zip(self.modalnorm,z)]
            fused_z[1]=fused_z[1]*available[:,1:2].to(fused_z[1])
        f,weights=self.fusion(fused_z,available)
        if hasattr(self,'residual'):f=f+self.residual(torch.cat(z,dim=-1))
        with torch.no_grad():
            self.last_diagnostics=dict(norm_before=torch.stack([v.float().norm(dim=-1).mean() for v in z]),
                norm_after=torch.stack([v.float().norm(dim=-1).mean() for v in fused_z]),
                gate_mean=weights.float().mean(0),gate_entropy=native.entropy(weights),
                gate_saturation=(weights.float().max(-1).values>=.95).float().mean())
        return f,weights

    def predict(self,batch):
        z,_,_=self.encode_modalities(batch);f,weights=self.fuse(z,batch.availability)
        if not torch.isfinite(f).all() or not torch.isfinite(weights).all():raise FloatingPointError('nonfinite fusion')
        if torch.count_nonzero(z[1][~batch.availability[:,1]]):raise ValueError('missing 3D not strictly zero')
        return self.head(f).view(-1),weights,z

    def training_loss(self,batch):
        pred,weights,z=self.predict(batch);mse=nn.functional.mse_loss(pred,batch.y.view(-1))
        sums,counts=alignment_terms(self.contrastive_features(z),batch.availability,batch.identity)
        align=sum(v/counts[k].clamp_min(1) for k,v in sums.items());h=native.entropy(weights)
        loss=mse+self.ft['align']*align-self.ft['entropy']*h
        return loss,dict(mse=mse.detach(),align=align.detach(),entropy=h.detach())

    def target_counts(self,batch,labels):
        counts={k:v for k,v in super().target_counts(batch,labels).items() if not k.startswith('align_')}
        for a,b in self.pt['directions']:
            n=len(native.unique_indices(batch.identity,batch.availability[:,a]&batch.availability[:,b]))
            counts[f'align_{a}_{b}']=torch.tensor(float(n if n>=2 else 0))
        return counts

    def pretrain_result(self,batch,labels):
        z,atoms,bonds=self.encode_modalities(batch,labels);data=batch.two_d;graphs=len(batch.availability)
        selected=labels['atom_selected'];selected3=labels['token_selected']&(data.bond_length_weight>0)
        logits=self.atom_head(atoms[selected]);logits3=self.line_head(bonds[selected3])
        with torch.autocast(atoms.device.type,enabled=False):
            loss=nn.functional.cross_entropy(logits.float(),labels['atom_label'][selected],reduction='none')
            loss3=nn.functional.cross_entropy(logits3.float(),labels['token_label'][selected3],reduction='none')
            sums={};counts={}
            sums['mask_2d'],counts['mask_2d']=native.task_sum(loss,data.canonical_graph_index[selected],graphs)
            sums['mask_3d'],counts['mask_3d']=native.task_sum(loss3,data.bond_batch[selected3],graphs,data.bond_length_weight[selected3])
            align,n=alignment_terms(self.contrastive_features(z),batch.availability,batch.identity,self.pt['directions'])
            sums.update(align);counts.update(n)
        return dict(sums=sums,counts=counts)

    def weights(self):
        return dict(mask_2d=1.,mask_3d=1.,**{f'align_{a}_{b}':self.pt['align_each']*min(1.,self.completed_updates/1000)
            for a,b in self.pt['directions']})

    def runtime_contract(self):
        return dict(stage=self.stage,training=self.training,
            dropout={n:dict(p=v.p,training=v.training) for n,v in self.named_modules() if isinstance(v,nn.Dropout)},
            native_modes={n:getattr(self,n).training for n in NATIVE})

def validate_package(package,pt=None):
    if (package.get('schema')!=PT_SCHEMA or package.get('plan')!='UA3D-BASE-OPT-20261026-01'
        or package.get('revision')!='r6' or package.get('step')!=10000 or package.get('config')!=CONFIG
        or package.get('fp_prediction') is not False or package.get('stream',{}).get('counters')!=[60000]*3
        or (pt is not None and package.get('pt_config')!=pt)):
        raise ValueError('r6 PT checkpoint identity mismatch')

def ft_state(package,model):
    """Exact allowed source conversion; discard only declared PT auxiliary heads."""
    wanted={k:v for k,v in package['state_dict'].items() if not k.startswith(('atom_head.','line_head.','QFP.'))}
    expected={k for k in model.state_dict() if not k.startswith(('fusion.','head.','residual.','modalnorm.'))}
    if set(wanted)!=expected:raise ValueError('explicit PT/FT state set mismatch')
    for name in (*NATIVE,*native.modal_modules('TMOL3M').keys()):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in wanted.items() if k.startswith(prefix)})

def initialize_ft(package,head,recipe):
    if recipe['pt_id']=='HIST_R4_BASE':
        spatial.validate_package(package)
        if package['arm']!='BASE':raise ValueError('historical PT must be BASE')
    else:
        validate_package(package,recipe['pt'])
        if package.get('pt_id')!=recipe['pt_id']:raise ValueError('PT endpoint source mismatch')
    model=BaseOptFuse(False,ft=recipe['ft']);ft_state(package,model);strict_module_state(model.head,head)
    return model

def optimizer_ft(model,stage):
    opt,groups=native.optimizer_ft(model,stage)
    if stage=='joint':
        opt.param_groups[1]['lr']=model.ft['joint_lr'];groups[1]['lr']=model.ft['joint_lr']
    return opt,groups
