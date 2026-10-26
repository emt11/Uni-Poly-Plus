"""Fixed A020 neural FP adapters. No pretrained FP objective or kernel path."""
from dataclasses import asdict
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel, strict_module_state
from src.modules.ua2d_p10k_nofp_align import arm_config, parse_model_stream_state, LEGACY_STREAM_POLICY
from src.training.ua2d_p10k_nofp_align import validate_parent as validate_nofp_parent

ARMS = ('UNI_ZERO', 'UNI_FP', 'TMOL_ZERO', 'TMOL_FP', 'FP_MLP')
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
SCHEMA = 'ua2d-a020-fp-fusion-best-r1'
COMMON_SCHEMA = 'ua2d-a020-fp-fusion-common-r1'
BRANCH_SEED = 20261024
ARCHITECTURE = dict(parent='D_A020', o8_layers=6, dimension=512, hop=2, attention_denominator=512,
    readout='mean_norm2', fp='2048-128-GELU-128-LayerNorm', graph_projection='512-128-LayerNorm',
    uni='2token-128dim-4head-residual-LN-FFN128x256x128-residual-LN-attention128x32x1',
    tmol='concat(graph128,fp128,task7)-64-GELU-2-softmax;zero-last',
    residual='H(h+R*u);R128x512-no-bias-zero', graph_head='512-512-GELU-dropout0.1-1',
    fp_head='128-512-GELU-dropout0.1-1', adapter_dropout=0., encoder_dropout=0., modality_id_init_normal_std=.02, branch_seed=BRANCH_SEED)


def encoder():
    return nn.Sequential(nn.Linear(2048,128), nn.GELU(), nn.Linear(128,128), nn.LayerNorm(128))


class UniAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality = nn.Parameter(torch.empty(2,128)); nn.init.normal_(self.modality, std=.02)
        self.attention = nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.norm1 = nn.LayerNorm(128); self.norm2 = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.score = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,g,z,task_index):
        tokens=torch.stack((g,z),dim=1)+self.modality
        attended,_=self.attention(tokens,tokens,tokens,need_weights=False)
        x=self.norm1(tokens+attended);x=self.norm2(x+self.ffn(x))
        weights=torch.softmax(self.score(x),dim=1)
        return (weights*x).sum(dim=1),weights.squeeze(-1)


class TmolAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(263,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
    def forward(self,g,z,task_index):
        task=nn.functional.one_hot(torch.full((len(g),),task_index,device=g.device,dtype=torch.long),7).to(g.dtype)
        weights=torch.softmax(self.gate(torch.cat((g,z,task),dim=-1)),dim=-1)
        return weights[:,:1]*g+weights[:,1:]*z,weights


def new_branches():
    # All additions are initialized together under an isolated CPU RNG. E is shared.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(BRANCH_SEED)
        modules=nn.ModuleDict(dict(fp_encoder=encoder(),graph_projection=nn.Sequential(nn.Linear(512,128),nn.LayerNorm(128)),
            uni=UniAdapter(),tmol=TmolAdapter(),residual=nn.Linear(128,512,bias=False),
            fp_head=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))))
        nn.init.zeros_(modules['residual'].weight)
    return modules


def common_package():
    branches=new_branches()
    return dict(schema=COMMON_SCHEMA,branch_seed=BRANCH_SEED,architecture=ARCHITECTURE,
                state_dict={k:v.detach().cpu().clone() for k,v in branches.state_dict().items()})


def load_common(common):
    if common.get('schema')!=COMMON_SCHEMA or common.get('branch_seed')!=BRANCH_SEED or common.get('architecture')!=ARCHITECTURE:
        raise ValueError('common initialization identity mismatch')
    branches=new_branches();strict_module_state(branches,common['state_dict']);return branches


class FusionModel(Pure2DGLTModel):
    def __init__(self,arm,task,common):
        if arm not in ARMS[:-1] or task not in TASKS:raise ValueError('unregistered graph adapter')
        super().__init__(dropout=0.,attention_scale_denominator=512)
        self.head[2].p=.1;self.stage='head';self.arm=arm;self.task_index=TASKS.index(task)
        branches=load_common(common)
        self.fp_encoder=branches['fp_encoder'];self.graph_projection=branches['graph_projection']
        self.residual=branches['residual'];self.adapter=branches['uni' if arm.startswith('UNI') else 'tmol']
    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self
    def predict_features(self,h,fp):
        validate_fp_tensor(fp)
        value=torch.zeros_like(fp) if self.arm.endswith('ZERO') else fp
        z=self.fp_encoder(value.float());g=self.graph_projection(h)
        u,weights=self.adapter(g,z,self.task_index)
        self.last_modality_weights=weights.detach()
        return self.head(h+self.residual(u)).view(-1)
    def forward(self,data):return self.predict_features(self.features(data),data.morgan_fp)


class FPModel(nn.Module):
    def __init__(self,common):
        super().__init__();branches=load_common(common)
        self.fp_encoder=branches['fp_encoder'];self.head=branches['fp_head'];self.stage='head';self.arm='FP_MLP'
    def forward(self,data):
        validate_fp_tensor(data.morgan_fp)
        return self.head(self.fp_encoder(data.morgan_fp.float())).view(-1)


def validate_fp_tensor(fp):
    if fp.ndim!=2 or fp.shape[1]!=2048 or not torch.isfinite(fp).all() or not ((fp==0)|(fp==1)).all():
        raise ValueError('expected finite binary Morgan [batch,2048]')
    if not ((fp.sum(dim=1)>0)&(fp.sum(dim=1)<2048)).all():raise ValueError('invalid empty/saturated real FP before ZERO injection')


def validate_parent(parent):
    validate_nofp_parent(parent,'D_A020')
    if parent['config']!=asdict(arm_config('D_A020')):
        raise ValueError('A020 config mismatch')
    if any(k.startswith('fp_head.') for k in parent['state_dict']):raise ValueError('FP head in no-FP parent')
    policy,_=parse_model_stream_state(parent['stream_positions'],True)
    if policy!=LEGACY_STREAM_POLICY:raise ValueError('historical legacy parent required')
    return parent


def initialize(arm,task,common,parent=None,head=None):
    if arm=='FP_MLP':return FPModel(common)
    validate_parent(parent);model=FusionModel(arm,task,common)
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('invalid stage')
    model.stage=stage
    def backbone(name):return name.startswith(('o8.','norm2.'))
    for name,p in model.named_parameters():p.requires_grad_(stage=='joint' or not backbone(name))
    groups=[]
    if stage=='joint' and model.arm!='FP_MLP':
        groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if backbone(n)],lr=1e-4))
    groups.append(dict(name='head_adapter',params=[p for n,p in model.named_parameters() if not backbone(n)],lr=3e-4 if stage=='head' else 1e-3))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage mismatch')
    evidence=[dict(name=g['name'],lr=g['lr'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def restore(saved,cfg,common):
    if saved.get('schema')!=SCHEMA or saved.get('config')!=cfg or saved.get('architecture')!=ARCHITECTURE:
        raise ValueError('fusion checkpoint identity mismatch')
    model=FPModel(common) if cfg['arm']=='FP_MLP' else FusionModel(cfg['arm'],cfg['task'],common)
    strict_module_state(model,saved['state_dict']);return model
