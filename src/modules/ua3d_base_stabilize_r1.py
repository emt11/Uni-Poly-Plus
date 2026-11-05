"""New-cycle random factory and strict native BASE conversion. No old tensor inputs."""
import copy,torch
from torch import nn
from src.modules import ua3d_base_opt_r6 as base
from src.modules import ua3m_paperfuse as native
from src.modules.ua2d_galformer_pretrain import UA2DGalformerPretrainer,PretrainConfig
from src.modules.ua3d_stability import configure,check_det,numerical_state,rng_state,clone_tree,exact_compare
PLAN='UA3D-BASE-STABILIZE-20261105-01';REVISION='r1'
PT_SCHEMA='ua3d-base-stabilize-pretrain-r1';INITIAL_SCHEMA='ua3d-base-stabilize-initial-r1';HEAD_SCHEMA='ua3d-base-stabilize-head-r1';FT_SCHEMA='ua3d-base-stabilize-estimator-r1'
TASKS=native.TASKS;NATIVE=native.NATIVE;MODAL=('A2','A3','fp_encoder','Q2','Q3');MASK=('atom_head','line_head')
GROUPS=('S42_A','S42_B','S142','S242');CANONICAL=('S42_A','S142','S242');SEEDS=dict(S42_A=42,S42_B=42,S142=142,S242=242)
PT=base.pt_config('P_CURRENT_SHARED',.1,0.);FT=copy.deepcopy(base.DEFAULT_FT)
CONFIG=dict(architecture=base.CONFIG,pt=PT,ft=FT,native_factory='UA2DGalformerPretrainer(mask_align)',modal_seed=20261025,head_seed='20261025+100*task_index+fold')
strict_module_state=native.strict_module_state;optimizer_ft=base.optimizer_ft

def fresh_factory():
    # Factory order consumes proj2/proj3 RNG before atom_head; line_head has its
    # historical isolated20261019 stream. Never reset generated modules again.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);factory=UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align'))
        model=base.BaseOptFuse.__new__(base.BaseOptFuse);nn.Module.__init__(model)
        model.arm='TMOL3M';model.pretraining=True;model.stage='head';model.completed_updates=0;model.counters=[0,0,0];model.pt=copy.deepcopy(PT);model.ft=copy.deepcopy(FT);model.last_diagnostics={}
        for name in NATIVE:setattr(model,name,getattr(factory,name))
        for name,mod in native.modal_modules('TMOL3M').items():setattr(model,name,mod)
        for name in MASK:setattr(model,name,getattr(factory,name))
        base.set_dropout(model.o8,.1);base.set_dropout(model.encoder3d,.1);base.set_dropout(model.fp_encoder,0.)
        state=clone_tree(model.state_dict())
        expected={k for k in factory.state_dict() if k.startswith(tuple(n+'.' for n in (*NATIVE,*MASK)))}
        expected.update(k for k in state if k.startswith(tuple(n+'.' for n in MODAL)))
        if set(state)!=expected:raise ValueError('factory key/role set mismatch')
        for name in (*NATIVE,*MASK):
            if not exact_compare(getattr(factory,name).state_dict(),getattr(model,name).state_dict())['equal']:raise ValueError('factory module extraction drift')
        for tensor in state.values():
            if tensor.is_floating_point() and tensor.dtype!=torch.float32:raise ValueError('factory must be FP32')
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():raise FloatingPointError('factory nonfinite')
        factory_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(20261025);fusion=native.Tmol3MGating()
    return state,clone_tree(fusion.state_dict()),factory_rng

def pretrainer():return base.BaseOptFuse(True,pt=PT)

def validate_pt(p,step):
    if p.get('plan')!=PLAN or p.get('revision')!=REVISION or p.get('schema')!=(INITIAL_SCHEMA if step==0 else PT_SCHEMA) or p.get('step')!=step or p.get('config')!=CONFIG or p.get('historical_tensor_reads')!=0 or p.get('fp_prediction') is not False:raise ValueError('new-cycle PT identity/schema')
    expected_stream=dict(policy='disjoint_modal_v1',B=86400010,counters=[step*6]*3,forward_domain=[0,60000],terminal_counter=60000)
    if p.get('stream')!=expected_stream:raise ValueError('PT stream endpoint identity mismatch')
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in p['state_dict'].values()):raise ValueError('PT master parameter dtype must be FP32')
    allowed=(*NATIVE,*MODAL,*MASK)
    if any(not k.startswith(tuple(n+'.' for n in allowed)) for k in p['state_dict']):raise ValueError('PT parameter role outside schema')

def initialize_ft(p,h,task,fold):
    validate_pt(p,10000)
    if h.get('plan')!=PLAN or h.get('schema')!=HEAD_SCHEMA or h.get('task')!=task or h.get('fold')!=fold or h.get('seed')!=20261025+100*TASKS.index(task)+fold:raise ValueError('new head schema/seed')
    model=base.BaseOptFuse(False,ft=FT)
    wanted={k:v for k,v in p['state_dict'].items() if not k.startswith(('atom_head.','line_head.'))}
    expected={k for k in model.state_dict() if not k.startswith(('fusion.','head.'))}
    if set(wanted)!=expected:raise ValueError('explicit PT to FT key conversion')
    for name in (*NATIVE,*MODAL):
        pre=name+'.';strict_module_state(getattr(model,name),{k[len(pre):]:v for k,v in wanted.items() if k.startswith(pre)})
    strict_module_state(model.head,h['state_dict'])
    if not exact_compare(model.fusion.state_dict(),p['ft_fusion_state'])['equal']:raise ValueError('historical gate factory rule differs')
    current=model.state_dict()
    if any(not torch.equal(current[k].cpu(),v) for k,v in wanted.items()) or not exact_compare(model.head.state_dict(),h['state_dict'])['equal']:raise ValueError('shared FT parameter initialization drift')
    return model

class SWA:
    def __init__(self):self.epochs=[];self.sums={};self.dtypes={};self.buffers=None
    def collect(self,model,epoch):
        if epoch not in (40,50,60,70) or epoch in self.epochs or epoch!=(40,50,60,70)[len(self.epochs)]:raise ValueError('fixed SWA epoch/order')
        if any(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in model.modules()):raise ValueError('BatchNorm requires unbudgeted update_bn; forbidden')
        buffers=clone_tree(dict(model.named_buffers()))
        if self.buffers is None:self.buffers=buffers
        elif not exact_compare(self.buffers,buffers)['equal']:raise ValueError('SWA non-parameter buffer changed; no update_bn allowed')
        for name,p in model.named_parameters():
            value=p.detach().to(device='cpu',dtype=torch.float64)
            if not torch.isfinite(value).all():raise FloatingPointError('SWA nonfinite')
            if name not in self.sums:self.sums[name]=value.clone();self.dtypes[name]=p.dtype
            else:self.sums[name].add_(value)
        self.epochs.append(epoch)
    def state(self):
        if self.epochs!=[40,50,60,70]:raise ValueError('SWA incomplete fixed window')
        return {**{k:(v/4).to(self.dtypes[k]) for k,v in self.sums.items()},**self.buffers}
