"""Two prespecified FT policy factors; original model and scientific functions stay read only."""
import math
import torch
from torch import nn
from src.modules import ua3d_base_opt_r6 as base
from src.modules.ua3d_base_opt_r6 import (CONFIG,DEFAULT_FT,NATIVE,TASKS,native,pt_config,validate_package,strict_module_state)

POLICIES={
    'C0_BASE':dict(loss_schedule='constant',lr_policy='original'),
    'A_ALIGN_DECAY':dict(loss_schedule='joint_late_decay',lr_policy='original'),
    'B_MODAL_LR':dict(loss_schedule='constant',lr_policy='modal_low'),
    'AB_DECAY_LR':dict(loss_schedule='joint_late_decay',lr_policy='modal_low'),
}
MODAL=('A2','A3','fp_encoder','Q2','Q3')
TASK_MODULES=('fusion','head')

def align_coefficient(schedule,epoch):
    if schedule not in ('constant','joint_late_decay') or type(epoch) is not int or not 1<=epoch<=70:
        raise ValueError('invalid registered schedule/epoch')
    if schedule=='constant' or epoch<=40:return .08
    return .08-.06*(epoch-40)/30

class BalanceFuse(base.BaseOptFuse):
    def __init__(self,ft,loss_schedule,lr_policy):
        super().__init__(False,ft=ft)
        if ft!=DEFAULT_FT or (loss_schedule,lr_policy) not in {(p['loss_schedule'],p['lr_policy']) for p in POLICIES.values()}:
            raise ValueError('outside fixed scientific policy matrix')
        self.loss_schedule=loss_schedule;self.lr_policy=lr_policy;self.global_epoch=1;self.current_align=.08
    def set_epoch(self,epoch):
        self.global_epoch=epoch;self.current_align=align_coefficient(self.loss_schedule,epoch)
    def training_loss(self,batch):
        if self.current_align==.08:return super().training_loss(batch)
        pred,weights,z=self.predict(batch);mse=nn.functional.mse_loss(pred,batch.y.view(-1))
        sums,counts=base.alignment_terms(self.contrastive_features(z),batch.availability,batch.identity)
        align=sum(v/counts[k].clamp_min(1) for k,v in sums.items());h=native.entropy(weights)
        loss=mse+self.current_align*align-self.ft['entropy']*h
        return loss,dict(mse=mse.detach(),align=align.detach(),entropy=h.detach())
    def runtime_contract(self):
        return dict(**super().runtime_contract(),loss_schedule=self.loss_schedule,lr_policy=self.lr_policy,
            global_epoch=self.global_epoch,align_coefficient=self.current_align,entropy_coefficient=self.ft['entropy'])

def initialize_ft(package,head,recipe):
    if recipe['pt_id']!='F_PD_N1_F0':raise ValueError('source must be original F PT')
    base.validate_package(package,recipe['pt'])
    if package.get('pt_id')!=recipe['pt_id']:raise ValueError('PT endpoint identity mismatch')
    model=BalanceFuse(recipe['ft'],recipe['loss_schedule'],recipe['lr_policy'])
    base.ft_state(package,model);strict_module_state(model.head,head)
    return model

def group_kind(name):
    module=name.split('.',1)[0]
    if module in NATIVE:return 'native'
    if module in MODAL:return 'pretrained_modal'
    if module in TASK_MODULES:return 'fusion_head'
    raise ValueError('unregistered FT parameter '+name)

def optimizer_snapshot(model,opt):
    names={id(p):n for n,p in model.named_parameters()};ids=[id(p) for g in opt.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('optimizer parameter overlap/coverage mismatch')
    if model.encoder3d.mask_token.requires_grad:raise ValueError('mask_token must remain frozen')
    if any(g['weight_decay']!=.02 or g['betas']!=(.9,.999) or g['eps']!=1e-8 for g in opt.param_groups):raise ValueError('optimizer defaults changed')
    expected=1 if model.stage=='head' else 3 if model.lr_policy=='modal_low' else 2
    if len(opt.param_groups)!=expected:raise ValueError('optimizer group count mismatch')
    result=[]
    for g in opt.param_groups:
        ns=[names[id(p)] for p in g['params']]
        if model.stage=='head':
            if any(group_kind(n)=='native' for n in ns):raise ValueError('native training in head stage')
        elif model.lr_policy=='modal_low':
            if any(group_kind(n)!=g['name'] for n in ns):raise ValueError('modal LR group assignment mismatch')
        elif g['name']=='native':
            if any(group_kind(n)!='native' for n in ns):raise ValueError('original native group mismatch')
        elif any(group_kind(n)=='native' for n in ns):raise ValueError('original nonnative group mismatch')
        result.append(dict(name=g['name'],lr=g['lr'],initial_lr=g.get('initial_lr',g['lr']),parameters=sum(p.numel() for p in g['params']),parameter_names=ns,weight_decay=g['weight_decay'],betas=list(g['betas']),eps=g['eps']))
    return result

def optimizer_ft(model,stage):
    if stage=='head' or model.lr_policy=='original':
        opt,_=base.optimizer_ft(model,stage)
    elif stage=='joint' and model.lr_policy=='modal_low':
        model.stage=stage
        for n,p in model.named_parameters():p.requires_grad_(n!='encoder3d.mask_token')
        groups=[]
        for kind,lr in (('native',1e-4),('pretrained_modal',3e-4),('fusion_head',1e-3)):
            groups.append(dict(name=kind,params=[p for n,p in model.named_parameters() if p.requires_grad and group_kind(n)==kind],lr=lr))
        opt=torch.optim.AdamW(groups,weight_decay=.02)
    else:raise ValueError('invalid FT stage/policy')
    return opt,optimizer_snapshot(model,opt)

def expected_learning_rates(stage,stage_epoch,policy):
    if stage not in ('head','joint') or policy not in ('original','modal_low'):raise ValueError('invalid LR strategy')
    if stage=='head':bases=[3e-4];eta=3e-5;fraction=stage_epoch/10
    else:bases=[1e-4,3e-4,1e-3] if policy=='modal_low' else [1e-4,1e-3];eta=1e-6;fraction=(stage_epoch%10)/10
    return [eta+(lr-eta)*(1+math.cos(math.pi*fraction))/2 for lr in bases]
