"""Isolated FCE r2 recipe wiring; original epoch/scaler/cache semantics."""
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import train_epoch, evaluate
from scripts.finetune_mcl_ph import (periodic_tdl_head, PERIODIC_TDL_WEIGHT_DECAY,
    PERIODIC_TDL_HEAD_DROPOUT, PERIODIC_TDL_EPOCHS)
ARMS=('R0','H_LR','B_LR','S_COS')
TASKS=('eea','egb','egc','ei','eps','nc','xc')
RUNTIME='fce_param_r2_cache_global_loop_v1'
T1_ARMS=('T1_R0','T1_D','T1_W','T1_DW')
T1_RUNTIME='fce_tune_t1_r2_v1'
T2_ARMS=('T2_R0','T2_G_LR')
T2_RUNTIME='fce_tune_t2_r3_glt_groups_v1'

def t2_recipe_for(arm,task):
    if arm not in T2_ARMS or task not in TASKS:raise ValueError('undefined T2 arm/task')
    recipe=recipe_for('R0',task)
    recipe.update(revision=T2_RUNTIME,arm=arm,
        joint_glt_lr=recipe['joint_backbone_lr']*(0.3 if arm=='T2_G_LR' else 1),
        joint_parameter_groups=['glt','backbone_other','head'])
    return recipe

def t1_recipe_for(arm,task):
    if arm not in T1_ARMS or task not in TASKS:raise ValueError('undefined T1 arm/task')
    recipe=recipe_for('R0',task)
    recipe.update(revision=T1_RUNTIME,arm=arm,
        head_dropout=recipe['head_dropout']+(0.10 if arm in ('T1_D','T1_DW') else 0),
        head_weight_decay=recipe['weight_decay'],
        joint_backbone_weight_decay=recipe['weight_decay']+(0.02 if arm in ('T1_W','T1_DW') else 0))
    return recipe

def recipe_for(arm,task):
    if arm not in ARMS or task not in TASKS:raise ValueError('undefined arm/task')
    return dict(revision=RUNTIME,arm=arm,task=task,head_epochs=10,
        joint_epochs=50 if task=='egc' else 60,head_lr=3e-4,
        joint_head_lr=3e-4 if arm=='H_LR' else 1e-3,
        joint_backbone_lr=(2e-4 if task=='xc' else 1e-4)*(0.3 if arm=='B_LR' else 1),
        joint_scheduler='CosineAnnealingLR' if arm=='S_COS' else 'CosineAnnealingWarmRestarts',
        head_scheduler=dict(T_max=10,eta_min=3e-5),
        joint_scheduler_options=dict(T_max=50 if task=='egc' else 60,eta_min=1e-6) if arm=='S_COS' else dict(T_0=10,T_mult=1,eta_min=1e-6),
        weight_decay=PERIODIC_TDL_WEIGHT_DECAY[task],head_dropout=PERIODIC_TDL_HEAD_DROPOUT[task],
        encoder_dropout=0.0,grad_clip=5.0,batch=24,precision='fp32',workers=0)

def validate_recipe(recipe,task):
    factory=t2_recipe_for if recipe.get('arm') in T2_ARMS else (t1_recipe_for if recipe.get('arm') in T1_ARMS else recipe_for)
    expected=factory(recipe.get('arm'),task)
    if recipe!=expected:raise ValueError('recipe mismatch')
    return recipe

NORM_ADAPTATION_POLICY = dict(name='encoder_layernorm_affine_and_head', revision='r14',
    selection='actual nn.LayerNorm direct weight/bias by parameter identity; exclude head',
    head_stage='original head only', joint_stage='encoder LayerNorm affine plus original head')


def validate_adaptation(policy, *, ft_candidate=None, profile=None, backbone_lr=None, recipe=None, smoke=False):
    if policy is None:return
    if policy != NORM_ADAPTATION_POLICY:raise ValueError('unknown adaptation policy')
    if ft_candidate is not None or profile is not None or backbone_lr is not None or smoke:
        raise ValueError('adaptation policy cannot mix overrides')
    if recipe is not None and recipe.get('arm')!='R0':raise ValueError('adaptation requires original R0')


def norm_parameter_partition(model):
    head_ids={id(p) for p in periodic_tdl_head(model).parameters()}
    encoder_ids={id(p) for p in model.model.parameters()}-head_ids
    ln_ids={id(p) for module in model.model.modules() if isinstance(module,nn.LayerNorm)
        for name,p in module.named_parameters(recurse=False) if name in ('weight','bias')}-head_ids
    if not ln_ids or not ln_ids<=encoder_ids:raise ValueError('invalid encoder LayerNorm identity partition')
    named=dict(model.named_parameters())
    def group(ids):
        return dict(parameter_names=[n for n,p in named.items() if id(p) in ids],
            num_tensors=sum(id(p) in ids for p in named.values()),
            num_parameters=sum(p.numel() for p in named.values() if id(p) in ids))
    return dict(total=sum(p.numel() for p in named.values()),head=group(head_ids),
        encoder_layernorm=group(ln_ids),frozen_encoder=group(encoder_ids-ln_ids),
        joint=group(head_ids|ln_ids))


def periodic_tdl_trainability(model, *, head_only, ft_candidate=None, adaptation_policy=None):
    validate_adaptation(adaptation_policy,ft_candidate=ft_candidate)
    if adaptation_policy is not None:
        partition=norm_parameter_partition(model)
        allowed=set(partition["head" if head_only else "joint"]["parameter_names"])
        allowed_ids={id(p) for name,p in model.named_parameters() if name in allowed}
        for p in model.parameters():p.requires_grad_(False)
        for p in model.parameters():
            if id(p) in allowed_ids:p.requires_grad_(True)
        return
    if ft_candidate not in (None, 'FT_PARTIAL', 'FT_LOW_LR', 'FT_REG'):
        raise ValueError('unknown FT generalization candidate')
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    allowed = head_ids
    if not head_only and ft_candidate == 'FT_PARTIAL':
        encoder = model.model
        modules = [*encoder.o8.layers[-2:], *encoder.glt.layers[-2:],
                   encoder.norm2, encoder.norm3]
        allowed = head_ids | {id(p) for module in modules for p in module.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_((not head_only and ft_candidate != 'FT_PARTIAL')
                                 or id(parameter) in allowed)


def periodic_tdl_optimizer(model, task, *, head_only, backbone_lr=None, ft_candidate=None, profile=None, recipe=None, adaptation_policy=None):
    """Paper learning rates and per-task weight decay on all trainable tensors."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    if recipe is not None:
        validate_recipe(recipe, task)
        if profile is not None or ft_candidate is not None or backbone_lr is not None:
            raise ValueError('explicit recipe cannot mix overrides')
    validate_adaptation(adaptation_policy,ft_candidate=ft_candidate,profile=profile,backbone_lr=backbone_lr,recipe=recipe)
    if adaptation_policy is not None:
        expected=set(norm_parameter_partition(model)["head" if head_only else "joint"]["parameter_names"])
        if {n for n,p in model.named_parameters() if p.requires_grad}!=expected:raise ValueError("adaptation flags must precede optimizer")
    decay = 0.02 if profile else PERIODIC_TDL_WEIGHT_DECAY[task]
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    groups = []
    if not head_only:
        backbone = [parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in head_ids]
        lr = 1e-4 if profile else (backbone_lr if backbone_lr is not None else (2e-4 if task == 'xc' else 1e-4))
        if recipe is not None:lr = recipe['joint_backbone_lr']
        if ft_candidate == 'FT_LOW_LR':
            lr *= 0.3
        if recipe is not None and recipe['arm'] in T2_ARMS:
            glt_ids={id(p) for p in model.model.glt.parameters()}
            if glt_ids & head_ids:raise ValueError('glt/head overlap')
            glt=[p for p in backbone if id(p) in glt_ids]
            other=[p for p in backbone if id(p) not in glt_ids]
            if not glt or not other or {id(p) for p in glt}!=glt_ids:
                raise ValueError('incomplete glt/backbone_other parameter groups')
            groups.extend([dict(params=glt,lr=recipe['joint_glt_lr'],weight_decay=decay,name='glt'),
                           dict(params=other,lr=lr,weight_decay=decay,name='backbone_other')])
        else:
            groups.append({'params': backbone, 'lr': lr,
                           'weight_decay': recipe.get('joint_backbone_weight_decay',decay) if recipe is not None else decay + (0.01 if ft_candidate == 'FT_REG' else 0),
                           'name': 'backbone'})
    head = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in head_ids]
    groups.append({'params': head, 'lr': 3e-4 if head_only else (recipe['joint_head_lr'] if recipe is not None else 1e-3),
                   'weight_decay': recipe.get('head_weight_decay',decay) if recipe is not None else decay, 'name': 'head'})
    if adaptation_policy is not None or (recipe is not None and recipe['arm'] in T2_ARMS):
        ids=[id(p) for g in groups for p in g['params']]
        if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('optimizer parameter omission/duplication')
    evidence = [{'name': group['name'], 'lr': group['lr'],
                 'weight_decay': group['weight_decay'], 'num_tensors': len(group['params']),
                 'num_parameters': sum(p.numel() for p in group['params'])}
                for group in groups]
    return torch.optim.AdamW(groups), evidence


def configure_periodic_tdl_dropout(model, task, *, profile=None, recipe=None):
    """Use paper task-specific head dropout; disable encoder module dropout."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown dropout profile')
    if recipe is not None:
        validate_recipe(recipe,task)
        if profile is not None:raise ValueError('explicit recipe cannot mix profile')
    head = periodic_tdl_head(model)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for module in head.modules():
        if isinstance(module, nn.Dropout):
            module.p = recipe['head_dropout'] if recipe is not None else (0.10 if profile else PERIODIC_TDL_HEAD_DROPOUT[task])


def run_periodic_tdl_epochs(model, train_loader, validation_loader, criterion,
                            device, *, task, scaler, progress=None, smoke=False, backbone_lr=None,
                            ft_candidate=None, profile=None, step_observer=None, recipe=None, adaptation_policy=None):
    """Ten frozen-head epochs, then 50/60 joint epochs; select minimum raw RMSE."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (smoke or ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    if recipe is not None:
        validate_recipe(recipe, task)
        if smoke or profile is not None or ft_candidate is not None or backbone_lr is not None:
            raise ValueError('explicit recipe cannot mix overrides')
    validate_adaptation(adaptation_policy,ft_candidate=ft_candidate,profile=profile,backbone_lr=backbone_lr,recipe=recipe,smoke=smoke)
    class HeadOnlyAdapter(AdaptationAdapter):
        def forward(self,data):
            if hasattr(data,'cached_features'):
                return periodic_tdl_head(model)(data.cached_features),None
            return super().forward(data)
        def train(self, mode=True):
            super().train(mode)
            if mode:
                model.eval()
                periodic_tdl_head(model).train()
            return self

    class EpochSchedulerProxy:
        def step(self):
            pass

    best_state, best_epoch, best_rmse, best_r2 = None, -1, float('inf'), None
    best_predictions = None
    history, updates, group_evidence = [], 0, {}
    stages = (('head', 1), ('joint', 1)) if smoke else (
        ('head', 10), ('joint', 60 if profile else PERIODIC_TDL_EPOCHS.get(task, 60)))
    online_train,online_validation=train_loader,validation_loader
    feature_cache=None
    for stage, stage_epochs in stages:
        if stage=='joint' and feature_cache is not None:
            feature_cache.invalidate()
            train_loader,validation_loader=online_train,online_validation
        wrapped = HeadOnlyAdapter(model) if stage == 'head' else AdaptationAdapter(model)
        periodic_tdl_trainability(model, head_only=(stage == 'head'), ft_candidate=ft_candidate, adaptation_policy=adaptation_policy)
        context=getattr(model,'feature_cache_context',None)
        if stage=='head' and context is not None:
            from src.modules.glt_adaptation import HeadFeatureCache
            from torch.utils.data import Subset
            model.eval()
            dataset=online_train.dataset.dataset if isinstance(online_train.dataset,Subset) else online_train.dataset
            feature_cache=HeadFeatureCache(model,dataset,context)
            feature_cache.build(device,online_train.batch_size,online_train.collate_fn)
            train_loader=feature_cache.loader(online_train)
            validation_loader=feature_cache.loader(online_validation)
        optimizer, group_evidence[stage] = periodic_tdl_optimizer(
            model, task, head_only=(stage == 'head'), backbone_lr=backbone_lr,
            ft_candidate=ft_candidate, profile=profile, recipe=recipe, adaptation_policy=adaptation_policy)
        if step_observer is not None:
            optimizer.register_step_pre_hook(lambda opt,args,kwargs:step_observer('attempted'))
            optimizer.register_step_post_hook(lambda opt,args,kwargs:step_observer('completed'))
        if adaptation_policy is not None or ft_candidate or (recipe is not None and recipe['arm'] in T1_ARMS+T2_ARMS):
            named = dict(model.named_parameters())
            for group in group_evidence[stage]:
                members = {id(p) for item in optimizer.param_groups
                           if item['name'] == group['name'] for p in item['params']}
                group['parameter_names'] = [name for name, p in named.items() if id(p) in members]
        scheduler = scheduler_for(optimizer,stage,stage_epochs,recipe)
        for stage_epoch in range(1, stage_epochs + 1):
            used_learning_rates=[g['lr'] for g in optimizer.param_groups]
            result = train_epoch(wrapped, train_loader, criterion, optimizer,
                                 EpochSchedulerProxy(),
                                 device, epoch=len(history) + 1, amp_dtype='fp32',
                                 max_grad_norm=5.0, fail_nonfinite=True,
                                 return_timing=True)
            scheduler.step()
            updates += int(result[-1]['training_steps'])
            validation_loss, r2, targets, predictions = evaluate(
                wrapped, validation_loader, criterion, device, scaler=scaler,
                amp_dtype='fp32')
            targets = np.asarray(targets, dtype=np.float64).reshape(-1)
            predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
            rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
            if not all(np.isfinite([validation_loss, r2, rmse])):
                raise FloatingPointError('nonfinite validation metrics')
            record = {'epoch': len(history) + 1, 'stage': stage,
                      'stage_epoch': stage_epoch, 'train_loss': float(result[0]),
                      'validation_loss': float(validation_loss),
                      'validation_r2': float(r2), 'validation_rmse': rmse,
                      'learning_rates': [group['lr'] for group in optimizer.param_groups],
                      'training_steps': int(result[-1]['training_steps'])}
            if profile:
                record.update(ft_profile=profile,optimizer_groups=group_evidence[stage],encoder_dropout=0.0,head_dropout=0.1,grad_clip=5.0,scheduler=dict(name=type(scheduler).__name__,T_max=10,eta_min=3e-5) if stage=='head' else dict(name=type(scheduler).__name__,T_0=10,T_mult=1,eta_min=1e-6))
            record.update(used_learning_rates=used_learning_rates,optimizer_groups=group_evidence[stage],scheduler_name=type(scheduler).__name__)
            if recipe is not None:record['recipe']=recipe
            history.append(record)
            if progress is not None:
                progress(record)
            if rmse < best_rmse:
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                best_predictions = (targets.copy(), predictions.copy())
                best_rmse, best_r2, best_epoch = rmse, float(r2), len(history)
    if best_state is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    return {'history': history, 'state': best_state, 'best_r2': best_r2,
            'best_rmse': best_rmse, 'best_epoch': best_epoch,
            'optimizer_updates': updates, 'predictions': best_predictions,
            'optimizer_groups_by_stage': group_evidence}



def scheduler_for(optimizer,stage,stage_epochs,recipe=None):
    if stage=='head':return CosineAnnealingLR(optimizer,T_max=10,eta_min=3e-5)
    if recipe is not None and recipe['joint_scheduler']=='CosineAnnealingLR':
        return CosineAnnealingLR(optimizer,T_max=stage_epochs,eta_min=1e-6)
    return CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-6)


# Shared completion stage for all tasks after the neural validation-best is fixed.
# Existing epoch/optimizer semantics above remain the matched neural reference.
from src.training.fce_kernel import (POLICY as FINETUNE_PREDICTION_POLICY,
    fit_branch as fit_finetune_predictions,
    predict_branch as predict_finetune_predictions)

# r10 array-only continuation after all r9 model/alpha choices are fixed.
from src.training.fce_kernel import (GLOBAL_BLEND_POLICY as GLOBAL_FINETUNE_BLEND_POLICY,
    select_global_weight as select_global_finetune_weight,
    blend_frozen_predictions as blend_finetune_arrays)

# r11 retains the historical feature metadata while aligning the actual policy.
from src.training.fce_kernel import (ALIGNED_ALPHA_POLICY as ALIGNED_FINETUNE_POLICY,
    select_aligned_alpha as select_aligned_finetune_alpha,
    restore_aligned_model as restore_aligned_finetune_model,
    predict_aligned_branch as predict_aligned_finetune_predictions)

# r12: explicit radius3 representation, inherited alpha, fixed global .75.
from src.training.fce_kernel import (RADIUS3_POLICY as RADIUS3_FINETUNE_POLICY,
    fit_radius3_branch as fit_radius3_finetune_predictions,
    predict_radius3_branch as predict_radius3_finetune_predictions)


# r13: an explicit opt-in loss. Existing callers keep their MSE criterion.
HUBER_FT_LOSS_POLICY=dict(name='2xHuber',delta=1.0,multiplier=2.0,reduction='mean',
    target_space='existing train-only standardized targets',stages=['head','joint'],
    best_selection='original-unit neural validation RMSE; strict improvement')


class FixedTwiceHuberLoss(nn.Module):
    def __init__(self):
        super().__init__();self.huber=nn.HuberLoss(delta=1.0,reduction='mean')
    def forward(self,prediction,target):
        if prediction.shape!=target.shape or prediction.numel()==0:
            raise ValueError('Huber prediction/target shapes must match without broadcasting')
        if prediction.ndim not in (1,2) or (prediction.ndim==2 and prediction.shape[1]!=1):
            raise ValueError('Huber expects one standardized scalar per sample')
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite Huber input')
        return 2.0*self.huber(prediction,target)


def finetune_criterion(loss_policy=None):
    if loss_policy is None:return nn.MSELoss()
    if loss_policy!=HUBER_FT_LOSS_POLICY:raise ValueError('undefined FT loss policy')
    return FixedTwiceHuberLoss()
