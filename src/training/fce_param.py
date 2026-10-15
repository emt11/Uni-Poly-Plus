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
    if recipe!=recipe_for(recipe.get('arm'),task):raise ValueError('recipe mismatch')
    return recipe

def periodic_tdl_trainability(model, *, head_only, ft_candidate=None):
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


def periodic_tdl_optimizer(model, task, *, head_only, backbone_lr=None, ft_candidate=None, profile=None, recipe=None):
    """Paper learning rates and per-task weight decay on all trainable tensors."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    if recipe is not None:
        validate_recipe(recipe, task)
        if profile is not None or ft_candidate is not None or backbone_lr is not None:
            raise ValueError('explicit recipe cannot mix overrides')
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
        groups.append({'params': backbone, 'lr': lr,
                       'weight_decay': decay + (0.01 if ft_candidate == 'FT_REG' else 0),
                       'name': 'backbone'})
    head = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in head_ids]
    groups.append({'params': head, 'lr': 3e-4 if head_only else (recipe['joint_head_lr'] if recipe is not None else 1e-3),
                   'weight_decay': decay, 'name': 'head'})
    evidence = [{'name': group['name'], 'lr': group['lr'],
                 'weight_decay': group['weight_decay'], 'num_tensors': len(group['params']),
                 'num_parameters': sum(p.numel() for p in group['params'])}
                for group in groups]
    return torch.optim.AdamW(groups), evidence


def configure_periodic_tdl_dropout(model, task, *, profile=None):
    """Use paper task-specific head dropout; disable encoder module dropout."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown dropout profile')
    head = periodic_tdl_head(model)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for module in head.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.10 if profile else PERIODIC_TDL_HEAD_DROPOUT[task]


def run_periodic_tdl_epochs(model, train_loader, validation_loader, criterion,
                            device, *, task, scaler, progress=None, smoke=False, backbone_lr=None,
                            ft_candidate=None, profile=None, step_observer=None, recipe=None):
    """Ten frozen-head epochs, then 50/60 joint epochs; select minimum raw RMSE."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (smoke or ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    if recipe is not None:
        validate_recipe(recipe, task)
        if smoke or profile is not None or ft_candidate is not None or backbone_lr is not None:
            raise ValueError('explicit recipe cannot mix overrides')
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
        periodic_tdl_trainability(model, head_only=(stage == 'head'), ft_candidate=ft_candidate)
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
            ft_candidate=ft_candidate, profile=profile, recipe=recipe)
        if step_observer is not None:
            optimizer.register_step_pre_hook(lambda opt,args,kwargs:step_observer('attempted'))
            optimizer.register_step_post_hook(lambda opt,args,kwargs:step_observer('completed'))
        if ft_candidate:
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
