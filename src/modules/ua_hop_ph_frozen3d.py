"""F3D_PH: identical HOP_PH forward, frozen pretrained 3D encoder only."""
import copy
import torch
from torch import nn
from .ua_hop_ph import HopModel, ARCHITECTURE as R1_ARCHITECTURE
from .ua2d_galformer_pretrain import package_attention, SCHEMA as PT_SCHEMA
from .glt_dual import strict_module_state

SCHEMA = 'ua-hop-ph-frozen3d-ft-r1'
CANDIDATE = 'F3D_PH'
POLICY = dict(encoder3d='frozen_all_stages', encoder3d_mode='eval',
    real_3d_forward=True, embedding_cache=False, joint_trainable=['o8', 'norm2', 'adapter', 'router', 'head'])
ARCHITECTURE = copy.deepcopy(R1_ARCHITECTURE)
ARCHITECTURE.update(schema=SCHEMA, candidate_id=CANDIDATE, trainability_policy=POLICY)


class Frozen3DModel(HopModel):
    def __init__(self):
        super().__init__('HOP_PH')
        self.candidate_id = CANDIDATE
        self.freeze_encoder3d()

    def freeze_encoder3d(self):
        for p in self.encoder3d.parameters(): p.requires_grad_(False)
        self.encoder3d.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_encoder3d()
        return self


def initialize(parent, head, arm=CANDIDATE):
    if (arm != CANDIDATE or parent.get('schema') != PT_SCHEMA or parent.get('step') != 5000
            or parent.get('config', {}).get('strategy') != 'mask_align'
            or parent['config'].get('fp_enabled') is not True
            or parent.get('execution', {}).get('revision') != 'r4'
            or parent.get('execution', {}).get('plan') != 'UA2D-PRETRAIN-FP-20261019-01'
            or package_attention(parent)['scale_denominator'] != 512):
        raise ValueError('not the declared B_FP step5000 parent')
    model = Frozen3DModel()
    for name in ('o8', 'encoder3d', 'norm2'):
        prefix = name+'.'
        strict_module_state(getattr(model, name), {k[len(prefix):]: v for k, v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head, head)
    return model


def restore(saved, expected_config):
    if (saved.get('schema') != SCHEMA or saved.get('architecture') != ARCHITECTURE
            or saved.get('candidate_id') != CANDIDATE or saved.get('trainability_policy') != POLICY
            or saved.get('config') != expected_config or saved.get('best_stage') not in ('head', 'joint')):
        raise ValueError('F3D complete checkpoint/config/freeze policy mismatch')
    model = Frozen3DModel()
    strict_module_state(model, saved['state_dict'])
    model.freeze_encoder3d()
    return model


def optimizer(model, stage):
    if stage not in ('head', 'joint'): raise ValueError('wrong stage')
    model.stage = stage
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith('encoder3d.') and (stage == 'joint' or name.startswith('head.')))
    model.freeze_encoder3d()
    groups = []
    if stage == 'joint':
        groups.extend([dict(name='inherited', params=[p for m in (model.o8, model.norm2) for p in m.parameters()], lr=1e-4),
            dict(name='new', params=[p for name, p in model.named_parameters() if name.startswith(('adapter.', 'router.'))], lr=1e-3)])
    groups.append(dict(name='head', params=list(model.head.parameters()), lr=3e-4 if stage == 'head' else 1e-3))
    ids = [id(p) for g in groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    forbidden = {id(p) for p in model.encoder3d.parameters()}
    if len(ids) != len(set(ids)) or set(ids) != expected or set(ids) & forbidden:
        raise ValueError('optimizer coverage/overlap/frozen encoder violation')
    evidence = [dict(name=g['name'], lr=g['lr'], tensors=len(g['params']), parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups, weight_decay=.02), evidence
