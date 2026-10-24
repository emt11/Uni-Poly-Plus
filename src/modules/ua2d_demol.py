"""One 128-wide atom/bond adapter following the six-layer O8 encoder."""
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel, mean_pool, strict_module_state
from src.modules.ua2d_galformer_pretrain import package_attention, SCHEMA as PT_SCHEMA
from src.dataset.ua2d_demol import INPUT_SCHEMA, SEMANTICS
SCHEMA = 'ua2d-demol-complete-ft-r1'
ARCHITECTURE = dict(schema=SCHEMA, input=SEMANTICS, width=128, layers=1,
    atom_input='endpoint_sum_absdiff_chem14', bond_update='residual_mlp_self_and_mean_incoming',
    zero_output=True, readout='original_mean_norm2', attention_denominator=512)
ARMS = ('BOND_SELF', 'BOND_NEIGHBOR')

class BondAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(nn.Linear(1038, 128), nn.GELU())
        self.norm = nn.LayerNorm(128)
        self.update = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128))
        self.output = nn.Linear(128, 512)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, atoms, data, arm):
        a, b = data.demol_keys[:, :2].unbind(1)
        h = self.initial(torch.cat((atoms[a]+atoms[b], (atoms[a]-atoms[b]).abs(), data.demol_chem), -1))
        normalized = self.norm(h)
        if arm == 'BOND_SELF': incoming = normalized
        elif arm == 'BOND_NEIGHBOR':
            source, target = data.demol_line
            incoming = mean_pool(normalized[source], target, len(h))
        else: raise ValueError('unknown arm')
        h = h+self.update(torch.cat((normalized, incoming), -1))
        bond, atom = data.demol_incidence
        pooled = mean_pool(h[bond], atom, len(atoms))
        valid = torch.bincount(atom, minlength=len(atoms)) > 0
        return torch.where(valid[:, None], self.output(pooled), torch.zeros_like(atoms))

class BondModel(Pure2DGLTModel):
    def __init__(self, arm):
        if arm not in ARMS: raise ValueError('unknown arm')
        super().__init__(dropout=0., attention_scale_denominator=512)
        self.head[2].p = .1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261020)
            self.adapter = BondAdapter()
        self.arm = arm; self.stage = 'joint'
    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == 'head':
            self.o8.eval(); self.norm2.eval(); self.adapter.eval()
        return self
    def features(self, data):
        if data.demol_schema != INPUT_SCHEMA: raise ValueError('wrong input schema')
        atoms, _ = self.o8(data)
        atoms = atoms+self.adapter(atoms, data, self.arm)
        return self.norm2(mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel()))
    def forward(self, data): return self.head(self.features(data)).view(-1)
    def forward_topology(self, topology, smiles):
        from src.dataset.ua2d_demol import build, collate
        return self(collate([build(topology, smiles)]).to(next(self.parameters()).device))


def initialize(parent, head, arm):
    if (parent.get('schema') != PT_SCHEMA or parent.get('step') != 5000
        or parent.get('config', {}).get('strategy') != 'mask_align'
        or parent['config'].get('fp_enabled') is not True
        or parent.get('execution', {}).get('revision') != 'r4'
        or parent.get('execution', {}).get('plan') != 'UA2D-PRETRAIN-FP-20261019-01'
        or package_attention(parent)['scale_denominator'] != 512):
        raise ValueError('not declared B_FP parent')
    model = BondModel(arm)
    for name in ('o8', 'norm2'):
        prefix = name+'.'
        strict_module_state(getattr(model, name), {k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head, head)
    return model


def restore(saved, expected_config):
    if (saved.get('schema') != SCHEMA or saved.get('architecture') != ARCHITECTURE
        or saved.get('config') != expected_config or saved.get('best_stage') not in ('head', 'joint')):
        raise ValueError('complete FT schema/config mismatch')
    model = BondModel(expected_config['arm'])
    strict_module_state(model, saved['state_dict'])
    return model


def optimizer(model, stage):
    if stage not in ('head', 'joint'): raise ValueError('wrong stage')
    model.stage = stage
    for name, p in model.named_parameters(): p.requires_grad_(stage == 'joint' or name.startswith('head.'))
    groups = []
    if stage == 'joint':
        groups.extend([dict(name='inherited', params=[p for m in (model.o8, model.norm2) for p in m.parameters()], lr=1e-4),
            dict(name='adapter', params=list(model.adapter.parameters()), lr=3e-4)])
    groups.append(dict(name='head', params=list(model.head.parameters()), lr=3e-4 if stage == 'head' else 1e-3))
    ids = [id(p) for g in groups for p in g['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('optimizer omits or duplicates parameters')
    evidence = [dict(name=g['name'], lr=g['lr'], tensors=len(g['params']), parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups, weight_decay=.02), evidence
