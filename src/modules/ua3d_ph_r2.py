"""PH r2 residuals; original encoder and alignment remain separate."""
import time
import numpy as np
import torch
from torch import nn
from src.dataset import ua3d_ph_r2 as d
from src.modules import ua3d_base_opt_r6 as base
from src.modules.glt_dual import mean_pool
from src.modules.glt_dual_opt import incident_pool

ARMS = ('C0_BASE', 'P_PE', 'P_FIXED', 'P_LEARN', 'M_EQ', 'M_GEOM', 'M_PH')
P_ARMS = ('P_PE', 'P_FIXED', 'P_LEARN')
M_ARMS = ('M_EQ', 'M_GEOM', 'M_PH')
CANDIDATES = ('P_FIXED', 'P_LEARN', 'M_PH')
SCHEMA = 'ua3d-ph-best-r2'
ADAPTER_SEED = 2026102801


def mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers.extend((nn.Linear(a, b), nn.GELU()))
    return nn.Sequential(*layers[:-1])


class GraphPHAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = mlp((32, 32, 16)); self.filter = mlp((16, 16, 4))
        self.h0 = mlp((4, 32, 32)); self.h1 = mlp((4, 32, 32)); self.out = mlp((80, 64, 512))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        self.reductions = 0; self.reduction_seconds = 0.

    def forward(self, *, side, arm):
        if arm not in P_ARMS:
            raise ValueError('invalid P arm')
        q = side['rwpe']; edges = side['edges']; n = len(q)
        p = q + self.pe(torch.cat((q, d.neighbor_mean(q, edges)), -1))
        learned = torch.sigmoid(self.filter(p)) if arm == 'P_LEARN' else None
        f = learned if learned is not None else q[:, [1, 3, 7, 15]]
        r0 = p.new_zeros((n, 32)); r1 = p.new_zeros((n, 32))
        for k in range(4):
            if arm == 'P_PE':
                v0, v1 = d.local_tuples(f[:, k], edges)
                r0 = r0 + self.h0(v0) / 4; r1 = r1 + self.h1(v1) / 4
            else:
                if arm == 'P_LEARN':
                    tick = time.monotonic(); self.reductions += 1
                    events = d.graph_events(f[:, k].detach().cpu().numpy(), edges)
                    self.reduction_seconds += time.monotonic() - tick
                else:
                    events = side['events'][k]
                for collection, encoder, target in ((events[0], self.h0, 0), (events[1], self.h1, 1)):
                    encoded = encoder(d.event_tuples(f[:, k], collection))
                    values = d.distribute(encoded, collection, n) / 4
                    if target == 0:
                        r0 = r0 + values
                    else:
                        r1 = r1 + values
        delta = self.out(torch.cat((p, r0, r1), -1))[side['center']]
        if not side.get('graph_available', True):
            delta = delta * 0.
        return delta, dict(pe=p.detach(), filtration=f.detach(), r0=r0.detach(), r1=r1.detach(),
                           learnable=learned is not None and learned.requires_grad)


class SpatialRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32); self.geometry = mlp((16, 32, 32))
        self.router = mlp((155, 64, 3)); self.up = nn.Linear(32, 256)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, *, states, side, arm):
        if arm not in M_ARMS or not side['available']:
            raise ValueError('invalid spatial branch input')
        ends = side['bond_atoms']; count = side['physical_atoms']
        if ends.shape != (2, len(states)):
            raise ValueError('physical bond state count mismatch')
        atoms = mean_pool(torch.cat((states, states), 0), ends.reshape(-1), count)
        hidden = self.down(atoms); messages = []
        for edges, distance in zip(side['sp_edges'], side['sp_distances']):
            if not edges.numel():
                messages.append(hidden.new_zeros((count, 32))); continue
            centers = torch.linspace(0., 4.5, 16, device=hidden.device)
            rbf = torch.exp(-.5 * ((distance[:, None] - centers) / (4.5 / 15)) ** 2)
            values = torch.sigmoid(self.geometry(rbf)) * hidden[edges[0]]
            atom_message = mean_pool(values, edges[1], count)
            bond = (atom_message[ends[0]] + atom_message[ends[1]]) / 2
            graph, valid = incident_pool(bond, side['incidence'], side['canonical_graph'], 1)
            if not valid.all():
                raise ValueError('physical incidence availability')
            messages.append(graph[0])
        # Empty scales must have graph, rather than atom, shape.
        messages = [x.new_zeros(32) if x.ndim == 2 else x for x in messages]
        if arm == 'M_EQ':
            alpha = hidden.new_full((3,), 1 / 3)
        else:
            x = side['router']
            if arm == 'M_GEOM':
                x = torch.cat((x[:, :3], torch.zeros_like(x[:, 3:])), -1)
            alpha = torch.softmax(self.router(x.flatten()), -1)
        delta = self.up((alpha[:, None] * torch.stack(messages)).sum(0))
        return delta, dict(alpha=alpha.detach(), router=side['router'].detach())


def attach_adapter(model, arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    model.__class__ = PHFuse; model.ph_arm = arm; model.last_ph = {}
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(ADAPTER_SEED)
        if arm in P_ARMS:
            model.graph_ph = GraphPHAdapter()
            if sum(x.numel() for x in model.graph_ph.parameters()) != 42820:
                raise ValueError('P parameter count')
        elif arm in M_ARMS:
            model.spatial_ph = SpatialRouter()
            if sum(x.numel() for x in model.spatial_ph.parameters()) != 36643:
                raise ValueError('M parameter count')
    return model


class PHFuse(base.BaseOptFuse):
    def __init__(self, arm, ft):
        super().__init__(False, ft=ft)
        attach_adapter(self, arm)

    def modified_modalities(self, batch, original, atoms, bonds):
        z = list(original); self.last_ph = {}
        if self.ph_arm in P_ARMS:
            increments = []; infos = []
            for side in batch.sides:
                delta, info = self.graph_ph(side=side, arm=self.ph_arm)
                increments.append(delta); infos.append(info)
            h = atoms + torch.cat(increments, 0)
            if len(h) != len(batch.two_d.canonical_graph_index):
                raise ValueError('center/canonical atom order')
            z[0] = self.A2(self.norm2(mean_pool(h, batch.two_d.canonical_graph_index, len(z[0]))))
            self.last_ph = dict(graphs=infos)
        elif self.ph_arm in M_ARMS:
            delta = torch.zeros_like(z[1]); infos = []
            if batch.three_d is not None:
                for local, row in enumerate(batch.three_d_rows.detach().cpu().tolist()):
                    side = batch.sides[row]
                    value, info = self.spatial_ph(states=bonds[batch.three_d.bond_batch == local], side=side, arm=self.ph_arm)
                    # Unique graph rows avoid any accumulating/index_add operation.
                    delta[row] = value; infos.append(info)
            z[1] = (z[1] + delta) * batch.availability[:, 1:2].to(z[1])
            self.last_ph = dict(graphs=infos)
        if torch.count_nonzero(z[1][~batch.availability[:, 1]]):
            raise ValueError('missing3D projected values must be zero')
        return z

    def predict(self, batch):
        original, atoms, bonds = self.encode_modalities(batch)
        modified = self.modified_modalities(batch, original, atoms, bonds)
        self.alignment_original = original
        fused, weights = self.fuse(modified, batch.availability)
        prediction = self.head(fused).view(-1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('prediction')
        return prediction, weights, modified

    def training_loss(self, batch):
        prediction, weights, _ = self.predict(batch)
        mse = nn.functional.mse_loss(prediction, batch.y.view(-1))
        sums, counts = base.alignment_terms(self.contrastive_features(self.alignment_original), batch.availability, batch.identity)
        align = sum(v / counts[k].clamp_min(1) for k, v in sums.items())
        entropy = base.native.entropy(weights)
        return mse + .08 * align - .08 * entropy, dict(mse=mse.detach(), align=align.detach(), entropy=entropy.detach())
