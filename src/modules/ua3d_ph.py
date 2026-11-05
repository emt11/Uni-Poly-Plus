"""Fixed PH side branches around the current BASE model.

A changes only the 2D readout input. B changes only the 3D graph feature used
for supervised prediction. Alignment always receives the original features.
"""
import math
import torch
from torch import nn
from src.modules.glt_dual import mean_pool
from src.modules.glt_dual_opt import incident_pool
from src.modules import ua3d_base_opt_r6 as base

ARMS = ("C0_BASE", "P_PE", "P_FIXED", "P_LEARN", "M_EQ", "M_GEOM", "M_PH")
P_ARMS = ("P_PE", "P_FIXED", "P_LEARN")
M_ARMS = ("M_EQ", "M_GEOM", "M_PH")
PH_CANDIDATES = ("P_FIXED", "P_LEARN", "M_PH")
ADAPTER_SEED = 2026102801
SCHEMA = "ua3d-ph-best-r1"


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers.pop()
    return nn.Sequential(*layers)


class GraphPHAdapter(nn.Module):
    """One readout-time residual. Parameter count is identical for all P arms."""
    def __init__(self):
        super().__init__()
        self.pe = _mlp((32, 32, 16))
        self.filter = _mlp((16, 16, 4))
        self.h0 = _mlp((4, 32, 32))
        self.h1 = _mlp((4, 32, 32))
        self.out = _mlp((80, 64, 512))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, rwpe, neighbor, events, center, active):
        pe = rwpe + self.pe(torch.cat((rwpe, neighbor), -1))
        learned = torch.sigmoid(self.filter(pe))
        # events are encoded before averaging. P_LEARN supplies values gathered
        # from this learned tensor; fixed arms pass their precomputed tuples.
        encoded = []
        for values, encoder in zip(events, (self.h0, self.h1)):
            if values.shape[-1] != 4:
                raise ValueError("PH tuple width mismatch")
            encoded.append(encoder(values).mean(1))
        delta = self.out(torch.cat((pe, *encoded), -1))
        delta = delta * active[:, None].to(delta)
        return delta[center], dict(learned_filter=learned, pe=pe)


class SpatialRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32)
        self.geometry = _mlp((16, 32, 32))
        self.router = _mlp((155, 64, 3))
        self.up = nn.Linear(32, 256)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, states, data, arm):
        if not bool(data.geometry_valid.all()):
            raise ValueError("unavailable geometry entered spatial branch")
        count = int(data.sp_degree.numel())
        endpoints = data.sp_bond_atoms.long()
        atoms = mean_pool(torch.cat((states, states)), endpoints.reshape(-1), count)
        hidden = self.down(atoms)
        messages = []
        for edges, distance in zip(data.sp_edges, data.sp_distance):
            if edges.numel() == 0:
                messages.append(hidden.new_zeros(count, 32))
                continue
            centers = torch.linspace(0, 4.5, 16, device=hidden.device)
            rbf = torch.exp(-.5 * ((distance[:, None] - centers) / (4.5 / 15)) ** 2)
            gate = torch.sigmoid(self.geometry(rbf))
            messages.append(mean_pool(gate * hidden[edges[0]], edges[1], count))
        stacked = torch.stack(messages, 0)
        if arm == "M_EQ":
            alpha = stacked.new_full((len(data.geometry_valid), 3), 1 / 3)
        else:
            features = data.sp_router.float()
            if arm == "M_GEOM":
                features = features.clone()
                features[:, :, 3:] = 0
            alpha = torch.softmax(self.router(features.flatten(1)), -1)
        graph = []
        for scale, value in enumerate(stacked):
            bond = (value[endpoints[0]] + value[endpoints[1]]) / 2
            pooled, valid = incident_pool(bond, data.incidence_index, data.canonical_graph_index, len(alpha))
            if not torch.equal(valid, data.geometry_valid):
                raise ValueError("spatial readout availability mismatch")
            graph.append(pooled)
        delta = self.up(torch.einsum("gr,rgd->gd", alpha, torch.stack(graph, 0)))
        return delta, dict(alpha=alpha.detach())


class PHFuse(base.BaseOptFuse):
    def __init__(self, arm, ft=None):
        if arm not in ARMS:
            raise ValueError("unregistered PH arm")
        super().__init__(False, ft=ft)
        self.ph_arm = arm
        if arm in P_ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(ADAPTER_SEED)
                self.graph_ph = GraphPHAdapter()
            if sum(p.numel() for p in self.graph_ph.parameters()) != 42820:
                raise RuntimeError("P adapter parameter count mismatch")
        if arm in M_ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(ADAPTER_SEED)
                self.spatial_ph = SpatialRouter()
            if sum(p.numel() for p in self.spatial_ph.parameters()) != 36643:
                raise RuntimeError("M adapter parameter count mismatch")

    def predict(self, batch):
        original = super().encode_modalities(batch)
        z = list(original[0])
        atoms, bonds = original[1], original[2]
        if self.ph_arm in P_ARMS:
            delta, info = self.graph_ph(batch.ph_rwpe, batch.ph_filtration if self.ph_arm == "P_LEARN" else None,
                                        batch.ph_h0, batch.ph_h1, batch.ph_center, batch.ph_active)
            self.last_ph = info
            h = atoms + delta
            z[0] = self.A2(self.norm2(mean_pool(h, batch.two_d.canonical_graph_index, len(z[0]))))
        if self.ph_arm in M_ARMS:
            delta = z[1].new_zeros(z[1].shape)
            if batch.three_d is not None:
                local, info = self.spatial_ph(bonds, batch.three_d, self.ph_arm)
                self.last_ph = info
                delta = delta.index_copy(0, batch.three_d_rows, local)
            if torch.count_nonzero(delta[~batch.availability[:, 1]]):
                raise ValueError("missing 3D spatial increment is not zero")
            z[1] = (z[1] + delta) * batch.availability[:, 1:2].to(z[1])
        # Supervised fusion sees side-branch output; alignment below uses original z.
        fused, weights = self.fuse(z, batch.availability)
        prediction = self.head(fused).view(-1)
        return prediction, weights, z, original[0]

    def training_loss(self, batch):
        prediction, weights, modified, original = self.predict(batch)
        mse = nn.functional.mse_loss(prediction, batch.y.view(-1))
        sums, counts = base.alignment_terms(self.contrastive_features(original), batch.availability, batch.identity)
        align = sum(v / counts[k].clamp_min(1) for k, v in sums.items())
        entropy = base.native.entropy(weights)
        loss = mse + self.ft["align"] * align - self.ft["entropy"] * entropy
        return loss, dict(mse=mse.detach(), align=align.detach(), entropy=entropy.detach())


def allowed_new_prefixes(arm):
    if arm in P_ARMS:
        return ("graph_ph.",)
    if arm in M_ARMS:
        return ("spatial_ph.",)
    return ()
