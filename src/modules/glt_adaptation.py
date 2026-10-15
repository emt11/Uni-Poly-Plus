"""Small, explicit downstream adaptation policies for GLT dual.

The helpers in this module deliberately leave the frozen deployment schema
untouched.  A deployment package is loaded first; LoRA modules are then
inserted for an adaptation run.  The default/full path therefore remains the
historical implementation.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRAQVMergedLinear(nn.Module):
    """A merged QKV projection with low-rank Q/V updates only.

    ``base`` is the original merged QKV layer and stays frozen.  B matrices
    are zero-initialized, so replacing a projection immediately after loading
    a deployment package is an exact identity.  The K slice has no update.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        if base.in_features != 512 or base.out_features != 1536:
            raise ValueError("GLT QKV projection must be 512 -> 1536")
        if int(rank) <= 0 or not torch.isfinite(torch.tensor(float(alpha))) or alpha <= 0:
            raise ValueError("invalid LoRA rank/alpha")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.q_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.q_b = nn.Parameter(torch.zeros(512, self.rank))
        self.v_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.v_b = nn.Parameter(torch.zeros(512, self.rank))
        nn.init.normal_(self.q_a, std=0.02)
        nn.init.normal_(self.v_a, std=0.02)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        output = self.base(inputs)
        dropped = self.dropout(inputs)
        q = (dropped @ self.q_a.t()) @ self.q_b.t() * self.scaling
        v = (dropped @ self.v_a.t()) @ self.v_b.t() * self.scaling
        delta = torch.cat((q, torch.zeros_like(q), v), dim=-1)
        return output + delta


def _replace_qkv(root: nn.Module, *, rank: int, alpha: float, dropout: float):
    replaced = []
    for module_name, module in list(root.named_modules()):
        if not module_name or not module_name.endswith("attention.qkv"):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = root.get_submodule(parent_name)
        old = getattr(parent, child_name)
        if isinstance(old, LoRAQVMergedLinear):
            replaced.append(module_name)
            continue
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{module_name} is not a merged Linear")
        replacement = LoRAQVMergedLinear(old, rank, alpha, dropout)
        # The deployment package is loaded after the encoder is moved to its
        # device.  New adapter parameters must follow the original projection
        # rather than defaulting to CPU.
        replacement = replacement.to(device=old.weight.device, dtype=old.weight.dtype)
        setattr(parent, child_name, replacement)
        replaced.append(module_name)
    if not replaced:
        raise ValueError("no GLT/O8 merged QKV projections found for LoRA")
    return replaced


def _set_requires_grad(module: nn.Module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def configure_adaptation(encoder: nn.Module, adaptation: str, *, rank: int = 8,
                         alpha: float = 8.0, dropout: float = 0.0):
    """Configure ``full``, ``head`` or ``lora`` in-place.

    Returns metadata used in the run record.  ``ridge`` is feature-only and
    intentionally does not mutate the encoder.
    """

    adaptation = str(adaptation).lower()
    if adaptation not in {"full", "head", "lora", "ridge"}:
        raise ValueError("adaptation must be full, head, lora or ridge")
    if adaptation == "full" or adaptation == "ridge":
        _set_requires_grad(encoder, adaptation == "full")
        return {"adaptation": adaptation, "lora_modules": []}

    if not hasattr(encoder, "predictor"):
        raise ValueError("dual encoder has no downstream predictor")
    _set_requires_grad(encoder, False)
    encoder.eval()
    _set_requires_grad(encoder.predictor, True)
    if adaptation == "head":
        encoder.predictor.train()
        return {"adaptation": adaptation, "lora_modules": []}

    # Load the historical deployment first, then replace QKV.  The original
    # base remains a child module and is frozen; only Q/V A/B and predictor
    # parameters are trainable.
    replaced = _replace_qkv(encoder, rank=rank, alpha=alpha, dropout=dropout)
    for name, module in encoder.named_modules():
        if isinstance(module, LoRAQVMergedLinear):
            module.eval()
            module.dropout.train()
            for parameter in (module.q_a, module.q_b, module.v_a, module.v_b):
                parameter.requires_grad = True
    encoder.predictor.train()
    return {"adaptation": adaptation, "lora_modules": replaced,
            "rank": int(rank), "alpha": float(alpha), "dropout": float(dropout)}


def trainable_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in module.named_parameters()
            if parameter.requires_grad]


class AdaptationAdapter(nn.Module):
    """Evaluation wrapper that preserves frozen-module eval mode on ``train``."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self._frozen = []

    def register_frozen(self, *modules: nn.Module):
        self._frozen = [module for module in modules if module is not None]
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen:
            module.eval()
        return self

    def forward(self, data):
        return self.encoder(data), None


class HeadFeatureCache:
    """In-memory features for one frozen head stage; labels remain dataset-owned."""

    def __init__(self, model, dataset, context):
        import copy
        self.model=model;self.dataset=dataset
        self.context=copy.deepcopy(context);self.active=True;self.features=None;self.label_shape=None
        if set(context)!= {'parent','configuration','input_view','dtype','sample_indices'}:
            raise ValueError('incomplete head feature cache binding')
        if len(context['sample_indices'])!=len(dataset) or context['dtype']!='fp32':
            raise ValueError('head feature cache indices/dtype mismatch')

    def validate(self, context):
        if not self.active or context!=self.context:
            raise ValueError('inactive or mismatched head feature cache')
        if self.model.training:
            raise ValueError('head feature cache requires an eval encoder')
        head=getattr(self.model,'head',None)
        if head is None:head=self.model.model.predictor
        ids={id(p) for p in head.parameters()}
        if any(p.requires_grad for p in self.model.parameters() if id(p) not in ids):
            raise ValueError('head feature cache encoder is not frozen')

    def build(self, device, batch_size, collate):
        from torch.utils.data import DataLoader
        from src.training.glt_dual_runtime import rng_state,restore_rng
        self.validate(self.context)
        saved=rng_state();was_training=self.model.training
        values=[]
        try:
            loader=DataLoader(self.dataset,batch_size=batch_size,shuffle=False,
                num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                for data in loader:
                    features=self.model.features(data.to(device)).detach().float().cpu()
                    if features.ndim!=2 or features.shape[1] not in (512,1024) or not torch.isfinite(features).all():
                        raise ValueError('invalid cached head features')
                    label_shape=tuple(data.y.shape[1:])
                    if self.label_shape is None:self.label_shape=label_shape
                    if self.label_shape!=label_shape or data.y.shape[0]!=features.shape[0]:
                        raise ValueError('inconsistent cached label batch shape')
                    values.append(features)
            self.features=torch.cat(values)
            if self.features.shape[0]!=len(self.dataset):raise ValueError('cache feature row count mismatch')
        finally:
            self.model.train(was_training);restore_rng(saved)

    def loader(self, original):
        from torch.utils.data import DataLoader,Dataset,Subset
        from torch_geometric.data import Data,Batch
        if original.num_workers!=0:raise ValueError('head feature cache requires the authorized workers0 recipe')
        if isinstance(original.dataset,Subset):
            if original.dataset.dataset is not self.dataset:raise ValueError('cache dataset/view mismatch')
            indices=list(original.dataset.indices)
        elif original.dataset is self.dataset:indices=list(range(len(self.dataset)))
        else:raise ValueError('cache dataset/view mismatch')
        cache=self
        class Features(Dataset):
            def __len__(self):return len(indices)
            def __getitem__(self,index):
                cache.validate(getattr(cache.model,'feature_cache_context',cache.context))
                row=indices[index]
                if hasattr(cache.dataset,'targets'):label=torch.tensor([float(cache.dataset.targets[row])],dtype=torch.float32)
                else:label=cache.dataset[row].y.clone()
                return Data(cached_features=cache.features[row:row+1],y=label.reshape((1,)+cache.label_shape))
        return DataLoader(Features(),batch_sampler=original.batch_sampler,
            num_workers=0,collate_fn=Batch.from_data_list,generator=original.generator)

    def invalidate(self):
        self.active=False;self.features=None
