import os
import pickle
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class KGEmbeddingStore:
    embedding_weight: torch.Tensor
    element_to_entity_id: Dict[int, int]
    functional_group_to_entity_id: Dict[str, int]
    embedding_dim: int
    padding_idx: int = 0


_KG_STORE_CACHE: Dict[str, KGEmbeddingStore] = {}


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _as_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def _normalize_element_key(key) -> Optional[int]:
    if isinstance(key, int):
        return key
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _resolve_path(root: str, relative_path: str) -> str:
    return relative_path if os.path.isabs(relative_path) else os.path.join(root, relative_path)


def load_kg_embedding_store(
    root: str = '.',
    ele2emb_path: str = 'knowledge_graph/KANO_initial/ele2emb.pkl',
    fg2emb_path: str = 'knowledge_graph/KANO_initial/fg2emb.pkl',
) -> KGEmbeddingStore:
    ele_path = _resolve_path(root, ele2emb_path)
    fg_path = _resolve_path(root, fg2emb_path)
    cache_key = f"{os.path.abspath(ele_path)}::{os.path.abspath(fg_path)}"
    if cache_key in _KG_STORE_CACHE:
        return _KG_STORE_CACHE[cache_key]

    if not os.path.exists(ele_path):
        raise FileNotFoundError(f"KG element embedding file not found: {ele_path}")
    if not os.path.exists(fg_path):
        raise FileNotFoundError(f"KG functional group embedding file not found: {fg_path}")

    ele2emb = _load_pickle(ele_path)
    fg2emb = _load_pickle(fg_path)

    vectors = []
    element_to_entity_id = {}
    functional_group_to_entity_id = {}

    first_vector = None
    for mapping in (ele2emb, fg2emb):
        for value in mapping.values():
            first_vector = _as_vector(value)
            break
        if first_vector is not None:
            break
    if first_vector is None:
        raise ValueError("KG embedding files are empty.")

    embedding_dim = int(first_vector.shape[0])
    vectors.append(np.zeros(embedding_dim, dtype=np.float32))

    for raw_key in sorted(ele2emb.keys(), key=lambda item: int(item)):
        element_index = _normalize_element_key(raw_key)
        if element_index is None:
            continue
        vector = _as_vector(ele2emb[raw_key])
        if vector.shape[0] != embedding_dim:
            raise ValueError(f"Inconsistent KG element embedding dim for key {raw_key}.")
        element_to_entity_id[element_index] = len(vectors)
        vectors.append(vector)

    for fg_name in sorted(fg2emb.keys()):
        vector = _as_vector(fg2emb[fg_name])
        if vector.shape[0] != embedding_dim:
            raise ValueError(f"Inconsistent KG functional group embedding dim for key {fg_name}.")
        functional_group_to_entity_id[str(fg_name)] = len(vectors)
        vectors.append(vector)

    store = KGEmbeddingStore(
        embedding_weight=torch.tensor(np.stack(vectors, axis=0), dtype=torch.float),
        element_to_entity_id=element_to_entity_id,
        functional_group_to_entity_id=functional_group_to_entity_id,
        embedding_dim=embedding_dim,
    )
    _KG_STORE_CACHE[cache_key] = store
    return store
