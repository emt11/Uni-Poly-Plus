import json
import os
from typing import Dict, Iterable, List, Optional, Set

from rdkit import Chem

from .kg_loader import KGEmbeddingStore


_SMARTS_CACHE = None
_ALIAS_CACHE = None
_ELEMENT_SYMBOL_CACHE = None


def _load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_path(root: str, relative_path: str) -> str:
    return relative_path if os.path.isabs(relative_path) else os.path.join(root, relative_path)


def load_functional_group_smarts(
    root: str = '.',
    path: str = 'knowledge_graph/mappings/functional_group_smarts.json',
):
    global _SMARTS_CACHE
    full_path = _resolve_path(root, path)
    if _SMARTS_CACHE is not None and _SMARTS_CACHE[0] == full_path:
        return _SMARTS_CACHE[1]

    smarts_dict = _load_json(full_path)
    compiled = []
    for name, smarts in smarts_dict.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            print(f"Invalid functional group SMARTS skipped: {name} -> {smarts}")
            continue
        compiled.append((name, pattern))
    _SMARTS_CACHE = (full_path, compiled)
    return compiled


def load_entity_alias(
    root: str = '.',
    path: str = 'knowledge_graph/mappings/entity_name_alias.json',
) -> Dict[str, str]:
    global _ALIAS_CACHE
    full_path = _resolve_path(root, path)
    if _ALIAS_CACHE is not None and _ALIAS_CACHE[0] == full_path:
        return _ALIAS_CACHE[1]

    alias = _load_json(full_path)
    _ALIAS_CACHE = (full_path, alias)
    return alias


def load_element_symbol_mapping(
    root: str = '.',
    path: str = 'knowledge_graph/mappings/element_symbol_mapping.json',
) -> Dict[str, int]:
    global _ELEMENT_SYMBOL_CACHE
    full_path = _resolve_path(root, path)
    if _ELEMENT_SYMBOL_CACHE is not None and _ELEMENT_SYMBOL_CACHE[0] == full_path:
        return _ELEMENT_SYMBOL_CACHE[1]

    mapping = {symbol: int(index) for symbol, index in _load_json(full_path).items()}
    _ELEMENT_SYMBOL_CACHE = (full_path, mapping)
    return mapping


def _canonical_name(name: str, alias: Dict[str, str]) -> str:
    return alias.get(name, alias.get(name.lower(), name))


def _extract_element_entity_ids(
    mol,
    kg_store: KGEmbeddingStore,
    element_symbol_mapping: Dict[str, int],
) -> List[int]:
    entity_ids = []
    seen: Set[int] = set()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 0:
            continue
        symbol = atom.GetSymbol()
        element_index = element_symbol_mapping.get(symbol, atomic_num - 1)
        entity_id = kg_store.element_to_entity_id.get(element_index)
        if entity_id is not None and entity_id not in seen:
            entity_ids.append(entity_id)
            seen.add(entity_id)
    return entity_ids


def _extract_functional_group_entity_ids(
    mol,
    kg_store: KGEmbeddingStore,
    functional_group_patterns,
    alias: Dict[str, str],
) -> List[int]:
    entity_ids = []
    seen: Set[int] = set()
    for name, pattern in functional_group_patterns:
        if not mol.HasSubstructMatch(pattern):
            continue
        canonical_name = _canonical_name(name, alias)
        entity_id = kg_store.functional_group_to_entity_id.get(canonical_name)
        if entity_id is not None and entity_id not in seen:
            entity_ids.append(entity_id)
            seen.add(entity_id)
    return entity_ids


def smiles_to_kg_entity_ids(
    smiles: str,
    kg_store: KGEmbeddingStore,
    root: str = '.',
    functional_group_smarts_path: str = 'knowledge_graph/mappings/functional_group_smarts.json',
    entity_alias_path: str = 'knowledge_graph/mappings/entity_name_alias.json',
    element_symbol_mapping_path: str = 'knowledge_graph/mappings/element_symbol_mapping.json',
) -> List[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    functional_group_patterns = load_functional_group_smarts(root, functional_group_smarts_path)
    alias = load_entity_alias(root, entity_alias_path)
    element_symbol_mapping = load_element_symbol_mapping(root, element_symbol_mapping_path)

    entity_ids = []
    entity_ids.extend(_extract_element_entity_ids(mol, kg_store, element_symbol_mapping))
    entity_ids.extend(_extract_functional_group_entity_ids(mol, kg_store, functional_group_patterns, alias))
    return entity_ids
