from typing import Iterable, List, Sequence

import torch


def pad_kg_entity_ids(
    entity_id_list: Sequence[torch.Tensor],
    padding_idx: int = 0,
):
    batch_size = len(entity_id_list)
    max_len = max((ids.numel() for ids in entity_id_list), default=0)
    if max_len == 0:
        max_len = 1

    padded = torch.full((batch_size, max_len), padding_idx, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for row, ids in enumerate(entity_id_list):
        ids = ids.to(dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            continue
        length = ids.numel()
        padded[row, :length] = ids
        mask[row, :length] = True

    return padded, mask
