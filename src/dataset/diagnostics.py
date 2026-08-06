"""Small diagnostics for the active MIPS-Trimer-SCAGE feature contract.

The old dense geometry diagnostics were coupled to the removed route and
could accidentally make a cache-only run look like it was using a second
model.  Production LMDB loading writes its bounded summary directly; these
helpers remain only for compatibility with callers that request a summary of
an in-memory feature collection.
"""

import json
import math
import os
from collections import Counter

import torch


def _finite_tensor(value):
    return torch.is_tensor(value) and bool(torch.isfinite(value).all())


def _as_bool(data, name, default=False):
    value = getattr(data, name, default)
    if torch.is_tensor(value):
        return bool(value.detach().cpu().bool().item()) if value.numel() == 1 else bool(value.bool().all())
    return bool(value)


def summarize_feature_cache(features, graph_input=None, geom_input=None, **_kwargs):
    """Return bounded graph/Trimer statistics for an in-memory collection."""
    total = int(len(features))
    graph_valid = 0
    trimer_valid = 0
    mcl_valid = 0
    mapping_failures = 0
    failure_reasons = Counter()
    for item in features.values() if isinstance(features, dict) else features:
        if _as_bool(item, "graph_available", False):
            graph_valid += 1
        if _as_bool(item, "trimer_geometry_valid", False):
            trimer_valid += 1
        if _as_bool(item, "mcl_valid", False):
            mcl_valid += 1
        mapping = getattr(item, "mips_to_trimer_central_index", None)
        trimer_nodes = getattr(item, "trimer_z", None)
        if mapping is not None and trimer_nodes is not None and torch.is_tensor(mapping):
            if mapping.numel() and (mapping.min() < 0 or mapping.max() >= trimer_nodes.numel()):
                mapping_failures += 1
        reason = str(getattr(item, "topology_failure_code", "") or "")
        if reason:
            failure_reasons[reason] += 1
    return {
        "schema": "mips-trimer-scage-diagnostics-v1",
        "total_unique_smiles": total,
        "graph": {
            "available_count": graph_valid,
            "unavailable_count": total - graph_valid,
            "available_rate": graph_valid / total if total else None,
        },
        "trimer": {
            "geometry_valid_count": trimer_valid,
            "geometry_invalid_count": total - trimer_valid,
            "mcl_valid_count": mcl_valid,
            "mapping_failure_count": mapping_failures,
            "failure_reason_counts": dict(failure_reasons.most_common()),
        },
        "contract": {
            "route": "MIPS-Trimer-SCAGE",
            "spatial_mode": "trimer_scage",
            "pbc": False,
            "graph_encoder": "mips_trimer_scage",
        },
    }


def print_dataset_diagnostics(summary):
    graph = summary.get("graph", {})
    trimer = summary.get("trimer", {})
    print(
        "[diagnostics] MTS graph availability: "
        f"{graph.get('available_count', 0)}/{summary.get('total_unique_smiles', 0)}"
    )
    print(
        "[diagnostics] MTS Trimer/MCL: "
        f"geometry_valid={trimer.get('geometry_valid_count', 0)}, "
        f"mcl_valid={trimer.get('mcl_valid_count', 0)}, "
        f"mapping_failures={trimer.get('mapping_failure_count', 0)}"
    )


def write_dataset_diagnostics(summary, output_path):
    directory = os.path.dirname(str(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{output_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(f"[diagnostics] wrote {output_path}")
