#!/usr/bin/env python3
"""Zero-training distance-information audit for MTS-GLT-v2.

This audit deliberately keeps the production Trimer -> RU distance contract
unchanged.  The only changed universe is the set of endpoint pairs: the
formal GLT bond tokens are used as the reproduction gate, and then every
distinct canonical atom pair supported by the same Trimer state mapping is
enumerated.  No model, cache, checkpoint, or training artifact is written.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UniDataset  # noqa: E402
from src.dataset.canonical_periodic import _neighbors  # noqa: E402
from src.dataset.dataset import sample_key_from_smiles  # noqa: E402
from src.dataset.mts_star_rbf_v2 import (  # noqa: E402
    prepare_topology,
    prepare_trimer,
)
from src.dataset.periodic_line_glt import (  # noqa: E402
    _distance,
    _token_instances,
    build_periodic_line_sample,
    canonical_line_token,
)
from src.training.pretrain.config import (  # noqa: E402
    dataset_kwargs_from_args,
    parse_arguments,
)


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
DEFAULT_CONFIG = ROOT / "configs/mts/glt_v2_formal_a6_h_w1_20k.json"
DEFAULT_OUTPUT = ROOT / "results/mts_glt_v2/distance_information_audit_v1"
DEFAULT_DOWNSTREAM_SIDECAR = (
    ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
)
CUTOFFS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
SHELLS = (
    (None, 2.0, "d_le_2"),
    (2.0, 2.5, "d_2_2_5"),
    (2.5, 3.0, "d_2_5_3"),
    (3.0, 3.5, "d_3_3_5"),
    (3.5, 4.0, "d_3_5_4"),
    (4.0, 4.5, "d_4_4_5"),
    (4.5, 5.0, "d_4_5_5"),
    (5.0, 6.0, "d_5_6"),
)
SPD_LABELS = ("1", "2", "3", "4", "5", "6", ">=7", "disconnected")
TASK_SCOPES = tuple(f"FINETUNE_{task.upper()}" for task in TASKS)
ALL_SCOPES = ("PI1M_10K", "FINETUNE_ALL", "FINETUNE_UNIQUE", *TASK_SCOPES)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["empty"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def finite_summary(values: Iterable[float]) -> dict:
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def compact_summary(values: Iterable[float]) -> dict:
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> tuple[float | None, float | None]:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None, None
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = rankdata(x), rankdata(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


def spd_label(value: int | None) -> str:
    if value is None:
        return "disconnected"
    value = int(value)
    if value <= 6:
        return str(value)
    return ">=7"


def is_spd_ge4(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64) >= 4


def sample_key_hex(item) -> str:
    key = sample_key_from_smiles(str(item.smiles))
    return bytes(key).hex()


def formal_dataset_from_config(config_path: Path):
    parsed = parse_arguments(["--experiment_config", str(config_path)])
    return UniDataset(**dataset_kwargs_from_args(parsed)), parsed


def downstream_dataset(task: str, sidecar_root: Path):
    return UniDataset(
        root=str(ROOT / "data"),
        dataset=f"smi_{task}",
        smiles_model_name=str(
            ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"
        ),
        graph_encoder_type="mips_trimer_scage",
        graph_input="star_linking",
        modalities=("graph",),
        cache_layers="ru_base,topology,trimer,md200",
        periodic_line_glt_sidecar=str(sidecar_root),
        experiment_id="glt-v2-distance-information-audit-v1",
        feature_config_hash="manual",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
    )


def lifted_spd_maps(item, max_abs_shift: int) -> dict[int, dict[tuple[int, int], int]]:
    """Build full lifted shortest-path maps using canonical_periodic neighbors.

    The neighbor function is the production ``right,q <-> left,q+1``
    contract.  No central-RU finite graph, minimum-image reduction, or
    distance cutoff is introduced here.
    """
    atom_count = int(item.canonical_atom_count)
    internal_edges = set()
    edge = np.asarray(item.ru_edge_index.detach().cpu(), dtype=np.int64)
    for left, right in edge.T.tolist():
        left, right = int(left), int(right)
        if left != right:
            internal_edges.add((min(left, right), max(left, right)))
    internal_edges = tuple(sorted(internal_edges))
    left_boundary = int(item.ru_left_boundary)
    right_boundary = int(item.ru_right_boundary)
    shift_limit = max(4, int(max_abs_shift) + 2)
    output = {}
    for source in range(atom_count):
        start = (int(source), 0)
        distances = {start: 0}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            distance = int(distances[state])
            for neighbor in _neighbors(
                state, internal_edges, left_boundary, right_boundary
            ):
                neighbor = (int(neighbor[0]), int(neighbor[1]))
                if abs(neighbor[1]) > shift_limit or neighbor in distances:
                    continue
                distances[neighbor] = distance + 1
                queue.append(neighbor)
        output[source] = distances
    return output


def _tensor_list(value) -> list[int]:
    return [int(x) for x in np.asarray(value.detach().cpu()).reshape(-1).tolist()]


def compare_formal_sidecar(item, formal: dict) -> tuple[bool, str]:
    """Check that the current builder is byte-semantic with the loaded sidecar."""
    if not hasattr(item, "glt_token_atom_a"):
        return True, "sidecar_fields_not_attached"
    fields = (
        "glt_token_atom_a",
        "glt_token_atom_b",
        "glt_token_shift",
        "glt_token_observation_distances",
        "glt_token_observation_count",
        "glt_token_valid",
    )
    expected = {
        "glt_token_atom_a": np.asarray([row["atom_a"] for row in formal["tokens"]]),
        "glt_token_atom_b": np.asarray([row["atom_b"] for row in formal["tokens"]]),
        "glt_token_shift": np.asarray([row["shift"] for row in formal["tokens"]]),
        "glt_token_observation_distances": np.asarray(
            [row["distances"] for row in formal["tokens"]], dtype=np.float32
        ),
        "glt_token_observation_count": np.asarray(
            [row["observation_count"] for row in formal["tokens"]]
        ),
        "glt_token_valid": np.asarray([row["valid"] for row in formal["tokens"]]),
    }
    for field in fields:
        observed = np.asarray(getattr(item, field).detach().cpu())
        if observed.shape != expected[field].shape:
            return False, f"sidecar_shape_mismatch:{field}"
        if observed.dtype.kind == "f":
            equal = np.allclose(observed, expected[field], rtol=0.0, atol=2e-6)
        else:
            equal = np.array_equal(observed, expected[field])
        if not equal:
            return False, f"sidecar_value_mismatch:{field}"
    return True, "ok"


def formal_true_sample(item) -> dict:
    """Run the actual formal GLT bond builder for one dataset item."""
    key = bytes.fromhex(sample_key_hex(item))
    try:
        formal = build_periodic_line_sample(key, item, item)
    except Exception as exc:  # malformed cache records are audit-invalid
        return {
            "valid": False,
            "reason": f"formal_builder_exception:{type(exc).__name__}:{exc}",
            "formal": None,
            "sidecar_match": False,
            "sidecar_reason": "builder_exception",
        }
    if not bool(formal.get("geometry_valid", False)):
        return {
            "valid": False,
            "reason": "formal_geometry_invalid",
            "formal": formal,
            "sidecar_match": True,
            "sidecar_reason": "not_compared_invalid_geometry",
        }
    sidecar_match, sidecar_reason = compare_formal_sidecar(item, formal)
    return {
        "valid": bool(sidecar_match),
        "reason": "ok" if sidecar_match else sidecar_reason,
        "formal": formal,
        "sidecar_match": bool(sidecar_match),
        "sidecar_reason": sidecar_reason,
    }


def formal_sidecar_sample(item) -> dict:
    """View the already validated formal GLT sidecar as builder-shaped rows.

    Phase 1 compares these rows against ``build_periodic_line_sample`` for
    every sampled pretraining row.  Phase 2 can therefore reuse the immutable
    production sidecar without invoking the builder a second time.
    """
    required = (
        "glt_token_atom_a", "glt_token_atom_b", "glt_token_shift",
        "glt_token_endpoint_z_a", "glt_token_endpoint_z_b",
        "glt_token_bond_type", "glt_token_label",
        "glt_token_observation_distances", "glt_token_observation_count",
        "glt_token_valid",
    )
    if not all(hasattr(item, name) for name in required):
        return {"valid": False, "reason": "formal_sidecar_fields_missing"}
    atom_a = np.asarray(item.glt_token_atom_a.detach().cpu(), dtype=np.int64)
    atom_b = np.asarray(item.glt_token_atom_b.detach().cpu(), dtype=np.int64)
    shift = np.asarray(item.glt_token_shift.detach().cpu(), dtype=np.int64)
    z_a = np.asarray(item.glt_token_endpoint_z_a.detach().cpu(), dtype=np.int64)
    z_b = np.asarray(item.glt_token_endpoint_z_b.detach().cpu(), dtype=np.int64)
    bond_type = np.asarray(item.glt_token_bond_type.detach().cpu(), dtype=np.int64)
    label = np.asarray(item.glt_token_label.detach().cpu(), dtype=np.int64)
    distances = np.asarray(
        item.glt_token_observation_distances.detach().cpu(), dtype=np.float64
    )
    counts = np.asarray(item.glt_token_observation_count.detach().cpu(), dtype=np.int64)
    valid = np.asarray(item.glt_token_valid.detach().cpu(), dtype=bool)
    if distances.ndim != 2 or distances.shape[1] < 3:
        return {"valid": False, "reason": "formal_sidecar_distance_shape_invalid"}
    n = atom_a.size
    if any(values.size != n for values in (atom_b, shift, z_a, z_b, bond_type, label, counts, valid)):
        return {"valid": False, "reason": "formal_sidecar_token_shape_mismatch"}
    tokens = []
    for index in range(n):
        tokens.append({
            "key": (int(atom_a[index]), int(atom_b[index]), int(shift[index])),
            "atom_a": int(atom_a[index]),
            "atom_b": int(atom_b[index]),
            "shift": int(shift[index]),
            "z_a": int(z_a[index]),
            "z_b": int(z_b[index]),
            "bond_type": int(bond_type[index]),
            "label": int(label[index]),
            "distances": distances[index, :3].astype(np.float64).tolist(),
            "observation_count": int(counts[index]),
            "valid": bool(valid[index]),
        })
    return {
        "valid": bool(getattr(item, "glt_geometry_valid", True)),
        "geometry_valid": bool(getattr(item, "glt_geometry_valid", True)),
        "tokens": tokens,
        "relations": [],
    }


def formal_bond_arrays(formal: dict) -> dict:
    rows = list(formal.get("tokens", [])) if formal else []
    distances = []
    means = []
    variances = []
    counts = []
    shifts = []
    for row in rows:
        count = int(row["observation_count"])
        values = np.asarray(row["distances"], dtype=np.float64)[:count]
        if not bool(row["valid"]) or count <= 0 or not np.all(np.isfinite(values)):
            continue
        distances.extend(values.tolist())
        means.append(float(values.mean()))
        variances.append(float(values.var()))
        counts.append(count)
        shifts.append(abs(int(row["shift"])))
    return {
        "distances": np.asarray(distances, dtype=np.float64),
        "means": np.asarray(means, dtype=np.float64),
        "variances": np.asarray(variances, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int16),
        "abs_shifts": np.asarray(shifts, dtype=np.int16),
    }


def baseline_gate_summary(records: list[dict]) -> dict:
    raw = np.concatenate(
        [record["arrays"]["distances"] for record in records if record["arrays"]["distances"].size]
    ) if any(record["arrays"]["distances"].size for record in records) else np.asarray([])
    means = np.concatenate(
        [record["arrays"]["means"] for record in records if record["arrays"]["means"].size]
    ) if any(record["arrays"]["means"].size for record in records) else np.asarray([])
    variances = np.concatenate(
        [record["arrays"]["variances"] for record in records if record["arrays"]["variances"].size]
    ) if any(record["arrays"]["variances"].size for record in records) else np.asarray([])
    counts = np.concatenate(
        [record["arrays"]["counts"] for record in records if record["arrays"]["counts"].size]
    ) if any(record["arrays"]["counts"].size for record in records) else np.asarray([])
    shifts = np.concatenate(
        [record["arrays"]["abs_shifts"] for record in records if record["arrays"]["abs_shifts"].size]
    ) if any(record["arrays"]["abs_shifts"].size for record in records) else np.asarray([])
    count_hist = Counter(int(value) for value in counts.tolist())
    shift_count = Counter(int(value) for value in shifts.tolist())
    expected_count_ok = bool(
        counts.size and np.all(counts == (3 - shifts))
    )
    raw_stats = finite_summary(raw)
    relation_stats = finite_summary(means)
    # These broad windows are deliberately a gate against a semantic mismatch,
    # not a claim about a confidence interval for the data-generating process.
    checks = {
        "raw_mean_near_1p446": raw_stats["mean"] is not None and abs(raw_stats["mean"] - 1.446) <= 0.02,
        "raw_std_near_0p113": raw_stats["std"] is not None and abs(raw_stats["std"] - 0.113) <= 0.02,
        "raw_min_in_expected_range": raw_stats["min"] is not None and 0.95 <= raw_stats["min"] <= 1.10,
        "raw_max_in_expected_range": raw_stats["max"] is not None and 2.35 <= raw_stats["max"] <= 2.80,
        "counts_are_2_or_3": bool(counts.size and set(count_hist).issubset({2, 3})),
        "count_equals_3_minus_abs_shift": expected_count_ok,
        "internal_count_3": bool(shift_count.get(0, 0) == count_hist.get(3, 0)),
        "cross_count_2": bool(
            sum(value for key, value in shift_count.items() if key != 0)
            == count_hist.get(2, 0)
        ),
    }
    return {
        "schema": "mts-glt-v2-distance-bond-reproduction-v1",
        "valid_sample_count": int(len(records)),
        "raw_observation_statistics": raw_stats,
        "relation_mean_statistics": relation_stats,
        "relation_population_variance_statistics": finite_summary(variances),
        "observation_count_histogram": {str(k): int(v) for k, v in sorted(count_hist.items())},
        "abs_shift_histogram": {str(k): int(v) for k, v in sorted(shift_count.items())},
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reference": {
            "raw_mean_A": 1.446,
            "raw_std_A": 0.113,
            "raw_range_A": [1.014, 2.627],
            "count_pattern": "count = 3 - abs_shift; internal=3, cross-RU=2",
        },
    }


def all_pair_sample(item, formal_result: dict) -> dict:
    """Expand the formal relation universe without changing distance semantics."""
    topology, topology_reason = prepare_topology(item)
    if topology is None or topology_reason is not None:
        return {"valid": False, "reason": topology_reason or "topology_invalid"}
    trimer, trimer_reason = prepare_trimer(item, item)
    if trimer is None or trimer_reason is not None:
        return {"valid": False, "reason": trimer_reason or "trimer_invalid"}

    mapping = np.asarray(item.canonical_to_trimer_base_atom_id.detach().cpu(), dtype=np.int64)
    atom_count = int(item.canonical_atom_count)
    if mapping.size != atom_count or len(set(mapping.tolist())) != atom_count:
        return {"valid": False, "reason": "canonical_to_trimer_mapping_not_bijective"}
    base_to_canonical = {int(base): int(index) for index, base in enumerate(mapping.tolist())}
    states = sorted(
        ((int(base), int(shift), int(local)) for (base, shift), local in trimer["state_to_local"].items()),
        key=lambda value: value[2],
    )
    # Build the same canonical relation keys from every mapped Trimer state
    # combination.  Distances are evaluated only once per formal observation
    # below; the raw observation count is the number of state combinations.
    offsets_by_canonical = defaultdict(set)
    for base, shift, _local in states:
        canonical = base_to_canonical.get(base)
        if canonical is None:
            return {"valid": False, "reason": "trimer_base_not_in_canonical_mapping"}
        offsets_by_canonical[canonical].add(int(shift))
    raw_count_by_token = Counter()
    for canonical_left in range(atom_count):
        left_offsets = sorted(offsets_by_canonical[canonical_left])
        for canonical_right in range(canonical_left, atom_count):
            right_offsets = sorted(offsets_by_canonical[canonical_right])
            if canonical_left == canonical_right:
                # Keep distinct periodic states of the same canonical atom.
                # The exact same (atom, shift) state is not a pair, but
                # (atom,q)-(atom,q') is a valid periodic self relation and is
                # required to reproduce formal boundary true-bond tokens.
                offset_pairs = (
                    (shift_left, shift_right)
                    for index, shift_left in enumerate(left_offsets)
                    for shift_right in left_offsets[index + 1:]
                )
            else:
                offset_pairs = (
                    (shift_left, shift_right)
                    for shift_left in left_offsets
                    for shift_right in right_offsets
                )
            for shift_left, shift_right in offset_pairs:
                token = canonical_line_token(
                    canonical_left,
                    shift_left,
                    canonical_right,
                    shift_right,
                )
                raw_count_by_token[token] += 1
    raw_total_state_pairs = len(states) * max(0, len(states) - 1) // 2
    # No exact duplicate state pairs are generated; distinct periodic states
    # of one canonical atom are intentionally retained above.
    raw_excluded_same_atom = 0
    raw_observation_count = int(sum(raw_count_by_token.values()))
    if raw_observation_count != raw_total_state_pairs - raw_excluded_same_atom:
        return {"valid": False, "reason": "all_pair_raw_pair_count_mismatch"}
    max_abs_shift = max((abs(int(token[2])) for token in raw_count_by_token), default=0)
    spd_maps = lifted_spd_maps(item, max_abs_shift)
    distance_cache = {}

    def cached_distance(u, q_u, v, q_v):
        key = (int(u), int(q_u), int(v), int(q_v))
        inverse = (int(v), int(q_v), int(u), int(q_u))
        cache_key = min(key, inverse)
        if cache_key not in distance_cache:
            value = _distance(trimer, item, u, q_u, v, q_v)
            if value is None or not np.isfinite(value) or float(value) <= 0.0:
                return None
            distance_cache[cache_key] = float(value)
        return distance_cache[cache_key]

    relations = []
    instance_mismatch = []
    for token in sorted(raw_count_by_token):
        instances = _token_instances(token)
        formal_values = []
        for left, shift_left, right, shift_right in instances:
            value = cached_distance(left, shift_left, right, shift_right)
            if value is None:
                return {"valid": False, "reason": "all_pair_formal_instance_invalid"}
            formal_values.append(float(value))
        raw_count = int(raw_count_by_token[token])
        if len(formal_values) != raw_count:
            instance_mismatch.append({
                "token": list(token),
                "raw_count": raw_count,
                "formal_count": int(len(formal_values)),
            })
            continue
        values = np.asarray(formal_values, dtype=np.float64)
        atom_a, atom_b, shift = map(int, token)
        target = (atom_b, int(shift))
        spd = spd_maps.get(atom_a, {}).get(target)
        relations.append({
            "token": token,
            "atom_a": atom_a,
            "atom_b": atom_b,
            "shift": shift,
            "abs_shift": abs(shift),
            "same_ru": bool(shift == 0),
            "spd": None if spd is None else int(spd),
            "spd_label": spd_label(spd),
            "distances": values,
            "distance_mean": float(values.mean()),
            "distance_variance": float(values.var()),
            "observation_count": int(values.size),
        })
    if instance_mismatch:
        return {
            "valid": False,
            "reason": "all_pair_instance_mismatch",
            "instance_mismatch": instance_mismatch[:10],
        }
    if len(relations) != len(raw_count_by_token):
        return {"valid": False, "reason": "all_pair_relation_count_mismatch"}

    formal = formal_result.get("formal") or {}
    formal_rows = list(formal.get("tokens", []))
    formal_map = {tuple(row["key"]): row for row in formal_rows}
    relation_map = {tuple(row["token"]): row for row in relations}
    true_token_mismatch = []
    for row in formal_rows:
        token = tuple(row["key"])
        candidate = relation_map.get(token)
        count = int(row["observation_count"])
        formal_values = np.asarray(row["distances"], dtype=np.float64)[:count]
        if candidate is None or count != int(candidate["observation_count"]):
            true_token_mismatch.append({"token": list(token), "reason": "missing_or_count"})
            continue
        if not np.allclose(
            formal_values,
            candidate["distances"],
            rtol=0.0,
            atol=2e-6,
        ):
            true_token_mismatch.append({"token": list(token), "reason": "distance_value"})
    if true_token_mismatch:
        return {
            "valid": False,
            "reason": "all_pair_true_bond_reproduction_mismatch",
            "true_token_mismatch": true_token_mismatch[:10],
        }
    canonical_relation_count = len(relations)
    return {
        "valid": True,
        "reason": "ok",
        "sample_key": sample_key_hex(item),
        "n_atoms": atom_count,
        "raw_observation_count": raw_observation_count,
        "canonical_relation_count": canonical_relation_count,
        "duplicates_removed": raw_observation_count - canonical_relation_count,
        "raw_excluded_same_canonical_atom_pairs": int(raw_excluded_same_atom),
        "periodic_self_relation_count": int(sum(
            1 for token in raw_count_by_token if token[0] == token[1]
        )),
        "relations": relations,
        "formal_token_count": len(formal_map),
        "formal_true_token_count_in_all_pairs": sum(
            1 for token in formal_map if token in relation_map
        ),
    }


class BondAccumulator:
    def __init__(self):
        self.records = []
        self.invalid = 0
        self.sidecar_mismatch = 0
        self.all_pair_mismatch = 0

    def add_formal(self, result: dict):
        if not result.get("valid"):
            self.invalid += 1
            if str(result.get("sidecar_reason", "")).startswith("sidecar_"):
                self.sidecar_mismatch += 1
            return
        self.records.append({"arrays": formal_bond_arrays(result["formal"])})

    def summary(self, total_samples: int) -> dict:
        result = baseline_gate_summary(self.records)
        result.update({
            "sampled_row_count": int(total_samples),
            "invalid_or_unusable_sample_count": int(self.invalid),
            "sidecar_mismatch_count": int(self.sidecar_mismatch),
        })
        return result


class ScopeAccumulator:
    """Streaming relation and per-polymer accumulator."""

    def __init__(self, name: str):
        self.name = str(name)
        self.sample_count = 0
        self.valid_sample_count = 0
        self.invalid_sample_count = 0
        self.invalid_reasons = Counter()
        self.raw_observation_count = 0
        self.canonical_relation_count = 0
        self.duplicates_removed = 0
        self.same_canonical_excluded = 0
        self.periodic_self_relation_count = 0
        self.distances = array("f")
        self.variances = array("f")
        self.spds = array("h")
        self.counts = array("b")
        self.abs_shifts = array("b")
        self.sample_ids = array("i")
        self.n_atoms_by_relation = array("h")
        self.item_records: list[dict] = []
        self.polymer_spd_counts = {label: array("I") for label in SPD_LABELS}
        self.scale_values = {percentile: array("f") for percentile in (20, 30, 40, 50, 60)}

    def add_invalid(self, n_atoms: int | None, reason: str):
        self.sample_count += 1
        self.invalid_sample_count += 1
        self.invalid_reasons[str(reason)] += 1
        self.item_records.append({
            "n_atoms": int(n_atoms or 0),
            "valid": False,
            "contact_counts": {cutoff: 0 for cutoff in CUTOFFS},
        })

    def add(self, sample: dict, sample_id: int):
        if not sample.get("valid"):
            self.add_invalid(sample.get("n_atoms"), sample.get("reason", "invalid"))
            return
        self.sample_count += 1
        self.valid_sample_count += 1
        relations = sample["relations"]
        self.raw_observation_count += int(sample["raw_observation_count"])
        self.canonical_relation_count += int(sample["canonical_relation_count"])
        self.duplicates_removed += int(sample["duplicates_removed"])
        self.same_canonical_excluded += int(sample.get("raw_excluded_same_canonical_atom_pairs", 0))
        self.periodic_self_relation_count += int(sample.get("periodic_self_relation_count", 0))
        n_atoms = int(sample["n_atoms"])
        distances = np.asarray([row["distance_mean"] for row in relations], dtype=np.float64)
        contact_counts = {}
        spd_group_counts = Counter()
        for row in relations:
            spd_value = -1 if row["spd"] is None else int(row["spd"])
            self.distances.append(float(row["distance_mean"]))
            self.variances.append(float(row["distance_variance"]))
            self.spds.append(spd_value)
            self.counts.append(int(row["observation_count"]))
            self.abs_shifts.append(int(row["abs_shift"]))
            self.sample_ids.append(int(sample_id))
            self.n_atoms_by_relation.append(n_atoms)
            spd_group_counts[row["spd_label"]] += 1
        for label in SPD_LABELS:
            self.polymer_spd_counts[label].append(int(spd_group_counts[label]))
        for percentile in self.scale_values:
            self.scale_values[percentile].append(
                float(np.quantile(distances, percentile / 100.0))
            )
        candidate = {}
        for cutoff in CUTOFFS:
            candidate[cutoff] = int(sum(
                row["spd"] is not None and int(row["spd"]) >= 4
                and float(row["distance_mean"]) < cutoff
                for row in relations
            ))
        pair_denominator = n_atoms * max(0, n_atoms - 1) / 2.0
        self.item_records.append({
            "n_atoms": n_atoms,
            "valid": True,
            "relation_count": len(relations),
            "raw_observation_count": int(sample["raw_observation_count"]),
            "contact_counts": candidate,
            "rho_atom": {cutoff: candidate[cutoff] / max(1, n_atoms) for cutoff in CUTOFFS},
            "rho_pair": {cutoff: candidate[cutoff] / max(1.0, pair_denominator) for cutoff in CUTOFFS},
        })

    def arrays_np(self) -> dict[str, np.ndarray]:
        return {
            "distance": np.asarray(self.distances, dtype=np.float64),
            "variance": np.asarray(self.variances, dtype=np.float64),
            "spd": np.asarray(self.spds, dtype=np.int64),
            "count": np.asarray(self.counts, dtype=np.int64),
            "abs_shift": np.asarray(self.abs_shifts, dtype=np.int64),
            "sample_id": np.asarray(self.sample_ids, dtype=np.int64),
            "n_atoms": np.asarray(self.n_atoms_by_relation, dtype=np.int64),
        }

    def summary_row(self) -> dict:
        arrays = self.arrays_np()
        distance = finite_summary(arrays["distance"])
        return {
            "scope": self.name,
            "sample_count": self.sample_count,
            "valid_sample_count": self.valid_sample_count,
            "invalid_sample_count": self.invalid_sample_count,
            "relation_count": int(arrays["distance"].size),
            "raw_observation_count": self.raw_observation_count,
            "canonical_relation_count": self.canonical_relation_count,
            "duplicates_removed": self.duplicates_removed,
            "duplicate_fraction_of_raw": self.duplicates_removed / max(1, self.raw_observation_count),
            "exact_duplicate_state_pairs_excluded": self.same_canonical_excluded,
            "periodic_self_relation_count": self.periodic_self_relation_count,
            "distance_mean_A": distance["mean"],
            "distance_std_A": distance["std"],
            "distance_min_A": distance["min"],
            "distance_p50_A": distance["p50"],
            "distance_p99_A": distance["p99"],
            "distance_max_A": distance["max"],
            "invalid_reasons": json.dumps(dict(self.invalid_reasons), sort_keys=True),
        }

    def distance_spd_rows(self) -> list[dict]:
        arrays = self.arrays_np()
        total = arrays["distance"].size
        rows = []
        for label in SPD_LABELS:
            if label == "disconnected":
                mask = arrays["spd"] < 0
            elif label == ">=7":
                mask = arrays["spd"] >= 7
            else:
                mask = arrays["spd"] == int(label)
            values = arrays["distance"][mask]
            stats = finite_summary(values)
            polymer_values = np.asarray(self.polymer_spd_counts[label], dtype=np.float64)
            polymer_stats = compact_summary(polymer_values)
            rows.append({
                "scope": self.name,
                "spd": label,
                "relation_count": int(values.size),
                "relation_fraction": float(values.size / max(1, total)),
                **{f"distance_{key}": value for key, value in stats.items()},
                "per_polymer_relation_count_mean": polymer_stats["mean"],
                "per_polymer_relation_count_median": polymer_stats["median"],
                "per_polymer_relation_count_p25": polymer_stats["p25"],
                "per_polymer_relation_count_p75": polymer_stats["p75"],
                "per_polymer_relation_count_p90": polymer_stats["p90"],
            })
        invalid_row = {
            "scope": self.name,
            "spd": "invalid",
            "relation_count": 0,
            "relation_fraction": 0.0,
            "sample_count": self.invalid_sample_count,
        }
        rows.append(invalid_row)
        return rows

    def cutoff_rows(self) -> list[dict]:
        arrays = self.arrays_np()
        distances, spd = arrays["distance"], arrays["spd"]
        rows = []
        for cutoff in CUTOFFS:
            short = distances < cutoff
            candidate = short & (spd >= 4)
            ge4 = spd >= 4
            row = {
                "scope": self.name,
                "cutoff_A": cutoff,
                "all_relation_count_d_lt_r": int(short.sum()),
                "spd_ge4_relation_count_d_lt_r": int(candidate.sum()),
                "spd_ge4_given_d_lt_r": float(candidate.sum() / max(1, short.sum())),
                "d_lt_r_given_spd_ge4": float(candidate.sum() / max(1, ge4.sum())),
                "same_ru_candidate_count": int((candidate & (arrays["abs_shift"] == 0)).sum()),
                "cross_ru_candidate_count": int((candidate & (arrays["abs_shift"] != 0)).sum()),
                "same_ru_candidate_fraction": float(
                    (candidate & (arrays["abs_shift"] == 0)).sum() / max(1, candidate.sum())
                ),
                "cross_ru_candidate_fraction": float(
                    (candidate & (arrays["abs_shift"] != 0)).sum() / max(1, candidate.sum())
                ),
            }
            for label in SPD_LABELS:
                if label == "disconnected":
                    mask = spd < 0
                elif label == ">=7":
                    mask = spd >= 7
                else:
                    mask = spd == int(label)
                row[f"spd_{label.replace('>=', 'ge')}_count"] = int((short & mask).sum())
                row[f"spd_{label.replace('>=', 'ge')}_fraction_of_short"] = float(
                    (short & mask).sum() / max(1, short.sum())
                )
            rows.append(row)
        return rows

    def shell_rows(self) -> list[dict]:
        arrays = self.arrays_np()
        distances, spd = arrays["distance"], arrays["spd"]
        rows = []
        for lower, upper, name in SHELLS:
            shell = distances <= upper
            if lower is not None:
                shell &= distances > lower
            row = {
                "scope": self.name,
                "shell": name,
                "lower_exclusive_A": lower,
                "upper_inclusive_A": upper,
                "relation_count": int(shell.sum()),
            }
            for label in SPD_LABELS:
                if label == "disconnected":
                    mask = spd < 0
                elif label == ">=7":
                    mask = spd >= 7
                else:
                    mask = spd == int(label)
                count = int((shell & mask).sum())
                row[f"spd_{label.replace('>=', 'ge')}_count"] = count
                row[f"spd_{label.replace('>=', 'ge')}_fraction"] = float(
                    count / max(1, shell.sum())
                )
            rows.append(row)
        return rows

    def polymer_density_rows(self) -> list[dict]:
        rows = []
        valid_items = [item for item in self.item_records if item.get("valid")]
        atom_counts = np.asarray([item["n_atoms"] for item in valid_items], dtype=np.float64)
        for cutoff in CUTOFFS:
            contacts = np.asarray([item["contact_counts"][cutoff] for item in valid_items], dtype=np.float64)
            rho_atom = np.asarray([item["rho_atom"][cutoff] for item in valid_items], dtype=np.float64)
            rho_pair = np.asarray([item["rho_pair"][cutoff] for item in valid_items], dtype=np.float64)
            pearson_n, spearman_n = correlation(atom_counts, contacts)
            pearson_atom, spearman_atom = correlation(atom_counts, rho_atom)
            pearson_pair, spearman_pair = correlation(atom_counts, rho_pair)
            summary = compact_summary(contacts)
            atom_summary = compact_summary(rho_atom)
            pair_summary = compact_summary(rho_pair)
            row = {
                "scope": self.name,
                "cutoff_A": cutoff,
                "polymer_count": int(contacts.size),
                "N_NB_mean": summary["mean"],
                "N_NB_median": summary["median"],
                "N_NB_p25": summary["p25"],
                "N_NB_p75": summary["p75"],
                "N_NB_p90": summary["p90"],
                "fraction_N_NB_eq_0": float(np.mean(contacts == 0)) if contacts.size else None,
                "fraction_N_NB_ge_1": float(np.mean(contacts >= 1)) if contacts.size else None,
                "fraction_N_NB_ge_3": float(np.mean(contacts >= 3)) if contacts.size else None,
                "fraction_N_NB_ge_5": float(np.mean(contacts >= 5)) if contacts.size else None,
                "fraction_N_NB_ge_10": float(np.mean(contacts >= 10)) if contacts.size else None,
                "rho_atom_mean": atom_summary["mean"],
                "rho_atom_median": atom_summary["median"],
                "rho_atom_p25": atom_summary["p25"],
                "rho_atom_p75": atom_summary["p75"],
                "rho_pair_mean": pair_summary["mean"],
                "rho_pair_median": pair_summary["median"],
                "rho_pair_p25": pair_summary["p25"],
                "rho_pair_p75": pair_summary["p75"],
                "pearson_N_NB_vs_N_atom": pearson_n,
                "spearman_N_NB_vs_N_atom": spearman_n,
                "pearson_rho_atom_vs_N_atom": pearson_atom,
                "spearman_rho_atom_vs_N_atom": spearman_atom,
                "pearson_rho_pair_vs_N_atom": pearson_pair,
                "spearman_rho_pair_vs_N_atom": spearman_pair,
            }
            rows.append(row)
        return rows

    def variance_rows(self) -> list[dict]:
        arrays = self.arrays_np()
        distance, variance, spd = arrays["distance"], arrays["variance"], arrays["spd"]
        groups = {"all": np.ones(distance.size, dtype=bool), "spd_ge4": spd >= 4}
        for cutoff in (3.0, 4.0, 5.0):
            groups[f"spd_ge4_d_lt_{str(cutoff).replace('.', '_')}"] = (spd >= 4) & (distance < cutoff)
        for label in SPD_LABELS:
            if label == "disconnected":
                groups["spd_disconnected"] = spd < 0
            elif label == ">=7":
                groups["spd_ge7"] = spd >= 7
            else:
                groups[f"spd_{label}"] = spd == int(label)
        rows = []
        for group, mask in groups.items():
            values = variance[mask]
            stats = finite_summary(values)
            rows.append({
                "scope": self.name,
                "group": group,
                "relation_count": int(values.size),
                "variance_mean_A2": stats["mean"],
                "variance_median_A2": stats["p50"],
                "variance_p90_A2": stats["p90"],
                "variance_p95_A2": stats["p95"],
                "variance_p99_A2": stats["p99"],
                "variance_max_A2": stats["max"],
                "fraction_low_variance_lt_0p01_A2": float(np.mean(values < 0.01)) if values.size else None,
                "fraction_high_variance_ge_0p25_A2": float(np.mean(values >= 0.25)) if values.size else None,
            })
        return rows

    def count_rows(self) -> list[dict]:
        arrays = self.arrays_np()
        distance, counts, shifts, spd = (
            arrays["distance"], arrays["count"], arrays["abs_shift"], arrays["spd"]
        )
        groups = {"all": np.ones(distance.size, dtype=bool), "same_ru": shifts == 0, "cross_ru": shifts != 0}
        for label in SPD_LABELS:
            if label == "disconnected":
                groups["spd_disconnected"] = spd < 0
            elif label == ">=7":
                groups["spd_ge7"] = spd >= 7
            else:
                groups[f"spd_{label}"] = spd == int(label)
        for cutoff in (3.0, 4.0, 5.0):
            groups[f"d_lt_{str(cutoff).replace('.', '_')}"] = distance < cutoff
            groups[f"spd_ge4_d_lt_{str(cutoff).replace('.', '_')}"] = (spd >= 4) & (distance < cutoff)
        rows = []
        for group, mask in groups.items():
            values = counts[mask]
            shift_values = shifts[mask]
            expected = 3 - shift_values
            rows.append({
                "scope": self.name,
                "group": group,
                "relation_count": int(values.size),
                "count_mean": float(values.mean()) if values.size else None,
                "count_median": float(np.quantile(values, 0.5)) if values.size else None,
                "count_min": int(values.min()) if values.size else None,
                "count_max": int(values.max()) if values.size else None,
                "fraction_count_eq_1": float(np.mean(values == 1)) if values.size else None,
                "fraction_count_eq_2": float(np.mean(values == 2)) if values.size else None,
                "fraction_count_eq_3": float(np.mean(values == 3)) if values.size else None,
                "mean_abs_shift": float(shift_values.mean()) if values.size else None,
                "fraction_count_eq_3_minus_abs_shift": float(np.mean(values == expected)) if values.size else None,
                "count_contract_mismatch": int(np.count_nonzero(values != expected)),
            })
        return rows

    def scale_rows(self) -> list[dict]:
        rows = []
        for percentile, values in self.scale_values.items():
            stats = finite_summary(values)
            rows.append({
                "scope": self.name,
                "scale_type": "per_polymer_percentile",
                "scale": f"p{percentile}",
                "polymer_count": int(len(values)),
                "distance_A_mean": stats["mean"],
                "distance_A_median": stats["p50"],
                "distance_A_p25": stats["p25"],
                "distance_A_p75": stats["p75"],
                "distance_A_p90": stats["p90"],
                "distance_A_min": stats["min"],
                "distance_A_max": stats["max"],
            })
        for cutoff in (3.0, 4.0, 5.0):
            values = np.asarray([
                item["contact_counts"][cutoff] for item in self.item_records if item.get("valid")
            ], dtype=np.float64)
            stats = compact_summary(values)
            rows.append({
                "scope": self.name,
                "scale_type": "fixed_physical_cutoff_candidate_contacts",
                "scale": f"{cutoff:g}A",
                "polymer_count": int(values.size),
                "distance_A_mean": None,
                "distance_A_median": None,
                "distance_A_p25": None,
                "distance_A_p75": None,
                "distance_A_p90": None,
                "contact_count_mean": stats["mean"],
                "contact_count_median": stats["median"],
                "fraction_polymer_any": float(np.mean(values > 0)) if values.size else None,
            })
        return rows


def validate_sidecar_result(formal_result: dict) -> bool:
    return bool(formal_result.get("valid") and formal_result.get("sidecar_match", True))


def process_scope_item(
    item,
    sample_id: int,
    scope: ScopeAccumulator,
    formal_check: bool = True,
):
    """Run one all-pair extraction and add it to an accumulator."""
    try:
        key = sample_key_hex(item)
    except Exception as exc:
        scope.add_invalid(0, f"sample_key_exception:{type(exc).__name__}")
        return {"valid": False, "reason": "sample_key_exception"}, None
    formal_result = formal_true_sample(item) if formal_check else formal_sidecar_sample(item)
    if not validate_sidecar_result(formal_result):
        scope.add_invalid(getattr(item, "canonical_atom_count", 0), formal_result.get("reason", "formal_invalid"))
        return {"valid": False, "reason": formal_result.get("reason", "formal_invalid"), "sample_key": key}, formal_result
    result = all_pair_sample(item, formal_result)
    if result.get("valid"):
        result["sample_key"] = key
        scope.add(result, sample_id)
    else:
        result["sample_key"] = key
        scope.add_invalid(getattr(item, "canonical_atom_count", 0), result.get("reason", "all_pair_invalid"))
    return result, formal_result


def task_row_metadata(task: str, item, sample_id: int) -> dict:
    return {
        "task": task,
        "row": int(sample_id),
        "sample_key": sample_key_hex(item),
        "smiles": str(item.smiles),
    }


def scope_metric_lookup(rows: list[dict], scope: str, cutoff: float) -> dict:
    for row in rows:
        if row.get("scope") == scope and abs(float(row.get("cutoff_A", -1)) - cutoff) < 1e-9:
            return row
    return {}


def choose_decision(
    bond_summary: dict,
    cutoff_rows: list[dict],
    density_rows: list[dict],
    variance_rows: list[dict],
    scope_rows: list[dict],
    validation_mismatch: dict | None = None,
) -> tuple[str, dict]:
    if not bond_summary.get("pass", False):
        return "STOP", {"reason": "true_bond_reproduction_gate_failed"}
    validation_mismatch = validation_mismatch or {}
    all_pair_true_bond_mismatches = int(
        validation_mismatch.get("all_pair_true_bond_reproduction_mismatch", 0)
    )
    if all_pair_true_bond_mismatches:
        return "STOP", {
            "reason": "all_pair_true_bond_reproduction_gate_failed",
            "all_pair_true_bond_reproduction_mismatch": all_pair_true_bond_mismatches,
        }
    def get(rows, scope, cutoff):
        return scope_metric_lookup(rows, scope, cutoff)
    pre4 = get(cutoff_rows, "PI1M_10K", 4.0)
    pre5 = get(cutoff_rows, "PI1M_10K", 5.0)
    ft4 = get(cutoff_rows, "FINETUNE_ALL", 4.0)
    pre_density = next((row for row in density_rows if row["scope"] == "PI1M_10K" and row["cutoff_A"] == 4.0), {})
    ft_density = next((row for row in density_rows if row["scope"] == "FINETUNE_ALL" and row["cutoff_A"] == 4.0), {})
    pre_var = next((row for row in variance_rows if row["scope"] == "PI1M_10K" and row["group"] == "spd_ge4_d_lt_4_0"), {})
    ft_var = next((row for row in variance_rows if row["scope"] == "FINETUNE_ALL" and row["group"] == "spd_ge4_d_lt_4_0"), {})
    scope_valid = {row["scope"]: int(row.get("valid_sample_count", 0)) for row in scope_rows}
    coverage = float(pre_density.get("fraction_N_NB_ge_1", 0.0) or 0.0)
    ft_coverage = float(ft_density.get("fraction_N_NB_ge_1", 0.0) or 0.0)
    candidate_fraction = float(pre4.get("spd_ge4_given_d_lt_r", 0.0) or 0.0)
    candidate_given_ge4 = float(pre4.get("d_lt_r_given_spd_ge4", 0.0) or 0.0)
    high_variance = float(pre_var.get("fraction_high_variance_ge_0p25_A2", 1.0) or 1.0)
    ft_high_variance = float(ft_var.get("fraction_high_variance_ge_0p25_A2", 1.0) or 1.0)
    present_in_scopes = scope_valid.get("PI1M_10K", 0) > 0 and scope_valid.get("FINETUNE_ALL", 0) > 0
    evidence = {
        "pretrain_candidate_fraction_among_d_lt_4": candidate_fraction,
        "pretrain_short_fraction_among_spd_ge4": candidate_given_ge4,
        "pretrain_polymer_fraction_with_candidate_lt_4": coverage,
        "finetune_polymer_fraction_with_candidate_lt_4": ft_coverage,
        "pretrain_candidate_high_variance_fraction": high_variance,
        "finetune_candidate_high_variance_fraction": ft_high_variance,
        "candidate_present_in_pretrain_and_finetune": present_in_scopes,
        "pretrain_spd_ge4_d_lt_5_count": int(pre5.get("spd_ge4_relation_count_d_lt_r", 0) or 0),
        "finetune_spd_ge4_d_lt_4_count": int(ft4.get("spd_ge4_relation_count_d_lt_r", 0) or 0),
        "all_pair_true_bond_reproduction_mismatch": all_pair_true_bond_mismatches,
    }
    # Conservative audit-only decision thresholds.  They are not model gates.
    if candidate_fraction < 0.01 or coverage < 0.20 or not present_in_scopes:
        decision = "STOP"
        reason = "SPD>=4 short contacts are too sparse or absent from a major scope"
    elif high_variance > 0.50 or ft_high_variance > 0.50:
        decision = "CONDITIONAL"
        reason = "candidate contacts exist but observation variance is frequently high"
    elif coverage >= 0.50 and ft_coverage >= 0.30 and candidate_fraction >= 0.10:
        decision = "GO"
        reason = "candidate contacts are prevalent, cross-scope, and not dominated by high variance"
    else:
        decision = "CONDITIONAL"
        reason = "candidate contacts are present but coverage or scale composition is task-dependent"
    return decision, {"reason": reason, "evidence": evidence}


def write_polymer_overlap(output: Path, task_keys: dict[str, dict[str, dict]]):
    all_keys = set().union(*(set(values) for values in task_keys.values())) if task_keys else set()
    rows = []
    for key in sorted(all_keys):
        tasks = sorted(task for task, values in task_keys.items() if key in values)
        rows.append({
            "sample_key": key,
            "task_count": len(tasks),
            "tasks": ";".join(tasks),
            "row_count": sum(int(task_keys[task][key]["rows"]) for task in tasks),
        })
    atomic_csv(output / "downstream_polymer_repeats.csv", rows)
    return {
        "unique_polymer_count": len(all_keys),
        "polymer_present_in_multiple_tasks": int(sum(len([task for task, values in task_keys.items() if key in values]) > 1 for key in all_keys)),
        "task_pair_overlap": {
            f"{left}__{right}": int(len(set(task_keys.get(left, {})) & set(task_keys.get(right, {}))))
            for left_index, left in enumerate(TASKS)
            for right in TASKS[left_index + 1:]
        },
    }


def plot_outputs(output: Path, accumulators: dict[str, ScopeAccumulator], shell_rows: list[dict], cutoff_rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        atomic_json(output / "plot_status.json", {"status": "SKIPPED", "reason": str(exc)})
        return
    rng = np.random.default_rng(42)
    plot_scopes = [scope for scope in ("PI1M_10K", "FINETUNE_ALL", "FINETUNE_UNIQUE") if scope in accumulators]
    colors = {"PI1M_10K": "#1f77b4", "FINETUNE_ALL": "#d62728", "FINETUNE_UNIQUE": "#2ca02c"}

    plt.figure(figsize=(8, 5))
    for scope in plot_scopes:
        values = accumulators[scope].arrays_np()["distance"]
        if values.size > 250000:
            values = values[rng.choice(values.size, 250000, replace=False)]
        if values.size:
            plt.hist(values, bins=120, density=True, alpha=0.30, label=scope, color=colors[scope])
    plt.xlabel("relation distance mean (Å)"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "all_distance_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    scope = accumulators["PI1M_10K"].arrays_np()
    for label in SPD_LABELS:
        mask = scope["spd"] < 0 if label == "disconnected" else scope["spd"] >= 7 if label == ">=7" else scope["spd"] == int(label)
        values = scope["distance"][mask]
        if values.size:
            if values.size > 100000:
                values = values[rng.choice(values.size, 100000, replace=False)]
            plt.hist(values, bins=100, density=True, histtype="step", linewidth=1.2, label=f"SPD {label}")
    plt.xlabel("relation distance mean (Å)"); plt.ylabel("density"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(output / "distance_by_spd.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    for scope_name in plot_scopes:
        arrays = accumulators[scope_name].arrays_np()
        values = arrays["distance"][arrays["spd"] >= 4]
        if values.size:
            if values.size > 250000:
                values = values[rng.choice(values.size, 250000, replace=False)]
            plt.hist(values, bins=100, density=True, alpha=0.28, label=scope_name, color=colors[scope_name])
    plt.xlabel("SPD≥4 relation distance mean (Å)"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "spd_ge4_distance_distribution.png", dpi=160); plt.close()

    def stacked_plot(rows, filename, x_key, title):
        selected = [row for row in rows if row["scope"] == "PI1M_10K"]
        x = [str(row[x_key]) for row in selected]
        bottom = np.zeros(len(selected))
        plt.figure(figsize=(10, 5))
        for label in ("1", "2", "3", "4", "5", "6", ">=7"):
            key = f"spd_{label.replace('>=', 'ge')}_fraction_of_short" if x_key == "cutoff_A" else f"spd_{label.replace('>=', 'ge')}_fraction"
            vals = np.asarray([float(row.get(key, 0.0) or 0.0) for row in selected])
            plt.bar(x, vals, bottom=bottom, label=f"SPD {label}")
            bottom += vals
        plt.title(title); plt.ylabel("fraction"); plt.xlabel(x_key); plt.legend(ncol=4, fontsize=8); plt.tight_layout()
        plt.savefig(output / filename, dpi=160); plt.close()
    stacked_plot(cutoff_rows, "cutoff_composition.png", "cutoff_A", "PI1M short-distance SPD composition")
    stacked_plot(shell_rows, "distance_shell_composition.png", "shell", "PI1M incremental shell SPD composition")

    plt.figure(figsize=(8, 5))
    for scope_name in plot_scopes:
        records = [item for item in accumulators[scope_name].item_records if item.get("valid")]
        values = np.asarray([item["contact_counts"][4.0] for item in records], dtype=np.float64)
        if values.size:
            plt.hist(values, bins=min(60, max(10, int(values.max()) + 1)), alpha=0.35, label=scope_name, color=colors[scope_name])
    plt.xlabel("N_NB (SPD≥4, d<4 Å) per polymer"); plt.ylabel("polymer count"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "contacts_per_polymer.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    for scope_name in plot_scopes:
        records = [item for item in accumulators[scope_name].item_records if item.get("valid")]
        x = np.asarray([item["n_atoms"] for item in records], dtype=np.float64)
        y = np.asarray([item["rho_atom"][4.0] for item in records], dtype=np.float64)
        if x.size:
            if x.size > 15000:
                indices = rng.choice(x.size, 15000, replace=False); x, y = x[indices], y[indices]
            plt.scatter(x, y, s=5, alpha=0.25, label=scope_name, color=colors[scope_name])
    plt.xlabel("canonical atom count"); plt.ylabel("rho_atom (SPD≥4, d<4 Å)"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "density_vs_ru_size.png", dpi=160); plt.close()

    arrays = accumulators["PI1M_10K"].arrays_np()
    indices = np.arange(arrays["distance"].size)
    if indices.size > 150000:
        indices = rng.choice(indices, 150000, replace=False)
    plt.figure(figsize=(8, 5)); plt.scatter(arrays["distance"][indices], arrays["variance"][indices], s=3, alpha=0.12)
    plt.yscale("log"); plt.xlabel("relation distance mean (Å)"); plt.ylabel("population variance (Å²)"); plt.tight_layout()
    plt.savefig(output / "variance_vs_distance.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    for scope_name in plot_scopes:
        values = accumulators[scope_name].scale_values
        x = np.arange(5)
        y = [float(np.median(values[p])) if len(values[p]) else np.nan for p in (20, 30, 40, 50, 60)]
        plt.plot(x, y, marker="o", label=scope_name, color=colors[scope_name])
    plt.axhline(3.0, color="black", linestyle="--", linewidth=0.8, label="3 Å")
    plt.axhline(4.0, color="black", linestyle=":", linewidth=0.8, label="4 Å")
    plt.axhline(5.0, color="black", linestyle="-.", linewidth=0.8, label="5 Å")
    plt.xticks(np.arange(5), ["p20", "p30", "p40", "p50", "p60"]); plt.ylabel("distance (Å)"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "fixed_cutoff_vs_percentile.png", dpi=160); plt.close()
    atomic_json(output / "plot_status.json", {"status": "OK", "scopes": plot_scopes})


def report_text(
    output: Path,
    decision: str,
    decision_detail: dict,
    bond_summary: dict,
    scope_rows: list[dict],
    cutoff_rows: list[dict],
    density_rows: list[dict],
    variance_rows: list[dict],
    count_rows: list[dict],
    overlap: dict,
    manifest: dict,
) -> str:
    def row(scope, cutoff):
        return scope_metric_lookup(cutoff_rows, scope, cutoff)
    pre4, pre5 = row("PI1M_10K", 4.0), row("PI1M_10K", 5.0)
    ft4 = row("FINETUNE_ALL", 4.0)
    pre_density = next((r for r in density_rows if r["scope"] == "PI1M_10K" and r["cutoff_A"] == 4.0), {})
    pre_var = next((r for r in variance_rows if r["scope"] == "PI1M_10K" and r["group"] == "spd_ge4_d_lt_4_0"), {})
    pre_count = next((r for r in count_rows if r["scope"] == "PI1M_10K" and r["group"] == "all"), {})
    valid = {r["scope"]: r for r in scope_rows}
    lines = [
        "# MTS-GLT-v2 zero-training distance information audit",
        "",
        f"## Decision: {decision}",
        "",
        decision_detail.get("reason", ""),
        "",
        "This is a data-layer audit only. No pretraining, fine-tuning, screening, checkpoint, architecture, gate, or RBF/basis change was run.",
        "",
        "## Distance pipeline",
        "",
        "The audit reuses the formal implementations below and changes only the pair universe from formal true-bond tokens to all distinct canonical atom pairs supported by the same Trimer states:",
        "",
        "- `src/dataset/mts_star_rbf_v2.py`: `prepare_topology`, `prepare_trimer`, canonical-to-Trimer mapping and `state_to_local` validation.",
        "- `src/dataset/periodic_line_glt.py`: `canonical_line_token`, `_token_instances`, `_distance`, and `build_periodic_line_sample`.",
        "- `src/dataset/canonical_periodic.py`: `_neighbors` lifted periodic chemical topology for SPD.",
        "",
        "No minimum-image, central-anchor, finite central-RU, or new conformer reduction was introduced. Relation distance is the existing observation mean; variance is population variance (`ddof=0`); count is the number of current observation slots.",
        "",
        "## Bond reproduction gate",
        "",
        f"- valid formal samples: {bond_summary.get('valid_sample_count', 0)} / sampled rows {bond_summary.get('sampled_row_count', 0)}",
        f"- raw distance mean/std/range: {bond_summary['raw_observation_statistics'].get('mean')} / {bond_summary['raw_observation_statistics'].get('std')} / {bond_summary['raw_observation_statistics'].get('min')}–{bond_summary['raw_observation_statistics'].get('max')} Å",
        f"- observation count histogram: `{bond_summary.get('observation_count_histogram', {})}`",
        f"- count contract `3 - abs_shift`: `{bond_summary['checks'].get('count_equals_3_minus_abs_shift')}`",
        f"- sidecar/builder reproduction: `{bond_summary.get('sidecar_mismatch_count', 0) == 0}`",
        f"- gate: **{'PASS' if bond_summary.get('pass') else 'FAIL'}**",
        "",
        "All-pair interpretation is permitted only because this gate passed." if bond_summary.get("pass") else "All-pair interpretation is forbidden because this gate failed.",
        f"- all-pair true-bond reproduction mismatch count: `{decision_detail.get('evidence', {}).get('all_pair_true_bond_reproduction_mismatch', decision_detail.get('all_pair_true_bond_reproduction_mismatch', 0))}`",
        f"- all-pair true-bond gate: **{'PASS' if decision_detail.get('evidence', {}).get('all_pair_true_bond_reproduction_mismatch', decision_detail.get('all_pair_true_bond_reproduction_mismatch', 0)) == 0 else 'FAIL'}**",
        "",
        "## Scope summary",
        "",
        "| scope | samples | valid | canonical relations | raw observations | duplicates removed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ALL_SCOPES:
        r = valid.get(scope, {})
        lines.append(
            f"| {scope} | {r.get('sample_count', 0)} | {r.get('valid_sample_count', 0)} | {r.get('canonical_relation_count', 0)} | {r.get('raw_observation_count', 0)} | {r.get('duplicates_removed', 0)} |"
        )
    lines += [
        "",
        "The `FINETUNE_<TASK>` rows above are the independent per-task scopes; `FINETUNE_ALL` is their row aggregate and `FINETUNE_UNIQUE` is the first-seen unique-polymer aggregate.",
        "",
        "## Top 10 numbers",
        "",
        f"1. Pretrain valid samples: **{valid.get('PI1M_10K', {}).get('valid_sample_count', 0)}**.",
        f"2. Fine-tune valid rows: **{valid.get('FINETUNE_ALL', {}).get('valid_sample_count', 0)}**; unique polymers: **{valid.get('FINETUNE_UNIQUE', {}).get('valid_sample_count', 0)}**.",
        f"3. Pretrain SPD≥4 relation fraction: **{pre4.get('spd_ge4_relation_count_d_lt_r', 0) / max(1, valid.get('PI1M_10K', {}).get('canonical_relation_count', 0)):.6f}** at d<4 Å; all-distance fraction is in `distance_summary_by_spd.csv`.",
        f"4. SPD≥4 among d<3 Å: **{row('PI1M_10K', 3.0).get('spd_ge4_given_d_lt_r', 0.0):.6f}**.",
        f"5. SPD≥4 among d<4 Å: **{pre4.get('spd_ge4_given_d_lt_r', 0.0):.6f}**.",
        f"6. SPD≥4 among d<5 Å: **{pre5.get('spd_ge4_given_d_lt_r', 0.0):.6f}**.",
        f"7. Pretrain polymer fraction with ≥1 SPD≥4 contact at d<4 Å: **{pre_density.get('fraction_N_NB_ge_1', 0.0)}**.",
        f"8. Median SPD≥4 contacts/polymer at d<4 Å: **{pre_density.get('N_NB_median', 0.0)}**.",
        f"9. Same-RU vs cross-RU candidate composition at d<4 Å: **{pre4.get('same_ru_candidate_fraction', 0.0)} / {pre4.get('cross_ru_candidate_fraction', 0.0)}**.",
        f"10. SPD≥4,d<4 Å variance: median **{pre_var.get('variance_median_A2')} Å²**, p90 **{pre_var.get('variance_p90_A2')} Å²**, high-variance fraction (≥0.25 Å²) **{pre_var.get('fraction_high_variance_ge_0p25_A2')}**.",
        "",
        "## Multi-scale conclusion",
        "",
        "The fixed 2/3/4 Å comparison and the non-overlapping shell composition are in `cutoff_composition.csv` and `distance_shell_composition.csv`. The percentile p20/p30/p40/p50/p60 summaries are in `scale_percentile_summary.csv`.",
        "",
        f"Decision evidence: `{json.dumps(decision_detail.get('evidence', {}), sort_keys=True)}`.",
        "",
        "Nested cutoffs are reported for direct comparison, but incremental shells are the scientifically cleaner view of what is newly added when the radius increases. Percentiles are auxiliary diagnostics only and do not alter the current distance definition.",
        "",
        "## Duplicate and observation semantics",
        "",
        "Raw Trimer state pairs are grouped with the same `canonical_line_token`; translation-equivalent observations remain in the relation mean/variance/count, while the report separately counts raw observations, canonical relation keys, and `raw - canonical` duplicate occurrences. Exact duplicate states are not paired, but distinct periodic states of one canonical atom are retained as formal periodic self relations. Same-RU means `shift==0`; all nonzero shifts are retained as cross-RU/periodic, with `abs_shift>=2` shown explicitly through count tables.",
        "",
        "## Reproducibility",
        "",
        f"- script: `scripts/audit_mts_glt_v2_distance_information_v1.py`",
        f"- manifest: `{manifest.get('manifest_path', 'run_manifest.json')}`",
        f"- sample ids: `{manifest.get('sample_ids_path', 'sample_ids.json')}`",
        "- output tables: `dataset_summary.csv`, `distance_summary_by_spd.csv`, `cutoff_composition.csv`, `distance_shell_composition.csv`, `polymer_contact_density.csv`, `variance_summary.csv`, `count_summary.csv`, `scale_percentile_summary.csv`.",
        "- figures: `all_distance_distribution.png`, `distance_by_spd.png`, `spd_ge4_distance_distribution.png`, `cutoff_composition.png`, `distance_shell_composition.png`, `contacts_per_polymer.png`, `density_vs_ru_size.png`, `variance_vs_distance.png`, `fixed_cutoff_vs_percentile.png`.",
        "",
        "## Limitations",
        "",
        "This audit establishes observed distance/topology composition and does not establish downstream predictive utility. The all-pair relation distance distribution is relation-level mean distance, not a new model input and not a claim that every relation should be encoded.",
        "",
    ]
    return "\n".join(lines)


def run(args) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.pretrain_config).resolve()
    start_manifest = {
        "schema": "mts-glt-v2-distance-information-audit-v1",
        "script": str(Path(__file__).resolve()),
        "cwd": str(ROOT),
        "training_executed": False,
        "seed": int(args.seed),
        "pretrain_requested_rows": int(args.pretrain_size),
        "pretrain_config": str(config_path),
        "downstream_tasks": list(TASKS),
        "distance_pipeline": {
            "prepare_topology": "src/dataset/mts_star_rbf_v2.py",
            "prepare_trimer": "src/dataset/mts_star_rbf_v2.py",
            "canonical_line_token": "src/dataset/periodic_line_glt.py",
            "token_instances": "src/dataset/periodic_line_glt.py",
            "distance": "src/dataset/periodic_line_glt.py",
            "spd_neighbors": "src/dataset/canonical_periodic.py:_neighbors",
        },
    }
    atomic_json(output / "run_manifest.json", start_manifest)

    print("loading formal PI1M_v2 dataset", flush=True)
    pretrain, parsed = formal_dataset_from_config(config_path)
    if int(args.pretrain_size) > len(pretrain):
        raise RuntimeError(f"requested {args.pretrain_size} rows but dataset has {len(pretrain)}")
    rng = np.random.default_rng(int(args.seed))
    sampled_ids = rng.choice(len(pretrain), size=int(args.pretrain_size), replace=False).astype(np.int64)
    sample_payload = {
        "schema": "mts-glt-v2-distance-information-sample-v1",
        "dataset": str(parsed.dataset_name),
        "dataset_rows": int(len(pretrain)),
        "seed": int(args.seed),
        "method": "numpy.default_rng(seed).choice(size, replace=False)",
        "replace": False,
        "sampled_row_ids": sampled_ids.tolist(),
    }
    atomic_json(output / "sample_ids.json", sample_payload)
    start_manifest["sample_ids_path"] = str(output / "sample_ids.json")

    # Phase 1 is intentionally bond-only.  No all-pair interpretation is
    # allowed until this formal current-distance gate has passed.
    print("phase 1/2: reproducing formal true-bond distances", flush=True)
    bond = BondAccumulator()
    for position, row_id in enumerate(sampled_ids.tolist(), start=1):
        item = pretrain[int(row_id)]
        result = formal_true_sample(item)
        bond.add_formal(result)
        if position % 1000 == 0:
            print(f"  bond {position}/{len(sampled_ids)}", flush=True)
    bond_summary = bond.summary(len(sampled_ids))
    atomic_json(output / "bond_reproduction.json", bond_summary)
    if not bond_summary.get("pass", False):
        scope_rows = [{
            "scope": "PI1M_10K",
            "sample_count": len(sampled_ids),
            "valid_sample_count": bond_summary.get("valid_sample_count", 0),
            "invalid_sample_count": bond_summary.get("invalid_or_unusable_sample_count", 0),
            "gate": "FAIL",
        }]
        atomic_csv(output / "dataset_summary.csv", scope_rows)
        manifest = dict(start_manifest)
        manifest.update({"status": "STOP_BOND_REPRODUCTION_FAILED", "bond_reproduction": str(output / "bond_reproduction.json")})
        atomic_json(output / "run_manifest.json", manifest)
        report = "# MTS-GLT-v2 zero-training distance information audit\n\n## Decision: STOP\n\nThe formal true-bond reproduction gate failed. All-pair statistics were not interpreted. See `bond_reproduction.json` for the failed checks.\n"
        (output / "REPORT.md").write_text(report, encoding="utf-8")
        print("STOP: formal true-bond reproduction gate failed", flush=True)
        return 2

    print("phase 2/2: all-pair extraction", flush=True)
    accumulators = {scope: ScopeAccumulator(scope) for scope in ALL_SCOPES}
    task_keys: dict[str, dict[str, dict]] = {task: {} for task in TASKS}
    unique_seen = set()
    validation_mismatch = Counter()
    pretrain_bond_validation = Counter()

    for position, row_id in enumerate(sampled_ids.tolist(), start=1):
        item = pretrain[int(row_id)]
        result, formal_result = process_scope_item(item, int(row_id), accumulators["PI1M_10K"])
        if result.get("reason") != "ok":
            validation_mismatch[result.get("reason", "unknown")] += 1
        if position % 1000 == 0:
            print(f"  PI1M all-pair {position}/{len(sampled_ids)}", flush=True)

    downstream_datasets = {}
    for task in TASKS:
        print(f"loading downstream {task}", flush=True)
        downstream_datasets[task] = downstream_dataset(task, Path(args.downstream_sidecar).resolve())
    downstream_row_counter = 0
    for task in TASKS:
        dataset = downstream_datasets[task]
        task_scope = accumulators[f"FINETUNE_{task.upper()}"]
        for index in range(len(dataset)):
            item = dataset[index]
            downstream_row_counter += 1
            result, formal_result = process_scope_item(item, index, accumulators["FINETUNE_ALL"])
            # The all-row accumulator performs the extraction once.  Mirror
            # its validated result into the task-specific scope so the audit
            # exposes per-task tables without a second geometry pass.
            if result.get("valid"):
                task_scope.add(result, index)
            else:
                task_scope.add_invalid(
                    getattr(item, "canonical_atom_count", 0),
                    result.get("reason", "invalid"),
                )
            if result.get("reason") != "ok":
                validation_mismatch[result.get("reason", "unknown")] += 1
            if result.get("sample_key"):
                key = result["sample_key"]
                task_keys[task].setdefault(key, {"rows": 0, "smiles": str(item.smiles)})["rows"] += 1
                if key not in unique_seen:
                    unique_seen.add(key)
                    if result.get("valid"):
                        accumulators["FINETUNE_UNIQUE"].add(result, index)
                    else:
                        accumulators["FINETUNE_UNIQUE"].add_invalid(getattr(item, "canonical_atom_count", 0), result.get("reason", "invalid"))
            if downstream_row_counter % 1000 == 0:
                print(f"  downstream all-pair {downstream_row_counter}", flush=True)

    scope_rows = [accumulators[scope].summary_row() for scope in ALL_SCOPES]
    distance_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].distance_spd_rows()]
    cutoff_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].cutoff_rows()]
    shell_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].shell_rows()]
    density_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].polymer_density_rows()]
    variance_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].variance_rows()]
    count_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].count_rows()]
    scale_rows = [row for scope in ALL_SCOPES for row in accumulators[scope].scale_rows()]

    overlap = write_polymer_overlap(output, task_keys)
    decision, decision_detail = choose_decision(
        bond_summary,
        cutoff_rows,
        density_rows,
        variance_rows,
        scope_rows,
        validation_mismatch,
    )
    manifest = dict(start_manifest)
    manifest.update({
        "status": "COMPLETED",
        "output_dir": str(output),
        "downstream_row_count": downstream_row_counter,
        "bond_reproduction": str(output / "bond_reproduction.json"),
        "validation_mismatch_counts": dict(validation_mismatch),
        "decision": decision,
        "decision_detail": decision_detail,
        "sample_ids_path": str(output / "sample_ids.json"),
        "manifest_path": str(output / "run_manifest.json"),
    })
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "decision.json", {"decision": decision, **decision_detail})
    atomic_json(output / "downstream_overlap_summary.json", overlap)
    atomic_json(output / "validation_summary.json", {
        "bond_reproduction_pass": bool(bond_summary.get("pass")),
        "all_pair_true_bond_reproduction_pass": int(
            validation_mismatch.get("all_pair_true_bond_reproduction_mismatch", 0)
        ) == 0,
        "all_pair_validation_mismatch_counts": dict(validation_mismatch),
        "training_executed": False,
    })

    atomic_csv(output / "dataset_summary.csv", scope_rows)
    atomic_csv(output / "distance_summary_by_spd.csv", distance_rows)
    atomic_csv(output / "cutoff_composition.csv", cutoff_rows)
    atomic_csv(output / "distance_shell_composition.csv", shell_rows)
    atomic_csv(output / "polymer_contact_density.csv", density_rows)
    atomic_csv(output / "variance_summary.csv", variance_rows)
    atomic_csv(output / "count_summary.csv", count_rows)
    atomic_csv(output / "scale_percentile_summary.csv", scale_rows)
    plot_outputs(output, accumulators, shell_rows, cutoff_rows)
    report_manifest = {
        "manifest_path": str(output / "run_manifest.json"),
        "sample_ids_path": str(output / "sample_ids.json"),
    }
    (output / "REPORT.md").write_text(
        report_text(
            output,
            decision,
            decision_detail,
            bond_summary,
            scope_rows,
            cutoff_rows,
            density_rows,
            variance_rows,
            count_rows,
            overlap,
            report_manifest,
        ),
        encoding="utf-8",
    )
    print(f"completed: {output}", flush=True)
    print(f"decision: {decision}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrain-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--downstream-sidecar", type=Path, default=DEFAULT_DOWNSTREAM_SIDECAR)
    parser.add_argument("--pretrain-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.pretrain_size <= 0:
        parser.error("--pretrain-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
