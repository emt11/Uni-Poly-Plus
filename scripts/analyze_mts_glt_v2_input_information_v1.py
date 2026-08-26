#!/usr/bin/env python3
"""Read-only input-information audit for the formal MTS-GLT-v2 Base-5k.

The audit reads the frozen periodic-line sidecars and the selected 5k probe.
It does not instantiate a training loop, change model inputs, or write caches.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.periodic_line_glt import PeriodicLineGLTSidecar  # noqa: E402
from src.modules.mts_glt_v2 import MTSGraphLineModelV2  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
OCCURRENCE_DTYPE = np.dtype([
    ("sample_id", "<i8"), ("relation_index", "<i4"),
    ("observation_slot", "i1"), ("cosine", "<f4"), ("angle", "<f4"),
    ("d_target", "<f4"), ("d_source", "<f4"),
    ("target_outer_z", "<i2"), ("center_z", "<i2"),
    ("source_outer_z", "<i2"),
])


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    result = {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
    result.update({f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES})
    return result


def histogram(values) -> dict:
    counts = Counter(int(value) for value in np.asarray(values).reshape(-1))
    return {str(key): int(counts[key]) for key in sorted(counts)}


def multiplicity_summary(values) -> dict:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    total = max(1, int(values.size))
    return {
        "count": int(values.size),
        "histogram": histogram(values),
        "fraction_count_1": float(np.count_nonzero(values == 1) / total),
        "fraction_count_2": float(np.count_nonzero(values == 2) / total),
        "fraction_count_ge_3": float(np.count_nonzero(values >= 3) / total),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p25": float(np.quantile(values, 0.25)) if values.size else None,
        "p75": float(np.quantile(values, 0.75)) if values.size else None,
        "max": int(values.max()) if values.size else None,
    }


def variance_norm_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    total = max(1, int(values.size))
    result = finite_summary(values)
    result.update({
        "fraction_exact_zero": float(np.count_nonzero(values == 0.0) / total),
        "fraction_norm_lt_1e_8": float(np.count_nonzero(values < 1e-8) / total),
        "fraction_norm_lt_1e_6": float(np.count_nonzero(values < 1e-6) / total),
        "fraction_norm_lt_1e_4": float(np.count_nonzero(values < 1e-4) / total),
    })
    return result


def _load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "mts-glt-v2-probe-v1" or int(payload.get("step", -1)) != 5000:
        raise RuntimeError("input audit requires the formal GLT-v2 5k probe")
    model = MTSGraphLineModelV2(
        glt_layers=int(payload["glt_layers"]),
        glt_attention_variant=str(payload["glt_attention_variant"]),
        use_compact19=False,
    )
    state = {
        key[len("model."):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval(), payload


def _cohort_root(name: str) -> Path:
    base = ROOT / "data/processed/mips_trimer_scage/cohorts" / name
    current = json.loads((base / "current.json").read_text(encoding="utf-8"))
    return base / str(current["cohort_hash"])


def _task_rows(sidecar: PeriodicLineGLTSidecar, task: str) -> np.ndarray:
    keys = np.load(_cohort_root(f"smi_{task}") / "sample_keys.npy", mmap_mode="r")
    unique = sorted({bytes(np.asarray(row, dtype=np.uint8)) for row in keys})
    return np.asarray([sidecar.index_for_key(key) for key in unique], dtype=np.int64)


def _token_incidence_options(tokens: dict, token_index: int, center: int):
    token_a = int(tokens["token_atom_a"][token_index])
    token_b = int(tokens["token_atom_b"][token_index])
    shift = int(tokens["token_shift"][token_index])
    options = []
    if int(center) == token_a:
        options.append((token_b, shift))
    if int(center) == token_b:
        options.append((token_a, -shift))
    if not options:
        raise ValueError("relation center is not incident to line token")
    return sorted(set(options))


def _token_distance_at_center(
    distances: np.ndarray,
    count: int,
    token_a: int,
    token_b: int,
    shift: int,
    center: int,
    center_translation: int,
    outer_offset: int,
) -> float:
    if int(center) == int(token_a) and int(outer_offset) == int(shift):
        q_a = int(center_translation)
    elif int(center) == int(token_b) and int(outer_offset) == -int(shift):
        q_a = int(center_translation) - int(shift)
    else:
        raise ValueError("outer offset does not identify a token incidence")
    translations = [
        value for value in range(-1, 2)
        if -1 <= value + int(shift) <= 1
    ]
    if q_a not in translations:
        raise ValueError("relation occurrence is outside token observation slots")
    position = translations.index(q_a)
    if position >= int(count):
        raise ValueError("relation occurrence points to an invalid distance slot")
    return float(distances[position])


def relation_occurrences(tokens: dict, relations: dict, sample_id: int):
    rows = []
    relation_rank = Counter()
    for relation_index in range(len(relations["relation_source"])):
        if bool(relations["relation_is_fallback"][relation_index]) or not bool(
            relations["relation_valid"][relation_index]
        ):
            continue
        source = int(relations["relation_source"][relation_index])
        target = int(relations["relation_target"][relation_index])
        center = int(relations["relation_center_atom"][relation_index])
        source_options = _token_incidence_options(tokens, source, center)
        target_options = _token_incidence_options(tokens, target, center)
        candidates = {}
        for source_position, (source_outer, source_offset) in enumerate(source_options):
            for target_position, (target_outer, target_offset) in enumerate(target_options):
                if source == target and source_position >= target_position:
                    continue
                direct = (source_outer, source_offset, center, target_outer, target_offset)
                inverse = (target_outer, target_offset, center, source_outer, source_offset)
                geometry_key = min(direct, inverse)
                candidates.setdefault(
                    geometry_key,
                    (source_outer, source_offset, target_outer, target_offset),
                )
        ordered_candidates = [candidates[key] for key in sorted(candidates)]
        duplicate_key = (source, target, center)
        candidate_index = relation_rank[duplicate_key]
        relation_rank[duplicate_key] += 1
        if candidate_index >= len(ordered_candidates):
            raise ValueError("stored relation duplicates exceed derived geometry keys")
        source_outer, source_offset, target_outer, target_offset = ordered_candidates[
            candidate_index
        ]
        translations = [
            value for value in range(-1, 2)
            if -1 <= value + source_offset <= 1
            and -1 <= value + target_offset <= 1
        ]
        count = int(relations["relation_observation_count"][relation_index])
        if len(translations) != int(relations["relation_multiplicity"][relation_index]):
            raise ValueError("derived relation occurrence count differs from multiplicity")
        if count != len(translations):
            raise ValueError("valid relation has incomplete angle observations")
        for slot, translation in enumerate(translations):
            d_source = _token_distance_at_center(
                tokens["token_observation_distances"][source],
                tokens["token_observation_count"][source],
                tokens["token_atom_a"][source], tokens["token_atom_b"][source],
                tokens["token_shift"][source], center, translation, source_offset,
            )
            d_target = _token_distance_at_center(
                tokens["token_observation_distances"][target],
                tokens["token_observation_count"][target],
                tokens["token_atom_a"][target], tokens["token_atom_b"][target],
                tokens["token_shift"][target], center, translation, target_offset,
            )
            angle = float(relations["relation_observation_angles"][relation_index, slot])
            rows.append({
                "sample_id": sample_id,
                "relation_index": int(relation_index),
                "observation_slot": int(slot),
                "cosine": float(math.cos(angle)),
                "angle": angle,
                "d_target": d_target,
                "d_source": d_source,
                "target_outer_z": int(tokens["token_endpoint_z_a"][target])
                if int(tokens["token_atom_a"][target]) == target_outer
                else int(tokens["token_endpoint_z_b"][target]),
                "center_z": int(tokens["token_endpoint_z_a"][target])
                if int(tokens["token_atom_a"][target]) == center
                else int(tokens["token_endpoint_z_b"][target]),
                "source_outer_z": int(tokens["token_endpoint_z_a"][source])
                if int(tokens["token_atom_a"][source]) == source_outer
                else int(tokens["token_endpoint_z_b"][source]),
            })
    if not rows:
        return np.empty(0, dtype=OCCURRENCE_DTYPE)
    return np.asarray([
        tuple(row[name] for name in OCCURRENCE_DTYPE.names) for row in rows
    ], dtype=OCCURRENCE_DTYPE)


def _encode_moments(module, observations, counts, z_a=None, z_b=None, chunk=65536):
    means, variances = [], []
    with torch.inference_mode():
        for start in range(0, len(counts), int(chunk)):
            stop = min(len(counts), start + int(chunk))
            kwargs = {}
            if z_a is not None:
                kwargs = {
                    "z_a": torch.from_numpy(np.asarray(z_a[start:stop], dtype=np.int64)),
                    "z_b": torch.from_numpy(np.asarray(z_b[start:stop], dtype=np.int64)),
                }
            mean, variance = module(
                torch.from_numpy(np.asarray(observations[start:stop], dtype=np.float32)),
                torch.from_numpy(np.asarray(counts[start:stop], dtype=np.int64)),
                **kwargs,
            )
            means.append(mean.cpu().numpy())
            variances.append(variance.cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def _collect_scope(sidecar, rows: np.ndarray, model, *, collect_occurrences=False):
    token_parts = defaultdict(list)
    relation_parts = defaultdict(list)
    occurrence_chunks = []
    for offset, row_index in enumerate(np.asarray(rows, dtype=np.int64)):
        row = sidecar.qc_row(int(row_index))
        tokens, relations = row["tokens"], row["relations"]
        token_cross = np.abs(tokens["token_shift"]) > 0
        real = ~relations["relation_is_fallback"].astype(bool)
        relation_cross = np.zeros(len(real), dtype=bool)
        if np.any(real):
            relation_cross[real] = (
                token_cross[relations["relation_source"][real]]
                | token_cross[relations["relation_target"][real]]
            )
        for name, value in tokens.items():
            token_parts[name].append(np.asarray(value))
        token_parts["token_cross"].append(token_cross)
        for name, value in relations.items():
            relation_parts[name].append(np.asarray(value))
        relation_parts["relation_cross"].append(relation_cross)
        if collect_occurrences:
            occurrence_chunks.append(relation_occurrences(
                tokens, relations, int(row_index)
            ))
        if offset and offset % 2500 == 0:
            print(f"collected rows={offset}/{len(rows)}", flush=True)
    tokens = {name: np.concatenate(parts, axis=0) for name, parts in token_parts.items()}
    relations = {name: np.concatenate(parts, axis=0) for name, parts in relation_parts.items()}
    token_valid = tokens["token_valid"].astype(bool)
    real_valid = (
        ~relations["relation_is_fallback"].astype(bool)
        & relations["relation_valid"].astype(bool)
    )
    _, distance_variance = _encode_moments(
        model.glt.distance_basis,
        tokens["token_observation_distances"][token_valid],
        tokens["token_observation_count"][token_valid],
        tokens["token_endpoint_z_a"][token_valid],
        tokens["token_endpoint_z_b"][token_valid],
    )
    _, angle_variance = _encode_moments(
        model.glt.angle_basis,
        relations["relation_observation_angles"][real_valid],
        relations["relation_observation_count"][real_valid],
    )
    distance_variance_norm = np.linalg.norm(distance_variance, axis=1)
    angle_variance_norm = np.linalg.norm(angle_variance, axis=1)
    valid_tokens = {name: value[token_valid] for name, value in tokens.items()}
    valid_relations = {name: value[real_valid] for name, value in relations.items()}
    valid_tokens["distance_variance_norm"] = distance_variance_norm
    valid_relations["angle_variance_norm"] = angle_variance_norm
    occurrences = (
        np.concatenate(occurrence_chunks)
        if occurrence_chunks else np.empty(0, dtype=OCCURRENCE_DTYPE)
    )
    return valid_tokens, valid_relations, occurrences


def _flatten_observations(values, counts):
    values = np.asarray(values)
    counts = np.asarray(counts, dtype=np.int64)
    mask = np.arange(values.shape[1])[None, :] < counts[:, None]
    return values[mask]


def _raw_distance_report(tokens):
    observations = _flatten_observations(
        tokens["token_observation_distances"], tokens["token_observation_count"]
    )
    internal = ~tokens["token_cross"]
    report = {
        "all": finite_summary(observations),
        "internal_true_bond": finite_summary(_flatten_observations(
            tokens["token_observation_distances"][internal],
            tokens["token_observation_count"][internal],
        )),
        "cross_ru_true_bond": finite_summary(_flatten_observations(
            tokens["token_observation_distances"][~internal],
            tokens["token_observation_count"][~internal],
        )),
        "observation_multiplicity": {
            "all": multiplicity_summary(tokens["token_observation_count"]),
            "internal": multiplicity_summary(tokens["token_observation_count"][internal]),
            "cross_ru": multiplicity_summary(tokens["token_observation_count"][~internal]),
        },
        "encoded_variance_norm": {
            "all": variance_norm_summary(tokens["distance_variance_norm"]),
            "internal": variance_norm_summary(tokens["distance_variance_norm"][internal]),
            "cross_ru": variance_norm_summary(tokens["distance_variance_norm"][~internal]),
        },
    }
    pairs = defaultdict(list)
    for index, count in enumerate(tokens["token_observation_count"]):
        pair = tuple(sorted((
            int(tokens["token_endpoint_z_a"][index]),
            int(tokens["token_endpoint_z_b"][index]),
        )))
        pairs[pair].extend(tokens["token_observation_distances"][index, :int(count)].tolist())
    pair_rows = []
    for pair, values in sorted(pairs.items(), key=lambda item: len(item[1]), reverse=True):
        if len(values) < 1000 or len(pair_rows) >= 10:
            continue
        pair_rows.append({"endpoint_atomic_numbers": list(pair), **finite_summary(values)})
    report["top_endpoint_pairs_min_1000"] = pair_rows
    return report


def _raw_angle_report(relations):
    angles = _flatten_observations(
        relations["relation_observation_angles"], relations["relation_observation_count"]
    )
    internal = ~relations["relation_cross"]
    return {
        "all_radians": finite_summary(angles),
        "all_cosine": finite_summary(np.cos(angles)),
        "internal_only_radians": finite_summary(_flatten_observations(
            relations["relation_observation_angles"][internal],
            relations["relation_observation_count"][internal],
        )),
        "cross_ru_involved_radians": finite_summary(_flatten_observations(
            relations["relation_observation_angles"][~internal],
            relations["relation_observation_count"][~internal],
        )),
        "observation_count": {
            "all": multiplicity_summary(relations["relation_observation_count"]),
            "internal": multiplicity_summary(relations["relation_observation_count"][internal]),
            "cross_ru": multiplicity_summary(relations["relation_observation_count"][~internal]),
        },
        "stored_multiplicity": {
            "all": multiplicity_summary(relations["relation_multiplicity"]),
            "internal": multiplicity_summary(relations["relation_multiplicity"][internal]),
            "cross_ru": multiplicity_summary(relations["relation_multiplicity"][~internal]),
        },
        "encoded_variance_norm": {
            "all": variance_norm_summary(relations["angle_variance_norm"]),
            "internal": variance_norm_summary(relations["angle_variance_norm"][internal]),
            "cross_ru": variance_norm_summary(relations["angle_variance_norm"][~internal]),
        },
    }


def _metadata_report(tokens, relations):
    shift = np.abs(tokens["token_shift"].astype(np.int64))
    cross = tokens["token_cross"].astype(bool)
    nonzero = shift > 0
    shift_report = {
        "histogram": histogram(shift),
        "fraction_zero": float(np.mean(shift == 0)),
        "fraction_nonzero": float(np.mean(nonzero)),
        "internal_histogram": histogram(shift[~cross]),
        "cross_ru_histogram": histogram(shift[cross]),
        "p_nonzero_given_cross_ru": float(np.mean(nonzero[cross])) if np.any(cross) else None,
        "p_cross_ru_given_nonzero": float(np.mean(cross[nonzero])) if np.any(nonzero) else None,
    }
    fields = {
        "distance_count": (tokens["token_observation_count"], cross),
        "distance_encoded_variance_norm": (tokens["distance_variance_norm"], cross),
        "angle_count": (relations["relation_observation_count"], relations["relation_cross"]),
        "angle_multiplicity": (relations["relation_multiplicity"], relations["relation_cross"]),
        "angle_encoded_variance_norm": (relations["angle_variance_norm"], relations["relation_cross"]),
        "abs_shift": (shift, cross),
    }
    table = []
    for name, (values, cross_mask) in fields.items():
        values = np.asarray(values)
        cross_mask = np.asarray(cross_mask, dtype=bool)
        discrete = np.issubdtype(values.dtype, np.integer)
        if discrete:
            counts = np.asarray(list(Counter(values.tolist()).values()), dtype=np.float64)
            probabilities = counts / counts.sum()
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            diversity = int(len(counts))
        else:
            entropy = None
            diversity = int(np.unique(np.round(values, 8)).size)
        table.append({
            "input_metadata": name,
            "support": histogram(values) if discrete else finite_summary(values),
            "entropy_bits": entropy,
            "diversity": diversity,
            "largest_value_fraction": float(max(Counter(values.tolist()).values()) / len(values))
            if discrete and len(values) else None,
            "internal_summary": histogram(values[~cross_mask]) if discrete else finite_summary(values[~cross_mask]),
            "cross_ru_summary": histogram(values[cross_mask]) if discrete else finite_summary(values[cross_mask]),
            "notes": "descriptive only; no mechanical threshold applied",
        })
    return {
        "abs_shift": shift_report,
        "exact_redundancies": {
            "distance_count_equals_3_minus_abs_shift": bool(np.array_equal(
                tokens["token_observation_count"].astype(np.int64), 3 - shift
            )),
            "angle_count_equals_stored_multiplicity": bool(np.array_equal(
                relations["relation_observation_count"], relations["relation_multiplicity"]
            )),
        },
        "redundancy_table": table,
    }


def _observation_encoding(module, observations, *, z_a=None, z_b=None, chunk=65536):
    observations = np.asarray(observations, dtype=np.float32).reshape(-1)
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(observations), chunk):
            stop = min(len(observations), start + chunk)
            values = torch.from_numpy(observations[start:stop, None])
            counts = torch.ones(stop - start, dtype=torch.long)
            kwargs = {}
            if z_a is not None:
                kwargs = {
                    "z_a": torch.from_numpy(np.asarray(z_a[start:stop], dtype=np.int64)),
                    "z_b": torch.from_numpy(np.asarray(z_b[start:stop], dtype=np.int64)),
                }
            mean, _ = module(values, counts, **kwargs)
            encoded.append(mean.cpu().numpy())
    return np.concatenate(encoded)


def _activation_statistics(encoded: np.ndarray, centers, widths, raw_values) -> dict:
    encoded = np.asarray(encoded, dtype=np.float64)
    means = encoded.mean(axis=0)
    stds = encoded.std(axis=0)
    covariance = np.cov(encoded, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues[eigenvalues > 0] / total if total > 0 else np.asarray([])
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if probabilities.size else 0.0
    participation = float(total * total / np.square(eigenvalues).sum()) if np.square(eigenvalues).sum() > 0 else 0.0
    safe = stds > 1e-12
    standardized = np.zeros_like(encoded)
    standardized[:, safe] = (encoded[:, safe] - means[safe]) / stds[safe]
    correlation = standardized.T @ standardized / max(1, len(encoded))
    upper = np.triu_indices(encoded.shape[1], k=1)
    pair_values = np.abs(correlation[upper])
    top = np.argsort(pair_values)[-20:][::-1]
    top_pairs = [{
        "channel_a": int(upper[0][index]),
        "channel_b": int(upper[1][index]),
        "absolute_correlation": float(pair_values[index]),
    } for index in top]
    maximum = np.max(np.abs(encoded), axis=1)
    low_cutoff = float(np.quantile(maximum, 0.01))
    channel_rows = [{
        "channel": int(index),
        "mean": float(means[index]),
        "std": float(stds[index]),
        "p01": float(np.quantile(encoded[:, index], 0.01)),
        "p50": float(np.quantile(encoded[:, index], 0.50)),
        "p99": float(np.quantile(encoded[:, index], 0.99)),
    } for index in range(encoded.shape[1])]
    return {
        "channels": int(encoded.shape[1]),
        "observations": int(encoded.shape[0]),
        "channel_statistics": channel_rows,
        "near_constant_channels_std_lt_1e_6": int(np.count_nonzero(stds < 1e-6)),
        "near_constant_channels_std_lt_1e_4": int(np.count_nonzero(stds < 1e-4)),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "fraction_pairs_abs_corr_gt_0_95": float(np.mean(pair_values > 0.95)),
        "fraction_pairs_abs_corr_gt_0_99": float(np.mean(pair_values > 0.99)),
        "top_pairwise_absolute_correlations": top_pairs,
        "learned_centers": np.asarray(centers, dtype=np.float64).tolist(),
        "learned_widths": np.asarray(widths, dtype=np.float64).tolist(),
        "raw_distribution": finite_summary(raw_values),
        "center_range": [float(np.min(centers)), float(np.max(centers))],
        "fraction_raw_below_center_range": float(np.mean(raw_values < np.min(centers))),
        "fraction_raw_above_center_range": float(np.mean(raw_values > np.max(centers))),
        "max_activation_lowest_1pct_cutoff": low_cutoff,
        "fraction_low_response_by_empirical_1pct": float(np.mean(maximum <= low_cutoff)),
        "max_activation_distribution": finite_summary(maximum),
    }


def _sample_observations(values, maximum: int, seed: int, *aligned):
    values = np.asarray(values)
    if len(values) <= int(maximum):
        indices = np.arange(len(values))
    else:
        indices = np.sort(np.random.default_rng(int(seed)).choice(
            len(values), size=int(maximum), replace=False
        ))
    return (values[indices],) + tuple(np.asarray(item)[indices] for item in aligned)


def _basis_reports(model, tokens, relations, maximum: int, seed: int):
    distance_values = []
    distance_a = []
    distance_b = []
    for index, count in enumerate(tokens["token_observation_count"]):
        count = int(count)
        distance_values.extend(tokens["token_observation_distances"][index, :count])
        distance_a.extend([tokens["token_endpoint_z_a"][index]] * count)
        distance_b.extend([tokens["token_endpoint_z_b"][index]] * count)
    distance_values, distance_a, distance_b = _sample_observations(
        np.asarray(distance_values), maximum, seed,
        np.asarray(distance_a), np.asarray(distance_b),
    )
    angle_values = _flatten_observations(
        relations["relation_observation_angles"], relations["relation_observation_count"]
    )
    (angle_values,) = _sample_observations(angle_values, maximum, seed)
    distance_encoded = _observation_encoding(
        model.glt.distance_basis, distance_values, z_a=distance_a, z_b=distance_b
    )
    angle_encoded = _observation_encoding(model.glt.angle_basis, angle_values)
    distance_module = model.glt.distance_basis
    angle_module = model.glt.angle_basis
    distance = _activation_statistics(
        distance_encoded,
        distance_module.centers.detach().cpu().numpy(),
        torch.nn.functional.softplus(distance_module.raw_width).detach().cpu().numpy(),
        distance_values,
    )
    affine = distance_module.pair_affine.weight.detach().cpu().numpy()
    with torch.inference_mode():
        active_pair_indices = torch.unique(distance_module._pair_index(
            torch.from_numpy(np.asarray(distance_a, dtype=np.int64)),
            torch.from_numpy(np.asarray(distance_b, dtype=np.int64)),
        )).cpu().numpy()
    active_affine = affine[active_pair_indices]
    distance["atom_type_conditioning"] = {
        "contract": "unordered endpoint pair -> learned affine scale,bias applied to distance; shared learned centers/widths",
        "endpoint_swap_invariant": True,
        "pair_vocab": int(distance_module.pair_vocab),
        "full_pair_table_scale_distribution": finite_summary(affine[:, 0]),
        "full_pair_table_bias_distribution": finite_summary(affine[:, 1]),
        "sampled_observed_pair_types": int(len(active_pair_indices)),
        "sampled_observed_scale_distribution": finite_summary(active_affine[:, 0]),
        "sampled_observed_bias_distribution": finite_summary(active_affine[:, 1]),
        "code_location": "src/modules/periodic_line_glt_v2.py:58-77",
    }
    angle = _activation_statistics(
        angle_encoded,
        angle_module.centers.detach().cpu().numpy(),
        torch.nn.functional.softplus(angle_module.raw_width).detach().cpu().numpy(),
        angle_values,
    )
    return distance, angle


def _radial_ambiguity(occurrences: np.ndarray):
    cosine = np.asarray(occurrences["cosine"], dtype=np.float64)
    target = np.asarray(occurrences["d_target"], dtype=np.float64)
    source = np.asarray(occurrences["d_source"], dtype=np.float64)
    edges = np.linspace(-1.0, 1.0, 41)
    bin_index = np.clip(np.digitize(cosine, edges, right=False) - 1, 0, 39)
    bins = []
    weighted_target = 0.0
    weighted_source = 0.0
    weight = 0
    for index in range(40):
        selected = bin_index == index
        count = int(selected.sum())
        if count < 500:
            continue
        covariance = np.cov(np.stack([target[selected], source[selected]], axis=0))
        row = {
            "bin": int(index), "left": float(edges[index]), "right": float(edges[index + 1]),
            "count": count,
            "std_d_target": float(target[selected].std()),
            "std_d_source": float(source[selected].std()),
            "iqr_d_target": float(np.quantile(target[selected], 0.75) - np.quantile(target[selected], 0.25)),
            "iqr_d_source": float(np.quantile(source[selected], 0.75) - np.quantile(source[selected], 0.25)),
            "joint_radial_covariance": covariance.tolist(),
        }
        bins.append(row)
        weighted_target += count * row["std_d_target"]
        weighted_source += count * row["std_d_source"]
        weight += count
    conditional_target = weighted_target / max(1, weight)
    conditional_source = weighted_source / max(1, weight)
    global_target = float(target.std())
    global_source = float(source.std())
    return {
        "occurrences": int(len(occurrences)),
        "cosine_bins": 40,
        "eligible_bins_min_500": bins,
        "eligible_occurrences": int(weight),
        "weighted_conditional_std_d_target": conditional_target,
        "weighted_conditional_std_d_source": conditional_source,
        "global_std_d_target": global_target,
        "global_std_d_source": global_source,
        "conditional_to_global_ratio_target": conditional_target / max(global_target, 1e-12),
        "conditional_to_global_ratio_source": conditional_source / max(global_source, 1e-12),
    }


def _ambiguity_examples(occurrences: np.ndarray, maximum=20):
    order = np.argsort(occurrences["cosine"], kind="stable")
    selected = []
    used = set()
    for left_position, left in enumerate(order):
        if len(selected) >= maximum:
            break
        first = occurrences[int(left)]
        for right in order[left_position + 1:]:
            second = occurrences[int(right)]
            cosine_gap = abs(first["cosine"] - second["cosine"])
            if cosine_gap >= 0.005:
                break
            radial_gap = max(
                abs(first["d_target"] - second["d_target"]),
                abs(first["d_source"] - second["d_source"]),
            )
            unordered_first = sorted((float(first["d_target"]), float(first["d_source"])))
            unordered_second = sorted((float(second["d_target"]), float(second["d_source"])))
            reverse_duplicate = (
                int(first["sample_id"]) == int(second["sample_id"])
                and np.allclose(unordered_first, unordered_second, rtol=0.0, atol=1e-6)
            )
            if radial_gap <= 0.10 or int(right) in used or reverse_duplicate:
                continue
            used.update((int(left), int(right)))
            selected.append({
                "sample_id_1": int(first["sample_id"]), "sample_id_2": int(second["sample_id"]),
                "relation_1": int(first["relation_index"]), "relation_2": int(second["relation_index"]),
                "cos_theta_1": float(first["cosine"]), "cos_theta_2": float(second["cosine"]),
                "angle_rad_1": float(first["angle"]), "angle_rad_2": float(second["angle"]),
                "d_target_1": float(first["d_target"]), "d_target_2": float(second["d_target"]),
                "d_source_1": float(first["d_source"]), "d_source_2": float(second["d_source"]),
                "context_1": f'{int(first["target_outer_z"])}-{int(first["center_z"])}-{int(first["source_outer_z"])}',
                "context_2": f'{int(second["target_outer_z"])}-{int(second["center_z"])}-{int(second["source_outer_z"])}',
                "cosine_gap": float(cosine_gap), "max_radial_gap": float(radial_gap),
            })
            break
    return selected


def _matrix_diagnostics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    std = values.std(axis=0)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = eigenvalues.sum()
    probabilities = eigenvalues[eigenvalues > 0] / total if total > 0 else np.asarray([])
    correlation = np.corrcoef(values, rowvar=False)
    upper = np.abs(correlation[np.triu_indices(values.shape[1], k=1)])
    return {
        "dimension": int(values.shape[1]),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if probabilities.size else 0.0,
        "condition_number_covariance": float(eigenvalues[-1] / max(eigenvalues[eigenvalues > 1e-12][0], 1e-12)) if np.any(eigenvalues > 1e-12) else None,
        "near_constant_channels_std_lt_1e_6": int(np.count_nonzero(std < 1e-6)),
        "near_constant_channels_std_lt_1e_4": int(np.count_nonzero(std < 1e-4)),
        "fraction_pairs_abs_corr_gt_0_95": float(np.mean(upper > 0.95)),
        "fraction_pairs_abs_corr_gt_0_99": float(np.mean(upper > 0.99)),
    }


def _basis_feasibility(occurrences: np.ndarray, maximum=50000, seed=42):
    try:
        from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer
    except Exception as error:
        return {"status": "SKIPPED", "reason": f"existing PyG implementation unavailable: {error}"}
    if len(occurrences) == 0:
        return {"status": "SKIPPED", "reason": "no valid relation occurrences"}
    indices = np.arange(len(occurrences))
    if len(indices) > maximum:
        indices = np.sort(np.random.default_rng(seed).choice(indices, maximum, replace=False))
    distances = torch.from_numpy(np.asarray(occurrences["d_source"][indices], dtype=np.float32))
    angles = torch.from_numpy(np.asarray(occurrences["angle"][indices], dtype=np.float32))
    cutoff = 5.0
    radial = BesselBasisLayer(6, cutoff, envelope_exponent=5)
    spherical = SphericalBasisLayer(7, 6, cutoff, envelope_exponent=5)
    with torch.inference_mode():
        rbf = radial(distances).numpy()
        sbf = spherical(distances, angles, torch.arange(len(distances))).numpy()
    return {
        "status": "AVAILABLE",
        "implementation": "torch_geometric.nn.models.dimenet",
        "sample_count": int(len(indices)),
        "radial_basis": _matrix_diagnostics(rbf),
        "joint_radial_angular_basis": _matrix_diagnostics(sbf),
        "note": "feasibility only; no model integration or property predictor",
    }


def _o8_trace():
    rows = [
        {
            "feature": "bond_type(single/double/triple/aromatic)",
            "available_in_data": True,
            "encoded": True,
            "present_in_batch": True,
            "passed_to_o8": False,
            "used_in_forward": False,
            "usage_location": "canonical edge_attr/ru_bond_type and GLT masked-line label/QC; O8 forward reads neither",
            "status": "NOT_USED_AS_O8_INPUT",
        },
        {
            "feature": "bond_conjugation",
            "available_in_data": False,
            "encoded": False,
            "present_in_batch": False,
            "passed_to_o8": False,
            "used_in_forward": False,
            "usage_location": "generic graph featurizer can derive it, but formal canonical topology does not preserve a bond-level conjugation tensor",
            "status": "NOT_USED_AS_O8_INPUT",
        },
        {
            "feature": "bond_ring_membership",
            "available_in_data": False,
            "encoded": False,
            "present_in_batch": False,
            "passed_to_o8": False,
            "used_in_forward": False,
            "usage_location": "atom-level MIPS137 includes aromaticity but formal O8 has no bond-ring input",
            "status": "NOT_USED_AS_O8_INPUT",
        },
        {
            "feature": "bond_stereo",
            "available_in_data": False,
            "encoded": False,
            "present_in_batch": False,
            "passed_to_o8": False,
            "used_in_forward": False,
            "usage_location": "generic graph featurizer can derive it; formal canonical O8 forward has no stereo tensor",
            "status": "NOT_USED_AS_O8_INPUT",
        },
        {
            "feature": "bond_existence/topological path",
            "available_in_data": True,
            "encoded": True,
            "present_in_batch": True,
            "passed_to_o8": True,
            "used_in_forward": True,
            "usage_location": "lga_edge_index, lga_spd and lga_path_index drive O8 attention and path-node bias",
            "status": "USED_AS_TOPOLOGY_NOT_CHEMICAL_ATTRIBUTE",
        },
    ]
    return {
        "o8_explicit_bond_chemistry": "NO",
        "features": rows,
        "forward_trace": [
            "canonical_periodic.py builds edge_attr/ru_bond_type plus lifted topology",
            "dataloader.py carries edge_attr and lga_path_bond_hist",
            "MIPSLocalAtomEmbedding consumes only mips_x and mips_backbone_mask",
            "MIPSLocalGraphEncoder consumes lga_spd and atom states along lga_path_index",
            "edge_attr, ru_bond_type and lga_path_bond_hist are absent from O8 numerical forward",
        ],
        "code_locations": {
            "formal_canonical_featurizer": "src/dataset/canonical_periodic.py:419-571",
            "batch": "src/dataset/dataloader.py:143-173,388-403",
            "atom_input": "src/modules/mips_local_graph.py:19-32",
            "attention_input": "src/modules/mips_local_graph.py:474-499",
        },
    }


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    model, checkpoint_payload = _load_model(checkpoint)
    pretrain_sidecar = PeriodicLineGLTSidecar(Path(args.pretrain_sidecar))
    downstream_sidecar = PeriodicLineGLTSidecar(Path(args.downstream_sidecar))
    sample_payload = json.loads(Path(args.sample_ids).read_text(encoding="utf-8"))
    pretrain_rows = np.asarray(sample_payload["sampled_graph_ids"], dtype=np.int64)
    if len(pretrain_rows) != 20000:
        raise RuntimeError("alignment audit sample must contain exactly 20,000 rows")

    print("collecting PRETRAIN-20K", flush=True)
    pre_tokens, pre_relations, occurrences = _collect_scope(
        pretrain_sidecar, pretrain_rows, model, collect_occurrences=True
    )
    distance_scopes = {"PRETRAIN-20K": _raw_distance_report(pre_tokens)}
    angle_scopes = {"PRETRAIN-20K": _raw_angle_report(pre_relations)}
    metadata_scopes = {"PRETRAIN-20K": _metadata_report(pre_tokens, pre_relations)}
    downstream_rows = {}
    for task in TASKS:
        rows = _task_rows(downstream_sidecar, task)
        downstream_rows[task] = int(len(rows))
        print(f"collecting downstream {task}: unique_rows={len(rows)}", flush=True)
        tokens, relations, _ = _collect_scope(downstream_sidecar, rows, model)
        distance_scopes[task] = _raw_distance_report(tokens)
        angle_scopes[task] = _raw_angle_report(relations)
        metadata_scopes[task] = _metadata_report(tokens, relations)

    print("auditing learned bases", flush=True)
    distance_basis, angle_basis = _basis_reports(
        model, pre_tokens, pre_relations, int(args.max_basis_observations), int(args.seed)
    )
    radial = _radial_ambiguity(occurrences)
    examples = _ambiguity_examples(occurrences)
    feasibility = _basis_feasibility(occurrences, seed=int(args.seed))
    bond_trace = _o8_trace()

    atomic_json(output / "o8_bond_feature_trace.json", bond_trace)
    atomic_json(output / "distance_raw_statistics.json", {"scopes": distance_scopes})
    atomic_json(output / "angle_raw_statistics.json", {"scopes": angle_scopes})
    atomic_json(output / "periodic_metadata_statistics.json", {"scopes": metadata_scopes})
    atomic_json(output / "distance_basis_statistics.json", distance_basis)
    atomic_json(output / "angle_basis_statistics.json", angle_basis)
    atomic_json(output / "radial_ambiguity_statistics.json", radial)
    atomic_json(output / "basis_feasibility.json", feasibility)
    example_fields = [
        "sample_id_1", "sample_id_2", "relation_1", "relation_2",
        "cos_theta_1", "cos_theta_2", "angle_rad_1", "angle_rad_2",
        "d_target_1", "d_target_2", "d_source_1", "d_source_2",
        "context_1", "context_2", "cosine_gap", "max_radial_gap",
    ]
    atomic_csv(output / "radial_ambiguity_examples.csv", examples, example_fields)

    pre_meta = metadata_scopes["PRETRAIN-20K"]
    redundant_fields = []
    shift = pre_meta["abs_shift"]
    if shift["p_nonzero_given_cross_ru"] == 1.0 and shift["p_cross_ru_given_nonzero"] == 1.0:
        redundant_fields.append("abs_shift (exact internal/cross-RU structural tag)")
    exact = pre_meta["exact_redundancies"]
    if exact["distance_count_equals_3_minus_abs_shift"]:
        redundant_fields.append("distance_count (exactly 3 - abs_shift)")
    if exact["angle_count_equals_stored_multiplicity"]:
        redundant_fields.append("angle_count/multiplicity (exact duplicates)")
    for entry in pre_meta["redundancy_table"]:
        if entry["largest_value_fraction"] is not None and entry["largest_value_fraction"] >= 0.95:
            label = f'{entry["input_metadata"]} (dominant value fraction {entry["largest_value_fraction"]:.6f})'
            if not any(existing.startswith(entry["input_metadata"] + " ") for existing in redundant_fields):
                redundant_fields.append(label)
    distance_redundancy = (
        distance_basis["near_constant_channels_std_lt_1e_4"] > 0
        or distance_basis["fraction_pairs_abs_corr_gt_0_99"] > 0.25
    )
    angle_redundancy = (
        angle_basis["near_constant_channels_std_lt_1e_4"] > 0
        or angle_basis["fraction_pairs_abs_corr_gt_0_99"] > 0.25
    )
    radial_evidence = max(
        radial["conditional_to_global_ratio_target"],
        radial["conditional_to_global_ratio_source"],
    ) > 0.5 and len(examples) > 0
    summary = {
        "schema": "mts-glt-v2-input-information-audit-v1",
        "baseline": "MTS-GLT-v2-Base-5k",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload["step"]),
        "pretraining_audit_set": "alignment_audit_v1 deterministic 20,000 geometry-valid PI1M_v2 rows",
        "pretraining_rows": int(len(pretrain_rows)),
        "downstream_unique_rows": downstream_rows,
        "o8_explicit_bond_chemistry": bond_trace["o8_explicit_bond_chemistry"],
        "distance": {
            "channels": distance_basis["channels"],
            "effective_rank": distance_basis["effective_rank"],
            "near_constant_channels_std_lt_1e_4": distance_basis["near_constant_channels_std_lt_1e_4"],
            "high_correlation_fraction_gt_0_99": distance_basis["fraction_pairs_abs_corr_gt_0_99"],
            "raw": distance_scopes["PRETRAIN-20K"]["all"],
            "observation_count": distance_scopes["PRETRAIN-20K"]["observation_multiplicity"],
            "encoded_variance": distance_scopes["PRETRAIN-20K"]["encoded_variance_norm"],
        },
        "angle": {
            "channels": angle_basis["channels"],
            "effective_rank": angle_basis["effective_rank"],
            "near_constant_channels_std_lt_1e_4": angle_basis["near_constant_channels_std_lt_1e_4"],
            "high_correlation_fraction_gt_0_99": angle_basis["fraction_pairs_abs_corr_gt_0_99"],
            "raw": angle_scopes["PRETRAIN-20K"]["all_radians"],
            "observation_count": angle_scopes["PRETRAIN-20K"]["observation_count"],
            "multiplicity": angle_scopes["PRETRAIN-20K"]["stored_multiplicity"],
            "encoded_variance": angle_scopes["PRETRAIN-20K"]["encoded_variance_norm"],
        },
        "periodic_metadata": pre_meta,
        "radial_ambiguity": radial,
        "near_angle_different_radial_examples": int(len(examples)),
        "dimenet_style_basis_feasibility": feasibility["status"],
        "candidate_decisions": {
            "2D_BOND_INPUT_CANDIDATE": {
                "decision": "YES",
                "reason": "formal canonical data carries bond type, but O8 numerical forward uses topology/path nodes rather than explicit bond chemistry",
            },
            "GEOMETRY_BASIS_CANDIDATE": {
                "decision": "YES" if distance_redundancy or angle_redundancy or radial_evidence else "NO",
                "reason": {
                    "distance_basis_redundancy": bool(distance_redundancy),
                    "angle_basis_redundancy": bool(angle_redundancy),
                    "fixed_angle_radial_diversity": bool(radial_evidence),
                },
            },
            "METADATA_ABLATION_CANDIDATE": {
                "decision": "YES" if redundant_fields else "NO",
                "fields": redundant_fields,
                "reason": "listed fields are constant/dominant or act as an exact structural class tag in the audited data",
            },
        },
        "anomalies": [],
        "restrictions": {
            "training_run": False,
            "downstream_finetuning": False,
            "baseline_modified": False,
            "property_prediction": False,
        },
    }
    atomic_json(output / "input_audit_summary.json", summary)
    manifest = {
        "schema": summary["schema"],
        "script": str(Path(__file__).resolve()),
        "seed": int(args.seed),
        "inputs": {
            "checkpoint": str(checkpoint),
            "pretrain_sidecar": str(Path(args.pretrain_sidecar).resolve()),
            "downstream_sidecar": str(Path(args.downstream_sidecar).resolve()),
            "sample_ids": str(Path(args.sample_ids).resolve()),
        },
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
        "note": "identity and path record only; no hash or integrity gate",
    }
    atomic_json(output / "run_manifest.json", manifest)
    print(json.dumps(summary["candidate_decisions"], indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=ROOT / "results/mts_glt_v2/formal/a6_h_w1_20k/mts_glt_v2_probe_005k.pth",
        type=Path,
    )
    parser.add_argument(
        "--pretrain-sidecar",
        default=ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/PI1M_v2",
        type=Path,
    )
    parser.add_argument(
        "--downstream-sidecar",
        default=ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union",
        type=Path,
    )
    parser.add_argument(
        "--sample-ids",
        default=ROOT / "results/mts_glt_v2/alignment_audit_v1/sampled_graph_ids.json",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "results/mts_glt_v2/input_information_audit_v1",
        type=Path,
    )
    parser.add_argument("--max-basis-observations", default=200000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
