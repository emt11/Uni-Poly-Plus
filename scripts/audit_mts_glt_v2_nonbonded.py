#!/usr/bin/env python3
"""Read-only non-bonded periodic spatial-contact audit for MTS GLT-v2."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UniDataset  # noqa: E402
from src.dataset.periodic_line_glt import canonical_line_token  # noqa: E402
from src.training.pretrain.config import dataset_kwargs_from_args, parse_arguments  # noqa: E402

CUTOFFS = (3.5, 4.0, 4.5, 5.0)
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
FORMAL_CONFIG = ROOT / "configs/mts/glt_v2_formal_a6_h_w1_20k.json"
SAMPLE_IDS = ROOT / "results/mts_glt_v2/alignment_audit_v1/sampled_graph_ids.json"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def downstream_dataset(task: str):
    return UniDataset(
        root=str(ROOT / "data"), dataset=f"smi_{task}",
        smiles_model_name=str(ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"),
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        modalities=("graph",), cache_layers="ru_base,topology,trimer,md200",
        periodic_line_glt_sidecar=str(SIDECAR),
        experiment_id="glt-v2-nonbonded-audit-v1", feature_config_hash="manual",
        mips_core="paper_corrected", mips_max_hops=2, mips_use_descriptors=True,
    )


def periodic_adjacency(item):
    atom_count = int(item.canonical_atom_count)
    adjacency = [[] for _ in range(atom_count)]
    edges = item.ru_edge_index.long().numpy()
    for column in range(edges.shape[1]):
        left, right = int(edges[0, column]), int(edges[1, column])
        adjacency[left].append((right, 0)); adjacency[right].append((left, 0))
    left, right = int(item.ru_left_boundary), int(item.ru_right_boundary)
    adjacency[right].append((left, 1)); adjacency[left].append((right, -1))
    return tuple(tuple(sorted(set(row))) for row in adjacency)


def periodic_spd(adjacency, source: int, target: int, relative_shift: int, max_distance=64):
    """Shortest lifted bond distance from source@0 to target@relative_shift."""
    goal = (int(target), int(relative_shift))
    start = (int(source), 0)
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    shift_limit = max(4, abs(int(relative_shift)) + 2)
    while queue:
        (atom, shift), distance = queue.popleft()
        if distance >= int(max_distance):
            continue
        for neighbor, delta in adjacency[atom]:
            state = (neighbor, shift + delta)
            if abs(state[1]) > shift_limit or state in visited:
                continue
            if state == goal:
                return distance + 1
            visited.add(state); queue.append((state, distance + 1))
    return None


def periodic_spd_map(adjacency, source: int, shift_limit: int, max_distance=64):
    """Compute all lifted shortest paths for one canonical source atom."""
    start = (int(source), 0)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        atom, shift = queue.popleft()
        distance = distances[(atom, shift)]
        if distance >= int(max_distance):
            continue
        for neighbor, delta in adjacency[atom]:
            state = (neighbor, shift + delta)
            if abs(state[1]) > int(shift_limit) or state in distances:
                continue
            distances[state] = distance + 1
            queue.append(state)
    return distances


def contact_observations(item):
    if not bool(item.trimer_geometry_valid) or not bool(item.trimer_geometry_is_3d):
        return {}
    positions = item.trimer_pos.detach().cpu().numpy().astype(np.float64)
    base = item.trimer_base_ru_atom_id.detach().cpu().numpy().astype(np.int64)
    offsets = item.trimer_ru_offset.detach().cpu().numpy().astype(np.int64)
    atomic = item.atomic_numbers.detach().cpu().numpy().astype(np.int64)
    adjacency = periodic_adjacency(item)
    all_observations = defaultdict(list)
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            atom_left, atom_right = int(base[left]), int(base[right])
            if atom_left == atom_right:
                continue
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if not np.isfinite(distance):
                continue
            identity = canonical_line_token(
                atom_left, int(offsets[left]), atom_right, int(offsets[right])
            )
            atom_a, atom_b, _shift = identity
            pair = tuple(sorted((int(atomic[atom_a]), int(atomic[atom_b]))))
            all_observations[identity].append({
                "distance": distance, "element_pair": pair,
                "translation": min(int(offsets[left]), int(offsets[right])),
            })
    result = {}
    candidates = {
        identity: observations for identity, observations in all_observations.items()
        if min(row["distance"] for row in observations) < max(CUTOFFS)
    }
    for identity, observations in candidates.items():
        atom_a, atom_b, shift = identity
        spd = periodic_spd(adjacency, atom_a, atom_b, shift)
        if spd == 1:
            continue
        for row in observations:
            row["spd"] = spd
        result[identity] = observations
    return result


def stratum(spd):
    if spd is None: return "INF"
    if spd >= 6: return ">=6"
    return str(int(spd))


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size: return {"count": 0}
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "p01": float(np.quantile(values, .01)), "p05": float(np.quantile(values, .05)),
        "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .5)),
        "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .9)),
        "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
        "max": float(values.max()),
    }


class AuditAccumulator:
    def __init__(self, name):
        self.name = name; self.graphs = 0
        self.counts = defaultdict(lambda: array("I")); self.distances = defaultdict(lambda: array("f"))
        self.spd = defaultdict(Counter); self.shift = defaultdict(Counter)
        self.elements = defaultdict(Counter)
        self.element_distances = defaultdict(lambda: defaultdict(lambda: array("f")))
        self.persistence = defaultdict(lambda: array("f")); self.asymmetry = defaultdict(lambda: array("f"))

    def add(self, contacts):
        self.graphs += 1
        for cutoff in CUTOFFS:
            # threshold=2 is retained as the explicit local-geometry overlap
            # reference; >=3 and >=4 remain the requested candidate views.
            for threshold in (2, 3, 4):
                key = (cutoff, threshold); selected = []
                for identity, observations in contacts.items():
                    representative = observations[0]
                    spd = representative["spd"]
                    eligible = spd is None or spd >= threshold
                    positive = [row for row in observations if row["distance"] < cutoff]
                    if not eligible or not positive: continue
                    selected.append((identity, observations, positive))
                    self.spd[key][stratum(spd)] += 1
                    self.shift[key][str(abs(int(identity[2])))] += 1
                    pair = f"{representative['element_pair'][0]}-{representative['element_pair'][1]}"
                    self.elements[key][pair] += 1
                    self.element_distances[key][pair].extend(row["distance"] for row in positive)
                    self.distances[key].extend(row["distance"] for row in positive)
                    ordered = sorted(observations, key=lambda row: row["translation"])
                    # Persistence uses every valid observation in the canonical
                    # identity denominator.  A single-observation identity is
                    # therefore a valid persistence=1 contact; paired
                    # asymmetry remains defined only when both sides exist.
                    self.persistence[key].append(len(positive) / len(observations))
                    if len(ordered) >= 2:
                        self.asymmetry[key].append(abs(ordered[0]["distance"] - ordered[-1]["distance"]))
                self.counts[key].append(len(selected))

    def rows(self):
        count_rows, spd_rows, distance_rows, element_rows, persistence_rows = [], [], [], [], []
        for key in sorted(self.counts):
            cutoff, threshold = key; counts = np.asarray(self.counts[key])
            count_rows.append({
                "dataset": self.name, "cutoff": cutoff, "spd_threshold": threshold,
                "graphs_total": self.graphs, "graphs_with_contact": int(np.count_nonzero(counts)),
                "coverage_fraction": float(np.mean(counts > 0)),
                **{f"contacts_{name}": value for name, value in quantile_summary(counts).items()},
                "unique_contacts_total": int(counts.sum()),
                "cross_ru_fraction": float(
                    sum(value for shift, value in self.shift[key].items() if shift != "0")
                    / max(1, sum(self.shift[key].values()))
                ),
            })
            total = max(1, sum(self.spd[key].values()))
            for label, value in sorted(self.spd[key].items()):
                spd_rows.append({"dataset": self.name, "cutoff": cutoff, "spd_threshold": threshold, "spd": label, "count": value, "fraction": value / total})
            distance_rows.append({"dataset": self.name, "cutoff": cutoff, "spd_threshold": threshold, **quantile_summary(self.distances[key])})
            element_total = max(1, sum(self.elements[key].values()))
            for pair, value in self.elements[key].most_common(20):
                element_rows.append({"dataset": self.name, "cutoff": cutoff, "spd_threshold": threshold, "element_pair": pair, "count": value, "fraction": value / element_total, "mean_distance": float(np.mean(self.element_distances[key][pair]))})
            persistence = np.asarray(self.persistence[key], dtype=np.float64)
            persistence_rows.append({
                "dataset": self.name, "cutoff": cutoff, "spd_threshold": threshold,
                "observations": int(persistence.size),
                "fraction_eq_1": float(np.mean(persistence == 1)) if persistence.size else None,
                "fraction_ge_2_3": float(np.mean(persistence >= 2/3)) if persistence.size else None,
                "fraction_ge_1_2": float(np.mean(persistence >= .5)) if persistence.size else None,
                "fraction_lt_1_2": float(np.mean(persistence < .5)) if persistence.size else None,
                "left_right_absolute_difference": quantile_summary(self.asymmetry[key]),
            })
        return count_rows, spd_rows, distance_rows, element_rows, persistence_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pi1m", type=int, default=20000)
    args = parser.parse_args(argv)
    sampled = json.loads(SAMPLE_IDS.read_text())["sampled_graph_ids"][:args.max_pi1m]
    dataset_args = parse_arguments(["--experiment_config", str(FORMAL_CONFIG)])
    pretrain = UniDataset(**dataset_kwargs_from_args(dataset_args))
    accumulators = []
    pi = AuditAccumulator("PI1M_v2")
    for position, index in enumerate(sampled):
        pi.add(contact_observations(pretrain[int(index)]))
        if position and position % 2000 == 0: print(f"PI1M nonbonded audit {position}/{len(sampled)}", flush=True)
    accumulators.append(pi)
    for task in TASKS:
        accumulator = AuditAccumulator(task); dataset = downstream_dataset(task)
        for index in range(len(dataset)): accumulator.add(contact_observations(dataset[index]))
        accumulators.append(accumulator)
    count_rows=[]; spd_rows=[]; distance_rows=[]; element_rows=[]; persistence_rows=[]
    for accumulator in accumulators:
        rows = accumulator.rows()
        for target, source in zip((count_rows, spd_rows, distance_rows, element_rows, persistence_rows), rows): target.extend(source)
    downstream_rows = [row for row in count_rows if row["dataset"] in TASKS and row["spd_threshold"] == 4]
    pi_primary = [row for row in count_rows if row["dataset"] == "PI1M_v2" and row["spd_threshold"] == 4]
    coverage = [row["coverage_fraction"] for row in pi_primary]
    primary_persistence = {
        row["cutoff"]: row for row in persistence_rows
        if row["dataset"] == "PI1M_v2" and row["spd_threshold"] == 4
    }
    persistent_majority = coverage and all(
        (primary_persistence[row["cutoff"]]["fraction_ge_2_3"] or 0.0) > 0.5
        for row in pi_primary
    )
    if coverage and min(coverage) > 0.5 and persistent_majority:
        label = "STRONG"
    elif coverage and max(coverage) > 0.0:
        label = "MODERATE"
    else:
        label = "WEAK"
    # The label summarizes observed prevalence and persistence only. It is not
    # an experiment gate and does not select a cutoff.
    evidence = {}
    for row in pi_primary:
        cutoff = row["cutoff"]
        persistence = primary_persistence[cutoff]
        matching_spd = {
            spd_row["spd"]: spd_row["fraction"]
            for spd_row in spd_rows
            if spd_row["dataset"] == "PI1M_v2"
            and spd_row["cutoff"] == cutoff
            and spd_row["spd_threshold"] == 2
        }
        evidence[str(cutoff)] = {
            "graph_coverage_spd_ge4": row["coverage_fraction"],
            "contacts_median_spd_ge4": row["contacts_p50"],
            "contacts_p95_spd_ge4": row["contacts_p95"],
            "cross_ru_fraction_spd_ge4": row["cross_ru_fraction"],
            "persistence_eq_1_spd_ge4": persistence["fraction_eq_1"],
            "persistence_ge_2_3_spd_ge4": persistence["fraction_ge_2_3"],
            "all_nonbonded_spd_fraction": matching_spd,
        }
    summary = {
        "schema": "mts-glt-v2-nonbonded-information-audit-v1",
        "pi1m_deterministic_graphs": len(sampled), "cutoffs": CUTOFFS,
        "primary_definition": "distance < cutoff; not same canonical atom; SPD>=4; direct bonds excluded",
        "canonicalization": "canonical_line_token(atom_i,q_i,atom_j,q_j)",
        "NONBONDED_INFORMATION_CANDIDATE": label,
        "reason": {
            "interpretation": (
                "STRONG requires a majority of graphs and a majority of canonical "
                "contacts persistent in at least two thirds of valid observations "
                "at every fixed cutoff; MODERATE means contacts exist but that "
                "majority contract is not met."
            ),
            "pi1m_by_cutoff": evidence,
        },
        "relative_shift_distribution": {
            str(cutoff): dict(
                next(accumulator for accumulator in accumulators if accumulator.name == "PI1M_v2").shift[(cutoff, 4)]
            )
            for cutoff in CUTOFFS
        },
        "persistence_available": True,
    }
    atomic_csv(args.output_dir / "contact_counts_by_cutoff.csv", count_rows)
    atomic_csv(args.output_dir / "spd_distribution.csv", spd_rows)
    atomic_csv(args.output_dir / "contact_distance_statistics.csv", distance_rows)
    atomic_csv(args.output_dir / "element_pair_statistics.csv", element_rows)
    atomic_csv(args.output_dir / "downstream_coverage.csv", downstream_rows)
    atomic_csv(args.output_dir / "persistence_statistics.csv", persistence_rows)
    atomic_json(args.output_dir / "canonicalization_sanity.json", {
        "inverse_equal": canonical_line_token(2, 0, 5, 1) == canonical_line_token(5, 1, 2, 0),
        "translation_equal": canonical_line_token(2, -1, 5, 0) == canonical_line_token(2, 0, 5, 1),
    })
    atomic_json(args.output_dir / "nonbonded_summary.json", summary)


if __name__ == "__main__": main()
