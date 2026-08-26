"""Small, explicit helpers for the GLT-v2 metadata-dedup comparison.

The helpers intentionally operate only on state-dict names and shapes.  They
are used by the stage-1 sanity script; they are not a checkpoint identity or
hashing framework and do not alter the production loading path.
"""

from __future__ import annotations

import torch


def compare_shared_state_by_name_and_shape(source, target):
    """Compare shared tensors without modifying either module."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = []
    mismatched = []
    source_only = sorted(set(source_state) - set(target_state))
    target_only = sorted(set(target_state) - set(source_state))
    for name in sorted(set(source_state) & set(target_state)):
        src = source_state[name]
        dst = target_state[name]
        if tuple(src.shape) != tuple(dst.shape):
            mismatched.append({
                "name": name,
                "source_shape": list(src.shape),
                "target_shape": list(dst.shape),
            })
        else:
            shared.append(name)
    equal = [
        name for name in shared
        if torch.equal(source_state[name], target_state[name])
    ]
    return {
        "shared_tensors": shared,
        "equal_tensors": equal,
        "mismatch": mismatched,
        "source_only": source_only,
        "target_only": target_only,
    }


def copy_shared_state_by_name_and_shape(source, target):
    """Copy tensors shared by two modules and return an auditable report.

    A shared tensor is copied only when its name exists in both modules and
    has the same shape.  Parameters which exist only in FULL (the two removed
    metadata embeddings) are intentionally reported rather than fabricated in
    DEDUP.  No random state is touched by this function.
    """

    source_state = source.state_dict()
    target_state = target.state_dict()
    report = compare_shared_state_by_name_and_shape(source, target)
    with torch.no_grad():
        for name in report["shared_tensors"]:
            src = source_state[name]
            dst = target_state[name]
            dst.copy_(src)
    return compare_shared_state_by_name_and_shape(source, target)


def build_matched_pretrain_containers(factory, metadata_mode, *, seed):
    """Construct FULL reference and actual containers with matched init.

    ``factory`` must accept one metadata mode and return a complete
    ``MTSGLTV2PretrainContainer``.  Both constructions happen from the same
    CPU seed inside isolated RNG forks.  The actual container receives every
    shared tensor from the FULL reference by name and shape; no dummy
    metadata parameters are created in DEDUP.
    """

    caller_cpu_rng = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        reference = factory("full")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        actual = factory(str(metadata_mode))
    report = copy_shared_state_by_name_and_shape(reference, actual)
    if report["mismatch"] or len(report["shared_tensors"]) != len(report["equal_tensors"]):
        raise RuntimeError("matched metadata initialization has unequal shared tensors")
    report["metadata_mode"] = str(metadata_mode)
    report["reference_mode"] = "full"
    report["caller_cpu_rng_unchanged"] = bool(
        torch.equal(caller_cpu_rng, torch.get_rng_state())
    )
    report["reference_parameter_count"] = int(
        sum(value.numel() for value in reference.parameters())
    )
    report["actual_parameter_count"] = int(
        sum(value.numel() for value in actual.parameters())
    )
    return actual, reference, report


def parameter_accounting(full, dedup):
    """Return parameter/state accounting for the two explicit modes."""

    def _named(module):
        return {name: int(value.numel()) for name, value in module.named_parameters()}

    full_named = _named(full)
    dedup_named = _named(dedup)
    full_state = full.state_dict()
    dedup_state = dedup.state_dict()
    return {
        "full_parameters": int(sum(full_named.values())),
        "dedup_parameters": int(sum(dedup_named.values())),
        "parameter_delta_full_minus_dedup": int(
            sum(full_named.values()) - sum(dedup_named.values())
        ),
        "full_state_tensors": int(len(full_state)),
        "dedup_state_tensors": int(len(dedup_state)),
        "removed_parameter_names": sorted(set(full_named) - set(dedup_named)),
        "added_parameter_names": sorted(set(dedup_named) - set(full_named)),
        "removed_state_names": sorted(set(full_state) - set(dedup_state)),
        "added_state_names": sorted(set(dedup_state) - set(full_state)),
        "formula": {
            "full_token_input":
                "LN(endpoint + distance_mean + distance_variance "
                "+ distance_count_embedding + abs_shift_embedding)",
            "dedup_token_input":
                "LN(endpoint + distance_mean + distance_variance "
                "+ abs_shift_embedding)",
            "full_relation_bias":
                "angle_mean + angle_variance + angle_count_embedding "
                "+ angle_multiplicity_embedding",
            "dedup_relation_bias":
                "angle_mean + angle_variance + angle_count_embedding",
        },
    }


__all__ = [
    "build_matched_pretrain_containers",
    "compare_shared_state_by_name_and_shape",
    "copy_shared_state_by_name_and_shape",
    "parameter_accounting",
]
