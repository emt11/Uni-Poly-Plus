"""Validation helpers for the single active paper5_outer split contract."""

SPLIT_PROTOCOL = "paper5_outer"
SPLIT_NAMES = ("train", "validation", "test")


def split_indices(entry, name, sample_count):
    """Return one validated integer index array from a fold entry."""
    values = entry.get(f"{name}_indices")
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"the split manifest has no usable {name}_indices array")
    indices = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name}_indices holds a non-integer entry: {value!r}")
        if not 0 <= value < sample_count:
            raise ValueError(f"{name}_indices holds an out-of-range index: {value}")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name}_indices repeats an index")
    return indices


def resolve_fold(manifest, task, fold, *, cohort_rows,
                 expected_protocol=SPLIT_PROTOCOL):
    """Validate and return train/validation indices for one paper5_outer fold."""
    if str(manifest.get("protocol")) != expected_protocol:
        raise ValueError(
            f"split manifest protocol {manifest.get('protocol')!r} != {expected_protocol!r}"
        )
    if manifest.get("validation_is_test") is not False:
        raise ValueError("split manifest does not declare validation_is_test=false")
    if str(manifest.get("task")) != str(task):
        raise ValueError(f"split manifest task {manifest.get('task')!r} != {task!r}")
    entry = next((item for item in manifest["folds"] if int(item["fold"]) == int(fold)), None)
    if entry is None:
        raise ValueError(f"split manifest has no fold {fold}")
    if int(manifest["sample_count"]) != int(cohort_rows):
        raise ValueError("downstream cohort task row count differs from the fixed split")
    count = int(manifest["sample_count"])
    splits = {name: split_indices(entry, name, count) for name in SPLIT_NAMES}
    if entry.get("sets_disjoint") is not True:
        raise ValueError("the fold does not assert split mutual exclusion")
    if entry.get("union_equals_full_cohort") is not True:
        raise ValueError("the fold does not assert complete cohort coverage")
    sets = {name: set(values) for name, values in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if sets[left] & sets[right]:
            raise ValueError(f"{left} and {right} overlap")
    union = set().union(*sets.values())
    if union != set(range(count)):
        raise ValueError(f"the fold does not cover the cohort exactly: {count - len(union)} missing")
    evidence = {
        "protocol": manifest["protocol"], "fold": int(fold), "task": str(task),
        "validation_is_test": False, "sample_count": count,
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "sets_disjoint": True, "indices": "integral, in range, duplicate-free",
        "union_equals_full_cohort": True, "outer_test": "NOT_RUN",
    }
    return splits["train"], splits["validation"], evidence
