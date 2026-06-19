"""Shared IO, stable identifiers, leakage checks, and numeric helpers for KG pipeline."""

import csv
import hashlib
import json
import os
import re
from pathlib import Path


def stable_hash(*parts, length=16):
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix, *parts, length=16):
    return f"{prefix}_{stable_hash(*parts, length=length)}"


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_alias(value):
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, value):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_val_fields(value, location="payload"):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == "val":
                raise ValueError(f"Label leakage: forbidden field 'val' in {location}")
            assert_no_val_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_val_fields(child, f"{location}[{index}]")


# Numeric feature helpers
import math


def normalize_mass(value, unit):
    if value is None:
        return None, None, "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, None, "invalid"
    normalized_unit = str(unit or "g/mol").lower().replace(" ", "")
    factors = {"g/mol": 1.0, "gmol-1": 1.0, "kg/mol": 1000.0, "kgmol-1": 1000.0, "kda": 1000.0, "da": 1.0}
    factor = factors.get(normalized_unit)
    if factor is None:
        return None, None, "unsupported_unit"
    return number * factor, "g/mol", "normalized"


def log_numeric_bin(kind, value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    exponent = math.floor(math.log10(number))
    mantissa = number / (10 ** exponent)
    bucket = 1 if mantissa < 2 else 2 if mantissa < 5 else 5
    upper = {1: 2, 2: 5, 5: 10}[bucket]
    return f"{kind}_bin_{bucket}e{exponent}_{upper}e{exponent}"
