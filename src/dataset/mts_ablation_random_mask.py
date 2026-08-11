"""Count-matched random MCL visibility sidecar for the A4 ablation.

Plan mts_geometry_injection_ablation_a0_a4_v2 §4.  For every MCL-valid
downstream-union polymer and every central-RU query atom the sidecar stores the
20%/50% visible key sets (sample-local Trimer indices, compact offsets + values)
chosen by a stateless SHA256 priority so the correct-space-neighbour test
(A3 vs A4) keeps identical per-query sparsity but random key identity.

The sidecar never modifies the frozen Trimer LMDB or MCL threshold artifacts,
never runs ETKDG/MMFF, and never materialises dense [Q, K] bool matrices.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.dataset.mips_cache_validation import _sha256_file
from src.dataset.mips_trimer_contract import (
    ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
    ABLATION_RANDOM_MASK_SCHEMA,
    ABLATION_RANDOM_MASK_SEED,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stable_key_priority(
    schema: str, seed: int, sample_key: bytes,
    query_canonical_atom_id: int, key_canonical_atom_id: int, key_ru_offset: int,
) -> bytes:
    material = (
        schema.encode("ascii")
        + b"||"
        + str(seed).encode("ascii")
        + b"||"
        + sample_key
        + b"||"
        + str(query_canonical_atom_id).encode("ascii")
        + b"||"
        + str(key_canonical_atom_id).encode("ascii")
        + b"||"
        + str(key_ru_offset).encode("ascii")
    )
    return hashlib.sha256(material).digest()


def _real_mcl_visible_counts(
    trimer_pos, central_local, thresholds, key_ru_offsets, valid_trimer_mask,
):
    """Return per-query (c20, c50) real 20%/50% visible key counts."""
    pos = trimer_pos.float()
    n_trimer = int(pos.size(0))
    if n_trimer == 0 or pos.ndim != 2 or pos.size(1) != 3:
        return None
    if not bool(torch.isfinite(pos).all()):
        return None
    central_pos = pos[central_local]          # [Q, 3]
    distances = torch.cdist(central_pos, pos)  # [Q, N]
    t20, t50 = float(thresholds[0]), float(thresholds[1])
    vis20 = distances <= t20
    vis50 = distances <= t50
    valid = valid_trimer_mask.unsqueeze(0)     # [1, N]
    vis20 = vis20 & valid
    vis50 = vis50 & valid
    c20 = vis20.sum(dim=1).tolist()
    c50 = vis50.sum(dim=1).tolist()
    return c20, c50


def _build_record(sample_key, trimer, thresholds, schema, seed):
    """Build one sample's compact random-mask record.

    Returns a dict with the per-query compact 20%/50% key sets (sample-local
    Trimer indices) or None for geometry/MCL-invalid samples.  MCL validity is
    judged with the exact function the collator uses so A4 falls back exactly
    like A3.
    """
    from src.dataset.mips_cache_validation import trimer_can_enter_mcl

    if not trimer_can_enter_mcl(trimer, 0):
        return None
    pos = getattr(trimer, "trimer_pos", None)
    if pos is None:
        return None
    central_local = torch.nonzero(
        torch.as_tensor(getattr(trimer, "trimer_central_ru_mask")).bool(),
        as_tuple=False,
    ).flatten()
    if central_local.numel() == 0:
        return None
    n_trimer = int(pos.size(0))
    valid_mask = torch.arange(n_trimer) >= 0
    counts = _real_mcl_visible_counts(
        pos, central_local, thresholds,
        getattr(trimer, "trimer_ru_offset", torch.zeros(n_trimer)),
        valid_mask,
    )
    if counts is None:
        return None
    c20s, c50s = counts
    ru_offset = torch.as_tensor(
        getattr(trimer, "trimer_ru_offset", torch.zeros(n_trimer)),
        dtype=torch.long,
    )
    canonical_ids = torch.as_tensor(
        getattr(trimer, "trimer_base_ru_atom_id", torch.zeros(n_trimer)),
        dtype=torch.long,
    )
    self_local = central_local.tolist()
    ptr20 = [0]
    ptr50 = [0]
    keys20 = []
    keys50 = []
    for q in range(int(central_local.numel())):
        self_idx = self_local[q]
        candidates = [
            k for k in range(n_trimer)
            if k != self_idx and valid_mask[k]
        ]
        priorities = []
        for k in candidates:
            priorities.append((
                _stable_key_priority(
                    schema, seed, sample_key,
                    int(canonical_ids[self_idx]), int(canonical_ids[k]),
                    int(ru_offset[k]),
                ),
                k,
            ))
        priorities.sort(key=lambda item: item[0])
        ordered = [k for _, k in priorities]
        vis20 = [self_idx] + ordered[: max(0, c20s[q] - 1)]
        vis50 = [self_idx] + ordered[: max(0, c50s[q] - 1)]
        # Nesting by construction: R20 is a prefix of R50 after self.
        vis50 = vis20 + [k for k in ordered[c20s[q] - 1: c50s[q] - 1]]
        keys20.extend(vis20)
        keys50.extend(vis50)
        ptr20.append(ptr20[-1] + len(vis20))
        ptr50.append(ptr50[-1] + len(vis50))
    return {
        "n_query": int(central_local.numel()),
        "n_trimer": n_trimer,
        "ptr20": ptr20,
        "ptr50": ptr50,
        "keys20": np.asarray(keys20, dtype=np.int32),
        "keys50": np.asarray(keys50, dtype=np.int32),
        "self_local": self_local,
    }


def build_random_mask_sidecar(
    cohort,
    trimer_root: Path,
    mcl_threshold_path: Path,
    output_root: Path,
) -> Path:
    """Stream-build the downstream-union random-mask sidecar.

    Read-only over the frozen Trimer LMDB; single parent writer.  Returns the
    sidecar root.
    """
    from src.dataset.lmdb_cache import LmdbLayerStore

    output_root.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(cohort["keys"], dtype=np.uint8)
    if keys.ndim != 2 or keys.shape[1] != 32:
        raise ValueError("sidecar cohort keys must have shape [N,32]")
    thresholds_all = np.load(mcl_threshold_path, mmap_mode="r")
    if thresholds_all.shape != (len(keys), 2):
        raise ValueError("MCL threshold array shape mismatch")
    # Only downstream_union is needed (this cycle is finetune-only).
    schema = ABLATION_RANDOM_MASK_SCHEMA
    seed = ABLATION_RANDOM_MASK_SEED

    store = LmdbLayerStore(str(trimer_root))
    sample_ptr = [0]
    ptr20_all = [0]
    ptr50_all = [0]
    keys20_all = []
    keys50_all = []
    per_sample = {}
    for i, raw_key in enumerate(keys):
        sample_key = bytes(raw_key)
        thresholds = thresholds_all[i]
        if not bool(np.isfinite(thresholds).all()):
            per_sample[i] = None
            sample_ptr.append(sample_ptr[-1])
            continue
        trimer = store[sample_key]
        record = _build_record(
            sample_key, trimer, thresholds, schema, seed
        )
        per_sample[i] = record
        if record is None:
            sample_ptr.append(sample_ptr[-1])
            continue
        base20 = ptr20_all[-1]
        ptr20_all.extend(base20 + int(v) for v in record["ptr20"][1:])
        base50 = ptr50_all[-1]
        ptr50_all.extend(base50 + int(v) for v in record["ptr50"][1:])
        keys20_all.append(record["keys20"])
        keys50_all.append(record["keys50"])
        sample_ptr.append(sample_ptr[-1] + record["n_query"])
        if i % 100 == 0:
            print(f"[a4-mask] {i}/{len(keys)}", flush=True)
    store.close()

    keys20_arr = (
        np.concatenate(keys20_all).astype(np.int32)
        if keys20_all else np.zeros(0, dtype=np.int32)
    )
    keys50_arr = (
        np.concatenate(keys50_all).astype(np.int32)
        if keys50_all else np.zeros(0, dtype=np.int32)
    )
    np.save(output_root / "sample_ptr.npy", np.asarray(sample_ptr, dtype=np.int64))
    np.save(output_root / "query_ptr20.npy", np.asarray(ptr20_all, dtype=np.int64))
    np.save(output_root / "query_ptr50.npy", np.asarray(ptr50_all, dtype=np.int64))
    np.save(output_root / "query_keys20.npy", keys20_arr)
    np.save(output_root / "query_keys50.npy", keys50_arr)
    np.save(output_root / "sample_keys.npy", keys)
    _write_sidecar_metadata(
        output_root, cohort, trimer_root, mcl_threshold_path,
        keys20_arr, keys50_arr,
    )
    return output_root


def _write_sidecar_metadata(
    output_root, cohort, trimer_root, mcl_threshold_path,
    keys20_arr, keys50_arr,
) -> None:
    from src.dataset.mips_trimer_contract import (
        ABLATION_RANDOM_MASK_SCHEMA,
        ABLATION_RANDOM_MASK_SEED,
    )
    from src.dataset.mips_cache_validation import _sha256_file as _sha

    trimer_root = Path(trimer_root)
    mcl_path = Path(mcl_threshold_path)
    payload_files = (
        "sample_ptr.npy", "query_ptr20.npy", "query_ptr50.npy",
        "query_keys20.npy", "query_keys50.npy", "sample_keys.npy",
    )
    payload_hashes = {
        name: _sha(output_root / name) for name in payload_files
    }
    payload = {
        "schema": ABLATION_RANDOM_MASK_SCHEMA,
        "payload_version": ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
        "seed": ABLATION_RANDOM_MASK_SEED,
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "record_count": int(len(cohort["keys"])),
        "trimer_artifact_hash": (trimer_root / ".done").read_text().strip(),
        "trimer_done_file_sha256": _sha(trimer_root / ".done"),
        "trimer_frozen_file_sha256": _sha(trimer_root / ".frozen"),
        "mcl_threshold_artifact_hash": (
            (mcl_path.parent / "mcl_thresholds_metadata.json")
            and json.loads(
                (mcl_path.parent / "mcl_thresholds_metadata.json").read_text()
            ).get("trimer_artifact_hash")
        ),
        "mcl_threshold_file_sha256": _sha(mcl_path),
        "self_key_policy": "central_ru_self_key_always_visible",
        "identity_tuple": "sample_key+query_canonical_atom_id+key_canonical_atom_id+key_ru_offset",
        "payload_files": payload_hashes,
        "done_payload_sha256": hashlib.sha256(
            json.dumps(payload_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    temporary = output_root / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_root / "metadata.json")
    (output_root / ".done").write_text(payload["done_payload_sha256"] + "\n")


class AblationRandomMaskSidecar:
    """Read-only loader for the count-matched random MCL visibility sidecar.

    ``record_for(sample_key)`` returns the sample-local compact sets for one
    downstream-union key, or ``None`` when the sample is not MCL-valid.  A
    task-local integer index is deliberately rejected: downstream task CSVs do
    not share the union ordering.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.meta.get("schema") != ABLATION_RANDOM_MASK_SCHEMA:
            raise ValueError("unsupported A4 random-mask sidecar schema")
        if int(self.meta.get("seed", -1)) != ABLATION_RANDOM_MASK_SEED:
            raise ValueError("A4 random-mask seed mismatch")
        if int(self.meta.get("payload_version", -1)) != ABLATION_RANDOM_MASK_PAYLOAD_VERSION:
            raise ValueError("A4 random-mask payload version mismatch")
        self.sample_ptr = np.load(self.root / "sample_ptr.npy")      # [n+1]
        self.ptr20 = np.load(self.root / "query_ptr20.npy")           # [TQ+1]
        self.ptr50 = np.load(self.root / "query_ptr50.npy")           # [TQ+1]
        self.keys20 = np.load(self.root / "query_keys20.npy")         # int32
        self.keys50 = np.load(self.root / "query_keys50.npy")         # int32
        sample_keys_path = self.root / "sample_keys.npy"
        if not sample_keys_path.is_file():
            # Upgrade compatibility: a pre-v2 sidecar did not carry the key
            # matrix, but its cohort manifest remains immutable and local.
            cohort_root = (
                PROJECT_ROOT / "data/processed/mips_trimer_scage/cohorts"
                / "downstream_union" / str(self.meta["cohort_hash"])
            )
            sample_keys_path = cohort_root / "sample_keys.npy"
        self.sample_keys = np.load(sample_keys_path, mmap_mode="r")
        self._validate()
        self._row_for_key = {
            bytes(row): int(index) for index, row in enumerate(self.sample_keys)
        }
        if len(self._row_for_key) != int(self.sample_keys.shape[0]):
            raise ValueError("A4 sidecar sample keys contain duplicates")

    @property
    def cohort_hash(self):
        return self.meta["cohort_hash"]

    def _validate(self):
        record_count = int(self.meta.get("record_count", -1))
        if self.sample_keys.ndim != 2 or self.sample_keys.shape[1] != 32:
            raise ValueError("A4 sample-key payload must have shape [N,32]")
        if int(self.sample_keys.shape[0]) != record_count:
            raise ValueError("A4 sample-key count mismatch")
        if self.sample_ptr.ndim != 1 or self.sample_ptr.shape[0] != record_count + 1:
            raise ValueError("A4 sample_ptr shape mismatch")
        if self.ptr20.ndim != 1 or self.ptr50.ndim != 1:
            raise ValueError("A4 query pointer arrays must be one-dimensional")
        if self.ptr20.shape != self.ptr50.shape:
            raise ValueError("A4 query pointer arrays have different shapes")
        if self.ptr20.size == 0 or self.ptr20[0] != 0 or self.ptr50[0] != 0:
            raise ValueError("A4 query pointers must start at zero")
        if bool(np.any(np.diff(self.sample_ptr) < 0)):
            raise ValueError("A4 sample_ptr is not monotonic")
        if bool(np.any(np.diff(self.ptr20) < 0)) or bool(np.any(np.diff(self.ptr50) < 0)):
            raise ValueError("A4 query pointers are not monotonic")
        if int(self.ptr20[-1]) != int(self.keys20.size) or int(self.ptr50[-1]) != int(self.keys50.size):
            raise ValueError("A4 query pointer/payload length mismatch")
        for name, array in (("query_keys20", self.keys20), ("query_keys50", self.keys50)):
            if array.ndim != 1 or array.dtype != np.int32:
                raise ValueError(f"A4 {name} has invalid dtype/shape")
        payload_files = self.meta.get("payload_files")
        if not isinstance(payload_files, dict):
            raise ValueError("A4 sidecar is missing payload file manifest")
        for name, expected in payload_files.items():
            path = self.root / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"A4 sidecar payload hash mismatch: {name}")
        done_hash = hashlib.sha256(
            json.dumps(payload_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.meta.get("done_payload_sha256") != done_hash:
            raise ValueError("A4 sidecar payload manifest hash mismatch")
        done_path = self.root / ".done"
        if not done_path.is_file() or done_path.read_text().strip() != done_hash:
            raise ValueError("A4 sidecar .done marker is stale")
        # A sidecar is only usable with a frozen Trimer and threshold artifact.
        if self.root.is_relative_to(PROJECT_ROOT / "data"):
            cohort_root = (
                PROJECT_ROOT / "data/processed/mips_trimer_scage/cohorts"
                / "downstream_union" / str(self.meta["cohort_hash"])
            )
            manifest = cohort_root / "manifest.json"
            if not manifest.is_file():
                raise ValueError("A4 sidecar cohort manifest is missing")
            cohort_meta = json.loads(manifest.read_text(encoding="utf-8"))
            if cohort_meta.get("ordered_sample_key_hash") != self.meta.get("ordered_key_hash"):
                raise ValueError("A4 sidecar ordered-key hash mismatch")
            ordered_hash = hashlib.sha256(
                np.asarray(self.sample_keys, dtype=np.uint8).tobytes()
            ).hexdigest()
            if ordered_hash != self.meta.get("ordered_key_hash"):
                raise ValueError("A4 sidecar sample-key payload hash mismatch")
            threshold_root = cohort_root
            threshold_path = threshold_root / "mcl_thresholds.npy"
            threshold_meta_path = threshold_root / "mcl_thresholds_metadata.json"
            if not threshold_path.is_file() or not threshold_meta_path.is_file():
                raise ValueError("A4 MCL threshold artifact is missing")
            if _sha256_file(threshold_path) != self.meta.get("mcl_threshold_file_sha256"):
                raise ValueError("A4 MCL threshold file hash mismatch")
            threshold_meta = json.loads(threshold_meta_path.read_text())
            if threshold_meta.get("trimer_artifact_hash") != self.meta.get("trimer_artifact_hash"):
                raise ValueError("A4 threshold/Trimer artifact mismatch")
            trimer_roots = list((PROJECT_ROOT / "data/processed/mips_trimer_scage/trimer").glob("*/.done"))
            matched = [path.parent for path in trimer_roots if path.read_text().strip() == self.meta.get("trimer_artifact_hash")]
            if len(matched) != 1 or not (matched[0] / ".frozen").is_file():
                raise ValueError("A4 sidecar is not bound to one frozen Trimer root")
            if _sha256_file(matched[0] / ".done") != self.meta.get("trimer_done_file_sha256"):
                raise ValueError("A4 Trimer .done hash mismatch")
            if _sha256_file(matched[0] / ".frozen") != self.meta.get("trimer_frozen_file_sha256"):
                raise ValueError("A4 Trimer .frozen hash mismatch")

    def record_for(self, sample_key: bytes):
        if isinstance(sample_key, (int, np.integer)):
            raise TypeError("A4 sidecar lookup requires sample_key, not task-local index")
        sample_key = bytes(sample_key)
        if len(sample_key) != 32:
            raise ValueError("A4 sample_key must contain 32 bytes")
        sample_index = self._row_for_key.get(sample_key)
        if sample_index is None:
            raise KeyError(sample_key.hex())
        start = int(self.sample_ptr[sample_index])
        end = int(self.sample_ptr[sample_index + 1])
        if start == end:
            return None
        p20 = self.ptr20[start:end + 1]
        p50 = self.ptr50[start:end + 1]
        return {
            "n_query": int(end - start),
            "ptr20": p20 - p20[0],
            "ptr50": p50 - p50[0],
            "keys20": self.keys20[p20[0]:p20[-1]],
            "keys50": self.keys50[p50[0]:p50[-1]],
        }
