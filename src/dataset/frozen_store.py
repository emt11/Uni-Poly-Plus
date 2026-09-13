"""Read-only frozen-cache reader for formal MTS training.

This is the production data path described in PIPELINE.md:

    store.json -> frozen artifacts -> readonly LMDB -> per-key lookup

Design (after the MIPS reference implementation, ``vemol/dataset/
mol_graph_dataset.py``):

* the LMDB environment is opened ``readonly=True, lock=False,
  readahead=False, meminit=False`` exactly like MIPS ``init_on_disk_dataset``;
* a miss is an error (``CacheMissError``).  Nothing in this module imports
  RDKit, ETKDG or MMFF, and nothing writes: there is no online builder, no
  repair and no migration on the training path;
* compatibility is decided by ``store.json`` bindings, the ``.frozen``
  marker, the artifact ``build_spec`` and per-record required fields — never
  by schema or builder version numbers.

Record envelope compatibility: records are the existing ``torch.save``
payloads ``{"sample_key": bytes, "data": PyG Data}``.  The reader ignores any
other envelope fields, so it reads both the historical frozen artifacts and
any future build produced with the same envelope.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import lmdb
import numpy as np
import torch

from .cache_spec import (
    REQUIRED_FIELDS,
    ROUTE_BUILD_SPEC_HASHES,
    StoreError,
    describe_route_mismatch,
    load_store,
    route_mismatches,
)


class CacheMissError(KeyError):
    """A sample key is absent from a frozen artifact. Never rebuilt online."""


class ArtifactNotFrozen(RuntimeError):
    """A binding points at an artifact without a .frozen marker."""


class ArtifactIdentityError(RuntimeError):
    """Hard data corruption: identity/join/required-field violation."""


class StoreRouteMismatch(RuntimeError):
    """A bound artifact was built with a different build_spec than the route."""


class CacheMissingDerivedArtifact(StoreError):
    """A derived array (e.g. MD200 mmap) is absent and will NOT be
    generated on the training path.  Run the explicit offline
    materialization command (scripts/materialize_cache_derived.py)."""


def _coerce_key(key) -> bytes:
    if isinstance(key, np.ndarray):
        array = np.asarray(key, dtype=np.uint8).reshape(-1)
        if array.size == 32:
            return array.tobytes()
    if isinstance(key, memoryview):
        key = key.tobytes()
    if isinstance(key, (bytes, bytearray)):
        key = bytes(key)
    elif isinstance(key, str):
        try:
            key = bytes.fromhex(key)
        except ValueError:
            raise ArtifactIdentityError(
                f"sample key must be 32 bytes or 64 hex chars: {key[:16]}…"
            )
    else:
        raise ArtifactIdentityError(f"unsupported sample key type: {type(key)!r}")
    if len(key) != 32:
        raise ArtifactIdentityError("sample keys must contain 32 bytes")
    return key


def _as_bool(value) -> bool:
    if torch.is_tensor(value):
        flat = value.reshape(-1)
        return bool(flat[0].item()) if flat.numel() else False
    return bool(value)


# python-lmdb rejects opening the same environment twice in one process.
# Whole-dataset caches are shared by every fold/task Dataset in the process,
# so readonly handles are shared by (pid, realpath) with reference counting
# (same solution as the historical LmdbLayerStore registry).
_READ_ENV_REGISTRY = {}
_READ_ENV_REGISTRY_LOCK = threading.Lock()


def _acquire_readonly_env(data_path: Path):
    registry_key = (os.getpid(), os.path.realpath(str(data_path)))
    with _READ_ENV_REGISTRY_LOCK:
        entry = _READ_ENV_REGISTRY.get(registry_key)
        if entry is None:
            environment = lmdb.open(
                str(data_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            )
            entry = [environment, 0]
            _READ_ENV_REGISTRY[registry_key] = entry
        entry[1] += 1
        return entry[0], registry_key


def _release_readonly_env(registry_key, environment):
    with _READ_ENV_REGISTRY_LOCK:
        entry = _READ_ENV_REGISTRY.get(registry_key)
        if entry is not None and entry[0] is environment:
            entry[1] -= 1
            if entry[1] <= 0:
                _READ_ENV_REGISTRY.pop(registry_key, None)
                entry[0].close()
        else:
            environment.close()


def _tensor_values_equal(left, right) -> bool:
    try:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return bool(torch.equal(left, right))
        return bool(left == right)
    except (TypeError, RuntimeError):
        return False


class FrozenArtifact:
    """One frozen LMDB artifact opened strictly read-only."""

    def __init__(self, cache_root, name: str, binding: dict,
                 *, validate_route: bool = True):
        self.name = str(name)
        self.binding = binding
        self.build_spec_hash = str(binding["build_spec_hash"])
        self.artifact_hash = str(binding["artifact_hash"])
        self.root = (Path(cache_root) / binding["path"]).resolve()
        self.required_fields = REQUIRED_FIELDS.get(self.name, ())

        data_path = self.root / "data.lmdb"
        if not data_path.is_dir():
            raise StoreError(
                f"bound {self.name} artifact has no data.lmdb: {self.root}"
            )
        frozen_path = self.root / ".frozen"
        if not frozen_path.is_file():
            raise ArtifactNotFrozen(
                f"bound {self.name} artifact is not frozen: {self.root}"
            )
        try:
            import json
            frozen_payload = json.loads(
                frozen_path.read_text(encoding="utf-8")
            )
        except Exception:
            frozen_payload = {}
        recorded = frozen_payload.get("build_spec_hash")
        if recorded is not None and str(recorded) != self.build_spec_hash:
            raise ArtifactIdentityError(
                f"{self.name} .frozen build_spec_hash does not match the "
                f"store binding: {self.root}"
            )
        if validate_route and self.name in ROUTE_BUILD_SPEC_HASHES:
            if self.build_spec_hash != ROUTE_BUILD_SPEC_HASHES[self.name]:
                raise StoreRouteMismatch(describe_route_mismatch(self.name, {
                    "artifacts": {self.name: binding}
                }))
        self._environment, self._registry_key = _acquire_readonly_env(data_path)
        self._transaction = self._environment.begin(buffers=False)

    # -- lifecycle ---------------------------------------------------------
    def close(self):
        if self._transaction is not None:
            self._transaction.abort()
            self._transaction = None
        if self._environment is not None:
            _release_readonly_env(self._registry_key, self._environment)
            self._environment = None
            self._registry_key = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def renew(self):
        if self._environment is not None:
            if self._transaction is not None:
                self._transaction.abort()
            self._transaction = self._environment.begin(buffers=False)

    # -- access ------------------------------------------------------------
    def __contains__(self, key) -> bool:
        key = _coerce_key(key)
        return bool(self._transaction.cursor().set_key(key))

    def missing_mask(self, keys) -> np.ndarray:
        keys = list(keys)
        mask = np.zeros(len(keys), dtype=np.uint8)
        cursor = self._transaction.cursor()
        for index, key in enumerate(keys):
            if not cursor.set_key(_coerce_key(key)):
                mask[index] = 1
        return mask

    def _deserialize(self, payload: bytes, key: bytes):
        record = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=False
        )
        if bytes(record.get("sample_key", b"")) != key:
            raise ArtifactIdentityError(
                f"{self.name} record sample_key mismatch for {key.hex()[:16]}…"
            )
        data = record.get("data")
        if data is None:
            raise ArtifactIdentityError(
                f"{self.name} record has no data payload: {key.hex()[:16]}…"
            )
        return data

    def get(self, key):
        """One sample, or ``CacheMissError``.  No online generation exists."""

        key = _coerce_key(key)
        value = self._transaction.get(key)
        if value is None:
            raise CacheMissError(
                f"{self.name} frozen cache has no record for {key.hex()[:16]}…"
            )
        data = self._deserialize(bytes(value), key)
        self._check_required_fields(data, key)
        return data

    __getitem__ = get

    def _check_required_fields(self, data, key: bytes):
        try:
            present = set(data.keys())
        except Exception:
            present = set()
        missing = [
            name for name in self.required_fields
            if name not in present and not hasattr(data, name)
        ]
        if missing:
            raise ArtifactIdentityError(
                f"{self.name} record {key.hex()[:16]}… is missing required "
                f"fields: {missing}"
            )


class FrozenFeatureStore:
    """Compose frozen topology + optional trimer + optional MD200 by key.

    The merge/join semantics mirror the historical ``LmdbFeatureStore``:
    only the allowlisted O8→Trimer mapping placeholder may be overridden and
    every other duplicate field must compare equal, otherwise the join is a
    hard identity error.
    """

    _PLACEHOLDER_ALLOWLIST = frozenset({"mips_to_trimer_central_index"})

    def __init__(self, *, cohort, topology: FrozenArtifact,
                 trimer: FrozenArtifact | None = None,
                 md200: FrozenArtifact | None = None,
                 md200_path=None, md200_valid_path=None):
        self.roots = {"topology": str(topology.root)}
        self.topology = topology
        self.trimer = trimer
        self.md200_artifact = md200
        if trimer is not None:
            self.roots["trimer"] = str(trimer.root)
        if md200 is not None:
            self.roots["md200"] = str(md200.root)
        self.cohort = cohort
        self.keys = cohort["keys_array"]
        self.md200_path = md200_path
        self.md200_valid_path = md200_valid_path
        self.md200 = (
            np.load(md200_path, mmap_mode="r") if md200_path is not None else None
        )
        self.md200_valid = (
            np.load(md200_valid_path, mmap_mode="r")
            if md200_valid_path is not None else None
        )
        if self.md200 is not None:
            if self.md200.ndim != 2 or self.md200.shape[1] != 200:
                raise StoreError("MD200 mmap must have shape [N,200]")
            if self.md200.dtype != np.float32:
                raise StoreError("MD200 mmap must use float32")
        if self.md200_valid is not None:
            if self.md200_valid.ndim != 1 or self.md200_valid.shape[0] != len(self.keys):
                raise StoreError("MD200 validity mmap has an invalid shape")
        self._sorted_key_order = None
        self._sorted_key_view = None
        if self.md200 is not None or self.md200_valid is not None:
            key_view = np.asarray(self.keys, dtype=np.uint8).view("S32").reshape(-1)
            self._sorted_key_order = np.argsort(key_view, kind="mergesort")
            self._sorted_key_view = key_view[self._sorted_key_order]

    def close(self):
        self.topology.close()
        if self.trimer is not None:
            self.trimer.close()
        if self.md200_artifact is not None:
            self.md200_artifact.close()
        self.md200 = None
        self.md200_valid = None

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key) -> bool:
        key = _coerce_key(key)
        if key not in self.topology:
            return False
        if self.trimer is not None and key not in self.trimer:
            return False
        return True

    def _row_for_key(self, key):
        if self._sorted_key_view is None:
            return None
        scalar = np.frombuffer(bytes(key), dtype=np.dtype("S32"), count=1)[0]
        position = int(np.searchsorted(self._sorted_key_view, scalar))
        if (
            position >= int(self._sorted_key_view.size)
            or self._sorted_key_view[position] != scalar
        ):
            return None
        return int(self._sorted_key_order[position])

    def _validate_mapping_override(self, name, topology_value, trimer_value,
                                   *, n_nodes, central_ru_mask, key):
        """Mirror of the historical placeholder-override contract."""

        if name not in self._PLACEHOLDER_ALLOWLIST:
            raise ArtifactIdentityError(
                f"duplicate frozen cache field: {name} ({key.hex()[:16]}…)"
            )
        if not (
            isinstance(topology_value, torch.Tensor)
            and topology_value.numel() > 0
            and bool(torch.all(topology_value == -1))
        ):
            raise ArtifactIdentityError(
                f"duplicate frozen cache field: {name} ({key.hex()[:16]}…)"
            )
        if not isinstance(trimer_value, torch.Tensor) or trimer_value.dtype != torch.long:
            raise ArtifactIdentityError(f"invalid {name} override dtype")
        if trimer_value.ndim != 1 or int(trimer_value.size(0)) != int(n_nodes):
            raise ArtifactIdentityError(f"invalid {name} override shape")
        if central_ru_mask is None:
            raise ArtifactIdentityError(f"{name} central_ru_mask is missing")
        mask = torch.as_tensor(central_ru_mask)
        if mask.ndim != 1 or mask.size(0) == 0 or mask.dtype != torch.bool:
            raise ArtifactIdentityError(f"invalid {name} central_ru_mask")
        if trimer_value.numel() > 0:
            indices = trimer_value.long()
            if bool((indices < 0).any()) or bool((indices >= int(mask.size(0))).any()):
                raise ArtifactIdentityError(f"invalid {name} override index range")
            if not bool(mask[indices].all()):
                raise ArtifactIdentityError(
                    f"{name} mapping does not point to the central RU"
                )

    def _check_identity(self, key, topology_data, trimer_data):
        """Hard identity checks on the topology↔trimer join."""

        mapping = getattr(trimer_data, "mips_to_trimer_central_index", None)
        if mapping is None:
            raise ArtifactIdentityError(
                f"trimer record {key.hex()[:16]}… has no O8 mapping"
            )
        mapping = torch.as_tensor(mapping).long().reshape(-1)
        node_count = int(getattr(topology_data, "num_nodes", 0) or 0)
        if node_count and int(mapping.numel()) != node_count:
            raise ArtifactIdentityError(
                f"topology↔trimer identity mismatch for {key.hex()[:16]}…: "
                f"mapping length {int(mapping.numel())} != topology nodes {node_count}"
            )
        geometry_valid = _as_bool(getattr(trimer_data, "trimer_geometry_valid", False))
        if not geometry_valid:
            return
        positions = torch.as_tensor(getattr(trimer_data, "trimer_pos"))
        if positions.ndim != 2 or positions.size(-1) != 3:
            raise ArtifactIdentityError(
                f"trimer record {key.hex()[:16]}… has invalid position shape"
            )
        if bool((mapping < 0).any()) or bool((mapping >= positions.size(0)).any()):
            raise ArtifactIdentityError(
                f"O8↔trimer mapping corruption for {key.hex()[:16]}…"
            )
        central_mask = getattr(trimer_data, "trimer_central_ru_mask", None)
        if central_mask is None:
            raise ArtifactIdentityError(
                f"trimer record {key.hex()[:16]}… has no central RU mask"
            )
        central_mask = torch.as_tensor(central_mask).bool().reshape(-1)
        if int(central_mask.numel()) != positions.size(0):
            raise ArtifactIdentityError(
                f"central RU mask length mismatch for {key.hex()[:16]}…"
            )
        if not bool(central_mask[mapping].all()):
            raise ArtifactIdentityError(
                f"O8 mapping does not point to the central RU for "
                f"{key.hex()[:16]}…"
            )

    def __getitem__(self, key):
        key = _coerce_key(key)
        merged = self.topology.get(key)
        if self.trimer is not None:
            layer = self.trimer.get(key)
            self._check_identity(key, merged, layer)
            for name in layer.keys():
                layer_value = layer[name]
                if name in merged:
                    if name in self._PLACEHOLDER_ALLOWLIST:
                        if self._validate_mapping_override(
                            name, merged[name], layer_value,
                            n_nodes=int(merged.x.size(0)),
                            central_ru_mask=layer.get("trimer_central_ru_mask"),
                            key=key,
                        ):
                            merged[name] = layer_value
                        continue
                    if _tensor_values_equal(merged[name], layer_value):
                        continue
                    raise ArtifactIdentityError(
                        f"duplicate frozen cache field: {name} "
                        f"({key.hex()[:16]}…)"
                    )
                merged[name] = layer_value
        if self.md200 is not None:
            row = self._row_for_key(key)
            if row is None:
                raise CacheMissError(f"MD200 row missing for {key.hex()[:16]}…")
            merged.mips_md = torch.from_numpy(
                np.asarray(self.md200[row]).copy()
            ).float()
            merged.mips_md_valid = bool(self.md200_valid[row])
            merged.mips_descriptor_source = "source_star_sub"
            merged.mips_descriptor_optimizer = "not_applicable_2d"
            merged.mips_descriptor_schema_version = 5
            merged.descriptor_failure_code = (
                "" if merged.mips_md_valid else "md200_content_cache_invalid"
            )
        return merged

    def values(self):
        for key in self.keys:
            yield self[key]


def open_frozen_feature_store(cache_root, cache_layers, *, cohort,
                              validate_route: bool = True,
                              validate_coverage: str = "sample"):
    """Open the store.json-bound frozen artifacts requested by a route.

    Returns ``(feature_store, store)``.  ``cache_layers`` is the resolved
    layer tuple (``ru_base`` is a parent dependency, never opened for
    reading).  ``validate_coverage`` mirrors the historical ``cache_validate``
    values: ``sample`` spot-checks 128 keys, ``full`` checks every key.
    """

    store = load_store(cache_root)
    artifacts = store["artifacts"]
    requested = [
        name for name in ("ru_base", "topology", "trimer", "md200")
        if name in tuple(cache_layers)
    ]
    if "topology" not in requested:
        raise StoreError(
            "the frozen reader always binds the topology artifact; requested "
            f"layers were: {sorted(requested)}"
        )
    for name in requested:
        if name not in artifacts:
            raise StoreError(
                f"active {name} artifact missing: store.json has no binding "
                f"for it (rebuild and re-run scripts/create_cache_store.py)"
            )
    if validate_route:
        mismatches = route_mismatches(store)
        for name in requested:
            if name in mismatches:
                raise StoreRouteMismatch(
                    describe_route_mismatch(name, store)
                )

    def _open(name):
        return FrozenArtifact(
            cache_root, name, artifacts[name], validate_route=validate_route
        )

    topology = _open("topology")
    trimer = _open("trimer") if "trimer" in requested else None
    md200 = _open("md200") if "md200" in requested else None
    try:
        md_path = md_valid_path = None
        if md200 is not None:
            md_path, md_valid_path = resolve_md200_paths(cohort, md200)
        feature_store = FrozenFeatureStore(
            cohort=cohort,
            topology=topology,
            trimer=trimer,
            md200=md200,
            md200_path=md_path,
            md200_valid_path=md_valid_path,
        )
    except BaseException:
        topology.close()
        if trimer is not None:
            trimer.close()
        if md200 is not None:
            md200.close()
        raise

    validate_coverage_keys(feature_store, cohort, mode=validate_coverage)
    return feature_store, store


def validate_coverage_keys(feature_store, cohort, *, mode="sample"):
    """Coverage + identity spot-check; a miss is an error, never a rebuild."""

    keys = cohort["keys_array"]
    if mode == "full":
        selected = range(len(keys))
    else:
        selected = range(min(128, len(keys)))
    for index in selected:
        key = keys[int(index)]
        if key not in feature_store:
            missing_topology = bytes(key) not in feature_store.topology
            raise CacheMissError(
                f"frozen cache does not cover cohort key {bytes(key).hex()[:16]}… "
                f"(topology_present={not missing_topology})"
            )


def _md200_metadata_ok(metadata, cohort_hash, ordered_hash, count) -> bool:
    return (
        metadata.get("cohort_hash") == cohort_hash
        and metadata.get("ordered_sample_key_hash") == ordered_hash
        and metadata.get("shape") == [count, 200]
        and metadata.get("dtype") == "float32"
    )


def resolve_md200_paths(cohort, md_artifact: FrozenArtifact):
    """Locate this cohort's materialized MD200 mmap pair.  Read-only.

    Lookup order: the store binding's registered array, then the
    deterministic ``md200_<build_spec_hash>/`` directory.  A missing or
    inconsistent array is a hard error — the training path never
    materializes anything.
    """

    import json

    cohort_dir = Path(cohort["root"])
    cohort_hash = str(cohort["manifest"]["cohort_hash"])
    ordered_hash = cohort["manifest"]["ordered_sample_key_hash"]
    count = len(cohort["keys_array"])
    # Registered arrays are keyed by cohort NAME; the cohort root itself is
    # the content-addressed hash directory.
    cohort_name = str(cohort["manifest"].get("dataset_name") or "")
    if not cohort_name:
        parts = cohort_dir.parts
        cohort_name = (
            parts[parts.index("cohorts") + 1]
            if "cohorts" in parts else cohort_dir.parent.name
        )

    candidates = []
    registered = md_artifact.binding.get("materialized") or {}
    relative = registered.get(cohort_name)
    if relative:
        candidates.append((cache_root_of(md_artifact)) / relative)
    candidates.append(cohort_dir / f"md200_{md_artifact.build_spec_hash}")

    last_checked = None
    for position, directory in enumerate(candidates):
        metadata_path = directory / "md200_metadata.json"
        values_path = directory / "md200.npy"
        valid_path = directory / "md200_valid.npy"
        if not (metadata_path.is_file() and values_path.is_file()
                and valid_path.is_file()):
            if position == 0 and relative:
                # The store binding explicitly registered this array; a
                # missing file is binding corruption, not a fallback case.
                raise CacheMissingDerivedArtifact(
                    f"registered MD200 array is incomplete: {directory}"
                )
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _md200_metadata_ok(metadata, cohort_hash, ordered_hash, count):
            return str(values_path), str(valid_path)
        raise CacheMissingDerivedArtifact(
            f"registered MD200 array for cohort {cohort_name} does not "
            f"match the cohort identity: {directory}"
        )
    raise CacheMissingDerivedArtifact(
        f"no materialized MD200 array for cohort {cohort_name} "
        f"(expected md200_<build_spec_hash>/md200.npy under {cohort_dir}); "
        "run scripts/materialize_cache_derived.py offline — the training "
        "reader never materializes"
    )


def cache_root_of(artifact: FrozenArtifact) -> Path:
    return artifact.root.parents[1]
