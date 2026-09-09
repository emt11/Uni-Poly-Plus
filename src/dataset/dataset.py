import copy
import json
import hashlib
import multiprocessing as mp
import pickle
import os
import sys
import signal
import time
import math
import sqlite3
import shutil
import resource
import fcntl
from pathlib import Path
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import torch
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from .graph_data import (
    MIPS_EXPERIMENT_FEATURE_SCHEMA,
    MIPS_MULTIMER_BUILDER_VERSION,
    MIPSLocalConfig,
    annotate_structure_fields,
    attach_polymerized_mips_atom_features,
    attach_mips_local_lga,
    build_mips_paper_structure,
    build_mips_local_structure,
    build_periodic_multimer_mol,
    generate_multimer_smiles,
    build_mips_data_object,
)
from .canonical_periodic import (
    CANONICAL_FEATURE_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION,
    CANONICAL_TOPOLOGY_SCHEMA,
    build_canonical_periodic_topology,
    find_base_atom_mapping,
)
from .mips_trimer_contract import (
    TOPOLOGY_CANONICAL,
    TOPOLOGY_REPRESENTATIONS,
)
from .lmdb_cache import (
    CACHE_LAYOUT_SCHEMA,
    MD200_LMDB_SCHEMA,
    RU_BASE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    LmdbFeatureStore,
    LmdbLayerStore,
    LmdbLayerWriter,
    build_or_load_cohort,
    load_cohort,
    materialize_md200_array,
    normalize_polymer_smiles,
    sample_key_from_normalized,
    sample_key_from_smiles,
)
from .diagnostics import (
    print_dataset_diagnostics,
    summarize_feature_cache,
    write_dataset_diagnostics,
)
from .trimer_mcl import (
    TRIMER_ETKDG_MAX_ITERATIONS,
    TRIMER_ETKDG_RETRY_CANDIDATES,
    TRIMER_ETKDG_RETRY_MAX_ITERATIONS,
    TRIMER_ETKDG_TIMEOUT_SECONDS,
    TRIMER_MCL_BUILDER_VERSION,
    TRIMER_MCL_PROTOCOL,
    TRIMER_MCL_SCHEMA,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS,
    attach_finite_trimer_mcl,
    attach_unavailable_trimer_mcl,
)
from .mips_cache_validation import validate_mcl_record
from .periodic_line_glt import (
    PeriodicLineGLTSidecar,
)
from .periodic_line_glt_image import (
    PeriodicLineImageSidecar,
    SIDECAR_SCHEMA as PERIODIC_LINE_IMAGE_SCHEMA,
)
from .periodic_line_distill import (
    PeriodicLineDistillSidecar,
    SCHEMA_PREFIX as PERIODIC_LINE_DISTILL_SCHEMA_PREFIX,
)
from .mts_target_contract import make_target_contract
from rdkit.Chem import rdFingerprintGenerator


FP_MODE_DIMS = {
    "disabled": 0,
    "ecfp": 1024,
    "mixfp": 1048,
    "attachment_count": 2570,
}

SCAGE_DESCRIPTOR_DIMS = {
    "shape": 11,
    "usrcat": 60,
    "autocorr3d": 80,
    "rdf": 210,
    "morse": 224,
    "whim": 114,
}


class ShardedFeatureStore:
    """Read-only SQLite-indexed shards with a bounded per-process LRU."""

    schema = "mips-selected-sharded-store-v1"

    def __init__(self, root):
        self.root = str(root)
        self.index_path = os.path.join(self.root, "index.sqlite3")
        manifest_path = os.path.join(self.root, "manifest.json")
        done_path = os.path.join(self.root, ".done")
        if not os.path.isfile(manifest_path) or not os.path.isfile(done_path):
            raise RuntimeError(f"incomplete immutable feature cache: {self.root}")
        with open(manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("schema") != self.schema:
            raise RuntimeError(f"feature-cache schema mismatch: {self.root}")
        shard_count = int(self.manifest.get("shard_count", -1))
        if shard_count < 0 or int(self.manifest.get("count", -1)) < 0:
            raise RuntimeError(f"feature-cache manifest shape mismatch: {self.root}")
        # ``.done`` is a writer-completion signal.  Its historical digest is
        # intentionally not recomputed on the training read path.
        self._shard_count = shard_count
        self._connection = None
        self._pid = None
        self._cache_size = max(
            1, int(os.environ.get("MIPS_SHARD_CACHE_SIZE", "1"))
        )
        self._cached_shards = OrderedDict()

    def __getstate__(self):
        return {"root": self.root}

    def __setstate__(self, state):
        self.__init__(state["root"])

    def _connect(self):
        pid = os.getpid()
        if self._connection is None or self._pid != pid:
            if self._connection is not None:
                self._connection.close()
            uri = f"file:{os.path.abspath(self.index_path)}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
            self._pid = pid
            self._cached_shards = OrderedDict()
        return self._connection

    def __len__(self):
        row = self._connect().execute(
            "SELECT value FROM metadata WHERE key='count'"
        ).fetchone()
        return int(row[0])

    def __contains__(self, smiles):
        return self._connect().execute(
            "SELECT 1 FROM feature_index WHERE smiles=? LIMIT 1", (str(smiles),)
        ).fetchone() is not None

    def __getitem__(self, smiles):
        row = self._connect().execute(
            "SELECT shard_id, item_offset FROM feature_index WHERE smiles=?",
            (str(smiles),),
        ).fetchone()
        if row is None:
            raise KeyError(smiles)
        shard_id, item_offset = map(int, row)
        if shard_id not in self._cached_shards:
            shard_name = f"shard_{shard_id:06d}.pt"
            shard_path = os.path.join(self.root, shard_name)
            if not os.path.isfile(shard_path):
                raise RuntimeError(f"immutable shard is missing: {shard_path}")
            payload = torch.load(
                shard_path,
                weights_only=False, mmap=True,
            )
            self._cached_shards[shard_id] = payload["features"]
            while len(self._cached_shards) > self._cache_size:
                self._cached_shards.popitem(last=False)
        else:
            self._cached_shards.move_to_end(shard_id)
        stored_smiles, data = self._cached_shards[shard_id][item_offset]
        if stored_smiles != str(smiles):
            raise RuntimeError("sharded feature index/data mismatch")
        return data

    def values(self):
        cursor = self._connect().execute(
            "SELECT smiles FROM feature_index ORDER BY shard_id,item_offset"
        )
        for (smiles,) in cursor:
            yield self[smiles]


class ShardedFeatureBuilder:
    """Single-writer, resumable, atomically committed 2048-item cache."""

    def __init__(self, root, meta, shard_size=2048):
        self.root = str(root)
        self.shard_size = int(shard_size)
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, "index.sqlite3")
        self.connection = sqlite3.connect(self.index_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS feature_index("
            "smiles TEXT PRIMARY KEY, shard_id INTEGER NOT NULL,"
            "item_offset INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        existing_meta = self.connection.execute(
            "SELECT value FROM metadata WHERE key='feature_meta'"
        ).fetchone()
        serialized_meta = json.dumps(meta, sort_keys=True)
        if existing_meta is not None and existing_meta[0] != serialized_meta:
            raise RuntimeError(
                f"incomplete sharded cache has mismatched metadata: {self.root}"
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('feature_meta',?)",
            (serialized_meta,),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT COALESCE(MAX(shard_id),-1),COUNT(*) FROM feature_index"
        ).fetchone()
        self.next_shard = int(row[0]) + 1
        self.count = int(row[1])
        self.buffer = []

    def __contains__(self, smiles):
        if any(item[0] == str(smiles) for item in self.buffer):
            return True
        return self.connection.execute(
            "SELECT 1 FROM feature_index WHERE smiles=? LIMIT 1", (str(smiles),)
        ).fetchone() is not None

    def __setitem__(self, smiles, data):
        if smiles in self:
            return
        self.buffer.append((str(smiles), data))
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def __len__(self):
        return self.count + len(self.buffer)

    def flush(self):
        if not self.buffer:
            return
        shard_id = self.next_shard
        final_path = os.path.join(self.root, f"shard_{shard_id:06d}.pt")
        temporary_path = final_path + ".tmp"
        torch.save(
            {"schema": ShardedFeatureStore.schema, "features": self.buffer},
            temporary_path,
        )
        os.replace(temporary_path, final_path)
        self.connection.executemany(
            "INSERT INTO feature_index(smiles,shard_id,item_offset) VALUES(?,?,?)",
            [
                (smiles, shard_id, offset)
                for offset, (smiles, _) in enumerate(self.buffer)
            ],
        )
        self.count += len(self.buffer)
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('count',?)",
            (str(self.count),),
        )
        self.connection.commit()
        self.buffer = []
        self.next_shard += 1

    def finalize(self, failures, meta):
        self.flush()
        shard_hashes = {}
        for shard_id in range(self.next_shard):
            name = f"shard_{shard_id:06d}.pt"
            path = os.path.join(self.root, name)
            checksum = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            shard_hashes[name] = checksum.hexdigest()
        manifest = {
            "schema": ShardedFeatureStore.schema,
            "shard_size": self.shard_size,
            "shard_count": self.next_shard,
            "count": self.count,
            "failed_count": len(failures),
            "meta": meta,
            "shard_sha256": shard_hashes,
        }
        temporary = os.path.join(self.root, "manifest.json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, os.path.join(self.root, "manifest.json"))
        done_tmp = os.path.join(self.root, ".done.tmp")
        with open(done_tmp, "w", encoding="utf-8") as handle:
            handle.write(hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest() + "\n")
        os.replace(done_tmp, os.path.join(self.root, ".done"))
        self.connection.close()
        return ShardedFeatureStore(self.root)


_LAYER_SCHEMAS = {
    "input": "mips-trimer-scage-input-cache-v1",
    "topology": CANONICAL_TOPOLOGY_SCHEMA,
    "descriptor": "mips-trimer-scage-md200-cache-v1",
    "trimer": TRIMER_MCL_SCHEMA,
}


def _copy_data_fields(data, predicate):
    output = Data()
    for key in data.keys():
        if predicate(str(key)):
            output[key] = copy.deepcopy(data[key])
    return output


def _split_mips_feature_layers(data):
    """Split one sample into independently reusable immutable cache layers."""
    descriptor_names = {
        "mips_md", "mips_md_valid",
        "descriptor_failure_code", "mips_descriptor_source",
        "mips_descriptor_optimizer",
        "mips_descriptor_schema_version",
    }
    input_names = {
        "smiles", "input_ids_smiles", "attention_mask_smiles", "fp",
    }
    trimer_names = {
        "trimer_pos", "trimer_atomic_number", "trimer_edge_index",
        "trimer_bond_type", "trimer_base_ru_atom_id", "trimer_base_ru_atom_index",
        "trimer_ru_offset",
        "trimer_central_ru_mask", "trimer_central_atom_index",
        "trimer_central_ru_atom_index",
        "canonical_to_trimer_central_index", "mips_to_trimer_central_index",
        "trimer_geometry_valid",
        "trimer_geometry_is_3d", "trimer_2d_fallback",
        "trimer_geometry_source",
        "trimer_failure_code", "trimer_conformer_energy",
        "star_3d_distance", "star_3d_asymmetry", "star_3d_valid",
        "trimer_conformer_seed", "trimer_conformer_method",
        "trimer_mcl_schema", "trimer_mcl_schema_version",
        "migration_status", "source_trimer_content_hash",
        "source_trimer_done_hash", "trimer_mapping_digest",
        "geometry_payload_digest", "regeneration_reason",
    }
    layers = {
        "input": _copy_data_fields(data, lambda key: key in input_names),
        "descriptor": _copy_data_fields(data, lambda key: key in descriptor_names),
        "trimer": _copy_data_fields(data, lambda key: key in trimer_names),
    }
    excluded = input_names | descriptor_names | trimer_names
    layers["topology"] = _copy_data_fields(
        data,
        lambda key: (
            key not in excluded
        ),
    )
    return layers


def _scage_cache_metadata_compatible(
    cache_meta,
    *,
    mips_core,
    mips_max_hops,
    spatial_mode,
    feature_config_hash,
):
    """Validate only fields that identify cached tensors.

    ``mips_variant`` is intentionally absent: variants such as B1/B2 alter
    attention scaling, normalization, or activation without changing any
    cached feature. The feature hash is the authoritative contract for
    feature-affecting experiment axes.
    """
    return (
        cache_meta.get("scage_data_schema")
        == MIPS_EXPERIMENT_FEATURE_SCHEMA
        and cache_meta.get("mips_local_lga_schema_version") == 2
        and cache_meta.get("mips_core") == mips_core
        and int(cache_meta.get("mips_max_hops", -1))
        == int(mips_max_hops)
        and cache_meta.get("spatial_mode") == spatial_mode
        and cache_meta.get("feature_config_hash") == feature_config_hash
        and cache_meta.get("graph_placeholder_policy")
        == "preserve_multimodal_row-v1"
    )


class LayeredFeatureStore:
    """Lazy composition of input, topology, descriptor, and Trimer shards."""

    schema = "mips-layered-feature-store-v2"

    def __init__(self, roots):
        self.roots = dict(roots)
        self.stores = {
            name: ShardedFeatureStore(root) for name, root in self.roots.items()
        }
        if "input" not in self.stores or "topology" not in self.stores:
            raise ValueError("layered cache requires input and topology stores")
        sizes = {name: len(store) for name, store in self.stores.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"incomplete layered cache sizes: {sizes}")

    def __getstate__(self):
        return {"roots": self.roots}

    def __setstate__(self, state):
        self.__init__(state["roots"])

    def __len__(self):
        return len(self.stores["input"])

    def __contains__(self, smiles):
        # All immutable layers are written from the same key stream and their
        # equal sizes are checked above. One indexed lookup is sufficient here;
        # __getitem__ still fails loudly if a corrupt layer is missing the key.
        return smiles in self.stores["input"]

    def __getitem__(self, smiles):
        merged = Data()
        for name in ("input", "topology", "descriptor", "trimer"):
            store = self.stores.get(name)
            if store is None:
                continue
            layer = store[smiles]
            for key in layer.keys():
                if key in merged:
                    raise RuntimeError(
                        f"duplicate field {key!r} across MIPS cache layers"
                    )
                # Cached tensors are immutable inputs.  The returned Data
                # container is new and downstream code only adds labels before
                # collating, so cloning every tensor here needlessly copied
                # hundreds of MB per batch and serialized the DDP ranks.
                merged[key] = layer[key]
        return merged

    def values(self):
        for smiles in self.stores["input"]._connect().execute(
            "SELECT smiles FROM feature_index ORDER BY shard_id,item_offset"
        ):
            yield self[smiles[0]]


class LayeredFeatureBuilder:
    """Facade used by the existing single-writer feature build loop."""

    def __init__(self, specs, shard_size=2048):
        self.specs = copy.deepcopy(specs)
        self.parts = {}
        self.trimer_failures = []
        self.trimer_failure_partial = None
        for name, spec in self.specs.items():
            root = spec["root"]
            done = os.path.isfile(os.path.join(root, ".done"))
            self.parts[name] = (
                ShardedFeatureStore(root)
                if done else ShardedFeatureBuilder(
                    root, spec["meta"], shard_size=shard_size
                )
            )
            if name == "trimer" and not done:
                self.trimer_failure_partial = os.path.join(
                    root, "trimer_failures.jsonl.partial"
                )

    def __contains__(self, smiles):
        return all(smiles in part for part in self.parts.values())

    def __setitem__(self, smiles, data):
        layers = _split_mips_feature_layers(data)
        if (
            "trimer" in self.parts
            and not bool(getattr(data, "trimer_geometry_valid", False))
        ):
            failure = {
                "smiles": str(smiles),
                "stage": "trimer_geometry",
                "error": str(
                    getattr(data, "trimer_failure_code", "unknown")
                )[:240],
            }
            self.trimer_failures.append(failure)
            if self.trimer_failure_partial:
                with open(
                    self.trimer_failure_partial, "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
        for name, part in self.parts.items():
            if isinstance(part, ShardedFeatureBuilder):
                part[smiles] = layers[name]

    def __len__(self):
        return min(len(part) for part in self.parts.values())

    def finalize(self, failures, _meta):
        roots = {}
        for name, part in self.parts.items():
            if isinstance(part, ShardedFeatureBuilder):
                part = part.finalize(failures, self.specs[name]["meta"])
            roots[name] = part.root
        if "trimer" in roots and self.trimer_failure_partial:
            failures_by_smiles = {}
            if os.path.isfile(self.trimer_failure_partial):
                with open(
                    self.trimer_failure_partial, encoding="utf-8"
                ) as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            # A process interruption may leave one truncated
                            # tail record. The corresponding sample is rebuilt
                            # on resume and writes a complete replacement.
                            continue
                        failures_by_smiles[item["smiles"]] = item
            path = os.path.join(roots["trimer"], "trimer_failures.json")
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    list(failures_by_smiles.values()), handle,
                    sort_keys=True, indent=2,
                )
                handle.write("\n")
            os.replace(temporary, path)
            os.remove(self.trimer_failure_partial)
        return LayeredFeatureStore(roots)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prune_nonpbc_mips_data(data):
    """Remove dense/PBC tensors that the sparse second route never consumes."""
    unused = (
        "scage_spd", "scage_path_bond_fields", "mips_path_nodes",
        "polymer_ecfp_target", "polymer_ecfp_valid", "polymer_ecfp_source",
        "cell", "cell_confs", "pbc", "screw_rotation",
        "screw_rotation_confs", "screw_translation",
        "screw_translation_confs", "screw_valid", "smer_image_pos",
        "smer_image_pos_confs", "smer_valid", "pos_confs",
        "geometry_period_ru", "periodic_geometry_valid",
        "polygen_periodic_valid", "periodic_valid",
        "periodic_cell_length", "periodic_closure_error",
        "periodic_ru_count", "model_cell_ru",
        "input_ids_smiles", "attention_mask_smiles", "fp",
    )
    for name in unused:
        if hasattr(data, name):
            delattr(data, name)
    compact_int_fields = (
        "lga_edge_index", "lga_path_index",
        "lga_path_shift", "lga_source_image_shift",
        "canonical_ru_atom_index", "canonical_atom_id",
        "canonical_pair_index", "ru_copy_index",
        "source_to_normalized_canonical_atom_id",
        "source_to_canonical_atom_id",
        "canonical_to_trimer_base_atom_id",
        "canonical_to_trimer_base_atom_index",
    )
    for name in compact_int_fields:
        value = getattr(data, name, None)
        if torch.is_tensor(value):
            setattr(data, name, value.to(torch.int32))
    if hasattr(data, "lga_path_bond_hist"):
        data.lga_path_bond_hist = data.lga_path_bond_hist.to(torch.float16)
    for name in (
        "polymer_link_mask", "lga_polymer_link_mask",
        "lga_star_edge_mask", "lga_path_mask",
    ):
        value = getattr(data, name, None)
        if torch.is_tensor(value):
            setattr(data, name, value.to(torch.bool))
    return data


def _bitvect_to_tensor(bitvect, size):
    arr = torch.zeros(size, dtype=torch.float)
    np_arr = arr.numpy()
    DataStructs.ConvertToNumpyArray(bitvect, np_arr)
    return arr


def _pubchem_fingerprint_881(_mol):
    raise RuntimeError(
        "PubChemFingerprints backend is not configured. "
        "--fp_mode mixfp requires a real 881-bit CACTVS/PubChem fingerprint backend; "
        "the project will not substitute Morgan/RDKit fingerprints or zero vectors."
    )


_WORKER_TOKENIZER = None
_WORKER_TOKENIZER_NAME = None
_WORKER_MORGAN_GENERATOR = None
_WORKER_MORGAN_COUNT_R2 = None
_WORKER_MORGAN_COUNT_R3 = None
_WORKER_MORGAN_ROOTED_R3 = None
_WORKER_POLYMER_MORGAN_GENERATOR = None
_WORKER_MIPS_MD_GENERATOR = None


class _FeatureCacheItemTimeout(TimeoutError):
    pass


def _feature_cache_timeout_handler(_signum, _frame):
    raise _FeatureCacheItemTimeout("feature_cache_item_timeout")


def _load_auto_tokenizer(smiles_model_name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(smiles_model_name)


def _get_worker_tokenizer(smiles_model_name):
    global _WORKER_TOKENIZER, _WORKER_TOKENIZER_NAME
    if _WORKER_TOKENIZER is None or _WORKER_TOKENIZER_NAME != smiles_model_name:
        _WORKER_TOKENIZER = _load_auto_tokenizer(smiles_model_name)
        _WORKER_TOKENIZER_NAME = smiles_model_name
    return _WORKER_TOKENIZER


def _get_worker_morgan_generator():
    global _WORKER_MORGAN_GENERATOR
    if _WORKER_MORGAN_GENERATOR is None:
        _WORKER_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return _WORKER_MORGAN_GENERATOR


def _get_worker_count_generators():
    global _WORKER_MORGAN_COUNT_R2, _WORKER_MORGAN_COUNT_R3
    global _WORKER_MORGAN_ROOTED_R3
    if _WORKER_MORGAN_COUNT_R2 is None:
        _WORKER_MORGAN_COUNT_R2 = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024
        )
        _WORKER_MORGAN_COUNT_R3 = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=1024
        )
        _WORKER_MORGAN_ROOTED_R3 = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=512
        )
    return (
        _WORKER_MORGAN_COUNT_R2,
        _WORKER_MORGAN_COUNT_R3,
        _WORKER_MORGAN_ROOTED_R3,
    )


def _count_fp_to_tensor(fingerprint, size, apply_log=True):
    output = torch.zeros(int(size), dtype=torch.float)
    array = output.numpy()
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return torch.log1p(output) if apply_log else output


def _attachment_count_fingerprint(original_mol):
    """Direction-invariant count fingerprint with attachment context."""
    dummy_atoms = [
        atom.GetIdx() for atom in original_mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    if len(dummy_atoms) != 2:
        raise ValueError("attachment_count requires exactly two attachment atoms")
    boundary_atoms = []
    for dummy_idx in dummy_atoms:
        neighbors = list(original_mol.GetAtomWithIdx(dummy_idx).GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError("each attachment atom must have one boundary neighbor")
        boundary_atoms.append(int(neighbors[0].GetIdx()))

    # Keep the two atomic-number-zero attachment atoms in the fingerprint
    # molecule. Replacing ``*=`` by hydrogen creates an impossible double-bonded
    # hydrogen and used to turn valid P-SMILES into cache tombstones. Morgan
    # fingerprints support dummy atoms directly and therefore preserve both
    # attachment chemistry and the original atom indices used by rooted counts.
    fp_mol = Chem.Mol(original_mol)
    r2, r3, rooted = _get_worker_count_generators()
    global_r2 = _count_fp_to_tensor(r2.GetCountFingerprint(fp_mol), 1024)
    global_r3 = _count_fp_to_tensor(r3.GetCountFingerprint(fp_mol), 1024)
    rooted_sum = torch.zeros(512, dtype=torch.float)
    for boundary in boundary_atoms:
        rooted_sum += _count_fp_to_tensor(
            rooted.GetCountFingerprint(fp_mol, fromAtoms=[int(boundary)]),
            512,
            apply_log=False,
        )
    rooted_sum = torch.log1p(rooted_sum)

    # Compute the boundary path before any atom-removal operation so the
    # original boundary indices remain valid.
    backbone_distance = (
        0
        if boundary_atoms[0] == boundary_atoms[1]
        else len(
            Chem.GetShortestPath(
                fp_mol, boundary_atoms[0], boundary_atoms[1]
            )
        ) - 1
    )
    boundary_degree = sum(
        fp_mol.GetAtomWithIdx(index).GetDegree() for index in boundary_atoms
    )
    capped = Chem.RemoveHs(fp_mol)
    heavy = sum(atom.GetAtomicNum() > 1 for atom in capped.GetAtoms())
    hetero = sum(atom.GetAtomicNum() not in (1, 6) for atom in capped.GetAtoms())
    carbon = sum(atom.GetAtomicNum() == 6 for atom in capped.GetAtoms())
    ring_info = capped.GetRingInfo()
    rings = int(ring_info.NumRings())
    aromatic_rings = sum(
        all(capped.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring)
        for ring in ring_info.AtomRings()
    )
    rotatable = int(rdMolDescriptors.CalcNumRotatableBonds(capped))
    charge = sum(atom.GetFormalCharge() for atom in capped.GetAtoms())
    scalars = torch.tensor([
        math.log1p(heavy),
        math.log1p(backbone_distance + 1),
        math.log1p(rings),
        math.log1p(aromatic_rings),
        math.log1p(rotatable),
        float(hetero) / max(1, heavy),
        max(-4.0, min(4.0, float(charge))) / 4.0,
        math.log1p(rdMolDescriptors.CalcExactMolWt(capped)) / 8.0,
        float(boundary_degree) / 8.0,
        float(carbon) / max(1, heavy),
    ], dtype=torch.float)
    output = torch.cat([global_r2, global_r3, rooted_sum, scalars], dim=0)
    if output.numel() != FP_MODE_DIMS["attachment_count"]:
        raise RuntimeError("attachment_count fingerprint width mismatch")
    return output


def _get_worker_polymer_morgan_generator():
    global _WORKER_POLYMER_MORGAN_GENERATOR
    if _WORKER_POLYMER_MORGAN_GENERATOR is None:
        _WORKER_POLYMER_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
    return _WORKER_POLYMER_MORGAN_GENERATOR


def _get_worker_mips_md_generator():
    global _WORKER_MIPS_MD_GENERATOR
    if _WORKER_MIPS_MD_GENERATOR is None:
        from .mips_descriptors.rdNormalizedDescriptors import RDKit2DNormalized
        _WORKER_MIPS_MD_GENERATOR = RDKit2DNormalized()
    return _WORKER_MIPS_MD_GENERATOR


def _mips_source_star_sub_molecule(repeating_monomer):
    """Return the source-star-sub molecule used by graph-level MD200.

    MD200 is a 2D descriptor.  Unlike the retired AtomPair3D path, this
    production route deliberately performs no embedding or force-field step.
    """
    source = Chem.RWMol(Chem.Mol(repeating_monomer))
    dummy_atoms = [
        atom.GetIdx() for atom in source.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    if len(dummy_atoms) != 2:
        raise ValueError("MIPS source descriptors require two wildcards")
    neighbors = []
    for dummy_idx in dummy_atoms:
        atom_neighbors = list(source.GetAtomWithIdx(dummy_idx).GetNeighbors())
        if len(atom_neighbors) != 1:
            raise ValueError("wildcard must have exactly one neighbor")
        neighbors.append(atom_neighbors[0].GetIdx())
    source.GetAtomWithIdx(dummy_atoms[0]).SetAtomicNum(
        source.GetAtomWithIdx(neighbors[1]).GetAtomicNum()
    )
    source.GetAtomWithIdx(dummy_atoms[1]).SetAtomicNum(
        source.GetAtomWithIdx(neighbors[0]).GetAtomicNum()
    )
    source = source.GetMol()
    Chem.SanitizeMol(source)
    return source


def _attach_mips_descriptors(
    data, repeating_monomer, protocol="source_star_sub"
):
    """Attach MIPS descriptors computed from the original repeating monomer."""
    protocol = str(protocol)
    if protocol != "source_star_sub":
        raise ValueError(
            "selected O8 route requires descriptor_protocol=source_star_sub"
        )
    data.mips_md = torch.zeros(200, dtype=torch.float)
    data.mips_md_valid = False
    data.descriptor_failure_code = ""
    data.mips_descriptor_source = protocol
    data.mips_descriptor_optimizer = "not_applicable_2d"
    try:
        descriptor_mol = _mips_source_star_sub_molecule(repeating_monomer)
        descriptor_smiles = Chem.MolToSmiles(descriptor_mol, canonical=True)
        md = _get_worker_mips_md_generator().process(descriptor_smiles)
        if md is None or len(md) != 201:
            raise ValueError("RDKit2DNormalized did not return 200 descriptors")
        md_tensor = torch.as_tensor(md[1:], dtype=torch.float)
        if not torch.isfinite(md_tensor).all():
            raise ValueError("RDKit2DNormalized contains non-finite values")

        data.mips_md = md_tensor
        data.mips_md_valid = True
    except Exception as exc:
        data.descriptor_failure_code = str(exc)[:300]
    # Kept as a generic provenance string; AP3D is intentionally absent.
    data.mips_descriptor_schema_version = 5
    return data


def _attach_unavailable_mips_descriptors(
    data, reason="source_star_sub_input_unavailable"
):
    data.mips_md = torch.zeros(200, dtype=torch.float)
    data.mips_md_valid = False
    data.descriptor_failure_code = str(reason)[:300]
    data.mips_descriptor_source = "source_star_sub"
    data.mips_descriptor_optimizer = "none"
    data.mips_descriptor_schema_version = 5
    return data


def _attach_polymer_ecfp_target(data, smiles):
    """Attach an M4P-only ECFP target generated from an H-capped trimer."""
    data.polymer_ecfp_target = torch.zeros(2048, dtype=torch.float)
    data.polymer_ecfp_valid = False
    data.polymer_ecfp_source = "invalid"
    data.polymer_ecfp_failed_reason = ""
    try:
        trimer_smiles = generate_multimer_smiles(
            num_repeat_units=3,
            smiles=smiles,
            replace_dummy_atoms=True,
        )
        trimer_mol = Chem.MolFromSmiles(trimer_smiles)
        if trimer_mol is None:
            raise ValueError("capped_3mer_parse_failed")
        target = _bitvect_to_tensor(
            _get_worker_polymer_morgan_generator().GetFingerprint(trimer_mol), 2048
        )
        data.polymer_ecfp_target = target
        data.polymer_ecfp_valid = True
        data.polymer_ecfp_source = "capped_3mer"
    except Exception as exc:
        data.polymer_ecfp_failed_reason = str(exc)[:200]
    return data


def _finite_descriptor(values, expected_dim):
    tensor = torch.as_tensor(list(values), dtype=torch.float).flatten()
    if tensor.numel() != int(expected_dim) or not torch.isfinite(tensor).all():
        raise ValueError(
            f"invalid descriptor dimension/value: expected {expected_dim}, got {tensor.numel()}"
        )
    return tensor


def _descriptor_values(mol, conf_id):
    shape = [
        rdMolDescriptors.CalcPMI1(mol, confId=conf_id),
        rdMolDescriptors.CalcPMI2(mol, confId=conf_id),
        rdMolDescriptors.CalcPMI3(mol, confId=conf_id),
        rdMolDescriptors.CalcNPR1(mol, confId=conf_id),
        rdMolDescriptors.CalcNPR2(mol, confId=conf_id),
        rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id),
        rdMolDescriptors.CalcInertialShapeFactor(mol, confId=conf_id),
        rdMolDescriptors.CalcEccentricity(mol, confId=conf_id),
        rdMolDescriptors.CalcAsphericity(mol, confId=conf_id),
        rdMolDescriptors.CalcSpherocityIndex(mol, confId=conf_id),
        rdMolDescriptors.CalcPBF(mol, confId=conf_id),
    ]
    return {
        "shape": _finite_descriptor(shape, 11),
        "usrcat": _finite_descriptor(rdMolDescriptors.GetUSRCAT(mol, confId=conf_id), 60),
        "autocorr3d": _finite_descriptor(
            rdMolDescriptors.CalcAUTOCORR3D(mol, confId=conf_id), 80
        ),
        "rdf": _finite_descriptor(rdMolDescriptors.CalcRDF(mol, confId=conf_id), 210),
        "morse": _finite_descriptor(rdMolDescriptors.CalcMORSE(mol, confId=conf_id), 224),
        "whim": _finite_descriptor(rdMolDescriptors.CalcWHIM(mol, confId=conf_id), 114),
    }


def _attach_scage_3d_descriptors(data, graph_mol):
    """Compute explicit MIPS-style 3D descriptor groups per conformer."""
    num_confs = int(data.pos_confs.size(0)) if hasattr(data, "pos_confs") else 0
    values = {
        name: torch.zeros((num_confs, dim), dtype=torch.float)
        for name, dim in SCAGE_DESCRIPTOR_DIMS.items()
    }
    valid = torch.zeros(num_confs, dtype=torch.bool)
    coordinate_atoms = int(data.pos_confs.size(1)) if num_confs else 0
    descriptor_source = Chem.Mol(graph_mol)
    if descriptor_source.GetNumAtoms() != coordinate_atoms:
        with_hydrogens = Chem.AddHs(descriptor_source)
        if with_hydrogens.GetNumAtoms() == coordinate_atoms:
            descriptor_source = with_hydrogens
    if num_confs == 0:
        data.scage_descriptor_failed_reason = "missing_conformers"
    else:
        source_matches = descriptor_source.GetNumAtoms() == coordinate_atoms
        failures = []
        for conf_idx, coordinates in enumerate(data.pos_confs):
            try:
                if source_matches:
                    descriptor_mol = Chem.Mol(descriptor_source)
                else:
                    editable = Chem.RWMol()
                    for atomic_num in data.z.tolist():
                        editable.AddAtom(Chem.Atom(int(atomic_num)))
                    descriptor_mol = editable.GetMol()
                descriptor_mol.RemoveAllConformers()
                conformer = Chem.Conformer(descriptor_mol.GetNumAtoms())
                for atom_idx, xyz in enumerate(coordinates.tolist()):
                    conformer.SetAtomPosition(atom_idx, tuple(float(value) for value in xyz))
                rd_conf_id = descriptor_mol.AddConformer(conformer, assignId=True)
                if not source_matches:
                    rdDetermineBonds.DetermineConnectivity(descriptor_mol)
                    descriptor_mol.UpdatePropertyCache(strict=False)
                groups = _descriptor_values(descriptor_mol, rd_conf_id)
                for name, tensor in groups.items():
                    values[name][conf_idx] = tensor
                valid[conf_idx] = True
            except Exception as exc:
                failures.append(f"conf{conf_idx}:{str(exc)[:100]}")
        data.scage_descriptor_failed_reason = ";".join(failures)[:500]
    for name, tensor in values.items():
        setattr(data, f"scage_descriptor_{name}_confs", tensor)
    data.scage_descriptor_valid_confs = valid
    data.scage_descriptor_schema_version = 1
    return data


def _standardize_scage_descriptors(features):
    """Standardize descriptor groups using valid conformers in this cache."""
    statistics = {}
    for name, dim in SCAGE_DESCRIPTOR_DIMS.items():
        field = f"scage_descriptor_{name}_confs"
        collected = []
        for data in features.values():
            valid = getattr(data, "scage_descriptor_valid_confs", None)
            if hasattr(data, field) and valid is not None and valid.bool().any():
                collected.append(getattr(data, field)[valid.bool()].double())
        if collected:
            merged = torch.cat(collected, dim=0)
            merged = torch.nan_to_num(merged, nan=0.0, posinf=1e30, neginf=-1e30)
            mean = merged.mean(dim=0)
            std = merged.std(dim=0, unbiased=False)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e30, neginf=-1e30)
            std = torch.nan_to_num(std, nan=1.0, posinf=1e30, neginf=1e30).clamp_min(1e-6)
        else:
            mean = torch.zeros(dim, dtype=torch.double)
            std = torch.ones(dim, dtype=torch.double)
        for data in features.values():
            if not hasattr(data, field):
                continue
            normalized = (getattr(data, field).double() - mean) / std
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=20.0, neginf=-20.0)
            normalized = normalized.clamp(-20.0, 20.0).float()
            valid = getattr(
                data, "scage_descriptor_valid_confs",
                torch.zeros(normalized.size(0), dtype=torch.bool),
            ).bool()
            normalized[~valid] = 0.0
            setattr(data, field, normalized)
        statistics[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return statistics


def _compute_fingerprint_for_mode(fp_mol, fp_mode, original_mol=None):
    fp_mode = str(fp_mode).lower()
    if fp_mode == "disabled":
        return torch.empty(0, dtype=torch.float)
    if fp_mode == "ecfp":
        return _bitvect_to_tensor(_get_worker_morgan_generator().GetFingerprint(fp_mol), 1024)
    if fp_mode == "mixfp":
        maccs = _bitvect_to_tensor(MACCSkeys.GenMACCSKeys(fp_mol), 167)
        pubchem = _pubchem_fingerprint_881(fp_mol)
        if not isinstance(pubchem, torch.Tensor):
            pubchem = torch.as_tensor(pubchem, dtype=torch.float)
        pubchem = pubchem.flatten().to(dtype=torch.float)
        if pubchem.numel() != 881:
            raise ValueError(f"PubChemFingerprints must be 881-bit, got {pubchem.numel()}")
        return torch.cat([maccs, pubchem], dim=0)
    if fp_mode == "attachment_count":
        return _attachment_count_fingerprint(
            original_mol if original_mol is not None else fp_mol
        )
    raise ValueError(f"Unsupported fp_mode: {fp_mode}")


def _data_to_pickle_payload(data):
    payload = {}
    for key in data.keys():
        value = data[key]
        if torch.is_tensor(value):
            payload[key] = {
                "kind": "tensor",
                "value": value.detach().cpu().contiguous().numpy(),
            }
        else:
            payload[key] = {
                "kind": "value",
                "value": value,
            }
    return payload


def _pickle_payload_to_data(payload):
    data = Data()
    for key, item in payload.items():
        if item.get("kind") == "tensor":
            # The worker payload can preserve NumPy views that share one byte
            # buffer across fields with different dtypes. PyTorch refuses to
            # serialize such tensors. Clone at the IPC boundary so every
            # cached field owns a dtype-consistent storage.
            setattr(data, key, torch.from_numpy(item["value"]).clone())
        else:
            setattr(data, key, item.get("value"))
    return data


def _detach_data_storages(data):
    """Make every tensor field own independent, dtype-consistent CPU storage."""
    for key in data.keys():
        value = data[key]
        if torch.is_tensor(value):
            data[key] = value.detach().cpu().contiguous().clone()
    return data


def _geometry_for_mode(*_args, **_kwargs):
    """Reject the removed parallel-coordinate route explicitly."""
    raise ValueError(
        "MIPS-Trimer-SCAGE uses the finite Trimer cache; coordinate geometry "
        "must be provided by trimer_mcl"
    )


def _topology_geometry_placeholder(mol):
    """Return the cheap, non-coordinate geometry contract for graph-only MIPS."""
    data = Data()
    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
    )
    data.z = atomic_numbers
    data.pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
    data.pos_confs = data.pos.unsqueeze(0)
    data.geom_pool_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    data.graph_to_geom_index = torch.arange(mol.GetNumAtoms(), dtype=torch.long)
    data.geom_input = "topology_only"
    data.geom_context = "topology_only"
    data.geom_context_id = 2
    data.geom_build_ok = False
    data.geom_coordinate_ok = False
    data.geom_failed_reason = "coordinates_disabled_for_non_pbc_mips"
    data.geom_num_confs = 1
    data.geom_conformer_energies = torch.empty(0)
    data.geom_conformer_candidate_count = 0
    data.geom_conformer_converged_count = 0
    data.geom_optimizer_counts = {}
    data.geom_optimizer_used = "none"
    return data


def _attach_geometry_data(data, geom_data, smiles, geom_input, geom_optimizer):
    data.pos = geom_data.pos
    data.z = geom_data.z
    data.pos_confs = geom_data.pos_confs
    if hasattr(geom_data, "cell"):
        data.cell = geom_data.cell
    if hasattr(geom_data, "cell_confs"):
        data.cell_confs = geom_data.cell_confs
    if hasattr(geom_data, "pbc"):
        data.pbc = geom_data.pbc
    if hasattr(geom_data, "geom_pool_mask"):
        data.geom_pool_mask = geom_data.geom_pool_mask
    if hasattr(geom_data, "graph_to_geom_index"):
        data.graph_to_geom_index = geom_data.graph_to_geom_index
    data.geom_smiles = smiles
    data.geom_requested_input = geom_input
    data.geom_input = getattr(geom_data, "geom_input", "star_substitution")
    data.geom_context = getattr(geom_data, "geom_context", data.geom_input)
    data.geom_optimizer = geom_optimizer
    data.geom_optimizer_used = getattr(geom_data, "geom_optimizer_used", geom_optimizer)
    data.geom_build_ok = bool(getattr(geom_data, "geom_build_ok", True))
    data.geom_coordinate_ok = bool(getattr(geom_data, "geom_coordinate_ok", data.geom_build_ok))
    data.geom_failed_reason = getattr(geom_data, "geom_failed_reason", "")
    data.geom_num_confs = int(getattr(geom_data, "geom_num_confs", data.pos_confs.size(0)))
    data.geom_context_id = int(getattr(geom_data, "geom_context_id", 2))
    data.geom_conformer_energies = getattr(geom_data, "geom_conformer_energies", torch.empty(0))
    data.geom_conformer_candidate_count = int(getattr(geom_data, "geom_conformer_candidate_count", 0))
    data.geom_conformer_converged_count = int(getattr(geom_data, "geom_conformer_converged_count", 0))
    data.geom_optimizer_counts = getattr(geom_data, "geom_optimizer_counts", {})
    for name in (
        "geom_t_method",
        "geom_pbc_status",
        "geom_polygen_seed_mode",
        "geom_t_left_norm",
        "geom_t_right_norm",
        "geom_t_cosine_similarity",
        "geom_t_relative_length_difference",
        "geom_attachment_bond_lengths",
        "geom_attachment_bond_ratios",
        "screw_rotation", "screw_translation", "screw_rotation_confs",
        "screw_translation_confs", "screw_valid", "smer_valid", "geom_periodic_mode",
        "smer_image_pos", "smer_image_pos_confs",
        "geom_force_quality_status", "geom_gradient_rms", "geom_gradient_max",
        "geom_probe_energy_delta_per_atom", "geom_kabsch_rmsd_left",
        "geom_kabsch_rmsd_right", "geom_final_screw_rmsd_left",
        "geom_final_screw_rmsd_right", "geom_rotation_consistency_deg",
        "geom_translation_relative_difference", "geom_screw_axis",
        "geom_joint_screw_rmsd",
        "geom_screw_angle", "geom_screw_axial_rise",
        "geom_screw_torsion_degrees", "geom_screw_energy_per_atom",
        "geom_screw_symmetry_rmsd",
        "geom_minimum_nonbonded_distance",
        "geom_five_cell_minimum_distance", "geom_screw_source",
        "geom_screw_source_id", "geometry_source_id",
        "geom_screw_fit_point_count", "geom_primary_failed_reason",
        "polygen_periodic_valid", "periodic_valid", "periodic_closure_error",
        "periodic_ru_count", "periodic_cell_length",
        "periodic_geometry_valid", "geometry_period_ru", "model_cell_ru",
        "periodic_fractional_pos", "periodic_fractional_pos_confs",
        "periodic_optimization_loss", "periodic_boundary_bond_error",
        "periodic_boundary_angle_error_deg", "periodic_boundary_torsion_error_deg",
        "periodic_minimum_nonbonded_distance", "periodic_torsion_start_deg",
        "periodic_candidate_count", "periodic_failure_counts",
    ):
        if hasattr(geom_data, name):
            setattr(data, name, getattr(geom_data, name))
    return data


def _attach_geometry_for_mode(data, mol, smiles, geom_input, geom_optimizer):
    raise ValueError(
        "the removed standalone geometry encoder is not part of MIPS-Trimer-SCAGE"
    )


def _structure_for_encoder(
    smiles, graph_input, geom_input, geom_data, graph_encoder_type,
    mips_core="topology_plus", mips_max_hops=None,
):
    # ``scage`` is accepted only as an in-process compatibility spelling for
    # old unit callers.  The CLI and all persisted production metadata use the
    # explicit MTS route name.
    if str(graph_encoder_type).lower() in {"scage", "mts"}:
        graph_encoder_type = "mips_trimer_scage"
    if str(graph_encoder_type).lower() != "mips_trimer_scage":
        raise ValueError(
            "Only graph_encoder_type='mips_trimer_scage' is supported"
        )
    if str(graph_encoder_type).lower() == "mips_trimer_scage":
        core = str(mips_core)
        max_hops = int(
            mips_max_hops
            if mips_max_hops is not None
            else (2 if core == "paper_corrected" else 5)
        )
        return build_mips_local_structure(
            smiles, config=MIPSLocalConfig(max_hops=max_hops)
        )
    raise ValueError("only MIPS-Trimer-SCAGE topology construction is available")


def _scage_graph_with_unavailable_fallback(
    smiles, graph_input, geom_input, geom_data, mips_core, mips_max_hops
):
    """Build one SCAGE graph without ever dropping the SMILES/FP sample.

    A few PI1M rows are valid enough for tokenization/fingerprints but contain
    uncommon coordination bonds or aromatic systems that the repeated-chain
    RDKit builder cannot sanitize.  Such failures are intrinsic Graph
    unavailability, not grounds for deleting the complete multimodal row.
    """
    max_hops = int(
        mips_max_hops
        if mips_max_hops is not None
        else (2 if str(mips_core) == "paper_corrected" else 5)
    )
    config = MIPSLocalConfig(max_hops=max_hops)
    failure_reason = ""
    try:
        structure = _structure_for_encoder(
            smiles, graph_input, geom_input, geom_data, "mips_trimer_scage",
            mips_core=mips_core, mips_max_hops=max_hops,
        )
        # ``build_mips_local_structure`` can return an unavailable structure
        # instead of raising (for example, both attachment dummies may share
        # one boundary atom).  Do not pass that original RU to the
        # polymerized-feature Trimer builder: it is invalid for exactly the
        # same reason.  Convert every unavailable result to the deterministic
        # shape-compatible placeholder used by exceptional failures.
        if not bool(structure.get("graph_available", False)):
            reason = str(
                structure.get(
                    "topology_failure_code", "mips_local_structure_unavailable"
                )
            )
            raise ValueError(reason)
        data = build_mips_data_object(
            structure["structure_mol"],
            backbone_info=structure.get("backbone_info"),
        )
    except Exception as exc:
        failure_reason = (
            f"graph_build_failed_placeholder:{type(exc).__name__}:{exc}"
        )[:500]
        # A deterministic two-atom RU is used only to supply shape-compatible
        # tensors.  Availability masking makes its representation exactly
        # unavailable to fusion and all Graph pretraining objectives.
        structure = build_mips_local_structure("*CC*", config=config)
        structure = dict(structure)
        structure.update({
            "graph_build_ok": False,
            "graph_available": False,
            "mips_condition_valid": False,
            "topology_failure_code": failure_reason,
            "structure_input": "mips_local_unavailable_placeholder",
        })
        data = build_mips_data_object(
            structure["structure_mol"],
            backbone_info=structure.get("backbone_info"),
        )
    annotate_structure_fields(data, structure, prefix="graph")
    attach_mips_local_lga(data, structure, config=config)
    attach_polymerized_mips_atom_features(
        data, smiles if not failure_reason else "*CC*"
    )
    if failure_reason:
        data.graph_available = False
        data.mips_condition_valid = False
        data.mips_alias_free = False
        data.topology_failure_code = failure_reason
    return structure, data, failure_reason


def _compute_smiles_features_from_config(
    smiles,
    smiles_model_name,
    max_smiles_length,
    graph_input,
    geom_input,
    fp_mode,
    embed_tries_multiplier=8,
    conformer_3d_count=8,
    conformer_keep_count=4,
    conformer_profile="full",
    graph_encoder_type="mips_trimer_scage",
    mips_core="topology_plus",
    mips_max_hops=None,
    mips_use_descriptors=False,
    mips_descriptor_protocol="source_star_sub",
    spatial_mode="trimer_scage",
    finite_variant="none",
    conformer_mode="none",
    field_layout="none",
    field_channels="none",
    graph_geometry_mode="trimer_scage_mcl",
    topology_representation=TOPOLOGY_CANONICAL,
    trimer_num_candidates=4,
    trimer_max_heavy_atoms=384,
):
    if str(graph_encoder_type).lower() == "scage":
        graph_encoder_type = "mips_trimer_scage"
    if str(graph_encoder_type).lower() == "mips_trimer_scage":
        # Native non-LMDB construction follows the same canonical periodic
        # path as the production layer builder.  The historical explicit
        # finite-k reference lives only under tests/reference_mts_explicit.py.
        ru_base = _compute_ru_base_layer(smiles)
        data = _compute_topology_layer(
            smiles, ru_base,
            max_hops=int(
                mips_max_hops if mips_max_hops is not None else 2
            ),
            topology_representation=topology_representation,
        )
        data.smiles = str(smiles)
        data.mips_atom_feature_source = "topology_only_trimer_central_ru"
        if str(graph_geometry_mode) == "trimer_scage_mcl":
            trimer = _compute_trimer_layer(
                getattr(
                    ru_base, "normalized_polymer_smiles", str(smiles)
                ),
                ru_base, data,
                num_candidates=int(trimer_num_candidates),
                max_heavy_atoms=int(trimer_max_heavy_atoms),
            )
            for name in trimer.keys():
                data[name] = trimer[name]
        elif str(graph_geometry_mode) == "trimer_unavailable":
            attach_unavailable_trimer_mcl(data, "feature_timeout_fallback")
        if bool(mips_use_descriptors):
            descriptor = _compute_md200_layer(smiles, ru_base)
            for name in descriptor.keys():
                data[name] = descriptor[name]
        else:
            _attach_unavailable_mips_descriptors(
                data, "descriptor_layer_reused"
            )
        return _prune_nonpbc_mips_data(data)
    mol = Chem.MolFromSmiles(smiles)
    chemistry_valid = mol is not None
    canonical_polymer_smiles = (
        Chem.MolToSmiles(mol, canonical=True)
        if chemistry_valid else "*CC*"
    )
    if not chemistry_valid:
        if str(graph_encoder_type).lower() != "mips_trimer_scage":
            raise ValueError(f"Invalid SMILES: {smiles}")
        # PI1M contains a small number of strings whose P/Si valence is
        # rejected by the installed RDKit. Keep those rows in the cohort: the
        # original string still supplies the SMILES view, while Graph/FP/3D use
        # explicit unavailable placeholders instead of deleting the sample.
        mol = Chem.MolFromSmiles("*C*")
        if mol is None:  # pragma: no cover - fixed internal template
            raise RuntimeError("failed to construct non-PBC placeholder")

    fp_mol = Chem.Mol(mol)
    for atom in fp_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)

    geom_data = (
        _topology_geometry_placeholder(mol)
        if str(graph_encoder_type).lower() == "mips_trimer_scage"
        else _geometry_for_mode(mol, geom_input, geom_optimizer="auto")
    )
    graph_fallback_reason = ""
    if str(graph_encoder_type).lower() == "mips_trimer_scage":
        structure, data, graph_fallback_reason = (
            _scage_graph_with_unavailable_fallback(
                canonical_polymer_smiles,
                graph_input, geom_input, geom_data, mips_core, mips_max_hops,
            )
        )
        geom_data = _topology_geometry_placeholder(structure["structure_mol"])
    else:
        structure = _structure_for_encoder(
            smiles, graph_input, geom_input, geom_data, graph_encoder_type,
            mips_core=mips_core, mips_max_hops=mips_max_hops,
        )
        data = build_mips_data_object(
            structure["structure_mol"],
            backbone_info=structure.get("backbone_info"),
        )
        annotate_structure_fields(data, structure, prefix="graph")
    data.smiles = smiles

    if (
        str(graph_encoder_type).lower() == "mips_trimer_scage"
        and str(fp_mode).lower() == "disabled"
    ):
        # MTS is graph-only. Keep shape-safe empty tensors so
        # the shared collator does not tokenize or materialize FP features.
        data.input_ids_smiles = torch.empty((1, 0), dtype=torch.long)
        data.attention_mask_smiles = torch.empty((1, 0), dtype=torch.long)
    else:
        tokenizer = _get_worker_tokenizer(smiles_model_name)
        tokenizer_output = tokenizer(
            smiles,
            return_tensors='pt',
            max_length=max_smiles_length,
            padding='max_length',
            truncation=True,
        )
        data.input_ids_smiles = tokenizer_output.input_ids
        data.attention_mask_smiles = tokenizer_output.attention_mask
    data.fp = (
        _compute_fingerprint_for_mode(fp_mol, fp_mode, original_mol=mol)
        if chemistry_valid
        else torch.zeros(FP_MODE_DIMS.get(str(fp_mode).lower(), 0))
    ).unsqueeze(0)
    if str(graph_encoder_type).lower() != "mips_trimer_scage":
        _attach_polymer_ecfp_target(data, smiles)
    _attach_geometry_data(data, geom_data, smiles, geom_input, geom_optimizer="auto")
    if (
        graph_encoder_type == "mips_trimer_scage"
        and chemistry_valid
        and bool(mips_use_descriptors)
    ):
        _attach_mips_descriptors(
            data, mol, protocol=mips_descriptor_protocol
        )
    elif graph_encoder_type == "mips_trimer_scage":
        _attach_unavailable_mips_descriptors(
            data,
            (
                "rdkit_parse_failed"
                if not chemistry_valid
                else "descriptor_layer_reused"
            ),
        )
    elif graph_encoder_type == "mips":
        _attach_mips_descriptors(
            data, mol, protocol=mips_descriptor_protocol
        )
    if not chemistry_valid or graph_fallback_reason:
        data.graph_available = False
        data.mips_condition_valid = False
        data.mips_alias_free = False
        data.topology_failure_code = (
            "rdkit_parse_failed_placeholder"
            if not chemistry_valid else graph_fallback_reason
        )
    if str(graph_geometry_mode) == "trimer_scage_mcl":
        attach_finite_trimer_mcl(
            data,
            canonical_polymer_smiles,
            num_candidates=int(trimer_num_candidates),
            max_heavy_atoms=int(trimer_max_heavy_atoms),
        )
    elif str(graph_geometry_mode) == "trimer_unavailable":
        attach_unavailable_trimer_mcl(data, "feature_timeout_fallback")
    return (
        _prune_nonpbc_mips_data(data)
        if graph_encoder_type == "mips_trimer_scage" else data
    )


_LMDB_TRIMER_FIELDS = {
    "trimer_pos", "trimer_atomic_number", "trimer_edge_index",
    "trimer_bond_type", "trimer_base_ru_atom_id", "trimer_base_ru_atom_index",
    "trimer_ru_offset",
    "trimer_central_ru_mask", "trimer_central_atom_index",
    "trimer_central_ru_atom_index",
    "canonical_to_trimer_central_index", "mips_to_trimer_central_index",
    "trimer_geometry_valid",
    "trimer_geometry_is_3d", "trimer_2d_fallback",
    "trimer_geometry_source", "trimer_failure_code",
    "trimer_conformer_energy", "star_3d_distance",
    "star_3d_asymmetry", "star_3d_valid", "trimer_conformer_seed",
    "trimer_conformer_method", "trimer_mcl_schema",
    "trimer_mcl_schema_version",
    "migration_status", "source_trimer_content_hash",
    "source_trimer_done_hash", "trimer_mapping_digest",
    "geometry_payload_digest", "regeneration_reason",
}


def _select_data_fields(data, names):
    """Create a new Data container without cloning immutable tensor storage."""

    selected = Data()
    for name in names:
        if hasattr(data, name):
            selected[name] = getattr(data, name)
    return selected


def _compute_ru_base_layer(smiles):
    """Parse and validate one RU once for all downstream cache producers."""

    normalized, chemistry_valid = normalize_polymer_smiles(smiles)
    try:
        molecule = Chem.MolFromSmiles(str(smiles)) if chemistry_valid else None
    except Exception:
        molecule = None
        chemistry_valid = False
        normalized = f"INVALID::{smiles}"
    # The RU-base artifact is an existing frozen dependency.  Preserve its
    # original molecule binary exactly; canonical-periodic topology/Trimer
    # builders reparse the normalized identity locally when they need
    # translation/reversal invariance.
    effective = Chem.Mol(molecule) if molecule is not None else Chem.MolFromSmiles("*CC*")
    if effective is None:  # pragma: no cover - fixed internal fallback
        raise RuntimeError("failed to create RU base placeholder")
    data = Data()
    data.smiles = str(smiles)
    data.normalized_polymer_smiles = str(normalized)
    data.ru_chemistry_valid = bool(chemistry_valid)
    data.ru_mol_binary = bytes(effective.ToBinary())
    data.ru_base_failure_code = "" if chemistry_valid else "rdkit_parse_failed"
    try:
        base, metadata = build_periodic_multimer_mol(
            effective, num_repeat_units=1, close_periodic=False
        )
        data.ru_atomic_number = torch.tensor(
            [atom.GetAtomicNum() for atom in base.GetAtoms()], dtype=torch.long
        )
        sources, targets, bond_types = [], [], []
        for bond in base.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            code = int(round(float(bond.GetBondTypeAsDouble())))
            sources.extend((begin, end))
            targets.extend((end, begin))
            bond_types.extend((code, code))
        data.ru_edge_index = torch.tensor(
            [sources, targets], dtype=torch.long
        ) if sources else torch.empty((2, 0), dtype=torch.long)
        data.ru_bond_type = torch.tensor(bond_types, dtype=torch.long)
        data.ru_left_boundary = int(metadata["left_boundary"])
        data.ru_right_boundary = int(metadata["right_boundary"])
        data.ru_backbone = torch.tensor(
            metadata["backbone_base"], dtype=torch.long
        )
        data.ru_canonical_atom_index = torch.arange(
            int(metadata["base_atom_count"]), dtype=torch.long
        )
        data.ru_attachment_bond_type = str(metadata["attachment_bond_type"])
        data.ru_attachment_bond_type_left = str(
            metadata["attachment_bond_type_left"]
        )
        data.ru_attachment_bond_type_right = str(
            metadata["attachment_bond_type_right"]
        )
        data.ru_attachment_bond_mismatch = bool(
            metadata["attachment_bond_mismatch"]
        )
        data.ru_connection_bond_policy = str(
            metadata["connection_bond_policy"]
        )
        data.ru_shared_boundary = bool(metadata["shared_boundary"])
        data.ru_multimer_builder_version = int(
            metadata["multimer_builder_version"]
        )
        data.ru_base_valid = bool(chemistry_valid)
    except Exception as exc:
        data.ru_atomic_number = torch.empty(0, dtype=torch.long)
        data.ru_edge_index = torch.empty((2, 0), dtype=torch.long)
        data.ru_bond_type = torch.empty(0, dtype=torch.long)
        data.ru_left_boundary = -1
        data.ru_right_boundary = -1
        data.ru_backbone = torch.empty(0, dtype=torch.long)
        data.ru_canonical_atom_index = torch.empty(0, dtype=torch.long)
        data.ru_attachment_bond_type = "unknown"
        data.ru_attachment_bond_type_left = "unknown"
        data.ru_attachment_bond_type_right = "unknown"
        data.ru_attachment_bond_mismatch = False
        data.ru_connection_bond_policy = "unavailable"
        data.ru_shared_boundary = False
        data.ru_multimer_builder_version = MIPS_MULTIMER_BUILDER_VERSION
        data.ru_base_valid = False
        data.ru_base_failure_code = f"{type(exc).__name__}:{exc}"[:240]
    data.ru_base_schema = RU_BASE_SCHEMA
    return data


def _mol_from_ru_base(ru_base):
    payload = getattr(ru_base, "ru_mol_binary", None)
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("RU base cache is missing its molecule binary")
    molecule = Chem.Mol(bytes(payload))
    if molecule is None:
        raise RuntimeError("RU base molecule binary is unreadable")
    return molecule


def _compute_topology_layer(
    smiles, ru_base, *, max_hops=2,
    topology_representation=TOPOLOGY_CANONICAL,
):
    try:
        return _compute_topology_layer_impl(
            smiles, ru_base, max_hops=max_hops,
            topology_representation=topology_representation,
        )
    except Exception as exc:
        # Any unexpected failure during topology construction produces a
        # shape-compatible placeholder so the cache build never aborts.
        placeholder = _topology_placeholder_from_ru(
            ru_base, str(smiles),
            topology_representation=topology_representation,
        )
        placeholder.topology_failure_code = f"{type(exc).__name__}:{exc}"[:240]
        placeholder.graph_available = False
        placeholder.mips_condition_valid = False
        placeholder.mips_alias_free = False
        return placeholder


def _topology_placeholder_from_ru(
    ru_base, smiles, *, topology_representation=TOPOLOGY_CANONICAL,
):
    """Build a canonical shape-safe placeholder reusing the RU base."""
    molecule = _mol_from_ru_base(ru_base)
    normalized, _ = normalize_polymer_smiles(smiles)
    canonical_molecule = Chem.MolFromSmiles(str(normalized))
    if canonical_molecule is not None:
        molecule = canonical_molecule
    try:
        data = build_canonical_periodic_topology(molecule, max_hops=2)
    except Exception:
        data = build_canonical_periodic_topology("*CC*", max_hops=2)
    data.smiles = str(smiles)
    data.graph_available = False
    data.mips_condition_valid = False
    data.mips_alias_free = False
    data.mts_canonical_periodic = True
    data.mts_topology_representation = TOPOLOGY_CANONICAL
    data.topology_representation = TOPOLOGY_CANONICAL
    data.feature_schema = CANONICAL_FEATURE_SCHEMA
    data.canonical_periodic_topology_schema = CANONICAL_TOPOLOGY_SCHEMA
    data.mips_local_lga_schema_version = CANONICAL_LGA_SCHEMA_VERSION
    data.topology_failure_code = "canonical_topology_unavailable"
    for name in (
        "ru_attachment_bond_type",
        "ru_attachment_bond_type_left",
        "ru_attachment_bond_type_right",
        "ru_attachment_bond_mismatch",
        "ru_connection_bond_policy",
        "ru_shared_boundary",
        "ru_multimer_builder_version",
    ):
        if hasattr(ru_base, name):
            data[name] = getattr(ru_base, name)
    data.feature_compute_seconds = 0.0
    return _prune_nonpbc_mips_data(data)


def _compute_topology_layer_impl(
    smiles, ru_base, *, max_hops=2,
    topology_representation=TOPOLOGY_CANONICAL,
):
    if topology_representation != TOPOLOGY_CANONICAL:
        raise ValueError("MTS-GLT-v2 only supports canonical_lifted topology")
    source_molecule = _mol_from_ru_base(ru_base)
    molecule = source_molecule
    normalized, _ = normalize_polymer_smiles(smiles)
    canonical_molecule = Chem.MolFromSmiles(str(normalized))
    if canonical_molecule is not None:
        molecule = canonical_molecule
    data = build_canonical_periodic_topology(molecule, max_hops=int(max_hops))
    data.smiles = str(smiles)
    data.feature_schema = CANONICAL_FEATURE_SCHEMA
    data.canonical_periodic_topology_schema = CANONICAL_TOPOLOGY_SCHEMA
    data.mips_local_lga_schema_version = CANONICAL_LGA_SCHEMA_VERSION
    data.mts_canonical_periodic = True
    data.mts_topology_representation = TOPOLOGY_CANONICAL
    data.topology_representation = TOPOLOGY_CANONICAL
    # Persist the explicit atom identity table used by Trimer migration.  The
    # canonical topology itself is in normalized-RU order; the source row may
    # use a reversed or otherwise non-canonical P-SMILES order.
    source_to_normalized = find_base_atom_mapping(source_molecule, molecule)
    data.source_to_normalized_canonical_atom_id = source_to_normalized
    data.source_to_canonical_atom_id = source_to_normalized
    data.normalized_canonical_smiles = str(normalized)
    canonical_count = (
        int(torch.as_tensor(data.canonical_ru_atom_index).max().item()) + 1
        if torch.as_tensor(data.canonical_ru_atom_index).numel()
        else 0
    )
    data.canonical_to_trimer_base_atom_id = torch.arange(
        canonical_count, dtype=torch.long
    )
    data.canonical_to_trimer_base_atom_index = (
        data.canonical_to_trimer_base_atom_id
    )
    chemistry_valid = bool(getattr(ru_base, "ru_chemistry_valid", False))
    builder_available = bool(getattr(data, "graph_available", True))
    builder_condition = bool(getattr(data, "mips_condition_valid", True))
    data.graph_available = bool(
        chemistry_valid and builder_available and builder_condition
    )
    data.mips_condition_valid = bool(data.graph_available)
    data.mips_alias_free = bool(
        data.graph_available and getattr(data, "mips_alias_free", True)
    )
    if data.graph_available:
        data.topology_failure_code = ""
    elif not chemistry_valid:
        data.topology_failure_code = "rdkit_parse_failed_placeholder"
    else:
        data.topology_failure_code = str(getattr(
            data, "topology_failure_code", "mips_condition_failed"
        ))
    for name in (
        "ru_attachment_bond_type",
        "ru_attachment_bond_type_left",
        "ru_attachment_bond_type_right",
        "ru_attachment_bond_mismatch",
        "ru_connection_bond_policy",
        "ru_shared_boundary",
        "ru_multimer_builder_version",
    ):
        if hasattr(ru_base, name):
            data[name] = getattr(ru_base, name)
    data.feature_compute_seconds = 0.0
    return _prune_nonpbc_mips_data(data)


def _compute_trimer_layer(
    smiles,
    ru_base,
    topology,
    *,
    num_candidates=4,
    max_heavy_atoms=384,
    force_unavailable=None,
):
    carrier = Data()
    carrier.smiles = str(smiles)
    carrier.x = topology.x
    carrier.z = topology.z
    carrier.canonical_ru_atom_index = topology.canonical_ru_atom_index
    if hasattr(topology, "canonical_to_trimer_base_atom_id"):
        carrier.canonical_to_trimer_base_atom_id = (
            topology.canonical_to_trimer_base_atom_id
        )
        carrier.canonical_to_trimer_base_atom_index = (
            topology.canonical_to_trimer_base_atom_id
        )
    carrier.graph_available = bool(
        getattr(topology, "graph_available", True)
    )
    if not carrier.graph_available:
        attach_unavailable_trimer_mcl(carrier, "graph_unavailable")
    elif not bool(getattr(ru_base, "ru_base_valid", False)):
        attach_unavailable_trimer_mcl(carrier, "ru_base_unavailable")
    elif force_unavailable:
        attach_unavailable_trimer_mcl(carrier, str(force_unavailable))
    else:
        normalized, _ = normalize_polymer_smiles(smiles)
        trimer_source = Chem.MolFromSmiles(str(normalized))
        if trimer_source is None:
            trimer_source = _mol_from_ru_base(ru_base)
        attach_finite_trimer_mcl(
            carrier,
            trimer_source,
            num_candidates=int(num_candidates),
            max_heavy_atoms=int(max_heavy_atoms),
        )
    return _select_data_fields(carrier, _LMDB_TRIMER_FIELDS)


def _compute_md200_layer(smiles, ru_base):
    data = Data()
    if bool(getattr(ru_base, "ru_chemistry_valid", False)):
        _attach_mips_descriptors(
            data, _mol_from_ru_base(ru_base), protocol="source_star_sub"
        )
    else:
        _attach_unavailable_mips_descriptors(data, "rdkit_parse_failed")
    return _select_data_fields(
        data,
        {
            "mips_md", "mips_md_valid", "descriptor_failure_code",
            "mips_descriptor_source", "mips_descriptor_optimizer",
            "mips_descriptor_schema_version",
        },
    )


def _compute_lmdb_layers_worker(payload):
    """Compute exactly the requested missing layers for one content key."""

    required = set(payload["lmdb_required_layers"])
    smiles = payload["smiles"]
    ru_base = (
        _pickle_payload_to_data(payload["ru_base_payload"])
        if payload.get("ru_base_payload") is not None else None
    )
    topology = (
        _pickle_payload_to_data(payload["topology_payload"])
        if payload.get("topology_payload") is not None else None
    )
    output = {}
    if ru_base is None:
        ru_base = _compute_ru_base_layer(smiles)
        if "ru_base" in required:
            output["ru_base"] = ru_base
    if "topology" in required:
        topology = _compute_topology_layer(
            smiles, ru_base, max_hops=payload.get("mips_max_hops", 2),
            topology_representation=payload.get(
                "topology_representation", TOPOLOGY_CANONICAL
            ),
        )
        output["topology"] = topology
    if "trimer" in required:
        if topology is None:
            raise RuntimeError("Trimer computation requires cached O8 topology")
        output["trimer"] = _compute_trimer_layer(
            smiles,
            ru_base,
            topology,
            num_candidates=payload.get("trimer_num_candidates", 4),
            max_heavy_atoms=payload.get("trimer_max_heavy_atoms", 384),
            force_unavailable=payload.get("force_trimer_unavailable"),
        )
    if "md200" in required:
        output["md200"] = _compute_md200_layer(smiles, ru_base)
    return {
        "smiles": smiles,
        "sample_key": payload["sample_key"],
        "layer_payloads": {
            name: _data_to_pickle_payload(data) for name, data in output.items()
        },
        "ok": True,
        "error": "",
    }


def _compute_smiles_features_worker(payload):
    if payload.get("lmdb_required_layers") is None:
        raise ValueError("MTS-GLT-v2 cache workers require LMDB layer requests")
    try:
        with rdBase.BlockLogs():
            return _compute_lmdb_layers_worker(payload)
    except Exception as exc:
        # Preserve a cache record for the failing sample so the immutable
        # cohort remains complete and the next run can reuse all successes.
        error_reason = f"{type(exc).__name__}:{exc}"[:500]
        smiles = payload.get("smiles", "")
        required = set(payload["lmdb_required_layers"])
        try:
            ru_base = _compute_ru_base_layer(smiles)
            topology = (
                _compute_topology_layer(
                    smiles, ru_base,
                    max_hops=payload.get("mips_max_hops", 2),
                    topology_representation=payload.get(
                        "topology_representation", TOPOLOGY_CANONICAL
                    ),
                )
                if "topology" in required else None
            )
            trimer = (
                _compute_trimer_layer(
                    smiles, ru_base, topology,
                    force_unavailable=f"cache_worker_error:{error_reason}"[:240],
                )
                if "trimer" in required else None
            )
            md200 = (
                _compute_md200_layer(smiles, ru_base)
                if "md200" in required else None
            )
            layers = {
                "ru_base": ru_base,
                "topology": topology,
                "trimer": trimer,
                "md200": md200,
            }
            return {
                "smiles": smiles,
                "sample_key": payload.get("sample_key"),
                "layer_payloads": {
                    name: _data_to_pickle_payload(layers[name])
                    for name in required if layers.get(name) is not None
                },
                "ok": True,
                "error": error_reason,
            }
        except Exception:
            return {
                "smiles": smiles,
                "sample_key": payload.get("sample_key"),
                "layer_payloads": {},
                "ok": False,
                "error": f"placeholder_construction_failed:{error_reason}"[:500],
            }


def _feature_cache_process_loop(connection):
    """Persistent cache worker controlled by a parent-side hard timeout."""
    # Geometry cache construction is CPU-only. In particular, PolyGen uses
    # torch.optim on CPU tensors; allowing a spawned worker to see CUDA makes
    # the optimizer run CUDA graph-capture health checks and can initialize a
    # CUDA context unnecessarily. A forked worker is worse because it may
    # inherit a partially initialized context from downstream training.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The process may already have initialized its inter-op pool while
        # importing dependencies. It is still isolated from the parent CUDA
        # context because cache workers use the spawn start method below.
        pass
    try:
        while True:
            task = connection.recv()
            if task is None:
                break
            job_id, payload = task
            payload = dict(payload)
            # SIGALRM cannot interrupt every RDKit C++ call. The parent owns
            # the real deadline and terminates this process if it is exceeded.
            payload["feature_cache_item_timeout"] = 0
            result = _compute_smiles_features_worker(payload)
            connection.send_bytes(pickle.dumps(
                (job_id, result), protocol=pickle.HIGHEST_PROTOCOL
            ))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        connection.close()


def _mark_hard_timeout_fallback(data, timeout_seconds, requested_geom_input):
    """Ensure a timeout fallback cannot be interpreted as periodic geometry."""
    data.geom_build_ok = False
    data.geom_coordinate_ok = False
    data.geom_requested_input = str(requested_geom_input)
    data.geom_input = "repeat_unit_fallback"
    data.geom_context = "feature_hard_timeout_topology_fallback"
    data.geom_context_id = 0
    data.geom_failed_reason = f"feature_cache_hard_timeout:{timeout_seconds}s"
    data.geom_primary_failed_reason = "feature_cache_hard_timeout"
    data.geom_screw_source = "topology_fallback"
    data.geom_screw_source_id = 0
    data.geometry_source_id = 0
    data.screw_valid = False
    data.smer_valid = False
    data.polygen_periodic_valid = False
    data.periodic_geometry_valid = False
    data.periodic_valid = False
    data.periodic_closure_error = float("inf")
    data.periodic_ru_count = 1
    data.periodic_cell_length = 0.0
    data.pbc = torch.tensor([False, False, False], dtype=torch.bool)
    data.cell = torch.zeros((3, 3), dtype=torch.float)
    data.feature_timeout_fallback = True
    return data


_SCAGE_GRAPH_CACHE_KEYS = {
    "x", "edge_index", "edge_attr", "atomic_num", "chiral_tag", "degree",
    "explicit_valence", "formal_charge", "hybridization", "is_aromatic",
    "total_numHs", "atom_is_in_ring", "mass", "van_der_waals_radius",
    "partial_charge", "scage_backbone_role", "scage_spd",
    "scage_path_bond_fields", "scage_topology_schema_version",
    "attachment_pair", "star_link_edge", "ordered_backbone_path",
    "star_link_metadata_valid", "graph_build_ok", "topology_failure_code",
    "graph_smiles", "graph_input", "structure_smiles", "structure_input",
    "requested_graph_input", "attachment_count", "has_backbone_features",
}


class UniDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        smiles_model_name,
        # ``geometry_encoder`` and the old graph route spelling are retained
        # only as a narrow Python-call compatibility shim.  They are ignored
        # below; production construction is MIPS-Trimer-SCAGE only.
        geometry_encoder='none',
        graph_encoder_type='mips_trimer_scage',
        graph_input='repeat_unit',
        geom_input='repeat_unit',
        use_feature_cache=True,
        feature_source_dataset=None,
        rebuild_feature_cache=False,
        max_smiles_length=None,
        max_smiles_length_cap=256,
        fp_mode='ecfp',
        feature_cache_workers=0,
        feature_cache_chunksize=4,
        feature_cache_partial_every=200,
        feature_cache_item_timeout=45,
        cache_layers=None,
        cache_validate="sample",
        cache_commit_size=128,
        embed_tries_multiplier=8,
        conformer_3d_count=8,
        conformer_keep_count=4,
        conformer_profile='full',
        scage_distance_mode='bias',
        scage_distance_rbf=32,
        scage_distance_cutoff=12.0,
        mips_core='topology_plus',
        mips_max_hops=None,
        mips_use_descriptors=False,
        mips_descriptor_protocol='source_star_sub',
        spatial_mode='trimer_scage',
        mips_variant=None,
        finite_variant='none',
        conformer_mode='none',
        field_layout='none',
        field_channels='none',
        graph_geometry_mode='trimer_scage_mcl',
        topology_representation=TOPOLOGY_CANONICAL,
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        modalities=None,
        experiment_id='manual',
        feature_config_hash='manual',
        transform=None,
        pre_transform=None,
        periodic_line_glt_sidecar=None,
    ):
        self.dataset = dataset
        self.periodic_line_glt_sidecar_root = (
            str(periodic_line_glt_sidecar) if periodic_line_glt_sidecar else None
        )
        self._periodic_line_glt_sidecar = None
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_list = []
        # Targets are kept separately from cached graph objects.  In the LMDB
        # path ``__getitem__`` creates a fresh Data copy, so mutating that copy
        # (the old target-scaling implementation) cannot persist across a
        # fold.  These arrays are the authoritative per-row target state.
        self._raw_targets = []
        self._target_override = None
        self._cohort_row_mode = False
        self.graph_input = str(graph_input).lower()
        if self.graph_input not in {'repeat_unit', 'star_linking'}:
            raise ValueError("graph_input must be 'repeat_unit' or 'star_linking'")
        self.geom_input = str(geom_input).lower()
        if self.geom_input != 'repeat_unit':
            raise ValueError(
                "MIPS-Trimer-SCAGE is non-periodic and requires geom_input=repeat_unit"
            )
        self.fp_mode = str(fp_mode).lower()
        self.modalities = tuple(modalities or ("graph",))
        if self.modalities != ("graph",):
            raise ValueError("MTS-GLT-v2-Base-5k supports graph modality only")
        if self.fp_mode != "disabled":
            raise ValueError("MTS-GLT-v2-Base-5k disables fingerprints")
        self.mts_use_smiles = False
        self.mts_use_fp = False
        self.fp_dim = 0
        self.feature_cache_workers = max(0, int(feature_cache_workers))
        self.feature_cache_chunksize = max(1, int(feature_cache_chunksize))
        self.feature_cache_partial_every = max(0, int(feature_cache_partial_every))
        self.feature_cache_item_timeout = max(0, int(feature_cache_item_timeout))
        if cache_layers is None:
            cache_layers = ("ru_base", "topology", "trimer", "md200")
        elif isinstance(cache_layers, str):
            cache_layers = tuple(
                item.strip() for item in cache_layers.split(",") if item.strip()
            )
        else:
            cache_layers = tuple(str(item).strip() for item in cache_layers)
        unknown_cache_layers = set(cache_layers) - {
            "ru_base", "topology", "trimer", "md200"
        }
        if unknown_cache_layers:
            raise ValueError(
                f"unknown MIPS cache layers: {sorted(unknown_cache_layers)}"
            )
        self.requested_cache_layers = tuple(dict.fromkeys(cache_layers))
        requested_cache_layers = set(cache_layers)
        if "topology" in requested_cache_layers:
            requested_cache_layers.add("ru_base")
        if "trimer" in requested_cache_layers:
            requested_cache_layers.update({"ru_base", "topology"})
        if "md200" in requested_cache_layers:
            requested_cache_layers.add("ru_base")
        self.cache_layers = tuple(
            name for name in ("ru_base", "topology", "trimer", "md200")
            if name in requested_cache_layers
        )
        self.cache_validate = str(cache_validate).lower()
        if self.cache_validate not in {"sample", "full"}:
            raise ValueError("cache_validate must be sample or full")
        self.cache_commit_size = max(1, int(cache_commit_size))
        self.embed_tries_multiplier = max(1, int(embed_tries_multiplier))
        self.conformer_profile = str(conformer_profile).lower()
        if self.conformer_profile not in {'fast', 'full', 'quality'}:
            raise ValueError("conformer_profile must be 'fast', 'full', or 'quality'")
        self.conformer_3d_count = max(1, int(conformer_3d_count))
        self.conformer_keep_count = min(
            self.conformer_3d_count,
            max(1, int(conformer_keep_count)),
        )
        self.scage_distance_mode = str(scage_distance_mode).lower()
        if self.scage_distance_mode not in {'bias', 'mask', 'multiscale_bias', 'mips_dual'}:
            raise ValueError(
                "scage_distance_mode must be bias, mask, multiscale_bias, or mips_dual"
            )
        self.scage_distance_rbf = int(scage_distance_rbf)
        self.scage_distance_cutoff = float(scage_distance_cutoff)
        self.graph_encoder_type = str(graph_encoder_type).lower()
        if self.graph_encoder_type in {"scage", "mts"}:
            # Backward-compatible direct Python API only; command-line
            # parsers expose the canonical MTS spelling.
            self.graph_encoder_type = "mips_trimer_scage"
        if self.graph_encoder_type not in {'mips_trimer_scage'}:
            raise ValueError(
                "Only graph_encoder_type='mips_trimer_scage' is supported; "
                "the retired graph routes are unavailable"
            )
        if self.graph_encoder_type == "mips_trimer_scage" and "topology" not in self.cache_layers:
            # Every graph-only dataset item requires O8 even when cache-only is
            # invoked to add MD200. Dependencies are readable/reusable but only
            # explicitly requested layers are rebuilt.
            selected = set(self.cache_layers)
            selected.update({"ru_base", "topology"})
            self.cache_layers = tuple(
                name for name in ("ru_base", "topology", "trimer", "md200")
                if name in selected
            )
        self.topology_representation = str(topology_representation)
        if self.topology_representation not in TOPOLOGY_REPRESENTATIONS:
            raise ValueError(
                "MTS-GLT-v2 only supports topology_representation=canonical_lifted"
            )
        self.mips_core = str(mips_core)
        if self.mips_core != "paper_corrected":
            raise ValueError("selected O8 route requires mips_core=paper_corrected")
        self.mips_max_hops = int(2 if mips_max_hops is None else mips_max_hops)
        if self.mips_max_hops != 2:
            raise ValueError("selected O8 route requires mips_max_hops=2")
        self.mips_use_descriptors = bool(mips_use_descriptors)
        self.mips_descriptor_protocol = str(mips_descriptor_protocol)
        if self.mips_descriptor_protocol != "source_star_sub":
            raise ValueError(
                "selected O8 route requires mips_descriptor_protocol="
                "source_star_sub"
            )
        self.spatial_mode = str(spatial_mode)
        if self.graph_encoder_type == "mips_trimer_scage" and self.spatial_mode != "trimer_scage":
            raise ValueError(
                "MTS requires spatial_mode=trimer_scage"
            )
        self.finite_variant = "none"
        self.conformer_mode = "none"
        self.field_layout = "none"
        self.field_channels = "none"
        self.graph_geometry_mode = str(graph_geometry_mode)
        if self.graph_geometry_mode != "trimer_scage_mcl":
            raise ValueError(
                "MTS-GLT-v2 requires graph_geometry_mode=trimer_scage_mcl"
            )
        self.trimer_num_candidates = int(trimer_num_candidates)
        self.trimer_max_heavy_atoms = int(trimer_max_heavy_atoms)
        if self.graph_geometry_mode == "trimer_scage_mcl":
            if self.graph_encoder_type != "mips_trimer_scage":
                raise ValueError("Trimer geometry is only valid for the MTS-GLT-v2 route")
            if self.trimer_num_candidates != 4:
                raise ValueError("MTS-GLT-v2 Trimer requires 4 candidates")
            if self.trimer_max_heavy_atoms != 384:
                raise ValueError("MTS-GLT-v2 Trimer requires max 384 heavy atoms")
        self.mips_variant = "O8"
        if self.graph_encoder_type == "mips_trimer_scage":
            if self.graph_geometry_mode != "trimer_scage_mcl":
                raise ValueError(
                "MTS requires graph_geometry_mode="
                    "trimer_scage_mcl"
                )
            self.mips_use_descriptors = True
        self.experiment_id = str(experiment_id)
        self.feature_config_hash = str(feature_config_hash)
        self.smiles_model_name = smiles_model_name
        self.is_mts_route = self.graph_encoder_type == "mips_trimer_scage"
        self.is_graph_only_mips_route = self.is_mts_route
        self.smiles_tokenizer = None

        # No parallel geometry encoder is instantiated in the production
        # route.  Keep a stable metadata value even when an old direct Python
        # caller still supplies the retired compatibility argument.
        self.geometry_encoder = "none"
        if self.geom_input != "repeat_unit":
            raise ValueError(
                "MIPS-Trimer-SCAGE is non-periodic and requires geom_input=repeat_unit"
            )

        cache_namespace = "mips_trimer_scage"
        processed_dir = os.path.join(self.root, 'processed', cache_namespace)
        os.makedirs(processed_dir, exist_ok=True)

        graph_tag = f'{self.graph_encoder_type}-backbone'
        if self.graph_input == 'star_linking':
            graph_tag = f'{self.graph_encoder_type}-starlink-backbone'
        if self.graph_encoder_type == 'mips_trimer_scage':
            graph_tag = (
                f"{graph_tag}-{MIPS_EXPERIMENT_FEATURE_SCHEMA}-"
                f"{self.mips_core}-hop{self.mips_max_hops}-"
                f"{self.feature_config_hash[:12]}"
            )

        self.use_feature_cache = bool(use_feature_cache)
        if not self.use_feature_cache:
            raise ValueError(
                "MTS-GLT-v2-Base-5k requires the immutable LMDB feature cache"
            )

        if self.geom_input == "periodic_pbc":
            geom_version = "-quality-v4"
        elif self.geom_input == "polygen_periodic":
            geom_version = "-opt-v4-directseed-steric"
        elif self.geom_input == "screw_periodic":
            geom_version = "-screw-trimer-fit-v5-top4"
        elif self.geom_input == "smer_context":
            geom_version = "-trimer-v1"
        else:
            geom_version = ""
        geom_tag = (
            f"geom-{self.geom_input.replace('_', '-')}{geom_version}-energytop{self.conformer_keep_count}"
            f"_cand{self.conformer_3d_count}_{self.conformer_profile}"
            f"_hardtimeout{self.feature_cache_item_timeout}"
        )
        fp_tag = f"fp-{self.fp_mode}"

        self._init_with_feature_cache(
            processed_dir=processed_dir,
            graph_tag=graph_tag,
            geom_tag=geom_tag,
            fp_tag=fp_tag,
            max_smiles_length=max_smiles_length,
            max_smiles_length_cap=max_smiles_length_cap,
            feature_source_dataset=feature_source_dataset,
            rebuild_feature_cache=rebuild_feature_cache,
        )
        if self.periodic_line_glt_sidecar_root is not None:
            sidecar_metadata = Path(
                self.periodic_line_glt_sidecar_root, "metadata.json"
            )
            sidecar_schema = json.loads(
                sidecar_metadata.read_text(encoding="utf-8")
            ).get("schema")
            is_distill_sidecar = str(sidecar_schema).startswith(
                (PERIODIC_LINE_DISTILL_SCHEMA_PREFIX, "mts-periodic-line-distill-v2-")
            )
            sidecar_type = (
                PeriodicLineDistillSidecar
                if is_distill_sidecar
                else PeriodicLineImageSidecar
                if sidecar_schema == PERIODIC_LINE_IMAGE_SCHEMA
                else PeriodicLineGLTSidecar
            )
            sidecar_kwargs = (
                {"build_key_index": not self._cohort_row_mode}
                if sidecar_type in {PeriodicLineDistillSidecar, PeriodicLineImageSidecar}
                else {}
            )
            self._periodic_line_glt_sidecar = sidecar_type(
                self.periodic_line_glt_sidecar_root, **sidecar_kwargs
            )
            if self._cohort_row_mode and len(self._periodic_line_glt_sidecar) != len(self.data_list):
                raise RuntimeError("periodic line GLT sidecar record count does not match Dataset")

    # ------------------------------------------------------------------
    # Immutable LMDB feature-cache path
    # ------------------------------------------------------------------
    def _init_with_feature_cache(
        self,
        processed_dir,
        graph_tag,
        geom_tag,
        fp_tag,
        max_smiles_length,
        max_smiles_length_cap,
        feature_source_dataset,
        rebuild_feature_cache,
    ):
        self.feature_source_dataset = feature_source_dataset or self.dataset
        self.max_smiles_length = 0
        self.feature_source_dataset = feature_source_dataset or self.dataset
        return self._init_with_lmdb_feature_cache(
            processed_dir=processed_dir,
            graph_tag=graph_tag,
            geom_tag=geom_tag,
            fp_tag=fp_tag,
            rebuild_feature_cache=bool(rebuild_feature_cache),
        )

    def _lmdb_cache_specs(self, meta):
        """Resolve dataset-independent content-addressed layer roots."""

        topology_representation = getattr(
            self, "topology_representation", TOPOLOGY_CANONICAL
        )
        root = os.path.join(
            self.root, "processed", "mips_trimer_scage"
        )
        common = {
            "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
            "rdkit_version": rdBase.rdkitVersion,
            "random_seed": 42,
        }
        definitions = {}

        def add_definition(name, schema, build_config):
            layer_meta = {
                **common,
                "schema": schema,
                "build_config": build_config,
            }
            digest = hashlib.sha256(
                json.dumps(layer_meta, sort_keys=True).encode("utf-8")
            ).hexdigest()
            layer_meta["feature_config_hash"] = digest
            definitions[name] = {
                "root": os.path.join(root, name, digest),
                "meta": layer_meta,
            }

        # Resolve in dependency order.  Downstream content hashes deliberately
        # include their upstream content hashes so a corrected RU protocol can
        # never silently reuse stale topology or Trimer records.
        add_definition(
            "ru_base",
            RU_BASE_SCHEMA,
            {
                "normalization": "rdkit_canonical_psmiles_v1",
                "molecule_binary": "rdkit_mol_binary",
                "attachment_site_policy":
                    "two_sites_shared_boundary_allowed",
                "mismatched_bond_policy": "single",
                "boundary_distance_algorithm":
                    "expanded_graph_shortest_path",
                "multimer_builder_version": MIPS_MULTIMER_BUILDER_VERSION,
            },
        )
        ru_base_hash = definitions["ru_base"]["meta"]["feature_config_hash"]
        target_contract = make_target_contract(
            ru_base_hash=ru_base_hash,
            rdkit_version=rdBase.rdkitVersion,
            random_seed=42,
        )
        for name in ("topology", "trimer"):
            layer_meta = copy.deepcopy(target_contract[name])
            definitions[name] = {
                "root": os.path.join(
                    root, name, layer_meta["feature_config_hash"]
                ),
                "meta": layer_meta,
            }
        # MD200 is intentionally independent of RU/topology/Trimer protocols.
        add_definition(
            "md200",
            MD200_LMDB_SCHEMA,
            {
                "descriptor_schema": 5,
                "protocol": "source_star_sub",
                "components": ["RDKit2DNormalized200"],
                "embedding": "none",
                "optimizer": "none",
            },
        )
        return {name: definitions[name] for name in self.cache_layers}

    def _lmdb_worker_payload(
        self, smiles, key, required_layers, parts
    ):
        ru_base = None
        topology = None
        if "ru_base" not in required_layers:
            source = parts["ru_base"]
            ru_base = (
                source.get(key)
                if isinstance(source, LmdbLayerWriter)
                else source[key]
            )
        if "trimer" in required_layers and "topology" not in required_layers:
            source = parts["topology"]
            topology = (
                source.get(key)
                if isinstance(source, LmdbLayerWriter)
                else source[key]
            )
        return {
            "smiles": str(smiles),
            "sample_key": bytes(key),
            "lmdb_required_layers": tuple(required_layers),
            "ru_base_payload": (
                _data_to_pickle_payload(ru_base) if ru_base is not None else None
            ),
            "topology_payload": (
                _data_to_pickle_payload(topology) if topology is not None else None
            ),
            "mips_max_hops": self.mips_max_hops,
            "topology_representation": self.topology_representation,
            "trimer_num_candidates": self.trimer_num_candidates,
            "trimer_max_heavy_atoms": self.trimer_max_heavy_atoms,
            "feature_cache_item_timeout": self.feature_cache_item_timeout,
        }

    def _commit_lmdb_worker_result(self, result, writers):
        key = bytes(result["sample_key"])
        decoded = {}
        for name, payload in result["layer_payloads"].items():
            data = _pickle_payload_to_data(payload)
            writers[name].add(key, data)
            decoded[name] = data
        return decoded

    def _run_lmdb_jobs(self, jobs, writers, parts, total_jobs=None):
        """Run layer-selective jobs with parent-owned hard timeouts."""

        total_jobs = int(total_jobs if total_jobs is not None else len(jobs))
        if total_jobs <= 0:
            return {"failure_count": 0, "failure_counts": {}}
        failure_counts = Counter()
        completed = 0
        graph_valid = 0
        graph_unavailable = 0
        trimer_valid = 0
        trimer_unavailable = 0
        worker_timeouts = 0
        worker_restarts = 0
        started_at = time.monotonic()
        last_report = started_at

        def cache_bytes():
            total = 0
            for writer in writers.values():
                data_root = os.path.join(writer.root, "data.lmdb")
                if os.path.isdir(data_root):
                    for root, _, filenames in os.walk(data_root):
                        for filename in filenames:
                            try:
                                total += os.path.getsize(
                                    os.path.join(root, filename)
                                )
                            except OSError:
                                pass
            return total

        initial_cache_bytes = cache_bytes()

        def committed_count():
            return sum(
                int(writer.inserted) + len(writer.buffer)
                for writer in writers.values()
            )

        def record_result(result, key, smiles):
            nonlocal completed, graph_valid, graph_unavailable
            nonlocal trimer_valid, trimer_unavailable
            decoded = self._commit_lmdb_worker_result(result, writers)
            topology = decoded.get("topology")
            if topology is not None:
                available = bool(getattr(topology, "graph_available", False))
                graph_valid += int(available)
                graph_unavailable += int(not available)
                if not available:
                    reason = " ".join(str(getattr(
                        topology, "topology_failure_code", "graph_unavailable"
                    )).split())
                    failure_counts[f"graph:{reason[:80]}"] += 1
            trimer = decoded.get("trimer")
            if trimer is not None:
                available = bool(getattr(
                    trimer, "trimer_geometry_valid", False
                ))
                trimer_valid += int(available)
                trimer_unavailable += int(not available)
                if not available:
                    reason = " ".join(str(getattr(
                        trimer, "trimer_failure_code", "unknown"
                    )).split())
                    failure_counts[f"trimer:{reason[:80]}"] += 1
            completed += 1

        progress_bar = None
        if sys.stderr.isatty():
            progress_bar = tqdm(
                total=total_jobs, desc="Building LMDB cache",
                unit="samples", mininterval=2.0,
                file=sys.stderr,
            )

        def report(force=False):
            nonlocal last_report
            now = time.monotonic()
            if not force and now - last_report < 5.0:
                return
            elapsed = max(now - started_at, 1e-9)
            rate = completed / elapsed
            remaining = max(0, total_jobs - completed)
            eta_minutes = (
                remaining / rate / 60.0 if completed > 0 else None
            )
            top_failures = ",".join(
                f"{key}={value}"
                for key, value in failure_counts.most_common(5)
            ) or "none"
            status_line = (
                "[lmdb_cache] "
                f"completed={completed}/{total_jobs} "
                f"committed={committed_count()} "
                f"samples_per_s={rate:.2f} "
                f"eta_min={'unknown' if eta_minutes is None else f'{eta_minutes:.1f}'} "
                f"graph_valid={graph_valid} "
                f"graph_unavailable={graph_unavailable} "
                f"trimer_valid={trimer_valid} "
                f"trimer_unavailable={trimer_unavailable} "
                f"worker_timeouts={worker_timeouts} "
                f"worker_restarts={worker_restarts} "
                f"failure_counts={top_failures} "
                f"rss_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2):.2f} "
                f"cache_delta_gib={(cache_bytes() - initial_cache_bytes) / (1024 ** 3):.2f}"
            )
            if progress_bar is not None:
                progress_bar.n = completed
                progress_bar.refresh()
            else:
                print(status_line, flush=True)
            last_report = now

        workers = int(self.feature_cache_workers)
        if workers <= 1:
            for smiles, key, required in jobs:
                payload = self._lmdb_worker_payload(
                    smiles, key, required, parts
                )
                result = _compute_smiles_features_worker(payload)
                if not result["ok"] and "trimer" in required:
                    payload["force_trimer_unavailable"] = (
                        f"cache_worker_error:{result['error']}"
                    )
                    result = _compute_smiles_features_worker(payload)
                if not result["ok"]:
                    raise RuntimeError(
                        f"failed to build required cache layers for {smiles}: "
                        f"{result['error']}"
                    )
                record_result(result, key, smiles)
                report()
            report(force=True)
            return {
                "failure_count": int(trimer_unavailable),
                "failure_counts": dict(failure_counts),
            }

        context = mp.get_context("spawn")
        job_iter = iter(jobs)
        primary_timeout = max(1, int(self.feature_cache_item_timeout))
        fallback_timeout = min(30, max(5, primary_timeout // 4))
        job_counter = 0

        def start_worker():
            parent_conn, child_conn = context.Pipe(duplex=True)
            process = context.Process(
                target=_feature_cache_process_loop, args=(child_conn,)
            )
            process.start()
            child_conn.close()
            return {
                "process": process,
                "connection": parent_conn,
                "job": None,
                "payload": None,
                "phase": None,
                "started": 0.0,
                "job_id": None,
            }

        def stop_worker(state):
            try:
                state["connection"].close()
            except Exception:
                pass
            process = state["process"]
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
            process.join(timeout=2.0)

        def assign(state, job, phase):
            nonlocal job_counter
            smiles, key, required = job
            payload = self._lmdb_worker_payload(
                smiles, key, required, parts
            )
            if phase == "fallback":
                payload["force_trimer_unavailable"] = (
                    f"feature_cache_hard_timeout:{primary_timeout}s"
                )
            job_counter += 1
            state.update({
                "job": job,
                "payload": payload,
                "phase": phase,
                "started": time.monotonic(),
                "job_id": job_counter,
            })
            state["connection"].send((job_counter, payload))

        states = [
            start_worker() for _ in range(min(workers, total_jobs))
        ]
        exhausted = False

        def assign_next(state):
            nonlocal exhausted
            if exhausted:
                return False
            try:
                job = next(job_iter)
            except StopIteration:
                exhausted = True
                return False
            assign(state, job, "primary")
            return True

        for state in states:
            assign_next(state)
        try:
            while any(state["job"] is not None for state in states):
                made_progress = False
                for state_index, state in enumerate(states):
                    if state["job"] is None:
                        continue
                    result = None
                    if state["connection"].poll():
                        try:
                            returned_job, result = pickle.loads(
                                state["connection"].recv_bytes()
                            )
                            if returned_job != state["job_id"]:
                                raise RuntimeError(
                                    "feature_cache_job_id_mismatch"
                                )
                        except Exception as exc:
                            result = {
                                "ok": False,
                                "error": f"worker_result_error:{exc}",
                            }
                    else:
                        deadline = (
                            fallback_timeout
                            if state["phase"] == "fallback"
                            else primary_timeout
                        )
                        timed_out = (
                            time.monotonic() - state["started"] > deadline
                        )
                        crashed = not state["process"].is_alive()
                        if timed_out or crashed:
                            worker_restarts += 1
                            worker_timeouts += int(timed_out)
                            previous_phase = state["phase"]
                            job = state["job"]
                            previous_payload = state.get("payload")
                            stop_worker(state)
                            states[state_index] = start_worker()
                            state = states[state_index]
                            if (
                                previous_phase == "primary"
                                and "trimer" in job[2]
                            ):
                                assign(state, job, "fallback")
                                made_progress = True
                                continue
                            if "trimer" in job[2]:
                                # A Trimer timeout is an expected sample-level
                                # failure.  Materialise its explicit
                                # unavailable record in the parent rather
                                # than aborting a million-row cohort.
                                payload = previous_payload or {}
                                payload["force_trimer_unavailable"] = (
                                    "cache_worker_hard_timeout_or_crash"
                                )
                                result = _compute_smiles_features_worker(
                                    payload
                                )
                            else:
                                result = {
                                    "ok": False,
                                    "error": (
                                        "cache_worker_hard_timeout_or_crash"
                                    ),
                                }
                    if result is None:
                        continue
                    smiles, key, required = state["job"]
                    if not result["ok"]:
                        raise RuntimeError(
                            f"failed to build required LMDB layers for "
                            f"{smiles}: {result['error']}"
                        )
                    record_result(result, key, smiles)
                    made_progress = True
                    state.update({
                        "job": None, "payload": None, "phase": None,
                        "job_id": None,
                    })
                    assign_next(state)
                report()
                if not made_progress:
                    time.sleep(0.02)
        finally:
            report(force=True)
            if progress_bar is not None:
                progress_bar.close()
            # Notify every persistent worker first, then wait against one
            # shared deadline.  Sequential two-second joins made a completed
            # 48-worker preflight appear to hang for nearly a minute.
            for state in states:
                try:
                    if state["process"].is_alive():
                        state["connection"].send(None)
                except Exception:
                    pass
            shutdown_deadline = time.monotonic() + 5.0
            for state in states:
                remaining = max(0.0, shutdown_deadline - time.monotonic())
                if state["process"].is_alive() and remaining > 0:
                    state["process"].join(timeout=remaining)
            for state in states:
                stop_worker(state)
        report(force=True)
        return {
            "failure_count": int(trimer_unavailable),
            "failure_counts": dict(failure_counts),
        }

    def _build_lmdb_layers(self, cohort, specs, rebuild):
        writers = {}
        parts = {}
        keys = cohort["keys"]
        missing_masks = {}
        layer_bits = {"ru_base": 1, "topology": 2, "trimer": 4, "md200": 8}
        try:
            requested_rebuild = set(self.requested_cache_layers)
            for name, spec in specs.items():
                rebuild_layer = bool(rebuild) and name in requested_rebuild
                done_path = os.path.join(spec["root"], ".done")
                existing = None
                if os.path.isfile(done_path) and not rebuild_layer:
                    existing = LmdbLayerStore(
                        spec["root"], expected_meta=spec["meta"]
                    )
                    mask = existing.missing_mask(
                        keys, cohort_hash=cohort["manifest"].get("cohort_hash")
                    )
                    if not bool(mask.any()):
                        parts[name] = existing
                        missing_masks[name] = mask
                        continue
                    existing.close()
                writer = LmdbLayerWriter(
                    spec["root"],
                    spec["meta"],
                    commit_size=self.cache_commit_size,
                    commit_seconds=30.0,
                    rebuild=rebuild_layer,
                )
                writers[name] = writer
                parts[name] = writer
                missing_masks[name] = writer.missing_mask(keys)

            union_mask = np.zeros(len(keys), dtype=np.uint8)
            for name in self.cache_layers:
                mask = missing_masks.get(name)
                if mask is not None:
                    union_mask |= np.where(
                        mask != 0, layer_bits[name], 0
                    ).astype(np.uint8, copy=False)
            total_jobs = int(np.count_nonzero(union_mask))

            # Graph-only training never needs the million SMILES strings.  A
            # cache writer does need them only when at least one layer is
            # missing; load the text lazily in that case.
            cohort_smiles = cohort.get("smiles")
            if total_jobs and cohort_smiles is None:
                loaded_cohort = load_cohort(
                    cohort["root"], load_text=True, verify_integrity=False
                )
                cohort_smiles = loaded_cohort["smiles"]

            def jobs():
                for index, key in enumerate(keys):
                    if union_mask[index] == 0:
                        continue
                    required = tuple(
                        name for name in self.cache_layers
                        if missing_masks.get(name) is not None
                        and bool(missing_masks[name][index])
                    )
                    if required:
                        yield cohort_smiles[index], key, required

            print(
                "[lmdb_cache] "
                + ", ".join(
                    f"{name}_missing={int(np.count_nonzero(missing_masks.get(name, [])))}"
                    for name in self.cache_layers
                )
                + f", jobs={total_jobs}, workers={self.feature_cache_workers}"
            )
            job_stats = self._run_lmdb_jobs(
                jobs(), writers, parts, total_jobs=total_jobs
            )
            stores = {
                name: part
                for name, part in parts.items()
                if isinstance(part, LmdbLayerStore)
            }
            stores.update({
                name: writer.finalize(
                    cohort_hashes=[cohort["manifest"]["cohort_hash"]],
                    failure_count=(
                        int(job_stats.get("failure_count", 0))
                        if name == "trimer" else 0
                    ),
                )
                for name, writer in writers.items()
            })
            if self.cache_validate == "full":
                self._validate_and_export_lmdb_cache(
                    cohort, stores, specs
                )
        except BaseException:
            # Preserve every complete parent-received record on a controlled
            # interruption.  No .done marker is written, so resume still
            # verifies and schedules every genuinely missing key.
            for writer in writers.values():
                try:
                    writer.flush()
                except Exception:
                    pass
                writer.close()
            for part in parts.values():
                if isinstance(part, LmdbLayerStore):
                    part.close()
            raise
        # Unavailable samples are represented by explicit LMDB tombstone
        # records.  Do not accumulate a million-row Python failure list; the
        # cohort-scoped validator exports the bounded failure manifest.
        return stores, []

    @staticmethod
    def _cache_json_hash(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _validate_and_export_lmdb_cache(self, cohort, stores, specs):
        """Validate one cohort and export cohort-scoped Trimer failures.

        Content-addressed LMDB roots are shared by multiple cohorts.  Keeping
        validation artifacts below the cohort hash prevents a later 200k or
        downstream validation from overwriting the PI1M_v2 result.
        """

        marker_spec = specs.get("trimer")
        if marker_spec is None:
            # Stage 1 requests only RU base + topology, but validation is a
            # bundle-level artifact shared with Stage 2/3.  Resolve the
            # canonical Trimer root without adding Trimer to this pass's
            # requested build layers, so the finalizer report can be reused
            # instead of triggering another million-record scan.
            original_layers = self.cache_layers
            try:
                self.cache_layers = (
                    "ru_base", "topology", "trimer", "md200"
                )
                marker_spec = self._lmdb_cache_specs({})["trimer"]
            finally:
                self.cache_layers = original_layers
        marker_root = marker_spec["root"]
        cohort_hash = str(cohort["manifest"]["cohort_hash"])
        validation_dir = os.path.join(marker_root, "validation", "cohorts")
        failure_dir = os.path.join(marker_root, "failures")
        os.makedirs(validation_dir, exist_ok=True)
        os.makedirs(failure_dir, exist_ok=True)
        validation_path = os.path.join(validation_dir, f"{cohort_hash}.json")
        failure_path = os.path.join(failure_dir, f"{cohort_hash}.json")
        done_path = os.path.join(marker_root, ".done")
        if not os.path.isfile(done_path):
            # A topology-only cache may legitimately be built in an isolated
            # test/root before its Trimer layer exists.  Production training
            # performs bundle verification separately and still requires all
            # frozen layers.  Keep the independent topology builder usable.
            if "trimer" not in tuple(self.cache_layers):
                compatibility_report = {
                    "schema": CACHE_LAYOUT_SCHEMA,
                    "cohort_hash": cohort_hash,
                    "ordered_sample_key_hash": cohort["manifest"].get(
                        "ordered_sample_key_hash"
                    ),
                    "record_count": int(cohort["manifest"]["unique_count"]),
                    "layers": list(self.cache_layers),
                    "bundle_complete": False,
                }
                temporary = validation_path + ".tmp"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(compatibility_report, handle, sort_keys=True)
                os.replace(temporary, validation_path)
                legacy_path = os.path.join(marker_root, "validation.json")
                legacy_temporary = legacy_path + ".tmp"
                with open(legacy_temporary, "w", encoding="utf-8") as handle:
                    json.dump(compatibility_report, handle, sort_keys=True)
                os.replace(legacy_temporary, legacy_path)
                return
            raise RuntimeError(f"LMDB cache is missing .done: {marker_root}")
        with open(done_path, encoding="utf-8") as handle:
            store_artifact_hash = handle.read().strip()
        expected_identity = {
            "cohort_hash": cohort_hash,
            "ordered_sample_key_hash": cohort["manifest"][
                "ordered_sample_key_hash"
            ],
            "record_count": int(cohort["manifest"]["unique_count"]),
            "store_artifact_hash": store_artifact_hash,
            "metadata_hashes": {
                name: self._cache_json_hash(store.meta)
                for name, store in stores.items()
            },
        }
        if os.path.isfile(validation_path) and os.path.isfile(failure_path):
            try:
                with open(validation_path, encoding="utf-8") as handle:
                    previous = json.load(handle)
                # Finalizer reports use the v2 ``artifact_hashes`` mapping,
                # while older layer-local reports stored the Trimer .done
                # value as ``store_artifact_hash``.  Both identify the same
                # immutable content; accept either so a frozen cache is not
                # needlessly rescanned during every Dataset construction.
                identity_matches = all(
                    previous.get(key) == value
                    for key, value in expected_identity.items()
                    if key not in {"store_artifact_hash", "metadata_hashes"}
                )
                previous_metadata = previous.get("metadata_hashes")
                expected_metadata = expected_identity["metadata_hashes"]
                if isinstance(previous_metadata, dict):
                    # A finalizer report covers all four layers, while a
                    # Dataset cache pass may request only topology or
                    # topology+Trimer.  Compare the requested layer subset.
                    metadata_matches = all(
                        previous_metadata.get(name) == value
                        for name, value in expected_metadata.items()
                    )
                else:
                    metadata_matches = previous_metadata == expected_metadata
                previous_artifacts = previous.get("artifact_hashes")
                if isinstance(previous_artifacts, dict):
                    artifact_matches = all(
                        previous_artifacts.get(name)
                        == Path(store.done_path).read_text(
                            encoding="utf-8"
                        ).strip()
                        for name, store in stores.items()
                    )
                else:
                    previous_artifact = previous.get("store_artifact_hash")
                    artifact_matches = previous_artifact == store_artifact_hash
                if identity_matches and artifact_matches \
                        and metadata_matches \
                        and int(previous.get("mapping_failure", 1)) == 0 \
                        and int(previous.get("two_d_mcl", 1)) == 0 \
                        and (
                            "trimer" not in stores
                            or (
                                previous.get("geometry_rate_given_graph") is not None
                                and float(previous["geometry_rate_given_graph"]) >= 0.90
                            )
                        ):
                    return []
            except (OSError, ValueError, TypeError):
                pass

        keys = cohort["keys"]
        failures = []
        failure_counts = Counter()
        graph_valid = graph_unavailable = 0
        trimer_valid = trimer_unavailable = 0
        star_valid = 0
        mapping_failure = 0
        finite_coordinate_failure = 0
        two_d_mcl = 0
        mcl_valid = 0
        temporary = f"{failure_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as failure_handle:
            failure_handle.write("[\n")
            first_failure = True
            for index, key in enumerate(keys):
                decoded = {}
                for name, store in stores.items():
                    if key not in store:
                        raise RuntimeError(
                            f"LMDB cache validation found a missing {name} key: "
                            f"{bytes(key).hex()}"
                        )
                    data = store[key]
                    if not isinstance(data, Data):
                        raise RuntimeError(
                            "LMDB cache returned a non-Data record"
                        )
                    decoded[name] = data

                topology = decoded.get("topology")
                if topology is not None:
                    available = bool(getattr(
                        topology, "graph_available", False
                    ))
                    graph_valid += int(available)
                    graph_unavailable += int(not available)
                    if not available:
                        reason = str(getattr(
                            topology, "topology_failure_code", "unknown"
                        ))
                        failure_counts[f"graph:{reason[:120]}"] += 1

                trimer = decoded.get("trimer")
                if trimer is None:
                    continue
                quality = validate_mcl_record(topology, trimer)
                valid = quality["geometry_valid"]
                trimer_valid += int(valid)
                trimer_unavailable += int(not valid)
                star_valid += int(quality["star_3d_valid"])
                mapping_failure += int(quality["mapping_failure"])
                finite_coordinate_failure += int(
                    quality["finite_coordinate_failure"]
                )
                two_d_mcl += int(quality["two_d_mcl"])
                mcl_valid += int(quality["mcl_valid"])
                if not valid:
                    reason = str(getattr(
                        trimer, "trimer_failure_code", "unknown"
                    ))[:240]
                    failure_counts[f"trimer:{reason}"] += 1
                    item = {
                        "sample_key": bytes(key).hex(),
                        "error": reason,
                    }
                    if not first_failure:
                        failure_handle.write(",\n")
                    json.dump(item, failure_handle, sort_keys=True)
                    first_failure = False
            failure_handle.write("\n]\n")
        geometry_rate = mcl_valid / graph_valid if graph_valid else None
        has_trimer_layer = "trimer" in stores
        if (
            mapping_failure
            or two_d_mcl
            or (
                has_trimer_layer
                and (geometry_rate is None or geometry_rate < 0.90)
            )
        ):
            os.remove(temporary)
            raise RuntimeError(
                "LMDB validation failed: "
                f"mapping_failure={mapping_failure}, "
                f"two_d_mcl={two_d_mcl}, "
                f"geometry_rate_given_graph={geometry_rate}"
            )
        os.replace(temporary, failure_path)
        validation = {
            **expected_identity,
            "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
            "record_count": int(len(keys)),
            "graph_valid": graph_valid,
            "graph_unavailable": graph_unavailable,
            "trimer_valid": trimer_valid,
            "trimer_unavailable": trimer_unavailable,
            "mcl_valid": mcl_valid,
            "geometry_rate_given_graph": geometry_rate,
            "star_3d_valid": star_valid,
            "failure_counts": dict(failure_counts),
            "mapping_failure": mapping_failure,
            "finite_coordinate_failure": finite_coordinate_failure,
            "two_d_mcl": two_d_mcl,
            "validation_timestamp": time.time(),
        }
        temporary_validation = f"{validation_path}.tmp"
        with open(temporary_validation, "w", encoding="utf-8") as handle:
            json.dump(validation, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_validation, validation_path)
        # Keep a tiny compatibility index for tooling that only checks for a
        # validation artifact.  It is deliberately a pointer, never a
        # cohort's validation payload, so it cannot be mistaken for a global
        # result or overwrite another cohort's report.
        index_path = os.path.join(marker_root, "validation.json")
        temporary_index = f"{index_path}.tmp"
        with open(temporary_index, "w", encoding="utf-8") as handle:
            json.dump({
                "schema": "mips-trimer-scage-validation-index-v1",
                "latest_cohort_hash": cohort_hash,
                "cohort_validation": os.path.relpath(
                    validation_path, marker_root
                ),
            }, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_index, index_path)
        # Every failure is already represented by an unavailable LMDB record;
        # callers do not need to materialise a million-row JSON list.
        return []

    def _validate_lmdb_cache(self, cohort, stores, specs=None):
        if self.cache_validate == "full" and specs is not None:
            self._validate_and_export_lmdb_cache(cohort, stores, specs)
            return
        keys = cohort["keys"][:min(128, len(cohort["keys"]))]
        for key in keys:
            for name, store in stores.items():
                if key not in store:
                    raise RuntimeError(
                        f"LMDB cache validation found a missing {name} key: "
                        f"{bytes(key).hex()}"
                    )
                data = store[key]
                if not isinstance(data, Data):
                    raise RuntimeError("LMDB cache returned a non-Data record")

    def _init_with_lmdb_feature_cache(
        self,
        *,
        processed_dir,
        graph_tag,
        geom_tag,
        fp_tag,
        rebuild_feature_cache,
    ):
        source_csv = os.path.join(
            self.root, "raw", f"{self.feature_source_dataset}.csv"
        )
        cache_root = os.path.join(
            self.root, "processed", "mips_trimer_scage"
        )
        cohort = build_or_load_cohort(
            cache_root,
            self.feature_source_dataset,
            source_csv,
            load_text=not bool(self.is_graph_only_mips_route),
            # Training readers use the lightweight shape/key mode.  Full
            # cohort integrity is an explicit offline QC operation.
            verify_integrity=False,
        )
        meta = self._feature_cache_meta()
        specs = self._lmdb_cache_specs(meta)
        stores, failures = self._build_lmdb_layers(
            cohort, specs, rebuild_feature_cache
        )
        self._validate_lmdb_cache(cohort, stores, specs)

        md_path = md_valid_path = None
        if "md200" in stores:
            md_dir = os.path.join(
                cohort["root"],
                f"md200_{stores['md200'].meta['feature_config_hash']}",
            )
            md_path, md_valid_path, _ = materialize_md200_array(
                cohort, stores["md200"], md_dir
            )
        for store in stores.values():
            store.close()
        feature_store = LmdbFeatureStore(
            topology_root=specs["topology"]["root"],
            trimer_root=(
                specs["trimer"]["root"] if "trimer" in stores else None
            ),
            cohort=cohort,
            md200_path=md_path,
            md200_valid_path=md_valid_path,
        )
        self.feature_cache_path = os.path.join(
            cohort["root"], "manifest.json"
        )
        self.use_sharded_feature_cache = False
        self.feature_shard_root = None
        # The LMDB path must expose the same immutable topology identities as
        # the legacy feature-cache path.  Stage-1 training, cost-balanced
        # sampling and Stage-2 checkpoint binding all use the .done artifact
        # hash; leaving these fields unset silently made a valid derived cost
        # array look stale and produced an invalid cache-bundle hash.
        self.topology_cache_hash = stores["topology"].meta[
            "feature_config_hash"
        ]
        with open(
            os.path.join(specs["topology"]["root"], ".done"),
            encoding="utf-8",
        ) as handle:
            self.topology_cache_artifact_hash = handle.read().strip()
        if len(self.topology_cache_artifact_hash) != 64:
            raise RuntimeError("invalid Topology cache manifest hash")
        self.trimer_cache_hash = None
        self.trimer_cache_artifact_hash = None
        self.feature_cohort_hash = cohort["manifest"]["cohort_hash"]
        # A topology-only cache path can still bind the immutable Trimer
        # artifact identity as metadata without opening or materialising the
        # Trimer geometry payloads.
        trimer_binding_spec = specs.get("trimer")
        if trimer_binding_spec is None:
            original_layers = self.cache_layers
            try:
                self.cache_layers = (
                    "ru_base", "topology", "trimer", "md200"
                )
                trimer_binding_spec = self._lmdb_cache_specs(meta)["trimer"]
            finally:
                self.cache_layers = original_layers
        if trimer_binding_spec is not None:
            # Topology-only readers are valid before the optional Trimer layer
            # is materialised.  Bind its immutable identity when the artifact
            # exists, but do not turn a topology-only cache read into an
            # implicit Trimer build requirement.
            trimer_done_path = Path(trimer_binding_spec["root"]) / ".done"
            if trimer_done_path.is_file():
                self.trimer_cache_artifact_hash = trimer_done_path.read_text(
                    encoding="utf-8"
                ).strip()
                if len(self.trimer_cache_artifact_hash) != 64:
                    raise RuntimeError("invalid Trimer cache manifest hash")
                self.trimer_cache_hash = trimer_binding_spec["meta"][
                    "feature_config_hash"
                ]
        feature_cache = {
            "meta": {
                **meta,
                "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
                "cohort_hash": cohort["manifest"]["cohort_hash"],
                "cache_layers": list(self.cache_layers),
            },
            "features": feature_store,
            "failures": failures,
        }
        self._cohort = cohort
        self._write_feature_cache_diagnostics(
            feature_cache,
            processed_dir,
            graph_tag,
            geom_tag,
            fp_tag,
            reuse_existing=not rebuild_feature_cache,
        )
        task_csv = os.path.join(
            self.root, "raw", f"{self.dataset}.csv"
        )
        self._build_labeled_data_list(feature_cache, task_csv)
        if self.is_graph_only_mips_route and (
            self.smiles_tokenizer is not None or self.max_smiles_length != 0
        ):
            raise RuntimeError("graph-only MIPS tokenizer bypass invariant failed")

    def _migrate_model_dependent_scage_cache(self, processed_dir, graph_tag, geom_tag, fp_tag):
        raise RuntimeError(
            "legacy SCAGE/O/S/G feature-cache migration was removed; "
            "rebuild the mips-selected cache"
        )
        if self.graph_encoder_type != 'mips_trimer_scage':
            return False
        prefix = f"feature_cache_{self.feature_source_dataset}_{graph_tag}"
        patterns = [
            os.path.join(
                processed_dir,
                f"{prefix}-*-rbf*-cut*_{geom_tag}_{fp_tag}_tok{self.max_smiles_length}.pt",
            ),
            os.path.join(
                processed_dir,
                f"feature_cache_{self.feature_source_dataset}_scage-starlink-backbone-input-v1-m4p-v1_"
                f"{geom_tag}_{fp_tag}_tok{self.max_smiles_length}.pt",
            ),
        ]
        candidates = sorted(
            {path for pattern in patterns for path in glob.glob(pattern)},
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            return False
        source = candidates[0]
        cache = torch.load(source, weights_only=False)
        if not isinstance(cache, dict) or 'features' not in cache:
            return False
        features = cache['features']
        migrated = {}
        smiles_values = list(features)
        workers = max(1, int(self.feature_cache_workers))
        print(
            f"[feature_cache] migrating {len(smiles_values)} SCAGE entries with "
            f"workers={workers}; cached geometry is reused"
        )
        if workers == 1:
            iterator = (
                _migrate_scage_cached_feature((smiles, features[smiles]))
                for smiles in smiles_values
            )
            for smiles, data in tqdm(iterator, total=len(smiles_values), desc="Migrating SCAGE cache"):
                migrated[smiles] = data
        else:
            pending = iter(smiles_values)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                in_flight = {}

                def submit_one():
                    try:
                        smiles = next(pending)
                    except StopIteration:
                        return False
                    serialized = pickle.dumps(
                        features[smiles], protocol=pickle.HIGHEST_PROTOCOL
                    )
                    future = executor.submit(
                        _migrate_scage_cached_feature, (smiles, serialized)
                    )
                    in_flight[future] = smiles
                    return True

                for _ in range(min(workers * 2, len(smiles_values))):
                    submit_one()
                with tqdm(total=len(smiles_values), desc="Migrating SCAGE cache") as progress:
                    while in_flight:
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            submitted = in_flight.pop(future)
                            smiles, data = future.result()
                            if smiles != submitted:
                                raise RuntimeError("SCAGE cache migration returned a mismatched SMILES")
                            migrated[smiles] = data
                            progress.update(1)
                            submit_one()
            migrated = {smiles: migrated[smiles] for smiles in smiles_values}
        descriptor_statistics = _standardize_scage_descriptors(migrated)
        cache = {
            **cache,
            'meta': {
                **cache.get('meta', {}),
                **self._feature_cache_meta(),
                'migrated_from': source,
                'migration_workers': workers,
                'scage_descriptor_statistics': descriptor_statistics,
            },
            'features': migrated,
        }
        torch.save(cache, self.feature_cache_path)
        print(
            "[feature_cache] migrated legacy SCAGE cache without rebuilding geometry: "
            f"{source} -> {self.feature_cache_path}"
        )
        return True

    def _write_feature_cache_diagnostics(
        self,
        feature_cache,
        processed_dir,
        graph_tag,
        geom_tag,
        fp_tag,
        reuse_existing=False,
    ):
        diagnostics_path = os.path.join(
            processed_dir,
            (
                f"diagnostics_{self.feature_source_dataset}_{graph_tag}_"
                f"{geom_tag}_{fp_tag}_tok{self.max_smiles_length}.json"
            ),
        )
        if reuse_existing and os.path.exists(diagnostics_path):
            print(f"reusing feature-cache diagnostics {diagnostics_path}")
            return
        if isinstance(
            feature_cache["features"],
            (ShardedFeatureStore, LayeredFeatureStore, LmdbFeatureStore),
        ):
            summary = {
                "schema": (
                    CACHE_LAYOUT_SCHEMA
                    if isinstance(feature_cache["features"], LmdbFeatureStore)
                    else "layered-sharded-feature-cache-diagnostics-v2"
                ),
                "total": len(feature_cache["features"]),
                "note": (
                    "Full million-row startup scan is intentionally disabled; "
                    "campaign preflight stores correctness/performance statistics."
                ),
            }
            summary["cache"] = {
                "feature_cache_path": self.feature_cache_path,
                "feature_source_dataset": self.feature_source_dataset,
                "feature_config_hash": self.feature_config_hash,
                "layer_roots": getattr(
                    feature_cache["features"], "roots", {}
                ),
                "storage": (
                    "lmdb-single-record"
                    if isinstance(feature_cache["features"], LmdbFeatureStore)
                    else "torch-shard-2048"
                ),
            }
            with open(diagnostics_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, sort_keys=True, indent=2)
                handle.write("\n")
            # Keep the historical diagnostics location readable for direct
            # Python callers/tests.  It is not a cache root and is never used
            # by production loading; the active route remains the explicit
            # mips_trimer_scage namespace above.
            if self.graph_encoder_type == "mips_trimer_scage":
                legacy_dir = os.path.join(self.root, "processed", "scage")
                os.makedirs(legacy_dir, exist_ok=True)
                legacy_path = os.path.join(legacy_dir, os.path.basename(diagnostics_path))
                with open(legacy_path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, sort_keys=True, indent=2)
                    handle.write("\n")
            print(
                f"wrote lazy feature-cache diagnostics {diagnostics_path}"
            )
            return
        else:
            summary = summarize_feature_cache(
                feature_cache["features"],
                graph_input=self.graph_input,
                geom_input=self.geom_input,
            )
        summary["cache"] = {
            "feature_cache_path": self.feature_cache_path,
            "feature_source_dataset": self.feature_source_dataset,
            "dataset": self.dataset,
            "geometry_encoder": self.geometry_encoder,
            "graph_encoder_type": self.graph_encoder_type,
            "mips_core": self.mips_core,
            "mips_max_hops": self.mips_max_hops,
            "graph_tag": graph_tag,
            "geom_tag": geom_tag,
            "fp_mode": self.fp_mode,
            "fp_dim": self.fp_dim,
            "fp_components": self._fp_components(),
            "max_smiles_length": self.max_smiles_length,
            "embed_tries_multiplier": self.embed_tries_multiplier,
            "conformer_3d_count": self.conformer_3d_count,
            "conformer_keep_count": self.conformer_keep_count,
            "conformer_profile": self.conformer_profile,
            "feature_cache_item_timeout": self.feature_cache_item_timeout,
            "scage_distance_mode": self.scage_distance_mode if self.graph_encoder_type == "mips_trimer_scage" else None,
            "scage_distance_rbf": self.scage_distance_rbf if self.graph_encoder_type == "mips_trimer_scage" else None,
            "scage_distance_cutoff": self.scage_distance_cutoff if self.graph_encoder_type == "mips_trimer_scage" else None,
        }
        print_dataset_diagnostics(summary)
        write_dataset_diagnostics(summary, diagnostics_path)

    # ------------------------------------------------------------------
    # Feature cache builders
    # ------------------------------------------------------------------
    def _compute_max_token_length_for_file(self, csv_path):
        """Compute max token length by scanning a single CSV file."""
        df = pd.read_csv(csv_path)
        max_len = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing token lengths"):
            smiles = str(row.iloc[0]).strip()
            tokens = self.smiles_tokenizer.encode(smiles)
            max_len = max(max_len, len(tokens))
        print(f"Max SMILES token length in {os.path.basename(csv_path)}: {max_len}")
        return max_len

    def _feature_cache_meta(self):
        meta = {
            "feature_source_dataset": self.feature_source_dataset,
            "geometry_encoder": self.geometry_encoder,
            "graph_encoder_type": self.graph_encoder_type,
            "topology_representation": self.topology_representation,
            "graph_input": self.graph_input,
            "geometry_structure": "canonical_identity_trimer_mcl",
            "geom_input": self.geom_input,
            "graph_features": "canonical_mips137_backbone_lifted_periodic_relations",
            "scage_input": (
                "chemical_fields_backbone_no_ru_index"
                if self.graph_encoder_type == "mips_trimer_scage" else "not_applicable"
            ),
            "scage_input_schema_version": 2 if self.graph_encoder_type == "mips_trimer_scage" else 0,
            "scage_data_schema": (
                MIPS_EXPERIMENT_FEATURE_SCHEMA
                if self.graph_encoder_type == "mips_trimer_scage" else None
            ),
            "scage_topology_schema_version": 2 if self.graph_encoder_type == "mips_trimer_scage" else 0,
            "periodic_lga_schema_version": 2 if self.graph_encoder_type == "mips_trimer_scage" else 0,
            "mips_local_lga_schema_version": (
                2 if self.graph_encoder_type == "mips_trimer_scage" else 0
            ),
            "mips_core": self.mips_core if self.graph_encoder_type == "mips_trimer_scage" else None,
            "mips_max_hops": self.mips_max_hops if self.graph_encoder_type == "mips_trimer_scage" else 0,
            "mips_variant": self.mips_variant if self.graph_encoder_type == "mips_trimer_scage" else None,
            "mips_descriptor_protocol": (
                self.mips_descriptor_protocol
                if self.graph_encoder_type == "mips_trimer_scage" else None
            ),
            "spatial_mode": self.spatial_mode if self.graph_encoder_type == "mips_trimer_scage" else "legacy",
            "graph_geometry_mode": (
                self.graph_geometry_mode
                if self.graph_encoder_type == "mips_trimer_scage" else "none"
            ),
            "trimer_mcl_schema": (
                TRIMER_MCL_SCHEMA
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_lmdb_schema": (
                TRIMER_LMDB_SCHEMA
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_builder_version": (
                TRIMER_MCL_BUILDER_VERSION
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_conformer_protocol": (
                TRIMER_MCL_PROTOCOL
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_mmff_relax_steps": (
                TRIMER_MMFF_RELAX_MAX_ITERATIONS
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_require_mmff_convergence": (
                False if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_acceptance": (
                "finite_3d_coordinates_and_finite_mmff_energy"
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_selection": (
                "lowest_finite_post_relaxation_energy"
                if self.graph_geometry_mode == "trimer_scage_mcl" else None
            ),
            "trimer_num_candidates": (
                self.trimer_num_candidates
                if self.graph_geometry_mode == "trimer_scage_mcl" else 0
            ),
            "trimer_max_heavy_atoms": (
                self.trimer_max_heavy_atoms
                if self.graph_geometry_mode == "trimer_scage_mcl" else 0
            ),
            "experiment_id": self.experiment_id,
            "feature_config_hash": self.feature_config_hash,
            "graph_placeholder_policy": "preserve_multimodal_row-v1",
            "source_data_hash": _sha256_file(
                f"{self.root}/raw/{self.feature_source_dataset}.csv"
            ),
            "scage_descriptor_schema_version": 0,
            "mips_input_schema_version": 4,
            "mips_descriptor_schema_version": 5,
            "polymer_ecfp_target": "capped_3mer_morgan_r2_2048",
            "fp_mode": self.fp_mode,
            "fp_dim": self.fp_dim,
            "fp_components": self._fp_components(),
            "max_smiles_length": self.max_smiles_length,
            "tokenizer": str(type(self.smiles_tokenizer).__name__),
            "embed_tries_multiplier": self.embed_tries_multiplier,
            "conformer_3d_count": self.conformer_3d_count,
            "conformer_keep_count": self.conformer_keep_count,
            "conformer_profile": self.conformer_profile,
            "conformer_policy": TRIMER_MCL_PROTOCOL,
        }
        if self.graph_encoder_type == "mips_trimer_scage":
            for legacy_key in ("periodic_lga_schema_version", "max_smiles_length", "tokenizer"):
                meta.pop(legacy_key, None)
            meta["graph_only_tokenizer_bypassed"] = True
            meta["cache_layout_schema"] = CACHE_LAYOUT_SCHEMA
        return meta

    def _layered_cache_specs(self, meta):
        """Resolve reusable layer roots from feature-producing fields only."""
        layer_root = os.path.join(
            self.root, "processed", "mips_trimer_scage", "mips_layers",
            self.feature_source_dataset,
        )
        common = {
            "source_csv_sha256": meta["source_data_hash"],
            "rdkit_version": rdBase.rdkitVersion,
            "random_seed": 42,
            "shard_size": 2048,
        }
        configs = {
            "input": {
                "fp_mode": self.fp_mode,
                "max_smiles_length": self.max_smiles_length,
                "tokenizer": self.smiles_model_name,
                "graph_placeholder_policy": "preserve_multimodal_row-v1",
            },
            "topology": {
                "max_hops": self.mips_max_hops,
                "boundary_threshold": 2 * (self.mips_max_hops + 1) - 1,
                "max_model_atoms": 384,
                "lga_schema": 2,
                "atom_features": (
                    "mips137_topology_only_trimer_central_ru"
                ),
                "star_edge_mask": "direct_virtual_edge_only",
                "graph_placeholder_policy": "preserve_multimodal_row-v1",
            },
        }
        if self.mips_use_descriptors:
            configs["descriptor"] = {
                "descriptor_schema": 5,
                "protocol": "source_star_sub",
                "wildcard_completion": "opposite_attachment_element",
                "components": ["RDKit2DNormalized200"],
                "embedding": "none",
                "optimizer": "none",
            }
        if self.graph_geometry_mode == "trimer_scage_mcl":
            configs["trimer"] = {
                "schema": TRIMER_MCL_SCHEMA,
                "protocol": TRIMER_MCL_PROTOCOL,
                "repeat_units": 3,
                "open_chain": True,
                "outer_attachment_cap": "implicit_h_via_AddHs",
                "num_candidates": self.trimer_num_candidates,
                "builder_version": TRIMER_MCL_BUILDER_VERSION,
                "etkdg_use_random_coords": False,
                "etkdg_max_iterations": TRIMER_ETKDG_MAX_ITERATIONS,
                "etkdg_timeout_seconds": TRIMER_ETKDG_TIMEOUT_SECONDS,
                "etkdg_failure_retry": {
                    "num_candidates": TRIMER_ETKDG_RETRY_CANDIDATES,
                    "use_random_coords": True,
                    "max_iterations": TRIMER_ETKDG_RETRY_MAX_ITERATIONS,
                },
                "optimizer": "MMFF94",
                "mmff_relax_max_iterations": TRIMER_MMFF_RELAX_MAX_ITERATIONS,
                "conformer_selection": "lowest-finite-mmff-energy",
                "worker_hard_timeout_seconds": self.feature_cache_item_timeout,
                "allow_2d_for_mcl": False,
                "max_heavy_atoms": self.trimer_max_heavy_atoms,
                "large_molecule_policy": (
                    "rdkit_2d_diagnostic_mcl_star3d_disabled"
                ),
                "seed_rule": "sha256_schema_smiles_int31",
            }
        specs = {}
        for name, config in configs.items():
            schema = _LAYER_SCHEMAS[name]
            layer_meta = {
                "schema": schema,
                **common,
                "build_config": config,
            }
            digest = hashlib.sha256(
                json.dumps(layer_meta, sort_keys=True).encode("utf-8")
            ).hexdigest()
            specs[name] = {
                "root": os.path.join(layer_root, f"{schema}_{digest[:20]}"),
                "meta": {**layer_meta, "feature_config_hash": digest},
            }
        return specs

    @staticmethod
    def _partial_meta_matches(partial_meta, expected_meta):
        required_keys = [
            "feature_source_dataset",
            "geometry_encoder",
            "graph_encoder_type",
            "graph_input",
            "geom_input",
            "fp_mode",
            "max_smiles_length",
            "embed_tries_multiplier",
            "conformer_3d_count",
            "conformer_keep_count",
            "conformer_profile",
            "feature_cache_item_timeout",
            "conformer_policy",
            "pbc_geometry_version",
            "scage_data_schema",
            "periodic_lga_schema_version",
            "mips_local_lga_schema_version",
            "mips_core",
            "mips_max_hops",
            "mips_variant",
            "spatial_mode",
            "graph_geometry_mode",
            "trimer_mcl_schema",
            "trimer_conformer_protocol",
            "trimer_num_candidates",
            "trimer_max_heavy_atoms",
            "feature_config_hash",
            "source_data_hash",
            "mips_input_schema_version",
            "mips_descriptor_schema_version",
        ]
        return all(partial_meta.get(key) == expected_meta.get(key) for key in required_keys)

    def _load_partial_feature_cache(self, expected_meta):
        partial_path = f"{self.feature_cache_path}.partial"
        if self.feature_cache_partial_every <= 0 or not os.path.exists(partial_path):
            return {}, []
        try:
            partial = torch.load(partial_path, weights_only=False)
        except Exception as exc:
            print(f"[feature_cache] ignoring unreadable partial cache {partial_path}: {exc}")
            return {}, []
        if not self._partial_meta_matches(partial.get("meta", {}), expected_meta):
            print(f"[feature_cache] ignoring partial cache with mismatched meta: {partial_path}")
            return {}, []
        features = partial.get("features", {})
        for data in features.values():
            _detach_data_storages(data)
        failures = partial.get("failures", [])
        print(
            f"[feature_cache] resumed partial cache {partial_path} "
            f"with {len(features)} completed entries and {len(failures)} failures"
        )
        return features, failures

    def _save_partial_feature_cache(self, features, failures, meta):
        if self.feature_cache_partial_every <= 0:
            return
        partial_path = f"{self.feature_cache_path}.partial"
        temporary_path = f"{partial_path}.tmp"
        torch.save(
            {
                "meta": meta,
                "features": features,
                "failures": failures,
                "partial": True,
                "updated_at": time.time(),
            },
            temporary_path,
        )
        os.replace(temporary_path, partial_path)

    def _feature_cache_payload(self, smiles):
        payload = {
            "smiles": smiles,
            "smiles_model_name": self.smiles_model_name,
            "max_smiles_length": self.max_smiles_length,
            "graph_input": self.graph_input,
            "geom_input": self.geom_input,
            "geometry_encoder": self.geometry_encoder,
            "graph_encoder_type": self.graph_encoder_type,
            "fp_mode": self.fp_mode,
            "embed_tries_multiplier": self.embed_tries_multiplier,
            "conformer_3d_count": self.conformer_3d_count,
            "conformer_keep_count": self.conformer_keep_count,
            "conformer_profile": self.conformer_profile,
            "feature_cache_item_timeout": self.feature_cache_item_timeout,
            "mips_core": self.mips_core,
            "mips_max_hops": self.mips_max_hops,
            "mips_use_descriptors": (
                self.mips_use_descriptors
                and getattr(self, "_cache_descriptor_needed", True)
            ),
            "mips_descriptor_protocol": self.mips_descriptor_protocol,
            "spatial_mode": self.spatial_mode,
            "graph_geometry_mode": self.graph_geometry_mode,
            "topology_representation": self.topology_representation,
            "trimer_num_candidates": self.trimer_num_candidates,
            "trimer_max_heavy_atoms": self.trimer_max_heavy_atoms,
        }
        return payload

    def _build_feature_cache(self):
        """Build SMILES-level feature cache from feature_source_dataset CSV."""
        source_csv = f"{self.root}/raw/{self.feature_source_dataset}.csv"
        df = pd.read_csv(source_csv)

        # Deduplicate while preserving order
        unique_smiles = list(dict.fromkeys(
            str(row.iloc[0]).strip() for _, row in df.iterrows()
        ))
        print(
            f"Feature cache: {len(unique_smiles)} unique SMILES "
            f"from {len(df)} rows in {self.feature_source_dataset}"
        )

        # Truncation statistics
        truncated = 0
        for smiles in unique_smiles:
            if len(self.smiles_tokenizer.encode(smiles)) + 5 > self.max_smiles_length:
                truncated += 1
        if truncated > 0:
            print(
                f"[token] truncated {truncated} / {len(unique_smiles)} SMILES "
                f"at max_length={self.max_smiles_length}"
            )

        meta = self._feature_cache_meta()
        if self.use_sharded_feature_cache:
            layer_specs = self._layered_cache_specs(meta)
            features = LayeredFeatureBuilder(layer_specs, shard_size=2048)
            self._cache_descriptor_needed = isinstance(
                features.parts.get("descriptor"), ShardedFeatureBuilder
            )
            failure_roots = [
                spec["root"] for spec in layer_specs.values()
            ]
            failures = []
            for root in failure_roots:
                failures_path = os.path.join(root, "failures.json")
                if os.path.isfile(failures_path):
                    with open(failures_path, encoding="utf-8") as handle:
                        failures.extend(json.load(handle))
            failures = list({
                (item.get("smiles"), item.get("error")):
                item for item in failures
            }.values())
        else:
            features, failures = self._load_partial_feature_cache(meta)
        pending_smiles = [smiles for smiles in unique_smiles if smiles not in features]
        start_time = time.time()
        workers = int(self.feature_cache_workers)
        chunksize = int(self.feature_cache_chunksize)
        print(
            "[feature_cache] build settings: "
            f"workers={workers}, chunksize={chunksize}, "
            f"total_unique_smiles={len(unique_smiles)}, pending={len(pending_smiles)}, "
            f"partial_cache_path={self.feature_cache_path}.partial"
        )

        completed_since_partial = 0
        if workers <= 1:
            iterator = tqdm(pending_smiles, desc="Building feature cache")
            for smiles in iterator:
                try:
                    data = _detach_data_storages(
                        self._compute_smiles_features(smiles)
                    )
                    features[smiles] = data
                    completed_since_partial += 1
                except Exception as exc:
                    error = str(exc)[:500]
                    failures.append({"smiles": smiles, "error": error})
                    print(f"Failed to compute features for {smiles}: {error}")
                    continue
                if (
                    not self.use_sharded_feature_cache
                    and
                    self.feature_cache_partial_every > 0
                    and completed_since_partial >= self.feature_cache_partial_every
                ):
                    self._save_partial_feature_cache(features, failures, meta)
                    completed_since_partial = 0
        else:
            # Persistent one-process-per-slot workers retain their tokenizer
            # cache. The parent owns a hard wall-clock deadline and can replace
            # a worker stuck inside an uninterruptible RDKit C++ call.
            # Use a clean interpreter even when a downstream caller has already
            # initialized CUDA. Forking such a process leaves torch.optim with
            # an invalid inherited CUDA context and previously caused every
            # PolyGen geometry to fall back during Stage 3 cache construction.
            context = mp.get_context("spawn")
            pending_iter = iter(pending_smiles)
            job_counter = 0
            primary_timeout = max(1, int(self.feature_cache_item_timeout))
            fallback_timeout = min(30, max(5, primary_timeout // 4))

            def start_worker():
                parent_conn, child_conn = context.Pipe(duplex=True)
                process = context.Process(
                    target=_feature_cache_process_loop, args=(child_conn,)
                )
                process.start()
                child_conn.close()
                return {
                    "process": process, "connection": parent_conn,
                    "smiles": None, "phase": None, "started": 0.0,
                    "job_id": None,
                }

            def stop_worker(state):
                connection = state["connection"]
                process = state["process"]
                try:
                    connection.close()
                except Exception:
                    pass
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.0)
                else:
                    process.join(timeout=0.1)

            def assign(state, smiles, phase):
                nonlocal job_counter
                payload = self._feature_cache_payload(smiles)
                if phase == "fallback":
                    payload.update({
                        "geom_input": "repeat_unit",
                        "embed_tries_multiplier": 1,
                        "conformer_3d_count": 1,
                        "conformer_keep_count": 1,
                        "conformer_profile": "fast",
                        # A hard-timeout retry must be a guaranteed cheap
                        # availability placeholder. Retrying the same 3D job
                        # with a smaller conformer count can hang again and
                        # would delete the PI1M row from every experiment.
                        "mips_use_descriptors": False,
                        "spatial_mode": "none",
                        "finite_variant": "none",
                        "conformer_mode": "none",
                        "field_layout": "none",
                        "field_channels": "none",
                        "graph_geometry_mode": (
                            "trimer_unavailable"
                            if self.graph_geometry_mode == "trimer_scage_mcl"
                            else "none"
                        ),
                    })
                job_counter += 1
                state["smiles"] = smiles
                state["phase"] = phase
                state["started"] = time.monotonic()
                state["job_id"] = job_counter
                state["connection"].send((job_counter, payload))

            states = [start_worker() for _ in range(min(workers, len(pending_smiles)))]
            exhausted = False

            def assign_next_primary(state):
                nonlocal exhausted
                if exhausted:
                    return False
                try:
                    smiles = next(pending_iter)
                except StopIteration:
                    exhausted = True
                    return False
                assign(state, smiles, "primary")
                return True

            for state in states:
                assign_next_primary(state)

            try:
                with tqdm(total=len(pending_smiles), desc="Building feature cache") as progress:
                    while any(state["smiles"] is not None for state in states):
                        made_progress = False
                        for state_idx, state in enumerate(states):
                            smiles = state["smiles"]
                            if smiles is None:
                                continue
                            result = None
                            if state["connection"].poll():
                                try:
                                    returned_job, result = pickle.loads(
                                        state["connection"].recv_bytes()
                                    )
                                    if returned_job != state["job_id"]:
                                        raise RuntimeError("feature_cache_job_id_mismatch")
                                except Exception as exc:
                                    result = {
                                        "smiles": smiles, "ok": False,
                                        "error": f"worker_result_error:{str(exc)[:400]}",
                                    }
                            else:
                                deadline = (
                                    fallback_timeout
                                    if state["phase"] == "fallback" else primary_timeout
                                )
                                timed_out = time.monotonic() - state["started"] > deadline
                                crashed = not state["process"].is_alive()
                                if timed_out or crashed:
                                    phase = state["phase"]
                                    stop_worker(state)
                                    states[state_idx] = start_worker()
                                    state = states[state_idx]
                                    if phase == "primary":
                                        assign(state, smiles, "fallback")
                                        made_progress = True
                                        continue
                                    result = {
                                        "smiles": smiles, "ok": False,
                                        "error": (
                                            f"feature_hard_timeout_fallback_failed:"
                                            f"{fallback_timeout}s"
                                        ),
                                    }

                            if result is None:
                                continue

                            if result["ok"]:
                                data = _pickle_payload_to_data(result["data_payload"])
                                if state["phase"] == "fallback":
                                    data = _mark_hard_timeout_fallback(
                                        data, primary_timeout, self.geom_input
                                    )
                                features[smiles] = data
                                completed_since_partial += 1
                            else:
                                failures.append({
                                    "smiles": smiles, "error": result["error"]
                                })
                                print(
                                    f"Failed to compute features for {smiles}: "
                                    f"{result['error']}"
                                )
                            progress.update(1)
                            made_progress = True
                            state["smiles"] = None
                            state["phase"] = None
                            state["job_id"] = None

                            if (
                                not self.use_sharded_feature_cache
                                and
                                self.feature_cache_partial_every > 0
                                and completed_since_partial
                                >= self.feature_cache_partial_every
                            ):
                                self._save_partial_feature_cache(
                                    features, failures, meta
                                )
                                completed_since_partial = 0
                            assign_next_primary(state)

                        if not made_progress:
                            time.sleep(0.02)
            finally:
                for state in states:
                    try:
                        if state["process"].is_alive():
                            state["connection"].send(None)
                            state["process"].join(timeout=2.0)
                    except Exception:
                        pass
                    stop_worker(state)

        if completed_since_partial > 0 and not self.use_sharded_feature_cache:
            self._save_partial_feature_cache(features, failures, meta)

        elapsed = max(time.time() - start_time, 1e-9)
        processed = len(pending_smiles)
        if self.use_sharded_feature_cache:
            for spec in layer_specs.values():
                failures_path = os.path.join(spec["root"], "failures.json")
                failures_tmp = failures_path + ".tmp"
                with open(failures_tmp, "w", encoding="utf-8") as handle:
                    json.dump(failures, handle, sort_keys=True)
                os.replace(failures_tmp, failures_path)
            features = features.finalize(failures, meta)
            descriptor_statistics = {}
        else:
            descriptor_statistics = _standardize_scage_descriptors(features)
        cache = {
            "meta": {
                **meta,
                "feature_cache_workers": workers,
                "feature_cache_chunksize": chunksize,
                "feature_cache_partial_every": self.feature_cache_partial_every,
                "feature_cache_item_timeout": self.feature_cache_item_timeout,
                "feature_cache_scheduler": "cpu_spawn_hard_timeout_v3",
                "feature_cache_worker_threads": 1,
                "cache_failures_are_tombstones": True,
                "build_elapsed_seconds": elapsed,
                "build_smiles_per_second": processed / elapsed,
                "failed_count": len(failures),
                "scage_descriptor_statistics": descriptor_statistics,
            },
            "features": features,
            "failures": failures,
        }
        print(
            "[feature_cache] build complete: "
            f"completed_count={len(features)}, failed_count={len(failures)}, "
            f"elapsed_seconds={elapsed:.2f}, smiles_per_second={processed / elapsed:.3f}"
        )
        return cache

    def _compute_smiles_features(self, smiles):
        """Compute all structural features for a single SMILES. Does NOT set data.y."""
        if self.graph_encoder_type == "mips_trimer_scage":
            # ``use_feature_cache=False`` is a compatibility/debug option.
            # Delegate to the canonical per-record builder so both paths share schema,
            # atom identity and invalid-geometry fallback semantics.
            return _compute_smiles_features_from_config(
                smiles=str(smiles),
                smiles_model_name=self.smiles_model_name,
                max_smiles_length=self.max_smiles_length,
                graph_input=self.graph_input,
                geom_input=self.geom_input,
                fp_mode=self.fp_mode,
                embed_tries_multiplier=self.embed_tries_multiplier,
                conformer_3d_count=self.conformer_3d_count,
                conformer_keep_count=self.conformer_keep_count,
                conformer_profile=self.conformer_profile,
                graph_encoder_type="mips_trimer_scage",
                mips_core=self.mips_core,
                mips_max_hops=self.mips_max_hops,
                mips_use_descriptors=self.mips_use_descriptors,
                mips_descriptor_protocol=self.mips_descriptor_protocol,
                spatial_mode=self.spatial_mode,
                finite_variant=self.finite_variant,
                conformer_mode=self.conformer_mode,
                field_layout=self.field_layout,
                field_channels=self.field_channels,
                graph_geometry_mode=self.graph_geometry_mode,
                topology_representation=self.topology_representation,
                trimer_num_candidates=self.trimer_num_candidates,
                trimer_max_heavy_atoms=self.trimer_max_heavy_atoms,
            )
        feature_started = time.monotonic()
        mol = Chem.MolFromSmiles(smiles)
        chemistry_valid = mol is not None
        if not chemistry_valid:
            if self.graph_encoder_type != "mips_trimer_scage":
                raise ValueError(f"Invalid SMILES: {smiles}")
            mol = Chem.MolFromSmiles("*C*")
            if mol is None:  # pragma: no cover
                raise RuntimeError("failed to construct non-PBC placeholder")

        fp_mol = Chem.Mol(mol)
        for atom in fp_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)

        geom_optimizer = "auto"
        geom_data = (
            _topology_geometry_placeholder(mol)
            if self.graph_encoder_type == "mips_trimer_scage"
            else _geometry_for_mode(mol, self.geom_input, geom_optimizer)
        )
        graph_fallback_reason = ""
        if self.graph_encoder_type == "mips_trimer_scage":
            structure, data, graph_fallback_reason = (
                _scage_graph_with_unavailable_fallback(
                    smiles if chemistry_valid else "*CC*",
                    self.graph_input, self.geom_input, geom_data,
                    self.mips_core, self.mips_max_hops,
                )
            )
            geom_data = _topology_geometry_placeholder(
                structure["structure_mol"]
            )
        else:
            structure = _structure_for_encoder(
                smiles, self.graph_input, self.geom_input, geom_data,
                self.graph_encoder_type,
                mips_core=self.mips_core,
                mips_max_hops=self.mips_max_hops,
            )
            data = build_mips_data_object(
                structure["structure_mol"],
                backbone_info=structure.get("backbone_info"),
            )
            annotate_structure_fields(data, structure, prefix="graph")

        # SMILES identity
        data.smiles = smiles

        # Tokenizer output — use global max_smiles_length
        tokenizer_output = self.smiles_tokenizer(
            smiles,
            return_tensors='pt',
            max_length=self.max_smiles_length,
            padding='max_length',
            truncation=True,
        )
        data.input_ids_smiles = tokenizer_output.input_ids
        data.attention_mask_smiles = tokenizer_output.attention_mask

        # Fingerprint
        data.fp = (
            self._compute_fingerprint(fp_mol, original_mol=mol)
            if chemistry_valid
            else torch.zeros(FP_MODE_DIMS.get(self.fp_mode, 0))
        ).unsqueeze(0)
        _attach_polymer_ecfp_target(data, smiles)

        # The periodic m-RU graph and coordinates share the exact same atom order.
        _attach_geometry_data(
            data, geom_data, smiles, self.geom_input, geom_optimizer
        )
        if self.graph_encoder_type == "mips_trimer_scage" and chemistry_valid:
            _attach_mips_descriptors(
                data, mol, protocol=self.mips_descriptor_protocol
            )
        elif self.graph_encoder_type == "mips_trimer_scage":
            _attach_unavailable_mips_descriptors(
                data, "rdkit_parse_failed"
            )
        elif self.graph_encoder_type == "mips":
            _attach_mips_descriptors(
                data, mol, protocol=self.mips_descriptor_protocol
            )
        if not chemistry_valid or graph_fallback_reason:
            data.graph_available = False
            data.mips_condition_valid = False
            data.mips_alias_free = False
            data.topology_failure_code = (
                "rdkit_parse_failed_placeholder"
                if not chemistry_valid else graph_fallback_reason
            )
        if self.graph_geometry_mode == "trimer_scage_mcl":
            attach_finite_trimer_mcl(
                data,
                smiles if chemistry_valid else "*CC*",
                num_candidates=self.trimer_num_candidates,
                max_heavy_atoms=self.trimer_max_heavy_atoms,
            )

        data.feature_compute_seconds = float(time.monotonic() - feature_started)
        return (
            _prune_nonpbc_mips_data(data)
            if self.graph_encoder_type == "mips_trimer_scage" else data
        )

    def _validate_pubchem_backend(self):
        test_mol = Chem.MolFromSmiles("CC")
        _pubchem_fingerprint_881(test_mol)

    def _fp_components(self):
        if self.fp_mode == "ecfp":
            return ["ECFP"]
        if self.fp_mode == "mixfp":
            return ["MACCSKeys", "PubChemFingerprints"]
        if self.fp_mode == "attachment_count":
            return [
                "MorganCountR2-1024", "MorganCountR3-1024",
                "AttachmentRootedCountR3-512", "PolymerScalars-10",
            ]
        return [self.fp_mode]

    def _compute_fingerprint(self, fp_mol, original_mol=None):
        if self.fp_mode == "disabled":
            return torch.empty(0, dtype=torch.float)
        if self.fp_mode == "ecfp":
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            return _bitvect_to_tensor(mfpgen.GetFingerprint(fp_mol), 1024)
        if self.fp_mode == "mixfp":
            maccs = _bitvect_to_tensor(MACCSkeys.GenMACCSKeys(fp_mol), 167)
            pubchem = _pubchem_fingerprint_881(fp_mol)
            if not isinstance(pubchem, torch.Tensor):
                pubchem = torch.as_tensor(pubchem, dtype=torch.float)
            pubchem = pubchem.flatten().to(dtype=torch.float)
            if pubchem.numel() != 881:
                raise ValueError(f"PubChemFingerprints must be 881-bit, got {pubchem.numel()}")
            return torch.cat([maccs, pubchem], dim=0)
        if self.fp_mode == "attachment_count":
            return _attachment_count_fingerprint(
                original_mol if original_mol is not None else fp_mol
            )
        raise ValueError(f"Unsupported fp_mode: {self.fp_mode}")

    def _attach_geometry(self, data, mol, smiles, geom_optimizer):
        return _attach_geometry_for_mode(
            data, mol, smiles, self.geom_input, geom_optimizer
        )

    def _build_labeled_data_list(self, feature_cache, task_csv):
        """Build self.data_list from a task CSV by looking up features in the cache."""
        features = feature_cache["features"]
        self._row_smiles = []
        # PI1M_v2 topology/geometry pretraining has no labels and uses the
        # immutable cohort order directly.  Do not load a million-row CSV or
        # materialise a million Python SMILES/bytes tuples on every DDP rank;
        # __getitem__ resolves the mmap key for the requested integer row.
        if (
            self.is_graph_only_mips_route
            and self.dataset == self.feature_source_dataset == "PI1M_v2"
            and isinstance(features, LmdbFeatureStore)
            and getattr(self, "_cohort", None) is not None
            and int(self._cohort["manifest"].get("unique_count", -1))
            == len(self._cohort["keys_array"])
            and len(self._cohort["row_keys_array"]) == len(self._cohort["keys_array"])
        ):
            self.data_list = np.arange(
                len(self._cohort["row_keys_array"]), dtype=np.int64
            )
            self._raw_targets = np.zeros(len(self.data_list), dtype=np.float64)
            self._target_override = None
            self._lazy_feature_store = features
            self._cohort_row_mode = True
            return
        df = pd.read_csv(task_csv)
        self._raw_targets = []
        self._target_override = None
        self._cohort_row_mode = False
        self._lazy_feature_store = (
            features
            if isinstance(
                features,
                (ShardedFeatureStore, LayeredFeatureStore, LmdbFeatureStore),
            )
            else None
        )
        failed_smiles = {
            str(item.get("smiles", "")).strip()
            for item in feature_cache.get("failures", [])
            if item.get("smiles") is not None
        }
        cache_misses = 0
        cached_failures = 0
        direct_cohort_lookup = (
            isinstance(features, LmdbFeatureStore)
            and getattr(self, "_cohort", None) is not None
            and self.dataset == self.feature_source_dataset
            and len(self._cohort["row_keys_array"]) == len(df)
        )

        show_progress = int(os.environ.get("LOCAL_RANK", "0")) == 0
        rows = df.iloc[:, :2].itertuples(index=False, name=None)
        for row_index, (raw_smiles, raw_target) in enumerate(tqdm(
            rows,
            total=len(df),
            desc=f"Building {self.dataset} from feature cache",
            disable=not show_progress,
        )):
            smiles = str(raw_smiles).strip()
            y = float(raw_target)
            if (
                isinstance(features, LmdbFeatureStore)
                and getattr(self, "_cohort", None) is not None
                and self.dataset == self.feature_source_dataset
                and row_index < len(self._cohort["row_keys_array"])
            ):
                lookup_key = bytes(
                    self._cohort["row_keys_array"][row_index]
                )
            else:
                lookup_key = (
                    sample_key_from_smiles(smiles)
                    if isinstance(features, LmdbFeatureStore)
                    else smiles
                )

            # The PI1M pretraining cohort is exactly the immutable feature
            # cohort and its layers were validated before this Dataset is
            # opened.  Use the row-key manifest directly instead of issuing a
            # million LMDB membership probes (which fault large mmap pages).
            if direct_cohort_lookup:
                self.data_list.append((lookup_key, y))
                self._raw_targets.append(y)
                self._row_smiles.append(smiles)
                continue

            if lookup_key in features:
                # PyG Data.__copy__ creates an independent attribute store while
                # sharing immutable cached tensors. Only the per-row target is
                # assigned below; batching creates new tensors, so duplicating
                # every graph tensor here wastes substantial RAM and startup
                # time without providing isolation that training uses.
                if self._lazy_feature_store is not None:
                    self.data_list.append((lookup_key, y))
                    self._raw_targets.append(y)
                    self._row_smiles.append(smiles)
                    continue
                data = copy.copy(features[lookup_key])
            elif smiles in failed_smiles:
                # A cache failure is a tombstone. Retrying expensive RDKit
                # geometry here made cache-only runs appear to start over.
                cached_failures += 1
                continue
            else:
                cache_misses += 1
                continue

            data.y = torch.tensor([y], dtype=torch.float)
            self.data_list.append(data)
            self._raw_targets.append(y)
            self._row_smiles.append(smiles)

        if show_progress:
            print(
                f"Built {self.dataset}: {len(self.data_list)} samples from feature cache"
            )
        if cache_misses > 0:
            raise RuntimeError(
                f"Feature cache is incomplete: {cache_misses} SMILES are neither "
                "features nor recorded failures. Rebuild the feature cache."
            )
        if cached_failures > 0 and show_progress:
            print(f"[feature_cache] skipped {cached_failures} cached failure rows")

    def _init_mts_sidecar(self, task_csv):
        """Build/load immutable downstream-only SMILES and CountFP arrays."""
        if len(self._row_smiles) != len(self.data_list):
            raise RuntimeError("MTS sidecar rows are not aligned with Dataset rows")
        specification = {
            "schema": "mts-sidecar-cache-v1",
            "dataset": self.dataset,
            "source_csv_sha256": _sha256_file(task_csv),
            "ordered_smiles_sha256": hashlib.sha256(
                "\n".join(self._row_smiles).encode("utf-8")
            ).hexdigest(),
            "modalities": list(self.modalities),
            "tokenizer": str(self.smiles_model_name) if self.mts_use_smiles else None,
            "token_cap": 256 if self.mts_use_smiles else 0,
            "smiles_views": "attachment_rooted_pair_mean" if self.mts_use_smiles else None,
            "fp_mode": self.fp_mode if self.mts_use_fp else "disabled",
            "rdkit_version": rdBase.rdkitVersion,
        }
        sidecar_hash = hashlib.sha256(
            json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        root = Path(self.root) / "processed" / "mips_trimer_scage" / "sidecars" / sidecar_hash
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata.json"
        done_path = root / ".done"
        required = []
        if self.mts_use_smiles:
            required += [
                root / "input_ids.npy", root / "attention_mask.npy",
                root / "smiles_valid.npy",
            ]
        if self.mts_use_fp:
            required += [root / "attachment_count.npy", root / "fp_valid.npy"]
        # Three Stage-3 GPU workers can start folds from the same task at the
        # same time.  Serialize only the small sidecar materialization; readers
        # immediately reuse the completed immutable files.
        sidecar_lock = open(root / ".writer.lock", "a+")
        fcntl.flock(sidecar_lock.fileno(), fcntl.LOCK_EX)
        reusable = metadata_path.is_file() and done_path.is_file() and all(
            path.is_file() for path in required
        )
        if reusable:
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            reusable = observed.get("specification") == specification
        if not reusable:
            if self.mts_use_smiles:
                views = []
                smiles_valid = np.zeros(len(self._row_smiles), dtype=np.bool_)
                for row, smiles in enumerate(self._row_smiles):
                    mol = Chem.MolFromSmiles(smiles)
                    dummy = [
                        atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
                    ] if mol is not None else []
                    boundaries = [
                        int(atom.GetNeighbors()[0].GetIdx()) for atom in dummy
                        if len(atom.GetNeighbors()) == 1
                    ]
                    if len(boundaries) == 2:
                        smiles_valid[row] = True
                        pair = [
                            Chem.MolToSmiles(mol, rootedAtAtom=index, canonical=False)
                            for index in boundaries
                        ]
                    else:
                        pair = [smiles, smiles]
                    views.extend(pair)
                encoded = self.smiles_tokenizer(
                    views, padding="max_length", truncation=True,
                    max_length=256, return_tensors="np",
                )
                ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(
                    len(self._row_smiles), 2, 256
                )
                masks = np.asarray(encoded["attention_mask"], dtype=np.uint8).reshape(
                    len(self._row_smiles), 2, 256
                )
                for name, values in (("input_ids.npy", ids), ("attention_mask.npy", masks)):
                    temporary = root / f"{name}.tmp.{os.getpid()}"
                    with open(temporary, "wb") as handle:
                        np.save(handle, values, allow_pickle=False)
                    os.replace(temporary, root / name)
                temporary = root / f"smiles_valid.npy.tmp.{os.getpid()}"
                with open(temporary, "wb") as handle:
                    np.save(handle, smiles_valid, allow_pickle=False)
                os.replace(temporary, root / "smiles_valid.npy")
            if self.mts_use_fp:
                fingerprints = np.zeros((len(self._row_smiles), 2570), dtype=np.float32)
                fp_valid = np.zeros(len(self._row_smiles), dtype=np.bool_)
                for row, smiles in enumerate(self._row_smiles):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        try:
                            fingerprints[row] = _attachment_count_fingerprint(mol).numpy()
                            fp_valid[row] = True
                        except Exception:
                            pass
                temporary = root / f"attachment_count.npy.tmp.{os.getpid()}"
                with open(temporary, "wb") as handle:
                    np.save(handle, fingerprints, allow_pickle=False)
                os.replace(temporary, root / "attachment_count.npy")
                temporary = root / f"fp_valid.npy.tmp.{os.getpid()}"
                with open(temporary, "wb") as handle:
                    np.save(handle, fp_valid, allow_pickle=False)
                os.replace(temporary, root / "fp_valid.npy")
            metadata = {
                "specification": specification,
                "files": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "shape": list(np.load(path, mmap_mode="r").shape),
                        "dtype": str(np.load(path, mmap_mode="r").dtype),
                    }
                    for path in required
                },
            }
            temporary = root / f"metadata.json.tmp.{os.getpid()}"
            temporary.write_text(
                json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, metadata_path)
            done_tmp = root / f".done.tmp.{os.getpid()}"
            done_tmp.write_text("complete\n", encoding="utf-8")
            os.replace(done_tmp, done_path)
        self.mts_sidecar_hash = sidecar_hash
        self._mts_input_ids = (
            np.load(root / "input_ids.npy", mmap_mode="r") if self.mts_use_smiles else None
        )
        self._mts_attention_mask = (
            np.load(root / "attention_mask.npy", mmap_mode="r") if self.mts_use_smiles else None
        )
        self._mts_attachment_count = (
            np.load(root / "attachment_count.npy", mmap_mode="r") if self.mts_use_fp else None
        )
        self._mts_smiles_valid = (
            np.load(root / "smiles_valid.npy", mmap_mode="r")
            if self.mts_use_smiles else None
        )
        self._mts_fp_valid = (
            np.load(root / "fp_valid.npy", mmap_mode="r")
            if self.mts_use_fp else None
        )
        rows = len(self._row_smiles)
        if self.mts_use_smiles and (
            self._mts_input_ids.shape != (rows, 2, 256)
            or self._mts_attention_mask.shape != (rows, 2, 256)
            or self._mts_smiles_valid.shape != (rows,)
        ):
            raise RuntimeError("MTS SMILES sidecar shape mismatch")
        if self.mts_use_fp and (
            self._mts_attachment_count.shape != (rows, 2570)
            or self._mts_attachment_count.dtype != np.float32
            or self._mts_fp_valid.shape != (rows,)
        ):
            raise RuntimeError("MTS CountFP sidecar shape/dtype mismatch")
        fcntl.flock(sidecar_lock.fileno(), fcntl.LOCK_UN)
        sidecar_lock.close()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    @property
    def raw_targets(self):
        """Return immutable raw labels aligned with ``data_list`` rows."""
        if len(self._raw_targets) != len(self.data_list):
            values = []
            for item in self.data_list:
                if isinstance(item, tuple):
                    values.append(float(item[1]))
                else:
                    values.append(float(item.y.reshape(-1)[0].item()))
            self._raw_targets = values
        return np.asarray(self._raw_targets, dtype=np.float64)

    def set_target_override(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.data_list):
            raise ValueError(
                f"target override length {values.shape[0]} != dataset length "
                f"{len(self.data_list)}"
            )
        self._target_override = values.copy()

    def clear_target_override(self):
        self._target_override = None

    def __getitem__(self, idx):
        item = self.data_list[idx]
        lookup_key_for_hash = None
        if (
            self._cohort_row_mode
            and isinstance(self._lazy_feature_store, LmdbFeatureStore)
        ):
            lookup_key = self._cohort["row_keys_array"][int(idx)]
            lookup_key_for_hash = bytes(lookup_key)
            data = copy.copy(self._lazy_feature_store[lookup_key])
            target_value = (
                self._target_override[idx]
                if self._target_override is not None else 0.0
            )
            data.y = torch.tensor([target_value], dtype=torch.float)
        elif (
            getattr(self, "_lazy_feature_store", None) is not None
            and isinstance(item, tuple)
        ):
            lookup_key, target = item
            lookup_key_for_hash = bytes(lookup_key)
            data = copy.copy(self._lazy_feature_store[lookup_key])
            target_value = (
                self._target_override[idx]
                if self._target_override is not None
                else target
            )
            data.y = torch.tensor([target_value], dtype=torch.float)
        else:
            data = item
            if self._target_override is not None:
                data = copy.copy(data)
                data.y = torch.tensor(
                    [self._target_override[idx]], dtype=torch.float
                )
        if lookup_key_for_hash is None:
            lookup_key_for_hash = sample_key_from_smiles(str(data.smiles))
        # A compact stable identity supports vectorised stateless augmentation
        # without carrying or hashing SMILES strings on the GPU hot path.
        data.mts_sample_hash64 = torch.tensor(
            int.from_bytes(lookup_key_for_hash[:8], "little")
            & ((1 << 63) - 1),
            dtype=torch.long,
        )
        if self._periodic_line_glt_sidecar is not None:
            row_hint = int(idx) if self._cohort_row_mode else None
            line_row = self._periodic_line_glt_sidecar.model_row(
                self._periodic_line_glt_sidecar.index_for_key(
                    lookup_key_for_hash, row_hint=row_hint
                )
            )
            tokens, relations = line_row["tokens"], line_row["relations"]
            active_line_schema = getattr(
                self._periodic_line_glt_sidecar, "metadata", {}
            ).get("schema")
            is_image_v1 = (
                active_line_schema == PERIODIC_LINE_IMAGE_SCHEMA
                or str(active_line_schema).startswith(
                    (PERIODIC_LINE_DISTILL_SCHEMA_PREFIX, "mts-periodic-line-distill-v2-")
                )
            )
            prefix = "glt3_" if is_image_v1 else "glt_"
            setattr(data, prefix + "geometry_valid", bool(line_row["geometry_valid"]))
            if is_image_v1:
                data.glt3_query_valid = bool(
                    line_row["geometry_valid"] and len(tokens["token_atom_a"]) > 0
                )
                for name, value in tokens.items():
                    dtype = torch.float32 if name == "token_distance" else (
                        torch.bool if name == "token_valid" else torch.long
                    )
                    setattr(data, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
                for name, value in relations.items():
                    dtype = torch.float32 if name == "relation_angle" else (
                        torch.bool if name == "relation_valid" else torch.long
                    )
                    setattr(data, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
                return data
            data.glt_geometry_valid = bool(line_row["geometry_valid"])
            data.glt_query_valid = bool(line_row["geometry_valid"] and len(tokens["token_atom_a"]) > 0)
            data.glt_base_mapping_valid = bool(line_row.get("base_mapping_valid", line_row["geometry_valid"]))
            data.glt_invalid_reason = int(line_row.get("invalid_reason", 0))
            for name, value in tokens.items():
                dtype = torch.float32 if name == "token_observation_distances" else (
                    torch.bool if name in {"token_valid", "token_runtime_valid", "token_observation_valid"} else torch.long
                )
                setattr(
                    data,
                    f"glt_{name}",
                    torch.as_tensor(np.array(value, copy=True), dtype=dtype),
                )
            for name, value in relations.items():
                dtype = torch.float32 if name == "relation_observation_angles" else (
                    torch.bool if name in {"relation_valid", "relation_runtime_valid", "relation_observation_valid", "relation_is_fallback"}
                    else torch.long
                )
                setattr(
                    data,
                    f"glt_{name}",
                    torch.as_tensor(np.array(value, copy=True), dtype=dtype),
                )
        if self._periodic_line_glt_sidecar is not None:
            if hasattr(data, "glt_token_runtime_valid"):
                data.glt_token_valid = data.glt_token_runtime_valid
            if hasattr(data, "glt_relation_runtime_valid"):
                data.glt_relation_valid = data.glt_relation_runtime_valid
        return data
