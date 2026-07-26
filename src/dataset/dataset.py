import copy
import hashlib
import multiprocessing as mp
import pickle
import random
import glob
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import torch
import pandas as pd
from rdkit import Chem, rdBase
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from .geom_data import (
    mol2coords,
    mol2periodic_pbc_coords,
    mol2polygen_periodic_coords,
    mol2screw_periodic_coords,
    mol2smer_context_coords,
    set_conformer_generation_config,
)
from .graph_data import (
    MIPSPeriodicConfig,
    annotate_structure_fields,
    attach_periodic_lga_topology,
    build_mips_paper_structure,
    build_mips_periodic_structure,
    build_polygen_periodic_structure,
    build_structure_for_input,
    generate_multimer_smiles,
    mol_to_graph_data_obj_simple,
    periodicity_augment_smiles,
)
from .diagnostics import (
    print_dataset_diagnostics,
    summarize_feature_cache,
    write_dataset_diagnostics,
)
from transformers import AutoTokenizer
from rdkit.Chem import rdFingerprintGenerator


FP_MODE_DIMS = {
    "ecfp": 1024,
    "mixfp": 1048,
}

SCAGE_DESCRIPTOR_DIMS = {
    "shape": 11,
    "usrcat": 60,
    "autocorr3d": 80,
    "rdf": 210,
    "morse": 224,
    "whim": 114,
}


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
_WORKER_POLYMER_MORGAN_GENERATOR = None
_WORKER_MIPS_MD_GENERATOR = None


class _FeatureCacheItemTimeout(TimeoutError):
    pass


def _feature_cache_timeout_handler(_signum, _frame):
    raise _FeatureCacheItemTimeout("feature_cache_item_timeout")


def _get_worker_tokenizer(smiles_model_name):
    global _WORKER_TOKENIZER, _WORKER_TOKENIZER_NAME
    if _WORKER_TOKENIZER is None or _WORKER_TOKENIZER_NAME != smiles_model_name:
        _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(smiles_model_name)
        _WORKER_TOKENIZER_NAME = smiles_model_name
    return _WORKER_TOKENIZER


def _get_worker_morgan_generator():
    global _WORKER_MORGAN_GENERATOR
    if _WORKER_MORGAN_GENERATOR is None:
        _WORKER_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return _WORKER_MORGAN_GENERATOR


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


def _mips_capped_monomer_conformer(repeating_monomer, seed):
    """Create the paper's descriptor conformer from one H-capped repeat unit."""
    capped = Chem.RWMol(Chem.Mol(repeating_monomer))
    dummy_count = 0
    for atom in capped.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            dummy_count += 1
    if dummy_count != 2:
        raise ValueError("MIPS descriptors require exactly two attachment atoms")
    capped = capped.GetMol()
    Chem.SanitizeMol(capped)
    capped_h = Chem.AddHs(capped)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(capped_h, params) != 0:
        raise ValueError("MIPS monomer descriptor ETKDG embedding failed")
    if AllChem.MMFFHasAllMoleculeParams(capped_h):
        status = AllChem.MMFFOptimizeMolecule(capped_h, maxIters=200)
        optimizer = "MMFF"
    else:
        status = AllChem.UFFOptimizeMolecule(capped_h, maxIters=200)
        optimizer = "UFF"
    descriptor_mol = Chem.RemoveHs(capped_h)
    if descriptor_mol.GetNumConformers() != 1:
        raise ValueError("MIPS monomer descriptor conformer is missing")
    return descriptor_mol, optimizer, int(status)


def _attach_mips_descriptors(data, repeating_monomer):
    """Attach MIPS descriptors computed from the original repeating monomer."""
    data.mips_md = torch.zeros(200, dtype=torch.float)
    data.mips_atom_pair_3d = torch.zeros(512, dtype=torch.float)
    data.mips_descriptor_valid = False
    data.mips_descriptor_failed_reason = ""
    data.mips_descriptor_source = "h_capped_repeating_monomer"
    data.mips_descriptor_optimizer = "none"
    data.mips_descriptor_optimization_status = -1
    try:
        raw_smiles = Chem.MolToSmiles(repeating_monomer, canonical=True)
        seed = int(hashlib.sha1(raw_smiles.encode()).hexdigest()[:7], 16)
        descriptor_mol, optimizer, status = _mips_capped_monomer_conformer(
            repeating_monomer, seed
        )
        descriptor_smiles = Chem.MolToSmiles(descriptor_mol, canonical=True)
        md = _get_worker_mips_md_generator().process(descriptor_smiles)
        if md is None or len(md) != 201:
            raise ValueError("RDKit2DNormalized did not return 200 descriptors")
        md_tensor = torch.as_tensor(md[1:], dtype=torch.float)
        if not torch.isfinite(md_tensor).all():
            raise ValueError("RDKit2DNormalized contains non-finite values")

        atom_pair = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            descriptor_mol, nBits=512, use2D=False
        )
        data.mips_md = md_tensor
        data.mips_atom_pair_3d = _bitvect_to_tensor(atom_pair, 512)
        data.mips_descriptor_valid = True
        data.mips_descriptor_optimizer = optimizer
        data.mips_descriptor_optimization_status = status
    except Exception as exc:
        data.mips_descriptor_failed_reason = str(exc)[:300]
    data.mips_descriptor_schema_version = 2
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


def _attach_periodic_aug_views(data, smiles, max_views=4):
    seed = int.from_bytes(hashlib.sha256(str(smiles).encode("utf-8")).digest()[:8], "little")
    rng = random.Random(seed)
    views, cuts = [], []
    for _ in range(max(8, int(max_views) * 5)):
        if len(views) >= int(max_views):
            break
        try:
            augmented, _, metadata = periodicity_augment_smiles(
                smiles, max_mrus=3, return_n=True, return_metadata=True, rng=rng
            )
        except Exception:
            continue
        cut = metadata.get("cut_identity")
        if str(augmented) == str(smiles) or str(augmented) in views or cut in cuts:
            continue
        views.append(str(augmented))
        cuts.append(cut)
    data.periodic_aug_smiles = views
    data.periodic_aug_cut_identities = cuts
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


def _compute_fingerprint_for_mode(fp_mol, fp_mode):
    fp_mode = str(fp_mode).lower()
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


def _geometry_for_mode(mol, geom_input, geom_optimizer):
    if geom_input == "periodic_pbc":
        return mol2periodic_pbc_coords(mol, optimizer=geom_optimizer)
    if geom_input == "polygen_periodic":
        return mol2polygen_periodic_coords(mol, optimizer=geom_optimizer)
    if geom_input == "screw_periodic":
        return mol2screw_periodic_coords(mol, optimizer=geom_optimizer)
    if geom_input == "smer_context":
        return mol2smer_context_coords(mol, optimizer=geom_optimizer)
    return mol2coords(mol, process_stars=True, optimizer=geom_optimizer)


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
    geom_data = _geometry_for_mode(mol, geom_input, geom_optimizer)
    return _attach_geometry_data(data, geom_data, smiles, geom_input, geom_optimizer)


def _structure_for_geometry(smiles, graph_input, geom_input, geom_data):
    if (
        geom_input == "polygen_periodic"
        and bool(getattr(geom_data, "polygen_periodic_valid", False))
    ):
        return build_polygen_periodic_structure(
            smiles, int(getattr(geom_data, "periodic_ru_count", 1))
        )
    return build_structure_for_input(smiles, graph_input)


def _structure_for_encoder(
    smiles, graph_input, geom_input, geom_data, graph_encoder_type
):
    if str(graph_encoder_type).lower() == "mips":
        return build_mips_paper_structure(smiles, distance_threshold=3)
    if str(graph_encoder_type).lower() == "scage":
        geometry_valid = bool(
            geom_input == "polygen_periodic"
            and getattr(geom_data, "polygen_periodic_valid", False)
        )
        return build_mips_periodic_structure(
            smiles,
            geometry_num_ru=int(getattr(geom_data, "periodic_ru_count", 1)),
            geometry_valid=geometry_valid,
            config=MIPSPeriodicConfig(),
        )
    return _structure_for_geometry(smiles, graph_input, geom_input, geom_data)


def _expand_polygen_geometry_for_model(geom_data, structure):
    """Copy a strict primitive PBC cell into its MIPS model supercell."""
    geometry_valid = bool(getattr(geom_data, "polygen_periodic_valid", False))
    graph_available = bool(structure.get("graph_available", False))
    repeat_factor = int(structure.get("mips_repeat_factor", 0))
    if not geometry_valid or not graph_available or repeat_factor < 1:
        # New SCAGE never consumes a finite-chain/ordinary-3D fallback. Keep a
        # shape-compatible zero placeholder solely because the shared collate
        # contract also serves flat4/PaiNN.
        atoms = list(structure["structure_mol"].GetAtoms())
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in atoms], dtype=torch.long
        )
        geom_data.z = atomic_numbers
        geom_data.pos = torch.zeros((len(atoms), 3), dtype=torch.float)
        geom_data.pos_confs = geom_data.pos.unsqueeze(0)
        geom_data.cell = torch.zeros((3, 3), dtype=torch.float)
        geom_data.cell_confs = geom_data.cell.unsqueeze(0)
        geom_data.pbc = torch.tensor([False, False, False], dtype=torch.bool)
        geom_data.geom_pool_mask = torch.zeros(len(atoms), dtype=torch.bool)
        geom_data.graph_to_geom_index = torch.arange(len(atoms), dtype=torch.long)
        geom_data.geom_build_ok = False
        geom_data.geom_coordinate_ok = False
        geom_data.periodic_valid = False
        geom_data.periodic_cell_length = 0.0
        geom_data.geometry_period_ru = int(structure.get("geometry_period_ru", 0))
        geom_data.model_cell_ru = int(structure.get("model_cell_ru", 1))
        geom_data.periodic_geometry_valid = False
        return geom_data
    if repeat_factor == 1:
        geom_data.geometry_period_ru = int(
            structure.get("geometry_period_ru", getattr(geom_data, "periodic_ru_count", 1))
        )
        geom_data.model_cell_ru = int(structure.get("model_cell_ru", geom_data.periodic_ru_count))
        geom_data.periodic_geometry_valid = True
        return geom_data

    positions = geom_data.pos_confs.float()
    cells = geom_data.cell_confs.float()
    if positions.dim() != 3 or cells.dim() != 3 or cells.size(0) != positions.size(0):
        raise ValueError("polygen supercell expansion requires matching pos_confs/cell_confs")
    expanded_positions = []
    expanded_cells = []
    for conf_idx in range(positions.size(0)):
        base_translation = cells[conf_idx, 2]
        copies = [
            positions[conf_idx] + float(copy_idx) * base_translation
            for copy_idx in range(repeat_factor)
        ]
        expanded_positions.append(torch.cat(copies, dim=0))
        final_cell = cells[conf_idx].clone()
        final_cell[2] = final_cell[2] * float(repeat_factor)
        expanded_cells.append(final_cell)
    geom_data.pos_confs = torch.stack(expanded_positions, dim=0)
    geom_data.pos = geom_data.pos_confs[0]
    geom_data.cell_confs = torch.stack(expanded_cells, dim=0)
    geom_data.cell = geom_data.cell_confs[0]
    geom_data.z = geom_data.z.repeat(repeat_factor)
    geom_data.geom_pool_mask = torch.ones(geom_data.pos.size(0), dtype=torch.bool)
    geom_data.graph_to_geom_index = torch.arange(geom_data.pos.size(0), dtype=torch.long)
    geom_data.geometry_period_ru = int(structure["geometry_period_ru"])
    geom_data.model_cell_ru = int(structure["model_cell_ru"])
    geom_data.periodic_ru_count = int(structure["model_cell_ru"])
    geom_data.periodic_cell_length = float(torch.linalg.vector_norm(geom_data.cell[2]))
    fractional_confs = []
    for conf_idx in range(geom_data.pos_confs.size(0)):
        length = torch.linalg.vector_norm(geom_data.cell_confs[conf_idx, 2]).clamp_min(1e-8)
        position = geom_data.pos_confs[conf_idx]
        fractional_confs.append(torch.stack([
            position[:, 0] / 55.0 + 0.5,
            position[:, 1] / 55.0 + 0.5,
            position[:, 2] / length,
        ], dim=-1))
    geom_data.periodic_fractional_pos_confs = torch.stack(fractional_confs, dim=0)
    geom_data.periodic_fractional_pos = geom_data.periodic_fractional_pos_confs[0]
    geom_data.periodic_geometry_valid = True
    return geom_data


def _attach_lga_geometry(data):
    """Compute conformer-matched Euclidean distances for cached LGA edges."""
    edge_count = int(data.lga_edge_index.size(1))
    geometry_valid = bool(
        getattr(data, "graph_available", False)
        and getattr(data, "polygen_periodic_valid", False)
        and getattr(data, "periodic_geometry_valid", False)
        and data.pos_confs.dim() == 3
        and data.pos_confs.size(1) == data.x.size(0)
        and data.cell_confs.dim() == 3
    )
    if not geometry_valid:
        data.lga_pbc_distance_confs = torch.zeros((1, edge_count), dtype=torch.float)
        data.lga_geometry_valid = torch.zeros(edge_count, dtype=torch.bool)
        data.periodic_geometry_valid = False
        return data

    source, target = data.lga_edge_index.long()
    shift = data.lga_image_shift.float()
    distances = []
    for conf_idx in range(data.pos_confs.size(0)):
        positions = data.pos_confs[conf_idx]
        translation = data.cell_confs[conf_idx, 2]
        displacement = (
            positions[source]
            + shift.unsqueeze(-1) * translation.unsqueeze(0)
            - positions[target]
        )
        distances.append(torch.linalg.vector_norm(displacement, dim=-1))
    data.lga_pbc_distance_confs = torch.stack(distances, dim=0).float()
    data.lga_geometry_valid = torch.ones(edge_count, dtype=torch.bool)
    data.periodic_geometry_valid = True
    return data


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
    graph_encoder_type="gin",
):
    set_conformer_generation_config(
        conformer_3d_count=conformer_3d_count,
        conformer_keep_count=conformer_keep_count,
        embed_tries_multiplier=embed_tries_multiplier,
        conformer_profile=conformer_profile,
    )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp_mol = Chem.Mol(mol)
    for atom in fp_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)

    geom_data = _geometry_for_mode(mol, geom_input, geom_optimizer="auto")
    structure = _structure_for_encoder(
        smiles, graph_input, geom_input, geom_data, graph_encoder_type
    )
    data = mol_to_graph_data_obj_simple(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    annotate_structure_fields(data, structure, prefix="graph")
    if graph_encoder_type == "scage":
        attach_periodic_lga_topology(data, structure, config=MIPSPeriodicConfig())
        geom_data = _expand_polygen_geometry_for_model(geom_data, structure)
    data.smiles = smiles

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
    data.fp = _compute_fingerprint_for_mode(fp_mol, fp_mode).unsqueeze(0)
    _attach_polymer_ecfp_target(data, smiles)
    _attach_periodic_aug_views(data, smiles)
    _attach_geometry_data(data, geom_data, smiles, geom_input, geom_optimizer="auto")
    if graph_encoder_type == "scage":
        _attach_lga_geometry(data)
    elif graph_encoder_type == "mips":
        _attach_mips_descriptors(data, mol)
    return data


def _compute_smiles_features_worker(payload):
    smiles = payload["smiles"]
    feature_started = time.monotonic()
    timeout_seconds = max(0, int(payload.get("feature_cache_item_timeout", 0)))
    previous_handler = None
    try:
        if timeout_seconds > 0:
            previous_handler = signal.signal(signal.SIGALRM, _feature_cache_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        # Repeated UFF warnings from dozens of workers serialize stderr and
        # materially slow PI1M cache builds. Failure metadata remains attached
        # to each sample, so suppress only RDKit's console stream here.
        with rdBase.BlockLogs():
            data = _compute_smiles_features_from_config(
                smiles=smiles,
                smiles_model_name=payload["smiles_model_name"],
                max_smiles_length=payload["max_smiles_length"],
                graph_input=payload["graph_input"],
                geom_input=payload["geom_input"],
                fp_mode=payload["fp_mode"],
                embed_tries_multiplier=payload["embed_tries_multiplier"],
                conformer_3d_count=payload["conformer_3d_count"],
                conformer_keep_count=payload["conformer_keep_count"],
                conformer_profile=payload["conformer_profile"],
                graph_encoder_type=payload["graph_encoder_type"],
            )
        data.feature_compute_seconds = float(time.monotonic() - feature_started)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return {
            "smiles": smiles,
            "data_payload": _data_to_pickle_payload(data),
            "ok": True,
            "error": "",
        }
    except _FeatureCacheItemTimeout:
        # Preserve topology/SMILES/FP coverage without presenting a timed-out
        # conformer as valid geometry to SCAGE. The fallback has its own short
        # deadline so a pathological molecule cannot occupy a worker forever.
        signal.setitimer(signal.ITIMER_REAL, 30)
        try:
            data = _compute_smiles_features_from_config(
                smiles=smiles,
                smiles_model_name=payload["smiles_model_name"],
                max_smiles_length=payload["max_smiles_length"],
                graph_input=payload["graph_input"],
                geom_input="repeat_unit",
                fp_mode=payload["fp_mode"],
                embed_tries_multiplier=1,
                conformer_3d_count=1,
                conformer_keep_count=1,
                conformer_profile="fast",
                graph_encoder_type=payload["graph_encoder_type"],
            )
            data.feature_compute_seconds = float(time.monotonic() - feature_started)
            data.geom_build_ok = False
            data.geom_coordinate_ok = False
            data.geom_input = "repeat_unit_fallback"
            data.geom_context = "feature_timeout_topology_fallback"
            data.geom_context_id = 0
            data.geom_failed_reason = f"feature_cache_item_timeout:{timeout_seconds}s"
            data.geom_primary_failed_reason = "feature_cache_item_timeout"
            data.geom_screw_source = "topology_fallback"
            data.geom_screw_source_id = 0
            data.geometry_source_id = 0
            data.screw_valid = False
            data.smer_valid = False
            data.polygen_periodic_valid = False
            data.periodic_valid = False
            data.periodic_closure_error = float("inf")
            data.periodic_ru_count = 1
            data.periodic_cell_length = 0.0
            data.pbc = torch.tensor([False, False, False], dtype=torch.bool)
            data.cell = torch.zeros((3, 3), dtype=torch.float)
            data.feature_timeout_fallback = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            return {
                "smiles": smiles,
                "data_payload": _data_to_pickle_payload(data),
                "ok": True,
                "error": "",
            }
        except Exception as exc:
            return {
                "smiles": smiles,
                "data": None,
                "ok": False,
                "error": f"feature_timeout_fallback_failed:{str(exc)[:400]}",
            }
    except Exception as exc:
        return {"smiles": smiles, "data": None, "ok": False, "error": str(exc)[:500]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)


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
    if hasattr(data, "lga_edge_index"):
        edge_count = int(data.lga_edge_index.size(1))
        data.lga_geometry_valid = torch.zeros(edge_count, dtype=torch.bool)
        data.lga_pbc_distance_confs = torch.zeros(
            (1, edge_count), dtype=torch.float
        )
    data.feature_timeout_fallback = True
    return data


_SCAGE_GRAPH_CACHE_KEYS = {
    "x", "edge_index", "edge_attr", "atomic_num", "chiral_tag", "degree",
    "explicit_valence", "formal_charge", "hybridization", "is_aromatic",
    "total_numHs", "atom_is_in_ring", "mass", "van_der_waals_radius",
    "partial_charge", "scage_backbone_role", "scage_spd",
    "scage_path_bond_fields", "scage_topology_schema_version",
    "attachment_pair", "star_link_edge", "ordered_backbone_path",
    "star_link_metadata_valid", "graph_build_ok", "graph_failed_reason",
    "graph_smiles", "graph_input", "structure_smiles", "structure_input",
    "requested_graph_input", "attachment_count", "has_backbone_features",
}


def _migrate_scage_cached_feature(payload):
    smiles, old = payload
    if isinstance(old, (bytes, bytearray)):
        old = pickle.loads(old)
    structure = build_structure_for_input(smiles, "star_linking")
    new = mol_to_graph_data_obj_simple(
        structure["structure_mol"], backbone_info=structure.get("backbone_info")
    )
    annotate_structure_fields(new, structure, prefix="graph")
    for key in old.keys():
        if key not in _SCAGE_GRAPH_CACHE_KEYS and not key.startswith("scage_descriptor_"):
            setattr(new, key, copy.deepcopy(old[key]))
    new.smiles = smiles
    _attach_periodic_aug_views(new, smiles)
    _attach_scage_3d_descriptors(new, structure["structure_mol"])
    return smiles, new


class UniDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        smiles_model_name,
        geometry_encoder='painn',
        graph_encoder_type='gin',
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
        embed_tries_multiplier=8,
        conformer_3d_count=8,
        conformer_keep_count=4,
        conformer_profile='full',
        scage_distance_mode='bias',
        scage_distance_rbf=32,
        scage_distance_cutoff=12.0,
        transform=None,
        pre_transform=None
    ):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_list = []
        self.graph_input = str(graph_input).lower()
        if self.graph_input not in {'repeat_unit', 'star_linking'}:
            raise ValueError("graph_input must be 'repeat_unit' or 'star_linking'")
        self.geom_input = str(geom_input).lower()
        if self.geom_input not in {
            'repeat_unit', 'periodic_pbc', 'polygen_periodic', 'screw_periodic', 'smer_context'
        }:
            raise ValueError(
                "geom_input must be repeat_unit, periodic_pbc, polygen_periodic, "
                "screw_periodic, or smer_context"
            )
        self.fp_mode = str(fp_mode).lower()
        if self.fp_mode not in FP_MODE_DIMS:
            raise ValueError("fp_mode must be 'ecfp' or 'mixfp'")
        self.fp_dim = FP_MODE_DIMS[self.fp_mode]
        if self.fp_mode == "mixfp":
            self._validate_pubchem_backend()
        self.feature_cache_workers = max(0, int(feature_cache_workers))
        self.feature_cache_chunksize = max(1, int(feature_cache_chunksize))
        self.feature_cache_partial_every = max(0, int(feature_cache_partial_every))
        self.feature_cache_item_timeout = max(0, int(feature_cache_item_timeout))
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
        set_conformer_generation_config(
            conformer_3d_count=self.conformer_3d_count,
            conformer_keep_count=self.conformer_keep_count,
            embed_tries_multiplier=self.embed_tries_multiplier,
            conformer_profile=self.conformer_profile,
        )
        self.smiles_model_name = smiles_model_name
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)

        self.geometry_encoder = geometry_encoder.lower()
        if self.geometry_encoder != 'painn':
            raise ValueError("geometry_encoder must be 'painn'")
        self.graph_encoder_type = str(graph_encoder_type).lower()
        if self.graph_encoder_type not in {'gin', 'scage', 'mips'}:
            raise ValueError("graph_encoder_type must be 'gin', 'scage', or 'mips'")

        cache_namespace = self.graph_encoder_type if self.graph_encoder_type in {'scage', 'mips'} else 'painn'
        processed_dir = os.path.join(self.root, 'processed', cache_namespace)
        os.makedirs(processed_dir, exist_ok=True)

        graph_tag = f'{self.graph_encoder_type}-backbone'
        if self.graph_input == 'star_linking':
            graph_tag = f'{self.graph_encoder_type}-starlink-backbone'
        if self.graph_encoder_type == 'scage':
            scage_schema = 'scage-mips-pbc-lga-v1'
            graph_tag = f'{graph_tag}-{scage_schema}'

        self.use_feature_cache = bool(use_feature_cache)

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

        if self.use_feature_cache:
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
        else:
            self._init_legacy(processed_dir=processed_dir, graph_tag=f"{graph_tag}_{geom_tag}_{fp_tag}")

    # ------------------------------------------------------------------
    # Legacy path (--disable_feature_cache)
    # ------------------------------------------------------------------
    def _init_legacy(self, processed_dir, graph_tag):
        self.processed_file = os.path.join(processed_dir, f"{self.dataset}_{graph_tag}.pt")

        if os.path.exists(self.processed_file):
            self.data_list = torch.load(self.processed_file, weights_only=False)
            print(f"loaded {self.processed_file} with {len(self.data_list)} samples")
        else:
            csv_path = f"{self.root}/raw/{self.dataset}.csv"
            self.max_length_smiles = self._compute_max_token_length_for_file(csv_path)
            self.process()
            torch.save(self.data_list, self.processed_file)
            print(f"processed and saved {self.processed_file} with {len(self.data_list)} samples")

    def process(self):
        """Legacy per-dataset processing (used when use_feature_cache=False)."""
        csv_path = f"{self.root}/raw/{self.dataset}.csv"
        df = pd.read_csv(csv_path)
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            smiles = row[0]
            property = row[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp_mol = Chem.Mol(mol)
            for atom in fp_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)

            geom_optimizer = "auto"
            geom_data = _geometry_for_mode(mol, self.geom_input, geom_optimizer)
            structure = _structure_for_encoder(
                smiles, self.graph_input, self.geom_input, geom_data,
                self.graph_encoder_type,
            )
            data = mol_to_graph_data_obj_simple(
                structure["structure_mol"], backbone_info=structure.get("backbone_info")
            )
            annotate_structure_fields(data, structure, prefix="graph")
            if self.graph_encoder_type == "scage":
                attach_periodic_lga_topology(data, structure, config=MIPSPeriodicConfig())
                geom_data = _expand_polygen_geometry_for_model(geom_data, structure)

            # Labels
            data.y = torch.tensor([property], dtype=torch.float)
            data.smiles = smiles

            # Tokenize with dynamic max_length
            tokenizer_output = self.smiles_tokenizer(
                smiles,
                return_tensors='pt',
                max_length=self.max_length_smiles + 5,
                padding='max_length',
                truncation=True
            )
            data.input_ids_smiles = tokenizer_output.input_ids
            data.attention_mask_smiles = tokenizer_output.attention_mask

            data.fp = self._compute_fingerprint(fp_mol).unsqueeze(0)
            _attach_polymer_ecfp_target(data, smiles)
            _attach_periodic_aug_views(data, smiles)
            try:
                _attach_geometry_data(
                    data, geom_data, smiles, self.geom_input, geom_optimizer
                )
                if self.graph_encoder_type == "scage":
                    _attach_lga_geometry(data)
                elif self.graph_encoder_type == "mips":
                    _attach_mips_descriptors(data, mol)
            except Exception as e:
                print(e)
                print(f"Failed to generate 3D coordinates for {smiles}")
                continue
            self.data_list.append(data)

        print("Dataset processed. Total samples:", len(self.data_list))

    # ------------------------------------------------------------------
    # Feature cache path
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
        max_smiles_length_cap = int(max_smiles_length_cap)

        # Determine global max_smiles_length
        if max_smiles_length is not None:
            self.max_smiles_length = int(max_smiles_length)
        else:
            source_csv = f"{self.root}/raw/{self.feature_source_dataset}.csv"
            raw_max = self._compute_max_token_length_for_file(source_csv)
            self.max_smiles_length = min(raw_max + 5, max_smiles_length_cap)
            if raw_max + 5 > max_smiles_length_cap:
                print(
                    f"[token] max_length capped at {max_smiles_length_cap} "
                    f"(raw max + 5 = {raw_max + 5})"
                )

        # Feature cache file path
        self.feature_cache_path = os.path.join(
            processed_dir,
            (
                f"feature_cache_{self.feature_source_dataset}_{graph_tag}_{geom_tag}_"
                f"{fp_tag}_tok{self.max_smiles_length}.pt"
            ),
        )

        # Build or load feature cache
        cache_was_built = rebuild_feature_cache or not os.path.exists(
            self.feature_cache_path
        )
        if cache_was_built:
            print(f"building feature cache from {self.feature_source_dataset} ...")
            feature_cache = self._build_feature_cache()
            torch.save(feature_cache, self.feature_cache_path)
            print(f"saved feature cache to {self.feature_cache_path}")
        else:
            # Feature tensors are immutable dataset inputs. Private mmap avoids
            # eagerly copying the multi-GB tensor storages into every DDP rank;
            # PyTorch still unpickles independent Data objects and any accidental
            # write is copy-on-write rather than modifying the cache file.
            feature_cache = torch.load(
                self.feature_cache_path,
                weights_only=False,
                mmap=True,
            )
            if self.graph_encoder_type == "mips":
                cache_meta = feature_cache.get("meta", {})
                if (
                    cache_meta.get("mips_input_schema_version") != 2
                    or cache_meta.get("mips_descriptor_schema_version") != 2
                ):
                    raise RuntimeError(
                        "The existing MIPS feature cache predates the paper-aligned "
                        "short-RU/backbone/descriptor implementation. Re-run with "
                        "--rebuild_feature_cache."
                    )
            elif self.graph_encoder_type == "scage":
                cache_meta = feature_cache.get("meta", {})
                if (
                    cache_meta.get("scage_data_schema") != "scage-mips-pbc-lga-v1"
                    or cache_meta.get("periodic_lga_schema_version") != 1
                ):
                    raise RuntimeError(
                        "The existing SCAGE cache is incompatible with the sparse "
                        "MIPS-PBC LGA route. Re-run with --rebuild_feature_cache."
                    )
            print(
                f"loaded feature cache {self.feature_cache_path} "
                f"with {len(feature_cache['features'])} entries"
            )

        # Diagnostics scan every cached graph and write a shared JSON file. In
        # DDP this work is identical on every rank and concurrent writes race,
        # so only the global rank zero process performs it.
        if int(os.environ.get("RANK", "0")) == 0:
            self._write_feature_cache_diagnostics(
                feature_cache,
                processed_dir,
                graph_tag,
                geom_tag,
                fp_tag,
                reuse_existing=not cache_was_built,
            )

        # Build labeled data list from the task CSV
        task_csv = f"{self.root}/raw/{self.dataset}.csv"
        self._build_labeled_data_list(feature_cache, task_csv)

    def _migrate_model_dependent_scage_cache(self, processed_dir, graph_tag, geom_tag, fp_tag):
        if self.graph_encoder_type != 'scage':
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
        summary = summarize_feature_cache(
            feature_cache["features"],
            graph_input=self.graph_input,
            geom_input=self.geom_input,
            painn_cutoff=10.0,
        )
        summary["cache"] = {
            "feature_cache_path": self.feature_cache_path,
            "feature_source_dataset": self.feature_source_dataset,
            "dataset": self.dataset,
            "geometry_encoder": self.geometry_encoder,
            "graph_encoder_type": self.graph_encoder_type,
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
            "scage_distance_mode": self.scage_distance_mode if self.graph_encoder_type == "scage" else None,
            "scage_distance_rbf": self.scage_distance_rbf if self.graph_encoder_type == "scage" else None,
            "scage_distance_cutoff": self.scage_distance_cutoff if self.graph_encoder_type == "scage" else None,
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
        return {
            "feature_source_dataset": self.feature_source_dataset,
            "geometry_encoder": self.geometry_encoder,
            "graph_encoder_type": self.graph_encoder_type,
            "graph_input": self.graph_input,
            "geometry_structure": (
                "periodic_pbc_1d_quality_gated_center_ru_energy_top4"
                if self.geom_input == "periodic_pbc"
                else (
                    "polygen_fractional_adaptive_mru_exact_translation"
                    if self.geom_input == "polygen_periodic"
                    else (
                        "screw_periodic_rt_quality_gated_center_ru"
                        if self.geom_input == "screw_periodic"
                        else (
                            "smer_trimer_center_ru" if self.geom_input == "smer_context"
                            else "star_substitution_energy_top4"
                        )
                    )
                )
            ),
            "geom_input": self.geom_input,
            "periodic_shifts": (
                [-1, 0, 1]
                if self.geom_input in {"periodic_pbc", "polygen_periodic", "screw_periodic"}
                else []
            ),
            "painn_cutoff": 10,
            "painn_max_num_neighbors": 32,
            "graph_features": "backbone_attachment_starlink_edge",
            "scage_input": (
                "chemical_fields_backbone_no_ru_index"
                if self.graph_encoder_type == "scage" else "not_applicable"
            ),
            "scage_input_schema_version": 2 if self.graph_encoder_type == "scage" else 0,
            "scage_data_schema": (
                "scage-mips-pbc-lga-v1"
                if self.graph_encoder_type == "scage" else None
            ),
            "scage_topology_schema_version": 2 if self.graph_encoder_type == "scage" else 0,
            "periodic_lga_schema_version": 1 if self.graph_encoder_type == "scage" else 0,
            "scage_descriptor_schema_version": 0,
            "mips_input_schema_version": 2 if self.graph_encoder_type == "mips" else 0,
            "mips_descriptor_schema_version": 2 if self.graph_encoder_type == "mips" else 0,
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
            "conformer_policy": (
                "etkdg100_random300_ff50_large150_pbc2d120_forcedfallback_v5"
                if self.conformer_profile == "fast"
                else "polygen_direct_cell24_top16_steric_progressive_strict_gate_v4"
            ),
            "pbc_geometry_version": (
                14 if self.geom_input == "polygen_periodic"
                else (6 if self.geom_input == "smer_context" else (
                    10 if self.geom_input == "screw_periodic"
                    else (4 if self.geom_input == "periodic_pbc" else 0)
                ))
            ),
        }

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
        return {
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
        }

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

        if completed_since_partial > 0:
            self._save_partial_feature_cache(features, failures, meta)

        elapsed = max(time.time() - start_time, 1e-9)
        processed = len(pending_smiles)
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
        feature_started = time.monotonic()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp_mol = Chem.Mol(mol)
        for atom in fp_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)

        geom_optimizer = "auto"
        geom_data = _geometry_for_mode(mol, self.geom_input, geom_optimizer)
        structure = _structure_for_encoder(
            smiles, self.graph_input, self.geom_input, geom_data,
            self.graph_encoder_type,
        )
        data = mol_to_graph_data_obj_simple(
            structure["structure_mol"], backbone_info=structure.get("backbone_info")
        )
        annotate_structure_fields(data, structure, prefix="graph")
        if self.graph_encoder_type == "scage":
            attach_periodic_lga_topology(data, structure, config=MIPSPeriodicConfig())
            geom_data = _expand_polygen_geometry_for_model(geom_data, structure)

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
        data.fp = self._compute_fingerprint(fp_mol).unsqueeze(0)
        _attach_polymer_ecfp_target(data, smiles)
        _attach_periodic_aug_views(data, smiles)

        # The periodic m-RU graph and coordinates share the exact same atom order.
        _attach_geometry_data(
            data, geom_data, smiles, self.geom_input, geom_optimizer
        )
        if self.graph_encoder_type == "scage":
            _attach_lga_geometry(data)
        elif self.graph_encoder_type == "mips":
            _attach_mips_descriptors(data, mol)

        data.feature_compute_seconds = float(time.monotonic() - feature_started)
        return data

    def _validate_pubchem_backend(self):
        test_mol = Chem.MolFromSmiles("CC")
        _pubchem_fingerprint_881(test_mol)

    def _fp_components(self):
        if self.fp_mode == "ecfp":
            return ["ECFP"]
        if self.fp_mode == "mixfp":
            return ["MACCSKeys", "PubChemFingerprints"]
        return [self.fp_mode]

    def _compute_fingerprint(self, fp_mol):
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
        raise ValueError(f"Unsupported fp_mode: {self.fp_mode}")

    def _attach_geometry(self, data, mol, smiles, geom_optimizer):
        return _attach_geometry_for_mode(
            data, mol, smiles, self.geom_input, geom_optimizer
        )

    def _build_labeled_data_list(self, feature_cache, task_csv):
        """Build self.data_list from a task CSV by looking up features in the cache."""
        df = pd.read_csv(task_csv)
        features = feature_cache["features"]
        failed_smiles = {
            str(item.get("smiles", "")).strip()
            for item in feature_cache.get("failures", [])
            if item.get("smiles") is not None
        }
        cache_misses = 0
        cached_failures = 0

        show_progress = int(os.environ.get("LOCAL_RANK", "0")) == 0
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Building {self.dataset} from feature cache",
            disable=not show_progress,
        ):
            smiles = str(row.iloc[0]).strip()
            y = float(row.iloc[1])

            if smiles in features:
                # PyG Data.__copy__ creates an independent attribute store while
                # sharing immutable cached tensors. Only the per-row target is
                # assigned below; batching creates new tensors, so duplicating
                # every graph tensor here wastes substantial RAM and startup
                # time without providing isolation that training uses.
                data = copy.copy(features[smiles])
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

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_max_smiles_token_length(self, csv_path):
        """Public helper kept for external callers (legacy name)."""
        return self._compute_max_token_length_for_file(csv_path)
