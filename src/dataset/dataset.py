import copy
import os
import torch
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Dataset
from .geom_data import mol2coords
from .graph_data import annotate_structure_fields, build_structure_for_input, mol_to_graph_data_obj_simple
from transformers import AutoTokenizer
from rdkit.Chem import rdFingerprintGenerator


class UniDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        smiles_model_name,
        geometry_encoder='painn',
        graph_input='repeat_unit',
        use_feature_cache=True,
        feature_source_dataset=None,
        rebuild_feature_cache=False,
        max_smiles_length=None,
        max_smiles_length_cap=256,
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
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)

        geometry_cache_dirs = {
            'schnet': 'Schnet',
            'painn': 'PAINN',
        }
        self.geometry_encoder = geometry_encoder.lower()
        if self.geometry_encoder not in geometry_cache_dirs:
            raise ValueError("geometry_encoder must be 'painn' or 'schnet'")

        processed_dir = os.path.join(self.root, 'processed', geometry_cache_dirs[self.geometry_encoder])
        os.makedirs(processed_dir, exist_ok=True)

        graph_tag = 'gin-backbone'
        if self.graph_input == 'star_linking':
            graph_tag = 'gin-starlink-backbone'

        self.use_feature_cache = bool(use_feature_cache)

        if self.use_feature_cache:
            self._init_with_feature_cache(
                processed_dir=processed_dir,
                graph_tag=graph_tag,
                max_smiles_length=max_smiles_length,
                max_smiles_length_cap=max_smiles_length_cap,
                feature_source_dataset=feature_source_dataset,
                rebuild_feature_cache=rebuild_feature_cache,
            )
        else:
            self._init_legacy(processed_dir=processed_dir, graph_tag=graph_tag)

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
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
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

            structure = build_structure_for_input(smiles, self.graph_input)
            data = mol_to_graph_data_obj_simple(structure["structure_mol"])
            annotate_structure_fields(data, structure, prefix="graph")

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

            data.fp = torch.tensor(mfpgen.GetFingerprint(fp_mol), dtype=torch.float).unsqueeze(0)
            try:
                geom_optimizer = "auto"
                geom_data = mol2coords(mol, process_stars=True, optimizer=geom_optimizer)
                data.pos = geom_data.pos
                data.z = geom_data.z
                data.pos_confs = geom_data.pos_confs
                data.geom_smiles = smiles
                data.geom_input = getattr(geom_data, "geom_input", "star_substitution")
                data.geom_optimizer = geom_optimizer
                data.geom_optimizer_used = getattr(geom_data, "geom_optimizer_used", geom_optimizer)
                data.geom_build_ok = bool(getattr(geom_data, "geom_build_ok", True))
                data.geom_failed_reason = getattr(geom_data, "geom_failed_reason", "")
                data.geom_num_confs = int(getattr(geom_data, "geom_num_confs", data.pos_confs.size(0)))
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
            f"feature_cache_{self.feature_source_dataset}_{graph_tag}_tok{self.max_smiles_length}.pt",
        )

        # Build or load feature cache
        if rebuild_feature_cache or not os.path.exists(self.feature_cache_path):
            print(f"building feature cache from {self.feature_source_dataset} ...")
            feature_cache = self._build_feature_cache()
            torch.save(feature_cache, self.feature_cache_path)
            print(f"saved feature cache to {self.feature_cache_path}")
        else:
            feature_cache = torch.load(self.feature_cache_path, weights_only=False)
            print(
                f"loaded feature cache {self.feature_cache_path} "
                f"with {len(feature_cache['features'])} entries"
            )

        # Build labeled data list from the task CSV
        task_csv = f"{self.root}/raw/{self.dataset}.csv"
        self._build_labeled_data_list(feature_cache, task_csv)

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

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        features = {}

        for smiles in tqdm(unique_smiles, desc="Building feature cache"):
            try:
                data = self._compute_smiles_features(smiles, mfpgen)
                features[smiles] = data
            except Exception as e:
                print(f"Failed to compute features for {smiles}: {e}")
                continue

        cache = {
            "meta": {
                "feature_source_dataset": self.feature_source_dataset,
                "geometry_encoder": self.geometry_encoder,
                "graph_input": self.graph_input,
                "geometry_structure": "star_substitution_multi_conformer",
                "graph_features": "backbone_attachment_starlink_edge",
                "max_smiles_length": self.max_smiles_length,
                "tokenizer": str(type(self.smiles_tokenizer).__name__),
            },
            "features": features,
        }
        print(f"Feature cache built: {len(features)} entries")
        return cache

    def _compute_smiles_features(self, smiles, mfpgen):
        """Compute all structural features for a single SMILES. Does NOT set data.y."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp_mol = Chem.Mol(mol)
        for atom in fp_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)

        structure = build_structure_for_input(smiles, self.graph_input)
        data = mol_to_graph_data_obj_simple(structure["structure_mol"])
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

        # Morgan fingerprint
        data.fp = torch.tensor(mfpgen.GetFingerprint(fp_mol), dtype=torch.float).unsqueeze(0)

        # 3D geometry stays on the original repeat-unit structure. Star-linking is
        # a topology-only graph construction and is not used for SchNet/PaiNN.
        geom_optimizer = "auto"
        geom_data = mol2coords(mol, process_stars=True, optimizer=geom_optimizer)
        data.pos = geom_data.pos
        data.z = geom_data.z
        data.pos_confs = geom_data.pos_confs
        data.geom_smiles = smiles
        data.geom_input = getattr(geom_data, "geom_input", "star_substitution")
        data.geom_optimizer = geom_optimizer
        data.geom_optimizer_used = getattr(geom_data, "geom_optimizer_used", geom_optimizer)
        data.geom_build_ok = bool(getattr(geom_data, "geom_build_ok", True))
        data.geom_failed_reason = getattr(geom_data, "geom_failed_reason", "")
        data.geom_num_confs = int(getattr(geom_data, "geom_num_confs", data.pos_confs.size(0)))

        return data

    def _build_labeled_data_list(self, feature_cache, task_csv):
        """Build self.data_list from a task CSV by looking up features in the cache."""
        df = pd.read_csv(task_csv)
        features = feature_cache["features"]
        cache_misses = 0

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Building {self.dataset} from feature cache"):
            smiles = str(row.iloc[0]).strip()
            y = float(row.iloc[1])

            if smiles in features:
                data = copy.deepcopy(features[smiles])
            else:
                # Cache miss — compute on the fly and write back
                print(f"[feature_cache] cache miss, computed: {smiles}")
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                try:
                    data = self._compute_smiles_features(smiles, mfpgen)
                    features[smiles] = data  # write back to in-memory cache
                    data = copy.deepcopy(data)
                except Exception as e:
                    print(f"Failed to compute features for {smiles}: {e}, skipping")
                    cache_misses += 1
                    continue

            data.y = torch.tensor([y], dtype=torch.float)
            self.data_list.append(data)

        print(
            f"Built {self.dataset}: {len(self.data_list)} samples from feature cache"
        )
        if cache_misses > 0:
            print(
                f"[feature_cache] {cache_misses} cache misses could not be resolved"
            )

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
