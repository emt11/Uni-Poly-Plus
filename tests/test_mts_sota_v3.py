import copy

import torch
from torch import nn

from scripts.pretrain import _joint_canonical_mask
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.dataset import _compute_smiles_features_from_config
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from src.modules.uni_encoder import UniEncoderAttention


def graph_data(smiles):
    data = _compute_smiles_features_from_config(
        smiles,
        "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        32, "star_linking", "repeat_unit", "disabled",
        graph_encoder_type="scage", mips_core="paper_corrected",
        mips_max_hops=2, mips_use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        trimer_num_candidates=4, trimer_max_heavy_atoms=384,
    )
    data.y = torch.zeros(1)
    return data


def _batch():
    left = graph_data("*CCO*")
    right = graph_data("*CCCCCCC*")
    left.mts_sample_hash64 = torch.tensor(11)
    right.mts_sample_hash64 = torch.tensor(29)
    return mips_trimer_collate([left, right])


def test_vectorized_mask_is_stateless_and_copy_consistent():
    batch = _batch()
    first = _joint_canonical_mask(batch, seed=42, stream_step=7, mask_ratio=0.30)
    second = _joint_canonical_mask(batch, seed=42, stream_step=7, mask_ratio=0.30)
    changed = _joint_canonical_mask(batch, seed=42, stream_step=8, mask_ratio=0.30)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    canonical = batch.canonical_ru_atom_index
    for identity in torch.unique(canonical):
        assert first[canonical == identity].unique().numel() == 1
    for graph_id in range(2):
        assert bool(first[batch.batch == graph_id].any())


def test_mcl_rbf_zero_initialization_matches_current_mcl():
    torch.manual_seed(3)
    current = MIPSLocalGraphEncoder(graph_geometry_mode="current_mcl").eval()
    rbf = MIPSLocalGraphEncoder(graph_geometry_mode="mcl_rbf").eval()
    state = rbf.state_dict()
    for name, value in current.state_dict().items():
        if name in state and state[name].shape == value.shape:
            state[name] = value
    rbf.load_state_dict(state, strict=True)
    current.trimer_mcl.geometry_gate.data.fill_(0.2)
    rbf.trimer_mcl.geometry_gate.data.fill_(0.2)
    batch = _batch()
    with torch.no_grad():
        current_graph, current_nodes = current(batch)
        rbf_graph, rbf_nodes = rbf(batch)
    assert torch.allclose(current_nodes, rbf_nodes, atol=1e-5, rtol=1e-5)
    assert torch.allclose(current_graph, rbf_graph, atol=1e-5, rtol=1e-5)


def test_bucketed_mcl_matches_legacy_per_graph_reference():
    batch = _batch()
    legacy = copy.deepcopy(batch)
    for name in (
        "mcl_bucket_size", "mcl_key_index_padded",
        "mcl_query_index_padded", "mcl_query_local_index_padded",
        "mcl_query_canonical_index_padded",
    ):
        delattr(legacy, name)
    encoder = MIPSLocalGraphEncoder(graph_geometry_mode="current_mcl").eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.3)
    with torch.no_grad():
        packed_graph, packed_nodes = encoder(batch)
        legacy_graph, legacy_nodes = encoder(legacy)
    assert torch.allclose(packed_nodes, legacy_nodes, atol=1e-5, rtol=1e-5)
    assert torch.allclose(packed_graph, legacy_graph, atol=1e-5, rtol=1e-5)


def test_padded_mcl_canonical_indices_use_preincrement_graph_offset():
    batch = _batch()
    rows = batch.mcl_query_canonical_index_padded.long()
    valid = rows >= 0
    assert bool(valid.any())
    assert int(rows[valid].max()) < int(batch.canonical_graph_index.numel())
    for graph_id in range(rows.size(0)):
        graph_targets = rows[graph_id][rows[graph_id] >= 0]
        if graph_targets.numel():
            assert torch.equal(
                batch.canonical_graph_index[graph_targets],
                torch.full_like(graph_targets, graph_id),
            )


def test_zero_gated_residual_is_exact_graph_anchor():
    model = UniEncoderAttention.__new__(UniEncoderAttention)
    nn.Module.__init__(model)
    model.fusion_type = "zero_gated_residual"
    model.modality_list = ["graph", "smiles", "fp"]
    model.residual_modality_gates = nn.ParameterDict({
        "smiles": nn.Parameter(torch.zeros(())),
        "fp": nn.Parameter(torch.zeros(())),
    })
    embeddings = torch.randn(4, 3, 16)
    availability = torch.ones(4, 3, dtype=torch.bool)
    fused, weights = model.fuse_embeddings(embeddings, availability)
    assert torch.equal(fused, embeddings[:, 0])
    assert torch.equal(weights[:, 0], torch.ones(4))
    assert torch.equal(weights[:, 1:], torch.zeros(4, 2))


def test_disabled_mcl_retains_star_topology_path():
    batch = _batch()
    encoder = MIPSLocalGraphEncoder(graph_geometry_mode="disabled").eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(1.0)
    with torch.no_grad():
        observed, nodes = encoder(batch)
        expected, expected_nodes = encoder._forward_impl(
            batch, use_star=True, use_geometry=False, use_md=True
        )
    assert torch.equal(nodes, expected_nodes)
    assert torch.equal(observed, expected)
