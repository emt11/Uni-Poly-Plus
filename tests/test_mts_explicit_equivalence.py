"""Independent and production explicit/canonical relation checks."""

from collections import Counter
import copy

import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.explicit_k_ru import build_explicit_k_ru_topology
from src.dataset.lmdb_cache import _expand_explicit_trimer_mapping
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from scripts.pretrain import _joint_canonical_mask, _joint_masked_atom_terms
from tests.reference_mts_explicit import build_corrected_explicit_k_ru_reference


def test_reference_is_independent_and_matches_lifted_relation_multiset():
    topology = build_canonical_periodic_topology("*CCO*")
    reference = build_corrected_explicit_k_ru_reference(topology, 7)
    n = int(topology.mips_x.size(0))
    relation = topology.lga_edge_index.long()
    shifts = topology.lga_source_image_shift.long()
    expected = []
    observed = []
    for copy_index in range(7):
        for row in range(relation.size(1)):
            source = int(relation[0, row]) + (
                (copy_index + int(shifts[row])) % 7
            ) * n
            target = int(relation[1, row]) + copy_index * n
            expected.append(
                (source, target, int(shifts[row]), int(topology.lga_spd[row]))
            )
    for row in range(reference.lga_edge_index.size(1)):
        observed.append(
            (
                int(reference.lga_edge_index[0, row]),
                int(reference.lga_edge_index[1, row]),
                int(reference.lga_source_image_shift[row]),
                int(reference.lga_spd[row]),
            )
        )
    assert sorted(expected) == sorted(observed)
    assert reference.lga_edge_index.size(1) == 7 * relation.size(1)
    assert torch.equal(
        reference.mips_x.view(7, n, -1),
        topology.mips_x.unsqueeze(0).expand(7, -1, -1),
    )


def _canonical_relation_counter(data):
    return Counter(
        (
            int(data.lga_edge_index[0, row]),
            int(data.lga_edge_index[1, row]),
            int(data.lga_source_image_shift[row]),
            int(data.lga_spd[row]),
            bool(data.lga_star_edge_mask[row]),
        )
        for row in range(int(data.lga_spd.numel()))
    )


def _explicit_relation_counter(data, canonical_count, repeat_units):
    values = Counter()
    for row in range(int(data.lga_spd.numel())):
        source = int(data.lga_edge_index[0, row])
        target = int(data.lga_edge_index[1, row])
        source_copy = source // canonical_count
        target_copy = target // canonical_count
        shift = source_copy - target_copy
        if shift > repeat_units // 2:
            shift -= repeat_units
        elif shift < -(repeat_units // 2):
            shift += repeat_units
        values[
            source % canonical_count,
            target % canonical_count,
            shift,
            int(data.lga_spd[row]),
            bool(data.lga_star_edge_mask[row]),
        ] += 1
    return values


def test_production_explicit_builder_is_minimal_and_relation_equivalent():
    cases = {
        "*CCO*": (3, 8),
        "*C(*)C(=O)OCC(C)(C)C": (7, 6),
        "*CC*": (4, 7),
    }
    for smiles, (expected_k, expected_boundary) in cases.items():
        canonical = build_canonical_periodic_topology(smiles)
        explicit = build_explicit_k_ru_topology(smiles)
        n = int(canonical.mips_x.size(0))
        k = int(explicit.mips_repeat_units)
        assert (k, int(explicit.mips_boundary_distance)) == (
            expected_k, expected_boundary
        )
        expected = _canonical_relation_counter(canonical)
        observed = _explicit_relation_counter(explicit, n, k)
        assert observed == Counter({key: k * count for key, count in expected.items()})
        assert not bool(
            (
                explicit.lga_star_edge_mask
                & (explicit.lga_edge_index[0] == explicit.lga_edge_index[1])
            ).any()
        )


def test_production_explicit_six_layer_topology_equivalence():
    for smiles in ("*CCCCCCC*", "*CCCC*", "*CCO*"):
        canonical = build_canonical_periodic_topology(smiles)
        explicit = build_explicit_k_ru_topology(smiles)
        canonical_batch = mips_trimer_collate([canonical])
        explicit_batch = mips_trimer_collate([explicit])
        torch.manual_seed(123)
        model = MIPSLocalGraphEncoder().eval()
        with torch.no_grad():
            canonical_graph, canonical_nodes = model._forward_impl(
                canonical_batch, use_star=False, use_geometry=False, use_md=False
            )
            explicit_graph, explicit_nodes = model._forward_impl(
                explicit_batch, use_star=False, use_geometry=False, use_md=False
            )
        pooled = torch.stack([
            explicit_nodes[explicit.canonical_ru_atom_index == atom].mean(0)
            for atom in range(int(canonical.num_nodes))
        ])
        assert torch.allclose(canonical_nodes, pooled, atol=1e-5, rtol=1e-5)
        assert torch.allclose(
            canonical_graph, explicit_graph, atol=1e-5, rtol=1e-5
        )


def test_explicit_mapping_expands_by_canonical_identity_not_copy_id():
    explicit = build_explicit_k_ru_topology("*CCO*")
    canonical_mapping = torch.tensor([7, 8, 9], dtype=torch.long)
    expanded = _expand_explicit_trimer_mapping(
        explicit,
        "mips_to_trimer_central_index",
        canonical_mapping,
        sample_key=b"a" * 32,
    )
    assert torch.equal(expanded, canonical_mapping.repeat(explicit.mips_repeat_units))
    for atom in range(3):
        assert torch.unique(expanded[explicit.canonical_ru_atom_index == atom]).tolist() == [7 + atom]


def test_explicit_joint_mask_is_copy_tied_and_one_target_per_atom():
    explicit = build_explicit_k_ru_topology("*CCO*")
    batch = mips_trimer_collate([explicit])
    mask = _joint_canonical_mask(batch, seed=42, stream_step=17, mask_ratio=0.30)
    for atom in range(3):
        selected = mask[explicit.canonical_ru_atom_index == atom]
        assert bool((selected == selected[0]).all())
    head = torch.nn.Linear(512, 119)
    terms = _joint_masked_atom_terms(
        batch, torch.randn(explicit.num_nodes, 512), head, mask
    )
    assert terms[2] == int(mask[batch.canonical_first_node_index].sum())


def _attach_same_canonical_trimer(canonical, explicit):
    attach_finite_trimer_mcl(canonical, "*CCO*", num_candidates=1)
    for name in canonical.keys():
        if name.startswith("trimer_") or name.startswith("star_3d_"):
            explicit[name] = copy.deepcopy(canonical[name])
    explicit.mips_to_trimer_central_index = (
        canonical.mips_to_trimer_central_index.long()[
            explicit.canonical_ru_atom_index.long()
        ]
    )


def test_explicit_and_canonical_geometry_forward_are_equivalent():
    canonical = build_canonical_periodic_topology("*CCO*")
    explicit = build_explicit_k_ru_topology("*CCO*")
    _attach_same_canonical_trimer(canonical, explicit)
    canonical_batch = mips_trimer_collate([canonical])
    explicit_batch = mips_trimer_collate([explicit])
    torch.manual_seed(29)
    model = MIPSLocalGraphEncoder().eval()
    with torch.no_grad():
        model.star_distance_bias.projection.weight.fill_(0.05)
        model.trimer_mcl.geometry_gate.fill_(0.20)
        canonical_graph, canonical_nodes = model._forward_impl(
            canonical_batch, use_star=True, use_geometry=True, use_md=False
        )
        explicit_graph, explicit_nodes = model._forward_impl(
            explicit_batch, use_star=True, use_geometry=True, use_md=False
        )
    pooled = torch.stack([
        explicit_nodes[explicit.canonical_ru_atom_index == atom].mean(0)
        for atom in range(int(canonical.num_nodes))
    ])
    assert torch.allclose(canonical_nodes, pooled, atol=1e-5, rtol=1e-5)
    assert torch.allclose(canonical_graph, explicit_graph, atol=1e-5, rtol=1e-5)


def test_both_representations_complete_joint_pretrain_backward():
    batches = []
    for representation in ("canonical", "explicit"):
        canonical = build_canonical_periodic_topology("*CCO*")
        if representation == "canonical":
            attach_finite_trimer_mcl(canonical, "*CCO*", num_candidates=1)
            sample = canonical
        else:
            sample = build_explicit_k_ru_topology("*CCO*")
            _attach_same_canonical_trimer(canonical, sample)
        batches.append(mips_trimer_collate([sample]))

    torch.manual_seed(31)
    model = MIPSLocalGraphEncoder().eval()
    total = torch.zeros(())
    for batch in batches:
        mask = _joint_canonical_mask(
            batch, seed=42, stream_step=3, mask_ratio=0.30
        )
        graph, nodes, aux = model.forward_joint_pretrain(batch, mask)
        assert torch.isfinite(graph).all()
        assert torch.isfinite(nodes).all()
        assert torch.isfinite(aux["final_trimer_states"]).all()
        total = total + graph.square().mean() + nodes.square().mean()
    total.backward()
    assert model.atom_embedding.projection.weight.grad is not None
    assert torch.isfinite(model.atom_embedding.projection.weight.grad).all()


def test_explicit_atom_cap_is_a_supported_unavailable_record():
    explicit = build_explicit_k_ru_topology("*CCO*", max_model_atoms=5)
    assert not bool(explicit.graph_available)
    assert not bool(explicit.mips_condition_valid)


def test_single_ru_terminal_star_remains_a_polymer_link():
    explicit = build_explicit_k_ru_topology("*CCCCCCC*")
    assert int(explicit.mips_repeat_units) == 1
    assert bool(explicit.lga_star_edge_mask.any())
    assert torch.equal(
        explicit.lga_star_edge_mask.bool(), explicit.polymer_link_mask.bool()
    )
