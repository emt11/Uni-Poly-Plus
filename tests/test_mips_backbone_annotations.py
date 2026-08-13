from rdkit import Chem

from src.dataset.graph_data import _graph_backbone_annotations


def test_backbone_without_ring_is_the_selected_attachment_path():
    mol = Chem.MolFromSmiles("CCCCC")

    backbone, neighbors, side_chain, star_edges = _graph_backbone_annotations(
        mol,
        original_neighbors=[0, 4],
        star_link_edge=[1, 2],
        ordered_backbone_path=[0, 1, 2, 3, 4],
    )

    assert backbone == {0, 1, 2, 3, 4}
    assert neighbors == {0, 4}
    assert side_chain == set()
    assert star_edges == {(1, 2), (2, 1)}


def test_backbone_path_through_ring_does_not_expand_to_entire_ring():
    mol = Chem.MolFromSmiles("C1CCCCC1")

    backbone, _, side_chain, _ = _graph_backbone_annotations(
        mol,
        original_neighbors=[0, 3],
        ordered_backbone_path=[0, 1, 2, 3],
    )

    assert backbone == {0, 1, 2, 3}
    assert side_chain == {4, 5}
    assert side_chain == set(range(mol.GetNumAtoms())) - backbone


def test_side_chain_ring_is_not_added_when_path_only_touches_ring_atom():
    mol = Chem.MolFromSmiles("CC1CCCCC1")

    backbone, _, side_chain, _ = _graph_backbone_annotations(
        mol,
        original_neighbors=[0, 1],
        ordered_backbone_path=[0, 1],
    )

    assert backbone == {0, 1}
    assert side_chain == set(range(mol.GetNumAtoms())) - backbone
    assert {2, 3, 4, 5, 6}.issubset(side_chain)


def test_same_attachment_and_exception_fallback_keep_existing_behavior():
    mol = Chem.MolFromSmiles("CC")

    same_backbone, same_neighbors, same_side_chain, _ = (
        _graph_backbone_annotations(mol, original_neighbors=[0, 0])
    )
    assert same_backbone == {0}
    assert same_neighbors == {0}
    assert same_side_chain == {1}

    fallback_backbone, fallback_neighbors, fallback_side_chain, _ = (
        _graph_backbone_annotations(
            mol,
            original_neighbors=[0, 99],
        )
    )
    assert fallback_backbone == {0}
    assert fallback_neighbors == {0, 99}
    assert fallback_side_chain == {1}
