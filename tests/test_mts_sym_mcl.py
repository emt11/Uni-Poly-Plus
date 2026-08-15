"""Sym-MCL v1: inverse-OR symmetric |shift|=1 hard-mask unit tests."""

import pytest
import torch

from src.modules.trimer_mcl import build_sym_mcl_mask


def _layout(c, k, q_cap=None):
    """RU(-1)|RU(0)|RU(+1) column blocks with optional trailing padding."""
    q_cap = c if q_cap is None else q_cap
    distances = torch.full((2, q_cap, k), 10.0)
    key_mask = torch.zeros((2, k), dtype=torch.bool)
    key_mask[:, : 3 * c] = True

    def place(ru, atom):
        return ru * c + atom

    return distances, key_mask, place


def test_both_observations_invisible():
    """Test 1: d_plus[a,b] > q and d_minus[b,a] > q -> both masked out."""
    c, k = 4, 12
    distances, key_mask, place = _layout(c, k)
    q = 1.0
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    a, b = 1, 2
    assert not bool(visible[0, a, place(2, b)].item())  # M_plus[a,b]
    assert not bool(visible[0, b, place(0, a)].item())  # M_minus[b,a]


def test_only_plus_observation_visible():
    """Test 2: d_plus[a,b] <= q, d_minus[b,a] > q -> both entries visible."""
    c, k = 4, 12
    distances, key_mask, place = _layout(c, k)
    q = 1.0
    a, b = 1, 2
    distances[0, a, place(2, b)] = 0.5  # RU+1 observation inside q
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    assert bool(visible[0, a, place(2, b)].item())  # M_plus[a,b] own
    assert bool(visible[0, b, place(0, a)].item())  # M_minus[b,a] inverse


def test_only_inverse_observation_visible():
    """Test 3: d_plus[a,b] > q, d_minus[b,a] <= q -> both entries visible."""
    c, k = 4, 12
    distances, key_mask, place = _layout(c, k)
    q = 1.0
    a, b = 1, 2
    distances[0, b, place(0, a)] = 0.5  # RU-1 inverse observation inside q
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    assert bool(visible[0, a, place(2, b)].item())  # M_plus[a,b] via inverse
    assert bool(visible[0, b, place(0, a)].item())  # M_minus[b,a] own


def test_shift_zero_unchanged():
    """Test 4: the RU(0) block keeps the plain hard threshold exactly."""
    c, k = 4, 12
    distances, key_mask, place = _layout(c, k)
    q = 1.0
    distances[0, 0, place(1, 0)] = 0.2
    distances[0, 3, place(1, 3)] = 0.7
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    old_zero = distances[:, :, c : 2 * c] <= q
    new_zero = visible[:, :, c : 2 * c]
    assert torch.equal(old_zero, new_zero)


def test_periodic_inverse_invariant():
    """M_plus == M_minus.T on the canonical-aligned blocks (full bucket)."""
    c, k = 4, 12  # full bucket: padded width equals 3C
    distances, key_mask, _ = _layout(c, k)
    distances[0, 1, 2 * c + 2] = 0.3
    distances[0, 2, 0 * c + 1] = 0.4
    distances[1, 0, 2 * c + 3] = 0.5
    q = 1.0
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    m_minus = visible[:, :, 0:c]
    m_plus = visible[:, :, 2 * c : 3 * c]
    assert torch.equal(m_plus, m_minus.transpose(-1, -2))


def test_ragged_padding_columns_stay_invisible():
    """Padded columns beyond 3C must never become visible."""
    c, k = 3, 12  # ragged: 3 padding columns
    distances, key_mask, _ = _layout(c, k)
    distances[:, :, 3 * c :] = 0.0  # garbage in padding columns
    q = 1.0
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    assert not bool(visible[:, :, 3 * c :].any())


def test_ragged_blocks_align_with_true_ru_boundaries():
    """Ragged buckets must align blocks at C, not at the padded width."""
    c, k = 3, 12
    distances, key_mask, place = _layout(c, k)
    q = 1.0
    a, b = 1, 2
    distances[0, a, place(2, b)] = 0.5
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    assert bool(visible[0, a, 2 * c + b].item())  # M_plus own at true offset
    assert bool(visible[0, b, 0 * c + a].item())  # M_minus inverse at true offset
    # A padded-width slice would put 2C+b at column 8 (padding), so this also
    # guards the alignment itself: the visible entry must sit inside the real
    # RU(+1) block [2C, 3C).
    assert 2 * c + b < 3 * c


def test_padded_query_rows_do_not_leak_into_valid_rows():
    """Padded query rows (>= C) must stay invisible and not pollute others."""
    c, k, q_cap = 3, 12, 4
    distances, key_mask, _ = _layout(c, k, q_cap=q_cap)
    # Distances for padded query row 3 are garbage-close; they must not make
    # any valid column visible through the transpose path.
    distances[0, 3, :] = 0.0
    q = 1.0
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
    assert not bool(visible[0, 3, : 3 * c].any())


def test_key_mask_none_treats_all_columns_real():
    """key_mask=None keeps the plain full-width semantics (3C == K)."""
    c, k = 4, 12
    distances, _, _ = _layout(c, k)
    q = 1.0
    visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), None)
    assert visible.shape == (2, 4, 12)
    assert bool((visible.sum() == (distances <= q).sum()).item())


def test_sym_mask_never_reduces_visibility():
    """The inverse-OR union can only add visible pairs, never remove them."""
    torch.manual_seed(0)
    for c, k in ((4, 12), (3, 12)):
        distances = torch.rand(2, c, k) * 5.0
        key_mask = torch.zeros((2, k), dtype=torch.bool)
        key_mask[:, : 3 * c] = True
        q = 2.0
        visible = build_sym_mcl_mask(distances, q, torch.tensor([c, c]), key_mask)
        old = (distances <= q) & key_mask.unsqueeze(1)
        assert bool((visible >= old).all())
        assert visible.shape == (2, c, k)
        assert not bool(torch.isnan(distances).any())


def _reference_sym_mask(distances, threshold, ru, canonical_key,
                        canonical_query, key_mask, query_mask):
    """Independent per-pair reference built from raw Trimer metadata.

    Uses the physical semantics directly: a |shift|=1 pair is visible when
    either its own observation or the inverse observation (query row with the
    key's canonical identity, key column with the query's canonical identity
    in the opposite RU) lies inside the threshold.  Deliberately written as
    plain loops so it shares no indexing logic with the production gather.
    """
    batch, queries, keys = distances.shape
    out = torch.zeros(batch, queries, keys, dtype=torch.bool)
    for b in range(batch):
        for qi in range(queries):
            if not bool(query_mask[b, qi].item()):
                continue
            for kj in range(keys):
                if not bool(key_mask[b, kj].item()):
                    continue
                ru_kj = int(ru[b, kj].item())
                own = bool((distances[b, qi, kj] <= threshold).item())
                if ru_kj == 0:
                    out[b, qi, kj] = own
                    continue
                inverse = False
                for jj in range(keys):
                    if (
                        bool(key_mask[b, jj].item())
                        and int(ru[b, jj].item()) == -ru_kj
                        and int(canonical_key[b, jj].item())
                        == int(canonical_query[b, qi].item())
                    ):
                        for qq in range(queries):
                            if (
                                bool(query_mask[b, qq].item())
                                and int(canonical_query[b, qq].item())
                                == int(canonical_key[b, kj].item())
                                and bool((distances[b, qq, jj] <= threshold).item())
                            ):
                                inverse = True
                out[b, qi, kj] = own or inverse
    return out


def test_real_field_independent_reference_ragged():
    """Production mask equals a metadata-driven reference on real Trimer data.

    Covers the ragged case C < Q (padded query rows) and padded key columns
    using the real ``trimer_ru_offset``, canonical atom order and query/key
    valid masks extracted from a collated real batch.
    """
    from src.dataset.dataloader import custom_collate
    from src.dataset.dataset import _compute_smiles_features_from_config
    from src.dataset.mts_star_rbf_v2 import build_star_rbf_v2_sample

    def graph_data(smiles="*CCCCCCC*"):
        data = _compute_smiles_features_from_config(
            smiles,
            "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
            32, "star_linking", "repeat_unit", "disabled",
            graph_encoder_type="scage",
            mips_core="paper_corrected",
            mips_max_hops=2,
            mips_use_descriptors=True,
            spatial_mode="trimer_scage",
            graph_geometry_mode="trimer_scage_mcl",
            trimer_num_candidates=4,
            trimer_max_heavy_atoms=384,
        )
        record = build_star_rbf_v2_sample(b"k" * 32, data, data)
        data.mts_star_v2_relation_row = torch.tensor(
            [item["row"] for item in record["relations"]], dtype=torch.long
        )
        data.mts_star_v2_relation_pair_index = torch.tensor(
            [item["pair_index"] for item in record["relations"]], dtype=torch.long
        )
        data.mts_star_v2_relation_spd = torch.tensor(
            [item["spd"] for item in record["relations"]], dtype=torch.long
        )
        data.mts_star_v2_pair_observation_distances = torch.tensor(
            [item["distances"] for item in record["pairs"]], dtype=torch.float
        )
        data.mts_star_v2_pair_observation_count = torch.tensor(
            [item["observation_count"] for item in record["pairs"]], dtype=torch.long
        )
        data.mts_star_v2_pair_valid = torch.tensor(
            [item["valid"] for item in record["pairs"]], dtype=torch.bool
        )
        data.mts_star_v2_pair_geometry_source = torch.tensor(
            [item["geometry_source"] for item in record["pairs"]], dtype=torch.long
        )
        data.mts_star_v2_sidecar_artifact = "a" * 64
        data.mts_star_v2_model_semantic_hash = "b" * 64
        data.mts_star_v2_rbf_upper = 6.0
        data.y = torch.zeros(1)
        return data

    batch = custom_collate([graph_data("*CCCCCCC*"), graph_data("*CCO*")])
    bucket_sizes = batch.mcl_bucket_size.long()
    keys_table = batch.mcl_key_index_padded.long()
    queries_table = batch.mcl_query_index_padded.long()
    ru_all = batch.trimer_ru_offset.long()
    canonical_all = batch.trimer_base_ru_atom_index.long()
    positions = batch.trimer_pos.float()
    assert int(bucket_sizes[1]) > 3 * 3  # second graph is ragged C < Q

    checked = 0
    for graph_id in (0, 1):
        capacity = int(bucket_sizes[graph_id].item())
        if capacity <= 0:
            continue
        query_capacity = capacity // 3
        keys = keys_table[graph_id, :capacity]
        queries = queries_table[graph_id, :query_capacity]
        key_mask = keys >= 0
        query_mask = queries >= 0
        safe_keys = keys.clamp_min(0)
        safe_queries = queries.clamp_min(0)
        central = int(query_mask.sum().item())
        distances = torch.cdist(
            positions[safe_queries], positions[safe_keys]
        ).unsqueeze(0)  # [1, Q, K]
        ru = ru_all[safe_keys].unsqueeze(0)
        canonical_key = canonical_all[safe_keys].unsqueeze(0)
        canonical_query = canonical_all[safe_queries].unsqueeze(0)
        assert int(key_mask.sum().item()) == 3 * central
        for scale_index in (0, 1):
            threshold = float(batch.trimer_mcl_thresholds[graph_id, scale_index])
            production = build_sym_mcl_mask(
                distances, threshold, torch.tensor([central]),
                key_mask.unsqueeze(0),
            )[0]
            reference = _reference_sym_mask(
                distances, threshold, ru, canonical_key, canonical_query,
                key_mask.unsqueeze(0), query_mask.unsqueeze(0),
            )[0]
            assert torch.equal(production, reference)
            old = (distances[0] <= threshold) & key_mask & query_mask.unsqueeze(-1)
            assert bool((production >= old).all())
            m_plus = production[:central, 2 * central : 3 * central]
            m_minus = production[:central, 0:central]
            assert torch.equal(m_plus, m_minus.transpose(-1, -2))
            zero_old = old[:, central : 2 * central]
            zero_new = production[:, central : 2 * central]
            assert torch.equal(zero_old, zero_new)
            if capacity > 3 * central:
                assert not bool(production[:, 3 * central :].any())
            if query_capacity > central:
                assert not bool(production[central:, : 3 * central].any())
            checked += 1
    assert checked >= 2
