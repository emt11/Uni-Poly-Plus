"""Two-layer SCAGE-style MCL residual for an open finite Trimer."""

from __future__ import annotations

import torch
import torch.nn as nn
import os
from torch_scatter import scatter

from src.dataset.mips_cache_validation import trimer_can_enter_mcl


class _TrimerMCLLayer(nn.Module):
    """Central-RU cross-attention over full-Trimer tokens at two scales."""

    def __init__(
        self, dim=512, num_heads=8, dropout=0.10,
        use_distance_bias=False, num_rbf=64, rbf_upper=8.0,
    ):
        super().__init__()
        if int(dim) % int(num_heads):
            raise ValueError("MCL dimension must divide num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        # Q/K/V are shared by the 20% and 50% distance scales.
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.multiscale_projection = nn.Linear(
            2 * self.dim, self.dim, bias=False
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.residual_dropout = nn.Dropout(float(dropout))
        self.attention_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.dim, self.dim),
        )
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.use_distance_bias = bool(use_distance_bias)
        # Register optional MCL-v2 tensors only for the explicit RBF mode.
        # The legacy/current hard-mask path must remain exactly isomorphic to
        # the completed joint checkpoint; unconditional registration would
        # incorrectly report missing parameters during G0 transfer.
        if self.use_distance_bias:
            centers = torch.linspace(0.0, float(rbf_upper), int(num_rbf))
            self.register_buffer("distance_centers", centers)
            spacing = float(rbf_upper) / max(1, int(num_rbf) - 1)
            self.distance_gamma = 0.5 / max(spacing * spacing, 1e-12)
            self.distance_projection = nn.Linear(
                int(num_rbf), self.num_heads, bias=False
            )
            # MCL-v2 is an exact functional no-op at construction time.  This
            # is essential for transferring the immutable MTS-v2 checkpoint.
            nn.init.zeros_(self.distance_projection.weight)
        else:
            self.distance_centers = None
            self.distance_gamma = None
            self.distance_projection = None
        self.debug_attention = os.environ.get("MIPS_DEBUG_ATTENTION", "0") == "1"
        self.last_attention = None

    def _attend(self, query, key, value, visible):
        logits = torch.einsum("qhd,khd->qkh", query, key) * self.scale
        logits = logits.float().masked_fill(
            ~visible.unsqueeze(-1), float("-inf")
        )
        weights = torch.softmax(logits, dim=1).to(value.dtype)
        output = torch.einsum(
            "qkh,khd->qhd", self.attention_dropout(weights), value
        )
        return output.reshape(query.size(0), self.dim), weights

    def forward(self, central_state, memory, distances, thresholds):
        query = self.query(central_state).view(
            central_state.size(0), self.num_heads, self.head_dim
        )
        key = self.key(memory).view(
            memory.size(0), self.num_heads, self.head_dim
        )
        value = self.value(memory).view(
            memory.size(0), self.num_heads, self.head_dim
        )
        outputs = []
        self.last_attention = [] if self.debug_attention else None
        for threshold in thresholds:
            visible = distances <= threshold
            if not bool(visible.any(dim=1).all()):
                raise ValueError("MCL query has no visible Trimer key")
            output, weights = self._attend(query, key, value, visible)
            outputs.append(output)
            if self.debug_attention:
                self.last_attention.append(weights.detach())
        update = self.multiscale_projection(torch.cat(outputs, dim=-1))
        state = self.attention_norm(
            central_state + self.residual_dropout(update)
        )
        return self.ffn_norm(
            state + self.residual_dropout(self.ffn(state))
        )

    def _project_distance(self, distances):
        if not self.use_distance_bias:
            return distances.new_zeros((*distances.shape, self.num_heads))
        flat = distances.float().reshape(-1)
        outputs = []
        # Avoid materialising [B,Q,K,64] for large 384-atom buckets.
        for start in range(0, int(flat.numel()), 65536):
            values = flat[start:start + 65536].unsqueeze(-1)
            rbf = torch.exp(
                -self.distance_gamma
                * (values - self.distance_centers.float()) ** 2
            )
            outputs.append(self.distance_projection(
                rbf.to(self.distance_projection.weight.dtype)
            ).to(distances.dtype))
        return torch.cat(outputs, dim=0).reshape(
            *distances.shape, self.num_heads
        )

    def forward_batched(
        self, central_state, memory, distances, thresholds,
        key_mask=None, query_mask=None, precomputed_visible=None,
    ):
        """Padded batched MCL with exact masks for ragged Trimer records.

        ``precomputed_visible`` (A4 random-mask mode) is an optional fixed
        [B, Q, K] bool set that replaces the distance-threshold visibility so
        the only difference vs A3 is the key identity, never the sparsity.
        """
        query = self.query(central_state).view(
            central_state.size(0), central_state.size(1),
            self.num_heads, self.head_dim
        )
        key = self.key(memory).view(
            memory.size(0), memory.size(1), self.num_heads, self.head_dim
        )
        value = self.value(memory).view(
            memory.size(0), memory.size(1), self.num_heads, self.head_dim
        )
        outputs = []
        if key_mask is None:
            key_mask = torch.ones(
                memory.shape[:2], dtype=torch.bool, device=memory.device
            )
        if query_mask is None:
            query_mask = torch.ones(
                central_state.shape[:2], dtype=torch.bool,
                device=central_state.device,
            )
        distance_bias = self._project_distance(distances)
        if self.debug_attention:
            self.last_attention = []
        num_scales = (
            2 if precomputed_visible is not None
            else thresholds.size(1)
        )
        for scale_index in range(num_scales):
            if precomputed_visible is not None:
                visible = precomputed_visible[:, :, :, scale_index].bool()
                visible = visible & key_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
            else:
                visible = (
                    distances <= thresholds[:, scale_index].view(-1, 1, 1)
                ) & key_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
            # Padded query rows need one finite logit to keep softmax finite;
            # their output is zeroed immediately afterwards.
            if visible.size(-1):
                visible[:, :, 0] |= ~query_mask
            logits = torch.einsum("bqhd,bkhd->bqkh", query, key) * self.scale
            logits = logits + distance_bias
            logits = logits.float().masked_fill(
                ~visible.unsqueeze(-1), float("-inf")
            )
            weights = torch.softmax(logits, dim=2).to(value.dtype)
            if self.debug_attention:
                self.last_attention.append(weights.detach())
            output = torch.einsum(
                "bqkh,bkhd->bqhd",
                self.attention_dropout(weights), value,
            ).reshape(central_state.size(0), central_state.size(1), self.dim)
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
            outputs.append(output)
        update = self.multiscale_projection(
            torch.cat(outputs, dim=-1)
        )
        state = self.attention_norm(
            central_state + self.residual_dropout(update)
        )
        result = self.ffn_norm(
            state + self.residual_dropout(self.ffn(state))
        )
        return result * query_mask.unsqueeze(-1).to(result.dtype)


class TrimerSCAGEMCLResidual(nn.Module):
    """Map a two-layer central-RU geometry update back to every O8 copy.

    Coordinates only determine hard visibility masks.  No coordinate,
    continuous distance, direction, or RU-offset embedding enters the MCL.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        percentiles=(0.20, 0.50),
        dropout: float = 0.10,
        use_distance_bias: bool = False,
        coordinate_shuffle: bool = False,
        mask_mode: str = "real",
    ):
        super().__init__()
        if tuple(float(value) for value in percentiles) != (0.20, 0.50):
            raise ValueError("MIPS-Trimer-SCAGE requires percentiles 0.20/0.50")
        if mask_mode not in ("real", "count_matched_random"):
            raise ValueError(
                f"unsupported MCL mask mode: {mask_mode!r}"
            )
        self.dim = int(dim)
        self.percentiles = (0.20, 0.50)
        self.mask_mode = str(mask_mode)
        self.input_norm = nn.LayerNorm(self.dim)
        self.layers = nn.ModuleList(
            _TrimerMCLLayer(
                dim, num_heads, dropout,
                use_distance_bias=use_distance_bias,
                num_rbf=64,
                rbf_upper=8.0,
            ) for _ in range(2)
        )
        self.use_distance_bias = bool(use_distance_bias)
        self.coordinate_shuffle = bool(coordinate_shuffle)
        self.geometry_gate = nn.Parameter(torch.zeros(self.dim))
        self.debug_attention = os.environ.get("MIPS_DEBUG_ATTENTION", "0") == "1"
        self.last_thresholds = [] if self.debug_attention else None

    def _forward_padded(
        self, topology_nodes, data, canonical, tokens,
        final_trimer_states, delta_canonical, valid,
    ):
        """Execute five fixed ragged buckets without a per-graph CUDA loop."""
        bucket_sizes = data.mcl_bucket_size.long()
        key_table = data.mcl_key_index_padded.long()
        query_table = data.mcl_query_index_padded.long()
        query_local_table = data.mcl_query_local_index_padded.long()
        query_canonical_table = (
            data.mcl_query_canonical_index_padded.long()
        )
        thresholds_all = data.trimer_mcl_thresholds.float()
        valid = valid.clone()
        if self.mask_mode == "count_matched_random":
            random_valid = getattr(data, "mcl_random_mask_valid", None)
            if random_valid is None:
                raise ValueError(
                    "count_matched_random MCL requires mcl_random_mask_valid"
                )
            random_valid = torch.as_tensor(
                random_valid, dtype=torch.bool, device=valid.device
            ).flatten()
            if random_valid.numel() != valid.numel():
                raise ValueError("random-mask validity length mismatch")
            valid &= random_valid

        for key_capacity in (24, 48, 96, 192, 384):
            graph_ids = torch.nonzero(
                valid & (bucket_sizes == key_capacity), as_tuple=False
            ).flatten()
            if graph_ids.numel() == 0:
                continue
            query_capacity = key_capacity // 3
            keys = key_table[graph_ids, :key_capacity]
            queries = query_table[graph_ids, :query_capacity]
            query_local = query_local_table[graph_ids, :query_capacity]
            query_canonical = query_canonical_table[
                graph_ids, :query_capacity
            ]
            key_mask = keys >= 0
            query_mask = queries >= 0
            safe_keys = keys.clamp_min(0)
            safe_queries = queries.clamp_min(0)
            safe_query_local = query_local.clamp_min(0)

            memory = tokens[safe_keys]
            central_state = tokens[safe_queries]
            memory_positions = data.trimer_pos[safe_keys].float()
            if self.coordinate_shuffle:
                counts = key_mask.sum(dim=1, keepdim=True)
                local = torch.arange(
                    key_capacity, device=memory_positions.device
                ).view(1, -1)
                reverse = (counts - 1 - local).clamp_min(0)
                memory_positions = memory_positions.gather(
                    1, reverse.unsqueeze(-1).expand(-1, -1, 3)
                )
                central_positions = memory_positions.gather(
                    1,
                    safe_query_local.unsqueeze(-1).expand(-1, -1, 3),
                )
            else:
                central_positions = data.trimer_pos[safe_queries].float()
            distances = torch.cdist(central_positions, memory_positions)
            thresholds = thresholds_all[graph_ids]
            threshold_valid = torch.isfinite(thresholds).all(dim=1)
            invalid_graphs = graph_ids[~threshold_valid]
            valid[invalid_graphs] = False
            query_mask = query_mask & threshold_valid.unsqueeze(1)

            precomputed_visible = None
            if self.mask_mode == "count_matched_random":
                # A4: replace distance-threshold visibility with the fixed
                # count-matched random sets from the sidecar (Plan §4).  Same
                # per-query sparsity as A3, different key identity.
                if not hasattr(data, "mcl_random_visible20") or not hasattr(
                    data, "mcl_random_visible50"
                ):
                    raise ValueError("random-mask visible tables are missing")
                trimer_start = data.mcl_trimer_start[graph_ids]
                query_start = data.mcl_query_start[graph_ids]
                # The sidecar rows are ordered by the central-query column
                # (0..Q-1), whereas ``safe_query_local`` is the Trimer atom
                # position used to replace the central token in ``memory``.
                # Mixing those two coordinate systems sends padded MCL
                # batches to arbitrary/out-of-range sidecar rows.  Index the
                # sidecar with the query column and clamp only padded columns
                # to a harmless real row; ``query_mask`` below prevents them
                # from contributing to the update.
                query_columns = torch.arange(
                    query_capacity, device=query_start.device
                ).view(1, -1).expand(graph_ids.numel(), -1)
                q_global = query_start.unsqueeze(1) + query_columns
                sidecar_rows = int(data.mcl_random_visible20.size(0))
                if sidecar_rows <= 0:
                    raise ValueError("A4 random sidecar has no query rows")
                q_global = q_global.clamp(0, sidecar_rows - 1)
                vis20 = data.mcl_random_visible20[q_global]
                vis50 = data.mcl_random_visible50[q_global]
                keys_local = keys.unsqueeze(1) - trimer_start[:, None, None]
                valid_key = (keys_local >= 0) & (keys_local < 384)
                kl = keys_local.clamp(0, 383).expand(
                    -1, q_global.size(1), -1
                )
                v20 = (
                    vis20.gather(-1, kl)
                    & valid_key & key_mask.unsqueeze(1)
                )
                v50 = (
                    vis50.gather(-1, kl)
                    & valid_key & key_mask.unsqueeze(1)
                )
                precomputed_visible = torch.stack([v20, v50], dim=-1)

            for layer in self.layers:
                memory_current = memory.clone()
                batch_index, query_index = torch.nonzero(
                    query_mask, as_tuple=True
                )
                memory_current[
                    batch_index, safe_query_local[batch_index, query_index]
                ] = central_state[batch_index, query_index]
                central_state = layer.forward_batched(
                    central_state,
                    memory_current,
                    distances,
                    thresholds,
                    key_mask=key_mask,
                    query_mask=query_mask,
                    precomputed_visible=precomputed_visible,
                )

            final_memory = memory.clone()
            batch_index, query_index = torch.nonzero(
                query_mask, as_tuple=True
            )
            final_memory[
                batch_index, safe_query_local[batch_index, query_index]
            ] = central_state[batch_index, query_index]
            final_trimer_states[safe_keys[key_mask]] = final_memory[key_mask]
            targets = query_canonical[batch_index, query_index]
            delta_canonical[targets] = (
                central_state[batch_index, query_index]
                - canonical[targets]
            )

        canonical_periodic = bool(
            getattr(data, "mts_canonical_periodic", False)
            or getattr(data, "mips_local_lga_schema_version", 0) == 2
        )
        if canonical_periodic:
            # Canonical topology has exactly one node state per atom.  The
            # Trimer copies were lifted from ``canonical[base_index]`` above;
            # scatter-mean and copy broadcast would reintroduce the retired
            # explicit O8 representation.
            node_delta = delta_canonical * torch.tanh(
                self.geometry_gate
            ).unsqueeze(0)
        else:
            canonical_index = data.canonical_ru_atom_index.long()
            node_delta = (
                delta_canonical
                * torch.tanh(self.geometry_gate).unsqueeze(0)
            )[canonical_index]
        if not bool(valid.any()):
            zero_anchor = topology_nodes.new_zeros(())
            for parameter in self.parameters():
                zero_anchor = zero_anchor + parameter.reshape(-1)[0] * 0.0
            node_delta = node_delta + zero_anchor
        if not bool(torch.isfinite(node_delta).all()):
            raise FloatingPointError("non-finite Trimer-MCL residual")
        return node_delta, final_trimer_states, valid

    def forward_with_aux(self, topology_nodes, data):
        canonical_periodic = bool(
            getattr(data, "mts_canonical_periodic", False)
            or getattr(data, "mips_local_lga_schema_version", 0) == 2
        )
        required = (
            "trimer_pos", "trimer_batch",
            "trimer_central_ru_mask",
            "trimer_geometry_valid", "trimer_geometry_is_3d",
            "trimer_2d_fallback", "mips_to_trimer_central_index",
        )
        if not canonical_periodic:
            required = ("canonical_ru_atom_index",) + required
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(
                "Trimer-MCL input is missing " + ", ".join(missing)
            )
        canonical_periodic = bool(
            getattr(data, "mts_canonical_periodic", False)
            or getattr(data, "mips_local_lga_schema_version", 0) == 2
        )
        if canonical_periodic:
            # Production canonical path: topology_nodes already is [N, D]
            # canonical RU state.  Do not scatter copies or broadcast a
            # geometry residual back to explicit copies.
            canonical_index = torch.arange(
                topology_nodes.size(0), device=topology_nodes.device,
                dtype=torch.long,
            )
            canonical = topology_nodes
            canonical_count = int(topology_nodes.size(0))
        else:
            canonical_index = data.canonical_ru_atom_index.long()
            canonical_count = int(getattr(
                data, "canonical_graph_index",
                canonical_index.new_empty((0,)),
            ).numel())
            if canonical_count == 0 and canonical_index.numel():
                # Diagnostic batches created outside the production collator keep
                # the legacy fallback; production never synchronizes max().item().
                canonical_count = int(canonical_index.max().item()) + 1
            canonical = scatter(
                topology_nodes, canonical_index, dim=0,
                dim_size=canonical_count, reduce="mean",
            )
        base_name = (
            "trimer_base_ru_atom_index"
            if hasattr(data, "trimer_base_ru_atom_index")
            else "trimer_base_ru_atom_id"
        )
        if not hasattr(data, base_name):
            raise ValueError("Trimer input is missing base canonical atom IDs")
        base_index = getattr(data, base_name).long()
        if base_index.numel() != data.trimer_pos.size(0):
            raise ValueError("Trimer identity/coordinate lengths differ")
        tokens = self.input_norm(canonical[base_index])
        final_trimer_states = tokens.clone()
        delta_canonical = canonical.new_zeros(canonical.shape)
        # Use the same complete eligibility predicate as cache validation and
        # Stage 2.  This prevents a malformed mapping or accidental 2-D
        # fallback from entering the geometry branch merely because its flag
        # happened to be set to True.
        cached_valid = getattr(data, "mcl_valid", None)
        if cached_valid is not None:
            valid = torch.as_tensor(
                cached_valid, dtype=torch.bool, device=topology_nodes.device
            ).flatten()
            if valid.numel() != int(data.trimer_geometry_valid.numel()):
                raise ValueError("cached MCL validity length mismatch")
            # Cache predicates are an optimisation, never an override for the
            # immutable geometry validity flags.  This also guarantees an
            # exact O8 fallback if a caller invalidates a geometry record
            # after collation (or if a stale validity sidecar is encountered).
            geometry_valid = torch.as_tensor(
                data.trimer_geometry_valid,
                dtype=torch.bool,
                device=topology_nodes.device,
            ).flatten()
            geometry_is_3d = torch.as_tensor(
                data.trimer_geometry_is_3d,
                dtype=torch.bool,
                device=topology_nodes.device,
            ).flatten()
            fallback_2d = torch.as_tensor(
                data.trimer_2d_fallback,
                dtype=torch.bool,
                device=topology_nodes.device,
            ).flatten()
            valid = valid & geometry_valid & geometry_is_3d & ~fallback_2d
        else:
            # Compatibility path for hand-built diagnostic batches.  Dataset
            # collate computes this once on the CPU; production batches never
            # execute the per-graph Python predicate here.
            valid = torch.tensor(
                [trimer_can_enter_mcl(data, graph_idx)
                 for graph_idx in range(int(data.trimer_geometry_valid.numel()))],
                dtype=torch.bool,
                device=topology_nodes.device,
            )
        trimer_batch = data.trimer_batch.long()
        node_mapping = data.mips_to_trimer_central_index.long()
        if node_mapping.numel() != topology_nodes.size(0):
            raise ValueError("O8-to-Trimer mapping must contain one ID per node")

        padded_fields = (
            "mcl_bucket_size",
            "mcl_key_index_padded",
            "mcl_query_index_padded",
            "mcl_query_local_index_padded",
            "mcl_query_canonical_index_padded",
        )
        if all(hasattr(data, name) for name in padded_fields):
            return self._forward_padded(
                topology_nodes, data, canonical, tokens,
                final_trimer_states, delta_canonical, valid,
            )
        if self.mask_mode == "count_matched_random":
            raise ValueError(
                "count_matched_random MCL requires the padded production collator"
            )

        # Validate records once, then group equal (Trimer atom count, central
        # atom count) records so their two-scale MCL can run as one padded
        # batched attention operation.  Invalid records remain exact O8
        # fallbacks and never enter a distance tensor.
        groups = {}
        cached_thresholds = getattr(data, "trimer_mcl_thresholds", None)
        for graph_idx in range(valid.numel()):
            if not bool(valid[graph_idx]):
                continue
            atoms = torch.nonzero(
                trimer_batch == graph_idx, as_tuple=False
            ).flatten()
            graph_nodes = torch.nonzero(
                data.batch == graph_idx, as_tuple=False
            ).flatten()
            central_mask = data.trimer_central_ru_mask[atoms].bool()
            central = atoms[central_mask]
            if atoms.numel() < 3 or not central.numel():
                valid[graph_idx] = False
                continue
            graph_mapping = node_mapping[graph_nodes]
            if (
                graph_mapping.numel() != graph_nodes.numel()
                or bool((graph_mapping < 0).any())
                or bool((graph_mapping >= data.trimer_pos.size(0)).any())
                or not bool(torch.isin(graph_mapping, atoms).all())
                or not bool(data.trimer_central_ru_mask[graph_mapping].all())
            ):
                # An invalid mapping is an unavailable geometry sample, not a
                # process-wide failure.  Its canonical residual remains zero.
                valid[graph_idx] = False
                continue
            positions = data.trimer_pos[atoms].float()
            if not bool(torch.isfinite(positions).all()):
                valid[graph_idx] = False
                continue
            if cached_thresholds is not None:
                thresholds = torch.as_tensor(cached_thresholds)[graph_idx].to(
                    device=positions.device, dtype=positions.dtype
                )
            else:
                pair_distances = torch.pdist(positions)
                if not pair_distances.numel():
                    raise ValueError("valid Trimer has no pair distances")
                thresholds = torch.quantile(
                    pair_distances,
                    pair_distances.new_tensor(self.percentiles),
                )
            if thresholds.numel() != 2 or not bool(torch.isfinite(thresholds).all()):
                # A malformed precomputed threshold is equivalent to an
                # unavailable geometry record; do not silently create a
                # dense mask from it.
                valid[graph_idx] = False
                continue
            central_local = torch.nonzero(
                central_mask, as_tuple=False
            ).flatten()
            targets = base_index[central]
            if targets.unique().numel() != central.numel():
                raise ValueError("central canonical identity is not one-to-one")
            key = (int(atoms.numel()), int(central.numel()))
            groups.setdefault(key, []).append(
                (atoms, central, central_local, positions, thresholds, targets)
            )

        self.last_thresholds = [] if self.debug_attention else None
        for records in groups.values():
            memory = torch.stack([tokens[item[0]] for item in records], dim=0)
            central_state = torch.stack(
                [tokens[item[1]] for item in records], dim=0
            )
            # Coordinates remain the only geometric input to the visibility
            # masks; tokens carry the learned topology representation.
            distances = torch.stack([
                torch.cdist(
                    data.trimer_pos[item[1]].float(),
                    data.trimer_pos[item[0]].float(),
                )
                for item in records
            ], dim=0)
            thresholds = torch.stack([item[4] for item in records], dim=0)
            central_positions = torch.stack([item[2] for item in records], dim=0)
            for layer in self.layers:
                # Outer-unit memory remains the canonical topology token;
                # only central tokens are replaced by the previous layer.
                memory_current = memory.clone()
                memory_current.scatter_(
                    1,
                    central_positions.unsqueeze(-1).expand(
                        -1, -1, self.dim
                    ),
                    central_state,
                )
                central_state = layer.forward_batched(
                    central_state, memory_current, distances, thresholds
                )
            final_memory = memory.clone()
            final_memory.scatter_(
                1,
                central_positions.unsqueeze(-1).expand(-1, -1, self.dim),
                central_state,
            )
            target_indices = torch.stack([item[5] for item in records], dim=0)
            delta_canonical[target_indices.reshape(-1)] = (
                central_state - canonical[target_indices]
            ).reshape(-1, self.dim)
            for record_index, item in enumerate(records):
                final_trimer_states[item[0]] = final_memory[record_index]
            if self.debug_attention:
                self.last_thresholds.extend(
                    [value.detach() for value in thresholds]
                )

        if canonical_periodic:
            node_delta = delta_canonical * torch.tanh(
                self.geometry_gate
            ).unsqueeze(0)
        else:
            node_delta = (
                delta_canonical
                * torch.tanh(self.geometry_gate).unsqueeze(0)
            )[canonical_index]
        if not bool(valid.any()):
            # Keep DDP parameter usage identical across ranks while retaining
            # an exact numerical zero for all invalid samples.
            zero_anchor = topology_nodes.new_zeros(())
            for parameter in self.parameters():
                zero_anchor = zero_anchor + parameter.reshape(-1)[0] * 0.0
            node_delta = node_delta + zero_anchor
        if not bool(torch.isfinite(node_delta).all()):
            raise FloatingPointError("non-finite Trimer-MCL residual")
        return node_delta, final_trimer_states, valid

    def forward(self, topology_nodes, data):
        node_delta, _, _ = self.forward_with_aux(topology_nodes, data)
        return node_delta
