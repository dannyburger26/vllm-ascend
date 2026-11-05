from dataclasses import dataclass, field

import torch

from vllm_ascend import envs
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, TreeBuildState


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
    depth_hidden: torch.Tensor | None = None,
    frontier_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add DSpark bias in draft-vocab space from target-vocab frontier tokens."""
    base = depth_logits.unsqueeze(1)
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1), None
    if depth_hidden is not None and frontier_states is not None:
        markov_step = getattr(draft_model, "markov_step", None)
        if markov_step is not None:
            num_reqs, width = frontier_tokens.shape
            flat_tokens = frontier_tokens.reshape(-1)
            flat_hidden = (
                depth_hidden.unsqueeze(1)
                .expand(-1, width, -1)
                .reshape(-1, depth_hidden.shape[-1])
            )
            next_states, bias = markov_step(
                frontier_states.reshape(-1, frontier_states.shape[-1]),
                flat_tokens,
                flat_hidden,
            )
            return (
                base + bias.view(num_reqs, width, -1),
                next_states.view(num_reqs, width, -1),
            )
    markov_emb = draft_model.markov_embed(frontier_tokens)
    # NPU unquantized gemm is 2D-only; restore the frontier shape afterward.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1), None


@dataclass
class BeamScoreContext:
    """Per-build device tensors; the greedy-pool list holds device node IDs."""

    draft_logits: torch.Tensor
    frontier_tokens: torch.Tensor
    draft_hidden: torch.Tensor | None = None
    frontier_states: torch.Tensor | None = None
    next_states: torch.Tensor | None = None
    req_ids: torch.Tensor | None = None
    greedy_frontier: torch.Tensor | None = None
    greedy_candidate: torch.Tensor | None = None
    greedy_pools: list[torch.Tensor] = field(default_factory=list)


class BeamTreeBuilder(TreeBuilder[BeamScoreContext]):
    """Level-wise beam expansion with optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(
        self,
        budget: int,
        topk: int,
        draft_model=None,
        branch_ks: list[int] | None = None,
    ):
        super().__init__(budget, topk, branch_ks=branch_ks)
        self.draft_model = draft_model
        self.keep_greedy = envs.VLLM_ASCEND_TREE_KEEP_GREEDY

    def _beam_width(self, vocab: int) -> int:
        # Buffer sizing uses the widest layer; per-layer widths come from
        # ``_depth_width`` below.
        if self.branch_ks is not None:
            return min(max(self.branch_ks), vocab)
        return min(self.topk, vocab)

    def _depth_width(self, max_width: int, depth_slot: int) -> int:
        if self.branch_ks is not None:
            return min(self.branch_ks[depth_slot], max_width)
        return min(self.topk, max_width)

    def _pool_capacity(self, width: int, spec_num: int) -> int:
        # Keep every parent-child edge. Only the frontier is pruned to ``width``.
        return width + max(spec_num - 1, 0) * width * width

    def _prepare_scores(
        self,
        state: TreeBuildState,
        draft_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None,
    ) -> BeamScoreContext:
        state.frontier = state.frontier[:, :1]
        frontier_states = None
        if self.draft_model is not None:
            markov_init_state = getattr(self.draft_model, "markov_init_state", None)
            if markov_init_state is not None:
                frontier_states = markov_init_state(draft_logits.shape[0]).unsqueeze(1)
        ctx = BeamScoreContext(
            draft_logits,
            root_token_ids.unsqueeze(1),
            draft_hidden=draft_hidden,
            frontier_states=frontier_states,
        )
        if self.keep_greedy:
            num_reqs, spec_num, _ = draft_logits.shape
            if self.budget < spec_num:
                raise ValueError("Keeping the complete greedy chain requires budget >= depth")
            ctx.req_ids = torch.arange(num_reqs, device=draft_logits.device)
            ctx.greedy_frontier = torch.zeros(
                num_reqs, dtype=torch.long, device=draft_logits.device
            )
        return ctx

    def _score_layer(
        self,
        state: TreeBuildState,
        ctx: BeamScoreContext,
        depth_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth_hidden = (
            ctx.draft_hidden[:, depth_slot]
            if ctx.draft_hidden is not None
            else None
        )
        step_logits, next_states = _markov_correct_logits(
            self.draft_model,
            ctx.draft_logits[:, depth_slot],
            ctx.frontier_tokens,
            depth_hidden,
            ctx.frontier_states,
        )
        ctx.next_states = next_states
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        depth_width = self._depth_width(state.width, depth_slot)
        top_vals, top_ids = log_probs.topk(depth_width, dim=-1)
        if self.keep_greedy:
            # topk can omit argmax on exact-logit ties. Keep the sequential edge.
            greedy_token = step_logits[ctx.req_ids, ctx.greedy_frontier].argmax(-1)
            local_ids = top_ids[ctx.req_ids, ctx.greedy_frontier]
            present = (local_ids == greedy_token[:, None]).any(-1)
            last_id = torch.where(present, local_ids[:, -1], greedy_token)
            top_ids[ctx.req_ids, ctx.greedy_frontier, -1] = last_id
            top_vals[ctx.req_ids, ctx.greedy_frontier, -1] = log_probs[
                ctx.req_ids, ctx.greedy_frontier, last_id
            ]
            greedy_child = (
                top_ids[ctx.req_ids, ctx.greedy_frontier] == greedy_token[:, None]
            ).long().argmax(-1)
            ctx.greedy_candidate = ctx.greedy_frontier * depth_width + greedy_child

        if self.draft_model is not None:
            top_ids = self.draft_model.map_draft_to_target(top_ids)
        return top_vals, top_ids

    def _expand_layer(
        self,
        state: TreeBuildState,
        ctx: BeamScoreContext,
        top_scores: torch.Tensor,
        cand_ids: torch.Tensor,
        child_depth: int,
        graph_safe: bool,
    ) -> tuple[torch.Tensor, int]:
        # Shape values and slice bounds are host ints; selections stay on device.
        num_reqs, batch, k = top_scores.shape
        frontier_scores = torch.gather(state.path_scores, 1, state.frontier)
        candidate_scores = frontier_scores.unsqueeze(-1) + top_scores
        num_candidates = batch * k
        cand_flat = candidate_scores.reshape(num_reqs, -1)
        top_ids_flat = cand_ids.reshape(num_reqs, -1)
        parent_flat = state.frontier.repeat_interleave(k, dim=-1)
        selected = cand_flat.topk(min(k, num_candidates), dim=-1).indices
        if self.keep_greedy:
            present = (selected == ctx.greedy_candidate[:, None]).any(-1)
            selected[:, -1] = torch.where(
                present, selected[:, -1], ctx.greedy_candidate
            )
            ctx.greedy_frontier = (
                selected == ctx.greedy_candidate[:, None]
            ).long().argmax(-1)

        n_add = num_candidates
        start = state.num_nodes + 1
        state.tokens[:, state.num_nodes : state.num_nodes + n_add] = top_ids_flat
        state.depths[:, state.num_nodes : state.num_nodes + n_add] = child_depth
        state.parents[:, start : start + n_add] = parent_flat
        state.path_scores[:, start : start + n_add] = cand_flat
        pool_ids = self._node_arange[start : start + n_add].unsqueeze(0).expand(
            num_reqs, -1
        )
        ctx.frontier_tokens = torch.gather(top_ids_flat, 1, selected)
        if ctx.next_states is not None:
            ctx.frontier_states = torch.gather(
                ctx.next_states,
                1,
                selected.unsqueeze(-1).expand(-1, -1, ctx.next_states.shape[-1]),
            )
        frontier = torch.gather(pool_ids, 1, selected)
        if self.keep_greedy:
            ctx.greedy_pools.append(
                torch.gather(pool_ids, 1, ctx.greedy_candidate[:, None]).squeeze(1)
            )
        return frontier, n_add

    def _prune_priority(
        self, state: TreeBuildState, ctx: BeamScoreContext
    ) -> tuple[torch.Tensor, int]:
        """Return (priority, num_nodes) without pruning; the caller decides
        whether to prune via the Triton kernel or the Torch fallback."""
        num_pool = state.num_nodes
        num_nodes = min(self.budget, num_pool)
        pool_scores = state.path_scores[:, 1 : num_pool + 1]
        # Stable score order puts ancestors first, including exact-score ties.
        if self.keep_greedy:
            priority = pool_scores.clone()
            greedy_slots = torch.stack(ctx.greedy_pools, dim=1) - 1
            priority.scatter_(1, greedy_slots, float("inf"))
        else:
            priority = pool_scores
        return priority, num_nodes

    def _select_nodes(
        self, state: TreeBuildState, ctx: BeamScoreContext
    ) -> torch.Tensor:
        priority, num_nodes = self._prune_priority(state, ctx)
        num_pool = state.num_nodes
        packed = priority.argsort(dim=-1, descending=True, stable=True)[:, :num_nodes]
        packed_depth = torch.gather(state.depths[:, :num_pool], 1, packed)
        packed_rank = packed_depth * num_pool + packed
        packed = torch.gather(packed, 1, packed_rank.argsort(dim=-1))
        return packed + 1
