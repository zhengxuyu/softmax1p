"""``log_prob_of`` from ``softmax_torch.py`` for logits sharded over the class dimension across a
``torch.distributed`` group (tensor-parallel language-model heads). Chunked over rows; recomputes
the softmax in backward to bound memory.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["vocab_parallel_log_prob_of"]

_NEG_INF = float("-inf")
_DEFAULT_CHUNK = 1024


def _group_info(group) -> tuple[int, int]:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(group), dist.get_world_size(group)


def _all_reduce(x: torch.Tensor, group, world: int, op=None) -> None:
    if world > 1:
        dist.all_reduce(x, op=op or dist.ReduceOp.SUM, group=group)


def _slot_mask(t: torch.Tensor, vp: int) -> torch.Tensor:
    """[n, vp] mask of the slot holding local index ``t`` (all False where ``t`` is on another rank)."""
    return torch.arange(vp, device=t.device) == t.unsqueeze(-1)


def _others(l: torch.Tensor, z_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``l - z_tok`` with the target's slot (on the rank that holds it) set to -inf."""
    return (l - z_tok.unsqueeze(-1)).masked_fill(mask, _NEG_INF)


def _lse_masked(d: torch.Tensor, group, world: int) -> torch.Tensor:
    """log-sum-exp over the full class dimension of sharded values, -inf entries excluded; -inf
    where every entry is -inf."""
    m = d.max(dim=-1).values
    _all_reduce(m, group, world, dist.ReduceOp.MAX)
    finite = torch.isfinite(m)
    safe = torch.where(finite, m, torch.zeros_like(m))
    s = torch.exp(d - safe.unsqueeze(-1)).sum(dim=-1)
    _all_reduce(s, group, world)
    return torch.where(finite, safe + s.log(), torch.full_like(m, _NEG_INF))


class _VocabParallelLogProb(torch.autograd.Function):
    """log p(tok) = -softplus(LSE_{j != tok}(z_j - z_tok)) on logits ``[n, vp]`` holding classes
    ``[rank * vp, (rank + 1) * vp)``, chunked over rows. Saves the logits and two numbers per row
    and recomputes the softmax chunk by chunk in backward."""

    @staticmethod
    def forward(ctx, logits, tokens, group, chunk):
        rank, world = _group_info(group)
        n, vp = logits.shape
        lo = rank * vp
        out = logits.new_empty(n)
        z_tok = logits.new_empty(n)
        for s in range(0, n, chunk):
            l = logits[s:s + chunk]
            t = tokens[s:s + chunk] - lo
            mask = _slot_mask(t, vp)
            inside = mask.any(dim=-1)
            picked = torch.where(inside, l.gather(1, t.clamp(0, vp - 1).unsqueeze(-1)).squeeze(-1), torch.zeros_like(t, dtype=l.dtype))
            _all_reduce(picked, group, world)                           # target logit on every rank
            out[s:s + chunk] = -F.softplus(_lse_masked(_others(l, picked, mask), group, world))
            z_tok[s:s + chunk] = picked
        ctx.save_for_backward(logits, tokens, z_tok, out)
        ctx.group, ctx.chunk, ctx.lo = group, chunk, lo
        return out

    @staticmethod
    def backward(ctx, grad):
        logits, tokens, z_tok, out = ctx.saved_tensors
        _, world = _group_info(ctx.group)
        n, vp = logits.shape
        grad = grad.contiguous()
        grad_in = torch.empty_like(logits)
        for s in range(0, n, ctx.chunk):
            l = logits[s:s + ctx.chunk]
            g = grad[s:s + ctx.chunk]
            mask = _slot_mask(tokens[s:s + ctx.chunk] - ctx.lo, vp)
            # p_j = exp(z_j - z_tok + log p(tok)) for j != tok, 0 in the target's slot
            p = _others(l, z_tok[s:s + ctx.chunk], mask).add_(out[s:s + ctx.chunk].unsqueeze(-1)).exp_()
            rest = p.sum(dim=-1)
            _all_reduce(rest, ctx.group, world)                         # sum of the other classes' p over all ranks
            # d log p(tok) / d z_j = -p_j for j != tok and sum_j p_j at tok, so the row sums to zero
            grad_in[s:s + ctx.chunk] = torch.where(mask, (rest * g).unsqueeze(-1), p.mul_(-g.unsqueeze(-1)))
        return grad_in, None, None, None


def vocab_parallel_log_prob_of(logits_shard: torch.Tensor, target: torch.Tensor, group=None,
                               chunk: int = _DEFAULT_CHUNK) -> torch.Tensor:
    """``log_prob_of`` for logits split over the last (class) dimension across ``group``.
    ``logits_shard`` is ``[..., C / world]`` and holds classes ``[rank * C / world, (rank + 1) *
    C / world)`` (equal shards, ranks in order); ``target`` is ``[...]`` with *global* class indices,
    the same on every rank. Returns the log-probability ``[...]``, identical on every rank, and
    backpropagates the gradient of this rank's shard. Every rank must call it with the same
    ``target`` and ``chunk``. ``group=None`` (or an uninitialised ``torch.distributed``) treats the
    logits as unsharded. ``chunk`` rows are processed at a time to bound temporary memory."""
    vp = logits_shard.size(-1)
    out = _VocabParallelLogProb.apply(logits_shard.reshape(-1, vp), target.reshape(-1), group, chunk)
    return out.reshape(target.shape)
