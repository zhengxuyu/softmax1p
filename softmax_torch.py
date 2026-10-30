"""log-softmax, softmax and cross-entropy without Softmax Collapse, in PyTorch (depends only on torch).

Author: Ke Xu (https://github.com/kevinkexu)

Reference: J. T. Barron, D. Gnanapragasam and M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2.

Softmax Collapse. The standard log-softmax shifts by the largest logit and adds ``e^0 = 1`` to the
other exponentials. Once the largest logit leads the rest by a margin ``m`` with ``e^{-m}`` below
half an ulp of 1 (``m > 16.6`` in float32, ``m > 5.5`` in bfloat16, ``m > 7.6`` in float16),
``1 + e^{-m}`` rounds to 1: the largest entry's log-probability becomes exactly 0, its gradient
becomes 0, and the gradients of a row no longer sum to zero.

The fix never forms ``1 + e^{-m}``. The argmax slot is set to an exact 0, the margin comes from a
log-sum-exp over the other slots only, and the two are joined by softplus:

    s_k = 1[k != argmax] (z_k - z_max),   m = -LSE_{j != argmax} s_j,   log_softmax_k = s_k - softplus(-m)

so the argmax logit enters only through the differences ``z_j - z_max`` and autodiff gives it minus
the sum of the other gradients. ``log_prob_of`` is the same with the target index in place of the
argmax (the paper's hard-target cross-entropy, negated), and ``cross_entropy`` is its negation with
``ignore_index`` and ``reduction``.

Usage:

    import torch
    from softmax_torch import log_softmax, softmax, log_prob_of, cross_entropy

    logits = torch.randn(8, 32000, requires_grad=True)
    target = torch.randint(0, 32000, (8,))
    loss = cross_entropy(logits, target)            # class dimension last: logits [..., C], target [...]
    lp = log_prob_of(logits, target)                # log p(target) per row, e.g. for policy gradients
    lp_all = log_softmax(logits, dim=-1)            # same shapes as torch.log_softmax

``vocab_parallel_log_prob_of`` is the version for logits sharded over the class dimension across a
``torch.distributed`` group (tensor-parallel language-model heads).

Run ``python softmax_torch.py`` for a demo of the difference on a large-margin row.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["log_softmax", "softmax", "log_prob_of", "cross_entropy", "vocab_parallel_log_prob_of"]

_NEG_INF = float("-inf")


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = z.movedim(dim, -1)
    i_max = z.argmax(dim=-1, keepdim=True)
    z_max = z.gather(-1, i_max)                                        # max logit
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, i_max, True)
    shifted = torch.where(is_max, torch.zeros_like(z), z - z_max)      # exact 0 at the argmax
    others = shifted.masked_fill(is_max, _NEG_INF)                     # argmax left out of the sum
    lse = torch.logsumexp(others, dim=-1, keepdim=True)                # = -margin
    return (shifted - F.softplus(lse)).movedim(-1, dim)


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return log_softmax(z, dim).exp()


def log_prob_of(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """log_softmax(z)[..., target] over the last dimension, for integer ``target`` of shape
    ``z.shape[:-1]``."""
    t = target.unsqueeze(-1)
    z_t = z.gather(-1, t)                                              # target logit
    is_t = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, t, True)
    others = (z - z_t).masked_fill(is_t, _NEG_INF)                     # target left out of the sum
    return -F.softplus(torch.logsumexp(others, dim=-1))


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
                  reduction: str = "mean") -> torch.Tensor:
    """Hard-target cross-entropy ``-log_prob_of(logits, target)``. Unlike
    ``torch.nn.functional.cross_entropy`` the class dimension is the last one: ``logits`` is
    ``[..., C]`` and ``target`` is ``[...]`` (integer class indices). Rows whose target equals
    ``ignore_index`` contribute 0 and are left out of the ``"mean"`` denominator. ``reduction`` is
    ``"none"`` (shape ``target.shape``), ``"sum"`` or ``"mean"``."""
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"reduction={reduction!r}: use none, sum or mean")
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))  # any in-range index for ignored rows
    loss = torch.where(valid, -log_prob_of(logits, safe_target), torch.zeros((), dtype=logits.dtype, device=logits.device))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / valid.sum()                                     # nan if every row is ignored, as in torch


# ── class dimension sharded over a torch.distributed group ─────────────────────────────────────

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


if __name__ == "__main__":
    z = torch.tensor([[20.0, 0.0, 0.0, 0.0]])        # the target leads by 20 > 16.6
    t = torch.tensor([0])
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        x = z.to(dtype).detach().clone().requires_grad_(True)
        std_lp = torch.log_softmax(x, -1)[0, 0]
        std_lp.backward()
        std_grad = x.grad.clone()
        x.grad = None
        new_lp = log_prob_of(x, t)[0]
        new_lp.backward()
        print(f"{name}: logits {z.tolist()[0]}, target 0")
        print(f"  torch.log_softmax  log p = {std_lp.item():.6g}   grad = {[round(v, 12) for v in std_grad[0].float().tolist()]}")
        print(f"  softmax_torch      log p = {new_lp.item():.6g}   grad = {[round(v, 12) for v in x.grad[0].float().tolist()]}")
    print("float64 reference log p =", torch.log_softmax(z.double(), -1)[0, 0].item())
