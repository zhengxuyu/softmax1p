"""log_softmax, softmax and cross_entropy without Softmax Collapse, in PyTorch.

Author: Ke Xu (https://github.com/kevinkexu)
Reference: J. T. Barron, D. Gnanapragasam, M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2.

The standard log-softmax forms ``1 + e^{-m}``, which rounds to 1 once the top logit leads by a
margin ``m`` past half an ulp (16.6 in float32, 5.5 in bfloat16): the top entry's log-probability
and gradient become 0 and the row's gradient no longer sums to zero. The fix never forms that sum:

    s_k = 1[k != argmax] (z_k - z_max),   log_softmax_k = s_k - softplus(LSE_{j != argmax} s_j)

so the argmax logit enters only through differences and autodiff gives it minus the sum of the
other gradients. ``log_prob_of`` uses the target in place of the argmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["log_softmax", "softmax", "log_prob_of", "cross_entropy"]

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
