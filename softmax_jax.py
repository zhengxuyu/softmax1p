"""log_softmax, softmax and cross_entropy without Softmax Collapse, in JAX.

Reference: J. T. Barron, D. Gnanapragasam, M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2. Same math as ``softmax_torch.py``:

    s_k = 1[k != argmax] (z_k - z_max),   log_softmax_k = s_k - softplus(LSE_{j != argmax} s_j)

Everything is jit / vmap / grad compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["log_softmax", "softmax", "log_prob_of", "cross_entropy"]


def log_softmax(z: jax.Array, axis: int = -1) -> jax.Array:
    z = jnp.moveaxis(z, axis, -1)
    i_max = jnp.argmax(z, axis=-1, keepdims=True)
    z_max = jnp.take_along_axis(z, i_max, axis=-1)                    # max logit
    is_max = jnp.arange(z.shape[-1]) == i_max
    shifted = jnp.where(is_max, 0, z - z_max)                          # exact 0 at the argmax
    others = jnp.where(is_max, -jnp.inf, shifted)                      # argmax left out of the sum
    lse = jax.nn.logsumexp(others, axis=-1, keepdims=True)             # = -margin
    return jnp.moveaxis(shifted - jax.nn.softplus(lse), -1, axis)


def softmax(z: jax.Array, axis: int = -1) -> jax.Array:
    return jnp.exp(log_softmax(z, axis))


def log_prob_of(z: jax.Array, target: jax.Array) -> jax.Array:
    """log_softmax(z)[..., target] over the last dimension, for integer ``target`` of shape
    ``z.shape[:-1]``."""
    t = target[..., None]
    z_t = jnp.take_along_axis(z, t, axis=-1)                           # target logit
    is_t = jnp.arange(z.shape[-1]) == t
    others = jnp.where(is_t, -jnp.inf, z - z_t)                        # target left out of the sum
    return -jax.nn.softplus(jax.nn.logsumexp(others, axis=-1))


def cross_entropy(logits: jax.Array, target: jax.Array, ignore_index: int = -100,
                  reduction: str = "mean") -> jax.Array:
    """Hard-target cross-entropy ``-log_prob_of(logits, target)``. ``logits`` is ``[..., C]`` and
    ``target`` is ``[...]`` (integer class indices). Rows whose target equals ``ignore_index``
    contribute 0 and are left out of the ``"mean"`` denominator. ``reduction`` is ``"none"`` (shape
    ``target.shape``), ``"sum"`` or ``"mean"``."""
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"reduction={reduction!r}: use none, sum or mean")
    valid = target != ignore_index
    safe_target = jnp.where(valid, target, 0)                          # any in-range index for ignored rows
    loss = jnp.where(valid, -log_prob_of(logits, safe_target), 0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / valid.sum()                                     # nan if every row is ignored
