"""log-softmax, softmax and cross-entropy without Softmax Collapse, in JAX (depends only on jax).

Reference: J. T. Barron, D. Gnanapragasam and M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2.

Same math as ``softmax_torch.py`` (PyTorch); see that file for the derivation:

    s_k = 1[k != argmax] (z_k - z_max),   m = -LSE_{j != argmax} s_j,   log_softmax_k = s_k - softplus(-m)

The argmax slot is an exact 0 and the argmax logit enters only through ``z_j - z_max``, so autodiff
gives it minus the sum of the other gradients. Everything is jit / vmap / grad compatible.

Usage:

    import jax, jax.numpy as jnp
    from softmax_jax import log_softmax, softmax, log_prob_of, cross_entropy

    logits = jax.random.normal(jax.random.key(0), (8, 32000))
    target = jax.random.randint(jax.random.key(1), (8,), 0, 32000)
    loss = cross_entropy(logits, target)            # logits [..., C], target [...]
    lp = log_prob_of(logits, target)                # log p(target) per row
    lp_all = log_softmax(logits, axis=-1)           # same shapes as jax.nn.log_softmax

``vocab_parallel_log_prob_of`` is the version for logits sharded over the class dimension along a
named mesh axis (call it inside ``jax.shard_map``).

Run ``python softmax_jax.py`` for a demo of the difference on a large-margin row.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["log_softmax", "softmax", "log_prob_of", "cross_entropy", "vocab_parallel_log_prob_of"]


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


# ── class dimension sharded over a mesh axis ───────────────────────────────────────────────────

def vocab_parallel_log_prob_of(logits_shard: jax.Array, target: jax.Array, axis_name) -> jax.Array:
    """``log_prob_of`` for logits split over the last (class) dimension along mesh axis
    ``axis_name``; call it inside ``jax.shard_map``. ``logits_shard`` is ``[..., C / n]`` and holds
    classes ``[i * C / n, (i + 1) * C / n)`` on axis index ``i`` (equal shards, in order); ``target``
    is ``[...]`` with *global* class indices, the same on every shard. Returns the log-probability
    ``[...]``, identical on every shard."""
    vp = logits_shard.shape[-1]
    t = (target - lax.axis_index(axis_name) * vp)[..., None]           # local index, out of range elsewhere
    is_t = jnp.arange(vp) == t                                         # all False where the target is on another shard
    z_t = lax.psum(jnp.where(is_t, logits_shard, 0).sum(-1), axis_name)  # target logit on every shard
    others = jnp.where(is_t, -jnp.inf, logits_shard - z_t[..., None])
    m = lax.pmax(lax.stop_gradient(others.max(-1)), axis_name)          # stabiliser only, no gradient
    m = jnp.where(jnp.isfinite(m), m, 0)
    s = lax.psum(jnp.exp(others - m[..., None]).sum(-1), axis_name)
    return -jax.nn.softplus(m + jnp.log(s))


if __name__ == "__main__":
    z = jnp.array([[20.0, 0.0, 0.0, 0.0]])            # the target leads by 20 > 16.6
    t = jnp.array([0])
    std = lambda x: jax.nn.log_softmax(x, -1)[0, 0]
    new = lambda x: log_prob_of(x, t)[0]
    for name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
        x = z.astype(dtype)
        print(f"{name}: logits {z.tolist()[0]}, target 0")
        for label, f in (("jax.nn.log_softmax", std), ("softmax_jax       ", new)):
            lp, g = jax.value_and_grad(f)(x)
            print(f"  {label} log p = {float(lp):.6g}   grad = {[float(f'{v:.4g}') for v in g[0].astype(jnp.float32).tolist()]}")
