"""``log_prob_of`` from ``softmax_jax.py`` for logits sharded over the class dimension along a
named mesh axis (call it inside ``jax.shard_map``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["vocab_parallel_log_prob_of"]


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
