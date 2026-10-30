"""log_softmax, softmax and cross_entropy without Softmax Collapse, in TensorFlow.

Reference: J. T. Barron, D. Gnanapragasam, M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2. Same math as ``softmax_torch.py``:

    s_k = 1[k != argmax] (z_k - z_max),   log_softmax_k = s_k - softplus(LSE_{j != argmax} s_j)

Works eagerly and inside ``tf.function``.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["log_softmax", "softmax", "log_prob_of", "cross_entropy"]


def _pick(z: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """The one entry of ``z`` per row where ``mask`` is True, keepdims. Exact: a sum of one value and
    zeros. The gradient goes to that entry only."""
    return tf.reduce_sum(tf.where(mask, z, tf.zeros_like(z)), axis=-1, keepdims=True)


def log_softmax(z, axis: int = -1) -> tf.Tensor:
    z = tf.convert_to_tensor(z)
    axis = axis % z.shape.rank
    z = tf.experimental.numpy.moveaxis(z, axis, -1)
    is_max = tf.one_hot(tf.argmax(z, axis=-1), tf.shape(z)[-1], on_value=True, off_value=False, dtype=tf.bool)
    z_max = _pick(z, is_max)                                           # max logit
    shifted = tf.where(is_max, tf.zeros_like(z), z - z_max)            # exact 0 at the argmax
    others = tf.where(is_max, tf.fill(tf.shape(z), tf.constant(float("-inf"), z.dtype)), shifted)
    lse = tf.reduce_logsumexp(others, axis=-1, keepdims=True)          # = -margin
    return tf.experimental.numpy.moveaxis(shifted - tf.nn.softplus(lse), -1, axis)


def softmax(z, axis: int = -1) -> tf.Tensor:
    return tf.exp(log_softmax(z, axis))


def log_prob_of(z, target) -> tf.Tensor:
    """log_softmax(z)[..., target] over the last dimension, for integer ``target`` of shape
    ``z.shape[:-1]``."""
    z = tf.convert_to_tensor(z)
    is_t = tf.one_hot(target, tf.shape(z)[-1], on_value=True, off_value=False, dtype=tf.bool)
    z_t = _pick(z, is_t)                                               # target logit
    others = tf.where(is_t, tf.fill(tf.shape(z), tf.constant(float("-inf"), z.dtype)), z - z_t)
    return -tf.nn.softplus(tf.reduce_logsumexp(others, axis=-1))       # target left out of the sum


def cross_entropy(logits, target, ignore_index: int = -100, reduction: str = "mean") -> tf.Tensor:
    """Hard-target cross-entropy ``-log_prob_of(logits, target)``. ``logits`` is ``[..., C]`` and
    ``target`` is ``[...]`` (integer class indices). Rows whose target equals ``ignore_index``
    contribute 0 and are left out of the ``"mean"`` denominator. ``reduction`` is ``"none"`` (shape
    ``target.shape``), ``"sum"`` or ``"mean"``."""
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"reduction={reduction!r}: use none, sum or mean")
    logits = tf.convert_to_tensor(logits)
    target = tf.convert_to_tensor(target)
    valid = tf.not_equal(target, ignore_index)
    safe_target = tf.where(valid, target, tf.zeros_like(target))       # any in-range index for ignored rows
    nll = -log_prob_of(logits, safe_target)
    loss = tf.where(valid, nll, tf.zeros_like(nll))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return tf.reduce_sum(loss)
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(tf.cast(valid, tf.int32)), loss.dtype)  # nan if every row is ignored
