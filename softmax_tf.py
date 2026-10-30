"""log-softmax, softmax and cross-entropy without Softmax Collapse, in TensorFlow (depends only on
tensorflow).

Reference: J. T. Barron, D. Gnanapragasam and M. Tschannen, "A Fix for Softmax and Cross-Entropy",
arXiv 2610.31642, Section 2.

Same math as ``softmax_torch.py`` (PyTorch); see that file for the derivation:

    s_k = 1[k != argmax] (z_k - z_max),   m = -LSE_{j != argmax} s_j,   log_softmax_k = s_k - softplus(-m)

The argmax slot is an exact 0 and the argmax logit enters only through ``z_j - z_max``, so autodiff
gives it minus the sum of the other gradients. Works eagerly and inside ``tf.function``.

Usage:

    import tensorflow as tf
    from softmax_tf import log_softmax, softmax, log_prob_of, cross_entropy

    logits = tf.random.normal((8, 32000))
    target = tf.random.uniform((8,), 0, 32000, dtype=tf.int64)
    loss = cross_entropy(logits, target)            # logits [..., C], target [...]
    lp = log_prob_of(logits, target)                # log p(target) per row
    lp_all = log_softmax(logits, axis=-1)           # same shapes as tf.nn.log_softmax

Run ``python softmax_tf.py`` for a demo of the difference on a large-margin row.
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


if __name__ == "__main__":
    z = tf.constant([[20.0, 0.0, 0.0, 0.0]])          # the target leads by 20 > 16.6
    t = tf.constant([0])
    for name, dtype in (("float32", tf.float32), ("bfloat16", tf.bfloat16)):
        print(f"{name}: logits {z.numpy().tolist()[0]}, target 0")
        for label, f in (("tf.nn.log_softmax", lambda x: tf.nn.log_softmax(x)[0, 0]),
                         ("softmax_tf       ", lambda x: log_prob_of(x, t)[0])):
            x = tf.Variable(tf.cast(z, dtype))
            with tf.GradientTape() as tape:
                lp = f(x)
            g = tf.cast(tape.gradient(lp, x), tf.float32)
            print(f"  {label} log p = {float(lp):.6g}   grad = {[float(f'{v:.4g}') for v in g[0].numpy().tolist()]}")
