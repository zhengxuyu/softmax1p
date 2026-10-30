import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softmax_tf import cross_entropy, log_prob_of, log_softmax, softmax

DTYPES = [tf.float32, tf.bfloat16, tf.float16]
COLLAPSE_MARGIN = {tf.float32: 20.0, tf.bfloat16: 8.0, tf.float16: 10.0}
RNG = np.random.default_rng(0)


def _collapse_row(dtype, c=8):
    z = np.zeros((1, c))
    z[0, 0] = COLLAPSE_MARGIN[dtype]
    return tf.constant(z, dtype)


def _close(a, b, rtol=1e-7, atol=1e-12):
    np.testing.assert_allclose(np.asarray(tf.cast(a, tf.float64)), np.asarray(tf.cast(b, tf.float64)), rtol=rtol, atol=atol)


def _value_and_grad(f, z):
    x = tf.Variable(z)
    with tf.GradientTape() as tape:
        y = f(x)
    return y, tape.gradient(y, x)


# ── agreement with tf.nn on ordinary inputs ────────────────────────────────────────────────────

@pytest.mark.parametrize("axis", [0, 1, -1])
def test_log_softmax_matches_tf(axis):
    z = tf.constant(RNG.normal(size=(5, 7, 11)) * 3)
    _close(log_softmax(z, axis), tf.nn.log_softmax(z, axis))
    _close(softmax(z, axis), tf.nn.softmax(z, axis))


def test_log_prob_of_matches_tf():
    z = tf.constant(RNG.normal(size=(4, 6, 50)) * 4)
    t = tf.constant(RNG.integers(0, 50, (4, 6)))
    _close(log_prob_of(z, t), -tf.nn.sparse_softmax_cross_entropy_with_logits(t, z))


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_matches_tf(reduction):
    z = tf.constant(RNG.normal(size=(32, 100)) * 3)
    t = RNG.integers(0, 100, (32,))
    t[::5] = -100
    valid = t != -100
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(np.where(valid, t, 0), z)
    nll = tf.where(valid, nll, tf.zeros_like(nll))
    ref = {"none": nll, "sum": tf.reduce_sum(nll), "mean": tf.reduce_sum(nll) / valid.sum()}[reduction]
    _close(cross_entropy(z, tf.constant(t), reduction=reduction), ref)


def test_cross_entropy_bad_reduction():
    with pytest.raises(ValueError):
        cross_entropy(tf.zeros((2, 3)), tf.zeros(2, tf.int32), reduction="avg")


def test_gradient_check():
    z = tf.constant(RNG.normal(size=(3, 9)) * 5)
    t = tf.constant(RNG.integers(0, 9, (3,)))
    for f in (lambda x: log_softmax(x, -1), lambda x: log_prob_of(x, t), lambda x: cross_entropy(x, t)):
        theoretical, numerical = tf.test.compute_gradient(f, [z])
        np.testing.assert_allclose(theoretical[0], numerical[0], rtol=1e-6, atol=1e-8)


def test_tf_function():
    z = tf.constant(RNG.normal(size=(4, 10)))
    t = tf.constant(RNG.integers(0, 10, (4,)))
    _close(tf.function(cross_entropy)(z, t), cross_entropy(z, t))
    _close(tf.function(log_softmax)(z), tf.nn.log_softmax(z))


# ── Softmax Collapse ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_tf_nn_collapses(dtype):
    """Sanity check of the bug itself: past the margin, tf.nn returns log p = 0 and target grad 0."""
    lp, g = _value_and_grad(lambda x: tf.nn.log_softmax(x)[0, 0], _collapse_row(dtype))
    assert float(lp) == 0.0
    assert float(g[0, 0]) == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_collapse(dtype):
    z = _collapse_row(dtype)
    ref = tf.nn.log_softmax(tf.cast(z, tf.float64))[0, 0]
    lp, g = _value_and_grad(lambda x: log_prob_of(x, tf.constant([0]))[0], z)
    assert float(lp) < 0
    _close(lp, ref, rtol=2e-2, atol=0)
    assert float(g[0, 0]) > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", ["log_softmax", "log_prob_of"])
def test_row_gradient_sums_to_zero(dtype, fn):
    z = RNG.normal(size=(16, 64)) * 2
    z[:, 0] += np.linspace(0, 40, 16)
    t = tf.zeros(16, tf.int32)
    f = (lambda x: tf.reduce_sum(log_softmax(x, -1)[:, 0])) if fn == "log_softmax" else (lambda x: tf.reduce_sum(log_prob_of(x, t)))
    _, g = _value_and_grad(f, tf.constant(z, dtype))
    g = np.asarray(tf.cast(g, tf.float64))
    scale = np.maximum(np.abs(g).max(-1), 1e-30)
    assert (np.abs(g.sum(-1)) / scale < 0.05).all()


@pytest.mark.parametrize("dtype,rtol", [(tf.float32, 1e-4), (tf.bfloat16, 5e-2)])
def test_log_softmax_large_margin_matches_float64(dtype, rtol):
    z = tf.constant([[30.0, 1.0, 0.0, -2.0, 5.0]], tf.float64)
    _close(log_softmax(tf.cast(z, dtype), -1), tf.nn.log_softmax(z), rtol=rtol, atol=1e-30)
