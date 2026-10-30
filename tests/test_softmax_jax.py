import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")  # 4 CPU devices for shard_map

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softmax_jax import cross_entropy, log_prob_of, log_softmax, softmax
from vocab_parallel_jax import vocab_parallel_log_prob_of

try:
    from jax import shard_map
except ImportError:                                   # older jax
    from jax.experimental.shard_map import shard_map

DTYPES = [jnp.float32, jnp.bfloat16, jnp.float16]
COLLAPSE_MARGIN = {jnp.float32: 20.0, jnp.bfloat16: 8.0, jnp.float16: 10.0}


def _key(i=0):
    return jax.random.key(i)


def _collapse_row(dtype, c=8):
    return jnp.zeros((1, c)).at[0, 0].set(COLLAPSE_MARGIN[dtype]).astype(dtype)


def _close(a, b, rtol=1e-7, atol=1e-12):
    a, b = jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64)
    assert jnp.allclose(a, b, rtol=rtol, atol=atol), (a, b)


# ── agreement with jax.nn on ordinary inputs ───────────────────────────────────────────────────

@pytest.mark.parametrize("axis", [0, 1, -1])
def test_log_softmax_matches_jax(axis):
    z = jax.random.normal(_key(), (5, 7, 11), jnp.float64) * 3
    _close(log_softmax(z, axis), jax.nn.log_softmax(z, axis))
    _close(softmax(z, axis), jax.nn.softmax(z, axis))


def test_log_prob_of_matches_jax():
    z = jax.random.normal(_key(), (4, 6, 50), jnp.float64) * 4
    t = jax.random.randint(_key(1), (4, 6), 0, 50)
    ref = jnp.take_along_axis(jax.nn.log_softmax(z, -1), t[..., None], -1)[..., 0]
    _close(log_prob_of(z, t), ref)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_matches_reference(reduction):
    z = jax.random.normal(_key(), (32, 100), jnp.float64) * 3
    t = jax.random.randint(_key(1), (32,), 0, 100).at[::5].set(-100)
    valid = t != -100
    nll = -jnp.take_along_axis(jax.nn.log_softmax(z, -1), jnp.where(valid, t, 0)[:, None], -1)[:, 0]
    nll = jnp.where(valid, nll, 0)
    ref = {"none": nll, "sum": nll.sum(), "mean": nll.sum() / valid.sum()}[reduction]
    _close(cross_entropy(z, t, reduction=reduction), ref)


def test_cross_entropy_bad_reduction():
    with pytest.raises(ValueError):
        cross_entropy(jnp.zeros((2, 3)), jnp.zeros(2, jnp.int32), reduction="avg")


def test_grad_matches_jax_and_jit():
    z = jax.random.normal(_key(), (3, 9), jnp.float64) * 5
    t = jax.random.randint(_key(1), (3,), 0, 9)
    ref = jax.grad(lambda x: jnp.take_along_axis(jax.nn.log_softmax(x), t[:, None], -1).sum())(z)
    _close(jax.jit(jax.grad(lambda x: log_prob_of(x, t).sum()))(z), ref)
    _close(jax.jit(jax.grad(lambda x: jnp.take_along_axis(log_softmax(x), t[:, None], -1).sum()))(z), ref)
    _close(jax.jit(jax.grad(lambda x: -cross_entropy(x, t, reduction="sum")))(z), ref)


# ── Softmax Collapse ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_jax_nn_collapses(dtype):
    """Sanity check of the bug itself: past the margin, jax.nn returns log p = 0 and target grad 0."""
    lp, g = jax.value_and_grad(lambda x: jax.nn.log_softmax(x, -1)[0, 0])(_collapse_row(dtype))
    assert float(lp) == 0.0
    assert float(g[0, 0]) == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_collapse(dtype):
    z = _collapse_row(dtype)
    ref = jax.nn.log_softmax(z.astype(jnp.float64), -1)[0, 0]
    lp, g = jax.value_and_grad(lambda x: log_prob_of(x, jnp.array([0]))[0])(z)
    assert float(lp) < 0
    _close(lp, ref, rtol=2e-2, atol=0)
    assert float(g[0, 0]) > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", ["log_softmax", "log_prob_of"])
def test_row_gradient_sums_to_zero(dtype, fn):
    z = (jax.random.normal(_key(), (16, 64)) * 2).at[:, 0].add(jnp.linspace(0, 40, 16)).astype(dtype)
    t = jnp.zeros(16, jnp.int32)
    f = (lambda x: log_softmax(x, -1)[:, 0].sum()) if fn == "log_softmax" else (lambda x: log_prob_of(x, t).sum())
    g = jax.grad(f)(z).astype(jnp.float64)
    scale = jnp.maximum(jnp.abs(g).max(-1), 1e-30)
    assert bool((jnp.abs(g.sum(-1)) / scale < 0.05).all())


@pytest.mark.parametrize("dtype,rtol", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_log_softmax_large_margin_matches_float64(dtype, rtol):
    z = jnp.array([[30.0, 1.0, 0.0, -2.0, 5.0]])
    _close(log_softmax(z.astype(dtype), -1), jax.nn.log_softmax(z.astype(jnp.float64), -1), rtol=rtol, atol=1e-30)


# ── vocab-parallel version, 4 shards ───────────────────────────────────────────────────────────

@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices (XLA_FLAGS host device count)")
@pytest.mark.parametrize("dtype,rtol", [(jnp.float64, 1e-10), (jnp.float32, 1e-5)])
def test_vocab_parallel_sharded_matches(dtype, rtol):
    mesh = jax.make_mesh((4,), ("tp",), axis_types=(AxisType.Auto,))
    z = (jax.random.normal(_key(), (6, 40), jnp.float64) * 5).at[0, 3].add(40).astype(dtype)  # row 0: big margin
    t = jax.random.randint(_key(1), (6,), 0, 40).at[0].set(3)
    sharded = shard_map(lambda l, tt: vocab_parallel_log_prob_of(l, tt, "tp"),
                        mesh=mesh, in_specs=(P(None, "tp"), P()), out_specs=P())
    z_s = jax.device_put(z, NamedSharding(mesh, P(None, "tp")))
    t_s = jax.device_put(t, NamedSharding(mesh, P()))
    _close(sharded(z_s, t_s), log_prob_of(z, t), rtol=rtol, atol=1e-30)
    g_ref = jax.grad(lambda x: log_prob_of(x, t).sum())(z)
    g = jax.jit(jax.grad(lambda x: sharded(x, t_s).sum()))(z_s)
    _close(g, g_ref, rtol=rtol, atol=1e-30)
