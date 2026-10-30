"""Compare each framework's built-in log-softmax with this repo's fix on one large-margin row.

The row is [m, 0, 0, 0, 0, 0, 0, 0] with target 0, with m past the collapse margin of the dtype
(20 for float32 > 16.6, 8 for bfloat16 > 5.5). Prints log p(target), d log p / d z_target and the
row's gradient sum (0 in exact arithmetic) for PyTorch, JAX and TensorFlow, whichever are installed.

    python benchmarks/collapse_all_frameworks.py
"""

import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C = 8
CASES = [("float32", 20.0), ("bfloat16", 8.0)]


def ref_log_p(m):
    return -math.log1p((C - 1) * math.exp(-m))


def row(m):
    return [[m] + [0.0] * (C - 1)]


def torch_runs(dtype_name, m):
    import torch
    from softmax_torch import log_prob_of
    dtype = getattr(torch, dtype_name)

    def run(f):
        x = torch.tensor(row(m), dtype=dtype, requires_grad=True)
        lp = f(x)
        lp.backward()
        g = x.grad.double()
        return lp.item(), g[0, 0].item(), g.sum().item()

    return [("PyTorch", "torch.log_softmax", run(lambda x: torch.log_softmax(x, -1)[0, 0])),
            ("PyTorch", "softmax_torch", run(lambda x: log_prob_of(x, torch.tensor([0]))[0]))]


def jax_runs(dtype_name, m):
    import jax
    import jax.numpy as jnp
    from softmax_jax import log_prob_of
    x = jnp.array(row(m), getattr(jnp, dtype_name))

    def run(f):
        lp, g = jax.value_and_grad(f)(x)
        g = g.astype(jnp.float32)
        return float(lp), float(g[0, 0]), float(g.sum())

    return [("JAX", "jax.nn.log_softmax", run(lambda x: jax.nn.log_softmax(x, -1)[0, 0])),
            ("JAX", "softmax_jax", run(lambda x: log_prob_of(x, jnp.array([0]))[0]))]


def tf_runs(dtype_name, m):
    import tensorflow as tf
    from softmax_tf import log_prob_of
    dtype = getattr(tf, dtype_name)

    def run(f):
        x = tf.Variable(tf.constant(row(m), dtype))
        with tf.GradientTape() as tape:
            lp = f(x)
        g = tf.cast(tape.gradient(lp, x), tf.float64)
        return float(lp), float(g[0, 0]), float(tf.reduce_sum(g))

    return [("TensorFlow", "tf.nn.log_softmax", run(lambda x: tf.nn.log_softmax(x)[0, 0])),
            ("TensorFlow", "softmax_tf", run(lambda x: log_prob_of(x, tf.constant([0]))[0]))]


def main():
    frameworks = [(mod, fn) for mod, fn in (("torch", torch_runs), ("jax", jax_runs), ("tensorflow", tf_runs))
                  if importlib.util.find_spec(mod)]
    print("| dtype | margin | framework | implementation | log p | ∂ log p / ∂z_t | Σ grad |")
    print("|---|---:|---|---|---:|---:|---:|")
    for dtype_name, m in CASES:
        r = ref_log_p(m)
        print(f"| {dtype_name} | {m:g} | *exact* | | {r:.4e} | {-math.expm1(r):.4e} | 0 |")
        for _, fn in frameworks:
            for fw, impl, (lp, g_t, g_sum) in fn(dtype_name, m):
                print(f"| {dtype_name} | {m:g} | {fw} | `{impl}` | {lp:.4e} | {g_t:.4e} | {g_sum:.1e} |")


if __name__ == "__main__":
    main()
