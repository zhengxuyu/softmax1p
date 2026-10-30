# softmax1p

This repository is a PyTorch, JAX and TensorFlow implementation of the improvement proposed in:

> Jonathan T. Barron, Dan Gnanapragasam, Michael Tschannen. **[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)**. arXiv:2610.31642, 2026.

It provides drop-in `log_softmax`, `softmax` and `cross_entropy` that do not suffer from **Softmax Collapse**. There is one self-contained file per framework, and each depends only on that framework:

| Framework | File | Built-in it replaces |
|---|---|---|
| PyTorch | [`softmax_torch.py`](softmax_torch.py) | `torch.log_softmax`, `F.cross_entropy` |
| JAX | [`softmax_jax.py`](softmax_jax.py) | `jax.nn.log_softmax`, `optax.softmax_cross_entropy_with_integer_labels` |
| TensorFlow | [`softmax_tf.py`](softmax_tf.py) | `tf.nn.log_softmax`, `tf.nn.sparse_softmax_cross_entropy_with_logits` |

📝 Blog post with a full walkthrough of the paper (in Chinese): **[深度理解 "A Fix for Softmax and Cross-Entropy"](https://zhengxuyu.github.io/blog/softmax-cross-entropy-fix)**

## The problem

The standard log-softmax subtracts the largest logit and then computes `log(1 + Σ_{j≠max} e^{-(z_max - z_j)})`. Once the largest logit leads the others by a margin `m` where `e^{-m}` falls below half an ulp of 1, the sum `1 + e^{-m}` rounds to exactly `1`. This happens at:

| dtype | collapse margin |
|---|---:|
| bfloat16 | ≈ 5.5 |
| float16 | ≈ 7.6 |
| float32 | ≈ 16.6 |
| float64 | ≈ 36.7 |

Once this happens, the target's log-probability becomes exactly `0` and its gradient becomes exactly `0`, while the other classes keep nonzero gradients. The row's gradient no longer sums to zero. With Adam, this bias is amplified into slingshot loss spikes and parameter drift; see the blog post and paper for details.

## The fix

The fix never forms `1 + e^{-m}`. Instead it sets the argmax (or target) slot to an exact `0`, takes the log-sum-exp over the *other* slots only, and joins the two with `softplus`:

```
s_k = 1[k ≠ argmax] · (z_k − z_max)
log_softmax_k = s_k − softplus( LSE_{j ≠ argmax} s_j )
log p(t)      = −softplus( LSE_{j ≠ t} (z_j − z_t) )
```

The kept logit enters the result only through differences. Autodiff therefore gives it exactly minus the sum of the other gradients, so the row gradient sums to zero by construction.

## Install

```bash
pip install softmax1p             # no framework pulled in
pip install "softmax1p[torch]"    # or [jax], [tf]
```

Or copy the single file for your framework into your project; nothing else is needed. Documentation: **<https://zhengxuyu.github.io/softmax1p/>**

## Usage

All three files expose the same functions: `log_softmax`, `softmax`, `log_prob_of` and `cross_entropy`. The class dimension is the **last** one for `log_prob_of` and `cross_entropy`.

### PyTorch

```python
import torch
from softmax_torch import log_softmax, softmax, log_prob_of, cross_entropy

logits = torch.randn(8, 32000, requires_grad=True)
target = torch.randint(0, 32000, (8,))

loss  = cross_entropy(logits, target)        # class dim LAST: logits [..., C], target [...]
lp    = log_prob_of(logits, target)          # log p(target) per row (e.g. for policy gradients)
lp_all = log_softmax(logits, dim=-1)         # same signature as torch.log_softmax
```

- `cross_entropy` supports `ignore_index` (default `-100`) and `reduction` (`"none" | "sum" | "mean"`). Unlike `F.cross_entropy`, it expects the class dimension **last**.
- [`vocab_parallel_torch.py`](vocab_parallel_torch.py) (optional, separate file): `vocab_parallel_log_prob_of(logits_shard, target, group)` handles logits sharded over the vocabulary across a `torch.distributed` group, as in tensor-parallel LM heads. It is chunked over rows and recomputes the softmax in backward to bound memory.

### JAX

```python
import jax, jax.numpy as jnp
from softmax_jax import log_softmax, softmax, log_prob_of, cross_entropy

logits = jax.random.normal(jax.random.key(0), (8, 32000))
target = jax.random.randint(jax.random.key(1), (8,), 0, 32000)

loss = jax.jit(cross_entropy)(logits, target)   # jit / vmap / grad all work
lp_all = log_softmax(logits, axis=-1)
```

- [`vocab_parallel_jax.py`](vocab_parallel_jax.py) (optional, separate file): `vocab_parallel_log_prob_of(logits_shard, target, axis_name)` is the tensor-parallel version. Call it inside `jax.shard_map` with the vocabulary split along mesh axis `axis_name`; it communicates with `psum` / `pmax`.

### TensorFlow

```python
import tensorflow as tf
from softmax_tf import log_softmax, softmax, log_prob_of, cross_entropy

logits = tf.random.normal((8, 32000))
target = tf.random.uniform((8,), 0, 32000, dtype=tf.int64)

loss = cross_entropy(logits, target)            # eager or inside tf.function
lp_all = log_softmax(logits, axis=-1)
```

## Test results

Environment: Python 3.12.12, PyTorch 2.15.0, JAX 0.12.0.1, TensorFlow 2.21.0, macOS (CPU). The JAX sharding test uses 4 virtual CPU devices.

```bash
pip install pytest torch jax tensorflow   # each test file is skipped if its framework is missing
python -m pytest tests
```

| Test file | Result |
|---|---|
| `tests/test_softmax_torch.py` | **27 passed** |
| `tests/test_softmax_jax.py` | **25 passed** |
| `tests/test_softmax_tf.py` | **24 passed** |
| **Total** | **76 passed**, 0 failed |

The test suites have the same structure in all three frameworks:

| Test | Checks |
|---|---|
| `*_matches_{torch,jax,tf}` | Matches the framework's own `log_softmax` / `softmax` / sparse cross-entropy (incl. `ignore_index`, all reductions) on ordinary float64 inputs |
| gradient check | `torch.autograd.gradcheck`; `jax.grad` under `jit` vs `jax.nn`; `tf.test.compute_gradient` (analytic vs numerical Jacobian) |
| `test_*_collapses` | Reproduces the bug: past the collapse margin, the built-in `log_softmax` returns `log p = 0` and target gradient `0` (fp32 / bf16 / fp16, in all three frameworks) |
| `test_no_collapse` | On the same inputs, `log_prob_of` returns `log p < 0` matching float64 and a nonzero target gradient |
| `test_row_gradient_sums_to_zero` | Margins from 0 to 40 in fp32 / bf16 / fp16: each row's gradient sums to ≈ 0 |
| `test_tf_function` | TF only: identical results inside `tf.function` |
| `test_vocab_parallel_*` | PyTorch: same values and gradients as `log_prob_of` (unsharded path). JAX: logits split over **4 shards** with `shard_map`; values and gradients match the unsharded version |

### Same bug, same fix, in every framework

`python benchmarks/collapse_all_frameworks.py` evaluates the row `[m, 0, 0, 0, 0, 0, 0, 0]` with target 0, with `m` past the collapse margin. It reports `log p(target)`, its gradient with respect to the target logit, and the row's gradient sum (exactly 0 in exact arithmetic):

| dtype | margin | framework | implementation | log p | ∂ log p / ∂z_t | Σ grad |
|---|---:|---|---|---:|---:|---:|
| float32 | 20 | *exact* | | -1.4428e-08 | 1.4428e-08 | 0 |
| float32 | 20 | PyTorch | `torch.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | PyTorch | `softmax_torch` | -1.4428e-08 | 1.4428e-08 | 6.7e-16 |
| float32 | 20 | JAX | `jax.nn.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | JAX | `softmax_jax` | -1.4428e-08 | 1.4428e-08 | 4.4e-16 |
| float32 | 20 | TensorFlow | `tf.nn.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | TensorFlow | `softmax_tf` | -1.4428e-08 | 1.4428e-08 | 2.2e-16 |
| bfloat16 | 8 | *exact* | | -2.3455e-03 | 2.3427e-03 | 0 |
| bfloat16 | 8 | PyTorch | `torch.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | PyTorch | `softmax_torch` | -2.3193e-03 | 2.3499e-03 | 0 |
| bfloat16 | 8 | JAX | `jax.nn.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | JAX | `softmax_jax` | -2.3346e-03 | 2.3346e-03 | -1.9e-06 |
| bfloat16 | 8 | TensorFlow | `tf.nn.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | TensorFlow | `softmax_tf` | -2.3346e-03 | 2.3346e-03 | -1.9e-06 |

All three built-ins collapse in the same way: the target gets exactly 0 gradient, and the row is left with a net bias equal to the full non-target gradient. The fixed versions in all three frameworks recover the exact value to within the dtype's precision, and they reduce the row-sum error by 7–8 orders of magnitude in fp32 and 3 in bf16.

### Margin sweep (PyTorch)

`python benchmarks/margin_sweep.py` uses the same row across more margins and dtypes and compares against a float64 reference:

| dtype | margin | ref log p (fp64) | torch log p | fixed log p | torch ∂/∂z_t | fixed ∂/∂z_t | torch Σ grad | fixed Σ grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float16 | 4 | -1.206e-01 | -1.204e-01 | -1.205e-01 | 1.134e-01 | 1.136e-01 | -1.4e-04 | 0.0e+00 |
| float16 | 8 | -2.345e-03 | -1.951e-03 | -2.344e-03 | 1.949e-03 | 2.342e-03 | -4.0e-04 | -9.5e-07 |
| float16 | 12 | -4.301e-05 | **0.000e+00** | -4.297e-05 | **0.000e+00** | 4.297e-05 | -4.3e-05 | 0.0e+00 |
| float16 | 20 | -1.443e-08 | 0.000e+00 | -0.000e+00 | 0.000e+00 | 0.000e+00 | 0.0e+00 | 0.0e+00 |
| bfloat16 | 4 | -1.206e-01 | -1.177e-01 | -1.196e-01 | 1.108e-01 | 1.143e-01 | -2.0e-03 | -2.4e-04 |
| bfloat16 | 8 | -2.345e-03 | **0.000e+00** | -2.319e-03 | **0.000e+00** | 2.350e-03 | -2.3e-03 | 0.0e+00 |
| bfloat16 | 12 | -4.301e-05 | **0.000e+00** | -4.268e-05 | **0.000e+00** | 4.315e-05 | -4.3e-05 | -3.0e-08 |
| bfloat16 | 20 | -1.443e-08 | **0.000e+00** | -1.525e-08 | **0.000e+00** | 1.444e-08 | -1.4e-08 | -2.9e-11 |
| bfloat16 | 30 | -6.548e-13 | **0.000e+00** | -6.928e-13 | **0.000e+00** | 6.608e-13 | -6.6e-13 | 1.8e-15 |
| bfloat16 | 40 | 0.000e+00 | **0.000e+00** | -3.144e-17 | **0.000e+00** | 2.971e-17 | -3.0e-17 | -8.1e-20 |
| float32 | 4 | -1.206e-01 | -1.206e-01 | -1.206e-01 | 1.136e-01 | 1.136e-01 | -2.2e-08 | 5.6e-09 |
| float32 | 8 | -2.345e-03 | -2.345e-03 | -2.345e-03 | 2.343e-03 | 2.343e-03 | -3.7e-08 | 1.5e-10 |
| float32 | 12 | -4.301e-05 | -4.303e-05 | -4.301e-05 | 4.303e-05 | 4.301e-05 | 2.7e-08 | 5.5e-12 |
| float32 | 20 | -1.443e-08 | **0.000e+00** | -1.443e-08 | **0.000e+00** | 1.443e-08 | -1.4e-08 | 6.7e-16 |
| float32 | 30 | -6.548e-13 | **0.000e+00** | -6.550e-13 | **0.000e+00** | 6.550e-13 | -6.6e-13 | 4.1e-20 |
| float32 | 40 | 0.000e+00 | **0.000e+00** | -2.974e-17 | **0.000e+00** | 2.974e-17 | -3.0e-17 | 4.1e-25 |

How to read this table:

- **Bold** cells show where `torch.log_softmax` has collapsed: `log p = 0` and target gradient `0`. The "torch Σ grad" column then equals minus the sum of the other classes' gradients, so the row has a net bias. The fixed version keeps a nonzero target gradient, and its row gradient sums to roughly 0, which is 10³–10¹⁰× closer to zero than torch's.
- In float32 at margin 40, the float64 reference itself collapses to `0`, because 40 is past fp64's cliff of 36.7. The fixed version in float32 still returns the correct `-2.97e-17`.
- float16 at margin ≥ 20 is a **range** limit, not collapse: `e^{-20} ≈ 2e-9` is below fp16's smallest subnormal (6e-8), so every term underflows to 0 on both sides. The fix removes the rounding cliff, but it cannot represent numbers fp16 cannot hold. The remaining float16 rows (30 and 40) are omitted above for that reason.

## Related work

Discussed in the blog post:

- Prieto et al., *Grokking at the Edge of Numerical Stability* (StableMax), ICLR 2025.
- Thilak et al., *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon*, arXiv:2206.04817, 2022.
- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177, 2022.

## License

MIT. This is an independent implementation and is not affiliated with the paper's authors.
