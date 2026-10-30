# Migrating from built-ins

On inputs that do not collapse, every function matches the framework built-in to within floating-point rounding. The test suites check this in float64.

## PyTorch

| Built-in | softmax1p | Notes |
|---|---|---|
| `torch.log_softmax(z, dim)` | `log_softmax(z, dim)` | identical signature |
| `torch.softmax(z, dim)` | `softmax(z, dim)` | identical signature |
| `F.cross_entropy(logits, target)` | `cross_entropy(logits, target)` | **class dim last**; see below |
| `torch.log_softmax(z, -1).gather(-1, t[..., None])[..., 0]` | `log_prob_of(z, t)` | no full log-softmax is formed |

`F.cross_entropy` expects classes in dimension 1 (`[N, C, d1, ...]`). `cross_entropy` expects them last (`[..., C]`):

```python
# before
loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)   # logits [B, T, V]
# after
loss = cross_entropy(logits, labels, ignore_index=-100)
```

Supported keyword arguments are `ignore_index` and `reduction` (`"none"`, `"sum"`, `"mean"`). `weight` and `label_smoothing` are not supported.

## JAX

| Built-in | softmax1p |
|---|---|
| `jax.nn.log_softmax(z, axis)` | `log_softmax(z, axis)` |
| `jax.nn.softmax(z, axis)` | `softmax(z, axis)` |
| `optax.softmax_cross_entropy_with_integer_labels(logits, labels)` | `cross_entropy(logits, labels, reduction="none")` |

The default `reduction` is `"mean"`, whereas optax returns per-example losses.

## TensorFlow

| Built-in | softmax1p |
|---|---|
| `tf.nn.log_softmax(z, axis)` | `log_softmax(z, axis)` |
| `tf.nn.softmax(z, axis)` | `softmax(z, axis)` |
| `tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)` | `cross_entropy(logits, labels, reduction="none")` |

Note the argument order: softmax1p takes `logits` first.
