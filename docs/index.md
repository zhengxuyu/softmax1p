# softmax1p

Drop-in `log_softmax`, `softmax` and `cross_entropy` for **PyTorch**, **JAX** and **TensorFlow** that do not suffer from **Softmax Collapse**.

It implements the fix from Jonathan T. Barron, Dan Gnanapragasam and Michael Tschannen, [*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642) (arXiv:2610.31642, 2026).

```bash
pip install "softmax1p[torch]"    # or [jax], [tf]
```

```python
from softmax_torch import cross_entropy

loss = cross_entropy(logits, target)   # logits [..., C], target [...]
```

## Why

Once the top logit leads the others by more than about **16.6 in float32** (5.5 in bfloat16, 7.6 in float16), the standard log-softmax rounds `1 + e^{-m}` to exactly 1. The target's log-probability and gradient then become exactly 0, and the row's gradient no longer sums to zero. With Adam, that bias turns into loss spikes and parameter drift. See [Softmax Collapse and the fix](guide/background.md).

## Package contents

Each module is one self-contained file that depends only on its own framework. Install the package, or copy the single file you need into your project.

| Module | Framework | Exports | Replaces |
|---|---|---|---|
| [`softmax_torch`](api/softmax_torch.md) | PyTorch | `log_softmax`, `softmax`, `log_prob_of`, `cross_entropy` | `torch.log_softmax`, `F.cross_entropy` |
| [`softmax_jax`](api/softmax_jax.md) | JAX | same four | `jax.nn.log_softmax`, `optax.softmax_cross_entropy_with_integer_labels` |
| [`softmax_tf`](api/softmax_tf.md) | TensorFlow | same four | `tf.nn.log_softmax`, `tf.nn.sparse_softmax_cross_entropy_with_logits` |
| [`vocab_parallel_torch`](api/vocab_parallel_torch.md) | PyTorch | `vocab_parallel_log_prob_of` | tensor-parallel LM head over a `torch.distributed` group |
| [`vocab_parallel_jax`](api/vocab_parallel_jax.md) | JAX | `vocab_parallel_log_prob_of` | tensor-parallel LM head inside `jax.shard_map` |

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Migrating from built-ins](guide/migration.md)
- [API reference](api/softmax_torch.md)
