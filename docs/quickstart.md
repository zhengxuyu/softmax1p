# Quickstart

All three `softmax_*` modules expose the same four functions:

| Function | What it computes |
|---|---|
| `log_softmax(z, dim/axis=-1)` | Log-softmax along a dimension. Same shapes as the built-in. |
| `softmax(z, dim/axis=-1)` | `exp(log_softmax(z))` |
| `log_prob_of(z, target)` | `log_softmax(z)[..., target]` over the last dimension, without forming the full log-softmax |
| `cross_entropy(logits, target, ignore_index=-100, reduction="mean")` | Hard-target cross-entropy, `-log_prob_of(logits, target)` |

!!! warning "Class dimension is last"
    `log_prob_of` and `cross_entropy` take logits `[..., C]` and targets `[...]`. This differs from `torch.nn.functional.cross_entropy`, which puts classes in dimension 1.

=== "PyTorch"

    ```python
    import torch
    from softmax_torch import log_softmax, softmax, log_prob_of, cross_entropy

    logits = torch.randn(8, 32000, requires_grad=True)
    target = torch.randint(0, 32000, (8,))

    loss   = cross_entropy(logits, target)   # scalar, mean over rows
    lp     = log_prob_of(logits, target)     # [8], log p(target) per row
    lp_all = log_softmax(logits, dim=-1)     # [8, 32000]
    probs  = softmax(logits, dim=-1)         # [8, 32000]

    loss.backward()
    ```

=== "JAX"

    ```python
    import jax
    from softmax_jax import log_softmax, softmax, log_prob_of, cross_entropy

    logits = jax.random.normal(jax.random.key(0), (8, 32000))
    target = jax.random.randint(jax.random.key(1), (8,), 0, 32000)

    loss   = jax.jit(cross_entropy)(logits, target)
    grads  = jax.grad(cross_entropy)(logits, target)
    lp_all = log_softmax(logits, axis=-1)
    ```

=== "TensorFlow"

    ```python
    import tensorflow as tf
    from softmax_tf import log_softmax, softmax, log_prob_of, cross_entropy

    logits = tf.Variable(tf.random.normal((8, 32000)))
    target = tf.random.uniform((8,), 0, 32000, dtype=tf.int64)

    with tf.GradientTape() as tape:
        loss = cross_entropy(logits, target)   # also works inside tf.function
    grads = tape.gradient(loss, logits)
    ```

## Language-model loss

For a language model, flatten nothing. Pass logits `[batch, seq, vocab]` and token ids `[batch, seq]`, and mark padding with `ignore_index`:

```python
from softmax_torch import cross_entropy

logits = model(input_ids)                     # [B, T, V]
labels = input_ids[:, 1:].masked_fill(pad_mask[:, 1:], -100)
loss = cross_entropy(logits[:, :-1], labels)  # padding rows are left out of the mean
```

## Policy gradients

`log_prob_of` returns `log p(action)` per row directly:

```python
from softmax_torch import log_prob_of

logp = log_prob_of(policy_logits, actions)    # [B]
loss = -(advantages * logp).mean()
```
