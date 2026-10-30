# Tensor-parallel heads

Large language models often split the output projection over the vocabulary, so each device holds logits for a contiguous slice of classes. `vocab_parallel_log_prob_of` computes `log p(target)` from those shards without gathering the full vocabulary. It uses the same collapse-free formula as `log_prob_of`.

Both versions assume **equal shards in order**: shard `i` of `n` holds classes `[i * C / n, (i + 1) * C / n)`. Targets are **global** class indices and must be the same on every shard. The result is identical on every shard.

=== "PyTorch (torch.distributed)"

    ```python
    import torch.distributed as dist
    from vocab_parallel_torch import vocab_parallel_log_prob_of

    # logits_shard: [B, T, V / world] on this rank; target: [B, T] global ids, same on every rank
    logp = vocab_parallel_log_prob_of(logits_shard, target, group=tp_group)
    loss = -logp.mean()
    loss.backward()        # gradient flows to this rank's shard only
    ```

    - Rows are processed `chunk` at a time (default 1024). The backward pass recomputes the softmax for each chunk, so memory stays bounded for large vocabularies.
    - `group=None`, or an uninitialised `torch.distributed`, treats the logits as unsharded, which is handy for single-GPU tests.
    - Every rank must call it with the same `target` and `chunk`.

=== "JAX (shard_map)"

    ```python
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from vocab_parallel_jax import vocab_parallel_log_prob_of

    mesh = jax.make_mesh((4,), ("tp",))
    log_prob = jax.shard_map(
        lambda l, t: vocab_parallel_log_prob_of(l, t, "tp"),
        mesh=mesh, in_specs=(P(None, "tp"), P()), out_specs=P(),
    )

    logits = jax.device_put(logits, NamedSharding(mesh, P(None, "tp")))   # [N, V] split over V
    logp = log_prob(logits, target)                                        # [N]
    ```

    The function communicates with `psum` and `pmax`, and works under `jit` and `grad`.

The JAX test suite checks that the sharded result matches the unsharded `log_prob_of` across 4 shards, for both values and gradients.
