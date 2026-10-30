# softmax1p

Drop-in PyTorch `log_softmax`, `softmax` and `cross_entropy` that do not suffer from **Softmax Collapse**. This is a single file, [`safe_softmax.py`](safe_softmax.py), that depends only on `torch`.

It implements the fix from Barron, Gnanapragasam & Tschannen, [*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642) (arXiv:2610.31642, 2026).

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

## Usage

```python
import torch
from safe_softmax import log_softmax, softmax, log_prob_of, cross_entropy

logits = torch.randn(8, 32000, requires_grad=True)
target = torch.randint(0, 32000, (8,))

loss  = cross_entropy(logits, target)        # class dim LAST: logits [..., C], target [...]
lp    = log_prob_of(logits, target)          # log p(target) per row (e.g. for policy gradients)
lp_all = log_softmax(logits, dim=-1)         # same signature as torch.log_softmax
```

- `cross_entropy` supports `ignore_index` (default `-100`) and `reduction` (`"none" | "sum" | "mean"`). Unlike `F.cross_entropy`, it expects the class dimension **last**.
- `vocab_parallel_log_prob_of(logits_shard, target, group)` handles logits sharded over the vocabulary across a `torch.distributed` group, as in tensor-parallel LM heads. It is chunked over rows and recomputes the softmax in backward to bound memory.

## Test results

Environment: PyTorch 2.11.0, Python 3.12.12, macOS (CPU).

```bash
pip install torch pytest
python -m pytest -v tests
```

```
tests/test_safe_softmax.py::test_log_softmax_matches_torch[0] PASSED
tests/test_safe_softmax.py::test_log_softmax_matches_torch[1] PASSED
tests/test_safe_softmax.py::test_log_softmax_matches_torch[-1] PASSED
tests/test_safe_softmax.py::test_log_prob_of_matches_torch PASSED
tests/test_safe_softmax.py::test_cross_entropy_matches_torch[none] PASSED
tests/test_safe_softmax.py::test_cross_entropy_matches_torch[sum] PASSED
tests/test_safe_softmax.py::test_cross_entropy_matches_torch[mean] PASSED
tests/test_safe_softmax.py::test_cross_entropy_bad_reduction PASSED
tests/test_safe_softmax.py::test_gradcheck PASSED
tests/test_safe_softmax.py::test_torch_collapses[float32/bfloat16/float16] PASSED (x3)
tests/test_safe_softmax.py::test_no_collapse[float32/bfloat16/float16] PASSED (x3)
tests/test_safe_softmax.py::test_row_gradient_sums_to_zero[{log_softmax,log_prob_of,vocab_parallel} x 3 dtypes] PASSED (x9)
tests/test_safe_softmax.py::test_log_softmax_large_margin_matches_float64[float32/bfloat16] PASSED (x2)
tests/test_safe_softmax.py::test_vocab_parallel_unsharded_matches PASSED

============================== 27 passed in 2.66s ==============================
```

What the tests cover:

| Test | Checks |
|---|---|
| `*_matches_torch` | Matches `torch.log_softmax` / `softmax` / `F.cross_entropy` (incl. `ignore_index`, all reductions) on ordinary float64 inputs |
| `test_gradcheck` | `torch.autograd.gradcheck` on every function, including the custom-backward vocab-parallel path |
| `test_torch_collapses` | Reproduces the bug: past the collapse margin, `torch.log_softmax` returns `log p = 0` and target gradient `0` |
| `test_no_collapse` | On the same inputs, `log_prob_of` returns `log p < 0` matching float64 and a nonzero target gradient |
| `test_row_gradient_sums_to_zero` | Margins from 0 to 40 in fp32 / bf16 / fp16: each row's gradient sums to ≈ 0 |
| `test_vocab_parallel_unsharded_matches` | `vocab_parallel_log_prob_of` gives the same values and gradients as `log_prob_of` |

### Margin sweep

`python benchmarks/margin_sweep.py` uses the row `[m, 0, 0, 0, 0, 0, 0, 0]` with target 0. It compares the target's log-probability, its gradient, and the row's gradient sum against a float64 reference:

| dtype | margin | ref log p (fp64) | torch log p | safe log p | torch ∂/∂z_t | safe ∂/∂z_t | torch Σ grad | safe Σ grad |
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
| float32 | 8 | -2.345e-03 | -2.345e-03 | -2.345e-03 | 2.343e-03 | 2.343e-03 | -3.7e-08 | -8.7e-11 |
| float32 | 12 | -4.301e-05 | -4.303e-05 | -4.301e-05 | 4.303e-05 | 4.301e-05 | 2.7e-08 | 1.8e-12 |
| float32 | 20 | -1.443e-08 | **0.000e+00** | -1.443e-08 | **0.000e+00** | 1.443e-08 | -1.4e-08 | 6.7e-16 |
| float32 | 30 | -6.548e-13 | **0.000e+00** | -6.550e-13 | **0.000e+00** | 6.550e-13 | -6.6e-13 | 4.1e-20 |
| float32 | 40 | 0.000e+00 | **0.000e+00** | -2.974e-17 | **0.000e+00** | 2.974e-17 | -3.0e-17 | 4.1e-25 |

How to read this table:

- **Bold** cells show where `torch.log_softmax` has collapsed: `log p = 0` and target gradient `0`. The "torch Σ grad" column then equals minus the sum of the other classes' gradients, so the row has a net bias. `safe_softmax` keeps a nonzero target gradient, and its row gradient sums to roughly 0, which is 10³–10¹⁰× closer to zero than torch's.
- In float32 at margin 40, the float64 reference itself collapses to `0`, because 40 is past fp64's cliff of 36.7. `safe_softmax` in float32 still returns the correct `-2.97e-17`.
- float16 at margin ≥ 20 is a **range** limit, not collapse: `e^{-20} ≈ 2e-9` is below fp16's smallest subnormal (6e-8), so every term underflows to 0 on both sides. The fix removes the rounding cliff, but it cannot represent numbers fp16 cannot hold. The remaining float16 rows (30 and 40) are omitted above for that reason.

### Demo

`python safe_softmax.py`:

```
float32: logits [20.0, 0.0, 0.0, 0.0], target 0
  torch.log_softmax  log p = 0   grad = [0.0, -2.061e-09, -2.061e-09, -2.061e-09]
  safe_softmax       log p = -6.18346e-09   grad = [6.183e-09, -2.061e-09, -2.061e-09, -2.061e-09]
bfloat16: logits [20.0, 0.0, 0.0, 0.0], target 0
  torch.log_softmax  log p = 0   grad = [0.0, -2.066e-09, -2.066e-09, -2.066e-09]
  safe_softmax       log p = -6.34464e-09   grad = [6.17e-09, -2.052e-09, -2.052e-09, -2.052e-09]
float64 reference log p = -6.1834608353410505e-09
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{barron2026fix,
  title   = {A Fix for Softmax and Cross-Entropy},
  author  = {Barron, Jonathan T. and Gnanapragasam, Dan and Tschannen, Michael},
  journal = {arXiv preprint arXiv:2610.31642},
  year    = {2026}
}
```

Related work discussed in the blog post:

- Prieto et al., *Grokking at the Edge of Numerical Stability* (StableMax), ICLR 2025.
- Thilak et al., *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon*, arXiv:2206.04817, 2022.
- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177, 2022.

## License

MIT. This is an independent PyTorch implementation and is not affiliated with the paper's authors.
