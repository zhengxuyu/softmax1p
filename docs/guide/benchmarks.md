# Benchmarks

All numbers come from scripts in [`benchmarks/`](https://github.com/zhengxuyu/softmax1p/tree/main/benchmarks). They were run with Python 3.12, PyTorch 2.15.0, JAX 0.12.0.1 and TensorFlow 2.21.0 on CPU.

## Same bug, same fix, in every framework

`python benchmarks/collapse_all_frameworks.py` evaluates the row `[m, 0, 0, 0, 0, 0, 0, 0]` with target 0 and `m` past the collapse margin. The last column, Σ grad, is exactly 0 in exact arithmetic.

| dtype | margin | implementation | log p | ∂ log p / ∂z_t | Σ grad |
|---|---:|---|---:|---:|---:|
| float32 | 20 | *exact* | -1.4428e-08 | 1.4428e-08 | 0 |
| float32 | 20 | `torch.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | `softmax_torch` | -1.4428e-08 | 1.4428e-08 | 6.7e-16 |
| float32 | 20 | `jax.nn.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | `softmax_jax` | -1.4428e-08 | 1.4428e-08 | 4.4e-16 |
| float32 | 20 | `tf.nn.log_softmax` | **0** | **0** | -1.4e-08 |
| float32 | 20 | `softmax_tf` | -1.4428e-08 | 1.4428e-08 | 2.2e-16 |
| bfloat16 | 8 | *exact* | -2.3455e-03 | 2.3427e-03 | 0 |
| bfloat16 | 8 | `torch.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | `softmax_torch` | -2.3193e-03 | 2.3499e-03 | 0 |
| bfloat16 | 8 | `jax.nn.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | `softmax_jax` | -2.3346e-03 | 2.3346e-03 | -1.9e-06 |
| bfloat16 | 8 | `tf.nn.log_softmax` | **0** | **0** | -2.3e-03 |
| bfloat16 | 8 | `softmax_tf` | -2.3346e-03 | 2.3346e-03 | -1.9e-06 |

## Margin sweep (PyTorch)

`python benchmarks/margin_sweep.py` compares against a float64 reference across margins. Bold cells show where `torch.log_softmax` has collapsed.

| dtype | margin | ref log p (fp64) | torch log p | fixed log p | torch Σ grad | fixed Σ grad |
|---|---:|---:|---:|---:|---:|---:|
| float16 | 8 | -2.345e-03 | -1.951e-03 | -2.344e-03 | -4.0e-04 | -9.5e-07 |
| float16 | 12 | -4.301e-05 | **0** | -4.297e-05 | -4.3e-05 | 0 |
| bfloat16 | 8 | -2.345e-03 | **0** | -2.319e-03 | -2.3e-03 | 0 |
| bfloat16 | 20 | -1.443e-08 | **0** | -1.525e-08 | -1.4e-08 | -2.9e-11 |
| bfloat16 | 40 | 0 | **0** | -3.144e-17 | -3.0e-17 | -8.1e-20 |
| float32 | 12 | -4.301e-05 | -4.303e-05 | -4.301e-05 | 2.7e-08 | 5.5e-12 |
| float32 | 20 | -1.443e-08 | **0** | -1.443e-08 | -1.4e-08 | 6.7e-16 |
| float32 | 30 | -6.548e-13 | **0** | -6.550e-13 | -6.6e-13 | 4.1e-20 |
| float32 | 40 | 0 | **0** | -2.974e-17 | -3.0e-17 | 4.1e-25 |

In float32 at margin 40, even the float64 reference collapses (40 > 36.7), while the fixed float32 version still returns `-2.97e-17`. The full table is in the [README](https://github.com/zhengxuyu/softmax1p#margin-sweep-pytorch).
