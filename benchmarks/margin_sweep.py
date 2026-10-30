"""Sweep the logit margin and compare torch.log_softmax with safe_softmax.log_prob_of.

For each dtype and margin m, a row [m, 0, ..., 0] (C classes, target 0) is evaluated. Prints the
target's log-probability and gradient next to the float64 reference, and the row's gradient sum.

    python benchmarks/margin_sweep.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_softmax import log_prob_of

C = 8
MARGINS = [4.0, 8.0, 12.0, 20.0, 30.0, 40.0]
DTYPES = [("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)]


def run(fn, z):
    x = z.clone().requires_grad_(True)
    lp = fn(x)
    lp.backward()
    return lp.item(), x.grad[0, 0].item(), x.grad.double().sum().item()


def main():
    t = torch.tensor([0])
    print(f"| dtype | margin | ref log p (fp64) | torch log p | safe log p | torch ∂/∂z_t | safe ∂/∂z_t | torch Σ grad | safe Σ grad |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, dtype in DTYPES:
        for m in MARGINS:
            z = torch.zeros(1, C)
            z[0, 0] = m
            ref = torch.log_softmax(z.double(), -1)[0, 0].item()
            zd = z.to(dtype)
            a = run(lambda x: torch.log_softmax(x, -1)[0, 0], zd)
            b = run(lambda x: log_prob_of(x, t)[0], zd)
            print(f"| {name} | {m:g} | {ref:.3e} | {a[0]:.3e} | {b[0]:.3e} | {a[1]:.3e} | {b[1]:.3e} | {a[2]:.1e} | {b[2]:.1e} |")


if __name__ == "__main__":
    main()
