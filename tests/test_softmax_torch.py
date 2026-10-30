import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softmax_torch import cross_entropy, log_prob_of, log_softmax, softmax
from vocab_parallel_torch import vocab_parallel_log_prob_of

DTYPES = [torch.float32, torch.bfloat16, torch.float16]
# margins past the collapse threshold of each dtype (16.6 / 5.5 / 7.6)
COLLAPSE_MARGIN = {torch.float32: 20.0, torch.bfloat16: 8.0, torch.float16: 10.0}


def _collapse_row(dtype, c=8):
    z = torch.zeros(1, c)
    z[0, 0] = COLLAPSE_MARGIN[dtype]
    return z.to(dtype)


# ── agreement with torch on ordinary inputs ────────────────────────────────────────────────────

@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax_matches_torch(dim):
    torch.manual_seed(0)
    z = torch.randn(5, 7, 11, dtype=torch.float64) * 3
    torch.testing.assert_close(log_softmax(z, dim), torch.log_softmax(z, dim))
    torch.testing.assert_close(softmax(z, dim), torch.softmax(z, dim))


def test_log_prob_of_matches_torch():
    torch.manual_seed(0)
    z = torch.randn(4, 6, 50, dtype=torch.float64) * 4
    t = torch.randint(0, 50, (4, 6))
    ref = torch.log_softmax(z, -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(log_prob_of(z, t), ref)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_matches_torch(reduction):
    torch.manual_seed(0)
    z = torch.randn(32, 100, dtype=torch.float64) * 3
    t = torch.randint(0, 100, (32,))
    t[::5] = -100
    ref = F.cross_entropy(z, t, ignore_index=-100, reduction=reduction)
    torch.testing.assert_close(cross_entropy(z, t, reduction=reduction), ref)


def test_cross_entropy_bad_reduction():
    with pytest.raises(ValueError):
        cross_entropy(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), reduction="avg")


def test_gradcheck():
    torch.manual_seed(0)
    z = (torch.randn(3, 9, dtype=torch.float64) * 5).requires_grad_(True)
    t = torch.randint(0, 9, (3,))
    assert torch.autograd.gradcheck(lambda x: log_softmax(x, -1), (z,))
    assert torch.autograd.gradcheck(lambda x: log_prob_of(x, t), (z,))
    assert torch.autograd.gradcheck(lambda x: cross_entropy(x, t), (z,))
    assert torch.autograd.gradcheck(lambda x: vocab_parallel_log_prob_of(x, t, chunk=2), (z,))


# ── Softmax Collapse ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_collapses(dtype):
    """Sanity check of the bug itself: past the margin, torch returns log p = 0 and target grad 0."""
    z = _collapse_row(dtype).requires_grad_(True)
    lp = torch.log_softmax(z, -1)[0, 0]
    lp.backward()
    assert lp.item() == 0.0
    assert z.grad[0, 0].item() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_collapse(dtype):
    z = _collapse_row(dtype)
    ref = torch.log_softmax(z.double(), -1)[0, 0]
    x = z.clone().requires_grad_(True)
    lp = log_prob_of(x, torch.tensor([0]))[0]
    lp.backward()
    assert lp.item() < 0
    torch.testing.assert_close(lp.double(), ref, rtol=2e-2, atol=0)
    assert x.grad[0, 0].item() > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", ["log_softmax", "log_prob_of", "vocab_parallel"])
def test_row_gradient_sums_to_zero(dtype, fn):
    """d/dz of a log-probability sums to zero over the row (softmax is shift-invariant)."""
    torch.manual_seed(0)
    z = torch.randn(16, 64) * 2
    z[:, 0] += torch.linspace(0, 40, 16)          # margins from small to far past collapse
    x = z.to(dtype).requires_grad_(True)
    t = torch.zeros(16, dtype=torch.long)
    if fn == "log_softmax":
        out = log_softmax(x, -1)[:, 0]
    elif fn == "log_prob_of":
        out = log_prob_of(x, t)
    else:
        out = vocab_parallel_log_prob_of(x, t, chunk=5)
    out.sum().backward()
    g = x.grad.double()
    scale = g.abs().max(dim=-1).values.clamp_min(1e-30)
    assert ((g.sum(-1).abs() / scale) < 0.05).all()


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
def test_log_softmax_large_margin_matches_float64(dtype, rtol):
    z = torch.tensor([[30.0, 1.0, 0.0, -2.0, 5.0]])
    ref = torch.log_softmax(z.double(), -1)
    got = log_softmax(z.to(dtype), -1).double()
    torch.testing.assert_close(got, ref, rtol=rtol, atol=1e-30)


# ── vocab-parallel version (unsharded path) ────────────────────────────────────────────────────

def test_vocab_parallel_unsharded_matches():
    torch.manual_seed(0)
    z = torch.randn(3, 7, 40, dtype=torch.float64) * 5
    t = torch.randint(0, 40, (3, 7))
    a = z.clone().requires_grad_(True)
    b = z.clone().requires_grad_(True)
    la = log_prob_of(a, t)
    lb = vocab_parallel_log_prob_of(b, t, chunk=4)
    torch.testing.assert_close(lb, la)
    la.sum().backward()
    lb.sum().backward()
    torch.testing.assert_close(b.grad, a.grad)
