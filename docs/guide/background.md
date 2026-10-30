# Softmax Collapse and the fix

## The problem

The standard log-softmax subtracts the largest logit, then adds the other exponentials to $e^0 = 1$:

$$
\log p_{\max} = -\log\Big(1 + \sum_{j \ne \max} e^{-(z_{\max} - z_j)}\Big)
$$

When the top logit leads by a margin $m$ and $e^{-m}$ falls below half an ulp of 1, the sum $1 + e^{-m}$ rounds to exactly $1$:

| dtype | collapse margin |
|---|---:|
| bfloat16 | ≈ 5.5 |
| float16 | ≈ 7.6 |
| float32 | ≈ 16.6 |
| float64 | ≈ 36.7 |

Past that margin, the target's log-probability becomes exactly $0$ and so does its gradient. The other classes keep nonzero gradients $-p_j$, so the row gradient sums to $-\sum_{j \ne t} p_j$ instead of $0$. Adam then normalises this small, consistent bias into full-size updates, which shows up as slingshot loss spikes and drifting parameters.

## The fix

Never form $1 + e^{-m}$. Set the argmax slot to an exact $0$, take the log-sum-exp over the *other* slots only, and join the two with softplus:

$$
s_k = \mathbb{1}[k \ne \arg\max]\,(z_k - z_{\max}), \qquad
\log\mathrm{softmax}_k = s_k - \operatorname{softplus}\big(\mathrm{LSE}_{j \ne \arg\max}\, s_j\big)
$$

For one target class $t$, the same idea gives:

$$
\log p(t) = -\operatorname{softplus}\big(\mathrm{LSE}_{j \ne t}\,(z_j - z_t)\big)
$$

The margin now appears as $\mathrm{LSE} \approx -m$, and $\operatorname{softplus}(-m) \approx e^{-m}$ is computed directly instead of being rounded away against $1$.

Because the kept logit enters only through the differences $z_j - z_{\max}$ (or $z_j - z_t$), autodiff gives it exactly minus the sum of the other gradients. The row gradient therefore sums to zero by construction, with no custom backward.

## Limits

The fix removes the *rounding* cliff, not the *range* limit of a dtype. In float16, $e^{-20} \approx 2\times10^{-9}$ is below the smallest subnormal ($6\times10^{-8}$), so very large margins still underflow to $0$ on both sides.

## Further reading

- J. T. Barron, D. Gnanapragasam, M. Tschannen, [*A Fix for Softmax and Cross-Entropy*](https://arxiv.org/abs/2610.31642), arXiv:2610.31642, 2026.
- [深度理解 "A Fix for Softmax and Cross-Entropy"](https://zhengxuyu.github.io/blog/softmax-cross-entropy-fix) (a walkthrough in Chinese).
- Prieto et al., *Grokking at the Edge of Numerical Stability* (StableMax), ICLR 2025.
- Thilak et al., *The Slingshot Mechanism*, arXiv:2206.04817, 2022.
