"""The length bias is a claim about gradients, so test the gradients directly.

With one on-policy update (ratio == 1), d loss / d logp at token t of response i
must equal  -A_i / |o_i| / N   under GRPO     (per-response mean), and
            -A_i / budget / N  under Dr. GRPO (constant normaliser).
If those hold, the only thing distinguishing the arms is exactly what the paper
says distinguishes them.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import mlx.core as mx
from objective import VARIANTS, group_advantages, sequence_losses


def test_advantages_match_reference():
    r = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # 2 groups of 8
    g = r.reshape(-1, 8)
    ref_centred = (g - g.mean(1, keepdims=True)).reshape(-1)
    ref_std = ((g - g.mean(1, keepdims=True)) / (g.std(1, ddof=1, keepdims=True) + 1e-8)).reshape(-1)
    assert np.allclose(group_advantages(r, 8, std_norm=False), ref_centred)
    assert np.allclose(group_advantages(r, 8, std_norm=True), ref_std)
    # an all-wrong group carries no signal under either variant
    assert np.allclose(group_advantages(r, 8, std_norm=True)[8:], 0.0)
    assert np.allclose(group_advantages(r, 8, std_norm=False)[8:], 0.0)


def per_token_grads(length_norm, budget=1024):
    lens = [3, 7, 5]                       # three responses of different lengths
    T = max(lens)
    mask = mx.array([[1.0] * n + [0.0] * (T - n) for n in lens])
    adv = mx.array([1.5, -0.5, -1.0])
    logps = mx.array(np.random.default_rng(0).normal(-2, 0.3, (3, T)).astype(np.float32))

    def f(lp):
        seq = sequence_losses(lp, mx.stop_gradient(lp), mask, adv, length_norm, budget)
        return seq.mean()

    return np.array(mx.grad(f)(logps)), lens, np.array(adv), np.array(mask)


def test_grpo_weights_tokens_by_inverse_own_length():
    g, lens, adv, mask = per_token_grads(length_norm=True)
    N = len(lens)
    for i, n in enumerate(lens):
        assert np.allclose(g[i, :n], -adv[i] / n / N, atol=1e-6)
        assert np.allclose(g[i, n:], 0.0)


def test_drgrpo_weights_tokens_by_constant():
    budget = 1024
    g, lens, adv, mask = per_token_grads(length_norm=False, budget=budget)
    N = len(lens)
    for i, n in enumerate(lens):
        assert np.allclose(g[i, :n], -adv[i] / budget / N, atol=1e-9)
        assert np.allclose(g[i, n:], 0.0)


def test_the_bias_itself():
    """Under GRPO a longer wrong answer is penalised less per token than a short one."""
    g, lens, adv, _ = per_token_grads(length_norm=True)
    # responses 1 (len 7) and 2 (len 5) both have negative advantage
    per_token_penalty_long = abs(g[1, 0]) / abs(adv[1])
    per_token_penalty_short = abs(g[2, 0]) / abs(adv[2])
    assert per_token_penalty_long < per_token_penalty_short
    g2, *_ = per_token_grads(length_norm=False)
    assert np.isclose(abs(g2[1, 0]) / abs(adv[1]), abs(g2[2, 0]) / abs(adv[2]))


def test_variant_table_matches_paper():
    assert VARIANTS["grpo"].std_norm and VARIANTS["grpo"].length_norm
    assert not VARIANTS["drgrpo"].std_norm and not VARIANTS["drgrpo"].length_norm


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
