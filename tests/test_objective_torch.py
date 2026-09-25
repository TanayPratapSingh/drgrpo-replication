"""The PyTorch loss must give exactly the gradients the MLX loss is tested for.

Same checks as tests/test_objective.py: with ratio == 1, d loss / d logp at a
token of response i is -A_i / |o_i| / N under GRPO and -A_i / budget / N under
Dr. GRPO. A final check runs identical inputs through both backends.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch
from objective_torch import sequence_losses

LENS = [3, 7, 5]
ADV = [1.5, -0.5, -1.0]


def grads(length_norm, budget=1024):
    T = max(LENS)
    mask = torch.tensor([[1.0] * n + [0.0] * (T - n) for n in LENS], dtype=torch.float64)
    adv = torch.tensor(ADV, dtype=torch.float64)
    lp = torch.tensor(np.random.default_rng(0).normal(-2, 0.3, (3, T)), requires_grad=True)
    sequence_losses(lp, lp.detach(), mask, adv, length_norm, budget).mean().backward()
    return lp.grad.numpy()


def test_grpo_weights_tokens_by_inverse_own_length():
    g = grads(True)
    for i, n in enumerate(LENS):
        assert np.allclose(g[i, :n], -ADV[i] / n / len(LENS))
        assert np.allclose(g[i, n:], 0.0)


def test_drgrpo_weights_tokens_by_constant():
    g = grads(False, budget=1024)
    for i, n in enumerate(LENS):
        assert np.allclose(g[i, :n], -ADV[i] / 1024 / len(LENS))
        assert np.allclose(g[i, n:], 0.0)


def test_clipping_matches_ppo_when_off_policy():
    # ratio above 1 + eps with positive advantage: the clipped branch wins, gradient is zero
    lp = torch.tensor([[0.5]], requires_grad=True)
    loss = sequence_losses(lp, torch.tensor([[0.0]]), torch.ones(1, 1), torch.tensor([1.0]), True, 10)
    loss.sum().backward()
    assert lp.grad.item() == 0.0


def test_matches_the_mlx_backend():
    try:
        import mlx.core as mx
        from objective import sequence_losses as mlx_losses
    except ImportError:
        print("  (mlx not installed in this env; cross backend check skipped)")
        return
    T = max(LENS)
    m = np.array([[1.0] * n + [0.0] * (T - n) for n in LENS], dtype=np.float32)
    lp = np.random.default_rng(1).normal(-2, 0.3, (3, T)).astype(np.float32)
    old = lp - 0.05
    for ln in (True, False):
        t = sequence_losses(torch.tensor(lp), torch.tensor(old), torch.tensor(m), torch.tensor(ADV, dtype=torch.float32), ln, 1024).numpy()
        x = np.array(mlx_losses(mx.array(lp), mx.array(old), mx.array(m), mx.array(np.array(ADV, dtype=np.float32)), ln, 1024))
        assert np.allclose(t, x, atol=1e-6), (ln, t, x)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
