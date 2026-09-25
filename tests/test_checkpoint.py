"""Resuming from a checkpoint must reproduce an uninterrupted run exactly.

A tiny model stands in for the LoRA adapter (same tree shape: a list of layers),
trained with the same AdamW configuration as the real run. If Adam's moments or
its step counter were restored wrong, the bias correction and update directions
would drift and the final weights would differ.
"""
import sys, tempfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from train import latest_checkpoint, save_checkpoint


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(16, 16) for _ in range(3)]

    def __call__(self, x):
        for l in self.layers:
            x = mx.tanh(l(x))
        return x


def make():
    mx.random.seed(7)
    m = Toy()
    o = optim.AdamW(learning_rate=5e-2, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0)
    return m, o


def step(m, o, k):
    x = mx.random.normal((32, 16), key=mx.random.key(100 + k))
    y = mx.random.normal((32, 16), key=mx.random.key(200 + k))
    loss_grad = nn.value_and_grad(m, lambda mm: ((mm(x) - y) ** 2).mean())
    _, g = loss_grad(m)
    g, _ = optim.clip_grad_norm(g, 1.0)
    o.update(m, g)
    mx.eval(m.parameters(), o.state)


def weights(m):
    return {k: np.array(v) for k, v in tree_flatten(m.parameters())}


def test_resume_is_bit_exact():
    ref, ro = make()
    for k in range(5):
        step(ref, ro, k)

    m, o = make()
    for k in range(3):
        step(m, o, k)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ckpt"
        save_checkpoint(root, 3, m, o)
        save_checkpoint(root, 3, m, o)           # overwriting the same step is safe
        ck = latest_checkpoint(root)
        assert ck is not None and ck.name == "step_0003"
        fresh, fo = make()
        fresh.load_weights(str(ck / "adapter.safetensors"), strict=False)
        fo.state = tree_unflatten(list(mx.load(str(ck / "optimizer.safetensors")).items()))
        assert int(fo.state["step"]) == 3
    for k in range(3, 5):
        step(fresh, fo, k)

    a, b = weights(ref), weights(fresh)
    assert a.keys() == b.keys()
    worst = max(float(np.abs(a[k] - b[k]).max()) for k in a)
    assert worst == 0.0, f"resumed weights differ by up to {worst}"


def test_old_checkpoints_are_pruned():
    m, o = make()
    step(m, o, 0)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "ckpt"
        for s in (5, 10, 15):
            save_checkpoint(root, s, m, o)
        assert [p.name for p in root.iterdir()] == ["step_0015"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
