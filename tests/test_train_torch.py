"""The batched, padded log prob path must equal a naive per sequence forward.

response_logps pads sequences on the right, runs the transformer body once, and
applies the vocabulary projection only at positions that predict response
tokens. An off by one in that slicing would silently train on the wrong tokens,
so it is checked against the obvious computation, with non zero LoRA weights.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from train_torch import lora_config, micro_batches, response_logps
from objective_torch import sequence_losses

MODEL = "Qwen/Qwen2.5-0.5B"


class A:  # the two fields lora_config reads
    lora_rank, lora_scale = 8, 1.0


def build():
    torch.manual_seed(0)
    m = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32), lora_config(A))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)   # make the adapter actually change the output
    return m


def test_logps_match_naive_forward_and_train():
    m = build().eval()
    pairs = [([9707, 11, 1879, 25], [220, 18, 488, 16, 284, 19]),
             ([3838, 374, 220, 17], [20, 13])]
    got = response_logps(m, pairs, torch.device("cpu"))

    # References are computed in fp64. torch's CPU log_softmax over the 151,936 wide
    # vocabulary is itself off by ~1.7e-4 in fp32; the logsumexp form used in
    # response_logps is within ~1e-6 of fp64, so fp32 references would fail it.
    # 1. slicing, isolated: the same padded batch, full logits, then gather.
    T = max(len(p) + len(r) for p, r in pairs)
    ids = torch.tensor([p + r + [151643] * (T - len(p) - len(r)) for p, r in pairs])
    att = torch.tensor([[1] * (len(p) + len(r)) + [0] * (T - len(p) - len(r)) for p, r in pairs])
    with torch.no_grad():
        full = m(input_ids=ids, attention_mask=att).logits.double()
    for b, ((p, r), g) in enumerate(zip(pairs, got)):
        ref = torch.log_softmax(full[b, len(p) - 1:len(p) + len(r) - 1], -1).gather(1, torch.tensor(r)[:, None]).squeeze(1)
        assert torch.allclose(g.double(), ref, atol=1e-5), ("slicing", g, ref)

    # 2. padding, end to end: against each sequence run alone, unpadded. Masked and
    #    unmasked attention take different kernels, so allow float noise; a real
    #    padding or offset bug moves these by whole units, not the fourth decimal.
    for (p, r), g in zip(pairs, got):
        with torch.no_grad():
            logits = m(input_ids=torch.tensor([p + r])).logits[0].double()
        ref = torch.log_softmax(logits[len(p) - 1:len(p) + len(r) - 1], -1).gather(1, torch.tensor(r)[:, None]).squeeze(1)
        assert torch.allclose(g.double(), ref, atol=1e-3), ("padding", g, ref)
        print(f"  unpadded max abs diff {float((g.double() - ref).abs().max()):.2e}")
    m.train()
    lps = response_logps(m, pairs, torch.device("cpu"))
    loss = sum(sequence_losses(lp[None], lp.detach()[None], torch.ones_like(lp)[None],
                               torch.tensor([a]), True, 1024).sum() for lp, a in zip(lps, [1.0, -1.0]))
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


def test_micro_batches_respect_budget():
    items = [(0, 900), (1, 300), (2, 500), (3, 1200), (4, 100)]
    groups = list(micro_batches(items, 2000))
    lens = dict(items)
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3, 4]
    for g in groups:
        assert len(g) * max(lens[i] for i in g) <= 2000 or len(g) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
