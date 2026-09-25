"""Merged weights loaded into a plain model must reproduce the LoRA model's outputs.

This is what the vLLM sampler relies on: if a name or the scaling were wrong, the
sampler would silently generate from a different policy than the one being trained.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
import torch
from transformers import AutoModelForCausalLM
from test_train_torch import MODEL, build
from train_torch import merged_linear_weights


def test_merged_weights_reproduce_lora_outputs():
    lora = build().eval()                       # LoRA with non zero B
    plain = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    weights = merged_linear_weights(lora, torch.float32, torch.device("cpu"))
    assert len(weights) == 7 * plain.config.num_hidden_layers
    sd = plain.state_dict()
    for name, w in weights:
        assert name in sd, name
        sd[name] = w
    plain.load_state_dict(sd)
    ids = torch.tensor([[9707, 11, 1879, 25, 220, 18, 488, 16, 284, 19]])
    with torch.no_grad():
        a, b = lora(input_ids=ids).logits, plain(input_ids=ids).logits
        base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()(input_ids=ids).logits
    assert float((a - base).abs().max()) > 1e-2, "adapter should change outputs"
    assert torch.allclose(a, b, atol=1e-3), float((a - b).abs().max())
    print(f"  merged vs LoRA max logit diff {float((a - b).abs().max()):.1e}")


if __name__ == "__main__":
    test_merged_weights_reproduce_lora_outputs()
    print("PASS  test_merged_weights_reproduce_lora_outputs")
