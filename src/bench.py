"""Measure one paper sized rollout batch (16 questions x 8 samples) on this machine."""
import random, statistics, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import load
from datasets import Dataset
from rollout import r1_prompt, sample
from vendor.math_grader import answer_tag_reward_fn

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

model, tok = load(MODEL)
data = Dataset.from_file("data/math_12k.arrow")
rng = random.Random(0)
qs = [data[i] for i in rng.sample(range(len(data)), 16)]
prompts = [tok.encode(r1_prompt(q["problem"])) for q in qs for _ in range(8)]
answers = [q["answer"] for q in qs for _ in range(8)]

mx.random.seed(0)
mx.reset_peak_memory()
t0 = time.time()
outs = sample(model, tok, prompts, max_tokens=MAX_TOKENS, temperature=1.0, completion_batch_size=128)
dt = time.time() - t0

rewards = [answer_tag_reward_fn(o.text, a, fast=True)[1] for o, a in zip(outs, answers)]
lens = [len(o.tokens) for o in outs]
fin = {k: sum(o.finish == k for o in outs) for k in ("answer", "eos", "length")}
groups = [rewards[i:i + 8] for i in range(0, 128, 8)]
mixed = sum(0 < sum(g) < 8 for g in groups)
print(f"model {MODEL}  budget {MAX_TOKENS}")
print(f"wall {dt:.1f}s  generated {sum(lens):,} tokens  -> {sum(lens)/dt:,.0f} tok/s aggregate")
print(f"peak memory {mx.get_peak_memory()/2**30:.2f} GB")
print(f"response length mean {statistics.mean(lens):.0f}  median {statistics.median(lens):.0f}  max {max(lens)}")
print(f"finish reasons {fin}")
print(f"mean reward {statistics.mean(rewards):.3f}  | groups with learning signal (mixed rewards): {mixed}/16")
print("\n--- one sampled response ---\n" + outs[0].text[:700])
