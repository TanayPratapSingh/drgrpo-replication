"""R1-Zero style RL on Qwen2.5-1.5B with GRPO or Dr. GRPO, on one Apple Silicon laptop.

Faithful to the paper's Figure 5 setup (Qwen2.5-1.5B base, R1 template, MATH
questions from the authors' math_12k split, binary answer tag reward, G=8,
temperature 1.0, KL off, AdamW (0.9, 0.95), weight decay 0, grad clip 1.0,
constant LR, one PPO epoch) except where 16 GB of memory forces a change:

  * LoRA adapters instead of full fine tuning. Full fp32 AdamW on 1.5B needs
    about 24 GB, and pure bf16 would lose updates at a 1e-6 learning rate.
  * A 1024 token generation budget instead of 3000.
  * One on-policy optimizer step per 128 response rollout (16 questions x 8).
    The paper takes eight sequential steps per 1024 response rollout, so its
    later minibatches are slightly off-policy; here the ratio is exactly 1.

Every one of these applies identically to both arms, and neither of the two
things under test (the 1/|o| term and the std term) depends on any of them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import Dataset
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from objective import VARIANTS, group_advantages, sequence_losses
from rollout import r1_prompt, sample
from vendor.math_grader import answer_tag_reward_fn

ROOT = Path(__file__).resolve().parent.parent
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def grade(texts, answers):
    rewards, formatted = [], []
    for t, a in zip(texts, answers):
        info, r = answer_tag_reward_fn(t, a, fast=True)
        rewards.append(float(r))
        formatted.append(bool(info.get("formatted", False)))
    return np.array(rewards), np.array(formatted)


def response_logps(model, full_ids: list[int], prompt_len: int) -> mx.array:
    """Log probability of each response token under the current policy.

    Runs the transformer body over the whole sequence but applies the vocabulary
    projection only at the positions that predict response tokens, which keeps
    the 151,936 wide logits off the prompt and roughly halves peak memory.
    """
    ids = mx.array(full_ids)[None]
    h = model.model(ids)[0]                               # (T, H)
    h = h[prompt_len - 1 : len(full_ids) - 1]             # positions predicting the response
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    logits = logits.astype(mx.float32)
    targets = mx.array(full_ids[prompt_len:])
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
    return picked - lse


def evaluate(model, tok, problems, answers, max_tokens, gen_batch):
    prompts = [tok.encode(r1_prompt(p)) for p in problems]
    outs = sample(model, tok, prompts, max_tokens=max_tokens, temperature=0.0,
                  completion_batch_size=gen_batch)
    rewards, formatted = grade([o.text for o in outs], answers)
    lens = np.array([len(o.tokens) for o in outs])
    correct = rewards > 0
    return {
        "n": len(problems),
        "accuracy": float(rewards.mean()),
        "format_rate": float(formatted.mean()),
        "len_mean": float(lens.mean()),
        "len_correct": float(lens[correct].mean()) if correct.any() else None,
        "len_incorrect": float(lens[~correct].mean()) if (~correct).any() else None,
        "truncated": int(sum(o.finish == "length" for o in outs)),
    }, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", type=int, default=16, help="questions per step")
    ap.add_argument("--group", type=int, default=8, help="responses per question (G)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="generation budget")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--gen-batch", type=int, default=64, help="concurrent sequences while sampling")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=100, help="MATH500 problems per interim eval")
    ap.add_argument("--final-eval-n", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variant = VARIANTS[args.loss]
    out = ROOT / "runs" / (args.out or f"{args.loss}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    # A restarted run must not interleave its steps with a crashed run's lines.
    for stale in ("log.jsonl", "eval.jsonl", "rollouts.jsonl"):
        (out / stale).write_text("")
    (out / "config.json").write_text(json.dumps({
        **vars(args), "variant": variant.__dict__,
        "started": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # MLX keeps freed buffers in a cache that get_peak_memory does not report. On
    # a 16 GB machine that cache pushed the first pilot into swap until the Metal
    # watchdog killed a command buffer mid rollout, so cap it and clear it
    # between the rollout and training phases.
    mx.set_cache_limit(1 * 2**30)
    # Pin the GPU working set in RAM. Swap alone was survivable; what killed the
    # run was macOS paging out Metal buffers mid command, so the GPU stalled
    # past the watchdog. 9 GB covers the measured 7.6 GB peak and stays under
    # the M5's 11.84 GB recommended working set.
    mx.set_wired_limit(9 * 2**30)

    model, tok = load(args.model)
    model.freeze()
    linear_to_lora_layers(model, len(model.model.layers),
                          {"rank": args.lora_rank, "scale": args.lora_scale,
                           "dropout": 0.0, "keys": LORA_KEYS})
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[{args.loss} s{args.seed}] trainable LoRA params: {n_train:,}", flush=True)

    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], eps=1e-8,
                            weight_decay=0.0)

    train = Dataset.from_file(str(ROOT / "data" / "math_12k.arrow"))
    evald = Dataset.from_file(str(ROOT / "data" / "eval_math500.arrow"))
    order = np.random.default_rng(args.seed).permutation(len(train))
    eval_idx = np.random.default_rng(1234).permutation(len(evald))  # same subset for every run

    def run_eval(step, n):
        idx = eval_idx[:n]
        res, outs = evaluate(model, tok, [evald[int(i)]["problem"] for i in idx],
                             [evald[int(i)]["answer"] for i in idx], args.max_tokens,
                             args.gen_batch)
        res["step"] = step
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{args.loss} s{args.seed}] eval step {step} n={n}: acc {res['accuracy']:.3f} "
              f"len {res['len_mean']:.0f} (correct {res['len_correct']}, "
              f"incorrect {res['len_incorrect']})", flush=True)
        return outs

    run_eval(0, args.eval_n)

    def loss_fn(model, full_ids, prompt_len, adv, n_total):
        lp = response_logps(model, full_ids, prompt_len)[None]
        mask = mx.ones_like(lp)
        seq = sequence_losses(lp, mx.stop_gradient(lp), mask, mx.array([adv]),
                              variant.length_norm, args.max_tokens)
        return seq.sum() / n_total          # summed over sequences = mean over the rollout

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Gradient checkpointing for the backward passes only. A 1024 token response
    # otherwise pushes peak memory past what 16 GB can hold; during rollouts the
    # plain forward is restored so generation pays nothing for it.
    Block = type(model.model.layers[0])
    plain_call = Block.__call__

    def checkpointed_call(self, *a, **k):
        def inner(params, *a, **k):
            self.update(params)
            return plain_call(self, *a, **k)
        return mx.checkpoint(inner)(self.trainable_parameters(), *a, **k)

    cursor = 0

    for step in range(1, args.steps + 1):
        qs = [train[int(order[(cursor + k) % len(order)])] for k in range(args.prompts)]
        cursor += args.prompts
        prompt_ids = [tok.encode(r1_prompt(q["problem"])) for q in qs]
        answers = [q["answer"] for q in qs for _ in range(args.group)]
        expanded = [p for p in prompt_ids for _ in range(args.group)]

        # identical seed per (run seed, step): both arms see the same step 1 rollouts
        mx.random.seed(args.seed * 1_000_003 + step)
        t0 = time.time()
        rolls = sample(model, tok, expanded, max_tokens=args.max_tokens, temperature=1.0,
                       completion_batch_size=args.gen_batch)
        t_roll = time.time() - t0
        mx.clear_cache()

        rewards, formatted = grade([r.text for r in rolls], answers)
        adv = group_advantages(rewards, args.group, variant.std_norm)
        lens = np.array([len(r.tokens) for r in rolls])
        n_total = len(rolls)

        # a zero advantage sequence contributes exactly zero gradient, so skip its pass
        t1 = time.time()
        acc = None
        live = [i for i in range(n_total) if adv[i] != 0.0]
        Block.__call__ = checkpointed_call
        for i in live:
            full = expanded[i] + rolls[i].tokens
            _, g = loss_and_grad(model, full, len(expanded[i]), float(adv[i]), n_total)
            acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
            mx.eval(acc)
        Block.__call__ = plain_call
        grad_norm, clipped = 0.0, False
        if acc is not None:
            acc, gn = optim.clip_grad_norm(acc, max_norm=args.clip_grad)
            grad_norm = float(gn)
            clipped = grad_norm > args.clip_grad
            optimizer.update(model, acc)
            mx.eval(model.parameters(), optimizer.state)
        t_train = time.time() - t1
        mx.clear_cache()

        correct = rewards > 0
        groups = rewards.reshape(-1, args.group).sum(1)
        rec = {
            "step": step,
            "reward": float(rewards.mean()),
            "format_rate": float(formatted.mean()),
            "len_mean": float(lens.mean()),
            "len_median": float(np.median(lens)),
            "len_correct": float(lens[correct].mean()) if correct.any() else None,
            "len_incorrect": float(lens[~correct].mean()) if (~correct).any() else None,
            "truncated": int(sum(r.finish == "length" for r in rolls)),
            "signal_groups": int(((groups > 0) & (groups < args.group)).sum()),
            "trained_seqs": len(live),
            "grad_norm": grad_norm,
            "clipped": clipped,
            "tokens": int(lens.sum()),
            "t_rollout": round(t_roll, 1),
            "t_train": round(t_train, 1),
            "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
        }
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(out / "rollouts.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "rewards": rewards.tolist(),
                                "lens": lens.tolist(), "formatted": formatted.tolist(),
                                "sample": rolls[int(np.argmax(rewards))].text[:2000]}) + "\n")
        print(f"[{args.loss} s{args.seed}] step {step:>3} reward {rec['reward']:.3f} "
              f"fmt {rec['format_rate']:.2f} len {rec['len_mean']:.0f} "
              f"(c {rec['len_correct']} / i {rec['len_incorrect']}) sig {rec['signal_groups']}/16 "
              f"gn {grad_norm:.3f} roll {t_roll:.0f}s train {t_train:.0f}s "
              f"peak {rec['peak_gb']}GB", flush=True)
        mx.reset_peak_memory()

        if step % args.eval_every == 0 and step != args.steps:
            run_eval(step, args.eval_n)
            mx.save_safetensors(str(out / "adapter.safetensors"),
                                dict(tree_flatten(model.trainable_parameters())))

    outs = run_eval(args.steps, args.final_eval_n)
    mx.save_safetensors(str(out / "adapter.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    with open(out / "final_eval_responses.jsonl", "w") as f:
        for o in outs:
            f.write(json.dumps({"len": len(o.tokens), "finish": o.finish, "text": o.text}) + "\n")
    print(f"[{args.loss} s{args.seed}] done", flush=True)


if __name__ == "__main__":
    main()
