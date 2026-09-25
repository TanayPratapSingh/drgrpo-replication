"""R1-Zero style RL with GRPO or Dr. GRPO in PyTorch, for Kaggle GPUs.

The CUDA twin of train.py (MLX). Same data, template, grader, objective,
hyperparameters, log schema and checkpoint layout, so src/analyze.py reads
either. Differences are backend only:

  * Sampling runs on its own engine: vLLM when it imports (LoRA adapter hot
    swapped every step) or HF generate on a separate model copy whose LoRA
    weights are synced from the learner every step.
  * With two GPUs the sampler takes one and the learner the other.
  * bf16 where the GPU supports it, fp16 with loss scaling on T4.
  * --max-hours checkpoints and exits cleanly before Kaggle's session limit, and
    --resume-from continues from a previous session's output.

Both arms of a comparison must run on this backend: an MLX arm and a CUDA arm
would differ by hardware as well as by objective.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from objective_torch import VARIANTS, group_advantages, sequence_losses
from template import STOP_STR, r1_prompt
from vendor.math_grader import answer_tag_reward_fn

ROOT = Path(__file__).resolve().parent.parent
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Rollout:
    tokens: list[int]
    text: str
    finish: str  # "answer", "eos" or "length"


def grade(texts, answers):
    rewards, formatted = [], []
    for t, a in zip(texts, answers):
        info, r = answer_tag_reward_fn(t, a, fast=True)
        rewards.append(float(r))
        formatted.append(bool(info.get("formatted", False)))
    return np.array(rewards), np.array(formatted)


# --------------------------------------------------------------------------- samplers

class VLLMSampler:
    """vLLM with the current LoRA adapter loaded from disk before every call."""

    def __init__(self, args, work: Path, dtype: str):
        from vllm import LLM
        self.args, self.dir, self.version = args, work / "adapter_for_sampler", 0
        self.llm = LLM(model=args.model, dtype=dtype, enable_lora=True,
                       max_lora_rank=args.lora_rank, max_loras=1, seed=args.seed,
                       gpu_memory_utilization=args.vllm_mem,
                       max_model_len=args.max_prompt + args.max_tokens,
                       enable_prefix_caching=True)
        self.name = "vllm"

    def sync(self, learner):
        self.version += 1
        learner.save_pretrained(str(self.dir))
        from vllm.lora.request import LoRARequest
        self.lora = LoRARequest(f"policy_v{self.version}", self.version, str(self.dir))

    def generate(self, prompt_ids, n, temperature, seed):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=self.args.max_tokens,
                            stop=[STOP_STR], include_stop_str_in_output=True,
                            seed=seed if temperature > 0 else None)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp,
                                 lora_request=self.lora, use_tqdm=False)
        rolls = []
        for o in outs:
            for c in o.outputs:
                finish = ("length" if c.finish_reason == "length"
                          else "answer" if c.stop_reason == STOP_STR else "eos")
                rolls.append(Rollout(list(c.token_ids), c.text, finish))
        return rolls


class HFSampler:
    """HF generate on a separate model copy, LoRA weights synced from the learner."""

    def __init__(self, args, tok, device, dtype):
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
        self.model = get_peft_model(base, lora_config(args)).eval()
        self.tok, self.args, self.device, self.name = tok, args, device, "hf"

    def sync(self, learner):
        sd = {k: v.to(self.device, dtype=next(self.model.parameters()).dtype)
              for k, v in get_peft_model_state_dict(learner).items()}
        set_peft_model_state_dict(self.model, sd)

    @torch.no_grad()
    def generate(self, prompt_ids, n, temperature, seed):
        torch.manual_seed(seed)
        eos = self.tok.eos_token_id
        rolls = []
        expanded = [p for p in prompt_ids for _ in range(n)]
        bs = self.args.gen_batch
        for s in range(0, len(expanded), bs):
            chunk = expanded[s:s + bs]
            L = max(len(p) for p in chunk)
            ids = torch.tensor([[eos] * (L - len(p)) + p for p in chunk], device=self.device)
            att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in chunk], device=self.device)
            out = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=self.args.max_tokens,
                                      do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                                      top_p=1.0, top_k=0, stop_strings=[STOP_STR], tokenizer=self.tok,
                                      pad_token_id=eos, eos_token_id=eos)
            for row in out[:, L:].tolist():
                toks, finish = [], "length"
                for t in row:
                    toks.append(t)
                    if t == eos:
                        finish = "eos"; break
                    if STOP_STR in self.tok.decode(toks[-6:]):
                        finish = "answer"; break
                rolls.append(Rollout(toks, self.tok.decode(toks), finish))
        return rolls


# --------------------------------------------------------------------------- learner

def lora_config(args):
    return LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * args.lora_scale,
                      lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none",
                      task_type="CAUSAL_LM")


def response_logps(model, pairs, device):
    """Per token log probs of each response; the vocab projection only at response positions."""
    eos = model.config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    seqs = [p + r for p, r in pairs]
    T = max(len(s) for s in seqs)
    ids = torch.tensor([s + [eos] * (T - len(s)) for s in seqs], device=device)
    att = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], device=device)
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state
    rows, cols, targets = [], [], []
    for b, (p, r) in enumerate(pairs):
        rows += [b] * len(r)
        cols += list(range(len(p) - 1, len(p) + len(r) - 1))
        targets += r
    logits = base.lm_head(h[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]).float()
    tgt = torch.tensor(targets, device=device)
    lp = logits.gather(1, tgt[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)
    return torch.split(lp, [len(r) for _, r in pairs])


def micro_batches(items, budget_tokens):
    """Group (index, total_len) pairs, longest first, so each group stays under budget."""
    batch, size = [], 0
    for i, n in sorted(items, key=lambda x: -x[1]):
        if batch and (len(batch) + 1) * max(n, size) > budget_tokens:
            yield batch
            batch, size = [], 0
        batch.append(i)
        size = max(size, n)
    if batch:
        yield batch


# --------------------------------------------------------------------------- checkpoints

def save_checkpoint(root: Path, step: int, model, optimizer, scaler):
    root.mkdir(parents=True, exist_ok=True)
    final, tmp = root / f"step_{step:04d}", root / f"step_{step:04d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save(get_peft_model_state_dict(model), tmp / "adapter.pt")
    torch.save({"optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None}, tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "saved": now()}))
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)
    for old in root.glob("step_*"):
        if old != final:
            shutil.rmtree(old)


def latest_checkpoint(root: Path):
    if not root.exists():
        return None
    done = sorted(d for d in root.glob("step_*")
                  if d.is_dir() and not d.name.endswith(".tmp") and (d / "state.json").exists())
    return done[-1] if done else None


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--final-eval-n", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--sampler", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--gen-batch", type=int, default=64, help="HF sampler batch size")
    ap.add_argument("--micro-tokens", type=int, default=6144, help="padded tokens per backward micro batch")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--resume-from", default=None, help="a previous session's runs dir to continue from")
    ap.add_argument("--max-hours", type=float, default=None, help="checkpoint and exit before this")
    ap.add_argument("--stop-at", type=int, default=None, help="checkpoint and exit after this step")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t_start = time.time()

    variant = VARIANTS[args.loss]
    name = args.out or f"{args.loss}_s{args.seed}"
    out = Path(args.runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = out / "ckpt"

    # carry a previous session's checkpoint and logs forward
    if args.resume_from and latest_checkpoint(ckpt_root) is None:
        prev = Path(args.resume_from) / name
        if latest_checkpoint(prev / "ckpt") is not None:
            shutil.copytree(prev, out, dirs_exist_ok=True)
            print(f"[{name}] copied previous session state from {prev}", flush=True)

    n_gpu = torch.cuda.device_count()
    learner_dev = torch.device(f"cuda:{1 if n_gpu > 1 else 0}") if n_gpu else torch.device("cpu")
    sampler_dev = torch.device("cuda:0") if n_gpu else torch.device("cpu")
    # native bf16 needs compute capability 8.0 (Ampere). torch reports bf16 as supported on
    # a T4 (7.5) by emulating it, which is slow, and vLLM refuses bf16 there outright.
    bf16 = bool(n_gpu) and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if bf16 else (torch.float16 if n_gpu else torch.float32)
    use_scaler = compute_dtype == torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=compute_dtype,
                                                attn_implementation="sdpa").to(learner_dev)
    base.config.use_cache = False
    model = get_peft_model(base, lora_config(args))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()           # LoRA weights and their Adam state in fp32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[{name}] trainable LoRA params: {sum(p.numel() for p in params):,} | "
          f"gpus {n_gpu} | learner {learner_dev} | dtype {compute_dtype}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    sampler = None
    if args.sampler in ("auto", "vllm") and n_gpu:
        try:
            sampler = VLLMSampler(args, out, "bfloat16" if bf16 else "half")
        except Exception as e:                # vLLM missing or unsupported on this GPU
            if args.sampler == "vllm":
                raise
            print(f"[{name}] vLLM unavailable ({type(e).__name__}: {e}); using HF generate", flush=True)
    if sampler is None:
        sampler = HFSampler(args, tok, sampler_dev, compute_dtype)
    print(f"[{name}] sampler: {sampler.name} on {sampler_dev}", flush=True)

    data_dir = Path(args.data_dir)
    train = Dataset.from_file(str(data_dir / "math_12k.arrow"))
    evald = Dataset.from_file(str(data_dir / "eval_math500.arrow"))
    order = np.random.default_rng(args.seed).permutation(len(train))
    eval_idx = np.random.default_rng(1234).permutation(len(evald))

    def run_eval(step, n):
        idx = eval_idx[:n]
        prompts = [tok.encode(r1_prompt(evald[int(i)]["problem"])) for i in idx]
        answers = [evald[int(i)]["answer"] for i in idx]
        outs = sampler.generate(prompts, 1, 0.0, seed=0)
        rewards, formatted = grade([o.text for o in outs], answers)
        lens = np.array([len(o.tokens) for o in outs])
        c = rewards > 0
        res = {"step": step, "n": n, "accuracy": float(rewards.mean()),
               "format_rate": float(formatted.mean()), "len_mean": float(lens.mean()),
               "len_correct": float(lens[c].mean()) if c.any() else None,
               "len_incorrect": float(lens[~c].mean()) if (~c).any() else None,
               "truncated": int(sum(o.finish == "length" for o in outs))}
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{name}] eval step {step} n={n}: acc {res['accuracy']:.3f} len {res['len_mean']:.0f} "
              f"(correct {res['len_correct']}, incorrect {res['len_incorrect']})", flush=True)
        return outs

    start_step = 1
    resume_from = latest_checkpoint(ckpt_root)
    if resume_from is not None:
        st = json.loads((resume_from / "state.json").read_text())
        set_peft_model_state_dict(model, torch.load(resume_from / "adapter.pt", map_location=learner_dev))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        opt = torch.load(resume_from / "optimizer.pt", map_location=learner_dev, weights_only=False)
        optimizer.load_state_dict(opt["optimizer"])
        if scaler is not None and opt["scaler"] is not None:
            scaler.load_state_dict(opt["scaler"])
        start_step = st["step"] + 1
        for fname in ("log.jsonl", "rollouts.jsonl", "eval.jsonl"):
            f = out / fname
            if f.exists():
                kept = [l for l in f.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= st["step"]]
                f.write_text("".join(l + "\n" for l in kept))
        cfg = json.loads((out / "config.json").read_text())
        cfg.setdefault("resumed", []).append({"from_step": st["step"], "at": now()})
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[{name}] resumed from checkpoint at step {st['step']}", flush=True)
        sampler.sync(model)
    else:
        for stale in ("log.jsonl", "eval.jsonl", "rollouts.jsonl"):
            (out / stale).write_text("")
        (out / "config.json").write_text(json.dumps({
            **vars(args), "variant": variant.__dict__, "backend": "torch", "sampler": sampler.name,
            "gpu": torch.cuda.get_device_name(0) if n_gpu else "cpu", "n_gpu": n_gpu,
            "compute_dtype": str(compute_dtype), "started": now()}, indent=2))
        sampler.sync(model)
        run_eval(0, args.eval_n)

    step_times = []
    for step in range(start_step, args.steps + 1):
        if args.max_hours and step_times:
            elapsed_h = (time.time() - t_start) / 3600
            if elapsed_h + 2 * max(step_times[-5:]) / 3600 > args.max_hours:
                save_checkpoint(ckpt_root, step - 1, model, optimizer, scaler)
                print(f"[{name}] time limit: checkpointed at step {step - 1}, exiting", flush=True)
                return
        t_step = time.time()
        k0 = (step - 1) * args.prompts
        qs = [train[int(order[(k0 + k) % len(order)])] for k in range(args.prompts)]
        prompt_ids = [tok.encode(r1_prompt(q["problem"])) for q in qs]
        answers = [q["answer"] for q in qs for _ in range(args.group)]
        expanded = [p for p in prompt_ids for _ in range(args.group)]

        t0 = time.time()
        rolls = sampler.generate(prompt_ids, args.group, 1.0, seed=args.seed * 1_000_003 + step)
        t_roll = time.time() - t0

        rewards, formatted = grade([r.text for r in rolls], answers)
        adv = group_advantages(rewards, args.group, variant.std_norm)
        lens = np.array([len(r.tokens) for r in rolls])
        n_total = len(rolls)

        t1 = time.time()
        model.train()
        live = [i for i in range(n_total) if adv[i] != 0.0 and len(rolls[i].tokens) > 0]
        optimizer.zero_grad(set_to_none=True)
        for mb in micro_batches([(i, len(expanded[i]) + len(rolls[i].tokens)) for i in live], args.micro_tokens):
            pairs = [(expanded[i], rolls[i].tokens) for i in mb]
            with torch.autocast(device_type=learner_dev.type, dtype=compute_dtype, enabled=n_gpu > 0):
                lps = response_logps(model, pairs, learner_dev)
            loss = 0.0
            for lp, i in zip(lps, mb):
                seq = sequence_losses(lp[None], lp.detach()[None], torch.ones_like(lp)[None],
                                      torch.tensor([adv[i]], device=learner_dev, dtype=torch.float32),
                                      variant.length_norm, args.max_tokens)
                loss = loss + seq.sum() / n_total
            (scaler.scale(loss) if scaler else loss).backward()
        grad_norm, clipped, nonfinite = 0.0, False, False
        if live:
            if scaler:
                scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            grad_norm = float(gn)
            nonfinite = not np.isfinite(grad_norm)
            clipped = grad_norm > args.clip_grad
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            elif not nonfinite:
                optimizer.step()
        sampler.sync(model)
        t_train = time.time() - t1

        c = rewards > 0
        groups = rewards.reshape(-1, args.group).sum(1)
        rec = {"step": step, "reward": float(rewards.mean()), "format_rate": float(formatted.mean()),
               "len_mean": float(lens.mean()), "len_median": float(np.median(lens)),
               "len_correct": float(lens[c].mean()) if c.any() else None,
               "len_incorrect": float(lens[~c].mean()) if (~c).any() else None,
               "truncated": int(sum(r.finish == "length" for r in rolls)),
               "signal_groups": int(((groups > 0) & (groups < args.group)).sum()),
               "trained_seqs": len(live), "grad_norm": grad_norm, "clipped": clipped,
               "nonfinite_grad": nonfinite, "tokens": int(lens.sum()),
               "t_rollout": round(t_roll, 1), "t_train": round(t_train, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated(learner_dev) / 2**30, 2) if n_gpu else 0.0}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(out / "rollouts.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "rewards": rewards.tolist(), "lens": lens.tolist(),
                                "formatted": formatted.tolist(),
                                "sample": rolls[int(np.argmax(rewards))].text[:2000]}) + "\n")
        print(f"[{name}] step {step:>3} reward {rec['reward']:.3f} fmt {rec['format_rate']:.2f} "
              f"len {rec['len_mean']:.0f} (c {rec['len_correct']} / i {rec['len_incorrect']}) "
              f"sig {rec['signal_groups']}/{args.prompts} gn {grad_norm:.3f} roll {t_roll:.0f}s "
              f"train {t_train:.0f}s peak {rec['peak_gb']}GB", flush=True)
        if n_gpu:
            torch.cuda.reset_peak_memory_stats(learner_dev)
        step_times.append(time.time() - t_step)

        if step % args.eval_every == 0 and step != args.steps:
            run_eval(step, args.eval_n)
        if step % args.ckpt_every == 0 or step == args.stop_at:
            save_checkpoint(ckpt_root, step, model, optimizer, scaler)
        if args.stop_at and step == args.stop_at and step != args.steps:
            print(f"[{name}] stopped after step {step} as requested; checkpoint saved", flush=True)
            return

    outs = run_eval(args.steps, args.final_eval_n)
    torch.save(get_peft_model_state_dict(model), out / "adapter.pt")
    with open(out / "final_eval_responses.jsonl", "w") as f:
        for o in outs:
            f.write(json.dumps({"len": len(o.tokens), "finish": o.finish, "text": o.text}) + "\n")
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    main()
