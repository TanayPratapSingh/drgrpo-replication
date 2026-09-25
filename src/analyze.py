"""Compare GRPO and Dr. GRPO runs on the paper's claims.

Both arms of a seed see the same questions at every step (same data order, same
per step sampling seed), so step level differences are matched pairs rather
than two independent noisy curves. Claims tested, from the paper:

  C1  GRPO response length grows more than Dr. GRPO's during training (Fig 5, plot 2)
  C2  Dr. GRPO's incorrect responses are shorter on evaluation   (Fig 5, plot 4)
  C3  Dr. GRPO maintains reasoning performance                    (Fig 5, abstract)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
rng = np.random.default_rng(0)


def load(name):
    d = RUNS / name
    if not (d / "log.jsonl").exists():
        return None
    steps = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
    evals = [json.loads(l) for l in (d / "eval.jsonl").read_text().splitlines() if l.strip()]
    final = []
    if (d / "final_eval_responses.jsonl").exists():
        final = [json.loads(l) for l in (d / "final_eval_responses.jsonl").read_text().splitlines() if l.strip()]
    return {"steps": steps, "evals": evals, "final": final}


def boot_ci(x, n=10_000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def slope(y):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else float("nan")


def compare(seed):
    g, d = load(f"grpo_s{seed}"), load(f"drgrpo_s{seed}")
    if g is None or d is None:
        return None
    n = min(len(g["steps"]), len(d["steps"]))
    gl = np.array([s["len_mean"] for s in g["steps"][:n]])
    dl = np.array([s["len_mean"] for s in d["steps"][:n]])
    diff = gl - dl
    third = max(n // 3, 1)
    out = {
        "seed": seed,
        "steps_compared": n,
        "len_mean_grpo_last_third": float(gl[-third:].mean()),
        "len_mean_drgrpo_last_third": float(dl[-third:].mean()),
        "paired_len_diff_all_steps": boot_ci(diff),
        "paired_len_diff_last_third": boot_ci(diff[-third:]),
        "slope_len_grpo_per_step": slope(gl),
        "slope_len_drgrpo_per_step": slope(dl),
        "reward_grpo_last_third": float(np.mean([s["reward"] for s in g["steps"][n - third:n]])),
        "reward_drgrpo_last_third": float(np.mean([s["reward"] for s in d["steps"][n - third:n]])),
    }
    # incorrect response length during training, where both arms had incorrect responses
    gi = [s["len_incorrect"] for s in g["steps"][:n]]
    di = [s["len_incorrect"] for s in d["steps"][:n]]
    pairs = [(a, b) for a, b in zip(gi, di) if a is not None and b is not None]
    out["paired_incorrect_len_diff_all_steps"] = boot_ci([a - b for a, b in pairs])
    # final evaluation, paired by problem (same 500 problems, same order)
    if g["final"] and d["final"]:
        fe_g, fe_d = g["evals"][-1], d["evals"][-1]
        out["final_eval"] = {
            "grpo": {k: fe_g[k] for k in ("n", "accuracy", "len_mean", "len_correct", "len_incorrect", "truncated")},
            "drgrpo": {k: fe_d[k] for k in ("n", "accuracy", "len_mean", "len_correct", "len_incorrect", "truncated")},
        }
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    results = [r for r in (compare(s) for s in seeds) if r]
    print(json.dumps(results, indent=2))
    (RUNS / "analysis.json").write_text(json.dumps(results, indent=2))
