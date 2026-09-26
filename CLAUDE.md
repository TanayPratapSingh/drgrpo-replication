# Dr. GRPO replication: working context

Read this first in any new session. It holds the state and the decisions that
are not recoverable from the code alone.

## What this is

A laptop replication of Liu et al. 2025, "Understanding R1-Zero-Like Training:
A Critical Perspective" (arXiv 2503.20783). The claim under test: GRPO's
per response length normalisation (1/|o|) and per question std normalisation
bias training so that response length inflates, especially for incorrect
answers. Dr. GRPO removes both terms. Paper's Figure 5 is the target:
Qwen2.5-1.5B base, R1 template, MATH questions, binary answer tag reward.

## Current state (update this section at every checkpoint)

- **Seed 0 is complete on Kaggle (4.35 GPU h) and written up.** GRPO 51.2% vs
  Dr. GRPO 51.6% on MATH500; Dr. GRPO produced LONGER responses (paired +99
  tokens/step, +239 last third; wrong answers 680 vs 545). Accuracy claim holds,
  both length claims reversed. README, figures and runs/kaggle/ hold everything.
- The user declined seeds 1 and 2 ("too much"). Do not relaunch without asking.
- Possible next steps, only on request: portfolio entry, walkthrough video.
- Public repo: https://github.com/TanayPratapSingh/drgrpo-replication.

## How to operate

Kaggle (current): see the state section above. The laptop commands below still
work for the MLX pilot.

```bash
./resume.sh              # continue from the latest checkpoint, ask mode
ASK=0 ./resume.sh        # continue unattended to the end
./pause.sh               # finish the current step, checkpoint, stop
tail -f logs/grpo_s0.log # progress
```

Never run two trainers at once: 16 GB cannot hold both. resume.sh refuses if
one is already running.

## Decisions that must not be silently reversed

1. **Model 1.5B, not 0.5B.** Qwen2.5-0.5B base scored 0 of 128 on MATH and 1 of
   128 on GSM8K under the strict R1 format, so GRPO gets zero gradient. 1.5B is
   also the paper's Figure 5 model.
2. **LoRA (rank 64, scale 1.0, all 7 projections)** instead of full fine tuning:
   fp32 AdamW on 1.5B needs ~24 GB, and bf16 would lose 1e-6 updates.
3. **LR 5e-5.** At 2e-5 the format rate stayed flat (0.18, 0.20, 0.17, 0.15) over
   four pilot steps. Chosen on format learning speed only, identical in both arms.
4. **Budget 1024 tokens** (paper 3000). By steps 31 to 35 about 10% of GRPO
   responses were truncated at the cap. Report truncation separately; it caps
   the very length growth being measured.
5. **One on-policy optimizer step per 128 response rollout.** The paper takes 8
   steps per 1024 response rollout. Neither bias term depends on this.
6. **Memory:** `mx.set_cache_limit(1 GB)` and `mx.set_wired_limit(9 GB)`. Without
   wiring, macOS paged Metal buffers and a Metal GPU timeout killed pilot 1.
7. **The grader is the authors' own, vendored unmodified** (MIT, src/vendor/).
   Training and eval data are the authors' own arrow files (src/fetch_data.py).

## Corrections already made, do not repeat the mistakes

- Runs are NOT bit for bit deterministic after step 1. Step 1 rollouts are
  identical across runs and arms, but the bf16 backward pass on the GPU is not
  reproducible, so trajectories drift after the first update. What holds:
  both arms see identical questions at every step, identical first rollouts,
  and checkpoint restore is exact (tests/test_checkpoint.py).
- Therefore one seed pair cannot separate the effect from run to run noise.
  The paper used 3 seeds (Figure 9). Recommend seeds 1 and 2 after seed 0.

## GRPO seed 0 so far (5 step windows)

| steps | reward | len | correct | incorrect | truncated/128 |
|---|---|---|---|---|---|
| 6-10 | 0.173 | 225 | 158 | 236 | 2.2 |
| 16-20 | 0.298 | 293 | 165 | 348 | 2.6 |
| 26-30 | 0.328 | 384 | 262 | 444 | 5.8 |
| 31-35 | 0.381 | 483 | 324 | 587 | 12.4 |

Held out MATH500 (100 problems, greedy): 2% at step 0, 38% at step 25.
This is one arm. It is not a finding until Dr. GRPO on the same questions exists.

## Kaggle lessons (each cost a smoke run)

- Datasets mount at /kaggle/input/datasets/<user>/<slug>/, flattened; search recursively.
- Kaggle ships torchao 0.10, which peft refuses; uninstall it in the kernel.
- torch says bf16 is supported on a T4 (emulated); key bf16 on compute capability >= 8.
- vLLM per request seeds sampled one request at a time: 350 tok/s. Without them: ~1000.
- Build merged weights as a generator; materialising all of them OOMed the sampler GPU.
- vLLM logprobs=0 returned nothing on 0.30; logprobs=1 works.

## What is left

1. Finish GRPO s0 and Dr. GRPO s0 on Kaggle (both arms on the same backend).
2. `src/analyze.py`: paired per step length difference with bootstrap CIs,
   slopes, final eval incorrect length (claims C1 to C3 in its docstring).
   Extend it to regrade final_eval_responses.jsonl per problem.
3. Charts (load the dataviz skill first), README with deviations table,
   commits, optionally seeds 1 and 2, then the portfolio entry.

## House rules (from the user, apply everywhere)

No em dashes. No hyphens in compound phrases in prose. Never invent a metric:
every number traces to a file in runs/. Atomic commits. Ask before pushing.
