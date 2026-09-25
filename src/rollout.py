"""Batched sampling from the policy, capturing exactly the tokens that were sampled.

Two details matter for correctness here.

1. The loss needs log probabilities of the tokens that were actually sampled.
   mlx_lm.batch_generate returns decoded text only, and re-tokenising decoded
   text does not always round trip, so this drives BatchGenerator directly and
   records every token id as it is produced.

2. The paper stops R1 template generations at the string "</answer>" and keeps
   it in the output (vLLM stop=["</answer>"], include_stop_str_in_output=True).
   Qwen's tokenizer emits that string as "</", " </" or "}</" followed by ">",
   ">\n" or ">\n\n" depending on context, so a fixed token sequence stop would
   miss most occurrences. Stopping is done on the decoded string instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

STOP_STR = "</answer>"

# Template 1 of the paper, copied from apply_r1_template in train_zero_math.py.
R1_PREFIX = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the User with the answer. The reasoning process is "
    "enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer "
    "here </answer>.\nUser: "
)


def r1_prompt(question: str) -> str:
    return R1_PREFIX + question + "\nAssistant: <think>"


@dataclass
class Rollout:
    tokens: list[int]  # response tokens exactly as sampled, final stop token included
    text: str
    finish: str  # "answer" (hit </answer>), "eos", or "length" (hit the budget)


def sample(
    model,
    tokenizer,
    prompt_ids: list[list[int]],
    max_tokens: int,
    temperature: float,
    completion_batch_size: int = 64,
) -> list[Rollout]:
    """Generate one response per entry of prompt_ids, in order."""
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    eos = list(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    gen = BatchGenerator(
        model,
        max_tokens=max_tokens,
        stop_tokens=[[t] for t in eos],
        sampler=sampler,
        completion_batch_size=completion_batch_size,
        prefill_batch_size=min(16, completion_batch_size),
    )
    uids = gen.insert(prompt_ids, [max_tokens] * len(prompt_ids))
    toks: dict[int, list[int]] = {u: [] for u in uids}
    finish: dict[int, str] = {}

    while True:
        responses = gen.next_generated()
        if not responses:
            break
        done_by_string = []
        for r in responses:
            toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                finish[r.uid] = "eos" if r.finish_reason == "stop" else "length"
                continue
            # "</answer>" is at most three tokens; six gives margin for a merged prefix.
            if STOP_STR in tokenizer.decode(toks[r.uid][-6:]):
                finish[r.uid] = "answer"
                done_by_string.append(r.uid)
        if done_by_string:
            gen.remove(done_by_string)
    gen.close()

    return [
        Rollout(tokens=toks[u], text=tokenizer.decode(toks[u]), finish=finish.get(u, "length"))
        for u in uids
    ]
