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

from template import R1_PREFIX, STOP_STR, r1_prompt  # noqa: F401  (re-exported)


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
