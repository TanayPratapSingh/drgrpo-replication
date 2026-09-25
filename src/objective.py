"""The two quantities Dr. GRPO changes, transcribed from the authors' code.

Reference: sail-sg/understand-r1-zero, train_zero_math.py, and oat's ops.py.

    Modification 2 (difficulty bias), compute_monte_carlo_advantages:
        advantages = rewards - group_mean
        if critic_type == "grpo":
            advantages = advantages / (group_std + 1e-8)      # torch .std(): unbiased

    Modification 1 (length bias), the masked aggregator:
        grpo   -> masked_mean(loss, mask, axis=1)  = sum(loss * mask) / mask.sum()
        drgrpo -> masked_sum(loss, mask, axis=1, constant_normalizer=generate_max_length)

    Then, per oat's PPO learner: pg_loss = aggregated.mean() over the minibatch.

The four variants below are the four arms of the paper's Appendix C ablation.
"""

from __future__ import annotations

import mlx.core as mx

from variants import VARIANTS, Variant, group_advantages  # noqa: F401  (re-exported)


def sequence_losses(
    logps: mx.array,
    old_logps: mx.array,
    mask: mx.array,
    adv: mx.array,
    length_norm: bool,
    budget: int,
    clip_eps: float = 0.2,
) -> mx.array:
    """PPO clipped surrogate per token, aggregated to one loss per response.

    logps, old_logps, mask: (B, T). adv: (B,). Returns (B,).
    """
    ratio = mx.exp(logps - old_logps)
    a = adv[:, None]
    unclipped = -a * ratio
    clipped = -a * mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    per_token = mx.maximum(unclipped, clipped) * mask
    if length_norm:
        return per_token.sum(axis=1) / mask.sum(axis=1)
    return per_token.sum(axis=1) / budget
