"""PyTorch twin of objective.sequence_losses, for the CUDA trainer.

Same formula as the MLX version and as the authors' oat implementation:
PPO clipped surrogate per token, then either a per response mean (GRPO, the
1/|o| term) or a sum divided by a constant generation budget (Dr. GRPO).
"""

from __future__ import annotations

import torch

from variants import VARIANTS, Variant, group_advantages  # noqa: F401  (re-exported)


def sequence_losses(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor,
    adv: torch.Tensor,
    length_norm: bool,
    budget: int,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """logps, old_logps, mask: (B, T). adv: (B,). Returns one loss per response (B,)."""
    ratio = torch.exp(logps - old_logps)
    a = adv[:, None]
    unclipped = -a * ratio
    clipped = -a * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    per_token = torch.maximum(unclipped, clipped) * mask
    if length_norm:
        return per_token.sum(dim=1) / mask.sum(dim=1)
    return per_token.sum(dim=1) / budget
