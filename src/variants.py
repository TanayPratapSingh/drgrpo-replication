"""The four objective variants and the group advantage, framework free.

Modification 2 of the paper (difficulty bias), from compute_monte_carlo_advantages
in the authors' train_zero_math.py:

    advantages = rewards - group_mean
    if critic_type == "grpo":
        advantages = advantages / (group_std + 1e-8)      # torch .std(): unbiased

The four variants are the four arms of the paper's Appendix C ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Variant:
    name: str
    std_norm: bool  # divide the centred reward by the group std
    length_norm: bool  # divide each response's summed loss by its own length


VARIANTS = {
    "grpo": Variant("grpo", std_norm=True, length_norm=True),
    "drgrpo": Variant("drgrpo", std_norm=False, length_norm=False),
    "grpo_no_len": Variant("grpo_no_len", std_norm=True, length_norm=False),
    "grpo_no_std": Variant("grpo_no_std", std_norm=False, length_norm=True),
}


def group_advantages(rewards: np.ndarray, group_size: int, std_norm: bool) -> np.ndarray:
    """One scalar advantage per response, shared by all of its tokens."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    adv = r - r.mean(axis=1, keepdims=True)
    if std_norm:
        adv = adv / (r.std(axis=1, ddof=1, keepdims=True) + 1e-8)
    return adv.reshape(-1)
