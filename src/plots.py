"""Figures for the README, drawn from the committed Kaggle seed 0 artifacts only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "kaggle"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
C = {"grpo": "#2a78d6", "drgrpo": "#eb6834"}          # validated categorical slots 1 and 2
NAME = {"grpo": "GRPO", "drgrpo": "Dr. GRPO"}
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def steps(arm):
    return [json.loads(l) for l in (RUNS / f"{arm}_s0" / "log.jsonl").read_text().splitlines() if l.strip()]


def roll(x, k=5):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid"), np.arange(k, len(x) + 1)


def length_curves():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for arm in ("grpo", "drgrpo"):
        s = steps(arm)
        y = [r["len_mean"] for r in s]
        ax.plot(range(1, len(y) + 1), y, color=C[arm], lw=1, alpha=0.25)
        m, x = roll(y)
        ax.plot(x, m, color=C[arm], lw=2, label=NAME[arm])
        ax.annotate(NAME[arm], (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=10)
    ax.set_xlabel("training step"); ax.set_ylabel("mean response length (tokens)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_xlim(0, 108)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Training response length, same 16 questions per step\n(5 step rolling mean; raw per step behind)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "length_by_step.png", dpi=160); plt.close(fig)


def paired_diff():
    g, d = steps("grpo"), steps("drgrpo")
    diff = np.array([a["len_mean"] - b["len_mean"] for a, b in zip(g, d)])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(range(1, len(diff) + 1), diff, color=INK, lw=1, alpha=0.25)
    m, x = roll(diff)
    ax.plot(x, m, color=INK, lw=2)
    # labels sit where the curve is far below zero, so they never touch it
    ax.text(58, 25, "above 0: GRPO longer (what the paper predicts)", color=MUTED, fontsize=9, va="bottom")
    ax.text(58, -25, "below 0: Dr. GRPO longer", color=MUTED, fontsize=9, va="top")
    ax.set_xlabel("training step"); ax.set_ylabel("GRPO minus Dr. GRPO (tokens)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Paired length difference per step (5 step rolling mean)", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "paired_length_difference.png", dpi=160); plt.close(fig)


def final_eval():
    a = json.loads((RUNS / "analysis_seed0.json").read_text())[0]["final_eval_paired"]
    groups = ["correct answers", "wrong answers"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w, xs = 0.36, np.arange(2)
    for k, arm in enumerate(("grpo", "drgrpo")):
        vals = [a[f"len_correct_{arm}"], a[f"len_incorrect_{arm}"]]
        mean = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        pos = xs + (k - 0.5) * (w + 0.02)
        ax.bar(pos, mean, w, color=C[arm], label=f"{NAME[arm]}  ({a[f'accuracy_{arm}']:.1%} accurate)")
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for p, v, hi in zip(pos, mean, [v[2] for v in vals]):
            ax.text(p, hi + 14, f"{v:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_axisbelow(True)
    ax.set_xticks(xs, groups); ax.set_ylabel("mean length (tokens)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylim(0, 820)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Final eval, 500 MATH500 problems, greedy (95% bootstrap intervals)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "final_eval_lengths.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    length_curves(); paired_diff(); final_eval()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
