#!/usr/bin/env bash
# The experiment matrix, run sequentially (two runs would not fit in 16 GB).
# Arms are interleaved per seed so that stopping early still leaves matched
# GRPO / Dr. GRPO pairs. A run that already finished its final eval is skipped,
# so re-running this script resumes the matrix. caffeinate -i keeps the Mac from
# idle sleeping, which would otherwise suspend a run mid step.
set -u
cd "$(dirname "$0")"
STEPS="${STEPS:-100}"
SEEDS="${SEEDS:-0 1 2}"
# LoRA LR: 50x the paper's full fine tuning 1e-6. 2e-5 left the format rate flat
# (0.18, 0.20, 0.17, 0.15) over the first four pilot steps, too slow for a 100
# step laptop budget. Chosen on format learning speed alone; identical in both arms.
LR="${LR:-5e-5}"
for seed in $SEEDS; do
  for loss in grpo drgrpo; do
    name="${loss}_s${seed}"
    if [ -s "runs/$name/final_eval_responses.jsonl" ]; then
      echo "$(date -u +%FT%TZ) skip $name (complete)"; continue
    fi
    echo "$(date -u +%FT%TZ) start $name"
    caffeinate -i .venv/bin/python src/train.py --loss "$loss" --seed "$seed" \
      --steps "$STEPS" --lr "$LR" --gen-batch 128 > "logs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) end $name exit=$?"
  done
done
echo "$(date -u +%FT%TZ) matrix finished"
