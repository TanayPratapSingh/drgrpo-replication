#!/usr/bin/env bash
# The experiment matrix, run sequentially (two runs would not fit in 16 GB).
# Arms are interleaved per seed so that stopping early still leaves matched
# GRPO / Dr. GRPO pairs. Finished runs are skipped and unfinished ones resume
# from their latest checkpoint, so re-running this script picks up where it
# stopped. caffeinate -i keeps the Mac from idle sleeping mid step.
#
#   ./pause.sh     finish the current step, checkpoint, stop
#   ./resume.sh    continue from exactly that step
set -u
cd "$(dirname "$0")"
STEPS="${STEPS:-100}"
SEEDS="${SEEDS:-0}"          # SEEDS="0 1 2" for the paper's three seed robustness check
CKPT_EVERY="${CKPT_EVERY:-5}"
# LoRA LR: 50x the paper's full fine tuning 1e-6. 2e-5 left the format rate flat
# (0.18, 0.20, 0.17, 0.15) over the first four pilot steps, too slow for a 100
# step laptop budget. Chosen on format learning speed alone; identical in both arms.
LR="${LR:-5e-5}"
mkdir -p logs
for seed in $SEEDS; do
  for loss in grpo drgrpo; do
    name="${loss}_s${seed}"
    if [ -e PAUSE ]; then echo "$(date -u +%FT%TZ) paused before $name"; exit 0; fi
    if [ -s "runs/$name/final_eval_responses.jsonl" ]; then
      echo "$(date -u +%FT%TZ) skip $name (complete)"; continue
    fi
    echo "$(date -u +%FT%TZ) start $name"
    caffeinate -i .venv/bin/python src/train.py --loss "$loss" --seed "$seed" \
      --steps "$STEPS" --lr "$LR" --gen-batch 128 --resume --ckpt-every "$CKPT_EVERY" \
      >> "logs/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) end $name exit=$?"
    if [ -e PAUSE ]; then echo "$(date -u +%FT%TZ) paused after $name"; exit 0; fi
  done
done
echo "$(date -u +%FT%TZ) matrix finished"
