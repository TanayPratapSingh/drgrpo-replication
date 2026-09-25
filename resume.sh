#!/usr/bin/env bash
# Continue the experiment from its latest checkpoint, in the background, so the
# terminal can be closed. Pass SEEDS="0 1 2" to extend to more seeds.
cd "$(dirname "$0")"
if pgrep -f "src/train.py" >/dev/null; then
  echo "A run is already in progress; nothing to resume."; exit 1
fi
rm -f PAUSE
nohup ./run_all.sh >> logs/matrix.log 2>&1 &
echo "Resumed in the background (runner pid $!). Watch progress with: tail -f logs/*_s*.log"
