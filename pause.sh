#!/usr/bin/env bash
# Ask the running experiment to stop cleanly: the current step finishes, a
# checkpoint is written, and the process exits. Nothing is lost.
cd "$(dirname "$0")"
touch PAUSE
echo "Pause requested. The current step will finish and checkpoint (a few minutes)."
echo "When 'paused after step N' appears in logs/, it is safe to close the lid or shut down."
echo "Resume any time with: ./resume.sh"
