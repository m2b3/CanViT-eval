#!/bin/bash
# Train teacher depth probes at multiple resolutions for FLOP frontier.
# Run from canvit-eval-depth worktree on crockett.
set -euo pipefail

NYU_ROOT="/datasets/NYU/nyu"
STEPS=38400
EVAL_EVERY=4000
BATCH=4

for RES in 256 384; do
    GRID=$((RES / 16))
    DIR="checkpoints/depth-teacher-${RES}px"
    LOG="$HOME/depth_teacher_${RES}px.log"
    echo "=== Training teacher at ${RES}px (${GRID}×${GRID} grid) ==="
    uv run python depth_poc/train.py \
        --nyu-root "$NYU_ROOT" \
        --mode teacher \
        --scene-size "$RES" \
        --max-steps "$STEPS" \
        --eval-every "$EVAL_EVERY" \
        --batch-size "$BATCH" \
        --ckpt-dir "$DIR" \
        > "$LOG" 2>&1
    echo "Done. Best result:"
    grep "new best" "$LOG" | tail -1
    echo
done

echo "All teacher multi-res training complete."
