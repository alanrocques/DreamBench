#!/bin/bash
# Train DreamerV3 on a specified environment using NM512/dreamerv3-torch.
# Usage:
#   ./scripts/train_dreamerv3.sh breakout   # Atari Breakout
#   ./scripts/train_dreamerv3.sh crafter    # Crafter
#   ./scripts/train_dreamerv3.sh pong       # Atari Pong

set -e

TASK=${1:-breakout}
STEPS=${2:-500000}
D3_DIR="third_party/dreamerv3-torch"
CKPT_DIR="checkpoints/dreamerv3_${TASK}"

if [ ! -d "$D3_DIR" ]; then
    echo "Error: $D3_DIR not found. Run:"
    echo "  mkdir -p third_party && cd third_party && git clone https://github.com/NM512/dreamerv3-torch.git"
    exit 1
fi

echo "Training DreamerV3 on ${TASK} for ${STEPS} steps..."
echo "Checkpoint will be saved to: ${CKPT_DIR}"

case $TASK in
    breakout|pong|space_invaders|freeway|montezuma)
        CONFIG="atari100k"
        TASK_ARG="atari_${TASK}"
        ;;
    crafter)
        CONFIG="crafter"
        TASK_ARG="crafter_reward"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Supported: breakout, pong, space_invaders, freeway, montezuma, crafter"
        exit 1
        ;;
esac

cd "$D3_DIR"
python dreamer.py \
    --configs "$CONFIG" \
    --task "$TASK_ARG" \
    --logdir "../../${CKPT_DIR}" \
    --steps "$STEPS"

echo "Training complete. Checkpoint at: ${CKPT_DIR}/latest.pt"
echo ""
echo "Run benchmark:"
echo "  python scripts/run_benchmark.py model=dreamerv3_${TASK} env=atari"
