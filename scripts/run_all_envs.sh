#!/bin/bash
set -e

for env in minigrid crafter atari; do
    echo "========== $env =========="
    python scripts/run_benchmark.py model=mock env=$env model.noise_std=0
    python scripts/generate_report.py \
        --results "results/$env" \
        --output "reports/${env}_mock.html"
    echo ""
done

echo "All environments passed with mock adapter (noise_std=0)."
