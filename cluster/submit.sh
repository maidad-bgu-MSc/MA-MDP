#!/usr/bin/env bash
# Submit the experiment array, then chain the aggregation job to run after all
# array tasks succeed (afterok). Run from the repo root: bash cluster/submit.sh [env_name]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ENV_NAME=${1:-mmdp}

ARRAY_ID=$(sbatch --parsable cluster/run_array.sbatch "$ENV_NAME")
echo "Submitted array job: $ARRAY_ID  (5 seeds x 6 scenarios = 30 tasks)"

AGG_ID=$(sbatch --parsable --dependency=afterok:"${ARRAY_ID}" cluster/aggregate.sbatch "$ENV_NAME")
echo "Submitted aggregate job: $AGG_ID  (runs after the array completes)"

echo
echo "Monitor:        squeue -j ${ARRAY_ID},${AGG_ID}"
echo "Per-task logs:  logs/seed<seed>_<scenario>.log   (+ SLURM: logs/job_${ARRAY_ID}_*.txt)"
echo "Final outputs:  outputs/aggregated/  (summary.md, cross_algorithm_bars.png, per-scenario PNGs)"
echo "Then build deck: python make_presentation.py"
