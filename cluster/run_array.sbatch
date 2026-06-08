#!/bin/bash
# SLURM job array: one task per (seed, scenario).
#   5 seeds x 6 scenarios = 30 tasks (array 0-29).
#   run_seeded_experiment.py --job_id $SLURM_ARRAY_TASK_ID decodes:
#       scenario = ALL_SCENARIOS[id % 6];  seed = id // 6   (0..4)
# Each task writes its own log at logs/seed<seed>_<scenario>.log.
#
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --job-name=mmdp_array
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/job_%A_%a.txt
#SBATCH --mail-user=yoyo123233@gmail.com
#SBATCH --mail-type=ALL

ENV_NAME=${1:-mmdp}

echo "--- MMDP Array Start ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID  |  env: $ENV_NAME  |  host: $(hostname)"
date

module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p logs outputs/seeds models

# Reproducibility: PYTHONHASHSEED must be set BEFORE the interpreter starts.
export PYTHONHASHSEED=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python run_seeded_experiment.py --job_id "$SLURM_ARRAY_TASK_ID"

echo "--- MMDP Array Finish ---"
date
