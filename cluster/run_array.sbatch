#!/bin/bash
# SLURM job array: one task per (seed, scenario).
#   5 seeds x 6 scenarios = 30 tasks (array 0-29).
#   run_seeded_experiment.py --job_id $SLURM_ARRAY_TASK_ID decodes:
#       scenario = ALL_SCENARIOS[id % 6];  seed = id // 6   (0..4)
#
# Env handling mirrors the user's proven nomad_array launcher: module load
# anaconda -> source conda.sh -> conda activate -> plain `python`. (An earlier
# absolute-$PY-without-activation variant FATALed on `import gymnasium`; this
# activated pattern is the one that actually works on the BGU cluster.)
#
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --job-name=mmdp_array
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/job_%A_%a.txt
#SBATCH --mail-user=leashmil@post.bgu.ac.il
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

# Prefer libsumo (SUMO in-process) over spawning a sumo subprocess + TraCI socket
# on every env.reset(): on busy/shared nodes the per-reset subprocess+port dance
# fails ("Could not connect in 1 tries"). Only enable it if libsumo imports --
# sumo-rl does `import libsumo as traci` when LIBSUMO_AS_TRACI is set, which would
# crash every task if it's missing. Without it, throttle concurrency (e.g. %4).
if python -c "import libsumo" >/dev/null 2>&1; then
    export LIBSUMO_AS_TRACI=1
    echo "libsumo available -> running SUMO in-process"
else
    unset LIBSUMO_AS_TRACI
    echo "libsumo NOT available -> using traci subprocess (throttle concurrency to avoid contention)"
fi

python run_seeded_experiment.py --job_id "$SLURM_ARRAY_TASK_ID"
status=$?

echo "--- MMDP Array Finish (exit $status) ---"
date
# Propagate the real exit code so a crashed task is marked FAILED (not COMPLETED)
# and the afterok aggregation dependency won't fire on partial results.
exit $status