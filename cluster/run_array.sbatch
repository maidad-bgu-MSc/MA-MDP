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
# Call the env's interpreter by absolute path. `conda activate` is unreliable in a
# non-interactive batch shell on some compute nodes (it can silently leave the base/
# system python active -> ModuleNotFoundError for gymnasium/libsumo even though the
# package is installed in the env). $PY guarantees the right interpreter regardless.
PY="$HOME/.conda/envs/$ENV_NAME/bin/python"

echo "--- MMDP Array Start ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID  |  env: $ENV_NAME  |  host: $(hostname)"
date

module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME" || echo "WARN: conda activate failed; using \$PY directly"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p logs outputs/seeds models

# Fail loudly and early if the env interpreter is wrong/broken (instead of a cryptic
# ModuleNotFoundError seconds into the run).
echo "interpreter -> $PY"
if ! "$PY" -c "import gymnasium, sumo_rl" >/dev/null 2>&1; then
    echo "FATAL: $PY cannot import gymnasium/sumo_rl -- env '$ENV_NAME' missing or broken."
    "$PY" -c "import gymnasium, sumo_rl"   # surface the real traceback in the log
    exit 1
fi

# Reproducibility: PYTHONHASHSEED must be set BEFORE the interpreter starts.
export PYTHONHASHSEED=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Prefer libsumo (SUMO in-process) over spawning a sumo subprocess + TraCI socket on
# every env.reset(). On busy/shared nodes the per-reset subprocess+port dance fails
# ("TraCI server already finished" -> "Could not connect in 1 tries"); libsumo avoids
# it entirely and is much faster (same SUMO core + sumo_seed, so results are consistent).
# IMPORTANT: only enable it if libsumo actually imports -- sumo-rl does `import libsumo
# as traci` when LIBSUMO_AS_TRACI is set, which would crash every task if it's missing.
# If libsumo is absent we fall back to traci; throttle array concurrency (e.g. %4) to
# keep tasks off saturated nodes.
if "$PY" -c "import libsumo" >/dev/null 2>&1; then
    export LIBSUMO_AS_TRACI=1
    echo "libsumo available -> running SUMO in-process"
else
    unset LIBSUMO_AS_TRACI
    echo "libsumo NOT available -> using traci subprocess (throttle concurrency to avoid contention)"
fi

"$PY" run_seeded_experiment.py --job_id "$SLURM_ARRAY_TASK_ID"
status=$?

echo "--- MMDP Array Finish (exit $status) ---"
date
# Propagate the real exit code so a crashed task is marked FAILED (not COMPLETED),
# and the afterok aggregation dependency won't fire on partial results.
exit $status
