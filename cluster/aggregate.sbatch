#!/bin/bash
# Aggregate all per-seed CSVs into mean +/- std tables and plots.
# Submitted with an afterok dependency on the array by cluster/submit.sh,
# or run standalone after the array finishes.
#
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --job-name=mmdp_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/aggregate_%j.txt
#SBATCH --mail-user=yoyo123233@gmail.com
#SBATCH --mail-type=END,FAIL

ENV_NAME=${1:-mmdp}
# Absolute interpreter path; do NOT module-load/conda-activate (it pollutes
# PYTHONHOME/PYTHONPATH and makes $PY load the wrong site-packages). See run_array.sbatch.
PY="$HOME/.conda/envs/$ENV_NAME/bin/python"

cd "$SLURM_SUBMIT_DIR"
unset PYTHONHOME
export PYTHONPATH="$(pwd)"

"$PY" aggregate_seeds.py
echo "Aggregation complete -> outputs/aggregated/"
