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
#SBATCH --mail-user=leashmil@post.bgu.ac.il
#SBATCH --mail-type=END,FAIL

ENV_NAME=${1:-mmdp}
# Mirror run_array.sbatch's proven env handling: module load -> conda activate ->
# plain `python` (the activated pattern that actually works on the BGU cluster).
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"

python aggregate_seeds.py
echo "Aggregation complete -> outputs/aggregated/"
