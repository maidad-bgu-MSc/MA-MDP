#!/usr/bin/env bash
# One-time conda environment setup on the cluster (run on a login node).
#
# SUMO comes from the `eclipse-sumo` pip wheel (headless binary + `sumo` python
# package); the project's setup_sumo_env() finds it at runtime, so no system SUMO
# module is needed. Matches the lab convention of `module load anaconda` + conda.
#
#   bash cluster/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=${1:-mmdp}

module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.12
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo "--- Verifying headless SUMO from the eclipse-sumo wheel ---"
python - <<'PY'
import os, subprocess, sumo
home = os.path.dirname(sumo.__file__)
os.environ["SUMO_HOME"] = home
print("SUMO_HOME ->", home)
print(subprocess.run([os.path.join(home, "bin", "sumo"), "--version"],
                     capture_output=True, text=True).stdout.splitlines()[0])
import sumo_rl, torch  # noqa: F401
print("sumo_rl + torch import OK")
PY

echo
echo "Conda env '${ENV_NAME}' ready. Submit experiments with:  bash cluster/submit.sh"
