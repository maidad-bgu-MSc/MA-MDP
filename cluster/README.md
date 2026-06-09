# Cluster runbook (SLURM) — full multi-seed sweep

This is a standalone guide to reproduce all results on the BGU cluster from scratch.
It runs the experiment matrix as a job array: **5 seeds × 6 scenarios = 30 tasks**.

Scenarios: `baseline`, `dense_wave`, `cross_surge`, `split_rush` (1×4 custom) and
`cologne3`, `grid4x4` (RESCO). Per task:
- **1×4** → tabular IQL / Hysteretic / VDN **+ QMIX (CTDE)** vs coordinated fixed-time.
- **RESCO** → tabular IQL / Hysteretic / VDN vs **two** fixed-time baselines (SUMO-native + round-robin).

Everything is seeded and deterministic: each seed is a fixed trajectory, and results are
reported as mean ± std across the 5 seeds.

---

## 0. Prerequisites
- An account on the cluster with SSH access.
- The cluster uses SLURM and provides conda via `module load anaconda`.
- No system SUMO is needed — SUMO is installed from the `eclipse-sumo` pip wheel.

## 1. Get the code onto the cluster
```bash
ssh <user>@<cluster-login-host>
# clone into your home (or wherever you keep code)
git clone <REPO_URL> MA-MDP
cd MA-MDP
git checkout feature/resco-experiments     # branch with the RESCO + reproducibility work
```

## 2. Create the conda environment (one time)
```bash
bash cluster/setup_env.sh                   # creates env "mmdp"  (override: bash cluster/setup_env.sh myenv)
```
This runs `module load anaconda`, creates a Python 3.12 conda env, `pip install -r requirements.txt`
(incl. SUMO from the `eclipse-sumo` wheel), and prints a SUMO/torch verification line.

If you prefer to do it by hand:
```bash
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda create -y -n mmdp python=3.12
conda activate mmdp
pip install --upgrade pip wheel
pip install -r requirements.txt
python -c "import sumo, sumo_rl, torch; print('env OK')"
```

## 3. Submit the sweep (+ automatic aggregation)
From the repo root:
```bash
bash cluster/submit.sh                       # or: bash cluster/submit.sh myenv
```
This submits the 30-task array (`run_array.sbatch`) and then the aggregation job
(`aggregate.sbatch`) with an `afterok` dependency, so aggregation runs automatically
once **all** array tasks succeed. It prints both job IDs and how to monitor them.

> Submit one quick test task first if you want to be safe:
> `sbatch --array=0-0 cluster/run_array.sbatch` (runs seed 0 / baseline only).

## 4. Monitor
```bash
squeue -u $USER                              # all your jobs
squeue -j <ARRAY_ID>,<AGG_ID>                # the two jobs submit.sh printed
tail -f logs/seed0_baseline.log              # live progress of one task
```
- Per-task SLURM stdout: `logs/job_<ARRAY_ID>_<TASKID>.txt`
- Per-task experiment log: `logs/seed<seed>_<scenario>.log`

Array index → (seed, scenario):  `scenario = SCENARIOS[id % 6]`, `seed = id // 6`,
with `SCENARIOS = (baseline dense_wave cross_surge split_rush cologne3 grid4x4)`.

## 5. Results
After the array + aggregation finish:
- Per-seed CSVs: `outputs/seeds/training_evaluation_log_<scenario>_seed<seed>.csv`,
  `outputs/seeds/qmix_results_<scenario>_seed<seed>.csv`
- Per-seed models: `models/seed<seed>/`
- **Aggregated** (`outputs/aggregated/`): `summary.md` (mean ± std table),
  `cross_algorithm_bars.png`, and per-scenario learning-curve PNGs/CSVs with ±std bands.

If you ever need to (re)aggregate manually:
```bash
conda activate mmdp
python aggregate_seeds.py
```

## 6. Build the presentation
```bash
conda activate mmdp
python make_presentation.py                  # -> ATLC_MMDP_presentation.pptx (uses outputs/aggregated/)
```
Copy `ATLC_MMDP_presentation.pptx` back to your machine (e.g. `scp`).

---

## Tuning / troubleshooting
- **Walltime / memory**: defaults are `--time=24:00:00`, `--cpus-per-task=2`, `--mem-per-cpu=4G`
  in `run_array.sbatch`. `grid4x4` (16 agents) is the heaviest — raise `--time`/`--mem-per-cpu` if a
  task is killed (check the SLURM `.txt` for OOM/TIMEOUT).
- **Email**: `--mail-user` is preset; change it in `run_array.sbatch` / `aggregate.sbatch` if needed.
- **Partition**: preset to `--partition=main`; no `--account` is required on this cluster.
- **A task crashed**: its `logs/seed<seed>_<scenario>.log` ends with a full traceback. Fix and
  re-run just that task: `sbatch --array=<id> cluster/run_array.sbatch`, then `python aggregate_seeds.py`.
- **`FatalTraCIError: Could not connect` / "TraCI server already finished"**: SUMO couldn't launch its
  subprocess+TraCI socket, usually on a busy/shared node (port/process contention). `run_array.sbatch`
  sets `LIBSUMO_AS_TRACI=1` to run SUMO **in-process** (no subprocess/ports), which avoids this and is
  faster. If many tasks still pile onto the same nodes, throttle concurrency:
  `sbatch --array=0-29%4 cluster/run_array.sbatch` (at most 4 tasks at once).
- **Local smoke test** (no SLURM): `python run_seeded_experiment.py --seed 0 --scenario baseline --quick`
  then `python aggregate_seeds.py`.
