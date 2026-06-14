"""Unified, reproducible entrypoint for ONE (seed, scenario) run.

Designed to be launched once per SLURM array task. Given a seed and a scenario it:
  * sets global RNG seeds (numpy / torch / random) for reproducibility,
  * trains the appropriate algorithms for that scenario family:
       - 1x4 custom scenarios  -> tabular IQL/Hysteretic/VDN  +  QMIX (CTDE)
       - RESCO scenarios       -> tabular IQL/Hysteretic/VDN  +  QMIX (CTDE)  (vs both fixed baselines)
  * writes per-(scenario, seed) CSVs under outputs/seeds/ and models under models/seed<seed>/,
  * mirrors all stdout/stderr to logs/seed<seed>_<scenario>.log.

Aggregation across seeds (mean +/- std) is done separately by aggregate_seeds.py.

Examples
--------
    python run_seeded_experiment.py --seed 0 --scenario baseline
    python run_seeded_experiment.py --seed 3 --scenario grid4x4
    python run_seeded_experiment.py --seed 0 --scenario cologne3 --quick   # fast smoke test
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

from seeding import set_global_seeds

ONE_X_FOUR = ["baseline", "dense_wave", "cross_surge", "split_rush"]
RESCO = ["cologne3", "grid4x4"]
ALL_SCENARIOS = ONE_X_FOUR + RESCO

OUT_DIR = os.path.join("outputs", "seeds")
LOG_DIR = "logs"


class _Tee:
    """Duplicates writes to multiple streams so a mid-run crash leaves a full trace on disk."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _open_log(seed, scenario):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"seed{seed}_{scenario}.log")
    fh = open(path, "w", buffering=1, encoding="utf-8")
    fh.write(f"=== run_seeded_experiment.py seed={seed} scenario={scenario} "
             f"started {datetime.now().isoformat()} ===\n")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh


def resolve_job(job_id):
    """Map a SLURM array index to (seed, scenario): 5 seeds x 6 scenarios.
    scenario = ALL_SCENARIOS[job_id % 6];  seed = job_id // 6  (0..4)."""
    n = len(ALL_SCENARIOS)              # 6
    return job_id // n, ALL_SCENARIOS[job_id % n]


def main():
    parser = argparse.ArgumentParser(description="Run one (seed, scenario) experiment.")
    # Either pass --job-id (SLURM array index, decoded to seed+scenario) OR --seed + --scenario.
    parser.add_argument("--job-id", "--job_id", type=int, default=None, dest="job_id",
                        help="SLURM array task id; decoded to (seed, scenario) via resolve_job().")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None, choices=ALL_SCENARIOS)
    # Unified training budget: EVERY method (tabular IQL/Hysteretic/VDN and deep QMIX) on
    # EVERY scenario trains for the same number of episodes at the same 1000s horizon ->
    # identical environment-interaction budget (500 eps x 200 steps = 100k steps each) and
    # identical seeded 600s evaluation. This removes any "unequal training" disparity when
    # comparing methods (samples, not wall-clock, are the fair currency — QMIX is slower
    # per step but equally sampled). See run_tabular_experiment / run_ctde_experiment.
    parser.add_argument("--episodes", type=int, default=500, help="Tabular training episodes.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Tabular eval interval (epochs).")
    parser.add_argument("--qmix-episodes", type=int, default=500, help="QMIX training episodes.")
    parser.add_argument("--qmix-eval-interval", type=int, default=10, help="QMIX eval interval.")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny run for smoke testing (overrides episode counts).")
    parser.add_argument("--qmix-only", "--qmix_only", action="store_true", dest="qmix_only",
                        help="Train ONLY QMIX (skip the tabular methods). Use to backfill the "
                             "QMIX cells without recomputing existing tabular results.")
    args = parser.parse_args()

    if args.job_id is not None:
        args.seed, args.scenario = resolve_job(args.job_id)
    if args.seed is None or args.scenario is None:
        parser.error("provide either --job-id, or both --seed and --scenario")

    episodes = args.episodes
    eval_interval = args.eval_interval
    qmix_episodes = args.qmix_episodes
    qmix_eval_interval = args.qmix_eval_interval
    # QMIX uses the same 1000s horizon as the tabular methods (was 1800) so total
    # environment interactions match exactly across all methods.
    qmix_sim_seconds, qmix_eval_seconds = 1000, 600
    if args.quick:
        episodes, eval_interval = 4, 2
        qmix_episodes, qmix_eval_interval = 4, 2
        qmix_sim_seconds, qmix_eval_seconds = 200, 150

    log_fh = _open_log(args.seed, args.scenario)
    set_global_seeds(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        from run_ctde_experiment import run_qmix_scenario
        if args.scenario in ONE_X_FOUR:
            if not args.qmix_only:
                from run_tabular_experiment import run_scenario
                run_scenario(args.scenario, episodes=episodes, eval_interval=eval_interval,
                             seed=args.seed, out_dir=OUT_DIR)
            run_qmix_scenario(args.scenario, num_episodes=qmix_episodes,
                              eval_interval=qmix_eval_interval, seed=args.seed, out_dir=OUT_DIR,
                              sim_seconds=qmix_sim_seconds, eval_seconds=qmix_eval_seconds)
        elif args.scenario in RESCO:
            if not args.qmix_only:
                from run_resco_tabular_experiment import run_resco_scenario
                run_resco_scenario(args.scenario, episodes=episodes, eval_interval=eval_interval,
                                   seed=args.seed, out_dir=OUT_DIR)
            # QMIX (CTDE) now runs on RESCO too — same unified budget as the 1x4 path.
            run_qmix_scenario(args.scenario, num_episodes=qmix_episodes,
                              eval_interval=qmix_eval_interval, seed=args.seed, out_dir=OUT_DIR,
                              sim_seconds=qmix_sim_seconds, eval_seconds=qmix_eval_seconds)
        print(f"\n=== DONE seed={args.seed} scenario={args.scenario} "
              f"at {datetime.now().isoformat()} ===")
    except BaseException:
        print(f"\n!!! CRASHED seed={args.seed} scenario={args.scenario} "
              f"at {datetime.now().isoformat()} !!!")
        traceback.print_exc()
        raise
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()
    # sumo-rl/libsumo raises while CPython finalizes the interpreter (atexit teardown
    # of the in-process SUMO connection: "Exception ignored in sys.unraisablehook"),
    # so the process exits 120 *after* the run finished and every CSV/log was flushed
    # -- SLURM then marks an otherwise-successful task FAILED on pure shutdown noise.
    # main() has already completed and flushed all its work, and it re-raises on any
    # real error (so a genuine crash never reaches this line and still exits non-zero).
    # Hard-exit 0 here to skip the broken finalizer and give SLURM an honest code.
    os._exit(0)
