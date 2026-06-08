# RESCO Multi-Agent Traffic Control Benchmarks (Cologne3 & Grid4x4)

This extension integrates two standard environments from the **RESCO (Reinforcement Learning Benchmarks for Traffic Signal Control)** suite into the PettingZoo / `sumo-rl` pipeline:
1. **`cologne3`**: A realistic 3-intersection arterial corridor in Cologne, Germany, simulating a morning peak rush hour.
2. **`grid4x4`**: A synthetic $4 \times 4$ grid containing 16 intersections with `traffic_light_right_on_red` signals.

Both scenarios are formulated as a cooperative **Multi-agent MDP (MMDP)**: a 4D look-ahead queue
observation (local + rest-of-network queues, approximating the global state) with a single
synchronized team reward. We evaluate cooperative MARL algorithms (IQL, Hysteretic-Q, VDN) on them.

---

## 📂 Newly Added Files

*   **[`simulator/download_resco_scenarios.py`](file:///c:/Users/advam/Documents/MA-MDP/simulator/download_resco_scenarios.py)**: Auto-downloads the RESCO network and route files, extracts the `grid4x4_1.rou.xml` route from the compressed zip archive, and places them under `simulator/resco_environments/`.
*   **[`simulator/env_setup_resco.py`](file:///c:/Users/advam/Documents/MA-MDP/simulator/env_setup_resco.py)**: Configures both PettingZoo turn-based AEC (`make_resco_env`) and parallel (`make_resco_parallel_env`) environments. Implements a robust TraCI-based geometric classifier (`is_vertical_lane`) to categorize lane orientations.
*   **[`run_resco_tabular_experiment.py`](file:///c:/Users/advam/Documents/MA-MDP/run_resco_tabular_experiment.py)**: Trains Tabular Independent Q-Learning (IQL), Hysteretic Q-Learning, and Value Decomposition Networks (VDN) alongside a Fixed-Time baseline on both scenarios.
*   **[`plot_resco_results.py`](file:///c:/Users/advam/Documents/MA-MDP/plot_resco_results.py)**: Reads evaluation logs and generates comparative learning curves and cross-algorithm bar charts.
*   **[`watch_resco_agents.py`](file:///c:/Users/advam/Documents/MA-MDP/watch_resco_agents.py)**: Loads trained policy Q-tables and runs them visually in the SUMO-GUI.
*   **[`tests/test_resco_env.py`](file:///c:/Users/advam/Documents/MA-MDP/tests/test_resco_env.py)**: Statically asserts scenario downloads and executes AEC env loop tests.

---

## 🚀 Usage Instructions

### 1. Download Scenario Assets
Retrieve the SUMO `.net.xml` and `.rou.xml` files directly from the RESCO repository:
```bash
python simulator/download_resco_scenarios.py
```
*(Note: If you run any experiment or environment setup script directly, it will download the missing files automatically).*

### 2. Run Tabular Experiments
Train and evaluate Tabular IQL, Hysteretic Q-Learning, and VDN agents:
```bash
python run_resco_tabular_experiment.py
```
*   Training logs are written to `resco_tabular_training.log`.
*   Evaluation CSV results are saved under `outputs/training_evaluation_log_cologne3.csv` and `outputs/training_evaluation_log_grid4x4.csv`.
*   Learning curves and cross-algorithm bar charts are saved to `outputs/resco_tabular_learning_curves.png` and `outputs/resco_cross_algorithm_bar.png` respectively.

### 3. Visualize Trained Agents in SUMO-GUI
Run SUMO-GUI to watch the pre-trained agents control the intersections:
```bash
python watch_resco_agents.py --scenario cologne3 --algo iql_tabular --delay 0.1
```
*(Options: `--scenario` can be `cologne3` or `grid4x4`; `--algo` can be `iql_tabular`, `hysteretic`, or `fixed`)*

---

## 📝 Bugs Found & Fixed

### 1. Cologne3 Zero-Reward Bug
Initially all algorithms on `cologne3` evaluated to exactly `0.00` reward.
*   **Diagnosis**: Vehicle departures in `cologne3.rou.xml` only begin at **`23512.0` s** (~6:32 AM rush hour); with `begin_time = 0` the simulation ended before any vehicle spawned.
*   **Resolution**: `begin_time = 23500` for `cologne3` in `simulator/env_setup_resco.py` (grid4x4 starts at `0`).

### 2. No-learning Bug (action-space mismatch) — the big one
Every algorithm sat at ~the fixed-time baseline with **no learning trend** (e.g. grid4x4 ~-950k vs -919k).
*   **Diagnosis**: the tabular agents were built with the default `num_actions=2`, but real RESCO junctions
    have **more green phases** — `cologne3` has 3-4 and **`grid4x4` has 8**. The agents could therefore only
    ever select 2 of up to 8 phases, starving the remaining movements and gridlocking the network.
*   **Resolution**: size each agent's action space to its own junction via `env.action_space(agent).n`
    (`TabularVDNAgents` now accepts per-agent action counts).

### 3. Observation saturation (grid4x4)
*   **Diagnosis**: the "rest-of-network" queue summed over all 15 neighbours, which almost always hit the
    top discretisation bin (30+), making 2 of the 4 state dimensions constant and uninformative.
*   **Resolution**: average the rest-of-network queue **per neighbouring junction** before discretising.

### 4. Stronger, fairer baselines
The fixed-time baseline now reports **two** plans: the **SUMO-native** signal program (`fixed_ts=True`,
the canonical RESCO baseline) and a **round-robin all-phases** controller. The old EW/NS controller only
exercised 2 phases — itself a victim of bug #2.

### Post-fix sanity check
After the fixes, `cologne3` IQL improves from **-124k -> -30k in 15 episodes**, converging toward the
SUMO-native fixed-time baseline (~-26k) — a real learning curve where there was previously none.
**Validated multi-seed (5-seed) results are produced by the reproducibility pipeline below and written to
`outputs/aggregated/`.**

## 🔁 Reproducibility pipeline
All experiments are seeded and run as a **5-seed × 6-scenario SLURM job array**:
```bash
bash cluster/setup_env.sh      # one-time: venv + SUMO (eclipse-sumo wheel) + deps
bash cluster/submit.sh         # 30-task array, then chained aggregation
python make_presentation.py    # build the deck from outputs/aggregated/
```
Locally: `python run_seeded_experiment.py --seed 0 --scenario grid4x4 [--quick]` then
`python aggregate_seeds.py`. See [`cluster/README.md`](cluster/README.md).
