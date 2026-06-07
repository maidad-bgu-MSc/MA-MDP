# RESCO Multi-Agent Traffic Control Benchmarks (Cologne3 & Grid4x4)

This extension integrates two standard environments from the **RESCO (Reinforcement Learning Benchmarks for Traffic Signal Control)** suite into the PettingZoo / `sumo-rl` pipeline:
1. **`cologne3`**: A realistic 3-intersection arterial corridor in Cologne, Germany, simulating a morning peak rush hour.
2. **`grid4x4`**: A synthetic $4 \times 4$ grid containing 16 intersections with `traffic_light_right_on_red` signals.

Both scenarios are formulated as Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) utilizing a 4D look-ahead queue observation function and a synchronized global reward structure.

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

## 📝 Key Findings & Zero-Reward Resolution

### 1. Cologne3 Zero-Reward Bug
During initial training, we observed that all algorithms on `cologne3` evaluated to exactly `0.00` reward across all steps.
*   **Diagnosis**: Vehicle trip starts in `cologne3.rou.xml` only begin at **`23512.0` seconds** (around 6:32 AM morning rush hour). Because the simulation `begin_time` defaulted to `0.00`, standard runs of 1000s or 600s terminated before any vehicles could spawn in the network.
*   **Resolution**: Set `begin_time = 23500` for the `cologne3` scenario in [env_setup_resco.py](file:///c:/Users/advam/Documents/MA-MDP/simulator/env_setup_resco.py) to start the simulation 12 seconds prior to the first vehicle departure. Grid4x4 naturally starts departures at `0.00` and thus uses `begin_time = 0`.

### 2. Tabular Results on Grid4x4
The evaluation rewards (average negative waiting times) over a 600-second simulation run logged across 100 training epochs are summarized below:

| Epoch | Tabular IQL Return | Hysteretic Q-Learning Return | VDN Return | Fixed-Time Baseline Return |
| :--- | :---: | :---: | :---: | :---: |
| **5** | -927,360.00 | -946,910.00 | -960,257.00 | -919,449.00 |
| **20** | -962,168.00 | -942,889.00 | -986,298.00 | -919,449.00 |
| **25** | -950,267.00 | **-921,315.00** | -957,330.00 | -919,449.00 |
| **50** | -960,801.00 | -976,723.00 | **-939,101.00** | -919,449.00 |
| **80** | **-946,608.00** | -971,740.00 | -987,293.00 | -919,449.00 |
| **100** | -1,013,757.00 | -952,222.00 | -950,349.00 | -919,449.00 |
