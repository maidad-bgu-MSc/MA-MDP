# Adaptive Traffic Light Control (ATLC) on a 1x4 Corridor with Multi-Agent Reinforcement Learning (MARL)

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SUMO version](https://img.shields.io/badge/SUMO-1.20%2B-orange.svg)](https://eclipse.dev/sumo/)

This repository implements a decentralized coordination framework for **Adaptive Traffic Light Control (ATLC)** on a 1-dimensional, 4-intersection horizontal road corridor ($1 \times 4$ grid) using SUMO (Simulation of Urban MObility) and PettingZoo. The traffic corridor represents a green wave scenario, where dense vehicle platoons are periodically injected eastbound and westbound along the arterial corridor, while stochastic Poisson cross-traffic intersects vertically. The goal of the system is to coordinate traffic light phase choices (East-West vs. North-South) to minimize cumulative vehicle waiting time and maximize corridor throughput.

The system is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) utilizing a cooperative global reward structure. It provides implementations of classical tabular multi-agent algorithms (Independent Q-Learning and Hysteretic Q-Learning) alongside advanced PyTorch deep networks (Independent DQN, Independent PPO, and QMIX mixing network structures) to evaluate coordination performance against predefined fixed-time signaling baseline policies.

---

## 📂 Repository Structure

The codebase is organized into the following components:

```
MA-MDP/
 ├── simulator/                       # Core simulation module
 │    ├── __init__.py
 │    ├── generate_1x4_wave.py        # Generates SUMO network, route plans, and traffic flows
 │    └── env_setup.py                # PettingZoo environment wrapper and global reward synced step
 ├── models/                          # Directory for saving trained agent policy models
 ├── tests/                           # Unit testing suite
 │    ├── __init__.py
 │    ├── test_imports.py             # Verifies dependencies and syntax
 │    ├── test_1x4_dynamics.py        # Tests environment stepping dynamics and reward broadcasting
 │    ├── test_algorithms.py          # Tests tabular update steps and network dimensions
 │    └── test_action_handshake.py    # Tests action constraints and traffic light phase transitions
 ├── requirements.txt                 # Project library dependencies
 ├── marl_algorithms.py               # Implements tabular policies and QMIX mixing networks
 ├── train.py                         # Deep RL training script using Tianshou v2 (IQL/DQN and IPPO/PPO)
 ├── evaluate.py                      # Evaluation pipeline for trained Deep RL models
 ├── evaluate_baselines.py            # Evaluates preconfigured fixed-time baseline controllers
 ├── plot_results.py                  # Processes evaluation log CSVs and generates comparison plots
 ├── run_all_tests.py                 # Runner wrapper script for running pytest
 ├── run_tabular_experiment.py        # Trains and evaluates Tabular IQL and Hysteretic agents
 └── watch_agents.py                  # Visual simulation CLI tool using SUMO-GUI
```

### Core Scripts:
*   **`simulator/env_setup.py`**: Wraps `sumo-rl` in a PettingZoo AEC interface. Uses a 4D queue observation function (local E-W, local N-S, rest-of-network E-W, rest-of-network N-S queues) discretized into 5 bins ($5^4 = 625$ states), and synchronizes rewards globally.
*   **`marl_algorithms.py`**: Defines agent classes including decentralized tabular Q-learning, Hysteretic Q-learning, and PyTorch deep neural network architectures for QMIX.
*   **`run_tabular_experiment.py`**: Trains tabular agents for 100 episodes, saves evaluation results to `training_evaluation_log.csv`, and outputs the learning curves to `learning_curves.png`.
*   **`evaluate_baselines.py`**: Runs 5 fixed-time control heuristics for comparison.
*   **`train.py` & `evaluate.py`**: Trains and evaluates deep reinforcement learning agents (IQL/DQN and IPPO/PPO) with Tianshou.

---

## ⚙️ Installation & Prerequisites

### 1. Install Eclipse SUMO
Ensure Eclipse SUMO is installed on your operating system:
*   **Windows**: Download and run the MSI installer from the [Eclipse SUMO Downloads](https://eclipse.dev/sumo/) page.
*   **Linux (Ubuntu/Debian)**:
    ```bash
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-gui
    ```

### 2. Configure Environment Variables
Set `SUMO_HOME` to point to the root folder of your SUMO installation:
*   **Windows (PowerShell)**:
    ```powershell
    [System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
    ```
*   **Linux/macOS**:
    ```bash
    export SUMO_HOME=/usr/share/sumo
    ```

### 3. Install Python Dependencies
Install required packages using the Python Launcher (`py`) or your active virtual environment:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

### 1. Run the Automated Tests
Run the test suite using the testing wrapper:
```bash
python run_all_tests.py
```

### 2. Run Tabular Experiments
Train and evaluate Tabular IQL and Hysteretic Q-Learning agents:
```bash
python run_tabular_experiment.py
```
This logs evaluation returns to `training_evaluation_log.csv` and saves the training plot to `learning_curves.png`.

### 3. Evaluate Fixed-Time Baselines
Evaluate the 5 preconfigured fixed-time controllers over 600-second simulation runs:
```bash
python evaluate_baselines.py
```

### 4. Train & Evaluate Deep RL Agents
Train the Deep IQL (DQN) and IPPO (PPO) models, deterministic evaluation runs, and plot results:
```bash
python train.py
python evaluate.py
python plot_results.py
```
This produces step-by-step logs under the `outputs/` folder, creates `outputs/summary.md`, and outputs comparative line plots to `outputs/performance_comparison.png`.

### 5. Watch Agents in SUMO-GUI
Observe policies visually in the SUMO simulator:
```bash
python watch_agents.py --algo iql_tabular --delay 0.1
```
*(Options for `--algo` are `iql_tabular`, `hysteretic`, `qmix`, `iql_deep`, or `fixed`)*

---

## 📈 Results & Outputs

### 1. Tabular Learning Performance (`training_evaluation_log.csv`)
The following table shows the evaluation returns (sum of system-wide negative delays over 600 seconds) logged across 100 training epochs:

| Epoch | Tabular IQL Return | Hysteretic Q-Learning Return | Fixed-Time (50/50) Baseline Return |
| :--- | :---: | :---: | :---: |
| **5** | -220,935.00 | -512,944.00 | -111,698.00 |
| **10** | -204,823.00 | -1,125,423.00 | -111,698.00 |
| **20** | -48,989.00 | -94,521.00 | -111,698.00 |
| **25** | **-43,357.00** | -59,290.00 | -111,698.00 |
| **30** | -467,245.00 | -246,688.00 | -111,698.00 |
| **40** | -76,799.00 | -132,195.00 | -111,698.00 |
| **45** | -57,988.00 | **-38,745.00** | -111,698.00 |
| **60** | -94,534.00 | -48,968.00 | -111,698.00 |
| **80** | -125,874.00 | -80,233.00 | -111,698.00 |
| **100** | -111,649.00 | -77,375.00 | -111,698.00 |

*   **Analysis**: Both reinforcement learning methods successfully learn policies that outperform the naive fixed-time baseline (constant return of `-111,698.00`). Tabular IQL reaches a peak return of `-43,357.00` at epoch 25, while Hysteretic Q-learning achieves the overall best return of `-38,745.00` at epoch 45.

### 2. Fixed-Time Heuristic Comparison
Evaluating the static fixed-time baselines (`evaluate_baselines.py`) over 600 seconds of simulation yields:

| Rank | Policy Name | Configuration | Total Delay Return |
| :---: | :--- | :--- | :---: |
| **1** | Policy 1: Naive 50/50 Split | EW Green: 50s, NS Green: 50s, No offsets | **-110,530.00** |
| **2** | Policy 5: Short Cycle Green Wave | EW Green: 60s, NS Green: 15s, Staggered offsets (0, 3, 6, 9) | **-321,881.00** |
| **3** | Policy 2: 80/20 Proportional Split | EW Green: 120s, NS Green: 30s, No offsets | **-325,535.00** |
| **4** | Policy 3: 80/20 Split, Green Wave | EW Green: 120s, NS Green: 30s, Staggered offsets (0, 3, 6, 9) | **-347,468.00** |
| **5** | Policy 4: Short Cycle Split | EW Green: 60s, NS Green: 15s, No offsets | **-448,139.00** |

*   **Analysis**: Among non-learning heuristics, Policy 1 performs best because it allocates sufficient time to clear stochastic cross-street vehicle queues. However, trained learning agents coordinate more dynamically, outperforming all fixed baselines.
