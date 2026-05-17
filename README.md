# Adaptive Traffic Light Control (ATLC) using Multi-Agent Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![SUMO version](https://img.shields.io/badge/SUMO-1.20%2B-orange.svg)](https://eclipse.dev/sumo/)

This repository implements a highly modular, scalable Multi-Agent Reinforcement Learning (MARL) framework to solve the **Adaptive Traffic Light Control (ATLC)** optimization problem in complex urban road networks. The system features a progression from baseline non-MDP systems to classical game-theoretic decentralized coordinators, graph-based message-passing, and modern deep value-factorization mixing architectures.

---

## 📖 Abstract & Project Overview

Metropolitan traffic congestion poses a significant economic and environmental burden. Traditional systems rely on pre-programmed fixed-time phases that cannot adapt to stochastic, dynamic vehicular arrival rates. 

We formulate the adaptive traffic light control problem as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**, denoted by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:

*   **Agents ($\mathcal{N}$):** A grid of $K \times K$ intersections (e.g., a $2\times2$ grid containing 4 coordinated agents).
*   **State Space ($\mathcal{S}$):** The custom partially observable queue length (halted vehicle count) along all incoming lanes at each intersection.
*   **Action Space ($\mathcal{A}$):** Discrete phase selection indicating which direction receives the green light (e.g., North-South green vs. East-West green). Corner junctions use single-phase plans, while boundary/center junctions support multi-phase coordinated transitions.
*   **Transition Dynamics ($\mathcal{T}$):** Vehicular traffic flows generated using a stochastic **Poisson process** ($\lambda = 0.05$) to model realistic dynamic vehicle arrivals.
*   **Reward Function ($\mathcal{R}$):** The accumulated negative waiting time of all vehicles halted at the intersection's junctions, driving the agents to minimize travel delays:
    $$R_i(t) = - \sum_{v \in \mathcal{V}_i} W_v(t)$$
*   **Observations ($\Omega$):** Dimension-adapted local queue metrics standardized for continuous deep networks.

---

## 🛠️ Implemented Algorithms & Methodologies

The repository includes a complete roster of **7 coordinated and deep algorithms** to evaluate how modern deep MARL compares against classical coordination graphs and baselines:

```
ATLC Roster Progression
 ├── Category A: Baselines & Classical Game Theory
 │    ├── 1. Fixed-Time Controller (Non-MDP Baseline round-robin)
 │    ├── 2. Independent Tabular Q-Learning (Decentralized State-Binning)
 │    └── 3. Independent SARSA (On-Policy counterpart)
 ├── Category B: Classical Coordination Mechanics
 │    ├── 4. Distributed W-Learning (Resource competitor negotiations)
 │    ├── 5. Joint-Action Learners (JAL) (Action-history tracking)
 │    └── 6. Max-Plus Algorithm (Message-Passing Coordination Graphs)
 └── Category C: Modern Centralized Mixing
      └── 7. QMIX Centralized Mixing (Centralized Training, Decentralized Execution - CTDE)
```

### Key Engineering Features:
*   **Dynamic Lane Dimension Adapters:** Automates observation shape transformation (padding/slicing) to evaluate pre-trained models on arbitrary grid sizes ($2\times2$ to $5\times5$) without dimensional shape crashes.
*   **Coordination Graph KeyError Safeguards:** Configures Max-Plus message-passing to gracefully default missing communication links to neutral utility boundaries.
*   **Automated Scaling Pipeline:** Programmatically compiles networks, edits green durations, and runs multi-seed scaled environments.

---

## 💻 Technologies Used

*   **Traffic Simulator:** Eclipse SUMO (Simulation of Urban MObility) & `sumo-rl`
*   **Environment Standard:** PettingZoo (Parallel API Wrapper) & Gymnasium
*   **Core Logic:** Python, PyTorch (Deep Mixing Networks), NumPy, Pandas
*   **Visualizations:** Seaborn & Matplotlib
*   **Verification:** Pytest

---

## ⚙️ Setup and Installation

### 1. Install Eclipse SUMO
Ensure that SUMO is installed on your local operating system:
*   **Windows:** Download the MSI installer from [SUMO Downloads](https://eclipse.dev/sumo/) and install it.
*   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-gui
    ```

### 2. Configure Environment Variables
Set the `SUMO_HOME` environment variable to point to your SUMO installation folder:
*   **Windows (Powershell):**
    ```powershell
    [System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
    ```
*   **Linux/macOS:**
    ```bash
    export SUMO_HOME=/usr/share/sumo
    ```

### 3. Clone and Install Python Dependencies
```bash
git clone https://github.com/yourusername/ATLC-MARL.git
cd ATLC-MARL
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage & Testing

### 1. Running the Automated Testing Suite
We include a lightning-fast, comprehensive automated testing suite with **24 modular tests** verifying imports, environment stepping, XML generations, and coordination message passes in **under 10 seconds**:
```bash
python run_all_tests.py
```

### 2. Run Scaling Evaluation
Execute the scaling pipeline to train and evaluate all 7 algorithms across $2\times2$ to $5\times5$ grids and plot comparison metrics:
```bash
python run_followup_experiments.py
```
This writes numerical metrics to `outputs/scaled_results.csv` and a high-resolution comparative scaling plot to `outputs/scaled_performance_comparison.png`.

### 3. Visual Simulator & Real-time Console Monitor
Open the SUMO-GUI window and step slowly through a simulation episode to visually observe vehicle queues and coordinated green phase switches:
```bash
python watch_agents.py --size 2 --algo qmix --delay 0.1
```

---

## 📈 Evaluation Results & Interpretability

Numerical comparison metrics are automatically plotted and saved to disk. Independent decentralized models show moderate improvements over the Fixed-Time baseline on small networks, while centralized coordination graph message-passing (Max-Plus) and centralization mixing (QMIX) exhibit superior scaling properties by preventing queue congestion bottlenecks across high agent densities.

*Place your generated plot comparison from `outputs/scaled_performance_comparison.png` here to visually document algorithmic performance scaling!*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
