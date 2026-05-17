# Adaptive Traffic Light Control (ATLC) MARL Codebase Summary

This document provides a highly detailed, comprehensive analysis of the **Adaptive Traffic Light Control (ATLC)** repository. It outlines the project's folder structure, provides an overview of each file, and breaks down the exact meaning, inputs, outputs, and mathematical context of every class and function in the codebase.

---

## 📂 Codebase Directory Structure

```
c:\Users\advam\Documents\MA-MDP
 ├── .gitignore                     # Git exclusion patterns
 ├── README.md                      # Core project abstract, Dec-POMDP formulation, setup & usage
 ├── requirements.txt               # Python package dependencies
 ├── ATLC_FollowUp_Tasks.md         # Checklist for experiment execution & scaling updates
 ├── ATLC_GitHub_Instructions.md    # Reference guide for git and remote pushes
 ├── ATLC_Testing_Instructions.md   # Setup and CLI commands for Pytest testing
 ├── ATLC_Testing_Instructions_new.md # Updated testing guides with verbose flag documentation
 ├── env_setup.py                   # SUMO-RL environment creation & Queue observation wrapper
 ├── generate_network.py            # Baseline grid.net.xml / grid.rou.xml generator
 ├── scale_network.py               # Scalable grid generator for arbitrary sizes (2x2 to 5x5)
 ├── marl_algorithms.py             # Roster of 7 learning controllers and Max-Plus/QMIX networks
 ├── train.py                       # Deep MARL training pipeline (IQL-DQN and IPPO-PPO)
 ├── evaluate.py                    # Deterministic evaluation script for deep MARL policies
 ├── watch_agents.py                # Visual monitor and step decision-log inside SUMO-GUI
 ├── run_followup_experiments.py    # Master scaling trainer & evaluator with performance plotting
 ├── plot_results.py                # Log parser & metric aggregator (waiting time, queue, speed)
 ├── run_all_tests.py               # Auto-testing script runner (pytest)
 ├── models/                        # Pre-trained tabular (.npy) and deep PyTorch (.pth) policies
 │    ├── ippo_policy.pth
 │    ├── iql_policy.pth
 │    ├── iql_tabular_A0_2x2.npy
 │    └── ...
 └── tests/                         # Pytest modular verification suite
      ├── __init__.py
      ├── test_action_handshake.py
      ├── test_algorithms.py
      ├── test_coordination_graph.py
      ├── test_environment.py
      ├── test_imports.py
      ├── test_net_generation.py
      ├── test_pipeline_flow.py
      ├── test_scripts.py
      ├── test_state_discretization.py
      └── test_topology_sync.py
```

---

## 🛠️ Module & File Descriptions

### 1. Project Specifications & Documentation
*   [README.md](file:///c:/Users/advam/Documents/MA-MDP/README.md)
    *   **Purpose:** The central documentation portal. It establishes the mathematical formulation of the adaptive traffic light control problem as a **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process), presents the 7-algorithm roster, details installation instructions, and highlights scaling execution.
*   [ATLC_FollowUp_Tasks.md](file:///c:/Users/advam/Documents/MA-MDP/ATLC_FollowUp_Tasks.md)
    *   **Purpose:** Tracks development tasks, centering on automated scaling tests, custom metric collections, and visualization plot setups.
*   [ATLC_GitHub_Instructions.md](file:///c:/Users/advam/Documents/MA-MDP/ATLC_GitHub_Instructions.md)
    *   **Purpose:** Practical tutorial on setting up a remote repository, resolving merge collisions, and synchronizing local commits.
*   [ATLC_Testing_Instructions.md](file:///c:/Users/advam/Documents/MA-MDP/ATLC_Testing_Instructions.md) & [ATLC_Testing_Instructions_new.md](file:///c:/Users/advam/Documents/MA-MDP/ATLC_Testing_Instructions_new.md)
    *   **Purpose:** Detailed guide on setting up and running Pytest, explaining command line overrides (e.g., `-v`, `--durations`, `-W`).

---

### 2. Core Environment Setup
*   [env_setup.py](file:///c:/Users/advam/Documents/MA-MDP/env_setup.py)
    *   **Purpose:** Responsible for setting up system paths for SUMO, defining the custom state-space representation based strictly on vehicle queue length, overrides the default PettingZoo reward structure to match a strict wait-delay formulation, and provisions turn-based PettingZoo environment AEC wrapping.
    *   **Key Functions & Classes:**
        *   [`setup_sumo_env()`](file:///c:/Users/advam/Documents/MA-MDP/env_setup.py#L7-L27): Detects local `sumo` installations or defaults to environment variable paths, appending correct binary execution paths to the Windows system environment PATH variable.
        *   [`QueueObservationFunction`](file:///c:/Users/advam/Documents/MA-MDP/env_setup.py#L34-L51): Extends SUMO-RL's standard observation builder.
            *   `__call__(self) -> np.ndarray`: Queries incoming junction links and returns active halted vehicle counts.
            *   `observation_space(self) -> Box`: Configures continuous state-space bounds (lower bound `0.0`, upper bound `+inf`) matching lane dimensional capacities.
        *   [`custom_reward_fn(ts)`](file:///c:/Users/advam/Documents/MA-MDP/env_setup.py#L53-L56): Computes a strict MDP reward representation of negative accumulated vehicle delays:
            $$R_i(t) = - \sum_{v \in \mathcal{V}_i} W_v(t)$$
        *   [`make_env(...)`](file:///c:/Users/advam/Documents/MA-MDP/env_setup.py#L58-L73): Orchestrates the instantiation of parallel environments (`sumo_rl.parallel_env`) before casting them into turn-based PettingZoo AEC interfaces (`parallel_to_aec`) for full deep training compatibility.

---

### 3. XML Network Generation & Scaling Utilities
*   [generate_network.py](file:///c:/Users/advam/Documents/MA-MDP/generate_network.py)
    *   **Purpose:** Autonomously builds a baseline 2x2 grid intersection net-file and route-file containing Poisson-process vehicular arrivals.
    *   **Key Functions:**
        *   [`setup_sumo_env()`](file:///c:/Users/advam/Documents/MA-MDP/generate_network.py#L6-L28): Re-registers system paths.
        *   [`run_netgenerate()`](file:///c:/Users/advam/Documents/MA-MDP/generate_network.py#L29-L45): Executes SUMO's `netgenerate` tool as a subprocess to build `grid.net.xml`.
        *   [`modify_traffic_lights()`](file:///c:/Users/advam/Documents/MA-MDP/generate_network.py#L47-L65): Programmatically parses and standardizes light phases in `grid.net.xml` to two main phases with yellow transition times (42s Green, 3s Yellow).
        *   [`generate_routes()`](file:///c:/Users/advam/Documents/MA-MDP/generate_network.py#L67-L97): Generates route file `grid.rou.xml` outlining 8 intersecting corridor channels alongside stochastically scaled vehicle flows driven by exponential Poisson densities ($\lambda = 0.05$).

*   [scale_network.py](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py)
    *   **Purpose:** Programmatically builds high-dimensional grid networks (up to 5x5 grids with 25 independent coordinated junctions).
    *   **Key Functions:**
        *   [`setup_sumo_env()`](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py#L7-L29): System binary path configuration.
        *   [`run_netgenerate(size)`](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py#L30-L48): Scales networks to size $K \times K$, building `grid_KxK.net.xml`.
        *   [`modify_traffic_lights(net_file)`](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py#L50-L70): Standardizes 2-phase green cycles across scaled networks.
        *   [`generate_routes(size, rou_file)`](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py#L71-L124): Programmatically generates complete horizontal (Eastbound, Westbound) and vertical (Northbound, Southbound) transit corridors throughout the entire grid network, writing out corresponding exp(0.05) Poisson flow nodes.
        *   [`build_network_size(size)`](file:///c:/Users/advam/Documents/MA-MDP/scale_network.py#L125-L133): Combines grid generation, traffic phase edits, and route creation.

---

### 4. Reinforcement Learning Roster (Algorithms)
*   [marl_algorithms.py](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py)
    *   **Purpose:** The mathematical engine of the project, defining the baseline, classical, game-theoretic, and deep centralized mixing algorithms.
    *   **Key Functions & Classes:**
        *   [`discretize_queue(q_length)`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L10-L19): Maps continuous vehicle queues into 4 discrete bins (0: Empty, 1: Short queue, 2: Moderate queue, 3: Heavily congested queue).
        *   [`get_discrete_state(obs)`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L21-L26): Computes a 1D discrete state index (0 to 15) from a 2D lane observation input `obs` based on a $4 \times 4$ bin grid:
            $$\text{State} = q_1 \times 4 + q_2$$
        *   [`FixedTimeController`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L31-L48): Baseline round-robin controller.
            *   `compute_action(...)`: Alternates green phases every `cycle_steps` intervals regardless of traffic observations.
        *   [`TabularQLearningAgent`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L52-L77): Tabular Independent Q-Learning.
            *   `compute_action(...)`: Performs $\epsilon$-greedy action selection across discrete tables.
            *   `update(...)`: Adjusts actions based on the temporal difference equation:
                $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
        *   [`SARSAAgent`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L81-L105): Tabular On-Policy SARSA control.
            *   `update(...)`: Performs on-policy parameter updates utilizing the *actual* chosen next action $a'$:
                $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$
        *   [`WLearningAgent`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L109-L164): Competitor-negotiation priority allocator. Maintains sub-Q-tables for independent incoming lanes and W-tables representing how much each lane sub-agent "cares" about winning the junction green phase action.
            *   `compute_action(...)`: Every lane sub-agent nominates an action. The final selected action corresponds to the sub-agent exhibiting the largest regret (W-value).
            *   `update(...)`: Updates the Q-table of the winning lane sub-agent, and updates W-table parameters (regret estimation) for non-winning lanes using local learning rate $\beta$:
                $$W_i(s_i) \leftarrow W_i(s_i) + \beta \left( \left[ \max_{a} Q_i(s_i, a) - Q_i(s_i, \text{action}) \right] - W_i(s_i) \right)$$
        *   [`JALAgent`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L168-L210): Joint-Action Learner. Observes physical adjacent neighbors' actions to coordinate selections, building a multi-dimensional joint action Q-table: $Q(s_{\text{local}}, a_{\text{self}}, \mathbf{a}_{\text{neighbors}})$.
            *   `get_neighbor_action_state(...)`: Formulates a bitwise representation index for coordinated neighboring action profiles.
            *   `update(...)`: Updates joint action tables using historical neighbor choices.
        *   [`MaxPlusAgent`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L214-L258): Coordination graph node. Incorporates local utility functions $Q_i(s_i, a_i)$ and pairwise coordination matrices $Q_{ij}(s_i, s_j, a_i, a_j)$.
            *   `update(...)`: Performs localized Q-updates for individual payoff terms alongside stochastic pairwise edge coordination updates.
        *   [`run_max_plus_coordination(agents_dict, obs_dict, iterations=4)`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L259-L306): Coordinates the overall grid network via decentralized message-passing. Agents calculate outbound utility bids $m_{i \to j}(a_j)$ programmatically:
            $$m_{i \to j}(a_j) = \max_{a_i} \left[ Q_i(s_i, a_i) + Q_{ij}(s_i, s_j, a_i, a_j) + \sum_{k \in \mathcal{N}_i \setminus \{j\}} m_{k \to i}(a_i) \right]$$
            Normalizes message values at each iteration to maintain numerical stability, summing final messages with local payoff inputs to output the coordinated action plan.
        *   [`QMIXMixingNetwork`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L311-L359): Centralized PyTorch monotonic mixing network. Houses hypernetworks that dynamically map the centralized state (concatenation of all agent observations) to positive mixing weights $W_1$, $W_2$ and bias structures $b_1$, $b_2$.
            *   `forward(...)`: Mixing logic. Monotonicity is structurally guaranteed by forcing weights to absolute positive values:
                $$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0, \quad \forall i$$
        *   [`QMIXAgentNetwork`](file:///c:/Users/advam/Documents/MA-MDP/marl_algorithms.py#L360-L374): Individual deep utility network, structured as a standard 3-layer Multi-Layer Perceptron (MLP), mapping local continuous queue observation dimensions to individual action Q-values.

---

### 5. Training, Evaluation, and Plotting Workflows
*   [train.py](file:///c:/Users/advam/Documents/MA-MDP/train.py)
    *   **Purpose:** Contains the training procedures for continuous deep MARL models (Independent Q-Learning and Independent PPO) using Tianshou v2 and PettingZoo.
    *   **Key Functions & Classes:**
        *   [`QNet`](file:///c:/Users/advam/Documents/MA-MDP/train.py#L23-L41): Multi-Layer Perceptron Q-network.
        *   [`ActorNet`](file:///c:/Users/advam/Documents/MA-MDP/train.py#L43-L61) & [`CriticNet`](file:///c:/Users/advam/Documents/MA-MDP/train.py#L62-L80): Actor and Critic MLP layers for IPPO.
        *   [`train_iql(num_seconds, epochs)`](file:///c:/Users/advam/Documents/MA-MDP/train.py#L81-L155): Standardizes multi-agent configurations, creating independent Tianshou `DiscreteQLearningPolicy` DQN agents. Managers execute off-policy VectorReplayBuffer updates and log best policy weights to `models/iql_policy.pth`.
        *   [`train_ippo(num_seconds, epochs)`](file:///c:/Users/advam/Documents/MA-MDP/train.py#L156-L235): Analogous setup for on-policy PPO training. Employs `ProbabilisticActorPolicy` wrappers alongside specialized on-policy trainers, saving policy weights to `models/ippo_policy.pth`.

*   [evaluate.py](file:///c:/Users/advam/Documents/MA-MDP/evaluate.py)
    *   **Purpose:** Executes deterministic test runs using pre-trained deep policy models, collecting numerical metrics.
    *   **Key Functions:**
        *   [`evaluate_iql(num_seconds)`](file:///c:/Users/advam/Documents/MA-MDP/evaluate.py#L78-L119): Evaluates deep IQL models, outputting progress statistics and metric CSV files under the prefix `outputs/iql_eval`.
        *   [`evaluate_ippo(num_seconds)`](file:///c:/Users/advam/Documents/MA-MDP/evaluate.py#L120-L163): Evaluates deep IPPO models deterministically under the prefix `outputs/ippo_eval`.

*   [watch_agents.py](file:///c:/Users/advam/Documents/MA-MDP/watch_agents.py)
    *   **Purpose:** Real-time console monitor and visual debugger. Initiates the visual SUMO-GUI window and prints local traffic dynamics alongside real-time agent actions.
    *   **Key Functions:**
        *   [`adapt_obs_dict(obs_dict)`](file:///c:/Users/advam/Documents/MA-MDP/watch_agents.py#L21-L32): Slices or zero-pads continuous input arrays to enforce observation shapes of exactly size 2, preventing dimensionality mismatches.
        *   [`load_policy(algo, agent_ids, size)`](file:///c:/Users/advam/Documents/MA-MDP/watch_agents.py#L44-L149): Loads pre-trained model parameters (NumPy matrices or PyTorch weights) from `models/` based on selected algorithm and grid scale.
        *   [`run_gui_simulation()`](file:///c:/Users/advam/Documents/MA-MDP/watch_agents.py#L150-L250): Initializes physical grid networks under SUMO-GUI. Steps through simulations at custom sleep intervals, adapts observations, executes policy forward passes, verifies action bounds, and prints formatted real-time logs to the console:
            ```
            Step 0055 | Agent 'A0' | State (Queues) - N: 2, S: 0, E: 1, W: 0 | Action Selected: Green North-South
            ```

*   [run_followup_experiments.py](file:///c:/Users/advam/Documents/MA-MDP/run_followup_experiments.py)
    *   **Purpose:** The central testing suite for scaled evaluations. It automates training and evaluation cycles for all 7 algorithms across grid size progressions (2x2 up to 5x5 grids), plotting comparative performance trends.
    *   **Key Functions:**
        *   [`train_tabular_agents(env, algo, size, episodes)`](file:///c:/Users/advam/Documents/MA-MDP/run_followup_experiments.py#L36-L135): Manages multi-agent tabular updates (IQL, SARSA, JAL, W-Learning, and Max-Plus coordination graphs) inside the simulator, saving parameters as `.npy` matrices.
        *   [`train_qmix_agents(env, size, episodes)`](file:///c:/Users/advam/Documents/MA-MDP/run_followup_experiments.py#L136-L223): Custom training pipeline for QMIX. Combines agent network updates and centralized mixer network updates, minimizing global MSE loss:
            $$\mathcal{L}(\theta) = \left[ Q_{\text{tot}}(s, \mathbf{a}) - \left( R_{\text{tot}} + \gamma \max_{\mathbf{a}'} Q_{\text{tot}}(s', \mathbf{a}') \right) \right]^2$$
        *   [`evaluate_on_grid(algo, size)`](file:///c:/Users/advam/Documents/MA-MDP/run_followup_experiments.py#L224-L348): Evaluates pre-trained models deterministically, returning the mean vehicle waiting time.
        *   [`generate_scaling_plot(df_results)`](file:///c:/Users/advam/Documents/MA-MDP/run_followup_experiments.py#L349-L420): Formulates stylish Matplotlib scaling diagrams using curated, modern palettes, plotting curves for all 7 algorithms and saving outputs to `outputs/scaled_performance_comparison.png` and `outputs/scaled_results.csv`.

*   [plot_results.py](file:///c:/Users/advam/Documents/MA-MDP/plot_results.py)
    *   **Purpose:** Formulates evaluation comparisons.
    *   **Key Functions:**
        *   [`find_latest_csv(prefix)`](file:///c:/Users/advam/Documents/MA-MDP/plot_results.py#L7-L16): Searches the outputs directory for modified, non-empty evaluation CSV logs.
        *   [`generate_comparison_plots()`](file:///c:/Users/advam/Documents/MA-MDP/plot_results.py#L18-L121): Parses IQL and IPPO simulation CSV logs, printing performance summaries (mean waiting times, stopped queue sizes, vehicle speeds), creating the report `outputs/summary.md`, and generating triple-line time-series comparisons in `outputs/performance_comparison.png`.

*   [run_all_tests.py](file:///c:/Users/advam/Documents/MA-MDP/run_all_tests.py)
    *   **Purpose:** Automatically executes testing suites.
    *   **Key Functions:**
        *   [`install_pytest_if_missing()`](file:///c:/Users/advam/Documents/MA-MDP/run_all_tests.py#L5-L18): Installs `pytest` via `pip` if absent.
        *   [`run_tests()`](file:///c:/Users/advam/Documents/MA-MDP/run_all_tests.py#L20-L29): Launches `pytest` on the `tests/` directory with verbose flags, suppressing warning logs.

---

### 6. Modular Test Suite (`tests/`)
*   [test_imports.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_imports.py)
    *   **Purpose:** Confirms essential Python imports (numpy, pandas, torch, sumo_rl, tianshou).
*   [test_state_discretization.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_state_discretization.py)
    *   **Purpose:** Asserts correct bin boundaries and discrete state mappings for the queue observations.
*   [test_net_generation.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_net_generation.py)
    *   **Purpose:** Verifies that netgenerate builds correct XML structures with target road networks.
*   [test_environment.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_environment.py)
    *   **Purpose:** Verifies environment instantiation, resets, and custom reward function behaviors.
*   [test_pipeline_flow.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_pipeline_flow.py)
    *   **Purpose:** Asserts standard PettingZoo stepping, termination triggers, and observation space formats.
*   [test_algorithms.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_algorithms.py)
    *   **Purpose:** Tests each of the 7 algorithms in isolation. Confirms value updates in Q-tables, SARSA-updates, W-Learning regret changes, Joint Q-table adjustments, Max-Plus local and pairwise matrices, Max-Plus coordination passes, and deep PyTorch model shapes.
*   [test_action_handshake.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_action_handshake.py)
    *   **Purpose:** Asserts valid action indices mapping and phase bounds.
*   [test_coordination_graph.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_coordination_graph.py)
    *   **Purpose:** Validates neighbor maps, communication links, and Max-Plus edge utility updates.
*   [test_topology_sync.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_topology_sync.py)
    *   **Purpose:** Ensures grid networks scale cleanly without losing coordinate alignment.
*   [test_scripts.py](file:///c:/Users/advam/Documents/MA-MDP/tests/test_scripts.py)
    *   **Purpose:** Runs script drivers under automated verification pipelines.

---

## 📈 Summary of Algorithms

The 7 traffic light coordination algorithms implemented in this project are compared below:

| # | Algorithm Name | Category | State Space | Action Space | Key Mathematical / Conceptual Context |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **Fixed-Time Controller** | Baseline (Non-MDP) | N/A | Discrete phase index | Standard round-robin phase cycling; alternates green phases on a fixed time interval. |
| **2** | **Independent Tabular Q-Learning** | Decentralized Classical | Discretized ($4\times4$) | Discrete phase index | Decentralized control where each agent operates stochastically, maximizing individual temporal difference Q-targets. |
| **3** | **Independent SARSA** | Decentralized Classical | Discretized ($4\times4$) | Discrete phase index | On-policy temporal difference learning variant, adapting actions to actual target updates. |
| **4** | **Distributed W-Learning** | Coordinated Classical | Discretized per lane | Discrete phase index | Coordinates resource allocation by learning lane queue bidder regret metrics to resolve action-space conflicts. |
| **5** | **Joint-Action Learners (JAL)** | Coordinated Classical | Discretized + Joint Neighbor Actions | Discrete phase index | Models explicit joint action coordination patterns across neighboring agents, maintaining historical action maps. |
| **6** | **Max-Plus Algorithm** | Coordinated Classical | Discretized local + pairwise | Discrete phase index | Computes coordinated action plans via message-passing over local coordinate graphs using edge-utility matrices. |
| **7** | **QMIX Centralized Mixing** | Deep Centralized Mixing | Continuous local + central state | Continuous (Q-values) | Monotonic deep mixing network (CTDE paradigm), mapping individual agent Q-utilities to a centralized $Q_{\text{tot}}$ hypernetwork. |
