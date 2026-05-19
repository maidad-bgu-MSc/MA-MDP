# System Instructions: Codebase Pruning & Green Wave Simulator Isolation

## Objective
Carefully prune the `MA-MDP` repository to remove all legacy algorithms, obsolete grid-scaling scripts, and extraneous test files. The final state must be a streamlined, laser-focused repository containing *only* the $1 	imes 4$ Green Wave simulator, the visualization tools, and the essential training pipeline structure.

---

## 1. Safety Protocol (Mandatory First Step)
Before deleting a single file, you must create a backup of the entire repository.
- **Action:** Copy the entire `MA-MDP` folder and rename the copy to `MA-MDP_Archive_Pre_Prune`. Store this outside of your active working directory.

---

## 2. Files to DELETE (The Pruning List)
Carefully delete the following files and folders from your active `MA-MDP` directory:

### Legacy Generation Scripts
- `generate_network.py` (The old $2 	imes 2$ grid generator)
- `scale_network.py` (The dynamic $K 	imes K$ scaling script)

### Legacy Environment Setup
- `env_setup.py` (The old root-level Dec-POMDP wrapper. We are moving this logic to the `simulator/` folder).

### Obsolete Testing Files
- Inside the `tests/` folder, delete:
  - `test_coordination_graph.py`
  - `test_topology_sync.py`
  - `test_state_discretization.py`
  - `test_net_generation.py`

### Obsolete Markdown Instructions
- `ATLC_FollowUp_Tasks.md`
- `ATLC_Advanced_Scenarios_Instructions.md`
- `ATLC_Scenario_BlindArterial.md`
- `ATLC_Scenario_GodsEye.md`

---

## 3. Files to MODIFY (The Streamlining List)

### A. `marl_algorithms.py`
- **Delete** the following algorithm classes:
  - `FixedTimeController`
  - `SARSAAgent`
  - `WLearningAgent`
  - `JALAgent`
  - `MaxPlusAgent`
  - `run_max_plus_coordination`
- **Keep ONLY:**
  - The 5-Bin `discretize_queue` function.
  - `TabularQLearningAgent` (The baseline).
  - `QMIXMixingNetwork` & `QMIXAgentNetwork` (The advanced coordinator).

### B. `watch_agents.py` (The Visualizer)
- Update the import paths to load the environment from `simulator.env_setup` instead of the root directory.
- Update the `load_policy` function to only look for `IQL` or `QMIX` models.
- Ensure the visualization loop can handle both trained (loading weights) and untrained (random action sampling) states for debugging.

---

## 4. The Final Target Directory Structure
After the pruning is complete, your repository must look exactly like this:

```text
MA-MDP/
 ├── .gitignore
 ├── README.md
 ├── requirements.txt
 ├── ATLC_Simulator_1x4_GreenWave.md  # Your reference for the simulator
 ├── marl_algorithms.py               # (Pruned: Only IQL and QMIX)
 ├── train.py                         # (Pipeline for IQL and QMIX)
 ├── evaluate.py
 ├── plot_results.py
 ├── watch_agents.py                  # (Updated for 1x4 visualization)
 │
 ├── simulator/                       # THE CORE GREEN WAVE MODULE
 │    ├── __init__.py
 │    ├── generate_1x4_wave.py
 │    └── env_setup.py                # (Contains make_wave_env and global reward)
 │
 ├── models/                          # (Empty or containing only valid 1x4 weights)
 └── tests/
      ├── __init__.py
      ├── test_imports.py
      ├── test_1x4_dynamics.py        # (The new PettingZoo API checks)
      ├── test_algorithms.py          # (Updated to test only IQL and QMIX)
      └── test_action_handshake.py
```

## 5. Verification
Once the pruning is complete, run the following command from the root directory to ensure the surgery was successful and nothing critical was broken:
```bash
pytest tests/ -v
```
