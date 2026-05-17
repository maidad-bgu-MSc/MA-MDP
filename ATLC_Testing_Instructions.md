# System Instructions: ATLC MA-MDP Automated Testing Suite

## Objective
Create a fast, comprehensive automated testing suite to verify the integrity, validity, and execution capability of the entire ATLC codebase. The tests must run quickly without performing full training cycles, strictly ensuring that all modules connect and execute correctly.

## 1. Directory Structure
- Create a dedicated folder named `tests/` in the root directory.
- All testing scripts must be placed inside this folder.
- Ensure the root directory has an `__init__.py` (if necessary) so the tests can import the main codebase modules.

## 2. Test Requirements (Fast & Complete)
Write `pytest`-compatible test scripts to cover the following areas. Ensure every test runs in under a few seconds by minimizing episode counts and step limits.

### Test A: File Integrity & Dependency Check (`test_imports.py`)
- **Action:** Dynamically load and parse every `.py` file in the main directory.
- **Validation:** Assert that all files have valid Python syntax and that all required libraries (e.g., `sumo-rl`, `ray`, `tianshou`, `matplotlib`) can be imported without throwing `ModuleNotFoundError`.

### Test B: SUMO Network Generation (`test_net_generation.py`)
- **Action:** Call the network generation function/script for the base $2 \times 2$ grid.
- **Validation:** Assert that the expected SUMO output files (`.net.xml`, `.rou.xml`, `.sumocfg`) are physically created on the disk and are not empty.

### Test C: Environment Initialization & Stepping (`test_environment.py`)
- **Action:** Initialize the `sumo-rl` PettingZoo environment with the generated $2 \times 2$ configuration.
- **Validation:** 1. Assert the environment loads without crashing.
  2. Verify that the number of agents equals 4.
  3. Verify that the observation space and action space dictionaries are correctly formatted.
  4. Perform 5 random steps using `env.step({agent: env.action_space(agent).sample()})` and assert that the returned state is valid and non-null.

### Test D: Algorithm Forward-Pass / Dummy Execution (`test_algorithms.py`)
- **Action:** For **every** algorithm implemented (Fixed-Time, Q-Learning, SARSA, W-Learning, JAL, Max-Plus, and QMIX):
  - Initialize the agent/policy.
  - Run exactly **1 episode** with a hard limit of **10 steps**.
- **Validation:** Assert that the policy can successfully map a dummy state to a valid action, and that the execution loop completes without raising any runtime exceptions.

### Test E: Follow-Up Scripts Initialization (`test_scripts.py`)
- **Action:** For the scaling script and the `watch_agents.py` visualization script:
  - Mock the CLI arguments or environment variables.
  - Call the initialization functions (do not run the GUI or the full scaling loop).
- **Validation:** Assert that the scripts can parse arguments and initialize their respective classes/functions without error.

## 3. Execution Standard
- Create a top-level bash script `run_tests.sh` or a simple Python script `run_all_tests.py` that automatically triggers `pytest tests/ -v`.
- The tests must clean up after themselves (e.g., delete any temporary `.xml` network files generated during `test_net_generation.py` using Python's `tempfile` or `shutil` module).
