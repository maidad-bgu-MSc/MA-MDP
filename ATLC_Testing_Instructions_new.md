# System Instructions: ATLC MA-MDP Automated Testing Suite (Comprehensive Workflow Expansion)

## Objective
Enhance the testing framework by implementing aggressive integration, pipeline workflow, and data boundary tests. These tests must focus on isolating complex interaction failures, state corruption, boundary condition crashes, and configuration mismatches. The suite must remain isolated within the `tests/` directory and execute rapidly using lightweight mocks and short iteration counts.

## 1. Directory Structure
- All test scripts must reside inside the `tests/` folder.
- Ensure proper path linking so test scripts can dynamically import from the root module without dependency issues.

## 2. Advanced Workflow & Integration Tests (Targeting Breaking Points)

### Test F: Multi-Agent Class Discretization Wrapper Leakage (`test_state_discretization.py`)
- **Target Failure:** Out-of-bounds queue sizes causing unhandled `KeyError` exceptions or unexpected `NoneType` states in the tabular models (Q-Learning, SARSA, W-Learning).
- **Action:** Mock the environment to yield extreme continuous state scenarios (e.g., negative queues, a lane with `100+` halted vehicles, empty arrays, or floats instead of integers).
- **Validation:** Assert that the discretization function cleanly intercepts these anomalies, safely maps them into the designated bins (e.g., routing 100 to the highest bin `8+`), and never allows unmapped inputs to propagate into table indices.

### Test G: Network Topology Change & Config Synchronization (`test_topology_sync.py`)
- **Target Failure:** Grid scaling script (`netgenerate`) modifying the map topology without correctly altering the dictionary mappings of the algorithms (e.g., looking up a deleted intersection ID).
- **Action:** Execute a mocked pipeline sequence that transitions dynamically from a $2 	imes 2$ setup to a $3 	imes 3$ grid. 
- **Validation:** 
  1. Verify that the agent initialization script dynamically reconstructs its internal policy matrix keys strictly based on the newly parsed `.net.xml` file.
  2. Assert that no stale agent keys from the previous configuration persist in memory.

### Test H: Action Ingestion & State-Machine Handshaking (`test_action_handshake.py`)
- **Target Failure:** Invalid or out-of-bounds action integers passed down from advanced policies (like QMIX) causing the simulator to crash, or invalid yellow-phase transitions causing desynchronization.
- **Action:** Step the environment through invalid action IDs, rapid alternating actions, or simultaneous conflict actions.
- **Validation:** Assert that the environment wrapper safely rejects invalid action structures with descriptive errors and that valid action switches correctly trigger the mandatory intermediate yellow transition phases before registering the new green light.

### Test I: Complete Evaluation-to-Plot Pipeline Data Integrity (`test_pipeline_flow.py`)
- **Target Failure:** The evaluation loop completing successfully but writing malformed data shapes to the logging lists, causing Matplotlib plotting routines to crash at runtime due to dimension mismatches.
- **Action:** Run a fast mock training-and-evaluation routine across 2 different algorithms for exactly 2 environment steps.
- **Validation:** 
  1. Assert that the underlying logging dictionary captures metrics uniformly across all algorithms.
  2. Verify that the data array shapes match the expected dimensions (`(num_algorithms, num_grid_sizes, metrics)`).
  3. Call the plotting function wrapper using a headless backend (`matplotlib.use('Agg')`) to confirm that the generation of line plots and saving to disk executes with zero visualization errors.

### Test J: Coordination Graph Neighbor Discovery (`test_coordination_graph.py`)
- **Target Failure:** Message-passing algorithms (Max-Plus, Joint-Action Learners) failing or hanging due to looking up non-existent neighbor nodes at the boundaries of the grid network.
- **Action:** Initialize a Max-Plus/JAL instance on an edge intersection agent (e.g., a corner junction with only 2 neighbors instead of 4).
- **Validation:** Assert that the neighborhood discovery mechanism handles boundary nodes elegantly without raising index errors, and verify that missing edge connections default safely to a neutral communication value during message passing.

## 3. Test Execution Rigor
- Ensure the runner executes with: `pytest tests/ -v --durations=0` to immediately flag any integration test that exceeds 2 seconds.
- Maintain complete code hygiene by using localized file fixtures that dynamically wipe temporary testing folders upon test completion.
