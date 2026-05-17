# System Instructions: ATLC MA-MDP Follow-Up Tasks (Expanded Algorithm Roster)

## Objective
Expand the base Adaptive Traffic Light Control (ATLC) MA-MDP experiment by executing three specific follow-up tasks: scaling the problem size, visualizing agent behavior, and drastically expanding the algorithm comparison to include foundational, classic multi-agent coordination frameworks and advanced MARL methods. Ensure all code is modular and builds upon the previously defined $2 \times 2$ infrastructure.

---

## Task 1: Problem Size Scaling & Evaluation
Evaluate how well the algorithms scale as the multi-agent system grows in complexity.
- **Implementation:** Write a Python script using the `subprocess` module to loop through SUMO's `netgenerate` CLI commands. Automatically generate expanding grid networks (e.g., $2 \times 2$, $3 \times 3$, $4 \times 4$, $5 \times 5$).
- **Execution:** Run a deterministic evaluation episode for each trained algorithm on every grid size. Maintain a consistent traffic generation density (Poisson rate) across all sizes to ensure a fair comparison.
- **Output:** Generate a comparative line plot using Matplotlib. 
  - **X-axis:** Number of Agents (or Grid Size).
  - **Y-axis:** Average Waiting Time.
  - Include error bars or shaded regions if running multiple random seeds.

---

## Task 2: Visualization of Intersections & Decisions
Create a visual rendering pipeline to qualitatively observe the policies and understand the state-action mappings.
- **Implementation:** Create a separate evaluation script that forces the environment to use `sumo-gui` (via PettingZoo's `render_mode='human'` or SUMO-RL's internal render flags) instead of the headless `sumo` binary.
- **Decision Logging:** Write a step-by-step monitor that prints the active decision matrix to the console in real-time. 
  - *Format Example:* `Step 100 | Agent 'TL_0' | State (Queues) - N:5, S:2, E:0, W:10 | Action Selected: Green East-West`
- **Output:** A single, easily executable script (`watch_agents.py`) that loads a pre-trained policy, opens the SUMO GUI, and steps through exactly one episode slowly enough for human observation.

---

## Task 3: Comprehensive Multi-Agent Algorithm Comparison
Ground the performance of deep MARL by testing it against classical decentralized and coordinated multi-agent systems techniques.

### Category A: Baselines & Classical Game Theory Methods
1. **Fixed-Time Controller (Non-MDP Baseline):** A standard, non-adaptive round-robin traffic light cycle (e.g., 30s NS, 3s Yellow, 30s EW).
2. **Independent Tabular Q-Learning (Decentralized Classical):** The foundational multi-agent RL approach where each agent acts independently.
   - *State Discretization Constraint:* Because queue lengths can grow infinitely, write a state wrapper to group queue counts into specific discrete bins (e.g., `0`, `1-3`, `4-7`, `8+`) to prevent the Q-table size from exploding.
3. **Independent SARSA:** The classical on-policy counterpart to Tabular Q-learning, operating on the exact same discretized state space.
4. **Distributed W-Learning:** A classic multi-agent method tailored for resource allocation. Agents maintain separate W-tables to determine how much they "care" about winning a slot (green light direction) based on competing directional queues.

### Category B: Classical Coordination Mechanics
5. **Joint-Action Learners (JAL) / Coordinated Q-Learning:** A classic multi-agent method where agents observe both their own state and the *actions* taken by adjacent neighboring intersections to learn a coordinated joint-policy matrix.
6. **Max-Plus Algorithm (Coordination Graphs):** An elegant, graph-based message-passing approach where adjacent intersections pass messages to coordinate green-light switches along common corridors, optimizing a global joint-action utility approximation without central control.

### Category C: Modern Value-Factorization Method
7. **QMIX:** A centralized training with decentralized execution (CTDE) deep MARL method. Add this to evaluate how a modern mixing network that ensures a monotonic relationship between individual agent utilities and joint utility compares against the classical coordination graph approaches.

- **Output:** Update the training and execution scripts to support this complete roster of 7 models, and ensure the evaluation plots map their performance across different network sizes.
