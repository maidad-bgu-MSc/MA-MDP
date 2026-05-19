# System Instructions: 1x4 Arterial Green Wave Simulator Setup & Testing

## Objective
Isolate the environment configuration into a dedicated `simulator/` module. This setup constructs a simplified $1 \times 4$ arterial corridor designed specifically for the Green Wave scenario. The instructions also mandate rigorous dynamic testing using dummy agents to verify the PettingZoo API integrity and global reward synchronization.

---

## 1. Directory Structure Restructuring
Create a dedicated `simulator/` folder to encapsulate all environment logic. 

```text
simulator/
 ├── __init__.py                # Makes the folder an importable Python module
 ├── generate_1x4_wave.py       # Script to build the 1x4 SUMO network and routes
 └── env_setup.py               # The PettingZoo wrapper with Global Reward
tests/
 └── test_1x4_dynamics.py       # Pytest suite for environment and dummy agent testing
```

---

## 2. Network Generation (`generate_1x4_wave.py`)
This script must autonomously generate a single, horizontal arterial road with four cross-streets.
- **Topology Setup:** Use SUMO's `netgenerate` with a grid of `--grid.x-number 4 --grid.y-number 1`.
- **Traffic Routing (The Wave):**
  - **Arterial (East-West):** Define a continuous route passing straight through all four intersections. Inject a high-density platoon (e.g., 30 vehicles over a 10-second window) every 150 seconds.
  - **Cross-Streets (North-South):** Generate light, stochastic Poisson traffic ($\lambda = 0.02$) on the four intersecting vertical roads.

---

## 3. Environment API & State Management (`env_setup.py`)
This file must provide a clean function `make_wave_env()` that returns the fully configured PettingZoo environment.

### A. The Global Reward Function
To force coordination, the reward must be shared.
- Write a `global_reward_fn(ts)` that calculates the sum of all negative waiting times across the **entire** $1 \times 4$ network.
- Ensure the wrapper broadcasts this single scalar value to all four agents on every step.

### B. "Look-Ahead" Observation Space (Tabular-Safe)
Keep the state space perfectly sized for Tabular Q-Learning.
- Modify the `QueueObservationFunction` so that each agent observes exactly two things:
  1. The queue length on its local East-West arterial incoming lane.
  2. The queue length on the immediate **upstream** East-West arterial lane (its neighbor to the West).
- Apply a **5-Bin Discretization** to both values:
  - `0`: 0 cars
  - `1`: 1-5 cars
  - `2`: 6-15 cars
  - `3`: 16-29 cars
  - `4`: 30+ cars
- *Result:* The tabular state space is exactly $5 \times 5 = 25$ possible states per agent.

### C. Safety Rails
- Ensure `min_green=10` is passed to the core `sumo_rl` initialization.
- Implement the **Gridlock Reset** wrapper: if any single queue hits Bin 4 (30+ cars), return `done=True` (or terminations) for all agents and apply a massive terminal penalty (e.g., -10,000).

---

## 4. Environment Dynamics & PettingZoo Testing (`tests/test_1x4_dynamics.py`)
Before passing the environment to the RL algorithms, write a `pytest` suite to verify the state machine and PettingZoo AEC/Parallel compliance using dummy agents.

### Test A: Dummy "Pet" Agent Loop (Crash Test)
- **Action:** Initialize `make_wave_env()`. Loop through the environment using PettingZoo's AEC standard loop (`for agent in env.agent_iter():`), passing random actions using `env.action_space(agent).sample()`.
- **Validation:** Assert that the environment can run a full 3600-step dummy episode without throwing `KeyError`, `IndexError`, or desynchronization exceptions.

### Test B: Global Reward Synchronization
- **Action:** Step the environment forward by 100 ticks.
- **Validation:** Intercept the `rewards` dictionary and assert that `rewards["agent_0"] == rewards["agent_n"]` for all agents, proving that the global reward broadcast is functioning correctly.

### Test C: Early Termination Trigger
- **Action:** Write a mock test that forces the environment's max queue length to exceed 30 cars.
- **Validation:** Assert that the environment immediately issues the terminal penalty (-10,000) and that the `terminations` dictionary flags `True` for all agents simultaneously, cleanly resetting the state.
