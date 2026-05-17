# System Instructions: Advanced ATLC Coordination Scenarios (Implementation Guide)

## Objective
This document outlines the implementation requirements for four advanced traffic scenarios designed to stress-test our Multi-Agent Reinforcement Learning (MARL) algorithms. The goal is to prove the "Coordination Gap"—demonstrating that coordinated networks (e.g., Max-Plus, QMIX) significantly outperform independent agents (e.g., IQL, SARSA) under asymmetric, non-stationary, and high-stress conditions.

---

## 🏗️ Core Engineering Standards (Mandatory)

Before implementing the scenarios, all developers must adhere to the following standards:

1. **Clean & Structured Code:** - Keep scenario generation scripts modular. Do not hardcode parameters. 
   - Inherit from base classes where possible (e.g., create a `BaseScenarioEnv` and extend it).
2. **On-Run Outputs (Real-Time Logging):**
   - Implement verbose, clean console logging during simulation runs.
   - Example Output: `[Scenario: Spillback] | Step: 1200 | Avg Queue: 45.2 | Deadlocks Detected: 2 | Teleports: 0`
   - Ensure these logs can be written to a `.log` file automatically.
3. **Automated Testing (`pytest`):**
   - EVERY new scenario must have a corresponding test file in the `tests/` directory (e.g., `tests/test_scenario_bottleneck.py`).
   - Tests must verify network generation, traffic flows, and edge-case handling without requiring a full training loop.

---

## 🚦 Scenario 1: The "Green Wave" (Correlated Platoon Arrivals)
**Goal:** Test if coordinated algorithms can preemptively clear downstream intersections for massive bursts of traffic.

* **Implementation Steps:**
    1. Create `generate_platoon_network.py`.
    2. Instead of a uniform Poisson process ($\lambda = 0.05$), define a route generator that injects a tightly packed "platoon" of 30-50 vehicles at a single entry node every 300 seconds.
    3. Keep cross-traffic at a low, steady Poisson rate.
* **Testing Requirements:**
    * Assert that the XML route file correctly schedules high-density bursts.
    * Run a dummy episode and assert that queue lengths temporarily spike at the entry nodes.

## 🗺️ Scenario 2: Real-World Asymmetric Topologies
**Goal:** Test algorithmic robustness on non-uniform, messy physical road networks.

* **Implementation Steps:**
    1. Create `generate_osm_network.py`.
    2. Use the SUMO `netconvert` tool or OSM Web Wizard to import a complex, real-world arterial corridor from Beer Sheva.
    3. Update the PettingZoo environment wrapper to dynamically handle intersections with varying numbers of incoming lanes (e.g., 3-way T-junctions mixed with 4-way intersections).
* **Testing Requirements:**
    * Assert that the environment initializes successfully without dimension mismatch errors.
    * Verify that the action space dictionary correctly maps to the specific allowed phases of each unique intersection.

## 🛑 Scenario 3: Deliberate Bottlenecks & Capacity Drops
**Goal:** Force upstream agents to sacrifice local performance to prevent downstream gridlock.

* **Implementation Steps:**
    1. Create `generate_bottleneck_network.py`.
    2. Build a linear or grid topology where a central arterial road drops from 3 lanes to 1 lane.
    3. Configure high traffic volume routing directly through the bottleneck edge.
* **Testing Requirements:**
    * Assert via SUMO's TraCI API that the specific bottleneck edge only has 1 active lane.
    * Test that independent agents fail (generate massive queues) while coordinated agents maintain a throttled flow.

## 🌊 Scenario 4: Spillback and Near-Saturation
**Goal:** Evaluate policy behavior when physical lane capacity is fully exhausted, threatening cascading deadlocks.

* **Implementation Steps:**
    1. Create `generate_spillback_network.py`.
    2. Reduce the physical distance between intersections in the `.net.xml` file (e.g., from 150m to 50m).
    3. Increase the global Poisson arrival rate $\lambda$ to near-saturation levels.
* **Testing Requirements:**
    * Write a test that monitors SUMO's internal teleport warnings (which indicate deadlocks).
    * Assert that the observation wrapper safely truncates or normalizes queue lengths if they exceed the physical capacity of the shortened lanes.

---

## 📊 Evaluation & Metric Aggregation
Update the evaluation loop to calculate the **"Coordination Gap"**.
- Track the Mean Squared Error (MSE) of queue lengths across neighboring intersections. High variance indicates poor coordination.
- Output a specific `coordination_metrics.csv` file alongside the standard waiting time plots.
